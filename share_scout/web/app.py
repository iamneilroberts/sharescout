"""Flask web application for browsing the ShareScout catalog."""

import subprocess
import signal
import sys
import os
import json as json_module
import urllib.request
import urllib.error

from flask import Flask, render_template, request, abort, redirect, url_for, jsonify, send_file

from ..catalog import Catalog


# Module-level process tracking
_crawl_process = None
_embed_process = None


def _get_ollama_status(config: dict) -> dict:
    """Fetch Ollama model/GPU status via API."""
    endpoint = config.get("ollama", {}).get("endpoint", "http://localhost:11434")
    result = {"reachable": False, "models": [], "error": None}
    try:
        # Get running models (ollama ps)
        req = urllib.request.Request(f"{endpoint}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json_module.loads(resp.read())
            result["reachable"] = True
            for m in data.get("models", []):
                model_info = {
                    "name": m.get("name", "?"),
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "vram_gb": round(m.get("size_vram", 0) / 1e9, 2),
                    "parameter_size": m.get("details", {}).get("parameter_size", "?"),
                    "quantization": m.get("details", {}).get("quantization_level", "?"),
                    "family": m.get("details", {}).get("family", "?"),
                    "context_length": m.get("context_length", 0),
                    "expires_at": m.get("expires_at", "")[:19],
                }
                total = m.get("size", 0)
                vram = m.get("size_vram", 0)
                model_info["gpu_percent"] = round(vram / total * 100) if total > 0 else 0
                result["models"].append(model_info)
    except urllib.error.URLError as e:
        result["error"] = f"Connection failed: {e.reason}"
    except Exception as e:
        result["error"] = str(e)
    return result


def create_app(config: dict) -> Flask:
    app = Flask(__name__)
    app.config["APP_CONFIG"] = config
    app.secret_key = "sharescout-session-key"
    db_path = config["catalog"]["db_path"]

    def get_catalog() -> Catalog:
        cat = Catalog(db_path, root_path=config["crawl"]["root_path"])
        cat.connect()
        cat.init_schema()
        return cat

    def format_size(size_bytes):
        if size_bytes is None:
            return "—"
        for unit in ("B", "KB", "MB", "GB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    app.jinja_env.filters["format_size"] = format_size

    def project_name(path):
        """Extract project name from path (first dir under root_path)."""
        if not path:
            return "—"
        path = path.replace("\\", "/")
        root = config["crawl"]["root_path"].replace("\\", "/").rstrip("/") + "/"
        if path.startswith(root):
            rel = path[len(root):]
            parts = rel.split("/")
            return parts[0] if parts else "—"
        # Fallback
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part == "dev" and i + 1 < len(parts):
                return parts[i + 1]
        return "—"

    app.jinja_env.filters["project_name"] = project_name

    def subpath(path):
        """Extract project/subdir/filename from path."""
        if not path:
            return "—"
        path = path.replace("\\", "/")
        root = config["crawl"]["root_path"].replace("\\", "/").rstrip("/") + "/"
        if path.startswith(root):
            return path[len(root):]
        # Fallback
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part == "dev" and i + 1 < len(parts):
                return "/".join(parts[i + 1:])
        return path

    app.jinja_env.filters["subpath"] = subpath

    import re
    _preamble_re = re.compile(
        r'^(This\s+(document|file|README[\w.]*|spec|report|page)\s+'
        r'(describes|details|outlines|provides|defines|specifies|contains|covers|presents|summarizes|explains|is about|is a)\s+'
        r'(an?\s+|the\s+)?)',
        re.IGNORECASE
    )

    def strip_preamble(text):
        if not text:
            return ""
        stripped = _preamble_re.sub('', text, count=1)
        if stripped and stripped != text:
            # Capitalize first letter
            stripped = stripped[0].upper() + stripped[1:]
        return stripped

    app.jinja_env.filters["strip_preamble"] = strip_preamble

    @app.context_processor
    def utility_processor():
        def browse_url(**overrides):
            defaults = {
                "category": request.args.get("category"),
                "extension": request.args.get("extension"),
                "project": request.args.get("project"),
                "status": request.args.get("status"),
                "min_score": request.args.get("min_score"),
                "max_score": request.args.get("max_score"),
                "sort": request.args.get("sort", "relevance_score"),
                "dir": request.args.get("dir", "DESC"),
                "page": request.args.get("page", 1),
            }
            defaults.update(overrides)
            return url_for("browse", **{k: v for k, v in defaults.items() if v is not None})

        def crawl_status():
            return "running" if _is_crawl_running() else "stopped"

        return {"browse_url": browse_url, "crawl_status": crawl_status}

    @app.after_request
    def add_no_cache(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/")
    def dashboard():
        cat = get_catalog()
        try:
            stats = cat.get_stats()
            score_dist = cat.get_score_distribution()
            categories = cat.get_categories()
            top_ext = cat.get_top_extensions()
            crawl_runs = cat.get_all_crawl_runs()
            recent = cat.get_recent_analyses(20)
            project_stats = cat.get_project_stats()
            rate = cat.get_analysis_rate()
            recent_skips = cat.get_recent_errors(15)
            live = cat.get_live_status()
            ollama = _get_ollama_status(config)
            vision_gaps = cat.get_unprocessed_image_stats()
            embedding_stats = cat.get_embedding_stats()
            embedding_model = config.get("ollama", {}).get("embedding_model")
            embed_running = _is_embed_running()
            return render_template(
                "dashboard.html",
                stats=stats, score_dist=score_dist,
                categories=categories, top_ext=top_ext,
                crawl_runs=crawl_runs, recent=recent,
                project_stats=project_stats, rate=rate,
                recent_skips=recent_skips, live=live,
                ollama=ollama, vision_gaps=vision_gaps,
                embedding_stats=embedding_stats, embedding_model=embedding_model,
                embed_running=embed_running,
            )
        finally:
            cat.close()

    # -- Crawl control routes --

    def _is_crawl_running():
        """Check if any share_scout crawl process is running (ours or external)."""
        global _crawl_process
        # Check our tracked process first
        if _crawl_process is not None and _crawl_process.poll() is None:
            return True
        # Check for any crawl process system-wide
        try:
            result = subprocess.run(
                ["pgrep", "-f", "share_scout.*crawl"],
                capture_output=True, text=True,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            # Filter out our own web server PID
            own_pid = str(os.getpid())
            crawl_pids = [p for p in pids if p != own_pid]
            return len(crawl_pids) > 0
        except Exception:
            return False

    @app.route("/crawl/start", methods=["POST"])
    def crawl_start():
        global _crawl_process
        if _is_crawl_running():
            return redirect(url_for("dashboard"))

        # Start crawl as subprocess
        venv_python = os.path.join(os.path.dirname(sys.executable), "python")
        _crawl_process = subprocess.Popen(
            [venv_python, "-m", "share_scout", "crawl"],
            cwd=config.get("_project_root", os.getcwd()),
            stdout=open("/tmp/sharescout-crawl.log", "w"),
            stderr=subprocess.STDOUT,
        )
        return redirect(url_for("dashboard"))

    @app.route("/crawl/stop", methods=["POST"])
    def crawl_stop():
        global _crawl_process
        # Stop our tracked process
        if _crawl_process is not None and _crawl_process.poll() is None:
            _crawl_process.send_signal(signal.SIGINT)
            try:
                _crawl_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _crawl_process.kill()
        _crawl_process = None
        # Also kill any other crawl processes we don't track
        try:
            subprocess.run(
                ["pkill", "-INT", "-f", "share_scout.*crawl"],
                capture_output=True,
            )
        except Exception:
            pass
        return redirect(url_for("dashboard"))

    @app.route("/crawl/status")
    def crawl_status_api():
        """JSON endpoint for polling crawl + embed status."""
        running = _is_crawl_running()
        embedding = _is_embed_running()
        cat = get_catalog()
        try:
            stats = cat.get_stats()
            live = cat.get_live_status()
            rate = cat.get_analysis_rate()
            emb = cat.get_embedding_stats()
            return jsonify({
                "running": running,
                "embedding": embedding,
                "total_files": stats["total_files"],
                "analyzed": stats["analyzed"],
                "extracted": stats["extracted"],
                "skipped": stats["skipped"],
                "queue_depth": live["queue_depth"],
                "last_analyzed": live["last_analyzed"]["filename"] if live["last_analyzed"] else None,
                "last_analyzed_project": project_name(live["last_analyzed"]["path"]) if live["last_analyzed"] else None,
                "per_minute": rate.get("per_minute", 0),
                "eta_hours": rate.get("eta_hours", 0),
                "embedded_files": emb.get("files_with_embeddings", 0),
                "total_embeddings": emb.get("total_embeddings", 0),
                "unembedded_files": emb.get("files_without_embeddings", 0),
            })
        finally:
            cat.close()

    # -- Embed control routes --

    def _is_embed_running():
        """Check if an embed process is running."""
        global _embed_process
        if _embed_process is not None and _embed_process.poll() is None:
            return True
        try:
            result = subprocess.run(
                ["pgrep", "-f", "share_scout.*embed"],
                capture_output=True, text=True,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            own_pid = str(os.getpid())
            return len([p for p in pids if p != own_pid]) > 0
        except Exception:
            return False

    @app.route("/embed/start", methods=["POST"])
    def embed_start():
        global _embed_process
        if _is_embed_running():
            return redirect(url_for("dashboard"))
        venv_python = os.path.join(os.path.dirname(sys.executable), "python")
        _embed_process = subprocess.Popen(
            [venv_python, "-m", "share_scout", "embed"],
            cwd=config.get("_project_root", os.getcwd()),
            stdout=open("/tmp/sharescout-embed.log", "w"),
            stderr=subprocess.STDOUT,
        )
        return redirect(url_for("dashboard"))

    @app.route("/embed/status")
    def embed_status_api():
        """JSON endpoint for polling embed status."""
        running = _is_embed_running()
        cat = get_catalog()
        try:
            stats = cat.get_embedding_stats()
            return jsonify({"running": running, **stats})
        finally:
            cat.close()

    @app.route("/api/reset-stats", methods=["POST"])
    def api_reset_stats():
        """Reset throughput stats — only count analyses from now on."""
        cat = get_catalog()
        try:
            cat.reset_stats()
            return jsonify({"ok": True, "message": "Stats reset. Throughput will recalculate from new analyses."})
        finally:
            cat.close()

    @app.route("/api/clear-stats-reset", methods=["POST"])
    def api_clear_stats_reset():
        """Clear stats reset — show all-time stats again."""
        cat = get_catalog()
        try:
            cat.clear_stats_reset()
            return jsonify({"ok": True, "message": "Showing all-time stats."})
        finally:
            cat.close()

    @app.route("/crawl/log")
    def crawl_log():
        """Return last N lines of crawl log."""
        lines = 50
        try:
            with open("/tmp/sharescout-crawl.log") as f:
                all_lines = f.readlines()
                return "<pre>" + "".join(all_lines[-lines:]) + "</pre>"
        except FileNotFoundError:
            return "<pre>No crawl log yet.</pre>"

    # -- Browse, Search, Detail --

    @app.route("/browse")
    def browse():
        cat = get_catalog()
        try:
            category = request.args.get("category")
            extension = request.args.get("extension")
            project = request.args.get("project")
            status = request.args.get("status")
            min_score = request.args.get("min_score", type=float)
            max_score = request.args.get("max_score", type=float)
            sort_by = request.args.get("sort", "relevance_score")
            sort_dir = request.args.get("dir", "DESC")
            page = request.args.get("page", 1, type=int)
            per_page = 50
            offset = (page - 1) * per_page

            files, total = cat.browse_files(
                category=category, extension=extension,
                project=project, status=status,
                min_score=min_score, max_score=max_score,
                sort_by=sort_by, sort_dir=sort_dir,
                limit=per_page, offset=offset,
            )
            categories = cat.get_categories()
            projects = cat.get_projects()
            total_pages = (total + per_page - 1) // per_page

            # Group projects
            from ..config import load_project_groups
            group_defs = load_project_groups(
                os.path.join(config.get("_project_root", os.getcwd()), "project_groups.yaml")
            )
            # Build grouped structure: [(group_name, [(project, count)])]
            project_lookup = {p["project"]: p["count"] for p in projects}
            assigned = set()
            grouped_projects = []
            for group_name, members in group_defs.items():
                group_items = []
                for m in members:
                    if m in project_lookup:
                        group_items.append({"project": m, "count": project_lookup[m]})
                        assigned.add(m)
                if group_items:
                    group_items.sort(key=lambda x: -x["count"])
                    grouped_projects.append((group_name, group_items))
            # "Other" for ungrouped projects
            other = [p for p in projects if p["project"] not in assigned]
            if other:
                grouped_projects.append(("Other", other))

            return render_template(
                "browse.html",
                files=files, categories=categories,
                projects=projects, grouped_projects=grouped_projects,
                total=total, page=page, total_pages=total_pages,
                current_category=category, current_extension=extension,
                current_project=project, current_status=status,
                current_sort=sort_by, current_dir=sort_dir,
                min_score=min_score, max_score=max_score,
            )
        finally:
            cat.close()

    @app.route("/search")
    def search():
        cat = get_catalog()
        try:
            query = request.args.get("q", "").strip()
            page = request.args.get("page", 1, type=int)
            per_page = 50
            offset = (page - 1) * per_page

            results = []
            total = 0
            if query:
                results, total = cat.search(query, limit=per_page, offset=offset)

            total_pages = (total + per_page - 1) // per_page

            return render_template(
                "search.html",
                query=query, results=results,
                total=total, page=page, total_pages=total_pages,
            )
        finally:
            cat.close()

    @app.route("/compare")
    def compare():
        cat = get_catalog()
        try:
            projects = cat.get_projects()
            selected = request.args.getlist("projects")
            grid = {}  # {category: {project: [files]}}
            all_categories = []
            if len(selected) >= 2:
                files = cat.get_files_by_projects(selected)
                for f in files:
                    cat_name = f.get("category") or "Uncategorized"
                    proj = f["project"]
                    grid.setdefault(cat_name, {})
                    grid[cat_name].setdefault(proj, [])
                    grid[cat_name][proj].append(f)
                all_categories = sorted(grid.keys())
            return render_template(
                "compare.html",
                projects=projects, selected=selected,
                grid=grid, all_categories=all_categories,
            )
        finally:
            cat.close()

    @app.route("/tags")
    def tags():
        cat = get_catalog()
        try:
            keyword = request.args.get("keyword")
            all_keywords = cat.get_all_keywords()
            files = []
            if keyword:
                files = cat.get_files_by_keyword(keyword)
            return render_template(
                "tags.html",
                keywords=all_keywords, current_keyword=keyword,
                files=files,
            )
        finally:
            cat.close()

    @app.route("/timeline")
    def timeline():
        cat = get_catalog()
        try:
            projects = cat.get_projects()
            project = request.args.get("project")
            files = cat.get_timeline_files(project=project)
            # Group by month
            from collections import OrderedDict
            months = OrderedDict()
            for f in files:
                mod = f.get("modified_at") or ""
                if len(mod) >= 7:
                    # Parse YYYY-MM to "Month YYYY"
                    try:
                        from datetime import datetime as dt
                        d = dt.fromisoformat(mod[:10]) if len(mod) >= 10 else dt.fromisoformat(mod[:7] + "-01")
                        month_key = d.strftime("%B %Y")
                    except (ValueError, TypeError):
                        month_key = "Unknown"
                else:
                    month_key = "Unknown"
                months.setdefault(month_key, []).append(f)
            return render_template(
                "timeline.html",
                months=months, projects=projects,
                current_project=project,
            )
        finally:
            cat.close()

    @app.route("/insights")
    def insights():
        cat = get_catalog()
        try:
            from ..config import load_project_groups
            group_defs = load_project_groups(
                os.path.join(config.get("_project_root", os.getcwd()), "project_groups.yaml")
            )
            data = cat.get_insights(group_defs)
            return render_template("insights.html", **data)
        finally:
            cat.close()

    @app.route("/file/<int:file_id>")
    def file_detail(file_id):
        cat = get_catalog()
        try:
            detail = cat.get_file_detail(file_id)
            if not detail:
                abort(404)
            # Get related files
            related_same_name = cat.get_related_by_filename(file_id, detail["filename"])
            related_keywords = []
            if detail.get("analysis") and detail["analysis"].get("keywords"):
                related_keywords = cat.get_related_by_keywords(
                    file_id, detail["analysis"]["keywords"]
                )
            return render_template(
                "file_detail.html", file=detail,
                related_same_name=related_same_name,
                related_keywords=related_keywords,
            )
        finally:
            cat.close()

    @app.route("/file/<int:file_id>/raw")
    def file_raw(file_id):
        """Serve the original file for viewing."""
        cat = get_catalog()
        try:
            detail = cat.get_file(file_id)
            if not detail:
                abort(404)
            filepath = detail["path"]
            if not os.path.exists(filepath):
                abort(404, description="Original file not found on disk")
            return send_file(filepath)
        finally:
            cat.close()

    # -- Ask AI (RAG) --

    @app.route("/ask", methods=["GET", "POST"])
    def ask():
        from flask import session as flask_session
        from ..rag import ask as rag_ask
        cat = get_catalog()
        try:
            has_embeddings = cat.has_embeddings()
            question = None
            answer = None
            sources = []
            history = flask_session.get("ask_history", [])

            if request.method == "POST":
                action = request.form.get("action", "ask")

                if action == "clear":
                    flask_session.pop("ask_history", None)
                    return redirect(url_for("ask"))

                question = request.form.get("question", "").strip()
                if question and has_embeddings:
                    result = rag_ask(config, cat, question, history=history)
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    # Append to conversation history
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": answer})
                    flask_session["ask_history"] = history

            return render_template(
                "ask.html",
                has_embeddings=has_embeddings,
                question=question,
                answer=answer,
                sources=sources,
                history=history,
            )
        finally:
            cat.close()

    # -- Settings --

    @app.route("/settings", methods=["GET", "POST"])
    def settings():
        import yaml
        project_root = config.get("_project_root", os.getcwd())
        config_path = os.path.join(project_root, "config.yaml")
        rules_path = os.path.join(project_root, "scoring_rules.yaml")

        error = None
        success = None

        if request.method == "POST":
            try:
                with open(config_path) as f:
                    disk_config = yaml.safe_load(f) or {}

                root_path = request.form.get("root_path", "").strip()
                ollama_endpoint = request.form.get("ollama_endpoint", "").strip()
                ollama_model = request.form.get("ollama_model", "").strip()
                score_threshold = request.form.get("score_threshold", "").strip()

                if root_path:
                    disk_config.setdefault("crawl", {})["root_path"] = root_path.replace("\\", "/")
                if ollama_endpoint:
                    disk_config.setdefault("ollama", {})["endpoint"] = ollama_endpoint
                if ollama_model:
                    disk_config.setdefault("ollama", {})["model"] = ollama_model

                with open(config_path, "w") as f:
                    yaml.dump(disk_config, f, default_flow_style=False, allow_unicode=True)

                if score_threshold:
                    try:
                        threshold_val = int(score_threshold)
                        with open(rules_path) as f:
                            rules_raw = f.read()
                        import re as _re
                        rules_raw = _re.sub(
                            r'^score_threshold:\s*\d+',
                            f'score_threshold: {threshold_val}',
                            rules_raw, flags=_re.MULTILINE
                        )
                        with open(rules_path, "w") as f:
                            f.write(rules_raw)
                    except Exception as e:
                        error = f"Saved config.yaml but could not update score threshold: {e}"

                if root_path:
                    config["crawl"]["root_path"] = root_path.replace("\\", "/")
                if ollama_endpoint:
                    config.setdefault("ollama", {})["endpoint"] = ollama_endpoint
                if ollama_model:
                    config.setdefault("ollama", {})["model"] = ollama_model

                if not error:
                    success = "Settings saved. Changes take effect on next crawl."

            except Exception as e:
                error = f"Failed to save settings: {e}"

        from ..config import load_scoring_rules
        try:
            rules = load_scoring_rules(rules_path)
            current_threshold = rules.get("score_threshold", 35)
        except Exception:
            current_threshold = 35

        ollama_models = _list_ollama_models(config)

        return render_template(
            "settings.html",
            config=config,
            current_threshold=current_threshold,
            ollama_models=ollama_models,
            error=error,
            success=success,
        )

    @app.route("/api/ollama/models")
    def api_ollama_models():
        """Return list of locally available Ollama models as JSON."""
        models = _list_ollama_models(config)
        return jsonify({"models": models})

    @app.route("/api/browse-dirs")
    def api_browse_dirs():
        """Return subdirectories of a given path for the folder picker."""
        path = request.args.get("path", "").strip()

        if not path:
            if sys.platform == "win32":
                import string
                drives = []
                for letter in string.ascii_uppercase:
                    drive = f"{letter}:/"
                    if os.path.exists(drive):
                        drives.append({"name": f"{letter}:", "path": drive, "sep": "/"})
                return jsonify({"path": "", "parent": None, "dirs": drives})
            else:
                path = "/"

        path = path.replace("\\", "/")
        norm = os.path.normpath(path)

        if not os.path.isdir(norm):
            return jsonify({"error": f"Not a directory: {path}"}), 400

        parent = None
        parent_raw = os.path.dirname(norm)
        if parent_raw and parent_raw != norm:
            parent = parent_raw.replace("\\", "/")

        skip = {"$recycle.bin", "system volume information", "windows",
                "program files", "program files (x86)", "programdata"}
        dirs = []
        try:
            for entry in sorted(os.scandir(norm), key=lambda e: e.name.lower()):
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name.startswith("."):
                    continue
                if entry.name.lower() in skip:
                    continue
                dirs.append({
                    "name": entry.name,
                    "path": entry.path.replace("\\", "/"),
                })
        except PermissionError:
            pass

        return jsonify({
            "path": norm.replace("\\", "/"),
            "parent": parent,
            "dirs": dirs,
        })

    return app


def _list_ollama_models(config: dict) -> list:
    """Fetch list of available models from Ollama /api/tags."""
    endpoint = config.get("ollama", {}).get("endpoint", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json_module.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
