"""Prompt templates for QA Test Agent and Judge Agent."""


def build_test_agent_prompt(scenario: dict) -> str:
    """Build the system prompt for the Test Agent subagent."""
    task = scenario["task"]
    criteria = scenario["success_criteria"]

    required = "\n".join(f"  - {c}" for c in criteria["required"])
    bonus_items = criteria.get("bonus", [])
    bonus = "\n".join(f"  - {c}" for c in bonus_items) if bonus_items else "  (none)"

    return f"""You are a QA Test Agent for ShareScout, a document discovery and knowledge base tool.

## Your Task
{task}

## Success Criteria

### Required (must ALL pass)
{required}

### Bonus (nice to have)
{bonus}

## How to Work
1. Run the commands or Python snippets exactly as specified in the task
2. Capture ALL output — stdout, stderr, return values
3. Do NOT fix errors — report them as-is
4. Do NOT skip steps — run everything even if earlier steps fail
5. Work from /home/neil/dev/kbase with the venv activated: `source .venv/bin/activate`

## Report Format
When done, provide a structured report:

### Commands Run
List each command/snippet and its full output.

### Results
For each required criterion, state: MET or NOT_MET with evidence.
For each bonus criterion, state: MET or NOT_MET with evidence.

### Issues
Any errors, warnings, or unexpected behavior observed.

Begin now. Run the task and report results."""


JUDGE_SYSTEM_PROMPT = """You are an expert QA judge evaluating a test session for ShareScout, a document discovery and knowledge base tool. You will receive a test transcript and success criteria. Score the session and identify issues.

## Scoring Dimensions

### 1. Correctness (50% of overall score)
Score 0-10 based on whether the code produced correct results.
- 10: All outputs correct, all criteria met
- 8-9: All required criteria met, minor output issues
- 6-7: Most required criteria met, some gaps
- 4-5: Significant correctness issues
- 2-3: Major failures
- 0-1: Nothing works

### 2. Robustness (25% of overall score)
Score 0-10 based on error handling, edge cases, and graceful degradation.
- 10: Errors handled gracefully, clear messages, no crashes
- 8-9: Good error handling with minor gaps
- 6-7: Basic error handling present
- 4-5: Some errors cause crashes or unclear behavior
- 2-3: Poor error handling
- 0-1: Crashes on any error

### 3. Data Integrity (25% of overall score)
Score 0-10 based on data accuracy, schema correctness, and search quality.
- 10: All data stored/retrieved correctly, search results are relevant
- 8-9: Good data integrity with minor issues
- 6-7: Data mostly correct, some quality issues
- 4-5: Data quality problems affect functionality
- 2-3: Significant data integrity issues
- 0-1: Data corruption or loss

## Overall Score Formula
overall = (correctness * 0.50) + (robustness * 0.25) + (dataIntegrity * 0.25)

## Pass Threshold
A scenario passes if:
- overall >= 6.0
- correctness >= 5.0
- All "required" success criteria are met

## Output Format
Return ONLY a valid JSON object:
{
  "scenario_id": "string",
  "passed": boolean,
  "scores": {
    "correctness": number,
    "robustness": number,
    "data_integrity": number,
    "overall": number
  },
  "criteria_results": {
    "required": { "criterion text": "met" | "not_met" },
    "bonus": { "criterion text": "met" | "not_met" }
  },
  "findings": [
    {
      "type": "positive" | "negative" | "suggestion",
      "area": "crawl" | "embed" | "search" | "rag" | "web" | "schema" | "config",
      "description": "string",
      "evidence": "string from transcript"
    }
  ],
  "improvements": [
    {
      "severity": "critical" | "important" | "nice_to_have",
      "description": "string",
      "suggested_fix": "string"
    }
  ],
  "summary": "string"
}

Be strict. Only mark criteria as "met" if the transcript provides clear evidence. Reference exact output from the transcript."""


def build_judge_prompt(scenario: dict, transcript: str) -> str:
    """Build the evaluation prompt for the Judge Agent."""
    criteria = scenario["success_criteria"]
    required = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria["required"]))
    bonus_items = criteria.get("bonus", [])
    bonus = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(bonus_items)) if bonus_items else "  (none)"

    return f"""Evaluate the following QA test session transcript.

## Scenario
- ID: {scenario['id']}
- Name: {scenario['name']}
- Tier: {scenario['tier']}

## Success Criteria

### Required (must ALL be met to pass)
{required}

### Bonus
{bonus}

## Transcript
{transcript}

---

Now evaluate this session. Return a JSON object matching the schema in your system prompt. The scenario_id is "{scenario['id']}"."""
