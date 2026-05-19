import os
import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from anonymizer import anonymize_resume_text

load_dotenv()

# Model names read from .env — with sensible defaults
OPENAI_MODEL      = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def get_openai_client():
    """Create OpenAI client on demand — avoids crash at import if key is missing."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_deepseek_client():
    """Create Deepseek client on demand — avoids crash at import if key is missing."""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )


def safe_cut_text(text, max_chars=5000):
    """Trim resume text to avoid exceeding model token limits."""
    return (text or "").strip()[:max_chars]


# ── Stage 1: Deepseek fast pre-screening ─────────────────────────────────────

DEEPSEEK_SYSTEM = (
    "You are a recruitment screening assistant. "
    "Analyse resumes against job requirements. Return JSON only. No markdown."
)


def build_deepseek_prompt(job_requirement, keywords, resume_text):
    """Build a lightweight prompt for Deepseek quick scoring."""
    return f"""
Score this anonymised resume against the job requirement.

Job Requirement:
{job_requirement}

Mandatory Keywords: {", ".join(keywords)}

Resume:
{safe_cut_text(resume_text, 5000)}

Return ONLY a JSON object with these fields (no extra text, no markdown):
{{
  "quick_score": <integer 0-100>,
  "keyword_coverage": <integer, count of mandatory keywords found>,
  "role_relevance": "<High|Medium|Low>",
  "quick_reason": "<2 sentences max>"
}}
"""


def deepseek_screen_one(job_requirement, keywords, resume_text, max_retries=2):
    """
    Send one anonymised resume to Deepseek for quick scoring.
    Falls back to a zero-score result if all retries fail.
    """
    prompt = build_deepseek_prompt(job_requirement, keywords, resume_text)

    for attempt in range(max_retries + 1):
        try:
            response = get_deepseek_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": DEEPSEEK_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r"```(?:json)?|```", "", content).strip()
            return json.loads(content)

        except Exception as e:
            if attempt >= max_retries:
                print(f"❌ Deepseek failed: {e}")
                return {
                    "quick_score":      0,
                    "keyword_coverage": 0,
                    "role_relevance":   "Low",
                    "quick_reason":     f"Deepseek error: {e}",
                }
            time.sleep(1.0)


def deepseek_batch_screen(job_requirement, keywords, results, top_n=10):
    """
    Run Deepseek quick screening in parallel across all keyword-filtered candidates.
    max_workers=5 — enough to saturate Deepseek without hitting rate limits.
    """
    def screen_one(resume):
        anon_text = anonymize_resume_text(
            resume.get("resume_text", ""),
            resume.get("employee_name", ""),
        )
        ds_result = deepseek_screen_one(job_requirement, keywords, anon_text)

        merged = dict(resume)
        merged["ds_quick_score"]      = ds_result.get("quick_score", 0)
        merged["ds_keyword_coverage"] = ds_result.get("keyword_coverage", 0)
        merged["ds_role_relevance"]   = ds_result.get("role_relevance", "")
        merged["ds_quick_reason"]     = ds_result.get("quick_reason", "")
        merged["anon_text"]           = anon_text  # cache for GPT stage
        return merged

    scored = []
    print(f"🔵 Deepseek: screening {len(results)} candidates in parallel (workers=5)...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(screen_one, r): r for r in results}
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            scored.append(result)
            print(f"   ✓ Deepseek {i}/{len(results)} — {result.get('employee_name', '')} scored {result['ds_quick_score']}")

    scored.sort(key=lambda x: x["ds_quick_score"], reverse=True)
    print(f"🔵 Deepseek done. Top {top_n} forwarded to GPT.")
    return scored[:top_n]


# ── Stage 2: GPT-4o mini deep analysis ───────────────────────────────────────

GPT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "resume_match_analysis",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "match_score":          {"type": "integer", "minimum": 0, "maximum": 100},
                "recommendation":       {
                    "type": "string",
                    "enum": ["Strong Match", "Good Match", "Possible Match", "Weak Match", "Not Suitable"]
                },
                "matched_requirements": {"type": "array", "items": {"type": "string"}},
                "missing_requirements": {"type": "array", "items": {"type": "string"}},
                "matched_keywords":     {"type": "array", "items": {"type": "string"}},
                "missing_keywords":     {"type": "array", "items": {"type": "string"}},
                "role_match":           {"type": "string"},
                "industry_match":       {"type": "string"},
                "licence_ticket_match": {"type": "string"},
                "experience_evidence":  {"type": "array", "items": {"type": "string"}},
                "risk_flags":           {"type": "array", "items": {"type": "string"}},
                "short_summary":        {"type": "string"},
                "reason":               {"type": "string"},
            },
            "required": [
                "match_score", "recommendation",
                "matched_requirements", "missing_requirements",
                "matched_keywords", "missing_keywords",
                "role_match", "industry_match", "licence_ticket_match",
                "experience_evidence", "risk_flags", "short_summary", "reason",
            ],
        },
    },
}


def build_gpt_prompt(job_requirement, keywords, resume_text):
    """Build the detailed prompt for GPT-4o mini deep analysis."""
    return f"""
You are an experienced recruitment screening assistant for labour hire, mining, civil, construction, shutdown, and industrial roles.

Analyse whether this candidate's anonymised resume matches the job requirement.

Rules:
1. Only use the provided job requirement, mandatory keywords, and resume text.
2. Do not invent experience, tickets, licences, certificates, companies, or job titles.
3. If evidence is not clearly found, mark it as missing.
4. Mandatory keywords are hard screening criteria.
5. Be strict but fair.
6. Return valid JSON only.

Scoring guide:
90-100 = Excellent fit
75-89  = Strong match
60-74  = Possible match
40-59  = Weak match
0-39   = Not suitable

Job Requirement:
{job_requirement}

Mandatory Keywords: {", ".join(keywords)}

Resume:
{safe_cut_text(resume_text, 6000)}
"""


def gpt_analyse_one(job_requirement, keywords, resume_text, max_retries=2):
    """
    Send one anonymised resume to GPT-4o mini for deep structured analysis.
    Uses OpenAI structured output (json_schema) to enforce response format.
    Falls back to a zero-score error result if all retries fail.
    """
    prompt = build_gpt_prompt(job_requirement, keywords, resume_text)

    for attempt in range(max_retries + 1):
        try:
            response = get_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict recruitment resume screening assistant. Return JSON only."},
                    {"role": "user",   "content": prompt},
                ],
                response_format=GPT_SCHEMA,
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            if attempt >= max_retries:
                print(f"❌ GPT failed: {e}")
                return {
                    "match_score":          0,
                    "recommendation":       "Not Suitable",
                    "matched_requirements": [],
                    "missing_requirements": [],
                    "matched_keywords":     [],
                    "missing_keywords":     keywords,
                    "role_match":           "",
                    "industry_match":       "",
                    "licence_ticket_match": "",
                    "experience_evidence":  [],
                    "risk_flags":           ["GPT analysis failed"],
                    "short_summary":        "GPT analysis failed.",
                    "reason":               f"GPT error: {e}",
                }
            time.sleep(1.5)


def gpt_batch_analyse(job_requirement, keywords, deepseek_results, top_n=5):
    """
    Run GPT-4o mini deep analysis in parallel on top_n Deepseek candidates.
    max_workers=3 — conservative to avoid OpenAI rate limits.
    """
    candidates = deepseek_results[:top_n]

    def analyse_one(resume):
        anon_text = resume.get("anon_text") or anonymize_resume_text(
            resume.get("resume_text", ""),
            resume.get("employee_name", ""),
        )
        ai = gpt_analyse_one(job_requirement, keywords, anon_text)

        merged = dict(resume)
        merged["ai_match_score"]          = ai.get("match_score", 0)
        merged["ai_recommendation"]       = ai.get("recommendation", "")
        merged["ai_matched_requirements"] = ai.get("matched_requirements", [])
        merged["ai_missing_requirements"] = ai.get("missing_requirements", [])
        merged["ai_matched_keywords"]     = ai.get("matched_keywords", [])
        merged["ai_missing_keywords"]     = ai.get("missing_keywords", [])
        merged["ai_role_match"]           = ai.get("role_match", "")
        merged["ai_industry_match"]       = ai.get("industry_match", "")
        merged["ai_licence_ticket_match"] = ai.get("licence_ticket_match", "")
        merged["ai_experience_evidence"]  = ai.get("experience_evidence", [])
        merged["ai_risk_flags"]           = ai.get("risk_flags", [])
        merged["ai_short_summary"]        = ai.get("short_summary", "")
        merged["ai_reason"]               = ai.get("reason", "")
        return merged

    analysed = []
    print(f"🟢 GPT: deep-analysing {len(candidates)} candidates in parallel (workers=3)...")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyse_one, r): r for r in candidates}
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            analysed.append(result)
            print(f"   ✓ GPT {i}/{len(candidates)} — {result.get('employee_name', '')} scored {result['ai_match_score']}")

    analysed.sort(key=lambda x: x.get("ai_match_score", 0), reverse=True)
    print(f"🟢 GPT done.")
    return analysed


# ── Main entry point ──────────────────────────────────────────────────────────

def run_two_stage_screening(
    job_requirement: str,
    keywords: list,
    keyword_results: list,
    deepseek_top_n: int = 10,
    gpt_top_n: int = 5,
):
    """
    Full two-stage AI screening pipeline:
      keyword_results → Deepseek parallel (top deepseek_top_n) → GPT parallel (top gpt_top_n)
    """
    deepseek_results = deepseek_batch_screen(
        job_requirement, keywords, keyword_results, top_n=deepseek_top_n
    )

    gpt_results = gpt_batch_analyse(
        job_requirement, keywords, deepseek_results, top_n=gpt_top_n
    )

    return {
        "deepseek_results": deepseek_results,
        "gpt_results":      gpt_results,
    }