from __future__ import annotations
import os
from typing import Dict, Any, List
from datetime import datetime, timezone

import config.env_loader  # load .env
from storage.embryology_db import list_updates

# Optional OpenAI (v1). We handle missing keys gracefully.
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

SYSTEM_PROMPT = """You are a fertility clinic assistant.
Summarize embryology progress for a patient in clear, supportive, neutral language.
Constraints:
- Be factual and concise (5–12 bullet points max + a one-paragraph overview).
- Do NOT give medical advice or diagnosis; do NOT predict outcomes.
- If specific data is missing, say so briefly.
- Explain any terms very briefly (e.g., ‘blastocyst (a day-5/6 embryo)’).
- Use headings and bullet points; keep it scannable for patients.
- If grades exist (e.g., 4BB), mention them in context but avoid judging prognosis.
- End with a gentle nudge: “Please discuss next steps with your clinical team.”
"""

def _fmt_date(ts: int | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

def _make_plain_bullets(updates: List[Dict[str, Any]]) -> str:
    if not updates:
        return "- No daily embryology updates have been recorded yet."
    lines = []
    for u in updates:
        day = u.get("day", "?")
        stage = u.get("stage") or "-"
        total = u.get("total")
        good = u.get("good")
        grades = (u.get("grades") or "").strip()
        notes = (u.get("notes") or "").strip()
        date_s = _fmt_date(u.get("date_utc"))
        parts = [f"**Day {day}** ({stage}) on {date_s}: total={total if total is not None else '-'}"]
        if good is not None:
            parts.append(f"good={good}")
        if grades:
            parts.append(f"grades: {grades}")
        line = ", ".join(parts)
        if notes:
            line += f". Notes: {notes}"
        lines.append(f"- {line}")
    return "\n".join(lines)

def _oai() -> Any:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=key)

def summarize_updates(patient_id: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Returns:
      {
        'markdown': '<summary for chat>',
        'used_openai': True|False,
        'diagnostics': { 'num_updates': int }
      }
    """
    updates = list_updates(patient_id)
    diag = {"num_updates": len(updates)}

    if not updates:
        return {
            "markdown": (
                "### 🧫 Embryology summary\n"
                "No daily embryology updates have been recorded yet. "
                "Once the clinic enters updates (e.g., fertilization, cleavage, blastocyst counts and grades), "
                "I’ll summarize them here for you."
            ),
            "used_openai": False,
            "diagnostics": diag,
        }

    # Build a compact, model-friendly table text
    table_lines = ["Day | Stage | Total | Good | Grades | Date (UTC) | Notes",
                   "--- | --- | --- | --- | --- | --- | ---"]
    for u in updates:
        table_lines.append(
            f"{u.get('day','')} | {u.get('stage','')} | {u.get('total','')} | {u.get('good','')} | "
            f"{(u.get('grades') or '').strip()} | { _fmt_date(u.get('date_utc')) } | {(u.get('notes') or '').strip()}"
        )
    table_text = "\n".join(table_lines)

    # Attempt OpenAI narrative
    try:
        client = _oai()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Patient daily embryology ledger:\n\n{table_text}\n\n"
                "Write a short overview paragraph and 5–12 bullet points tailored to this ledger. "
                "Avoid predictions and medical advice; keep it supportive and neutral."
            },
        ]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        summary = (resp.choices[0].message.content or "").strip()
        if not summary:
            raise RuntimeError("Empty model response")
        md = f"### 🧫 Embryology summary (auto-generated)\n\n{summary}\n\n—\n*Please discuss next steps with your clinical team.*"
        return {"markdown": md, "used_openai": True, "diagnostics": diag}
    except Exception:
        # Fallback: deterministic, rule-based summary + raw bullets
        bullets = _make_plain_bullets(updates)
        md = (
            "### 🧫 Embryology summary\n"
            "_OpenAI summary unavailable right now. Showing a plain summary instead._\n\n"
            f"{bullets}\n\n—\n*Please discuss next steps with your clinical team.*"
        )
        return {"markdown": md, "used_openai": False, "diagnostics": diag}
