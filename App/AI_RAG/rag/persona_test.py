"""Persona test.

  python persona_test.py          same question sent to all 8 personas  -> compare the style
  python persona_test.py --own    each persona gets its own tailored question
  python persona_test.py --module pipeline   (default: auto-detect pipeline / rag_pipeline_v2)

Writes persona_report_same.md (default) or persona_report_own.md (--own). The script only counts words/citations; YOU judge the style (see checklist).
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

SHARED_QUESTION = "Explain how a murabaha financing works and what the risks are for each party."

OWN_QUESTIONS = {
    "general_public": "What is the difference between an Islamic financing and a regular bank loan?",
    "student": "How is ijarah muntahia bittamleek defined? I need to cite it in my paper.",
    "entrepreneur": "I'm launching a bakery and need money for an oven and a partner who invests. Which Islamic structures fit?",
    "business_owner": "My company already uses a conventional credit line. How would Islamic financing change my cash flow, and how does it compare?",
    "investor": "What are the risks of sukuk compared with conventional bonds, and is a sukuk liquid?",
    "banker": "What does AAOIFI require about ownership and risk transfer before a bank sells an asset by murabaha?",
    "academic": "Is there scholarly disagreement about tawarruq, and what does AAOIFI say about organized tawarruq?",
    "finance_specialist": "Explain the structure of parallel salam and how the two contracts must stay independent, with figures.",
}

CHECKLIST = """
What to look for (persona -> expected / red flag)
- general_public : short sentences, analogy with a normal loan, every Arabic term explained. RED FLAG: unexplained jargon.
- student        : precise definition of Arabic terms, clear structure, standard number cited. RED FLAG: loose wording.
- entrepreneur   : direct, solution-oriented, murabaha for equipment / musharaka for the partner. RED FLAG: invented prices or amounts.
- business_owner : big picture, cash-flow/balance-sheet angle. RED FLAG: made-up numbers; it must say the conventional side is outside the excerpts.
- investor       : risk / return / liquidity vocabulary. RED FLAG: invented yields or market data (it must say what the excerpts do not cover).
- banker         : maximum precision, exact standard referenced systematically. RED FLAG: simplified language.
- academic       : separates established consensus from controversy, sources cited. RED FLAG: claiming a controversy the excerpts do not show.
- finance_specialist : no simplification, structures from the first lines. RED FLAG: INVENTED FIGURES (the persona asks for figures but grounding forbids inventing them).
For ALL: still no fatwa, still [Source : AAOIFI Standard X] citations, no sentence cut in the middle.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="auto")
    ap.add_argument("--own", action="store_true", help="tailored question per persona")
    ap.add_argument("--sleep", type=float, default=25.0)
    args = ap.parse_args()

    names = ["pipeline", "rag_pipeline_v2"] if args.module == "auto" else [args.module]
    rp = None
    for name in names:
        try:
            rp = importlib.import_module(name)
            print(f"using module: {name}")
            break
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
    if rp is None:
        raise SystemExit("Could not import the pipeline. Run from its folder or pass --module.")

    url = os.getenv("SUPABASE_DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("No database URL in .env")
    personas = list(getattr(rp, "PERSONA_PROMPTS", {}) or [])
    if not personas:
        raise SystemExit("PERSONA_PROMPTS is empty: the personas module was not imported (check the import path).")

    out = [f"# Persona report ({'own questions' if args.own else 'same question for all'})\n", CHECKLIST]
    print(f"\n{'persona':<20}{'words':>6}{'cites':>7}  note")
    for i, persona in enumerate(personas):
        q = OWN_QUESTIONS.get(persona, SHARED_QUESTION) if args.own else SHARED_QUESTION
        try:
            answer, _ = rp.answer_question(q, url, persona_name=persona)
            note = ""
        except Exception as e:  # noqa: BLE001
            answer, note = f"ERROR: {type(e).__name__}: {e}", "ERROR"
        words = len(answer.split())
        cites = len(re.findall(r"\[Source : AAOIFI Standard \d+\]", answer))
        if answer.startswith(rp.INSUFFICIENT[:40]):
            note = "refused (insufficient)"
        print(f"{persona:<20}{words:>6}{cites:>7}  {note}")
        out.append(f"## {persona}\n**Q:** {q}\n\nwords={words} citations={cites} {note}\n\n{answer}\n")
        if i < len(personas) - 1:
            time.sleep(args.sleep)
    fname = "persona_report_own.md" if args.own else "persona_report_same.md"
    Path(fname).write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved {fname}. If the 8 answers read almost the same, the persona prompt is too weak.")


if __name__ == "__main__":
    main()