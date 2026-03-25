"""
main.py – Full ESGReveal-style pipeline orchestration
=====================================================
Usage:
    python main.py --pdf  data/EPA-Report-East-Rockingham-Waste-to-Energy-Revised-Proposal.pdf \
                   --xlsx data/GBF_Monitoring_Framework_Supplementary_Tables.xlsx \
                   --max  20 --rebuild
"""

import argparse, json, os
from datetime import datetime
from config import OUTPUT_DIR
from metadata_module import load_metadata
from preprocessing_module import ReportPreprocessor
from llm_agent_module import LLMAgent


def parse_args():
    p = argparse.ArgumentParser(description="ESGReveal-style extraction pipeline")
    p.add_argument("--pdf",     default="data/EPA-Report-East-Rockingham-Waste-to-Energy-Revised-Proposal.pdf")
    p.add_argument("--xlsx",    default="data/GBF_Monitoring_Framework_Supplementary_Tables.xlsx")
    p.add_argument(
        "--sheet",
        default="preliminary trial",
        help="Metadata sheet name (recommended: preliminary trial)",
    )
    p.add_argument("--name",    default="epa_report",     help="Report name (used for index files)")
    p.add_argument("--max",     type=int, default=None,   help="Max metadata entries to process (None = all)")
    p.add_argument("--rebuild", action="store_true",      help="Rebuild FAISS indices even if they exist")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Step 1: Load Metadata Module ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 – Loading GBF Metadata Module")
    print("="*60)
    entries = load_metadata(args.xlsx, sheet_name=args.sheet)

    # ── Step 2: Report Preprocessing ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 – Report Preprocessing Module")
    print("="*60)
    proc = ReportPreprocessor(args.pdf, report_name=args.name)

    index_exists = os.path.exists(f"index/{args.name}_text.faiss")
    if args.rebuild or not index_exists:
        proc.extract()
        proc.build_knowledge_bases()
        proc.save()
    else:
        print("[main] Indices found – loading from disk (use --rebuild to force reprocess).")
        proc.load()

    # ── Step 3: LLM Agent Module ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 – LLM Agent Module")
    print("="*60)
    agent   = LLMAgent(proc)
    results = agent.run(entries, max_entries=args.max)

    # ── Save results ──────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(OUTPUT_DIR, f"extraction_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[main] Results saved to {out_path}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    disclosed = sum(1 for r in results if r.get("Disclosure", "no").lower() == "yes")
    partial   = sum(1 for r in results if r.get("Disclosure", "no").lower() == "partial")
    total     = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY  |  Total indicators processed: {total}")
    print(f"         |  Disclosed  : {disclosed}  ({disclosed/total*100:.1f}%)")
    print(f"         |  Partial    : {partial}   ({partial/total*100:.1f}%)")
    print(f"         |  Not found  : {total-disclosed-partial}  ({(total-disclosed-partial)/total*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
