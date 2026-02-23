import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import AvaliacaoDataMapping as adm
from AvaliacaoDataMapping.rag_engine import build_chunks, build_rag_index, evaluate_questions_with_rag


SUPPORTED_EXT = {".txt", ".csv", ".xlsx", ".docx", ".pdf"}


def _question_pairs() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    out.extend(list(adm.AREAS_QMAP.items()))
    out.extend(list(adm.PROCESSOS_QMAP.items()))
    out.extend(list(adm.SISTEMAS_QMAP.items()))
    return out


def _old_llm_map(text_norm: str, structured: dict) -> Dict[str, int]:
    text_for_llm = structured.get("__raw_text_src__") if isinstance(structured, dict) else None
    text_for_llm = adm._clip_text_for_llm(text_for_llm or text_norm)
    if not text_for_llm:
        return {}

    llm = adm.llm_evaluate_document(text_for_llm) or {}
    merged: Dict[str, int] = {}
    for sec in ("areas", "processos", "sistemas"):
        sec_map = (llm.get(sec, {}) or {}).get("map", {}) or {}
        for code, bit in sec_map.items():
            merged[str(code)] = 1 if int(bit) == 1 else 0
    return merged


def _rag_map(text_norm: str, structured: dict, top_k: int, model: str) -> Dict[str, int]:
    chunks = build_chunks(text_norm, structured)
    index = build_rag_index(
        text_norm,
        structured,
        chunks=chunks,
        question_sections=adm.QUESTION_SECTION_BY_CODE,
        embed_chunks=True,
    )
    rag_eval = evaluate_questions_with_rag(
        _question_pairs(),
        index,
        top_k=top_k,
        model=model,
    ) or {}
    out: Dict[str, int] = {}
    for code, row in ((rag_eval.get("results") or {}) or {}).items():
        out[str(code)] = 1 if int((row or {}).get("answerable", 0)) == 1 else 0
    return out


def _compare_maps(old_map: Dict[str, int], rag_map: Dict[str, int]) -> Dict[str, object]:
    all_codes = sorted(set(old_map.keys()) | set(rag_map.keys()))
    compared = len(all_codes)
    agree = 0
    diverged: List[str] = []
    for code in all_codes:
        old_bit = 1 if int(old_map.get(code, 0)) == 1 else 0
        rag_bit = 1 if int(rag_map.get(code, 0)) == 1 else 0
        if old_bit == rag_bit:
            agree += 1
        else:
            diverged.append(code)
    return {
        "compared": compared,
        "agree": agree,
        "disagree": max(0, compared - agree),
        "diverged_codes": diverged[:20],
    }


def _iter_files(root: Path, limit: int) -> List[Path]:
    selected: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXT:
            continue
        selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare old LLM flow vs RAG flow on sample files.")
    parser.add_argument("input_dir", help="Folder with files to evaluate")
    parser.add_argument("--limit", type=int, default=10, help="Max files to evaluate (default: 10)")
    parser.add_argument("--top-k", type=int, default=4, help="RAG top-k (default: 4)")
    parser.add_argument("--rag-model", default=os.getenv("RAG_EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    args = parser.parse_args()

    root = Path(args.input_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"input_dir invalid: {root}")

    files = _iter_files(root, max(1, args.limit))
    if not files:
        raise SystemExit("No supported files found")

    summary: Dict[str, object] = {
        "input_dir": str(root),
        "limit": args.limit,
        "processed": 0,
        "top_k": args.top_k,
        "rag_model": args.rag_model,
        "files": [],
        "aggregate": {"compared": 0, "agree": 0, "disagree": 0},
    }

    for path in files:
        started = time.time()
        try:
            data = path.read_bytes()
            text_norm, structured = adm.extract_text_from_bytes(path.name, data)
            old_map = _old_llm_map(text_norm, structured)
            rag_map = _rag_map(text_norm, structured, top_k=args.top_k, model=args.rag_model)
            cmp_row = _compare_maps(old_map, rag_map)
            elapsed_ms = int((time.time() - started) * 1000)
            summary["files"].append(
                {
                    "file": str(path),
                    "elapsed_ms": elapsed_ms,
                    **cmp_row,
                }
            )
            summary["processed"] = int(summary["processed"]) + 1
            summary["aggregate"]["compared"] += int(cmp_row["compared"])
            summary["aggregate"]["agree"] += int(cmp_row["agree"])
            summary["aggregate"]["disagree"] += int(cmp_row["disagree"])
        except Exception as ex:
            summary["files"].append({"file": str(path), "error": str(ex)})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
