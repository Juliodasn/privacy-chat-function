from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .answerability_guard import (
    looks_like_prompt,
    reason_to_metric,
    validate_answerability,
    is_too_generic,
)
from .anchors_catalog import ANCHORS, QUESTION_ANCHOR_MAP


_DIVIDER_RE = re.compile(r"^\s*[-_=]{3,}\s*$")
_MULTI_SPACE_COL_SPLIT_RE = re.compile(r"\s{2,}")
_ANSWER_EMPTY_RE = re.compile(r"^[\s\-_:.|/]*$")
_NA_VARIANTS_RE = re.compile(
    r"^(?:n/?a|na|n\.a\.|nao\s+se\s+aplica|nao\s+aplicavel|nao\s+informado|not\s+applicable|not\s+informed)$",
    re.IGNORECASE,
)


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: str) -> str:
    text = value or ""
    text = _strip_accents(text).lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_na_variants(value: str) -> str:
    text = normalize_text(value or "")
    return "n/a" if _NA_VARIANTS_RE.match(text) else text


def _clean_answer_text(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"^[\s\-:|]+", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    norm = normalize_na_variants(text)
    if norm == "n/a":
        return "n/a"
    return text


def _is_meaningful_answer(value: str, anchor_text: str = "") -> bool:
    text = _clean_answer_text(value)
    if not text:
        return False
    if _ANSWER_EMPTY_RE.match(text):
        return False
    if _looks_like_gap_or_task(text):
        return False
    if looks_like_prompt(text, question_text=anchor_text):
        return False
    if text.endswith("?"):
        return False

    signal = normalize_text(text)
    if signal not in {"sim", "nao", "n/a"} and len(signal) < 3:
        return False

    if is_too_generic(text):
        return False

    if anchor_text and normalize_text(text) == normalize_text(anchor_text):
        return False

    return True


def _is_divider_line(value: str) -> bool:
    return bool(_DIVIDER_RE.match(value or ""))


def _line_columns(line: str) -> List[str]:
    raw = (line or "").strip()
    if not raw:
        return []
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
    else:
        parts = [p.strip() for p in _MULTI_SPACE_COL_SPLIT_RE.split(raw)]
    return [p for p in parts if p]

def _looks_like_gap_or_task(value: str) -> bool:
    text = normalize_text(value or "")
    if not text:
        return False

    if re.match(r"^gap\s*\d+\b", text):
        return True

    return bool(
        re.match(
            r"^(verificar|identificar|validar|confirmar|avaliar|analisar|checar|revisar|mapear|definir|informar|enviar|obter)\b",
            text,
        )
    )

def _extract_inline_answer(line: str) -> str:
    raw = (line or "").strip()
    if not raw:
        return ""

    if _looks_like_gap_or_task(raw):
        return ""

    for sep in ("?", ":", " - ", ";"):
        if sep not in raw:
            continue

        candidate = raw.split(sep, 1)[1].strip()

        if _looks_like_gap_or_task(candidate):
            continue

        if _is_meaningful_answer(candidate):
            return _clean_answer_text(candidate)

    return ""


def _prepare_anchor_patterns(catalog: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for anchor_id, cfg in (catalog or {}).items():
        patterns = cfg.get("patterns", []) if isinstance(cfg, dict) else []
        compiled = []
        for pattern in patterns or []:
            try:
                compiled.append(re.compile(str(pattern), re.IGNORECASE))
            except Exception:
                continue
        if not compiled:
            continue
        out[anchor_id] = {
            "compiled": compiled,
            "section": str((cfg or {}).get("section") or "global"),
            "type": str((cfg or {}).get("type") or "single_line"),
        }
    return out


def _match_anchor_ids(line_norm: str, prepared_catalog: Dict[str, Dict[str, object]]) -> List[str]:
    matches: List[str] = []
    for anchor_id, cfg in prepared_catalog.items():
        for cre in cfg.get("compiled", []):
            if cre.search(line_norm):
                matches.append(anchor_id)
                break
    return matches


def _collect_multiline_answer(
    lines: Sequence[str],
    lines_norm: Sequence[str],
    start_idx: int,
    prepared_catalog: Dict[str, Dict[str, object]],
    max_follow_lines: int,
    current_anchor_id: Optional[str] = None,
) -> Tuple[str, int]:
    collected: List[str] = []
    line_end = start_idx
    for j in range(start_idx + 1, min(len(lines), start_idx + 1 + max_follow_lines)):
        line = (lines[j] or "").strip()
        line_norm = (lines_norm[j] or "").strip()
        if not line or _is_divider_line(line):
            break
        matched_ids = _match_anchor_ids(line_norm, prepared_catalog)
        if matched_ids:
            if current_anchor_id and all(mid == current_anchor_id for mid in matched_ids):
                pass
            else:
                break
        collected.append(line)
        line_end = j
    return _clean_answer_text(" ".join(collected).strip()), line_end


def extract_qa_blocks(
    text: str,
    anchors_catalog: Optional[Dict[str, Dict[str, object]]] = None,
    source: str = "pdf_raw_text",
    max_follow_lines: int = 10,
) -> Tuple[List[Dict[str, object]], Dict[str, List[int]]]:
    catalog = anchors_catalog or ANCHORS
    prepared_catalog = _prepare_anchor_patterns(catalog)
    raw_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw_text.split("\n")
    lines_norm = [normalize_text(line) for line in lines]

    blocks: List[Dict[str, object]] = []
    qa_index: Dict[str, List[int]] = {}
    dedupe: set[Tuple[str, str, int, int]] = set()

    for idx, (line, line_norm) in enumerate(zip(lines, lines_norm)):
        if not line_norm:
            continue
        matched_anchor_ids = _match_anchor_ids(line_norm, prepared_catalog)
        if not matched_anchor_ids:
            continue

        columns = _line_columns(line)
        inline_answer = _extract_inline_answer(line)

        for anchor_id in matched_anchor_ids:
            anchor_cfg = prepared_catalog.get(anchor_id) or {}
            anchor_type = str(anchor_cfg.get("type") or "single_line")
            answer_text = ""
            line_end = idx
            confidence = 0.0

            if columns and len(columns) > 1 and (anchor_type == "table_row" or "|" in line or _MULTI_SPACE_COL_SPLIT_RE.search(line)):
                answer_text = _clean_answer_text(" ".join(columns[1:]))
                confidence = 0.90 if _is_meaningful_answer(answer_text, anchor_text=line) else 0.55
            elif _is_meaningful_answer(inline_answer, anchor_text=line):
                answer_text = inline_answer
                confidence = 0.92
            elif anchor_type in ("multi_line", "table_row"):
                answer_text, line_end = _collect_multiline_answer(
                    lines,
                    lines_norm,
                    idx,
                    prepared_catalog,
                    max_follow_lines=max_follow_lines,
                    current_anchor_id=anchor_id,
                )
                confidence = 0.84 if _is_meaningful_answer(answer_text, anchor_text=line) else 0.52
            else:
                confidence = 0.50

            block = {
                "anchor_id": anchor_id,
                "anchor_text": (line or "").strip()[:240],
                "answer_text": answer_text[:420],
                "confidence": round(float(confidence), 4),
                "source": source,
                "span": {"line_start": int(idx + 1), "line_end": int(line_end + 1)},
            }
            dedupe_key = (
                str(block["anchor_id"]),
                normalize_text(str(block["answer_text"])),
                int(block["span"]["line_start"]),
                int(block["span"]["line_end"]),
            )
            if dedupe_key in dedupe:
                continue
            dedupe.add(dedupe_key)

            block_idx = len(blocks)
            blocks.append(block)
            qa_index.setdefault(anchor_id, []).append(block_idx)

    return blocks, qa_index


def resolve_question_from_qa_blocks(
    question_code: str,
    qa_blocks: Sequence[Dict[str, object]],
    qa_index: Dict[str, List[int]],
    question_anchor_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    anchor_map = question_anchor_map or QUESTION_ANCHOR_MAP
    anchor_ids = list(anchor_map.get(question_code) or [])
    if not anchor_ids:
        return {
            "answerable": 0,
            "source": "qa_block",
            "evidence": [],
            "used_chunk_ids": [],
            "reason": "no_anchor_mapping",
        }

    candidates: List[Tuple[float, int, str, Dict[str, object]]] = []

    for anchor_id in anchor_ids:
        for idx in qa_index.get(anchor_id, []) or []:
            if idx < 0 or idx >= len(qa_blocks):
                continue

            block = dict(qa_blocks[idx] or {})
            anchor_text = str(block.get("anchor_text") or "").strip()
            answer_text = _clean_answer_text(str(block.get("answer_text") or ""))

            candidate_text = (f"{anchor_text} {answer_text}".strip() or anchor_text).strip()
            answer_payload = answer_text or candidate_text

            # corta recomendações/gaps/tarefas, que não são resposta factual
            if _looks_like_gap_or_task(anchor_text):
                continue
            if _looks_like_gap_or_task(candidate_text):
                continue

            # valida primeiro o payload da resposta, não "pergunta + resposta"
            if not _is_meaningful_answer(answer_payload, anchor_text=anchor_text):
                continue

            confidence = float(block.get("confidence", 0.0) or 0.0)
            score = confidence + min(0.08, len(answer_payload) / 1000.0)

            candidates.append((score, idx, anchor_id, block))

    if not candidates:
        return {
            "answerable": 0,
            "source": "qa_block",
            "evidence": [],
            "used_chunk_ids": [],
            "reason": "no_filled_anchor_answer",
        }

    candidates.sort(key=lambda item: item[0], reverse=True)

    last_reason = "qa_guard_rejected"

    for _, block_idx, anchor_id, block in candidates:
        anchor_text = str(block.get("anchor_text") or "").strip()
        answer_text = _clean_answer_text(str(block.get("answer_text") or ""))

        # evidência principal = resposta
        # chunk completo = ancora + resposta (para permitir auto-repair quando necessário)
        evidence_text = (answer_text or anchor_text)[:240]
        chunk_payload = (f"{anchor_text} {answer_text}".strip() or anchor_text)[:500]

        used_chunk_id = f"qa:{anchor_id}:{block_idx}"

        validated_answerable, validated_reason, validated_evidence = validate_answerability(
            question_code=question_code,
            answerable=1,
            evidence=[evidence_text],
            used_ids=[used_chunk_id],
            chunk_texts={used_chunk_id: chunk_payload},
            question_text="",
        )

        if int(validated_answerable) == 1:
            return {
                "answerable": int(validated_answerable),
                "source": "qa_block",
                "evidence": list(validated_evidence),
                "used_chunk_ids": [used_chunk_id],
                "reason": validated_reason or "",
                "anchor_id": anchor_id,
                "confidence": float(block.get("confidence", 0.0) or 0.0),
            }

        metric = reason_to_metric(validated_reason)
        if metric:
            logging.info(
                "%s=1 source=qa_block question_code=%s reason=%s",
                metric,
                question_code,
                validated_reason,
            )

        last_reason = validated_reason or last_reason

    return {
        "answerable": 0,
        "source": "qa_block",
        "evidence": [],
        "used_chunk_ids": [],
        "reason": last_reason,
    }