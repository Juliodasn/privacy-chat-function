from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 800
MAX_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 120
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_RETRIEVAL_TOP_K = 4
DEFAULT_EMBED_MAX_INPUT_CHARS = 6000
DEFAULT_RAG_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_RAG_BATCH_SIZE = 6
DEFAULT_RAG_CONTEXT_MAX_CHARS = 12000
DEFAULT_RAG_CONTEXT_MAX_TOKENS = 3000


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    section: str
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _is_nonempty(value: Any) -> bool:
    text = _clean_text(value)
    return bool(text and text.lower() != "nan")


def _is_dataframe_like(value: Any) -> bool:
    return (
        value is not None
        and hasattr(value, "columns")
        and hasattr(value, "itertuples")
        and hasattr(value, "empty")
    )


def _normalize_chunk_size(chunk_size: int) -> int:
    try:
        value = int(chunk_size)
    except Exception:
        value = DEFAULT_CHUNK_SIZE
    value = max(MIN_CHUNK_SIZE, value)
    value = min(MAX_CHUNK_SIZE, value)
    return value


def _normalize_overlap(overlap: int, chunk_size: int) -> int:
    try:
        value = int(overlap)
    except Exception:
        value = DEFAULT_OVERLAP
    value = max(0, value)
    # Keep overlap light to avoid excessive duplication.
    return min(value, max(40, chunk_size // 5))


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    if value < low:
        return low
    if value > high:
        return high
    return value


def _table_block_from_section(section: str, table: Any) -> Optional[Dict[str, Any]]:
    if not _is_dataframe_like(table):
        return None
    if getattr(table, "empty", True):
        return None

    try:
        columns = [str(col) for col in table.columns]
    except Exception:
        return None

    rows: List[str] = []
    row_count = 0
    try:
        for row in table.itertuples(index=False, name=None):
            pairs: List[str] = []
            for col_name, value in zip(columns, row):
                if not _is_nonempty(value):
                    continue
                pairs.append(f"{col_name}: {_clean_text(value)}")
            if pairs:
                rows.append("; ".join(pairs))
                row_count += 1
    except Exception:
        return None

    if not rows:
        return None

    return {
        "section": section,
        "source": f"table:{section}",
        "text": "\n".join(rows),
        "metadata": {"kind": "table", "rows": row_count},
    }


def _iter_table_blocks(structured: Optional[dict]) -> Iterable[Dict[str, Any]]:
    if not isinstance(structured, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    for section in ("areas", "processos", "sistemas"):
        block = _table_block_from_section(section, structured.get(section))
        if block:
            blocks.append(block)
    return blocks


def _split_free_text_blocks(text: str) -> List[str]:
    base = _clean_text(text)
    if not base:
        return []

    by_paragraph = [p.strip() for p in re.split(r"\n\s*\n+", base) if p.strip()]
    if len(by_paragraph) > 1:
        return by_paragraph

    lines = [ln.strip() for ln in base.split("\n") if ln.strip()]
    if not lines:
        return [base]

    blocks: List[str] = []
    buf: List[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) >= 6:
            blocks.append("\n".join(buf))
            buf = []
    if buf:
        blocks.append("\n".join(buf))
    return blocks or [base]


def _find_split_boundary(text: str, low: int, high: int) -> int:
    if high <= low:
        return high

    priorities = ("\n\n", "\n", ". ", "; ", ", ", " ")
    for token in priorities:
        idx = text.rfind(token, low, high)
        if idx >= low:
            return idx + len(token)
    return high


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    content = _clean_text(text)
    if not content:
        return []

    if len(content) <= chunk_size:
        return [content]

    parts: List[str] = []
    start = 0
    total = len(content)

    while start < total:
        hard_end = min(start + chunk_size, total)
        if hard_end >= total:
            tail = content[start:].strip()
            if tail:
                parts.append(tail)
            break

        min_end = min(total, start + MIN_CHUNK_SIZE)
        if min_end >= hard_end:
            split_at = hard_end
        else:
            split_at = _find_split_boundary(content, min_end, hard_end)

        piece = content[start:split_at].strip()
        if piece:
            parts.append(piece)

        if split_at <= start:
            split_at = hard_end

        start = split_at - overlap if overlap > 0 else split_at
        if start < 0:
            start = 0
        if start >= total:
            break

    if len(parts) >= 2 and len(parts[-1]) < MIN_CHUNK_SIZE // 3:
        parts[-2] = f"{parts[-2]}\n{parts[-1]}".strip()
        parts.pop()

    return parts


def _make_chunk_id(section: str, source: str, text: str, ordinal: int) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    safe_section = re.sub(r"[^a-z0-9_-]", "-", (section or "geral").lower())
    safe_source = re.sub(r"[^a-z0-9_-]", "-", (source or "text").lower())
    return f"{safe_section}-{safe_source}-{ordinal:04d}-{digest}"


def _normalize_embedding(vec: Sequence[Any]) -> List[float]:
    floats: List[float] = []
    for item in vec:
        try:
            floats.append(float(item))
        except Exception:
            floats.append(0.0)
    norm = math.sqrt(sum(v * v for v in floats))
    if norm <= 0:
        return []
    return [v / norm for v in floats]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _truncate_embedding_input(text: str) -> str:
    max_chars = _env_int("RAG_EMBED_MAX_INPUT_CHARS", DEFAULT_EMBED_MAX_INPUT_CHARS, 512, 20000)
    content = _clean_text(text)
    if len(content) <= max_chars:
        return content
    return content[:max_chars]


def _resolve_embedding_client(explicit_model: Optional[str]) -> Tuple[Any, Optional[str], str]:
    try:
        from openai import OpenAI, AzureOpenAI
    except Exception:
        return None, explicit_model or DEFAULT_EMBEDDING_MODEL, "unavailable"

    azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    azure_key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
    if azure_endpoint and azure_key:
        model = (
            explicit_model
            or os.getenv("RAG_EMBEDDING_MODEL")
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return client, model, "azure_openai"

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_key:
        model = (
            explicit_model
            or os.getenv("RAG_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        client = OpenAI(api_key=openai_key)
        return client, model, "openai"

    model = (
        explicit_model
        or os.getenv("RAG_EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    return None, model, "none"


def _embed_texts(client: Any, model: str, texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    if client is None or not model:
        return [[] for _ in texts]

    batch_size = _env_int("RAG_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE, 1, 128)
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = [_truncate_embedding_input(t) for t in texts[start : start + batch_size]]
        try:
            resp = client.embeddings.create(model=model, input=batch)
        except Exception as ex:
            logging.exception("RAG embedding batch failed at start=%s: %s", start, ex)
            vectors.extend([[] for _ in batch])
            continue

        out_batch: List[List[float]] = [[] for _ in batch]
        for row in getattr(resp, "data", []) or []:
            try:
                idx = int(getattr(row, "index", 0))
            except Exception:
                idx = 0
            if idx < 0 or idx >= len(out_batch):
                continue
            out_batch[idx] = _normalize_embedding(getattr(row, "embedding", []) or [])
        vectors.extend(out_batch)

    if len(vectors) < len(texts):
        vectors.extend([[] for _ in range(len(texts) - len(vectors))])
    elif len(vectors) > len(texts):
        vectors = vectors[: len(texts)]
    return vectors


def build_chunks(
    text_norm: str,
    structured: Optional[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[RagChunk]:
    """
    Build semantically grouped chunks.

    Strategy:
    - prioritise section tables (areas/processos/sistemas) when available
    - then include free text split by paragraph/block
    - enforce chunk size in [800, 1500] with a light overlap
    """
    size = _normalize_chunk_size(chunk_size)
    ov = _normalize_overlap(overlap, size)

    blocks: List[Dict[str, Any]] = []
    blocks.extend(_iter_table_blocks(structured))

    raw_text = ""
    if isinstance(structured, dict):
        raw_text = _clean_text(structured.get("__raw_text_src__") or structured.get("__raw_text__") or "")
    if not raw_text:
        raw_text = _clean_text(text_norm)

    for idx, paragraph in enumerate(_split_free_text_blocks(raw_text), start=1):
        blocks.append(
            {
                "section": "geral",
                "source": "text:block",
                "text": paragraph,
                "metadata": {"kind": "text", "block": idx},
            }
        )

    if not blocks and _clean_text(text_norm):
        blocks.append(
            {
                "section": "geral",
                "source": "text:fallback",
                "text": _clean_text(text_norm),
                "metadata": {"kind": "text", "fallback": True},
            }
        )

    chunks: List[RagChunk] = []
    seen_hashes: set[str] = set()
    ordinal = 1

    for block in blocks:
        section = str(block.get("section") or "geral")
        source = str(block.get("source") or "text")
        metadata = dict(block.get("metadata") or {})
        for piece in _chunk_text(str(block.get("text") or ""), size, ov):
            digest = hashlib.sha1(piece.encode("utf-8", errors="ignore")).hexdigest()[:10]
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            chunk = RagChunk(
                chunk_id=_make_chunk_id(section, source, piece, ordinal),
                section=section,
                source=source,
                text=piece,
                metadata=metadata,
            )
            chunks.append(chunk)
            ordinal += 1

    return chunks


def build_rag_index(
    text_norm: str,
    structured: Optional[dict],
    chunks: Optional[Sequence[RagChunk]] = None,
    embedding_model: Optional[str] = None,
    question_sections: Optional[Dict[str, str]] = None,
    embed_chunks: bool = True,
) -> Dict[str, Any]:
    """
    Build in-memory RAG index with chunks + embeddings.
    """
    resolved_chunks = list(chunks) if chunks is not None else build_chunks(text_norm, structured)

    section_counts: Dict[str, int] = {}
    section_positions: Dict[str, List[int]] = {}
    for pos, chunk in enumerate(resolved_chunks):
        section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1
        section_positions.setdefault(chunk.section, []).append(pos)

    started_at = time.time()
    client, resolved_model, provider = _resolve_embedding_client(embedding_model)
    vectors: List[List[float]] = [[] for _ in resolved_chunks]
    has_embeddings = False

    if embed_chunks and resolved_chunks and client is not None and resolved_model:
        vectors = _embed_texts(client, resolved_model, [c.text for c in resolved_chunks])
        has_embeddings = any(bool(v) for v in vectors)

    elapsed_ms = int((time.time() - started_at) * 1000)
    first_vec = next((v for v in vectors if v), [])

    return {
        "version": "rag-index-v1",
        "embedding_model": resolved_model,
        "embedding_provider": provider,
        "has_embeddings": has_embeddings,
        "embedding_dim": len(first_vec),
        "chunks": resolved_chunks,
        "embeddings": vectors,
        "section_positions": section_positions,
        "question_sections": dict(question_sections or {}),
        "chunk_count": len(resolved_chunks),
        "section_counts": section_counts,
        "embed_elapsed_ms": elapsed_ms,
        "_embedding_client": client,
        "_question_embedding_cache": {},
    }


def _resolve_question_section(question_code: str, index: Dict[str, Any]) -> Optional[str]:
    mapping = index.get("question_sections") or {}
    section = mapping.get(question_code)
    if section in ("areas", "processos", "sistemas"):
        return section
    return None


def _question_embedding(question_code: str, question_text: str, index: Dict[str, Any]) -> List[float]:
    cache = index.setdefault("_question_embedding_cache", {})
    cache_key = f"{question_code}::{_clean_text(question_text)}"
    if cache_key in cache:
        return cache[cache_key]

    client = index.get("_embedding_client")
    model = index.get("embedding_model")
    vecs = _embed_texts(client, model, [question_text]) if client is not None and model else [[]]
    vec = vecs[0] if vecs else []
    cache[cache_key] = vec
    return vec


def retrieve_context_for_question(
    question_code: str,
    question_text: str,
    index: Dict[str, Any],
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Return top-k most similar chunks for a question.

    The section filter is inferred from index["question_sections"] by question_code.
    """
    if not isinstance(index, dict):
        return []
    if not index.get("has_embeddings"):
        return []

    chunks: List[RagChunk] = list(index.get("chunks") or [])
    vectors: List[List[float]] = list(index.get("embeddings") or [])
    if not chunks or not vectors:
        return []

    try:
        k = int(top_k)
    except Exception:
        k = DEFAULT_RETRIEVAL_TOP_K
    k = max(1, min(k, 8))

    question_vec = _question_embedding(question_code, question_text, index)
    if not question_vec:
        return []

    section = _resolve_question_section(question_code, index)
    section_positions = index.get("section_positions") or {}
    candidate_positions = list(section_positions.get(section, [])) if section else list(range(len(chunks)))
    filter_fallback = False
    if not candidate_positions:
        candidate_positions = list(range(len(chunks)))
        filter_fallback = True

    scored: List[Dict[str, Any]] = []
    for pos in candidate_positions:
        if pos < 0 or pos >= len(chunks) or pos >= len(vectors):
            continue
        vec = vectors[pos]
        if not vec:
            continue
        score = _dot(question_vec, vec)
        chunk = chunks[pos]
        scored.append(
            {
                "chunk_id": chunk.chunk_id,
                "score": score,
                "section": chunk.section,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "section_filter": section,
                "section_filter_fallback": filter_fallback,
            }
        )

    scored.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return scored[:k]


def _resolve_chat_client(explicit_model: Optional[str]) -> Tuple[Any, Optional[str], str]:
    try:
        from openai import OpenAI, AzureOpenAI
    except Exception:
        return None, explicit_model or DEFAULT_RAG_CHAT_MODEL, "unavailable"

    azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    azure_key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
    if azure_endpoint and azure_key:
        model = (
            explicit_model
            or os.getenv("RAG_EVAL_MODEL")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_RAG_CHAT_MODEL
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return client, model, "azure_openai"

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_key:
        model = (
            explicit_model
            or os.getenv("RAG_EVAL_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_RAG_CHAT_MODEL
        )
        client = OpenAI(api_key=openai_key)
        return client, model, "openai"

    model = explicit_model or os.getenv("RAG_EVAL_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_RAG_CHAT_MODEL
    return None, model, "none"


def _sanitize_rag_json_response(data: Any, allowed_chunk_ids: set[str]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"answerable": 0, "evidence": [], "used_chunk_ids": [], "reason": "invalid_json"}

    answer_raw = data.get("answerable", 0)
    answerable = 1 if answer_raw in (1, True, "1", "true", "sim", "yes") else 0

    evidence: List[str] = []
    for ev in data.get("evidence", []) or []:
        txt = _clean_text(ev)
        if not txt:
            continue
        if txt not in evidence:
            evidence.append(txt[:240])
        if len(evidence) >= 2:
            break

    used_chunk_ids: List[str] = []
    for cid in data.get("used_chunk_ids", []) or []:
        cid_txt = str(cid).strip()
        if not cid_txt:
            continue
        if cid_txt in allowed_chunk_ids and cid_txt not in used_chunk_ids:
            used_chunk_ids.append(cid_txt)
        if len(used_chunk_ids) >= 4:
            break

    reason = _clean_text(data.get("reason", ""))[:240]
    if answerable == 1 and not used_chunk_ids:
        answerable = 0
        reason = reason or "missing_chunk_trace"
    return {
        "answerable": answerable,
        "evidence": evidence,
        "used_chunk_ids": used_chunk_ids,
        "reason": reason,
    }


def _estimate_tokens(text: str) -> int:
    # Lightweight estimate to keep context bounded in serverless runtime.
    return max(1, int(math.ceil(len(text) / 4)))


def _iter_batches(items: Sequence[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield list(items[start : start + step])


def _build_batch_context(
    batch_questions: Sequence[Tuple[str, str]],
    question_hits: Dict[str, List[Dict[str, Any]]],
    max_context_chars: int,
    max_context_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, set[str]], bool]:
    chunk_map: Dict[str, Dict[str, Any]] = {}
    q_allowed: Dict[str, set[str]] = {code: set() for code, _ in batch_questions}

    for code, _ in batch_questions:
        for hit in question_hits.get(code, []) or []:
            cid = str(hit.get("chunk_id") or "").strip()
            if not cid:
                continue
            q_allowed[code].add(cid)
            score = float(hit.get("score", 0.0))
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "section": str(hit.get("section") or ""),
                    "source": str(hit.get("source") or ""),
                    "text": _clean_text(hit.get("text", "")),
                    "score": score,
                    "used_by": {code},
                }
            else:
                chunk_map[cid]["score"] = max(float(chunk_map[cid].get("score", 0.0)), score)
                chunk_map[cid]["used_by"].add(code)

    ordered_chunks = sorted(chunk_map.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
    selected: List[Dict[str, Any]] = []
    used_chars = 0
    used_tokens = 0
    truncated = False

    for row in ordered_chunks:
        context_line = (
            f"[chunk_id={row['chunk_id']} score={float(row['score']):.6f} section={row['section']} source={row['source']}] "
            f"{str(row['text'])[:1600]}"
        )
        line_chars = len(context_line)
        line_tokens = _estimate_tokens(context_line)

        if selected and (used_chars + line_chars > max_context_chars or used_tokens + line_tokens > max_context_tokens):
            truncated = True
            continue
        selected.append(
            {
                "chunk_id": row["chunk_id"],
                "section": row["section"],
                "source": row["source"],
                "text": str(row["text"])[:1600],
                "score": float(row["score"]),
                "used_by": sorted(list(row["used_by"])),
            }
        )
        used_chars += line_chars
        used_tokens += line_tokens

    return selected, q_allowed, truncated


def evaluate_questions_with_rag(
    questions: Sequence[Tuple[str, str]],
    index: Dict[str, Any],
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_context_chars: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate questions with retrieved RAG context.

    Returns:
    {
      "results": { "<code>": {"answerable":0|1, "evidence":[...], "used_chunk_ids":[...], "reason": "...", "rag_chunks":[...]} },
      "meta": {"provider":"...", "model":"...", "evaluated":N, "elapsed_ms":MS}
    }
    """
    started_at = time.time()
    out: Dict[str, Dict[str, Any]] = {}
    question_list = [(str(code), str(text or "")) for code, text in (questions or [])]
    client, resolved_model, provider = _resolve_chat_client(model)

    batch_size_resolved = (
        max(2, min(int(batch_size), 8))
        if batch_size is not None
        else _env_int("RAG_QUESTION_BATCH_SIZE", DEFAULT_RAG_BATCH_SIZE, 2, 8)
    )
    context_max_chars = (
        max(2000, min(int(max_context_chars), 60000))
        if max_context_chars is not None
        else _env_int("RAG_CONTEXT_MAX_CHARS", DEFAULT_RAG_CONTEXT_MAX_CHARS, 2000, 60000)
    )
    context_max_tokens = (
        max(500, min(int(max_context_tokens), 12000))
        if max_context_tokens is not None
        else _env_int("RAG_CONTEXT_MAX_TOKENS", DEFAULT_RAG_CONTEXT_MAX_TOKENS, 500, 12000)
    )

    system_prompt = (
        "Voce e um avaliador LGPD. Responda APENAS JSON estrito no formato "
        "{\"answers\":{\"<question_code>\":{\"answerable\":0|1,\"evidence\":[...],\"used_chunk_ids\":[...],\"reason\":\"...\"}}}. "
        "answerable=1 somente com evidencia textual explicita nos chunks."
    )

    # Retrieval stage (question -> top-k chunks)
    retrieval_started = time.time()
    question_hits: Dict[str, List[Dict[str, Any]]] = {}
    retrieval_times_ms: List[int] = []
    for question_code, question_text in question_list:
        t0 = time.time()
        hits = retrieve_context_for_question(question_code, question_text, index, top_k=top_k)
        retrieval_times_ms.append(int((time.time() - t0) * 1000))
        question_hits[question_code] = hits
        out[question_code] = {
            "answerable": 0,
            "evidence": [],
            "used_chunk_ids": [],
            "reason": "not_evaluated",
            "rag_chunks": [
                {
                    "chunk_id": hit.get("chunk_id"),
                    "score": round(float(hit.get("score", 0.0)), 6),
                    "section": hit.get("section"),
                    "source": hit.get("source"),
                    "trecho": _clean_text(hit.get("text", ""))[:260],
                }
                for hit in hits
            ],
        }
    retrieval_elapsed_ms = int((time.time() - retrieval_started) * 1000)

    if client is None or not resolved_model:
        for question_code, _ in question_list:
            if not question_hits.get(question_code):
                out[question_code]["reason"] = "no_retrieved_context"
            else:
                out[question_code]["reason"] = "no_chat_client"
        elapsed_ms = int((time.time() - started_at) * 1000)
        return {
            "results": out,
            "meta": {
                "provider": provider,
                "model": resolved_model,
                "evaluated": len(question_list),
                "batch_size": batch_size_resolved,
                "top_k": top_k,
                "context_max_chars": context_max_chars,
                "context_max_tokens": context_max_tokens,
                "retrieval_ms_total": retrieval_elapsed_ms,
                "retrieval_ms_per_question": retrieval_times_ms,
                "llm_calls": 0,
                "llm_call_ms": [],
                "llm_ms_total": 0,
                "elapsed_ms": elapsed_ms,
            },
        }

    llm_call_ms: List[int] = []
    truncated_batches = 0
    evaluated_questions = 0

    for batch in _iter_batches(question_list, batch_size_resolved):
        nonempty_batch = [(code, text) for code, text in batch if question_hits.get(code)]
        for code, _ in batch:
            if not question_hits.get(code):
                out[code]["reason"] = "no_retrieved_context"
        if not nonempty_batch:
            continue

        batch_chunks, q_allowed, is_truncated = _build_batch_context(
            nonempty_batch,
            question_hits,
            max_context_chars=context_max_chars,
            max_context_tokens=context_max_tokens,
        )
        if is_truncated:
            truncated_batches += 1

        if not batch_chunks:
            for code, _ in nonempty_batch:
                out[code]["reason"] = "no_context_after_limit"
            continue

        selected_chunk_ids = [str(row["chunk_id"]) for row in batch_chunks]
        context_lines = [
            f"[chunk_id={row['chunk_id']} score={float(row['score']):.6f} section={row['section']} source={row['source']}] {row['text']}"
            for row in batch_chunks
        ]
        questions_payload = {code: text for code, text in nonempty_batch}
        user_prompt = (
            "QUESTIONS_JSON:\n"
            + json.dumps(questions_payload, ensure_ascii=False)
            + "\n\nCONTEXT_CHUNKS:\n"
            + "\n\n".join(context_lines)
            + "\n\nRegras:\n"
            + "- Retorne answers para todos os question_code.\n"
            + "- used_chunk_ids deve conter somente chunk_ids do contexto.\n"
            + "- Se nao houver evidencia explicita, answerable=0."
        )

        response_data: Any = {}
        call_ms = 0
        try:
            call_t0 = time.time()
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=950,
            )
            call_ms = int((time.time() - call_t0) * 1000)
            content = (resp.choices[0].message.content or "{}").strip()
            response_data = json.loads(content)
        except Exception as ex:
            logging.exception(
                "RAG batch eval failed for questions=%s: %s",
                [code for code, _ in nonempty_batch],
                ex,
            )
            response_data = {"answers": {}}
        llm_call_ms.append(call_ms)

        answers = (response_data or {}).get("answers", {}) if isinstance(response_data, dict) else {}
        allowed_union = set(selected_chunk_ids)
        for code, _ in nonempty_batch:
            allowed_for_question = set(q_allowed.get(code, set())) & allowed_union
            if not allowed_for_question:
                out[code]["answerable"] = 0
                out[code]["evidence"] = []
                out[code]["used_chunk_ids"] = []
                out[code]["reason"] = "no_context_for_question_after_limit"
                continue
            raw = answers.get(code, {}) if isinstance(answers, dict) else {}
            normalized = _sanitize_rag_json_response(raw, allowed_for_question)
            normalized["rag_chunks"] = out[code].get("rag_chunks", [])
            out[code] = normalized
            evaluated_questions += 1

    elapsed_ms = int((time.time() - started_at) * 1000)
    return {
        "results": out,
        "meta": {
            "provider": provider,
            "model": resolved_model,
            "evaluated": len(question_list),
            "evaluated_with_context": evaluated_questions,
            "batch_size": batch_size_resolved,
            "top_k": top_k,
            "context_max_chars": context_max_chars,
            "context_max_tokens": context_max_tokens,
            "truncated_batches": truncated_batches,
            "retrieval_ms_total": retrieval_elapsed_ms,
            "retrieval_ms_per_question": retrieval_times_ms,
            "llm_calls": len(llm_call_ms),
            "llm_call_ms": llm_call_ms,
            "llm_ms_total": int(sum(llm_call_ms)),
            "elapsed_ms": elapsed_ms,
        },
    }


def chunk_text_for_llm(text: str, chunk_size: int) -> List[str]:
    """
    Backward-compatible chunking used by the current LLM evaluation flow.
    """
    content = text or ""
    try:
        size = int(chunk_size)
    except Exception:
        size = 1
    size = max(1, size)
    return [content[i : i + size] for i in range(0, len(content), size)] or [""]
