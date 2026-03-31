from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .answerability_guard import (
    bump_metric,
    is_too_generic,
    is_value_like,
    looks_like_prompt,
    reason_to_metric,
    strict_mode_enabled,
    validate_answerability,
)

DEFAULT_CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 800
MAX_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 120
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_RETRIEVAL_TOP_K = 4
DEFAULT_RETRIEVAL_EMBED_TOP_K = 12
DEFAULT_RETRIEVAL_ALPHA = 0.65
DEFAULT_EMBED_MAX_INPUT_CHARS = 6000
DEFAULT_RAG_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_RAG_BATCH_SIZE = 6
DEFAULT_RAG_CONTEXT_MAX_CHARS = 12000
DEFAULT_RAG_CONTEXT_MAX_TOKENS = 3000
QUESTION_KEYWORD_STOPWORDS = {
    "analisar",
    "answer",
    "avaliar",
    "can",
    "como",
    "does",
    "eh",
    "essa",
    "esse",
    "esta",
    "este",
    "existe",
    "ha",
    "how",
    "identificar",
    "is",
    "it",
    "its",
    "pode",
    "possible",
    "possivel",
    "possui",
    "qual",
    "quais",
    "question",
    "ser",
    "tem",
    "there",
    "what",
    "which",
}
QUESTION_DOMAIN_HINTS = {
    "acesso",
    "api",
    "armazenado",
    "armazenamento",
    "cibernetica",
    "classificacao",
    "compartilhamento",
    "compartilhar",
    "comprovadas",
    "contrato",
    "cpf",
    "criptografia",
    "dados",
    "descarte",
    "documentados",
    "email",
    "fisico",
    "fornecedor",
    "nda",
    "nuvem",
    "paises",
    "permanente",
    "politica",
    "processados",
    "processo",
    "protecao",
    "rede",
    "restricao",
    "retencao",
    "seguranca",
    "senha",
    "sigilo",
    "sistema",
    "terceiros",
    "transferencia",
}

LEXICAL_STOPWORDS = {
    "a", "o", "os", "as", "de", "do", "da", "dos", "das", "e", "ou", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "para", "por", "com", "sem", "que", "se", "ao", "aos", "a", "ha", "eh",
    "esta", "este", "isso", "isto", "sobre", "entre", "cada", "pelos", "pelas",
}
EXPLICIT_ANSWER_TOKENS = {
    "sim",
    "nao",
    "n/a",
    "na",
    "nao se aplica",
    "nao informado",
    "permanente",
    "apresentado",
}

SCHEMA_SIGNAL_YES = {"sim", "yes", "true", "1"}
SCHEMA_SIGNAL_NO = {"nao", "no", "false", "0"}
SCHEMA_SIGNAL_NA = {
    "n/a",
    "na",
    "nao se aplica",
    "nao aplicavel",
    "nao informado",
    "n.d.",
    "nd",
    "n d",
}

# Deterministic schema/table matching rules applied before chat completion.
SCHEMA_TABLE_RULES: Dict[str, Dict[str, Any]] = {
    "2_compart_dados_rede": {
        "header_terms": ["rede", "compart"],
        "value_terms": ["rede interna", "rede corporativa", "rede pessoal interna", "intranet", "lan", "vpn"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "2_2_politica_restricao_acesso": {
        "header_terms": ["acesso", "restricao", "controle", "perfil", "politica", "politica de acesso"],
        "value_terms": [
            "gestao de acesso",
            "controle de acesso",
            "perfil de acesso",
            "acesso restrito",
            "acesso por area",
            "segregacao de acesso",
            "liderancas",
            "controller",
            "financeiro",
        ],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "4_1_forma_compart": {
        "header_terms": ["compart", "forma", "canal", "transfer"],
        "value_terms": ["software", "email", "e-mail", "api", "arquivo", "formulario", "impresso", "integracao", "rede"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 1,
    },
    "4_2_termo_sigilo_terceiros": {
        "header_terms": ["nda", "sigilo", "confidencialidade"],
        "value_terms": ["nda", "confidencial", "n/a", "nao se aplica", "nao aplicavel", "nao informado", "n.d."],
        "allow_text": True,
        "header_optional": False,
        "min_tokens": 1,
    },
    "6_dados_tratados_quais": {
        "header_terms": [
            "detalhamento dos dados tratados",
            "dados tratados",
            "descricao do dado",
            "descrição do dado",
            "dados coletados",
            "dados utilizados",
        ],
        "value_terms": [
            "cpf",
            "nome",
            "nome completo",
            "email",
            "e-mail",
            "telefone",
            "endereco",
            "endereço",
            "data de nascimento",
            "foto",
            "genero",
            "gênero",
            "estado civil",
            "cargo",
            "funcao",
            "função",
            "dados bancarios",
            "dados bancários",
            "voz",
            "biometria",
            "saude",
            "saúde",
        ],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "6_classificacao_documentos": {
        "header_terms": [
            "classificacao de documentos",
            "classificacao da informacao",
            "classificacao documental",
            "nivel de sigilo",
            "informacao classificada",
        ],
        "value_terms": [
            "uso interno",
            "documento confidencial",
            "documento restrito",
            "documento sigiloso",
            "documento publico",
            "confidencial",
            "restrito",
            "sigiloso",
            "publico",
        ],
        "allow_text": True,
        "header_optional": False,
        "min_tokens": 1,
    },
    "3_trata_dados_sensiveis": {
        "header_terms": ["categoria", "dados", "sensivel", "pessoal"],
        "value_terms": ["pessoal", "sensivel", "dados pessoais", "dados sensiveis"],
        "allow_text": True,
        "header_optional": False,
        "min_tokens": 1,
    },
    "5_descricao_processo": {
        "header_terms": ["descricao", "processo", "etapa"],
        "value_terms": ["entrevista", "admissao", "rescisao", "processo interno", "relatorio", "proposta"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "6_1_fisico_ou_digital": {
        "header_terms": ["fisico", "digital", "canal", "formato", "coleta", "meio"],
        "value_terms": ["impresso", "formulario", "email", "telefone", "banco de dados", "sistema", "software", "digital", "fisico"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 1,
    },
    "6_3_compart_interno_terceiros": {
        "header_terms": ["compart", "interno", "terceiro", "acesso"],
        "value_terms": ["interno", "terceiro", "rh", "rede pessoal interna", "rede interna", "liderancas"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 1,
    },
    "6_4_descarte_apos_uso": {
        "header_terms": ["descarte", "eliminacao", "retencao", "exclusao"],
        "value_terms": ["exclusao", "eliminacao", "descarte", "retencao", "apos uso", "sempre"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 1,
    },
    "9_somente_quando_necessario": {
        "header_terms": [
            "necessario",
            "estritamente",
            "objetivo",
            "minimizacao",
        ],
        "value_terms": [
            "estritamente necessario",
            "estritamente necessarios",
            "somente quando necessario",
            "apenas quando necessario",
            "minimizacao",
            "minimizacao de dados",
            "atinge o seu objetivo",
            "objetivo do processo",
            "sim",
            "nao",
            "n/a",
            "nao se aplica",
            "nao aplicavel",
            "nao informado",
        ],
        "allow_text": False,
        "header_optional": False,
        "min_tokens": 2,
    },
        "12_formato_dados_disponiveis": {
        "header_terms": ["formato", "disponivel", "meio", "canal", "suporte", "armazenamento", "local"],
        "value_terms": [
            "email",
            "e-mail",
            "telefone",
            "whatsapp",
            "banco de dados",
            "sistema",
            "software",
            "portal",
            "arquivo",
            "planilha",
            "documento",
            "word",
            "pdf",
            "digital",
            "fisico",
            "formulario",
            "impresso",
            "pasta",
            "drive",
            "google drive",
            "onedrive",
            "pen drive",
            "nuvem",
        ],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 1,
    },
    "13_forma_transferencia_dados": {
        "header_terms": ["transfer", "forma", "canal", "meio"],
        "value_terms": ["rede interna", "intranet", "email", "e-mail", "api", "arquivo", "software", "integracao"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 1,
    },
    "14_processos_documentados": {
        "header_terms": ["processo", "documentado", "descricao"],
        "value_terms": ["processo interno", "relatorio", "proposta", "entrevista", "admissao", "rescisao", "documentado"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "17_medidas_seg_cibernetica": {
        "header_terms": ["seguranca", "cibernetica", "acesso", "controle", "medidas"],
        "value_terms": ["gestao de acesso", "controle de acesso", "perfil", "restricao", "liderancas", "controller", "financeiro", "senha"],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "17_2_medidas_seg_descritas": {
        "header_terms": ["seguranca", "cibernetica", "acesso", "controle", "descricao"],
        "value_terms": [
            "gestao de acesso",
            "controle de acesso",
            "perfil de acesso",
            "protegido por senha",
            "senha de acesso",
            "senha unica",
            "senha",
            "acesso restrito",
            "armario trancado",
            "fechada diariamente",
            "backup",
            "antivirus",
            "antivírus",
            "firewall",
            "mfa",
        ],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
        "18_medidas_admin_protecao": {
        "header_terms": ["administrativa", "protecao", "acesso", "controle", "medidas", "politica", "procedimento"],
        "value_terms": [
            "gestao de acesso",
            "controle de acesso",
            "perfil",
            "restricao",
            "liderancas",
            "controller",
            "financeiro",
            "politica de organizacao documental",
            "procedimento",
            "treinamento de acesso",
            "hierarquia de acesso",
            "permitido apenas",
            "exclusivamente por e-mail",
            "exclusivamente por whatsapp",
            "termo de adesao",
        ],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "18_2_admin_descritas": {
        "header_terms": ["administrativa", "protecao", "acesso", "controle", "descricao", "politica", "procedimento"],
        "value_terms": [
            "gestao de acesso",
            "controle de acesso",
            "perfil",
            "restricao",
            "liderancas",
            "controller",
            "financeiro",
            "politica de organizacao documental",
            "procedimento",
            "treinamento de acesso",
            "hierarquia de acesso",
            "permitido apenas",
            "exclusivamente por e-mail",
            "exclusivamente por whatsapp",
            "termo de adesao",
        ],
        "allow_text": True,
        "header_optional": True,
        "min_tokens": 2,
    },
    "4_possui_contrato": {
        # IMPORTANTE:
        #   Ser conservador aqui. Não basta a linha citar "contrato" em um fluxo de processo.
        #   Só aceitar quando houver um cabeçalho/âncora explicitamente contratual do sistema
        #   (ex.: "Possui contrato escrito assinado?: Sim", "Contrato com o fornecedor: Sim").
        "header_terms": [
            "possui contrato escrito assinado",
            "contrato com o fornecedor",
            "contrato do sistema",
            "fornecedor do sistema",
        ],
        "value_terms": [],
        "allow_text": False,
        "header_optional": False,
        "min_tokens": 1,
    },
}


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


def _env_float(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
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
                value_text = _clean_text(value)
                candidate = f"{col_name}: {value_text}"
                if looks_like_prompt(candidate, question_text=str(col_name)):
                    continue
                if is_too_generic(value_text):
                    continue
                pairs.append(candidate)
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

    # pdfminer often inserts blank lines between almost every row.
    # Collapse fake paragraphs to avoid exploding into micro-chunks.
    if base.count("\n\n") >= 20:
        base = re.sub(r"\n\s*\n+", "\n", base).strip()

    by_paragraph = [p.strip() for p in re.split(r"\n\s*\n+", base) if p.strip()]

    # If paragraph split created too many tiny blocks, prefer line grouping.
    if len(by_paragraph) > 12:
        avg_size = sum(len(p) for p in by_paragraph) / float(len(by_paragraph))
        if avg_size < 90:
            by_paragraph = []

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
    question_keywords: Optional[Dict[str, Dict[str, List[str]]]] = None,
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
        "question_keywords": dict(question_keywords or {}),
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


def _normalize_for_lexical(value: str) -> str:
    text = _clean_text(value or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9/\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_for_lexical(value: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]{2,}", _normalize_for_lexical(value))
    return [tok for tok in tokens if tok not in LEXICAL_STOPWORDS]


def _filter_question_keyword_tokens(tokens: Sequence[str]) -> List[str]:
    filtered: List[str] = []
    seen: set[str] = set()
    for raw in tokens or []:
        tok = _normalize_for_lexical(str(raw))
        if not tok or tok in seen:
            continue
        if tok in QUESTION_KEYWORD_STOPWORDS:
            continue
        seen.add(tok)
        filtered.append(tok)
    return filtered


def _prioritize_question_keyword_tokens(tokens: Sequence[str]) -> List[str]:
    weighted: List[Tuple[int, int, str]] = []
    for idx, tok in enumerate(tokens or []):
        score = 0
        if tok in QUESTION_DOMAIN_HINTS:
            score += 3
        if len(tok) >= 6:
            score += 1
        if re.search(r"\d", tok):
            score += 1
        weighted.append((-score, idx, tok))
    weighted.sort()
    return [tok for _, _, tok in weighted]


def _default_question_keywords(question_text: str) -> Dict[str, List[str]]:
    raw_tokens = _tokenize_for_lexical(question_text)
    filtered = _filter_question_keyword_tokens(raw_tokens)
    prioritized = _prioritize_question_keyword_tokens(filtered or raw_tokens)
    required = prioritized[:2]
    optional = prioritized[2:7]
    return {"required": required, "optional": optional, "negative": []}


def _resolve_question_keywords(question_code: str, question_text: str, index: Dict[str, Any]) -> Dict[str, List[str]]:
    configured = (index.get("question_keywords") or {}).get(question_code) if isinstance(index, dict) else None
    base = _default_question_keywords(question_text)
    if not isinstance(configured, dict):
        return base
    merged = {
        "required": list(base["required"]),
        "optional": list(base["optional"]),
        "negative": [],
    }
    for key in ("required", "optional", "negative"):
        values: List[str] = []
        for raw in configured.get(key, []) or []:
            txt = _normalize_for_lexical(str(raw))
            if txt:
                values.append(txt)
        if key != "negative":
            values = _prioritize_question_keyword_tokens(_filter_question_keyword_tokens(values) or values)
        if values:
            merged[key] = values
    return merged


def _keyword_hits(text_norm: str, keywords: Sequence[str]) -> int:
    return sum(1 for kw in keywords if kw and kw in text_norm)


def _proximity_bonus(text_norm: str, keywords: Sequence[str], window: int = 18) -> float:
    tokens = _tokenize_for_lexical(text_norm)
    if not tokens:
        return 0.0
    positions: Dict[str, List[int]] = {}
    for idx, tok in enumerate(tokens):
        positions.setdefault(tok, []).append(idx)
    observed: List[int] = []
    for kw in keywords:
        first_token = _tokenize_for_lexical(kw)
        if not first_token:
            continue
        tok = first_token[0]
        if tok in positions:
            observed.append(positions[tok][0])
    if len(observed) < 2:
        return 0.0
    observed.sort()
    min_gap = min(observed[i + 1] - observed[i] for i in range(len(observed) - 1))
    if min_gap <= 2:
        return 1.0
    if min_gap <= window:
        return max(0.0, 1.0 - (float(min_gap) / float(window)))
    return 0.0


def lexical_score_for_chunk(
    question_code: str,
    question_text: str,
    chunk_text: str,
    index: Dict[str, Any],
) -> float:
    chunk_norm = _normalize_for_lexical(chunk_text or "")
    if not chunk_norm:
        return 0.0

    hints = _resolve_question_keywords(question_code, question_text, index)
    required = [kw for kw in hints.get("required", []) if kw]
    optional = [kw for kw in hints.get("optional", []) if kw]
    negative = [kw for kw in hints.get("negative", []) if kw]

    required_ratio = 0.0
    if required:
        required_hits = _keyword_hits(chunk_norm, required)
        required_ratio = required_hits / float(len(required))
    else:
        required_hits = 0

    optional_ratio = 0.0
    if optional:
        optional_ratio = _keyword_hits(chunk_norm, optional) / float(len(optional))

    explicit_ratio = _keyword_hits(chunk_norm, list(EXPLICIT_ANSWER_TOKENS)) / float(max(1, len(EXPLICIT_ANSWER_TOKENS)))
    proximity = _proximity_bonus(chunk_norm, required + optional)

    score = (
        0.50 * required_ratio
        + 0.25 * optional_ratio
        + 0.15 * explicit_ratio
        + 0.10 * proximity
    )

    if required and required_hits == 0:
        score *= 0.40
    if any(neg in chunk_norm for neg in negative):
        score *= 0.65
    if len(chunk_norm) < 50:
        score *= 0.80
    if chunk_norm.count("?") >= 2 and explicit_ratio == 0.0 and required_hits == 0:
        score *= 0.80

    return max(0.0, min(1.0, float(score)))


def retrieve_context_for_question(
    question_code: str,
    question_text: str,
    index: Dict[str, Any],
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    include_debug: bool = False,
) -> Any:
    """
    Return top-k most similar chunks for a question.

    The section filter is inferred from index["question_sections"] by question_code.
    """
    if not isinstance(index, dict):
        return {"hits": [], "debug": {"before_rerank": [], "after_rerank": []}} if include_debug else []
    if not index.get("has_embeddings"):
        return {"hits": [], "debug": {"before_rerank": [], "after_rerank": []}} if include_debug else []

    chunks: List[RagChunk] = list(index.get("chunks") or [])
    vectors: List[List[float]] = list(index.get("embeddings") or [])
    if not chunks or not vectors:
        return {"hits": [], "debug": {"before_rerank": [], "after_rerank": []}} if include_debug else []

    try:
        final_k = int(top_k)
    except Exception:
        final_k = DEFAULT_RETRIEVAL_TOP_K
    final_k = max(1, min(final_k, 8))
    embed_top_k = _env_int("RAG_RETRIEVAL_EMBED_TOP_K", DEFAULT_RETRIEVAL_EMBED_TOP_K, 4, 24)
    alpha = _env_float("RAG_RETRIEVAL_ALPHA", DEFAULT_RETRIEVAL_ALPHA, 0.0, 1.0)

    question_vec = _question_embedding(question_code, question_text, index)
    if not question_vec:
        return {"hits": [], "debug": {"before_rerank": [], "after_rerank": []}} if include_debug else []

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
        emb_score_raw = _dot(question_vec, vec)
        emb_score = max(0.0, min(1.0, (emb_score_raw + 1.0) / 2.0))
        chunk = chunks[pos]
        lex_score = lexical_score_for_chunk(question_code, question_text, chunk.text, index)
        final_score = float(alpha) * emb_score + (1.0 - float(alpha)) * lex_score
        scored.append(
            {
                "chunk_id": chunk.chunk_id,
                "score": final_score,
                "emb_score": emb_score,
                "lex_score": lex_score,
                "final_score": final_score,
                "section": chunk.section,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "section_filter": section,
                "section_filter_fallback": filter_fallback,
            }
        )

    top_emb = sorted(scored, key=lambda row: float(row.get("emb_score", 0.0)), reverse=True)[:embed_top_k]
    top_lex = sorted(scored, key=lambda row: float(row.get("lex_score", 0.0)), reverse=True)[:embed_top_k]

    pool_map: Dict[str, Dict[str, Any]] = {}
    for row in top_emb + top_lex:
        cid = str(row.get("chunk_id") or "")
        if not cid:
            continue
        pool_map[cid] = row

    before_rerank = list(pool_map.values())
    for rank, row in enumerate(top_emb, start=1):
        row["rank_embed"] = rank
    for rank, row in enumerate(top_lex, start=1):
        row["rank_lex"] = rank

    reranked = sorted(before_rerank, key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
    final_hits = reranked[:final_k]
    for rank, row in enumerate(final_hits, start=1):
        row["rank_final"] = rank

    if include_debug:
        before_debug = [
            {
                "chunk_id": row.get("chunk_id"),
                "emb_score": round(float(row.get("emb_score", 0.0)), 6),
                "lex_score": round(float(row.get("lex_score", 0.0)), 6),
                "final_score": round(float(row.get("final_score", 0.0)), 6),
            }
            for row in before_rerank
        ]
        after_debug = [
            {
                "chunk_id": row.get("chunk_id"),
                "emb_score": round(float(row.get("emb_score", 0.0)), 6),
                "lex_score": round(float(row.get("lex_score", 0.0)), 6),
                "final_score": round(float(row.get("final_score", 0.0)), 6),
            }
            for row in final_hits
        ]
        return {"hits": final_hits, "debug": {"before_rerank": before_debug, "after_rerank": after_debug}}

    return final_hits


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


def _short_evidence_excerpt(text: str, max_len: int = 240) -> str:
    content = _clean_text(text)
    if not content:
        return ""
    if len(content) <= max_len:
        return content
    excerpt = content[:max_len].strip()
    split_at = max(excerpt.rfind(". "), excerpt.rfind("; "), excerpt.rfind(", "), excerpt.rfind(" "))
    if split_at >= int(max_len * 0.6):
        excerpt = excerpt[:split_at].strip()
    return excerpt


def _iter_line_value_candidates(line: str) -> List[str]:
    raw = _clean_text(line)
    if not raw:
        return []

    candidates: List[str] = []
    if "|" in raw:
        parts = [_clean_text(part) for part in raw.split("|")]
        candidates.extend(part for part in parts[1:] if part)

    for sep in (":", " - ", " => ", " = ", ";"):
        if sep not in raw:
            continue
        tail = _clean_text(raw.split(sep, 1)[1])
        if tail:
            candidates.append(tail)

    candidates.append(raw)

    deduped: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _normalize_for_schema(value: str) -> str:
    text = _normalize_for_lexical(value or "")
    text = text.replace(".", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _compact_for_schema(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", _normalize_for_schema(value)))


def _contains_schema_term(haystack_norm: str, haystack_compact: str, term: str) -> bool:
    term_norm = _normalize_for_schema(term or "")
    if not term_norm:
        return False
    if term_norm in haystack_norm:
        return True
    term_compact = _compact_for_schema(term_norm)
    return bool(term_compact and term_compact in haystack_compact)


def _contains_any_schema_terms(value: str, terms: Sequence[str]) -> bool:
    if not terms:
        return False
    haystack_norm = _normalize_for_schema(value)
    haystack_compact = _compact_for_schema(value)
    return any(_contains_schema_term(haystack_norm, haystack_compact, term) for term in (terms or []))


def _schema_signal(value: str) -> str:
    norm = _normalize_for_schema(value)
    if not norm:
        return ""
    if norm in SCHEMA_SIGNAL_NA:
        return "na"
    if "nao se aplica" in norm or "nao aplicavel" in norm or "nao informado" in norm:
        return "na"
    if re.search(r"(^|[^a-z0-9])n/a([^a-z0-9]|$)", norm):
        return "na"
    if re.search(r"(^|[^a-z0-9])n d([^a-z0-9]|$)", norm):
        return "na"
    if any(re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", norm) for tok in SCHEMA_SIGNAL_YES):
        return "yes"
    if any(re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", norm) for tok in SCHEMA_SIGNAL_NO):
        return "no"
    return ""


def _iter_schema_fragments(chunk_text: str) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in _clean_text(chunk_text).splitlines() if ln.strip()]
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for line in lines:
        fragments = [frag.strip() for frag in line.split(";") if frag.strip()]
        if not fragments:
            fragments = [line]

        for fragment in fragments:
            header = ""
            value = fragment
            if ":" in fragment:
                head, tail = fragment.split(":", 1)
                if _clean_text(tail):
                    header = _clean_text(head)
                    value = _clean_text(tail)
            elif "?" in fragment:
                head, tail = fragment.split("?", 1)
                if _clean_text(tail):
                    header = f"{_clean_text(head)}?"
                    value = _clean_text(tail)
            key = (_normalize_for_schema(header), _normalize_for_schema(value), _normalize_for_schema(fragment))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "line": line,
                    "fragment": fragment,
                    "header": header,
                    "value": value,
                }
            )

        whole_key = ("", _normalize_for_schema(line), _normalize_for_schema(line))
        if whole_key in seen:
            continue
        seen.add(whole_key)
        out.append({"line": line, "fragment": line, "header": "", "value": line})
    return out


def _schema_table_answer_from_hits(
    question_code: str,
    question_text: str,
    hits: Sequence[Dict[str, Any]],
    index: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    rule = SCHEMA_TABLE_RULES.get(str(question_code))
    if not rule:
        return None

    header_terms = list(rule.get("header_terms") or [])
    value_terms = list(rule.get("value_terms") or [])
    allow_text = bool(rule.get("allow_text", True))
    header_optional = bool(rule.get("header_optional", False))
    min_tokens = max(1, int(rule.get("min_tokens", 1) or 1))

    hints = _resolve_question_keywords(question_code, question_text, index)
    required = [kw for kw in hints.get("required", []) if kw]
    optional = [kw for kw in hints.get("optional", []) if kw]

    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for hit in hits or []:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        chunk_text = _clean_text(hit.get("text", ""))
        if not chunk_id or not chunk_text:
            continue

        for fragment in _iter_schema_fragments(chunk_text):
            header = fragment.get("header", "")
            value = fragment.get("value", "")
            literal = fragment.get("fragment", "") or value
            if not value:
                continue

            header_match = _contains_any_schema_terms(header or fragment.get("line", ""), header_terms)
            value_match = _contains_any_schema_terms(value or fragment.get("line", ""), value_terms)
            line_norm = _normalize_for_lexical(fragment.get("line", ""))
            keyword_hits = _keyword_hits(line_norm, required + optional)

            signal = _schema_signal(value)
            if not signal:
                signal = _schema_signal(literal)

            value_tokens = _tokenize_for_lexical(value)
            descriptive = len(value_tokens) >= min_tokens and not looks_like_prompt(literal, question_text=question_text)

            accepted = False
            reason = "schema_table_value_present"
            score = 0.0

            if signal == "na":
                if header_match or value_match or keyword_hits > 0 or header_optional:
                    accepted = True
                    reason = "schema_value_present_na"
                    score += 9.0
            elif signal in {"yes", "no"}:
                if header_match or value_match or keyword_hits > 0:
                    accepted = True
                    score += 7.0
            elif value_match:
                accepted = True
                score += 6.0
            elif allow_text and descriptive and (header_match or keyword_hits > 0 or header_optional):
                accepted = True
                score += 5.0

            if not accepted:
                continue
            if not header_optional and not (header_match or value_match):
                continue
            if is_too_generic(literal) and signal not in {"yes", "no", "na"} and not value_match:
                continue

            if signal in {"yes", "no"}:
                score += 1.0
            if header_match:
                score += 1.0
            if value_match:
                score += 1.0
            score += min(2.0, float(len(value_tokens)) / 2.0)
            score += min(2.0, float(keyword_hits))

            evidence = literal
            if _schema_signal(evidence) in {"yes", "no", "na"} and fragment.get("line"):
                evidence = fragment["line"]
            evidence = _clean_text(evidence)[:240]
            if not evidence:
                continue

            candidate = {
                "answerable": 1,
                "evidence": [evidence],
                "used_chunk_ids": [chunk_id],
                "reason": reason,
            }
            if best is None or score > best[0]:
                best = (score, candidate)

    if best is None:
        return None
    return best[1]


def _select_value_like_chunk_evidence(text: str, question_text: str = "", max_len: int = 240) -> str:
    content = _clean_text(text)
    if not content:
        return ""

    def _line_has_yesno_payload(line: str) -> bool:
        raw = _clean_text(line)
        if not raw:
            return False

        # Caso 1: "Pergunta? sim"
        if "?" in raw:
            _, tail = raw.split("?", 1)
            if _schema_signal(tail) in {"yes", "no", "na"}:
                return True

        # Caso 2: "Header: sim"
        if ":" in raw:
            _, tail = raw.split(":", 1)
            if _schema_signal(tail) in {"yes", "no", "na"}:
                return True

        return False

    for raw_line in content.splitlines():
        if _line_has_yesno_payload(raw_line):
            return _short_evidence_excerpt(raw_line, max_len=max_len)

        for candidate in _iter_line_value_candidates(raw_line):
            if candidate.endswith("?"):
                continue

            if looks_like_prompt(candidate, question_text=question_text) and not _line_has_yesno_payload(candidate):
                continue

            if is_too_generic(candidate):
                continue

            if is_value_like(candidate, question_text=question_text):
                return _short_evidence_excerpt(candidate, max_len=max_len)

    return ""

def _heuristic_system_name_from_chunks(chunks: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of at least one system name mentioned in chunks.
    Returns dict {answerable,evidence,used_chunk_ids,reason} or None.
    """
    if not chunks:
        return None

    known_names = {
        "datasul",
        "sap",
        "totvs",
        "oracle",
        "salesforce",
        "microsoft dynamics",
        "protheus",
        "rm",
    }

    name_patterns = [
        re.compile(r"\b(?:sistema|software)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9_\-\+/\.]{2,})", re.IGNORECASE),
        re.compile(r"\b([A-Za-z][A-Za-z0-9_\-\+/\.]{2,})\s*\(sistema\)", re.IGNORECASE),
    ]

    best: Optional[Tuple[str, str, str]] = None  # (score, evidence, chunk_id)
    for row in chunks:
        cid = str((row or {}).get("chunk_id") or "").strip()
        text = _clean_text((row or {}).get("text", ""))
        if not cid or not text:
            continue

        low = _normalize_for_lexical(text)

        for nm in known_names:
            if nm in low:
                m = re.search(re.escape(nm), text, flags=re.IGNORECASE)
                if m:
                    start = max(0, m.start() - 40)
                    end = min(len(text), m.end() + 60)
                    ev = text[start:end].strip()
                    best = ("2", ev, cid)
                    break
        if best and best[0] == "2":
            break

        for pat in name_patterns:
            m = pat.search(text)
            if not m:
                continue
            name = (m.group(1) or "").strip()
            if not name:
                continue
            if _normalize_for_lexical(name) in {"dados", "processo", "empresa", "area", "rh"}:
                continue
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 60)
            ev = text[start:end].strip()
            best = ("1", ev, cid)
            break
        if best and best[0] == "1":
            break

    if not best:
        return None

    _, ev, cid = best
    return {
        "answerable": 1,
        "evidence": [ev[:240]],
        "used_chunk_ids": [cid],
        "reason": "heuristic_system_name_found_partial",
    }


def _should_accept_system_vendor_evidence(evidence: Sequence[str]) -> bool:
    """Return True only if evidence indicates a *system vendor*, not just any third party list."""
    if not evidence:
        return False

    joined = " ".join(evidence)
    low = _normalize_for_lexical(joined)

    if not any(tok in low for tok in ("sistema", "software", "erp", "aplicacao", "aplicativo", "plataforma")):
        return False

    if any(tok in low for tok in ("fornecedor", "vendor", "fabricante", "totvs", "sap", "oracle", "microsoft", "salesforce")):
        return True

    if re.search(r"\bfornecedor\b", joined, flags=re.IGNORECASE):
        return True

    return False

def _sanitize_rag_json_response(
    data: Any,
    allowed_chunk_ids: set[str],
    fallback_chunks: Optional[Sequence[Dict[str, Any]]] = None,
    question_code: str = "",
    question_text: str = "",
    question_keywords: Optional[Dict[str, List[str]]] = None,
    debug_out: Optional[Dict[str, Any]] = None,
    strict_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"answerable": 0, "evidence": [], "used_chunk_ids": [], "reason": "invalid_json"}

    # 1) answerable + strict primeiro
    answer_raw = data.get("answerable", 0)
    answerable = 1 if answer_raw in (1, True, "1", "true", "sim", "yes") else 0
    strict = strict_mode_enabled() if strict_mode is None else bool(strict_mode)

    # 2) montar fallback chunks
    fallback_text_by_chunk_id: Dict[str, str] = {}
    fallback_chunk_ids: List[str] = []
    for row in fallback_chunks or []:
        cid = str((row or {}).get("chunk_id") or "").strip()
        if not cid or cid not in allowed_chunk_ids:
            continue
        if cid not in fallback_text_by_chunk_id:
            fallback_chunk_ids.append(cid)
        fallback_text_by_chunk_id[cid] = _clean_text((row or {}).get("text", ""))

    # 3) evidence + used_chunk_ids + reason
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

    # ------------------------------------------------------------------
    # Heuristic fallback for 4.1 (forma de compartilhamento)
    # Só tenta quando:
    #   - answerable ainda é 0
    #   - strict está ligado
    #   - question_code == 4_1_forma_compart
    # ------------------------------------------------------------------
    if answerable != 1 and strict and str(question_code or "").strip() == "4_1_forma_compart":
        method_re = re.compile(
            r"\b(e-?mail|email|api|arquivo|planilha|drive|formulario|software|sistema|portal)\b",
            re.IGNORECASE,
        )

        for cid in fallback_chunk_ids:
            chunk_text = fallback_text_by_chunk_id.get(cid, "") or ""
            if not chunk_text:
                continue
            if not method_re.search(chunk_text):
                continue

            # Preferir linha que contenha o termo
            chosen = ""
            for line in (chunk_text.splitlines() or []):
                if method_re.search(line):
                    chosen = line.strip()
                    break

            # Fallback: pega recorte próximo do match
            if not chosen:
                m = method_re.search(chunk_text)
                if m:
                    start = max(0, m.start() - 80)
                    end = min(len(chunk_text), m.end() + 120)
                    chosen = chunk_text[start:end].strip()

            chosen = _clean_text(chosen)[:240]
            if not chosen:
                continue
            if looks_like_prompt(chosen):
                continue

            proposed_evidence = [chosen]
            proposed_used_ids = [cid]
            used_chunk_texts = {cid: chunk_text}

            validated_answerable, guard_reason, validated_evidence = validate_answerability(
                question_code=question_code,
                answerable=1,
                evidence=proposed_evidence,
                used_ids=proposed_used_ids,
                chunk_texts=used_chunk_texts,
                question_text=question_text,
                question_keywords=question_keywords,
                debug_out=debug_out,
                strict_mode=strict,
            )

            if int(validated_answerable) == 1:
                answerable = 1
                evidence = list(validated_evidence)
                used_chunk_ids = proposed_used_ids
                reason = guard_reason or "heuristic_share_method_detected"
                break

    # ------------------------------------------------------------------
    # Heurística de resgate para 7 (finalidade legítima)
    # Quando o modelo estiver conservador demais, aceitamos como respondível
    # se o chunk trouxer uma finalidade/objetivo explícito junto do contexto
    # de tratamento/uso de dados ou do próprio processo.
    # ------------------------------------------------------------------
    if answerable != 1 and strict and str(question_code or "").strip() == "7_finalidade_legitima":
        purpose_terms = (
            "finalidade",
            "objetivo",
            "objetiva",
            "objetivar",
            "proposito",
            "propósito",
            "justific",
            "motivo",
            "interesse legitimo",
            "interesse legítimo",
        )
        context_terms = (
            "dados",
            "dado",
            "trat",
            "process",
            "uso",
            "coleta",
            "fluxo",
        )

        def _purpose_excerpt(chunk_text: str) -> str:
            parts = [seg.strip() for seg in re.split(r"(?<=[\.!?])\s+|\n+", chunk_text or "") if str(seg).strip()]
            best = ""
            best_score = -1

            for seg in parts:
                low = _normalize_for_lexical(seg)
                purpose_hits = sum(1 for term in purpose_terms if term in low)
                context_hits = sum(1 for term in context_terms if term in low)
                score = purpose_hits * 3 + context_hits

                if score > best_score and purpose_hits > 0:
                    best_score = score
                    best = seg

            if best_score >= 3:
                return _clean_text(best)[:240]

            return ""

        candidate_ids = list(used_chunk_ids) + [cid for cid in fallback_chunk_ids if cid not in used_chunk_ids]

        for cid in candidate_ids:
            chunk_text = fallback_text_by_chunk_id.get(cid, "") or ""
            if not chunk_text:
                continue

            low_chunk = _normalize_for_lexical(chunk_text)

            if not any(term in low_chunk for term in purpose_terms):
                continue

            if not any(term in low_chunk for term in context_terms):
                continue

            chosen = _purpose_excerpt(chunk_text)
            if not chosen:
                continue

            if looks_like_prompt(chosen, question_text=question_text):
                continue

            proposed_evidence = [chosen]
            proposed_used_ids = [cid]
            used_chunk_texts = {cid: chunk_text}

            validated_answerable, guard_reason, validated_evidence = validate_answerability(
                question_code=question_code,
                answerable=1,
                evidence=proposed_evidence,
                used_ids=proposed_used_ids,
                chunk_texts=used_chunk_texts,
                question_text=question_text,
                question_keywords=question_keywords,
                debug_out=debug_out,
                strict_mode=strict,
            )

            if int(validated_answerable) == 1:
                answerable = 1
                evidence = list(validated_evidence)
                used_chunk_ids = proposed_used_ids
                reason = guard_reason or "heuristic_legitimate_purpose_detected"
                break

    # 4) auto-assign chunk (apenas quando NÃO strict)
    _ALLOW_AUTO_ASSIGN_IN_STRICT = {
        "1_nome_identificado",        # sistemas - nome do sistema
        "2_fornecedor_identificado",  # sistemas - fornecedor (quando houver)
        "3_eh_web",                   # sistemas - sistema web (quando houver)
    }

    if answerable == 1 and not used_chunk_ids and fallback_chunk_ids and (not strict or question_code in _ALLOW_AUTO_ASSIGN_IN_STRICT):
        used_chunk_ids = [fallback_chunk_ids[0]]
        reason = "auto_assigned_chunk" if not strict else "auto_assigned_chunk_strict"

    # 5) se answerable=1 mas veio sem evidence, tenta gerar evidência do chunk
    if answerable == 1 and not evidence:
        candidate_ids = list(used_chunk_ids) + list(fallback_chunk_ids)
        for cid in candidate_ids:
            chunk_text = fallback_text_by_chunk_id.get(cid, "")
            if not chunk_text:
                continue
            snippet = _select_value_like_chunk_evidence(chunk_text, question_text=question_text)
            if not snippet:
                snippet = _short_evidence_excerpt(chunk_text)
            if not snippet:
                continue
            if looks_like_prompt(snippet):
                continue
            if snippet not in evidence:
                evidence.append(_clean_text(snippet)[:240])
            if len(evidence) >= 2:
                break

    # Heurística de "resgate" para 1_nome_identificado:
    # Se o LLM marcar como não respondível, mas houver chunk com um nome de sistema explícito
    # (ex.: "Sistema Datasul"), promovemos para answerable=1 com evidência rastreável.
    if answerable != 1 and question_code == "1_nome_identificado":
        candidate_ids = list(used_chunk_ids) + list(fallback_chunk_ids)
        rx = re.compile(r"\bsistema\s+([A-Z][A-Za-z0-9_.-]{1,})")
        for cid in candidate_ids:
            chunk_text = fallback_text_by_chunk_id.get(cid, "")
            m = rx.search(chunk_text or "")
            if not m:
                continue
            sys_name = m.group(1).strip()
            if not sys_name:
                continue
            used_chunk_ids = [cid]
            evidence = [f"Sistema {sys_name}"]
            answerable = 1
            reason = "heuristic_system_name_detected"
            break        

    # 6) Validação final (guarda strict / chunk trace)
    if answerable == 1:
        used_chunk_texts = {cid: fallback_text_by_chunk_id.get(cid, "") for cid in used_chunk_ids}
        validated_answerable, guard_reason, validated_evidence = validate_answerability(
            question_code=question_code,
            answerable=answerable,
            evidence=evidence,
            used_ids=used_chunk_ids,
            chunk_texts=used_chunk_texts,
            question_text=question_text,
            question_keywords=question_keywords,
            debug_out=debug_out,
            strict_mode=strict,
        )
        answerable = int(validated_answerable)
        evidence = list(validated_evidence)

        if answerable != 1:
            reason = guard_reason or reason or ("strict_missing_chunk_trace" if strict else "missing_chunk_trace")
            metric = reason_to_metric(reason)
            if metric:
                logging.info("%s=1 source=rag question_code=%s reason=%s", metric, question_code, reason)
        elif guard_reason:
            reason = guard_reason

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

    # ---------------------------------------------------------------------
    # Optional policy: allow answering "Não" by absence of evidence.
    #
    # OFF by default because it is an inference (open-world vs closed-world).
    # When enabled, we only flip an explicit allow-list of *existence* questions
    # to answerable=1 with a standardized evidence marker.
    #
    # Enable with: DM_ALLOW_NO_BY_ABSENCE=1
    # ---------------------------------------------------------------------
    allow_no_by_absence = str(os.getenv("DM_ALLOW_NO_BY_ABSENCE", "0")).strip() in (
        "1",
        "true",
        "True",
        "yes",
        "sim",
    )
    absence_no_codes: set[str] = {
        # AREAS
        "1_HD_externo_pendrive",
        "1_1_hd_tem_senha",
        "1_2_hd_armazenado_onde",
        "2_1_rede_protecao_senha",
        "3_compart_nuvem",
        "3_1_nuvem_criptografado",
        "4_1_forma_compart",
        "4_2_termo_sigilo_terceiros",
        "5_transfer_internacional",
        "5_1_paises",
        "5_2_termo_sigilo_transfer",
        "6_classificacao_documentos",
        # PROCESSOS
        "1_decisoes_automatizadas",
        "4_dados_menores",
        "6_2_armazenado_nuvem",
        "7_finalidade_legitima",
        "8_finalidade_explicita_ao_titular",
        "9_somente_quando_necessario",
        "10_quem_usa_precisa",
        "11_tempo_necessario_apenas",
        "13_forma_transferencia_dados",
        "15_dados_precisos_claros_atualizados",
        "16_titular_pode_corrigir",
        "17_1_medidas_seg_comprovadas",
        "18_1_admin_comprovadas",
        # SISTEMAS
        "1_nome_identificado",
        "3_eh_web",
    }

    system_prompt = (
        "Voce e um avaliador de respondibilidade LGPD. "
        "Responda APENAS JSON estrito no formato "
        "{\"answers\":{\"<question_code>\":{\"answerable\":0|1,\"evidence\":[...],\"used_chunk_ids\":[...],\"reason\":\"...\"}}}. "
        "Definicao: answerable=1 somente quando for possivel responder com base explicita no documento. "
        "N/A, nao se aplica e nao informado contam como respondiveis apenas quando estiverem explicitos. "
        "Nunca deduza resposta negativa por ausencia de evidencia. "
        "EVIDENCE deve ser copia literal do chunk, nunca parafraseie. "
        "EVIDENCE nao pode repetir apenas o texto da pergunta ou cabecalho. "
        "Se o chunk tiver resposta binaria literal, prefira o trecho completo no formato em que aparece no chunk, como 'Campo: Sim'. "
        "Quando houver mais de um trecho literal util, prefira retornar 2 evidencias curtas."
    )

    # Retrieval stage (question -> top-k chunks)
    retrieval_started = time.time()
    question_hits: Dict[str, List[Dict[str, Any]]] = {}
    retrieval_debug: Dict[str, Dict[str, Any]] = {}
    retrieval_times_ms: List[int] = []
    retrieval_embed_top_k = _env_int("RAG_RETRIEVAL_EMBED_TOP_K", DEFAULT_RETRIEVAL_EMBED_TOP_K, 4, 24)
    retrieval_alpha = _env_float("RAG_RETRIEVAL_ALPHA", DEFAULT_RETRIEVAL_ALPHA, 0.0, 1.0)
    for question_code, question_text in question_list:
        t0 = time.time()

        # FIX: algumas perguntas precisam de mais contexto porque o rerank pode
        # deixar de fora chunks que contêm a evidência (ex.: "E-mail", "Software", "API").
        per_question_top_k = top_k
        if question_code in {"13_forma_transferencia_dados", "6_dados_tratados_quais"}:
            per_question_top_k = max(int(top_k or 0), 8)

        retrieval_out = retrieve_context_for_question(
            question_code,
            question_text,
            index,
            top_k=per_question_top_k,
            include_debug=True,
        )
        if isinstance(retrieval_out, dict):
            hits = list(retrieval_out.get("hits") or [])
            retrieval_debug[question_code] = dict(retrieval_out.get("debug") or {})
        else:
            hits = list(retrieval_out or [])
            retrieval_debug[question_code] = {"before_rerank": [], "after_rerank": []}
        retrieval_times_ms.append(int((time.time() - t0) * 1000))
        question_hits[question_code] = hits
        out[question_code] = {
            "answerable": 0,
            "evidence": [],
            "used_chunk_ids": [],
            "reason": "not_evaluated",
            "retrieval_debug": retrieval_debug.get(question_code, {"before_rerank": [], "after_rerank": []}),
            "rag_chunks": [
                {
                    "chunk_id": hit.get("chunk_id"),
                    "score": round(float(hit.get("score", 0.0)), 6),
                    "emb_score": round(float(hit.get("emb_score", 0.0)), 6),
                    "lex_score": round(float(hit.get("lex_score", 0.0)), 6),
                    "final_score": round(float(hit.get("final_score", hit.get("score", 0.0))), 6),
                    "section": hit.get("section"),
                    "source": hit.get("source"),
                    "trecho": _clean_text(hit.get("text", ""))[:260],
                }
                for hit in hits
            ],
        }
    retrieval_elapsed_ms = int((time.time() - retrieval_started) * 1000)

    schema_table_candidate_total = 0
    schema_table_applied_total = 0
    for question_code, question_text in question_list:
        hits = list(question_hits.get(question_code) or [])
        if not hits:
            continue
        schema_row = _schema_table_answer_from_hits(
            question_code=question_code,
            question_text=question_text,
            hits=hits,
            index=index,
        )
        if not schema_row:
            continue
        schema_table_candidate_total += 1
        allowed_for_question = {
            str(hit.get("chunk_id") or "").strip()
            for hit in hits
            if str(hit.get("chunk_id") or "").strip()
        }
        if not allowed_for_question:
            continue
        fallback_chunks = [
            {"chunk_id": str(hit.get("chunk_id") or "").strip(), "text": _clean_text(hit.get("text", ""))}
            for hit in hits
            if str(hit.get("chunk_id") or "").strip() in allowed_for_question
        ]
        normalized = _sanitize_rag_json_response(
            schema_row,
            allowed_for_question,
            fallback_chunks=fallback_chunks,
            question_code=question_code,
            question_text=question_text,
            question_keywords=_resolve_question_keywords(question_code, question_text, index),
        )
        if int(normalized.get("answerable", 0)) != 1:
            continue
        normalized["rag_chunks"] = out[question_code].get("rag_chunks", [])
        normalized["retrieval_debug"] = out[question_code].get("retrieval_debug", {"before_rerank": [], "after_rerank": []})
        out[question_code] = normalized
        schema_table_applied_total += 1

    if client is None or not resolved_model:
        for question_code, _ in question_list:
            if int((out.get(question_code) or {}).get("answerable", 0)) == 1:
                continue
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
                "retrieval_embed_top_k": retrieval_embed_top_k,
                "retrieval_alpha": retrieval_alpha,
                "llm_calls": 0,
                "llm_call_ms": [],
                "llm_ms_total": 0,
                "guard_drop_stats": {},
                "guard_drop_reason_counts": {},
                "raw_llm_answerable_total": 0,
                "guard_dropped_answerable_total": 0,
                "guard_auto_repair_total": 0,
                "guard_auto_expand_total": 0,
                "schema_table_candidate_total": schema_table_candidate_total,
                "schema_table_applied_total": schema_table_applied_total,
                "guard_anchoring_debug": {},
                "elapsed_ms": elapsed_ms,
            },
        }

    llm_call_ms: List[int] = []
    truncated_batches = 0
    evaluated_questions = 0
    guard_drop_stats: Dict[str, int] = {}
    raw_llm_answerable_total = 0
    guard_dropped_answerable_total = 0
    guard_auto_repair_total = 0
    guard_auto_expand_total = 0
    guard_drop_reason_counts: Dict[str, int] = {}
    guard_anchoring_debug: Dict[str, Dict[str, Any]] = {}

    for batch in _iter_batches(question_list, batch_size_resolved):
        nonempty_batch = [
            (code, text)
            for code, text in batch
            if question_hits.get(code) and int((out.get(code) or {}).get("answerable", 0)) != 1
        ]
        for code, _ in batch:
            if int((out.get(code) or {}).get("answerable", 0)) == 1:
                continue
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
            + "- answerable=1 somente se o contexto permitir responder a pergunta sem inferencia.\n"
            + "- N/A, nao se aplica e nao informado contam como respondiveis quando explicitos.\n"
            + "- used_chunk_ids deve conter somente chunk_ids do contexto.\n"
            + "- Se nao houver evidencia explicita, answerable=0.\n"
            + "- Nao deduza NAO por ausencia de evidencia.\n"
            + "- Para answerable=1, inclua 1-2 evidence e ao menos 1 used_chunk_ids.\n"
            + "- evidence deve copiar e colar trecho literal do chunk usado.\n"
            + "- evidence nao pode ser so a pergunta, cabecalho ou label do campo.\n"
            + "- Se houver resposta literal em formato 'Campo: valor' ou 'Pergunta? valor', copie o trecho completo.\n"
            + "- Quando existirem 2 trechos literais curtos e relevantes, retorne ambos."
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
            # --------------------------------------------------------------
            # Heuristics for "SISTEMAS" questions to match the intent
            # "é possível responder?" (partial evidence counts as answerable).
            # --------------------------------------------------------------
            if str(code) == "1_nome_identificado":
                # If the model said not answerable, try extracting at least one system name (e.g., "Sistema Datasul").
                if not (isinstance(raw, dict) and raw.get("answerable", 0) in (1, True, "1", "true", "sim", "yes")):
                    heuristic = _heuristic_system_name_from_chunks(fallback_chunks)
                    if heuristic:
                        raw = dict(heuristic)

            if str(code) == "2_fornecedor_identificado":
                # Only accept as answerable if the evidence really refers to the SYSTEM vendor,
                # not just a generic list of third parties/benefit providers.
                if isinstance(raw, dict) and raw.get("answerable", 0) in (1, True, "1", "true", "sim", "yes"):
                    ev_list = list(raw.get("evidence", []) or [])
                    if not _should_accept_system_vendor_evidence(ev_list):
                        raw = {"answerable": 0, "evidence": [], "used_chunk_ids": [], "reason": "vendor_not_tied_to_system"}
            raw_answerable = 1 if isinstance(raw, dict) and raw.get("answerable", 0) in (1, True, "1", "true", "sim", "yes") else 0
            if raw_answerable == 1:
                raw_llm_answerable_total += 1
            fallback_chunks: List[Dict[str, Any]] = []
            seen_chunk_ids: set[str] = set()
            for hit in question_hits.get(code, []) or []:
                cid = str(hit.get("chunk_id") or "").strip()
                if not cid or cid in seen_chunk_ids or cid not in allowed_for_question:
                    continue
                fallback_chunks.append({"chunk_id": cid, "text": _clean_text(hit.get("text", ""))})
                seen_chunk_ids.add(cid)

            guard_debug: Dict[str, Any] = {}
            normalized = _sanitize_rag_json_response(
                raw,
                allowed_for_question,
                fallback_chunks=fallback_chunks,
                question_code=code,
                question_text=questions_payload.get(code, ""),
                question_keywords=_resolve_question_keywords(code, questions_payload.get(code, ""), index),
                debug_out=guard_debug,
            )
            normalized["rag_chunks"] = out[code].get("rag_chunks", [])
            normalized["retrieval_debug"] = out[code].get("retrieval_debug", {"before_rerank": [], "after_rerank": []})
            out[code] = normalized
            if guard_debug:
                guard_anchoring_debug[code] = dict(guard_debug)
            normalized_reason = str(normalized.get("reason") or "")
            if raw_answerable == 1 and int(normalized.get("answerable", 0)) != 1:
                guard_dropped_answerable_total += 1
                if normalized_reason:
                    guard_drop_reason_counts[normalized_reason] = int(guard_drop_reason_counts.get(normalized_reason, 0)) + 1
                bump_metric(guard_drop_stats, normalized_reason)
            elif normalized_reason == "auto_repaired_evidence_from_chunk":
                guard_auto_repair_total += 1
            elif normalized_reason == "auto_expanded_short_evidence":
                guard_auto_expand_total += 1
            evaluated_questions += 1

            # ------------------------------------------------------------------
            # Post-processing: optional "Não por ausência" policy.
            #
            # We only apply it when:
            # - DM_ALLOW_NO_BY_ABSENCE=1
            # - the question code is explicitly allow-listed
            # - the current result is answerable=0
            # - the reason indicates "not found / not explicit"
            #   (we do NOT flip guard failures like missing chunk trace).
            # ------------------------------------------------------------------
            if allow_no_by_absence:
                ok_no_reasons = {
                    "no_retrieved_context",
                    "no_context_after_limit",
                    "no_context_for_question_after_limit",
                    "no_context_after_guard",
                    "no_evidence_explicit",
                    "schema_free_text_not_specific",
                    "no_anchor_mapping",
                    "no_filled_anchor_answer",
                    "no_schema_evidence",
                    "no_output",
                    # Retrieval picked up only the questionnaire/prompt text.
                    "evidence_is_prompt",
                    "prompt_only_evidence",
                }
                blocked_reasons = {
                    "missing_chunk_trace",
                    "strict_missing_chunk_trace",
                    "evidence_not_in_any_chunk",
                    "used_ids_not_allowed",
                }

                for code, _ in question_list:
                    if code not in absence_no_codes:
                        continue
                    row = out.get(code)
                    if not row or int(row.get("answerable", 0)) == 1:
                        continue

                    reason = str(row.get("reason") or "").strip()
                    if reason in blocked_reasons:
                        continue

                    if (reason in ok_no_reasons) or reason.lower().startswith("não há") or reason.lower().startswith("nao ha"):
                        row["answerable"] = 1
                        row["used_chunk_ids"] = []
                        row["evidence"] = ["(ausência de menção explícita no documento)"]
                        row["reason"] = "absence_assumed_no"
                        out[code] = row

    elapsed_ms = int((time.time() - started_at) * 1000)
    if guard_drop_stats or guard_auto_repair_total or guard_auto_expand_total:
        logging.info(
            "RAG_GUARD_STATS=%s",
            json.dumps(
                {
                    "drop_metrics": guard_drop_stats,
                    "drop_reasons": guard_drop_reason_counts,
                    "guard_dropped_answerable_total": guard_dropped_answerable_total,
                    "guard_auto_repair_total": guard_auto_repair_total,
                    "guard_auto_expand_total": guard_auto_expand_total,
                },
                ensure_ascii=False,
            ),
        )
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
            "retrieval_embed_top_k": retrieval_embed_top_k,
            "retrieval_alpha": retrieval_alpha,
            "llm_calls": len(llm_call_ms),
            "llm_call_ms": llm_call_ms,
            "llm_ms_total": int(sum(llm_call_ms)),
            "guard_drop_stats": guard_drop_stats,
            "guard_drop_reason_counts": guard_drop_reason_counts,
            "raw_llm_answerable_total": raw_llm_answerable_total,
            "guard_dropped_answerable_total": guard_dropped_answerable_total,
            "guard_auto_repair_total": guard_auto_repair_total,
            "guard_auto_expand_total": guard_auto_expand_total,
            "schema_table_candidate_total": schema_table_candidate_total,
            "schema_table_applied_total": schema_table_applied_total,
            "guard_anchoring_debug": guard_anchoring_debug,
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
