from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .question_schema import DEFAULT_QUESTION_SCHEMA, QUESTION_SCHEMA


PROMPT_PATTERNS = [
    r"\?$",
    r"^\s*(possui|existe|e possivel|como e|para quais|onde|qual|quais|quem|por que|porque|quanto tempo)\b",
    r"\b(local de armazenamento dos dados|dados tratados quais|forma de transferencia de dados)\b",
    r"\bcategoria dos?\s+dados(?:\s*\(.*\))?\b",
    r"\bforma de coleta dos?\s+dados\b",
    r"\bquem pode acessar os?\s+dados\b",
    r"\bpor que estes?\s+dados sao processados\b",
    r"\bporque estes?\s+dados sao processados\b",
    r"\bquanto tempo o?s?\s*dados? e?s?\s+mantid[oa]s?\b",
]

GENERIC_PATTERNS = [
    r"^local de armazenamento dos dados$",
    r"^informacoes do processo$",
    r"^dados do processo$",
    r"^dados tratados quais$",
    r"^forma de transferencia de dados$",
    r"^forma de coleta dos dados$",
    r"^categoria dos dados(?: pessoal ou sensivel)?$",
    r"^quem pode acessar os dados$",
    r"^por que estes dados sao processados$",
    r"^quanto tempo o dado e mantido$",
    r"^nao informado$",
    r"^n/a$",
]

GENERIC_TOKENS = {
    "dados",
    "dado",
    "informacao",
    "informacoes",
    "processo",
    "local",
    "armazenamento",
    "campo",
    "resposta",
    "descricao",
    "geral",
    "coleta",
    "transferencia",
    "categoria",
    "tempo",
    "mantido",
}

GENERIC_HEADER_TOKENS = {
    "categoria",
    "forma",
    "coleta",
    "transferencia",
    "local",
    "armazenamento",
    "dados",
    "dado",
    "processo",
    "quem",
    "pode",
    "acessar",
    "quanto",
    "tempo",
    "porque",
    "por",
    "que",
    "quais",
}

YES_TOKENS = {"sim", "yes", "true"}
NO_TOKENS = {"nao", "no", "false"}
NA_VARIANTS = {
    "n/a",
    "na",
    "n.d.",
    "nd",
    "n d",
    "nao se aplica",
    "nao aplicavel",
    "nao informado",
    "not applicable",
    "not informed",
}
INLINE_VALUE_SEPARATORS = ("|", ":", " - ", " => ", " = ")
MIN_EVIDENCE_CHARS = 12
REPAIR_MAX_EVIDENCE_CHARS = 240
MAX_EXPAND_CANDIDATES = 12
SHORT_SIGNAL_TOKENS = {"sim", "nao", "ok", "yes", "no", "n/a", "na", "nd", "n.d."}

QUESTION_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "como",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "esta",
    "este",
    "existe",
    "ha",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "onde",
    "os",
    "para",
    "por",
    "porque",
    "possui",
    "qual",
    "quais",
    "quando",
    "quanto",
    "que",
    "quem",
    "se",
    "sem",
    "tempo",
    "um",
    "uma",
}

REASON_METRIC_MAP = {
    "evidence_is_prompt": "guard_drop_prompt_evidence_count",
    "evidence_header_like": "guard_drop_prompt_evidence_count",
    "evidence_too_generic": "guard_drop_generic_evidence_count",
    "filtered_evidence_empty": "guard_drop_generic_evidence_count",
    "strict_missing_chunk_trace": "guard_drop_missing_chunk_count",
    "missing_chunk_trace": "guard_drop_missing_chunk_count",
    "evidence_too_short": "guard_drop_evidence_not_in_chunk_count",
    "evidence_missing_in_chunk": "guard_drop_evidence_not_in_chunk_count",
    "missing_chunk_text": "guard_drop_missing_chunk_count",
    "missing_explicit_anchor_for_question": "guard_drop_generic_evidence_count",
}

STRICT_EXPLICIT_ANSWERABILITY = {
    "4_2_termo_sigilo_terceiros",
    "5_2_termo_sigilo_transfer",
    "3_1_nuvem_criptografado",
    "4_1_contrato_apresentado",
    "6_classificacao_documentos",
    "9_somente_quando_necessario",
    "17_1_medidas_seg_comprovadas",
    "17_2_medidas_seg_descritas",
    "18_1_admin_comprovadas",
    "8_finalidade_explicita_ao_titular",
    "4_dados_menores",
}

QUESTION_EXPLICIT_ANCHORS = {
    "4_2_termo_sigilo_terceiros": [
        "termo de sigilo",
        "termo de confidencialidade",
        "acordo de confidencialidade",
        "clausula de confidencialidade",
        "contrato de confidencialidade",
        "nda",
        "non disclosure",
        "non-disclosure",
        "sigilo",
        "confidencialidade",
    ],
    "9_somente_quando_necessario": [
        "necessario",
        "necessários",
        "necessarios",
        "estritamente necessario",
        "estritamente necessarios",
        "somente quando necessario",
        "apenas quando necessario",
        "minimizacao",
        "minimizacao de dados",
        "atinge o seu objetivo",
        "objetivo do processo",
    ],
    "4_dados_menores": [
        "menor",
        "menores",
        "menor de idade",
        "menores de 18 anos",
        "dados de menores",
        "filhos menores",
        "dependentes menores",
        "crianca",
        "adolescente",
    ],
    "5_2_termo_sigilo_transfer": [
        "termo de sigilo",
        "termo de confidencialidade",
        "acordo de confidencialidade",
        "clausula de confidencialidade",
        "contrato de confidencialidade",
        "nda",
        "non disclosure",
        "non-disclosure",
        "sigilo",
        "confidencialidade",
    ],
    "4_1_contrato_apresentado": [
        "contrato apresentado",
        "contrato anexado",
        "contrato disponível",
        "contrato disponivel",
        "contrato foi apresentado",
        "copia do contrato",
        "cópia do contrato",
        "anexo contrato",
    ],
    "6_classificacao_documentos": [
        "classificacao de documentos",
        "classificacao da informacao",
        "classificacao documental",
        "informacao classificada",
        "nivel de sigilo",
        "uso interno",
        "documento confidencial",
        "documento restrito",
        "documento sigiloso",
        "documento publico",
    ],
    "3_1_nuvem_criptografado": [
        "criptografado",
        "criptografia",
        "encryption",
        "encrypted",
    ],
    "17_1_medidas_seg_comprovadas": [
        "comprovado",
        "comprovacao",
        "comprovação",
        "evidencia",
        "evidência",
        "registro",
        "log",
        "auditoria",
        "documentado",
        "comprovavel",
        "comprovável",
    ],
        "17_2_medidas_seg_descritas": [
        "descrito",
        "descritas",
        "documentado",
        "politica",
        "política",
        "procedimento",
        "controle de acesso",
        "gestao de acesso",
        "perfil de acesso",
        "protegido por senha",
        "senha de acesso",
        "senha unica",
        "senha",
        "acesso restrito",
        "armario trancado",
        "fechada diariamente",
        "mfa",
        "backup",
        "antivirus",
        "antivírus",
        "firewall",
    ],
    "18_1_admin_comprovadas": [
        "comprovado",
        "comprovacao",
        "comprovação",
        "evidencia",
        "evidência",
        "registro",
        "documentado",
        "comprovavel",
        "comprovável",
    ],
    "8_finalidade_explicita_ao_titular": [
        "informada ao titular",
        "informada para o titular",
        "informada para o portador do dado",
        "portador do dado",
        "ciencia do titular",
        "ciência do titular",
        "aviso de privacidade",
        "consentimento",
        "finalidade informada",
        "finalidade explicita",
        "finalidade explícita",
        "finalidade e explicita",
        "finalidade é explícita",
        "finalidade e explicita e informada",
        "finalidade é explícita e informada",
        "comunicar o titular sobre o tratamento realizado e a finalidade",
    ],
}

def strict_mode_enabled() -> bool:
    return (os.getenv("STRICT_MODE", "true") or "true").strip().lower() == "true"


def _normalize_text(value: str) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


NA_VARIANTS_NORMALIZED = {_normalize_text(v) for v in NA_VARIANTS}


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", _normalize_text(value))


def _answer_signal(value: str) -> str:
    norm = _normalize_text(value)
    if not norm:
        return ""
    if any(re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", norm) for tok in YES_TOKENS):
        return "yes"
    if any(re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", norm) for tok in NO_TOKENS):
        return "no"
    if norm in NA_VARIANTS_NORMALIZED:
        return "na"
    return ""


def _has_inline_answer_payload(norm: str) -> bool:
    for sep in ("?", ":", "|", " - ", " => ", " = "):
        if sep not in norm:
            continue
        tail = norm.split(sep, 1)[1].strip()
        if not tail:
            continue
        signal = _answer_signal(tail)
        if signal in {"yes", "no", "na"}:
            return True
        tokens = _tokenize(tail)
        if len(tokens) >= 2 and not all(tok in GENERIC_TOKENS for tok in tokens):
            return True
        if len(tokens) == 1 and len(tokens[0]) >= 4 and tokens[0] not in GENERIC_TOKENS:
            return True
    return False


def _looks_like_header_sentence(norm: str) -> bool:
    tokens = _tokenize(norm)
    if not tokens:
        return False
    if len(tokens) > 12:
        return False
    if _has_inline_answer_payload(norm):
        return False
    if any(re.search(pattern, norm) for pattern in PROMPT_PATTERNS):
        return True
    has_data_context = any(tok in {"dados", "dado", "processo"} for tok in tokens)
    has_header_terms = sum(1 for tok in tokens if tok in GENERIC_HEADER_TOKENS) >= 3
    return bool(has_data_context and has_header_terms)


def looks_like_prompt(text: str, question_text: Optional[str] = None) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False

    # "Pergunta + resposta explicita" nao deve ser bloqueada como prompt.
    if _answer_signal(norm) or _has_inline_answer_payload(norm):
        return False

    for pattern in PROMPT_PATTERNS:
        if re.search(pattern, norm):
            return True

    if _looks_like_header_sentence(norm):
        return True

    if question_text:
        qnorm = _normalize_text(question_text)
        if qnorm:
            ratio = SequenceMatcher(None, norm, qnorm).ratio()
            if ratio >= 0.88:
                return True
            if norm in qnorm or qnorm in norm:
                return True

    return False


def is_too_generic(text: str) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return True
    if _answer_signal(norm):
        return False

    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, norm):
            return True

    if _looks_like_header_sentence(norm):
        return True

    tokens = _tokenize(norm)
    if len(tokens) <= 1:
        if len(tokens) == 1 and len(tokens[0]) >= 4 and tokens[0] not in GENERIC_TOKENS:
            return False
        return True
    if len(tokens) <= 4 and all(tok in GENERIC_TOKENS for tok in tokens):
        return True
    return False


def _legacy_is_value_like_unused(text: str, question_text: Optional[str] = None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.endswith("?"):
        return False
    if looks_like_prompt(raw, question_text=question_text):
        return False
    if is_too_generic(raw):
        return False

    if re.search(r"\d", raw):
        return True
    if "," in raw or ";" in raw:
        return True
    if re.search(r"\b[A-Z]{2,}\b", raw):
        return True
    if "@" in raw or re.search(r"https?://", raw, flags=re.IGNORECASE):
        return True

    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", raw)
    return len(tokens) >= 3


def evidence_matches_chunk(evidence: str, chunk_text: str) -> bool:
    ev = _normalize_text(evidence)
    chunk = _normalize_text(chunk_text)
    if not ev or not chunk:
        return False
    return ev in chunk


def _question_schema(question_code: str) -> Dict[str, Any]:
    schema = dict(DEFAULT_QUESTION_SCHEMA)
    schema.update(dict(QUESTION_SCHEMA.get(str(question_code), {})))
    return schema


def _normalize_evidence_list(evidence: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in evidence or []:
        txt = str(item or "").strip()
        if not txt:
            continue
        if txt not in out:
            out.append(txt[:240])
        if len(out) >= 4:
            break
    return out


def _normalize_ids(used_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in used_ids or []:
        txt = str(item or "").strip()
        if not txt or txt in out:
            continue
        out.append(txt[:120])
        if len(out) >= 8:
            break
    return out


def is_value_like(text: str, question_text: Optional[str] = None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.endswith("?"):
        return False
    if looks_like_prompt(raw, question_text=question_text):
        return False
    if is_too_generic(raw):
        return False

    if re.search(r"\d", raw):
        return True
    if "," in raw or ";" in raw:
        return True
    if re.search(r"\b[A-Z]{2,}\b", raw):
        return True
    if "@" in raw or re.search(r"https?://", raw, flags=re.IGNORECASE):
        return True

    tokens = _tokenize(raw)
    return len(tokens) >= 3


def _is_short_signal_evidence(text: str) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False
    if len(norm) >= MIN_EVIDENCE_CHARS:
        return False
    return norm in SHORT_SIGNAL_TOKENS or _answer_signal(norm) in {"yes", "no", "na"}


def _is_short_evidence(text: str) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False
    return len(norm) < MIN_EVIDENCE_CHARS or len(_tokenize(norm)) <= 1


def _extract_inline_payload(text: str) -> Tuple[str, str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", "", ""
    for sep in ("|", ":", " - ", " \u2013 ", " \u2014 ", " => ", " = "):
        if sep not in raw:
            continue
        head, tail = raw.split(sep, 1)
        return head.strip(), tail.strip(), sep
    return "", "", ""


def _is_relevant_keyword(term: str) -> bool:
    norm = _normalize_text(term)
    if not norm:
        return False
    if norm in QUESTION_STOPWORDS or norm in GENERIC_TOKENS or norm in GENERIC_HEADER_TOKENS:
        return False
    return len(_tokenize(norm)) >= 1


def _required_optional_keyword_terms(
    question_code: str,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
) -> Tuple[List[str], List[str]]:
    required: List[str] = []
    optional: List[str] = []

    def _append_unique(dest: List[str], raw_terms: Sequence[str]) -> None:
        for raw in raw_terms or []:
            norm = _normalize_text(str(raw or ""))
            if not _is_relevant_keyword(norm):
                continue
            compact = "".join(re.findall(r"[a-z0-9]+", norm))
            if compact and compact not in dest:
                dest.append(compact)

    schema = _question_schema(question_code)
    _append_unique(required, [str(term) for term in (schema.get("required_terms") or []) if str(term).strip()])
    if isinstance(question_keywords, dict):
        _append_unique(required, [str(term) for term in (question_keywords.get("required") or []) if str(term).strip()])
        _append_unique(optional, [str(term) for term in (question_keywords.get("optional") or []) if str(term).strip()])

    question_terms = [
        "".join(re.findall(r"[a-z0-9]+", tok))
        for tok in _tokenize(question_text or "")
        if _is_relevant_keyword(tok)
    ]
    for term in question_terms:
        if term and term not in required and term not in optional:
            optional.append(term)

    return required[:8], optional[:8]


def _compact_text(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", _normalize_text(value)))


def _keyword_overlap_count(text: str, keywords: Sequence[str]) -> int:
    norm = _normalize_text(text)
    compact = _compact_text(text)
    hits = 0
    for keyword in keywords:
        kw = _compact_text(keyword)
        if not kw:
            continue
        if keyword in norm or kw in compact:
            hits += 1
    return hits


def _classify_fragment_type(text: str) -> str:
    raw = str(text or "").strip()
    if re.match(r"^\s*[-*•]\s+", raw):
        return "bullet"
    _, payload, _ = _extract_inline_payload(raw)
    if payload and _clean_text_for_type(payload := payload):
        if _compact_text(raw) == _compact_text(payload):
            return "inline_payload"
    if re.search(r"[.!;:]\s", raw) or raw.endswith("."):
        return "sentence"
    return "line"


def _clean_text_for_type(text: str) -> str:
    return str(text or "").strip()


def _prompt_reason(text: str, question_text: Optional[str] = None) -> str:
    raw = str(text or "").strip()
    norm = _normalize_text(raw)
    if not norm:
        return ""

    if "?" in raw:
        _, tail = raw.split("?", 1)
        tail_norm = _normalize_text(tail)
        if tail_norm and (_answer_signal(tail_norm) in {"yes", "no", "na"} or is_value_like(tail, question_text=question_text)):
            return ""
    _, payload, _ = _extract_inline_payload(raw)
    if payload:
        payload_norm = _normalize_text(payload)
        if _answer_signal(payload_norm) in {"yes", "no", "na"} or is_value_like(payload, question_text=question_text):
            return ""
    if re.search(r"(?::|\||\s[-\u2013\u2014])\s*$", raw):
        return "evidence_header_like"
    if norm.endswith("?"):
        return "evidence_is_prompt"
    if question_text:
        qnorm = _normalize_text(question_text)
        if qnorm and (norm == qnorm or norm in qnorm or qnorm in norm):
            return "evidence_is_prompt"
    if _looks_like_header_sentence(norm):
        return "evidence_header_like"
    if looks_like_prompt(text, question_text=question_text):
        return "evidence_is_prompt"
    return ""


def _classify_fragment_type(text: str) -> str:
    raw = str(text or "").strip()
    if re.match(r"^\s*[-*]\s+", raw):
        return "bullet"
    _, payload, _ = _extract_inline_payload(raw)
    if payload and _compact_text(raw) == _compact_text(payload):
        return "inline_payload"
    if re.search(r"[.!;:]\s", raw) or raw.endswith("."):
        return "sentence"
    return "line"


def _iter_chunk_lines(chunk_text: str) -> List[str]:
    raw = str(chunk_text or "").replace("\r\n", "\n").replace("\r", "\n")
    out: List[str] = []
    for line in raw.splitlines():
        clean = str(line or "").strip()
        if clean:
            out.append(clean[:REPAIR_MAX_EVIDENCE_CHARS])
    return out


def _iter_chunk_fragments(chunk_text: str) -> List[str]:
    candidates: List[str] = []
    for clean in _iter_chunk_lines(chunk_text):
        candidates.append(clean)
        if "|" in clean:
            parts = [part.strip() for part in clean.split("|")]
            candidates.extend(part for part in parts[1:] if part)
        _, payload, _ = _extract_inline_payload(clean)
        if payload:
            candidates.append(payload)

    deduped: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clipped = candidate[:REPAIR_MAX_EVIDENCE_CHARS]
        if not clipped or clipped in seen:
            continue
        seen.add(clipped)
        deduped.append(clipped)
    return deduped


def _is_response_like(
    text: str,
    question_code: str,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False
    if _prompt_reason(text, question_text=question_text):
        return False
    if is_too_generic(text):
        return False
    if _answer_signal(norm):
        return True
    if _has_inline_answer_payload(norm):
        return True
    if is_value_like(text, question_text=question_text):
        return True
    if re.search(r"\d", text) or "," in text or ";" in text:
        return True
    required_terms, optional_terms = _required_optional_keyword_terms(question_code, question_text, question_keywords)
    if _keyword_overlap_count(text, required_terms + optional_terms) > 0:
        return True
    return False


def _score_anchored_fragment(
    fragment: str,
    question_code: str,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
    evidence_hint: str = "",
) -> Tuple[float, int, int]:
    required_terms, optional_terms = _required_optional_keyword_terms(question_code, question_text, question_keywords)
    required_hits = _keyword_overlap_count(fragment, required_terms)
    optional_hits = _keyword_overlap_count(fragment, optional_terms)
    norm = _normalize_text(fragment)

    score = required_hits * 4.0 + optional_hits * 2.0
    if _answer_signal(norm):
        score += 2.0
    if _has_inline_answer_payload(norm):
        score += 2.5
    if is_value_like(fragment, question_text=question_text):
        score += 2.0
    score += min(1.5, len(norm) / 80.0)

    hint_norm = _normalize_text(evidence_hint)
    if hint_norm:
        if _is_short_signal_evidence(evidence_hint):
            if re.search(rf"(^|[^a-z0-9]){re.escape(hint_norm)}([^a-z0-9]|$)", norm):
                score += 2.0
        elif hint_norm in norm or _compact_text(evidence_hint) in _compact_text(fragment):
            score += 1.5

    return score, required_hits, optional_hits


def _build_anchoring_debug(
    evidence: str,
    question_code: str,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    required_terms, optional_terms = _required_optional_keyword_terms(question_code, question_text, question_keywords)
    required_hits = _keyword_overlap_count(evidence, required_terms)
    optional_hits = _keyword_overlap_count(evidence, optional_terms)
    norm = _normalize_text(evidence)
    required_ratio = (required_hits / float(len(required_terms))) if required_terms else 0.0
    optional_ratio = (optional_hits / float(len(optional_terms))) if optional_terms else 0.0
    signal_or_value = 1.0 if (_answer_signal(norm) or is_value_like(evidence, question_text=question_text)) else 0.0
    length_score = min(1.0, len(norm) / 80.0)
    anchoring_score = min(
        1.0,
        0.45 * required_ratio + 0.20 * optional_ratio + 0.20 * signal_or_value + 0.15 * length_score,
    )
    return {
        "guard_anchoring_score": round(float(anchoring_score), 4),
        "guard_selected_fragment_type": _classify_fragment_type(evidence),
        "guard_keyword_hits_required": int(required_hits),
        "guard_keyword_hits_optional": int(optional_hits),
    }


def _direct_evidence_match(evidence: str, chunk_text: str) -> bool:
    if _is_short_signal_evidence(evidence):
        return False

    evidence_norm = _normalize_text(evidence)
    chunk_norm = _normalize_text(chunk_text)
    if not evidence_norm or not chunk_norm:
        return False

    if _is_short_evidence(evidence) and len(_tokenize(evidence_norm)) <= 1:
        compact_evidence = _compact_text(evidence)
        compact_chunk = _compact_text(chunk_text)
        if re.search(rf"(^|[^a-z0-9]){re.escape(evidence_norm)}([^a-z0-9]|$)", chunk_norm):
            return True
        return bool(compact_evidence and compact_evidence in compact_chunk)

    return evidence_matches_chunk(evidence, chunk_text)


def _expand_short_evidence(
    evidence: str,
    chunk_texts: Dict[str, str],
    question_code: str,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
) -> str:
    if not _is_short_evidence(evidence):
        return ""

    evidence_norm = _normalize_text(evidence)
    evidence_compact = _compact_text(evidence)
    if not evidence_norm or not evidence_compact:
        return ""

    required_terms, optional_terms = _required_optional_keyword_terms(question_code, question_text, question_keywords)
    candidates: List[Tuple[float, str]] = []

    for chunk_text in chunk_texts.values():
        for line in _iter_chunk_lines(chunk_text):
            line_norm = _normalize_text(line)
            line_compact = _compact_text(line)
            if not line_norm or len(line_norm) <= len(evidence_norm):
                continue

            matched = False
            if _is_short_signal_evidence(evidence):
                if re.search(rf"(^|[^a-z0-9]){re.escape(evidence_norm)}([^a-z0-9]|$)", line_norm):
                    matched = _keyword_overlap_count(line, required_terms + optional_terms) > 0
            else:
                if len(_tokenize(evidence_norm)) <= 1:
                    matched = bool(
                        re.search(rf"(^|[^a-z0-9]){re.escape(evidence_norm)}([^a-z0-9]|$)", line_norm)
                        or (evidence_compact and evidence_compact in line_compact)
                    )
                else:
                    matched = evidence_norm in line_norm or evidence_compact in line_compact

            if not matched:
                continue
            if not _is_response_like(
                line,
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
            ):
                continue
            score, _, _ = _score_anchored_fragment(
                line,
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
                evidence_hint=evidence,
            )
            candidates.append((score, line[:REPAIR_MAX_EVIDENCE_CHARS]))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:MAX_EXPAND_CANDIDATES][0][1]


def _repair_evidence_from_chunks(
    chunk_texts: Dict[str, str],
    question_code: str,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
    seed_evidence: Optional[Sequence[str]] = None,
) -> str:
    required_terms, optional_terms = _required_optional_keyword_terms(question_code, question_text, question_keywords)
    seed_terms = [tok for ev in (seed_evidence or []) for tok in _tokenize(ev) if _is_relevant_keyword(tok)]
    seed_has_short_signal = any(_is_short_signal_evidence(ev) for ev in (seed_evidence or []))
    best_candidate = ""
    best_score = -1.0

    for chunk_text in chunk_texts.values():
        for fragment in _iter_chunk_fragments(chunk_text):
            if not evidence_matches_chunk(fragment, chunk_text):
                continue
            if not _is_response_like(
                fragment,
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
            ):
                continue

            required_hits = _keyword_overlap_count(fragment, required_terms)
            optional_hits = _keyword_overlap_count(fragment, optional_terms)
            seed_overlap = _keyword_overlap_count(fragment, seed_terms)
            norm = _normalize_text(fragment)

            if seed_has_short_signal and (required_terms or optional_terms):
                if required_hits == 0 and optional_hits == 0:
                    continue
            if required_terms or optional_terms:
                if required_hits == 0 and optional_hits == 0 and seed_overlap == 0:
                    continue

            score, _, _ = _score_anchored_fragment(
                fragment,
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
            )
            score += seed_overlap * 2.0
            if score > best_score:
                best_score = score
                best_candidate = fragment[:REPAIR_MAX_EVIDENCE_CHARS]

    return best_candidate

def _has_explicit_anchor_for_question(question_code: str, evidence: Sequence[str]) -> bool:
    if str(question_code or "").strip() not in STRICT_EXPLICIT_ANSWERABILITY:
        return True

    anchors = [str(anchor or "").strip() for anchor in QUESTION_EXPLICIT_ANCHORS.get(str(question_code), []) if str(anchor or "").strip()]
    if not anchors:
        return True

    normalized_evidence = " ".join(_normalize_text(ev) for ev in (evidence or []) if str(ev or "").strip())
    compact_evidence = "".join(re.findall(r"[a-z0-9]+", normalized_evidence))
    if not normalized_evidence:
        return False

    for anchor in anchors:
        anchor_norm = _normalize_text(anchor)
        if not anchor_norm:
            continue
        if anchor_norm in normalized_evidence:
            return True
        anchor_compact = "".join(re.findall(r"[a-z0-9]+", anchor_norm))
        if anchor_compact and anchor_compact in compact_evidence:
            return True
    return False

def _matches_required_terms(evidence: Sequence[str], required_terms: Sequence[str]) -> bool:
    if not required_terms:
        return True
    flat = " ".join(_normalize_text(ev) for ev in evidence)
    flat_compact = "".join(re.findall(r"[a-z0-9]+", flat))
    for term in required_terms:
        norm_term = _normalize_text(term)
        if not norm_term:
            continue
        if norm_term in flat:
            return True
        compact_term = "".join(re.findall(r"[a-z0-9]+", norm_term))
        if compact_term and compact_term in flat_compact:
            return True
    return False

def _looks_like_country_answer(ev: str) -> bool:
    norm = _normalize_text(ev)
    if not norm:
        return False

    generic_only_terms = [
        "transferencia internacional",
        "transferencia internacional de dados",
        "fora do brasil",
        "fora do pais",
        "para o exterior",
        "cross-border",
    ]
    if any(term in norm for term in generic_only_terms):
        tokens = _tokenize(norm)
        if len(tokens) <= 5 and "," not in ev and ";" not in ev:
            return False

    word_candidates = re.findall(r"\b[a-z]{4,}\b", norm)
    blocked = {
        "transferencia",
        "internacional",
        "dados",
        "exterior",
        "brasil",
        "pais",
        "paises",
        "quais",
        "para",
        "fora",
    }
    meaningful = [tok for tok in word_candidates if tok not in blocked]
    if len(meaningful) >= 1:
        return True

    return "," in ev or ";" in ev or " e " in norm or " and " in norm

def _schema_validate(question_code: str, evidence: Sequence[str]) -> Tuple[bool, str]:
    schema = _question_schema(question_code)
    ev_list = [str(ev or "").strip() for ev in (evidence or []) if str(ev or "").strip()]
    if not ev_list:
        return False, "filtered_evidence_empty"

    qtype = str(schema.get("type") or "free_text").strip().lower()
    allow_na = bool(schema.get("allow_na", False))
    allow_binary_signal = bool(schema.get("allow_binary_signal", False))
    min_tokens = int(schema.get("min_tokens", 1) or 1)
    required_terms = [str(t) for t in (schema.get("required_terms") or []) if str(t).strip()]
    forbidden_terms = [str(t) for t in (schema.get("forbidden_terms") or []) if str(t).strip()]
    flat_text = " | ".join(ev_list)
    flat_norm = _normalize_text(flat_text)

    if forbidden_terms and any(_normalize_text(term) in flat_norm for term in forbidden_terms):
        return False, "schema_forbidden_terms"

    if question_code == "4_possui_contrato":
        # Evita falso positivo quando aparece "contrato" em contexto de processo/base legal,
        # mas sem evidência de contrato do sistema com fornecedor.
        strong_header_terms = [
            "possui contrato escrito assinado",
            "contrato com o fornecedor",
            "contrato do sistema",
            "fornecedor do sistema",
        ]
        contract_terms = ["contrato", "assinado", "instrumento contratual"]
        supplier_terms = ["fornecedor", "vendor", "provedor", "fabricante"]

        has_strong_header = any(_normalize_text(term) in flat_norm for term in strong_header_terms)
        has_contract_term = any(_normalize_text(term) in flat_norm for term in contract_terms)
        has_supplier_term = any(_normalize_text(term) in flat_norm for term in supplier_terms)

        if not has_strong_header and not (has_contract_term and has_supplier_term):
            return False, "schema_contract_not_system_specific"

    # Global rule: explicit NA values are respondable when anchored to document evidence,
    # but still must respect contextual required_terms when a question defines them.
    for ev in ev_list:
        if _answer_signal(ev) == "na":
            if not required_terms or _matches_required_terms([ev], required_terms):
                return True, ""

    if qtype == "yes_no":
        for ev in ev_list:
            signal = _answer_signal(ev)
            has_required_terms = _matches_required_terms([ev], required_terms) if required_terms else True

            if signal in {"yes", "no"}:
                if has_required_terms:
                    return True, ""
                continue

            if allow_na and signal == "na":
                if has_required_terms:
                    return True, ""
                continue

            if required_terms and has_required_terms:
                return True, ""

            if is_value_like(ev):
                if has_required_terms:
                    return True, ""
                continue

        return False, "schema_yes_no_not_explicit"

    if qtype == "location":
        for ev in ev_list:
            if is_too_generic(ev):
                continue
            if len(_tokenize(ev)) < min_tokens:
                continue
            if required_terms and not _matches_required_terms([ev], required_terms):
                continue
            return True, ""
        return False, "schema_location_not_specific"

    if qtype == "list":
        for ev in ev_list:
            ev_norm = _normalize_text(ev)

            if allow_na and _answer_signal(ev_norm) == "na":
                return True, ""

            if len(_tokenize(ev_norm)) < min_tokens:
                continue

            if required_terms and not _matches_required_terms([ev_norm], required_terms):
                continue

            if question_code == "5_1_paises":
                if _looks_like_country_answer(ev):
                    return True, ""
                continue

            if "," in ev or ";" in ev or " e " in ev_norm or " and " in ev_norm:
                return True, ""

            if len(_tokenize(ev_norm)) >= max(2, min_tokens):
                return True, ""

        return False, "schema_list_not_specific"

    # free_text
    for ev in ev_list:
        ev_norm = _normalize_text(ev)
        signal = _answer_signal(ev_norm)
        if allow_na and signal == "na":
            return True, ""
        if allow_binary_signal and signal in {"yes", "no"}:
            return True, ""
        if len(_tokenize(ev_norm)) < min_tokens:
            continue
        if required_terms and not _matches_required_terms([ev_norm], required_terms):
            continue
        if is_too_generic(ev_norm):
            continue
        return True, ""
    return False, "schema_free_text_not_specific"


def validate_answerability(
    question_code: str,
    answerable: int,
    evidence: Sequence[str],
    used_ids: Sequence[str],
    chunk_texts: Optional[Dict[str, str]] = None,
    question_text: Optional[str] = None,
    question_keywords: Optional[Dict[str, Sequence[str]]] = None,
    debug_out: Optional[Dict[str, Any]] = None,
    strict_mode: Optional[bool] = None,
) -> Tuple[int, str, List[str]]:
    if int(answerable) != 1:
        return 0, "not_answerable", []

    strict = strict_mode_enabled() if strict_mode is None else bool(strict_mode)
    evidence_list = _normalize_evidence_list(evidence)
    used_chunk_ids = _normalize_ids(used_ids)

    filtered: List[str] = []
    prompt_like: List[str] = []
    header_like: List[str] = []
    generic_like: List[str] = []
    for ev in evidence_list:
        prompt_reason = _prompt_reason(ev, question_text=question_text)
        if prompt_reason == "evidence_header_like":
            header_like.append(ev)
            continue
        if prompt_reason == "evidence_is_prompt":
            prompt_like.append(ev)
            continue
        if is_too_generic(ev):
            generic_like.append(ev)
            continue
        filtered.append(ev)

    if not used_chunk_ids:
        if not filtered:
            if header_like:
                return 0, "evidence_header_like", []
            if prompt_like:
                return 0, "evidence_is_prompt", []
            if generic_like:
                return 0, "evidence_too_generic", []
            return 0, "filtered_evidence_empty", []
        if strict:
            return 0, "strict_missing_chunk_trace", filtered[:2]
        return 0, "missing_chunk_trace", filtered[:2]

    repair_reason = ""
    if chunk_texts is not None:
        matched: List[str] = []
        normalized_chunk_texts = {
            str(cid): str(chunk_texts.get(cid, "") or "")
            for cid in used_chunk_ids
        }
        if strict and not any(normalized_chunk_texts.values()):
            return 0, "missing_chunk_text", filtered[:2]
        for ev in filtered:
            expanded = _expand_short_evidence(
                ev,
                normalized_chunk_texts,
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
            )
            if expanded:
                matched.append(expanded)
                if repair_reason != "auto_repaired_evidence_from_chunk":
                    repair_reason = "auto_expanded_short_evidence"
                continue
            if any(_direct_evidence_match(ev, txt) for txt in normalized_chunk_texts.values()):
                matched.append(ev)
        if not matched:
            repaired = _repair_evidence_from_chunks(
                normalized_chunk_texts,
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
                seed_evidence=evidence_list,
            )
            if repaired:
                matched = [repaired]
                repair_reason = "auto_repaired_evidence_from_chunk"
        if not matched:
            if header_like:
                return 0, "evidence_header_like", []
            if prompt_like:
                return 0, "evidence_is_prompt", []
            if generic_like and not filtered:
                return 0, "evidence_too_generic", []
            if filtered and all(_is_short_evidence(ev) for ev in filtered):
                return 0, "evidence_too_short", []
            if not filtered:
                return 0, "filtered_evidence_empty", []
            return 0, "evidence_missing_in_chunk", []
        filtered = matched
    elif not filtered:
        if header_like:
            return 0, "evidence_header_like", []
        if prompt_like:
            return 0, "evidence_is_prompt", []
        if generic_like:
            return 0, "evidence_too_generic", []
        return 0, "filtered_evidence_empty", []

    if not _has_explicit_anchor_for_question(question_code, filtered):
        return 0, "missing_explicit_anchor_for_question", []

    schema_ok, schema_reason = _schema_validate(question_code, filtered)
    if not schema_ok:
        return 0, schema_reason or "schema_validation_failed", []

    if debug_out is not None and filtered:
        debug_out.clear()
        debug_out.update(
            _build_anchoring_debug(
                filtered[0],
                question_code=question_code,
                question_text=question_text,
                question_keywords=question_keywords,
            )
        )

    return 1, repair_reason, filtered[:2]


def reason_to_metric(reason: str) -> Optional[str]:
    return REASON_METRIC_MAP.get(str(reason or "").strip())


def bump_metric(stats: Dict[str, int], reason: str) -> None:
    metric = reason_to_metric(reason)
    if not metric:
        return
    stats[metric] = int(stats.get(metric, 0)) + 1
