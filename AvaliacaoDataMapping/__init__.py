import azure.functions as func
import io, os, re, json, unicodedata, csv
from typing import Tuple, List, Dict, Optional
import logging, sys

import hashlib, time, tempfile
from collections import Counter

from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_LLM_EVAL = (os.getenv("USE_LLM_EVAL", "true").lower() == "true") and bool(os.getenv("OPENAI_API_KEY"))
LLM_DOUBLE_PASS = os.getenv("LLM_DOUBLE_PASS", "false").lower() == "true"
try:
    _chunk_env = int(os.getenv("LLM_CHUNK_SIZE", "6000"))
except Exception:
    _chunk_env = 6000
LLM_CHUNK_SIZE = max(2000, min(_chunk_env, 15000))
try:
    LLM_MAX_CHARS = int(os.getenv("LLM_MAX_CHARS", "60000"))
except Exception:
    LLM_MAX_CHARS = 60000


STRICT_MODE = os.getenv("STRICT_MODE", "true").lower() == "true"

_ANALYSIS_CACHE_KEY = "__pp_analysis_cache__"
_SENSITIVE_CACHE_ENTRY = "process_sensitive_eval"
_SENSITIVE_COLUMN_HINTS = [
    "categoria dos dados",
    "categoria do dado",
    "categoria dado",
    "natureza dos dados",
    "natureza do dado",
    "tipo de dado",
    "tipo dos dados",
    "classificacao dos dados",
    "classificacao do dado",
    "sensivel",
]
_SENSITIVE_SKIP_PATTERNS = [
    "pessoal ou sens",
    "apenas dados pessoais",
    "somente dados pessoais",
]
_SENSITIVE_NEGATIVE_PATTERNS = [
    "nao ha dado sens",
    "nao ha dados sens",
    "nao possui dado sens",
    "nao possui dados sens",
    "sem dado sens",
    "sem dados sens",
    "nao tratamos dado sens",
    "nao trata dado sens",
]
_SENSITIVE_VALUE_KEYWORDS = [
    "sensivel",
    "sensiveis",
    "dado sens",
    "dados sens",
    "biometr",
    "genet",
    "dna",
    "saude",
    "medic",
    "clinico",
    "hospital",
    "prontuario",
    "diagnost",
    "psicolog",
    "laudo",
    "exame",
    "tratamento",
    "doenca",
    "cid",
    "relig",
    "crenca",
    "convic",
    "orientacao sexual",
    "preferencia sexual",
    "vida sexual",
    "sexualidade",
    "opiniao politica",
    "conviccao politica",
    "partido politico",
    "filiacao partidaria",
    "sindic",
    "filiacao sindical",
    "etnia",
    "racial",
    "origem racial",
    "origem etnica",
    "raca",
    "deficienc",
    "deficiente",
    "pcd",
    "antecedente criminal",
    "historico criminal",
    "condenacao",
]
_PERSONAL_ONLY_KEYWORDS = [
    "pessoal",
    "dados pessoais",
    "somente dados pessoais",
    "so dados pessoais",
    "apenas dados pessoais",
    "only personal data",
]
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "3")) # Limite de requisição
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(24 * 60 * 60)))

_RATE_FILE = os.path.join(tempfile.gettempdir(), "pp_rate_limit.json")

try:
    import pandas as pd
except:
    pd = None

try:
    from docx import Document
except:
    Document = None

try:
    from pdfminer_high_level import extract_text as pdf_extract_text  # fallback de nome
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
    except Exception:
        pdf_extract_text = None

try:
    import cgi  # removido no Python 3.13
except Exception:
    cgi = None


def _rl_load() -> dict:
    try:
        with open(_RATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _rl_save(data: dict) -> None:
    try:
        with open(_RATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:

        pass

def _client_ip_from_headers(req: func.HttpRequest) -> str:
    hdrs = { (k or "").lower(): v for k, v in (req.headers or {}).items() }
    raw = (
        hdrs.get("x-forwarded-for")
        or hdrs.get("x-original-forwarded-for")
        or hdrs.get("x-real-ip")
        or hdrs.get("x-client-ip")
        or hdrs.get("x-functions-client-ip")  # Azure Functions local/proxy
        or hdrs.get("cf-connecting-ip")
        or hdrs.get("client-ip")
    )
    if raw:
        return raw.split(",")[0].strip()
    return "127.0.0.1"  # fallback pra dev local


def rate_limit_allow(req: func.HttpRequest, max_requests: int = RATE_LIMIT_MAX, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
    """
    Retorna (ok: bool, ip: str, retry_after_seconds: int).
    Persistência local por instância (arquivo em /tmp).
    Para produção distribuí­da, trocar por Table/Cosmos/Redis.
    """
    now = int(time.time())
    ip = _client_ip_from_headers(req)
    data = _rl_load()

    arr = [int(t) for t in (data.get(ip) or []) if now - int(t) < window_seconds]

    if len(arr) >= max_requests:
        retry_after = max(1, window_seconds - (now - min(arr)))
        return False, ip, retry_after

    # anota a tentativa atual
    arr.append(now)
    data[ip] = arr
    _rl_save(data)
    return True, ip, 0
# --- FIM RATE LIMIT ---

def _cors_headers():
    return {
        "Access-Control-Allow-Origin": os.getenv("CORS_ALLOW_ORIGIN", "*"),
        "Access-Control-Allow-Headers": "Content-Type, X-Requested-With",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Expose-Headers": "Retry-After",
    }

def _json_response(payload: dict, status: int = 200, extra_headers: dict | None = None):
    hdrs = _cors_headers()
    if extra_headers:
        hdrs.update(extra_headers)
    return func.HttpResponse(
        json.dumps(payload, ensure_ascii=False),
        status_code=status,
        mimetype="application/json",
        headers=hdrs
    )



# -----------------------------
# Util: normalizaÃ§Ã£o de texto
# -----------------------------
def norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def _get_analysis_cache(structured: dict | None) -> Optional[dict]:
    if not isinstance(structured, dict):
        return None
    cache = structured.get(_ANALYSIS_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        structured[_ANALYSIS_CACHE_KEY] = cache
    return cache

def _value_is_personal_only(text_norm: str) -> bool:
    if not text_norm:
        return False
    if "sens" in text_norm:
        return False
    if "pessoal" in text_norm:
        return True
    return any(token in text_norm for token in _PERSONAL_ONLY_KEYWORDS)

def _cell_mentions_sensitive(text_norm: str) -> bool:
    if not text_norm:
        return False
    for neg in _SENSITIVE_NEGATIVE_PATTERNS:
        if neg in text_norm:
            return False
    if any(skip in text_norm for skip in _SENSITIVE_SKIP_PATTERNS):
        return False
    return any(kw in text_norm for kw in _SENSITIVE_VALUE_KEYWORDS)

def _looks_like_sensitive_category_header(col_norm: str) -> bool:
    if not col_norm:
        return False
    return any(h in col_norm for h in _SENSITIVE_COLUMN_HINTS)

def _scan_sensitive_values_from_df(df) -> Tuple[List[str], bool]:
    hits: List[str] = []
    try:
        cols = list(df.columns)
    except Exception:
        return hits, False
    sensitive_idxs: List[int] = []
    for idx, col in enumerate(cols):
        col_norm = norm(str(col))
        if _looks_like_sensitive_category_header(col_norm):
            sensitive_idxs.append(idx)
    saw_personal_value = False
    all_personal = True
    try:
        row_iter = df.itertuples(index=False, name=None)
    except Exception:
        row_iter = []
    for row in row_iter:
        for cidx, raw_val in enumerate(row):
            if not _is_nonempty(raw_val):
                continue
            text = str(raw_val).strip()
            norm_text = norm(text)
            if not norm_text:
                continue
            if cidx in sensitive_idxs:
                saw_personal_value = True
                if not _value_is_personal_only(norm_text):
                    all_personal = False
            if _cell_mentions_sensitive(norm_text):
                colname = str(cols[cidx])
                snippet = f"{colname}: {text}"
                if snippet not in hits:
                    hits.append(snippet[:240])
                all_personal = False
                break
        if len(hits) >= 3:
            break
    table_only_personal = bool(sensitive_idxs) and saw_personal_value and all_personal and not hits
    return hits, table_only_personal

def _detect_sensitive_data_usage(structured: dict | None) -> Dict[str, object]:
    default = {"has_sensitive": False, "evidence": [], "only_personal": False}
    if pd is None:
        return default
    cache = _get_analysis_cache(structured)
    if cache is not None and _SENSITIVE_CACHE_ENTRY in cache:
        return cache[_SENSITIVE_CACHE_ENTRY]
    if not isinstance(structured, dict):
        return default
    df = structured.get("processos")
    if df is None or not hasattr(df, "columns") or getattr(df, "empty", True):
        if cache is not None:
            cache[_SENSITIVE_CACHE_ENTRY] = default
        return default
    try:
        evidence, only_personal = _scan_sensitive_values_from_df(df)
    except Exception:
        evidence, only_personal = [], False
    result = {
        "has_sensitive": bool(evidence),
        "evidence": (evidence or [])[:3],
        "only_personal": bool(only_personal),
    }
    if cache is not None:
        cache[_SENSITIVE_CACHE_ENTRY] = result
    return result

def parse_multipart_formdata(req: func.HttpRequest):
    content_type = req.headers.get("Content-Type") or req.headers.get("content-type") or ""
    body = req.get_body()
    if "multipart/form-data" not in content_type:
        raise ValueError("Envie como multipart/form-data com arquivo (campo 'file').")
    env = {
        "REQUEST_METHOD": "POST",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(body))
    }
    fp = io.BytesIO(body)
    form = cgi.FieldStorage(fp=fp, environ=env, keep_blank_values=True)
    organograma = None
    if "organograma" in form and getattr(form["organograma"], "value", None):
        v = (form["organograma"].value or "").strip().lower()
        organograma = "sim" if v in ("sim","yes","true","1") else ("nao" if v in ("nao","não","no","false","0") else None)
    if "file" not in form or not getattr(form["file"], "filename", None):
        raise ValueError("Arquivo não encontrado no form-data (campo 'file').")
    filename = form["file"].filename
    file_bytes = form["file"].file.read()
    return organograma, filename, file_bytes


# -------------------------------------------------------
# DefiniÃ§Ãµes das perguntas (cÃ³digos â†” textos humanos)
# -------------------------------------------------------
AREAS_15 = [
    ("1_HD_externo_pendrive", []),
    ("1_1_hd_tem_senha", []),
    ("1_2_hd_armazenado_onde", []),
    ("2_compart_dados_rede", []),
    ("2_1_rede_protecao_senha", []),
    ("2_2_politica_restricao_acesso", []),
    ("3_compart_nuvem", []),
    ("3_1_nuvem_criptografado", []),
    ("4_compart_terceiros", []),
    ("4_1_forma_compart", []),
    ("4_2_termo_sigilo_terceiros", []),
    ("5_transfer_internacional", []),
    ("5_1_paises", []),
    ("5_2_termo_sigilo_transfer", []),
    ("6_classificacao_documentos", [])
]

PROCESSOS_26 = [
    ("1_decisoes_automatizadas", []),
    ("2_trata_dados_pessoais", []),
    ("3_trata_dados_sensiveis", []),
    ("4_dados_menores", []),
    ("5_descricao_processo", []),
    ("6_dados_tratados_quais", []),
    ("6_1_fisico_ou_digital", []),
    ("6_2_armazenado_nuvem", []),
    ("6_3_compart_interno_terceiros", []),
    ("6_4_descarte_apos_uso", []),
    ("7_finalidade_legitima", []),
    ("8_finalidade_explicita_ao_titular", []),
    ("9_somente_quando_necessario", []),
    ("10_quem_usa_precisa", []),
    ("11_tempo_necessario_apenas", []),
    ("12_formato_dados_disponiveis", []),
    ("13_forma_transferencia_dados", []),
    ("14_processos_documentados", []),
    ("15_dados_precisos_claros_atualizados", []),
    ("16_titular_pode_corrigir", []),
    ("17_medidas_seg_cibernetica", []),
    ("17_1_medidas_seg_comprovadas", []),
    ("17_2_medidas_seg_descritas", []),
    ("18_medidas_admin_protecao", []),
    ("18_1_admin_comprovadas", []),
    ("18_2_admin_descritas", [])
]

SISTEMAS_5 = [
    ("1_nome_identificado", []),
    ("2_fornecedor_identificado", []),
    ("3_eh_web", []),
    ("4_possui_contrato", []),
    ("4_1_contrato_apresentado", [])
]

def _list_to_question_map(pairs: list[tuple[str, list]], human_texts: list[str]) -> dict[str, str]:
    mp: dict[str, str] = {}
    for (code, _), txt in zip(pairs, human_texts):
        mp[code] = txt
    return mp

AREAS_QTEXT = [
    "1- Possui HD Externo/Pen drive na Área?",
    "1.1 Este dispositivo tem senha?",
    "1.2 Onde fica armazenado?",
    "2- Possui compartilhamento de dados na rede?",
    "2.1 Existe proteção por senha?",
    "2.2 Existe política de restrição de acesso?",
    "3- Possui compartilhamento em nuvem?",
    "3.1 Os dados são criptografados?",
    "4- Possui compartilhamento com terceiros?",
    "4.1 Como é esse compartilhamento? (APIs, Arquivos, E-mail, etc.)",
    "4.2 Existe termo de sigilo?",
    "5- Existe transferência internacional de dados?",
    "5.1 Para quais paí­ses?",
    "5.2 Existe termo de sigilo?",
    "6- Existe classificação de documentos?"
]

PROCESSOS_QTEXT = [
    "1- Este processo possui decisões automatizadas?",
    "2- Possui tratamento de dados pessoais?",
    "3- Possui tratamento de dados sensíveis?",
    "4- Existem dados de menores de 18 anos?",
    "5- Existe uma descrição de como é realizado o processo?",
    "6- É informado quais dados são tratados?",
    "6.1 Pode ser identificado se é físico ou digital?",
    "6.2 Pode ser identificado se é armazenado em nuvem?",
    "6.3 Pode ser identificado se há compartilhamento interno ou com terceiros?",
    "6.4 Pode ser identificado se existe descarte/eliminação após o uso?",
    "7- É possível avaliar se os dados são tratados para finalidade legítima?",
    "8- Se a finalidade é explícita e informada ao titular?",
    "9- Os dados são utilizados somente quando necessário?",
    "10- Todos que utilizam os dados realmente possuem a necessidade?",
    "11- Os dados são mantidos somente pelo tempo necessário?",
    "12- Está documentado o formato no qual os dados estão disponíveis?",
    "13- Está documentada a forma como os dados são transferidos?",
    "14- Os processos onde os dados são utilizados estão documentados?",
    "15- Os dados são precisos, claros e atualizados?",
    "16- O titular do dado pode pedir sua correção?",
    "17- São aplicadas medidas básicas de segurança cibernética?",
    "17.1 Estas medidas podem ser comprovadas?",
    "17.2 Estas medidas estão descritas?",
    "18- Existem medidas administrativas para proteção dos dados?",
    "18.1 Estas medidas podem ser comprovadas?",
    "18.2 Estas medidas estão descritas?"
]

SISTEMAS_QTEXT = [
    "1- É possível identificar o nome de cada sistema utilizado pelas áreas?",
    "2- É possível identificar o fornecedor deste sistema?",
    "3- É possível identificar se trata-se de um sistema web?",
    "4- É possível identificar se este sistema possui algum contrato com o fornecedor?",
    "4.1 Este contrato é apresentado?"
]


AREAS_QMAP     = _list_to_question_map(AREAS_15, AREAS_QTEXT)
PROCESSOS_QMAP = _list_to_question_map(PROCESSOS_26, PROCESSOS_QTEXT)
SISTEMAS_QMAP  = _list_to_question_map(SISTEMAS_5, SISTEMAS_QTEXT)


def _slots_respondidos_por_media(df, excluir_cols, total_slots):
    cols = [c for c in df.columns if norm(str(c)) not in {norm(x) for x in excluir_cols}]
    if not cols or df.empty:
        return 0, total_slots

    ratios = []
    for _, row in df[cols].iterrows():
        non_empty = 0
        for v in row:
            if (pd is not None and pd.isna(v)):
                continue
            if str(v).strip() == "":
                continue
            non_empty += 1
        ratios.append(non_empty / len(cols))

    media = sum(ratios) / len(ratios) if ratios else 0
    answered = round(total_slots * media)
    return max(0, min(answered, total_slots)), total_slots


_HDR_HINTS = {
    "areas": [
        "nuvem", "cloud", "criptograf", "senha", "classifica", "compart",
        "termo de sigilo", "transfer", "terceir", "acl", "política de acesso"
    ],
    "processos": [
        "processo", "finalidade", "titular", "retenção", "descarte",
        "eliminação", "formato", "transferência", "dados pessoais",
        "dados sensíveis", "menor", "minimiza", "need-to-know"
    ],
    "sistemas": [
        "sistema", "fornecedor", "vendor", "contrato", "sla", "web",
        "browser", "saas", "datasul", "smile"
    ],
}

_SHEET_NAME_HINTS = {
    "areas": ["área", "departamento", "setor", "unidade", "gerência"],
    "processos": ["processo", "fluxo", "atividade", "operação"],
    "sistemas": ["sistema", "aplicação", "software", "plataforma", "tecnologia"],
}

_FIELD_SECTION_HINTS = {
    "cli.collection_method": ["areas", "processos"],
    "cli.who_access": ["areas", "processos"],
    "cli.nda": ["areas", "sistemas"],
    "cli.storage_location": ["areas"],
    "cli.name": ["sistemas"],
    "cli.contract_signed": ["sistemas"],
}


def _hint_section_from_sheet_name(sheet_name: str) -> Optional[str]:
    s = norm(str(sheet_name or ""))
    if not s:
        return None
    for sec, hints in _SHEET_NAME_HINTS.items():
        if any(h in s for h in hints):
            return sec
    return None


def _infer_section_from_field_tags(df) -> Optional[str]:
    if df is None or getattr(df, "empty", True):
        return None
    counts: Counter[str] = Counter()
    for col in getattr(df, "columns", []):
        fld = _extract_field_from_header(str(col))
        if not fld:
            continue
        for sec in _FIELD_SECTION_HINTS.get(fld, []):
            counts[sec] += 1
    if not counts:
        return None
    sec, _ = counts.most_common(1)[0]
    return sec

def _score_headers_to_section(headers: list[str]) -> str | None:
    hs = [norm(str(h)) for h in (headers or [])]
    best_sec, best_score = None, 0
    for sec, hints in _HDR_HINTS.items():
        score = sum(1 for h in hs for kw in hints if kw in h)
        if score > best_score:
            best_sec, best_score = sec, score
    return best_sec if best_score > 0 else None

def _merge_tables_vertically(dfs: list):
    try:
        import pandas as pd
        if not dfs:
            return None
        _normed = []
        for d in dfs:
            if d is None or d.empty:
                continue
            d = d.copy()
            d.columns = [str(c) for c in d.columns]
            _normed.append(d)
        if not _normed:
            return None
        return pd.concat(_normed, ignore_index=True, sort=False)
    except Exception:
        return None

def _distribute_xlsx_tables_into_sections(structured: dict) -> None:
    if not isinstance(structured, dict) or "__tables__" not in structured:
        return
    tables = structured.get("__tables__") or []
    by_sec = {"areas": [], "processos": [], "sistemas": []}
    for sheet_name, df in tables:
        try:
            section = _score_headers_to_section(list(df.columns))
            if not section:
                section = _infer_section_from_field_tags(df)
            if not section:
                section = _hint_section_from_sheet_name(sheet_name)
            if section:
                by_sec[section].append(df)
        except Exception:
            continue
    agg_areas     = _merge_tables_vertically(by_sec["areas"])
    agg_processos = _merge_tables_vertically(by_sec["processos"])
    agg_sistemas  = _merge_tables_vertically(by_sec["sistemas"])
    if agg_areas     is not None: structured["areas"]     = agg_areas
    if agg_processos is not None: structured["processos"] = agg_processos
    if agg_sistemas  is not None: structured["sistemas"]  = agg_sistemas


# ------------------------------------------
# EXTRAÃ‡ÃƒO de tabelas pipe do DOCX/PDF/TXT
# ------------------------------------------
def _try_extract_pipe_tables_to_dfs(text_original: str) -> List[Tuple[str, "pd.DataFrame"]]:
    if pd is None:
        return []
    lines = (text_original or "").splitlines()
    blocks = []
    cur = []
    for ln in lines:
        if ln.count("|") >= 2:
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)

    tables = []
    for i, blk in enumerate(blocks, 1):
        try:
            header = [c.strip() for c in blk[0].split("|")]
            header = [h for h in header if h != ""]
            rows = []
            for r in blk[1:]:
                cells = [c.strip() for c in r.split("|")]
                if len(cells) >= len(header):
                    rows.append(cells[:len(header)])
            if header and rows:
                df = pd.DataFrame(rows, columns=header)
                tables.append((f"doc_table_{i}", df))
        except Exception:
            continue
    return tables


# -----------------------------
# Helpers de normalizacao para XLSX
# -----------------------------
def _clean_excel_cell(val) -> str:
    if val is None:
        return ""
    try:
        if pd is not None and pd.isna(val):
            return ""
    except Exception:
        pass
    sval = str(val).strip()
    if not sval:
        return ""
    if sval.lower() in {"nan", "none", "null"}:
        return ""
    return sval


def _map_cells(df, fn):
    if df is None:
        return None
    try:
        mapper = getattr(df, "map", None)
        if callable(mapper):
            out = mapper(fn)
            if out is not None:
                return out
    except Exception:
        pass
    try:
        return df.apply(lambda col: col.map(fn) if hasattr(col, "map") else col)
    except Exception:
        return df


def _drop_empty_rows_cols(df):
    if df is None or getattr(df, "empty", True):
        return df
    mask_rows = ~(df.apply(lambda col: col.map(lambda v: _clean_excel_cell(v) == "")).all(axis=1))
    df = df.loc[mask_rows]
    mask_cols = ~(df.apply(lambda col: col.map(lambda v: _clean_excel_cell(v) == "")).all(axis=0))
    df = df.loc[:, mask_cols]
    return df.reset_index(drop=True)


def _dedupe_headers(headers: List[str]) -> List[str]:
    """
    Ensures column headers are non-empty and unique while preserving tag metadata.
    """
    seen: Dict[str, int] = {}
    result: List[str] = []
    for idx, raw in enumerate(headers or []):
        base = _clean_excel_cell(raw)
        if not base:
            base = f"col_{idx+1}"
        count = seen.get(base, 0)
        if count:
            deduped = f"{base}__{count+1}"
        else:
            deduped = base
        seen[base] = count + 1
        seen[deduped] = 1  # avoid collisions if it shows up again
        result.append(deduped)
    return result


def _infer_excel_header_row(df) -> int:
    """
    Picks the row that most likely carries the header / [field=...] tags.
    Combines density of tags, amount of text, and proximity to the top of the sheet.
    """
    if df is None or getattr(df, "empty", True):
        return 0
    try:
        total_rows = len(df.index)
        total_cols = max(1, len(df.columns))
    except Exception:
        return 0

    best_idx = 0
    best_score = float("-inf")
    max_scan = min(total_rows, 40)

    for ridx in range(max_scan):
        try:
            row_vals = [df.iloc[ridx, c] for c in range(total_cols)]
        except Exception:
            continue
        cleaned = [_clean_excel_cell(v) for v in row_vals]
        if not any(cleaned):
            continue

        non_empty = sum(1 for v in cleaned if v)
        field_tags = sum(1 for v in cleaned if FIELD_TAG_RE.search(v))
        rowkey_tags = sum(1 for v in cleaned if ROWKEY_TAG_RE.search(v))
        alphaish = sum(1 for v in cleaned if any(ch.isalpha() for ch in v))
        longish = sum(1 for v in cleaned if len(v) >= 4)
        coverage = non_empty / total_cols

        score = (
            field_tags * 12
            + rowkey_tags * 4
            + alphaish * 0.5
            + longish * 0.25
            + coverage * 3
            - ridx * 0.2
        )

        # linha com maioria absoluta de [field=...] vira head imediata
        if field_tags and field_tags >= max(2, int(non_empty * 0.6)):
            return ridx

        if score > best_score:
            best_score = score
            best_idx = ridx

    return best_idx if best_score > float("-inf") else 0


def _normalize_excel_dataframe(raw_df):
    if raw_df is None:
        return None
    try:
        df = raw_df.copy()
        df = _map_cells(df, _clean_excel_cell)
        df = _drop_empty_rows_cols(df)
        if df is None or df.empty:
            return df
        header_idx = _infer_excel_header_row(df)
        header_idx = max(0, min(header_idx, len(df) - 1))
        header_vals = [df.iloc[header_idx, c] for c in range(len(df.columns))]
        header = _dedupe_headers([str(v) for v in header_vals])
        data_start = header_idx + 1
        if data_start >= len(df):
            data = df.iloc[0:0].copy()
        else:
            data = df.iloc[data_start:].copy()
        data.columns = header
        data = _map_cells(data, _clean_excel_cell)
        return data.reset_index(drop=True)
    except Exception:
        return raw_df


# -----------------------------
# ExtraÃ§Ã£o de texto/estrutura
# -----------------------------
def extract_text_from_bytes(filename: str, content: bytes) -> Tuple[str, dict]:
    name = (filename or "").lower()
    text_original = ""
    structured: Dict[str, object] = {}

    if name.endswith(".txt"):
        text_original = content.decode("utf-8", errors="ignore")

    elif name.endswith(".csv"):
        try:
            decoded = content.decode("utf-8", errors="ignore").splitlines()
            reader = csv.reader(decoded)
            rows = list(reader)
            text_original = "\n".join([",".join(r) for r in rows])
            if pd is not None and rows:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                structured["__tables__"] = [("csv", df)]
        except Exception:
            text_original = content.decode("latin1", errors="ignore")

    elif name.endswith(".xlsx") and pd is not None:
        bio = io.BytesIO(content)
        try:
            xls = pd.ExcelFile(bio)
            structured["__tables__"] = []
            for sheet in sorted(xls.sheet_names, key=lambda s: str(s).lower()):
                try:
                    raw_df = xls.parse(sheet, header=None, dtype=str)
                except Exception:
                    continue
                df = _normalize_excel_dataframe(raw_df)
                target_df = df if df is not None else raw_df
                if target_df is None:
                    continue
                try:
                    text_original += target_df.fillna("").to_csv(index=False, lineterminator="\n") + "\n"
                except Exception:
                    pass
                structured["__tables__"].append((str(sheet), target_df))
            _distribute_xlsx_tables_into_sections(structured)
        except Exception:
            pass

    elif name.endswith(".docx") and Document is not None:
        bio = io.BytesIO(content)
        try:
            doc = Document(bio)
            chunks = []
            for p in doc.paragraphs:
                if p.text and p.text.strip():
                    chunks.append(p.text)
            for tb in getattr(doc, "tables", []):
                hdr = []
                if tb.rows:
                    hdr = [c.text.strip() for c in tb.rows[0].cells]
                    if any(hdr):
                        chunks.append(" | ".join(hdr))
                        for row in tb.rows[1:]:
                            cells = [c.text.strip() for c in row.cells]
                            chunks.append(" | ".join(cells))
            text_original = "\n".join(chunks)
        except Exception:
            text_original = content.decode("utf-8", errors="ignore")

    elif name.endswith(".pdf") and pdf_extract_text is not None:
        bio = io.BytesIO(content)
        try:
            text_original = pdf_extract_text(bio)
        except Exception:
            text_original = ""

    else:
        text_original = content.decode("utf-8", errors="ignore")

    try:
        pipe_tables = _try_extract_pipe_tables_to_dfs(text_original)
        if pipe_tables:
            structured.setdefault("__tables__", [])
            structured["__tables__"].extend(pipe_tables)
            _distribute_xlsx_tables_into_sections(structured)
    except Exception:
        pass

    text_norm = norm(text_original or "")
    structured["__raw_text__"] = text_norm
    structured["__raw_text_src__"] = text_original
    return text_norm, structured


def _answer_bit(row: Dict) -> int:
    if STRICT_MODE:
        return 1 if int(row.get("col", 0)) == 1 or int(row.get("llm", 0)) == 1 else 0
    else:
        return 1 if (row.get("keyword_hit") or (row.get("segments") and len(row["segments"]) > 0) or int(row.get("llm", 0)) == 1) else 0

def _compute_answered_from_per_question(perq: Dict[str, Dict], total_expected: int) -> dict:
    by_code = {}
    for code, row in perq.items():
        by_code[code] = _answer_bit(row)
    return {
        "answered": int(sum(by_code.values())),
        "total": int(total_expected),
        "by_code": by_code
    }


# ==========================
# SegmentaaÃ§Ã£o (fallback leve)
# ==========================
AREA_HEAD_PAT = re.compile(
    r'^\s*(?:Área|area|departamento|setor|ger(?:ência|encia)|unidade|diretoria)\s*[:\-\u2013\u2014]\s*(.+)$',
    re.I | re.M
)
PROCESS_HEAD_PAT = re.compile(
    r'^\s*(?:processo|process)\s*[:\-\u2013\u2014]\s*(.+)$',
    re.I | re.M
)

def _segment_text(raw_text: str) -> Dict[str, Dict[str, str]]:
    if not raw_text:
        return {"areas": {}, "processos": {}}
    lines = raw_text.splitlines()
    areas = {}
    processos = {}
    current_area = None
    current_area_buf = []
    for ln in lines:
        ma = AREA_HEAD_PAT.match(ln.strip())
        if ma:
            if current_area is not None:
                areas[current_area] = "\n".join(current_area_buf).strip()
            current_area = ma.group(1).strip()
            current_area_buf = []
        else:
            if current_area is not None:
                current_area_buf.append(ln)
    if current_area is not None:
        areas[current_area] = "\n".join(current_area_buf).strip()
    if not areas:
        areas = {"GERAL": raw_text}
    for area_name, area_text in areas.items():
        proc_lines = area_text.splitlines()
        cur_proc = None
        buf = []
        for ln in proc_lines:
            mp = PROCESS_HEAD_PAT.match(ln.strip())
            if mp:
                if cur_proc is not None:
                    processos[f"{area_name}::{cur_proc}"] = "\n".join(buf).strip()
                cur_proc = mp.group(1).strip()
                buf = []
            else:
                if cur_proc is not None:
                    buf.append(ln)
        if cur_proc is not None:
            processos[f"{area_name}::{cur_proc}"] = "\n".join(buf).strip()
    return {"areas": areas, "processos": processos}

def detect_organograma(text_norm: str, structured: dict) -> bool:
    kw = ["organograma","estrutura organizacional","estrutura organizativa","hierarquia","organizational chart","organization chart","organogram"]
    if any(k in text_norm for k in kw):
        return True
    if isinstance(structured, dict):
        if "organograma" in structured and structured["organograma"] is not None:
            return True
        df = structured.get("areas") or structured.get("Areas")
        if df is not None and hasattr(df, "columns"):
            cols_norm = [norm(str(c)) for c in df.columns]
            if any(c in cols_norm for c in ["superior","gestor","chefe","cargo","nivel","hierarquia","estrutura"]):
                return True
    return False


# ==== LLM helpers ====
def _clip_text_for_llm(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    if len(text) <= LLM_MAX_CHARS:
        return text
    keep = max(1000, LLM_MAX_CHARS // 2)
    head = text[:keep]
    tail = text[-keep:]
    logging.info("LLM text truncated from %s to %s chars (LLM_MAX_CHARS=%s)", len(text), len(head) + len(tail), LLM_MAX_CHARS)
    return head + "\n...\n" + tail


# ==============
# LLM auxiliar
# ==============
_original_extract = extract_text_from_bytes
def extract_text_from_bytes(filename: str, content: bytes) -> Tuple[str, dict]:
    text_norm, structured = _original_extract(filename, content)
    if isinstance(structured, dict):
        structured["__raw_text__"] = text_norm
    return text_norm, structured

def _init_llm_maps() -> dict:
    return {
        "areas":    {"map": {k: 0 for k in AREAS_QMAP.keys()},    "hits": {k: [] for k in AREAS_QMAP.keys()},    "total": 15, "answered": 0},
        "processos":{"map": {k: 0 for k in PROCESSOS_QMAP.keys()}, "hits": {k: [] for k in PROCESSOS_QMAP.keys()},"total": 26, "answered": 0},
        "sistemas": {"map": {k: 0 for k in SISTEMAS_QMAP.keys()},  "hits": {k: [] for k in SISTEMAS_QMAP.keys()}, "total": 5,  "answered": 0},
    }
LLM_REQUIRED_SNIPPET_KEYWORDS = {
    "areas": {
        "2_compart_dados_rede": [
            "compart",
            "rede",
            "intranet",
            "lan",
            "vpn",
            "fileserver",
            "network",
        ],
        "4_2_termo_sigilo_terceiros": [
            "sigilo",
            "confidenc",
            "nda",
            "acordo de confidencialidade",
            "non disclosure",
            "non-disclosure",
        ],
        "5_2_termo_sigilo_transfer": [
            "sigilo",
            "confidenc",
            "nda",
            "acordo de confidencialidade",
            "non disclosure",
            "non-disclosure",
        ],
        "4_1_forma_compart": [
            "compart",
            "forma",
            "via",
            "email",
            "software",
            "sistema",
            "portal",
            "api",
        ],
        "6_classificacao_documentos": [
            "classific",
            "document",
            "sigilo",
        ],
    },
    "processos": {
        "6_1_fisico_ou_digital": [
            "fisico",
            "impresso",
            "papel",
            "digital",
            "online",
            "portal",
            "eletron",
        ],
        "7_finalidade_legitima": [
            "finalidade",
            "legit",
            "objetivo",
            "interesse legitimo",
        ],
        "9_somente_quando_necessario": [
            "necess",
            "somente quando",
            "apenas quando",
            "need to know",
            "minimiz",
        ],
        "15_dados_precisos_claros_atualizados": [
            "atualiz",
            "precis",
            "qualidade",
            "integrid",
            "claro",
        ],
        "10_quem_usa_precisa": [
            "necess",
            "apenas",
            "autoriz",
            "gestor",
            "responsavel",
        ],
        "11_tempo_necessario_apenas": [
            "tempo",
            "prazo",
            "retenc",
            "anos",
            "duracao",
        ],
        "12_formato_dados_disponiveis": [
            "formato",
            "forma",
            "digital",
            "fisico",
            "portal",
            "arquivo",
        ],
        "17_medidas_seg_cibernetica": [
            "politica de seguranca",
            "seguranca",
            "risco",
            "firewall",
            "antivirus",
            "backup",
        ],
        "18_medidas_admin_protecao": [
            "controle de acesso",
            "politica",
            "procedimento",
            "responsabilidade",
            "admin",
        ],
    },
    "sistemas": {
        "2_fornecedor_identificado": [
            "fornecedor",
            "vendor",
            "provedor",
            "fabricante",
            "parceiro",
            "prestador",
        ],
        "3_eh_web": [
            "web",
            "browser",
            "online",
            "internet",
            "portal",
            "saas",
        ],
    }
}


def _enforce_llm_keyword_guards(agg: dict) -> None:
    for sec, reqs in LLM_REQUIRED_SNIPPET_KEYWORDS.items():
        sec_entry = agg.get(sec)
        if not isinstance(sec_entry, dict):
            continue
        sec_map = sec_entry.get("map") or {}
        sec_hits = sec_entry.get("hits") or {}
        for code, keywords in reqs.items():
            if int(sec_map.get(code, 0)) != 1:
                continue
            snippets = sec_hits.get(code) or []
            haystack = " ".join(norm(str(sn)) for sn in snippets)
            if not haystack:
                sec_map[code] = 0
                sec_hits.pop(code, None)
                continue
            if not any(kw in haystack for kw in keywords):
                sec_map[code] = 0
                sec_hits.pop(code, None)


QUESTION_CONTEXT_RULES = {
    "areas": {
        "2_compart_dados_rede": {
            "must_groups": [
                ["compart", "disponib", "acess", "repass", "envio"],
                ["rede", "intranet", "lan", "vpn", "fileserver", "servidor", "network", "fileshare", "pasta compart"],
            ],
            "neg_patterns": [
                "nao compart",
                "sem compartilhamento na rede",
                "inexistencia de compartilhamento na rede",
            ],
        },
        "4_1_forma_compart": {
            "must_groups": [
                ["compart", "disponib", "envio", "repasse", "transfer", "compartilhamento"],
                ["forma", "como", "meio", "metodo", "canal", "via", "realizado por", "realizado via"],
            ],
            "any_keywords": [
                "email",
                "formulario",
                "software",
                "sistema",
                "portal",
                "drive",
                "api",
                "arquivo",
                "planilha",
                "presencial",
                "telefone",
            ],
            "neg_patterns": [
                "nao ha forma",
                "sem forma definida",
                "nao compart",
            ],
        },
        "6_classificacao_documentos": {
            "must_groups": [
                ["classific", "categoria", "nivel", "etiqueta", "sigilo"],
                ["document", "informac", "registro", "dado"],
            ],
            "neg_patterns": [
                "nao possui class",
                "sem classificacao",
                "nao ha classificacao",
            ],
        },
    },
    "processos": {
        "9_somente_quando_necessario": {
            "must_groups": [
                ["necess", "somente quando", "apenas quando", "apenas quem", "need to know", "minimiz", "uso restrito"],
                ["uso", "trat", "acess", "process", "operac"],
            ],
            "neg_patterns": [
                "sem criterio de necessidade",
                "qualquer pessoa utiliza",
                "sem necessidade",
            ],
        },
        "6_1_fisico_ou_digital": {
            "must_groups": [
                ["fisico", "físico", "impresso", "papel", "manual"],
                ["digital", "eletron", "online", "portal", "sistema", "software", "formulario digital"],
            ],
            "neg_patterns": [
                "nao identificado",
                "nao informado",
            ],
        },
        "7_finalidade_legitima": {
            "must_groups": [
                ["finalidade", "legit", "proposit", "motivo", "justific"],
                ["trat", "process", "uso", "dados"],
            ],
            "any_keywords": [
                "interesse legitimo",
                "interesses legitimos",
                "legitimate interest",
                "controlador",
                "objetivo",
                "compliance",
                "lei",
                "obrigacao legal",
            ],
        },
        "10_quem_usa_precisa": {
            "must_groups": [
                ["quem", "usuarios", "setor", "responsavel", "gestor", "equipes"],
                ["necess", "precisa", "need to know", "autoriz", "habilit", "apenas"],
            ],
            "neg_patterns": [
                "qualquer pessoa utiliza",
                "todos acessam sem restricao",
            ],
        },
        "11_tempo_necessario_apenas": {
            "must_groups": [
                ["tempo", "prazo", "retenc", "duracao", "periodo", "manutenc"],
                ["necess", "limitado", "anos", "permanente", "prazo", "limite"],
            ],
        },
        "12_formato_dados_disponiveis": {
            "must_groups": [
                ["formato", "forma", "meio", "suporte", "disponivel"],
                ["digital", "fisico", "papel", "sistema", "planilha", "formulario", "arquivo", "pdf"],
            ],
        },
        "15_dados_precisos_claros_atualizados": {
            "must_groups": [
                ["dado", "informacao", "registro"],
                ["atualiz", "precis", "qualidade", "integrid", "claro", "confiavel", "complet"],
            ],
            "any_keywords": [
                "process",
                "controle",
                "rotina",
                "verific",
                "auditoria",
            ],
            "neg_patterns": [
                "dados desatualizados",
                "sem atualizacao",
                "nao estao atualizados",
            ],
        },
        "17_medidas_seg_cibernetica": {
            "must_groups": [
                ["medid", "politica", "procedimento", "controle", "plano", "programa"],
                ["seguranc", "cibernet", "risco", "proteca", "cyber", "firewall", "antivirus", "backup"],
            ],
            "any_keywords": [
                "gestao de riscos",
                "politica de seguranca",
                "controles tecnicos",
                "monitoramento",
            ],
        },
        "18_medidas_admin_protecao": {
            "must_groups": [
                ["medid", "politica", "procedimento", "controle", "manual"],
                ["admin", "acesso", "governanc", "responsavel", "papel", "cargo"],
            ],
            "any_keywords": [
                "controle de acesso",
                "politica de acesso",
                "responsabilidades",
                "processos administrativos",
            ],
        },
    },
    "sistemas": {
        "2_fornecedor_identificado": {
            "must_groups": [
                ["fornecedor", "vendor", "provedor", "fabricante", "parceiro", "prestador", "empresa responsavel", "responsavel"],
                ["sistema", "aplicacao", "software", "plataforma", "solucao", "ferramenta"],
            ],
            "neg_patterns": [
                "nao possui fornecedor",
                "sem fornecedor identificado",
            ],
        },
        "3_eh_web": {
            "must_groups": [
                ["sistema", "aplicacao", "software", "plataforma", "portal"],
                ["web", "browser", "online", "internet", "acesso remoto", "saas"],
            ],
        },
    },
}

_SENTENCE_SPLIT_RE = re.compile(r'[\n\r]+|(?<=[.!?])\s+')
_VENDOR_COL_TOKENS = ["fornecedor", "vendor", "provedor", "fabricante", "parceiro", "prestador"]
_SYSTEM_COL_TOKENS = ["sistema", "aplicacao", "software", "plataforma", "ferramenta", "solucao"]


def _iter_candidate_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _match_sentence_rule(sentence_norm: str, rule: dict) -> str | None:
    if not sentence_norm:
        return None
    for neg in rule.get("neg_patterns", []) or []:
        if neg and neg in sentence_norm:
            return "negative"
    for group in rule.get("must_groups", []) or []:
        if group and not any(token in sentence_norm for token in group):
            return None
    any_kw = rule.get("any_keywords") or []
    if any_kw and not any(token in sentence_norm for token in any_kw):
        return None
    return "positive"


def _collect_contextual_hits_from_text(text: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    results: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    if not text:
        return results
    for sentence in _iter_candidate_sentences(text):
        norm_sentence = norm(sentence)
        if not norm_sentence:
            continue
        for sec, rules in QUESTION_CONTEXT_RULES.items():
            for code, rule in rules.items():
                verdict = _match_sentence_rule(norm_sentence, rule)
                if not verdict:
                    continue
                sec_entry = results.setdefault(sec, {})
                code_entry = sec_entry.setdefault(code, {"positive": [], "negative": []})
                target = code_entry[verdict]
                snippet = sentence.strip()
                if snippet and snippet not in target and len(target) < 3:
                    target.append(snippet[:240])
    return results


def _merge_context_evidence(dst: Dict[str, Dict[str, Dict[str, List[str]]]], src: Dict[str, Dict[str, Dict[str, List[str]]]]) -> None:
    for sec, codes in (src or {}).items():
        sec_dst = dst.setdefault(sec, {})
        for code, buckets in codes.items():
            code_dst = sec_dst.setdefault(code, {"positive": [], "negative": []})
            for kind in ("positive", "negative"):
                for snippet in buckets.get(kind, []) or []:
                    if snippet not in code_dst[kind] and len(code_dst[kind]) < 3:
                        code_dst[kind].append(snippet)


def _extract_vendor_evidence(structured: dict) -> List[str]:
    if pd is None or not isinstance(structured, dict):
        return []
    df = structured.get("sistemas")
    if df is None or not hasattr(df, "columns") or getattr(df, "empty", True):
        return []
    try:
        cols = list(df.columns)
    except Exception:
        return []
    vendor_idx = []
    system_idx = []
    for idx, col in enumerate(cols):
        cname = norm(str(col))
        if any(tok in cname for tok in _VENDOR_COL_TOKENS):
            vendor_idx.append(idx)
        if any(tok in cname for tok in _SYSTEM_COL_TOKENS):
            system_idx.append(idx)
    if not vendor_idx:
        return []
    snippets: List[str] = []
    try:
        nrows = len(df.index)
    except Exception:
        nrows = 0
    for ridx in range(nrows):
        try:
            row = df.iloc[ridx]
        except Exception:
            continue
        for vidx in vendor_idx:
            try:
                raw_vendor = row.iloc[vidx]
            except Exception:
                continue
            if not _is_nonempty(raw_vendor):
                continue
            vendor_val = str(raw_vendor).strip()
            sistema_val = ""
            for sidx in system_idx:
                try:
                    sval = row.iloc[sidx]
                except Exception:
                    continue
                if _is_nonempty(sval):
                    sistema_val = str(sval).strip()
                    break
            snippet = f"{sistema_val} -> {vendor_val}" if sistema_val else f"Fornecedor identificado: {vendor_val}"
            if snippet not in snippets:
                snippets.append(snippet[:240])
            if len(snippets) >= 4:
                return snippets
    return snippets

def _structured_tables_as_text(structured: dict | None, row_limit: int = 400) -> Dict[str, str]:
    sections_text: Dict[str, str] = {"areas": "", "processos": "", "sistemas": ""}
    if pd is None or not isinstance(structured, dict):
        return sections_text
    for sec in sections_text.keys():
        df = structured.get(sec)
        if df is None or not hasattr(df, "columns") or getattr(df, "empty", True):
            continue
        lines: List[str] = []
        try:
            cols = [str(c) for c in df.columns]
            for ridx, row in enumerate(df.itertuples(index=False, name=None)):
                if ridx >= row_limit:
                    break
                parts = []
                for col_name, val in zip(cols, row):
                    if not _is_nonempty(val):
                        continue
                    sval = str(val).strip()
                    if not sval:
                        continue
                    parts.append(f"{col_name}: {sval}")
                if parts:
                    lines.append("; ".join(parts))
        except Exception:
            continue
        if lines:
            sections_text[sec] = "\n".join(lines)
    return sections_text


def _gather_additional_context_evidence(structured: dict | None) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    evidence: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    raw_text = ""
    if isinstance(structured, dict):
        raw_text = structured.get("__raw_text_src__") or structured.get("__raw_text__") or ""
    _merge_context_evidence(evidence, _collect_contextual_hits_from_text(raw_text))
    table_texts = _structured_tables_as_text(structured)
    for sec, text in table_texts.items():
        if not text:
            continue
        sec_hits = _collect_contextual_hits_from_text(text).get(sec)
        if sec_hits:
            _merge_context_evidence(evidence, {sec: sec_hits})
    vendor_hits = _extract_vendor_evidence(structured or {})
    if vendor_hits:
        vendor_map = {
            "sistemas": {
                "2_fornecedor_identificado": {"positive": vendor_hits, "negative": []}
            }
        }
        _merge_context_evidence(evidence, vendor_map)
    return evidence


def _apply_context_hits_to_maps(section_hits: dict | None, map_dst: Dict[str, int], hits_dst: Dict[str, List[str]]) -> None:
    if not section_hits:
        return
    for code, buckets in section_hits.items():
        positives = buckets.get("positive") or []
        negatives = buckets.get("negative") or []
        if positives and not negatives:
            map_dst[code] = 1
            if positives:
                hits_dst.setdefault(code, [])
                for snippet in positives:
                    if snippet not in hits_dst[code] and len(hits_dst[code]) < 2:
                        hits_dst[code].append(snippet[:240])
        elif negatives and not positives:
            map_dst[code] = 0
            hits_dst.pop(code, None)


def llm_evaluate_document(text_for_llm: str) -> dict:
    if not USE_LLM_EVAL or not text_for_llm or len(text_for_llm) < 40:
        return {}

    model = OPENAI_MODEL

    client = openai.OpenAI()

    def _eval_group_once(group_name: str, qmap: Dict[str, str], chunk: str) -> dict:
        sys_msg = (
            "Você é um avaliador LGPD. Dado um TRECHO do documento, para cada PERGUNTA do grupo {grp}, "
            "decida se o trecho permite RESPONDER explicitamente SIM ou NÃO (resposta conclusiva). "
            "Apenas marque 1 quando houver afirmação clara em frase ou tabela. "
            "Retorne JSON estrito: {\"map\": {\"<codigo>\": 0|1}, \"hits\": {\"<codigo>\":[\"trecho1\",\"trecho2\"]}}"
        ).replace("{grp}", group_name.upper())
        user_msg = "PERGUNTAS:\n" + json.dumps(qmap, ensure_ascii=False) + "\n\nTRECHO:\n" + chunk

        out = {"map": {}, "hits": {}}
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1400,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            if isinstance(data, dict):
                out["map"] = {k: int(1 if v in (1, True, "1", "true", "sim", "yes") else 0)
                              for k, v in (data.get("map", {}) or {}).items()}
                out["hits"] = data.get("hits", {}) or {}
        except Exception as e:
            logging.exception("LLM group '%s' falhou: %s", group_name, e)
        return out

    def _merge_intersection(dst: dict, src: dict):
        for code in dst["map"].keys():
            dst["map"][code] = 1 if (int(dst["map"][code]) == 1 and int(src["map"].get(code, 0)) == 1) else 0
        for code, arr in (src.get("hits") or {}).items():
            if not isinstance(arr, list):
                continue
            dst["hits"].setdefault(code, [])
            for snip in arr:
                if snip not in dst["hits"][code] and len(dst["hits"][code]) < 2:
                    dst["hits"][code].append(str(snip)[:240])

    def _merge_union(dst: dict, src: dict):
        for code in dst["map"].keys():
            if int(src["map"].get(code, 0)) == 1:
                dst["map"][code] = 1
        for code, arr in (src.get("hits") or {}).items():
            if not isinstance(arr, list):
                continue
            dst["hits"].setdefault(code, [])
            for snip in arr:
                if snip not in dst["hits"][code] and len(dst["hits"][code]) < 2:
                    dst["hits"][code].append(str(snip)[:240])

    def _normalize_group_result(res: dict, qmap: Dict[str, str]) -> dict:
        norm_map = {k: 0 for k in qmap.keys()}
        for code in norm_map.keys():
            try:
                norm_map[code] = 1 if int((res.get("map") or {}).get(code, 0)) == 1 else 0
            except Exception:
                norm_map[code] = 0
        hits = {}
        for code, arr in (res.get("hits") or {}).items():
            if not isinstance(arr, list):
                continue
            hits[code] = []
            for snip in arr:
                if len(hits[code]) >= 2:
                    break
                hits[code].append(str(snip)[:240])
        return {"map": norm_map, "hits": hits}

    def _reduce_pass_results(results: List[dict]) -> dict:
        if not results:
            return {"map": {}, "hits": {}}
        merged = results[0]
        for extra in results[1:]:
            _merge_intersection(merged, extra)
        return merged

    agg = {
        "areas":     {"map": {k: 0 for k in AREAS_QMAP.keys()},     "hits": {}, "total": 15, "answered": 0},
        "processos": {"map": {k: 0 for k in PROCESSOS_QMAP.keys()}, "hits": {}, "total": 26, "answered": 0},
        "sistemas":  {"map": {k: 0 for k in SISTEMAS_QMAP.keys()},  "hits": {}, "total": 5,  "answered": 0},
    }

    chunk_size = LLM_CHUNK_SIZE
    chunks = [text_for_llm[i:i+chunk_size] for i in range(0, len(text_for_llm), chunk_size)] or [""]
    passes = 2 if LLM_DOUBLE_PASS else 1

    for chunk_idx, ch in enumerate(chunks, start=1):
        pass_results = {"areas": [], "processos": [], "sistemas": []}
        for _ in range(passes):
            pass_results["areas"].append(_normalize_group_result(_eval_group_once("Áreas", AREAS_QMAP, ch), AREAS_QMAP))
            pass_results["processos"].append(_normalize_group_result(_eval_group_once("processos", PROCESSOS_QMAP, ch), PROCESSOS_QMAP))
            pass_results["sistemas"].append(_normalize_group_result(_eval_group_once("sistemas", SISTEMAS_QMAP, ch), SISTEMAS_QMAP))

        inter_a = _reduce_pass_results(pass_results["areas"])
        inter_p = _reduce_pass_results(pass_results["processos"])
        inter_s = _reduce_pass_results(pass_results["sistemas"])

        _merge_union(agg["areas"],     inter_a)
        _merge_union(agg["processos"], inter_p)
        _merge_union(agg["sistemas"],  inter_s)
        if all(agg[sec]["answered"] >= agg[sec]["total"] for sec in ["areas", "processos", "sistemas"]):
            logging.info("LLM early stop after chunk %s/%s (all sections answered)", chunk_idx, len(chunks))
            break

    for sec in ["areas", "processos", "sistemas"]:
        m = agg[sec]["map"]
        agg[sec]["answered"] = int(sum(1 for v in m.values() if v))

    return agg

# =========================================================
# ÃDICE DE TAGS: [field=...] e [rowkey=...]
# =========================================================
FIELD_TAG_RE  = re.compile(r"\[field\s*=\s*([A-Za-z0-9_.:-]+)\]", re.I)
ROWKEY_TAG_RE = re.compile(r"\[rowkey\s*=\s*([A-Za-z0-9_.:-]+)\]", re.I)

def _iter_tables(structured: dict):
    tables = (structured or {}).get("__tables__") or []
    for sheet, df in tables:
        yield sheet, df

def _extract_field_from_header(colname: str) -> Optional[str]:
    m = FIELD_TAG_RE.search(str(colname) or "")
    return m.group(1).strip() if m else None

def _find_rowkey_in_row(row_vals: List[str]) -> Optional[str]:
    for v in row_vals:
        if v is None: 
            continue
        m = ROWKEY_TAG_RE.search(str(v))
        if m:
            return m.group(1).strip()
    return None

def _is_nonempty(val) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"

def _parse_bool(val: str) -> Optional[bool]:
    if val is None:
        return None
    s = norm(str(val))
    # aceita sim/nao/yes/no/true/false/1/0
    if s in ("sim","yes","true","1","ok","habilitado","ativado","ativo"):
        return True
    if s in ("nao","não","no","false","0","desabilitado","inativo"):
        return False
    # frases curtas comuns
    if "protegido por senha" in s or "criptograf" in s:
        return True
    if "sem senha" in s or "sem criptografia" in s or "nao criptograf" in s or "não criptograf" in s:
        return False
    return None

class TagIndex:
    def __init__(self):
        self.rows_by_rowkey: Dict[str, List[Tuple[str, int, Dict[str, str]]]] = {}
        self.field_values: Dict[str, List[Tuple[str, int, str]]] = {}

def _build_tag_index(structured: dict) -> TagIndex:
    idx = TagIndex()
    for sheet, df in _iter_tables(structured):
        if df is None or getattr(df, "empty", True):
            continue
        cols = list(df.columns)
        col_fields: Dict[int, str] = {}
        for i, c in enumerate(cols):
            fld = _extract_field_from_header(str(c))
            if fld:
                col_fields[i] = fld

        if not col_fields:
            continue

        # percorre linhas por posiÃ§Ã£o (para termos nÃºmero de linha)
        try:
            nrows = len(df.index)
        except Exception:
            nrows = 0

        for ridx in range(nrows):
            try:
                row_series = df.iloc[ridx]
            except Exception:
                continue
            row_vals = [row_series.get(c, "") for c in cols]
            rowkey = _find_rowkey_in_row(row_vals)
            row_field_map: Dict[str, str] = {}
            for cidx, field_name in col_fields.items():
                try:
                    val = row_series.iloc[cidx]
                except Exception:
                    val = None
                if _is_nonempty(val):
                    sval = str(val)
                    row_field_map[field_name] = sval
                    idx.field_values.setdefault(field_name, []).append((sheet, ridx, sval))
            if rowkey and row_field_map:
                idx.rows_by_rowkey.setdefault(rowkey, []).append((sheet, ridx, row_field_map))
    return idx


# =========================================================
# REGRAS POR TAG (SEM KEYWORDS)
# =========================================================
def _evidence_rowkey_field(sheet: str, ridx: int, rowkey: str, field: str, value: str) -> str:
    return f"{sheet} | linha {ridx+2} | [rowkey={rowkey}] [field={field}]: {value}"

def _evidence_field_only(sheet: str, ridx: int, field: str, value: str) -> str:
    return f"{sheet} | linha {ridx+2} | [field={field}]: {value}"

def _rowkey_has_bool(idx: TagIndex, rowkey: str) -> Tuple[int, List[str]]:
    EV = []
    for (sheet, ridx, fmap) in idx.rows_by_rowkey.get(rowkey, []):
        if "has" in fmap:
            b = _parse_bool(fmap["has"])
            if b is True or b is False:
                EV.append(_evidence_rowkey_field(sheet, ridx, rowkey, "has", fmap["has"]))
                return 1, EV
    return 0, EV

def _field_any_nonempty(idx: TagIndex, field: str) -> Tuple[int, List[str]]:
    EV = []
    for sheet, ridx, val in idx.field_values.get(field, []):
        if _is_nonempty(val):
            EV.append(_evidence_field_only(sheet, ridx, field, val))
            if len(EV) >= 3:
                break
    return (1, EV) if EV else (0, EV)

# ---------- ÃƒÂREAS (mapeado por tags)
def _areas_rule_hd_tem_senha(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 1.1 Ã¢â‚¬â€ Medidas Administrativas: PolÃƒÂ­tica de senhas [rowkey=policy_passwords] + [field=has]
    return _rowkey_has_bool(idx, "policy_passwords")

def _areas_rule_rede_protecao_senha(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 2.1 Ã¢â‚¬â€ PolÃƒÂ­tica de controle de acesso [rowkey=policy_access_control] + [field=has]
    return _rowkey_has_bool(idx, "policy_access_control")

def _areas_rule_nuvem_criptografado(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 3.1 Ã¢â‚¬â€ PolÃƒÂ­tica de uso de controles criptogrÃƒÂ¡ficos [rowkey=policy_crypto] + [field=has]
    return _rowkey_has_bool(idx, "policy_crypto")

def _areas_rule_classificacao_docs(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 6 Ã¢â‚¬â€ PolÃƒÂ­tica de classificaÃƒÂ§ÃƒÂ£o da informaÃƒÂ§ÃƒÂ£o [rowkey=policy_classification] + [field=has]
    return _rowkey_has_bool(idx, "policy_classification")

def _areas_rule_forma_compart(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 4.1 Ã¢â‚¬â€ forma de compartilhamento: Clientes/Fornec. [field=cli.collection_method]
    return _field_any_nonempty(idx, "cli.collection_method")

def _areas_rule_compart_terceiros(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 4 Ã¢â‚¬â€ compartilhamento com terceiros: presenÃƒÂ§a de quem acessa [field=cli.who_access]
    return _field_any_nonempty(idx, "cli.who_access")

def _areas_rule_termo_sigilo_terceiros(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 4.2 Ã¢â‚¬â€ termo de sigilo com terceiros: [field=cli.nda] booleano
    EV = []
    for sheet, ridx, val in idx.field_values.get("cli.nda", []):
        b = _parse_bool(val)
        if b is True or b is False:
            EV.append(_evidence_field_only(sheet, ridx, "cli.nda", val))
            return 1, EV
    return 0, EV

def _areas_rule_hd_armazenado_onde(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 1.2 Ã¢â‚¬â€ onde fica armazenado: [field=cli.storage_location]
    return _field_any_nonempty(idx, "cli.storage_location")


# ---------- PROCESSOS (mapeado por tags essenciais)
def _proc_rule_3_trata_dados_sensiveis(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    info = _detect_sensitive_data_usage(structured)
    evidence = info.get("evidence") or []
    return (1 if evidence else 0), evidence[:2]
def _proc_rule_6_1_fisico_digital(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 6.1 Ã¢â‚¬â€ fÃƒÂ­sico/digital deduzÃƒÂ­vel pela presenÃƒÂ§a de [field=cli.collection_method]
    return _field_any_nonempty(idx, "cli.collection_method")

def _proc_rule_6_2_armazenado_nuvem(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 6.2 Ã¢â‚¬â€ nuvem: presenÃƒÂ§a de local/coleÃƒÂ§ÃƒÂ£o em [field=cli.collection_method] tambÃƒÂ©m responde
    return _field_any_nonempty(idx, "cli.collection_method")

def _proc_rule_6_3_compart_interno_terceiros(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 6.3 Ã¢â‚¬â€ compartilhamento interno/terceiros: [field=cli.who_access]
    return _field_any_nonempty(idx, "cli.who_access")

def _proc_rule_13_transfer(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 13 Ã¢â‚¬â€ forma de transferÃƒÂªncia: [field=cli.collection_method]
    return _field_any_nonempty(idx, "cli.collection_method")


# ---------- SISTEMAS (mapeado por tags disponÃƒÂ­veis)
def _sist_rule_1_nome(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 1 Ã¢â‚¬â€ nome do sistema: [field=cli.name]
    return _field_any_nonempty(idx, "cli.name")

def _sist_rule_4_contrato(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 4 Ã¢â‚¬â€ possui contrato: [field=cli.contract_signed] booleano
    EV = []
    for sheet, ridx, val in idx.field_values.get("cli.contract_signed", []):
        b = _parse_bool(val)
        if b is True or b is False:
            EV.append(_evidence_field_only(sheet, ridx, "cli.contract_signed", val))
            return 1, EV
    return 0, EV

def _sist_rule_4_1_contrato_apresentado(structured: dict, idx: TagIndex) -> Tuple[int, List[str]]:
    # 4.1 Ã¢â‚¬â€ contrato apresentado: [field=cli.nda] (outra evidÃƒÂªncia documental/assinado)
    EV = []
    for sheet, ridx, val in idx.field_values.get("cli.nda", []):
        b = _parse_bool(val)
        if b is True or b is False:
            EV.append(_evidence_field_only(sheet, ridx, "cli.nda", val))
            return 1, EV
    return 0, EV


# ---------- Tabela de regras (apenas tags, sem keywords)
COLUMN_RULES = {
    "areas": {
        "1_1_hd_tem_senha": _areas_rule_hd_tem_senha,
        "1_2_hd_armazenado_onde": _areas_rule_hd_armazenado_onde,
        "2_1_rede_protecao_senha": _areas_rule_rede_protecao_senha,
        "3_1_nuvem_criptografado": _areas_rule_nuvem_criptografado,
        "4_compart_terceiros": _areas_rule_compart_terceiros,
        "4_1_forma_compart": _areas_rule_forma_compart,
        "4_2_termo_sigilo_terceiros": _areas_rule_termo_sigilo_terceiros,
        "6_classificacao_documentos": _areas_rule_classificacao_docs,
    },
    "processos": {
        "3_trata_dados_sensiveis": _proc_rule_3_trata_dados_sensiveis,
        "6_1_fisico_ou_digital": _proc_rule_6_1_fisico_digital,
        "6_2_armazenado_nuvem": _proc_rule_6_2_armazenado_nuvem,
        "6_3_compart_interno_terceiros": _proc_rule_6_3_compart_interno_terceiros,
        "13_forma_transferencia_dados": _proc_rule_13_transfer,
    },
    "sistemas": {
        "1_nome_identificado": _sist_rule_1_nome,
        "4_possui_contrato": _sist_rule_4_contrato,
        "4_1_contrato_apresentado": _sist_rule_4_1_contrato_apresentado,
    }
}


def evaluate_by_columns(structured: dict) -> Dict[str, Dict]:
    """
    Retorna: {"areas":{"map":{code:int},"hits":{code:[evid...]}, "answered":int,"total":N}, ...}
    Base: SOMENTE tags [field]/[rowkey].
    """
    idx = _build_tag_index(structured)
    out = {
        "areas":     {"map": {}, "hits": {}, "answered": 0, "total": len(COLUMN_RULES.get("areas", {}))},
        "processos": {"map": {}, "hits": {}, "answered": 0, "total": len(COLUMN_RULES.get("processos", {}))},
        "sistemas":  {"map": {}, "hits": {}, "answered": 0, "total": len(COLUMN_RULES.get("sistemas", {}))},
    }
    for sec, rules in COLUMN_RULES.items():
        answered = 0
        for code, fn in rules.items():
            try:
                bit, ev = fn(structured, idx)
            except Exception as e:
                bit, ev = 0, [f"erro na regra: {e}"]
            out[sec]["map"][code] = int(bool(bit))
            if bit:
                answered += 1
            if ev:
                out[sec]["hits"][code] = ev
        out[sec]["answered"] = answered
    return out


# =========================================================
# MAIN
# =========================================================
def _build_per_question_report(
    qmap: Dict[str, str],
    keyword_hits_list: List[str],
    segment_by_code: Dict[str, List[str]],
    llm_map: Dict[str, int],
    llm_hits: Dict[str, List[str]],
    col_map: Optional[Dict[str, int]] = None,
    col_hits: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict]:
    rep: Dict[str, Dict] = {}
    kw_set = set(keyword_hits_list or [])
    col_map = col_map or {}
    col_hits = col_hits or {}
    for code, question_text in qmap.items():
        rep[code] = {
            "question": question_text,
            "keyword_hit": (code in kw_set),              # ficarÃƒÂ¡ False com STRICT_MODE
            "segments": sorted(segment_by_code.get(code, [])),
            "llm": int((llm_map or {}).get(code, 0)),
            "evidence": (llm_hits or {}).get(code, [])[:2],
            "col": int(col_map.get(code, 0)),
            "evidence_col": (col_hits or {}).get(code, [])[:2],
        }
    return rep

def _invert_segment_map_to_hits(by_segment: Dict[str, List[str]]) -> Dict[str, List[str]]:
    inv: Dict[str, List[str]] = {}
    for seg, codes in (by_segment or {}).items():
        for code in codes:
            inv.setdefault(code, []).append(seg)
    return inv

def main(req: func.HttpRequest) -> func.HttpResponse:
    if (req.method or "").upper() == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_cors_headers())

    ok, ip, retry_after = rate_limit_allow(req)
    if not ok:
        logging.warning("RATE_LIMIT: bloqueado ip=%s retry_after=%ss", ip, retry_after)
        return _json_response(
            {
                "error": "Limite de uso atingido",
                "message": "Você pode enviar no máximo 3 avaliações por IP a cada 24 horas.",
                "ip": ip,
                "retry_after_seconds": retry_after
            },
            status=429,
            extra_headers={"Retry-After": str(retry_after)}
        )

    # ---- A partir daqui, o fluxo normal fica dentro de um try global ----
    try:
        try:
            organograma, filename, file_bytes = parse_multipart_formdata(req)
        except ValueError as ve:
            logging.warning("FORM PARSE ERROR: %s", ve)
            return _json_response(
                {"error": "invalid_form_data", "message": str(ve)},
                status=400
            )
        except Exception as ex:
            logging.exception("FORM PARSE FAILURE")
            return _json_response(
                {"error": "invalid_form_data", "message": "Erro ao ler o formulário."},
                status=400
            )

        name_lower = (filename or "").lower()
        if name_lower.endswith(".doc"):
            return func.HttpResponse(
                json.dumps({
                    "error": "Formato .doc (Word antigo) nÃƒÂ£o suportado.",
                    "hint": "Envie .docx, .pdf ou .xlsx."
                }),
                status_code=415, mimetype="application/json"
            )

        text_norm, structured = extract_text_from_bytes(filename, file_bytes)
        logging.info("TEXT DEBUG: len=%s preview=%r", len(text_norm), text_norm[:240])
        logging.info("ENV DEBUG: py=%s has_pandas=%s", sys.executable, bool(pd))
        logging.info("STRUCT DEBUG: keys=%s", list(structured.keys()) if isinstance(structured, dict) else None)
        raw_text = structured.get("__raw_text__", "") if isinstance(structured, dict) else ""

        # 0) Verdade por TAG (coluna/aba)
        col_based = evaluate_by_columns(structured)
        sensitive_scan = _detect_sensitive_data_usage(structured if isinstance(structured, dict) else None)

        # Organograma automÃƒÂ¡tico (pode vir do parÃƒÂ¢metro tambÃƒÂ©m)
        if organograma is None:
            organograma = "sim" if detect_organograma(text_norm, structured) else "nao"

        # 1) HeurÃƒÂ­stica de preenchimento mÃƒÂ©dio por seÃƒÂ§ÃƒÂ£o (fallback)
        areas = processos = sistemas = None
        try:
            sc = { "areas": None, "processos": None, "sistemas": None }
            if "areas" in structured and pd is not None and isinstance(structured["areas"], pd.DataFrame):
                a_answ, a_tot = _slots_respondidos_por_media(structured["areas"], excluir_cols=["Area"], total_slots=15)
                sc["areas"] = {"answered": a_answ, "total": a_tot}
            if "processos" in structured and pd is not None and isinstance(structured["processos"], pd.DataFrame):
                p_answ, p_tot = _slots_respondidos_por_media(structured["processos"], excluir_cols=["Area","Processo"], total_slots=26)
                sc["processos"] = {"answered": p_answ, "total": p_tot}
            if "sistemas" in structured and pd is not None and isinstance(structured["sistemas"], pd.DataFrame):
                s_answ, s_tot = _slots_respondidos_por_media(structured["sistemas"], excluir_cols=["Area","Sistema"], total_slots=5)
                sc["sistemas"] = {"answered": s_answ, "total": s_tot}
            areas, processos, sistemas = sc["areas"], sc["processos"], sc["sistemas"]
        except Exception:
            pass

        # 1.1) SegmentaÃƒÂ§ÃƒÂ£o (somente para relatÃƒÂ³rio)
        segments = _segment_text(raw_text or text_norm)
        area_segments = segments.get("areas", {})
        process_segments = segments.get("processos", {})

        a_map = {} if not area_segments else {}
        p_map = {} if not process_segments else {}

        if areas is None:
            areas = {"answered": 0, "total": 15, "hits": [], "map": a_map}
        else:
            areas["map"] = a_map

        if processos is None:
            processos = {"answered": 0, "total": 26, "hits": [], "map": p_map}
        else:
            processos["map"] = p_map

        if sistemas is None:
            sistemas = {"answered": 0, "total": 5, "hits": []}

        # 2) IA semÃƒÂ¢ntica (opcional)
        llm_map_areas = {}
        llm_map_proc  = {}
        llm_map_sist  = {}
        llm_hits_areas = {}
        llm_hits_proc  = {}
        llm_hits_sist  = {}
        try:
            text_for_llm = structured.get("__raw_text_src__") if isinstance(structured, dict) else None
            text_for_llm = _clip_text_for_llm(text_for_llm or text_norm)
            if USE_LLM_EVAL and text_for_llm:
                llm = llm_evaluate_document(text_for_llm)
                llm_map_areas = llm.get("areas", {}).get("map", {}) or {}
                llm_map_proc  = llm.get("processos", {}).get("map", {}) or {}
                llm_map_sist  = llm.get("sistemas", {}).get("map", {}) or {}
                llm_hits_areas = llm.get("areas", {}).get("hits", {}) or {}
                llm_hits_proc  = llm.get("processos", {}).get("hits", {}) or {}
                llm_hits_sist  = llm.get("sistemas", {}).get("hits", {}) or {}
        except Exception:
            pass

        context_hits = _gather_additional_context_evidence(structured if isinstance(structured, dict) else None)
        _apply_context_hits_to_maps(context_hits.get("areas"), llm_map_areas, llm_hits_areas)
        _apply_context_hits_to_maps(context_hits.get("processos"), llm_map_proc, llm_hits_proc)
        _apply_context_hits_to_maps(context_hits.get("sistemas"), llm_map_sist, llm_hits_sist)

        if sensitive_scan.get("only_personal") and not (sensitive_scan.get("evidence") or []):
            if int(llm_map_proc.get("3_trata_dados_sensiveis", 0)) == 1:
                llm_map_proc["3_trata_dados_sensiveis"] = 0
                llm_hits_proc.pop("3_trata_dados_sensiveis", None)
        # 2.1) Merge da VERDADE por TAG (sobrepÃƒÂµe contagem base)
        def _merge_column_truth(base: dict, sec: str):
            col = col_based.get(sec) or {}
            base["answered"] = max(int(base.get("answered", 0)), int(col.get("answered", 0)))
            base["map_col"]  = col.get("map", {})
            base["hits_col"] = col.get("hits", {})

        _merge_column_truth(areas, "areas")
        _merge_column_truth(processos, "processos")
        _merge_column_truth(sistemas, "sistemas")

        # === RELATÃƒâ€œRIO pergunta-a-pergunta ===
        seg_by_code_areas = {}
        seg_by_code_proc  = {}
        seg_by_code_sist  = {}

        kw_hits_areas = set(areas.get("hits", []) if isinstance(areas, dict) else [])
        kw_hits_proc  = set(processos.get("hits", []) if isinstance(processos, dict) else [])
        kw_hits_sist  = set(sistemas.get("hits", []) if isinstance(sistemas, dict) else [])

        perq_areas = _build_per_question_report(
            AREAS_QMAP, list(kw_hits_areas), seg_by_code_areas,
            llm_map_areas, llm_hits_areas,
            col_map=(col_based.get("areas", {}) or {}).get("map", {}),
            col_hits=(col_based.get("areas", {}) or {}).get("hits", {})
        )

        perq_proc = _build_per_question_report(
            PROCESSOS_QMAP, list(kw_hits_proc), seg_by_code_proc,
            llm_map_proc, llm_hits_proc,
            col_map=(col_based.get("processos", {}) or {}).get("map", {}),
            col_hits=(col_based.get("processos", {}) or {}).get("hits", {})
        )

        perq_sist = _build_per_question_report(
            SISTEMAS_QMAP, list(kw_hits_sist), seg_by_code_sist,
            llm_map_sist, llm_hits_sist,
            col_map=(col_based.get("sistemas", {}) or {}).get("map", {}),
            col_hits=(col_based.get("sistemas", {}) or {}).get("hits", {})
        )

        areas_final     = _compute_answered_from_per_question(perq_areas, 15)
        processos_final = _compute_answered_from_per_question(perq_proc,  26)
        sistemas_final  = _compute_answered_from_per_question(perq_sist,   5)

        missing_areas = [code for code, v in areas_final["by_code"].items() if v == 0]
        missing_proc  = [code for code, v in processos_final["by_code"].items() if v == 0]
        missing_sist  = [code for code, v in sistemas_final["by_code"].items() if v == 0]

        logging.info("PERGUNTA-A-PERGUNTA AREAS: %s", json.dumps(perq_areas, ensure_ascii=False))
        logging.info("PERGUNTA-A-PERGUNTA PROCESSOS: %s", json.dumps(perq_proc, ensure_ascii=False))
        logging.info("PERGUNTA-A-PERGUNTA SISTEMAS: %s", json.dumps(perq_sist, ensure_ascii=False))
        logging.info("FALTANTES AREAS: %s", sorted(missing_areas))
        logging.info("FALTANTES PROCESSOS: %s", sorted(missing_proc))
        logging.info("FALTANTES SISTEMAS: %s", sorted(missing_sist))

        total_respondidas = (1 if organograma == "sim" else 0) + areas_final["answered"] + processos_final["answered"] + sistemas_final["answered"]
        total_perguntas  = 1 + areas_final["total"] + processos_final["total"] + sistemas_final["total"]
        percentual_total = round(100 * total_respondidas / total_perguntas) if total_perguntas else 0

        payload = {
            "organograma": {
                "resposta": organograma,
                "percentual": 100 if organograma == "sim" else 0,
                "respondidas": 1 if organograma == "sim" else 0,
                "total": 1
            },
            "areas": {
                "answered": areas_final["answered"],
                "total": areas_final["total"],
                "percentual": round(100 * areas_final["answered"] / areas_final["total"]) if areas_final["total"] else 0,
            },
            "processos": {
                "answered": processos_final["answered"],
                "total": processos_final["total"],
                "percentual": round(100 * processos_final["answered"] / processos_final["total"]) if processos_final["total"] else 0,
            },
            "sistemas": {
                "answered": sistemas_final["answered"],
                "total": sistemas_final["total"],
                "percentual": round(100 * sistemas_final["answered"] / sistemas_final["total"]) if sistemas_final["total"] else 0,
            },
            "total": {
                "respondidas": total_respondidas,
                "total": total_perguntas,
                "percentual": percentual_total
            },
            "debug": {
                "use_llm": bool(USE_LLM_EVAL),
                "len_text": len(text_norm),
                "per_question": {
                    "areas": perq_areas,
                    "processos": perq_proc,
                    "sistemas": perq_sist
                },
                "missing": {
                    "areas": sorted(missing_areas),
                    "processos": sorted(missing_proc),
                    "sistemas": sorted(missing_sist)
                }
            }
        }

        return func.HttpResponse(json.dumps(payload, ensure_ascii=False),
                                 status_code=200, mimetype="application/json")

    except ValueError as ve:
        return func.HttpResponse(json.dumps({"error": str(ve)}),
                                 status_code=400, mimetype="application/json")
    except Exception as ex:
        logging.exception("Erro geral na função")
        return func.HttpResponse(json.dumps({"error": str(ex)}),
                                 status_code=500, mimetype="application/json")
