import os
import json
import logging
import azure.functions as func
from src import UserThread
from src.defs import *

import AvaliacaoDataMapping as avaliacao_dm

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
ut = UserThread()

# -------------------- CORS --------------------
# Lista de origens permitidas (separa por vírgula)
# Ex: "https://homologa.privacypoint.com.br,https://privacypoint.com.br,http://localhost:3000"
# Suporta 2 formatos:
# 1) CSV: "https://a,https://b"
# 2) JSON array: ["https://a","https://b"]
def _parse_allowed_origins() -> set[str]:
    raw = (os.getenv("CORS_ALLOWED_ORIGINS") or "https://homologa.privacypoint.com.br").strip()
    if not raw:
        return set()
    if raw.lstrip().startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return {str(o).strip() for o in arr if str(o).strip()}
        except Exception:
            # cai para o fallback CSV
            pass
    return {o.strip() for o in raw.split(",") if o.strip()}

CORS_ALLOWED_ORIGINS = _parse_allowed_origins()


def _cors_headers(req: func.HttpRequest) -> dict:
    origin = req.headers.get("origin")
    # se o origin estiver na whitelist, retorna ele; senão retorna o primeiro permitido (ou "*")
    allow_origin = origin if origin in CORS_ALLOWED_ORIGINS else (next(iter(CORS_ALLOWED_ORIGINS), "*"))

    return {
        "Access-Control-Allow-Origin": allow_origin,
        "Vary": "Origin",
        "Access-Control-Allow-Methods": "POST, GET, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400",
    }


def _apply_cors(req: func.HttpRequest, resp: func.HttpResponse) -> func.HttpResponse:
    """Garante que qualquer HttpResponse (inclusive vindo de outro módulo) saia com CORS."""
    for k, v in _cors_headers(req).items():
        resp.headers[k] = v
    return resp


def _read_json_body(req: func.HttpRequest) -> dict:
    """Parseia JSON do body de forma mais tolerante (inclusive BOM UTF-8).
    Levanta Exception se não conseguir parsear.
    """
    raw = req.get_body() or b""
    if not raw:
        raise ValueError("Empty request body")

    # utf-8-sig remove BOM se vier de arquivos gerados no Windows (Out-File etc.)
    try:
        text = raw.decode("utf-8-sig")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    # Alguns clientes podem mandar whitespace antes do JSON
    text = text.strip()

    if not text:
        raise ValueError("Empty request body")

    return json.loads(text)


# -------------------- CHAT --------------------
@app.route(
    route="ask",
    methods=[func.HttpMethod.POST, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
def ask(req: func.HttpRequest) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    try:
        body = _read_json_body(req)
    except Exception as e:
        # Loga diagnóstico no App Insights / logs do host
        try:
            raw = req.get_body() or b""
            preview = raw[:200]
        except Exception:
            preview = b""
        logging.exception("Falha ao parsear JSON. content-type=%s body_preview=%r", req.headers.get('content-type'), preview)
        return JsonErrorResponse("Invalid JSON body.", 400, headers=_cors_headers(req))

    message = body.get("message", None)
    thread_id = body.get("thread-id", None)

    if message is None:
        return JsonErrorResponse("Message is required.", 400, headers=_cors_headers(req))

    try:
        thread_id, answer = ut.ask(message, thread_id)
        return JsonResponse(
            {
                "thread-id": thread_id,
                "answer": answer,
            },
            200,
            headers=_cors_headers(req),
        )
    except StatusCodeError as e:
        return JsonErrorResponse(e.message, e.status_code, headers=_cors_headers(req))
    except Exception as e:
        return JsonErrorResponse(e, headers=_cors_headers(req))


@app.route(
    route="thread",
    methods=[func.HttpMethod.GET, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
def get_thread(req: func.HttpRequest) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    thread_id = req.params.get("thread-id", None)

    if thread_id is None:
        return JsonErrorResponse("Thread ID is required", 400, headers=_cors_headers(req))

    asc = req.params.get("order", "desc").lower() == "asc"

    try:
        messages = ut.list_messages(thread_id, asc=asc)
        return JsonResponse(
            {
                "thread-id": thread_id,
                "messages": messages,
            },
            200,
            headers=_cors_headers(req),
        )
    except StatusCodeError as e:
        return JsonErrorResponse(e.message, e.status_code, headers=_cors_headers(req))
    except Exception as e:
        return JsonErrorResponse(e, headers=_cors_headers(req))


@app.route(
    route="thread",
    methods=[func.HttpMethod.DELETE, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
def clear_thread(req: func.HttpRequest) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    thread_id = req.params.get("thread-id", None) or req.form.get("thread-id", None)

    if thread_id is None:
        return JsonErrorResponse("Thread ID is required", 400, headers=_cors_headers(req))

    try:
        ut.delete_thread(thread_id)
        return EmptyResponse(status_code=200, headers=_cors_headers(req))
    except StatusCodeError as e:
        return JsonErrorResponse(e.message, e.status_code, headers=_cors_headers(req))
    except Exception as e:
        return JsonErrorResponse(e, headers=_cors_headers(req))


# ----------------- AVALIAÇÃO DATA MAPPING -----------------
# Deixo esta rota ANONYMOUS para o front consumir sem chave.
@app.route(
    route="avaliar-data-mapping",
    methods=[func.HttpMethod.POST, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
def avaliar_data_mapping(req: func.HttpRequest) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    # delega para a função main do package AvaliacaoDataMapping
    resp = avaliacao_dm.main(req)

    # garante CORS mesmo que o módulo retorne HttpResponse "cru"
    if isinstance(resp, func.HttpResponse):
        return _apply_cors(req, resp)

    # fallback (se retornar algo inesperado)
    return JsonErrorResponse("Unexpected response from AvaliacaoDataMapping.", 500, headers=_cors_headers(req))
