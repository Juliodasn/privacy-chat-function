import os
import azure.functions as func
from src import UserThread
from src.defs import *

import AvaliacaoDataMapping as avaliacao_dm

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
ut = UserThread()

# -------------------- CORS --------------------
# Lista de origens permitidas (separa por vírgula)
# Ex: "https://homologa.privacypoint.com.br,https://privacypoint.com.br,http://localhost:3000"
CORS_ALLOWED_ORIGINS = {
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", "https://homologa.privacypoint.com.br").split(",")
    if o.strip()
}


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
        body = req.get_json()
    except Exception:
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
