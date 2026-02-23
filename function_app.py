import os
import json
import logging
import uuid
import inspect
import re
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from src import UserThread
from src.defs import *

import AvaliacaoDataMapping as avaliacao_dm

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)
ut = UserThread()

# -------------------- CORS --------------------
# Lista de origens permitidas (separa por vírgula)
# Ex: "https://homologa.privacypoint.com.br,https://privacypoint.com.br,http://localhost:3000"
# Suporta 2 formatos:
# 1) CSV: "https://a,https://b"
# 2) JSON array: ["https://a","https://b"]
def _parse_allowed_origins() -> set[str]:
    raw = (os.getenv("CORS_ALLOWED_ORIGINS") or os.getenv("CORS_ALLOW_ORIGIN") or "https://homologa.privacypoint.com.br").strip()
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
    origin = req.headers.get("Origin") or req.headers.get("origin")
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


# -------------------- Durable/Storage helpers --------------------
DM_INPUT_CONTAINER = os.getenv("DM_INPUT_CONTAINER", "dm-inputs")
DM_RESULT_CONTAINER = os.getenv("DM_RESULT_CONTAINER", "dm-results")


def _resolve_storage_connection_string() -> str:
    conn = (os.getenv("AzureWebJobsStorage") or "").strip()
    if not conn:
        raise ValueError("AzureWebJobsStorage is not configured.")

    # Suporte explícito ao shortcut do Azurite em ambiente local
    if conn.lower() == "usedevelopmentstorage=true":
        return (
            "DefaultEndpointsProtocol=http;"
            "AccountName=devstoreaccount1;"
            "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
            "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
            "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
            "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
        )

    return conn


def _get_blob_service_client() -> BlobServiceClient:
    conn = _resolve_storage_connection_string()
    return BlobServiceClient.from_connection_string(conn)




def _ensure_container(name: str):
    client = _get_blob_service_client().get_container_client(name)
    try:
        client.create_container()
    except ResourceExistsError:
        pass
    return client


def _safe_filename(name: str | None) -> str:
    base = os.path.basename(name or "arquivo")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return (safe[:120] or "arquivo")


def _upload_json(container_name: str, blob_name: str, payload: dict) -> None:
    container = _ensure_container(container_name)
    data = json.dumps(payload, ensure_ascii=False)
    container.upload_blob(name=blob_name, data=data, overwrite=True, content_type="application/json")


def _download_json(container_name: str, blob_name: str) -> dict | None:
    container = _ensure_container(container_name)
    blob = container.get_blob_client(blob_name)
    try:
        data = blob.download_blob().readall()
    except ResourceNotFoundError:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _map_runtime_status(runtime: str) -> str:
    s = (runtime or "").lower()
    if s in ("pending", "queued"):
        return "queued"
    if s in ("running", "continuedasnew"):
        return "processing"
    if s == "completed":
        return "done"
    if s in ("failed", "terminated", "canceled", "cancelled"):
        return "error"
    return "processing"


def _build_status_payload(instance_id: str, runtime_status: str, custom_status: dict | None = None) -> dict:
    payload = {
        "jobId": instance_id,
        "status": _map_runtime_status(runtime_status),
        "runtimeStatus": runtime_status,
    }
    if isinstance(custom_status, dict):
        if custom_status.get("step"):
            payload["step"] = custom_status.get("step")
        if custom_status.get("progress") is not None:
            payload["progress"] = custom_status.get("progress")
        if custom_status.get("message"):
            payload["message"] = custom_status.get("message")
    return payload


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
    methods=[func.HttpMethod.GET, func.HttpMethod.DELETE, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
def thread(req: func.HttpRequest) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    if req.method == "GET":
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

    if req.method == "DELETE":
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

    return JsonErrorResponse("Method not allowed", 405, headers=_cors_headers(req))


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


# ----------------- AVALIACAO DATA MAPPING (ASYNC JOB) -----------------
@app.route(
    route="avaliar-data-mapping/start",
    methods=[func.HttpMethod.POST, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
@app.durable_client_input(client_name="client")
async def avaliar_data_mapping_start(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    ok, ip, retry_after = avaliacao_dm.rate_limit_allow(req)
    if not ok:
        return JsonResponse(
            {
                "error": "Limite de uso atingido",
                "message": "Voce pode enviar no maximo 3 avaliacoes a cada 24 horas.",
                "ip": ip,
                "retry_after_seconds": retry_after,
            },
            status_code=429,
            headers=_cors_headers(req),
        )

    try:
        organograma, filename, file_bytes = avaliacao_dm.parse_multipart_formdata(req)
    except ValueError as ve:
        logging.warning("FORM PARSE ERROR: %s", ve)
        return JsonResponse(
            {"error": "invalid_form_data", "message": str(ve)},
            status_code=400,
            headers=_cors_headers(req),
        )
    except Exception:
        logging.exception("FORM PARSE FAILURE")
        return JsonResponse(
            {"error": "invalid_form_data", "message": "Erro ao ler o formulario."},
            status_code=400,
            headers=_cors_headers(req),
        )

    name_lower = (filename or "").lower()
    if name_lower.endswith(".doc"):
        return JsonResponse(
            {
                "error": "Formato .doc (Word antigo) nao suportado.",
                "hint": "Envie .docx, .pdf ou .xlsx.",
            },
            status_code=415,
            headers=_cors_headers(req),
        )

    job_id = str(uuid.uuid4())
    safe_name = _safe_filename(filename)
    input_blob = f"{job_id}/{safe_name}"

    try:
        container = _ensure_container(DM_INPUT_CONTAINER)
        container.upload_blob(name=input_blob, data=file_bytes, overwrite=True)
    except Exception as ex:
        logging.exception("Erro ao salvar arquivo no blob")
        return JsonErrorResponse("Falha ao salvar arquivo.", 500, headers=_cors_headers(req))

    payload = {
        "jobId": job_id,
        "input_blob": input_blob,
        "filename": filename,
        "organograma": organograma,
    }

    instance_id = await _maybe_await(client.start_new("avaliar_data_mapping_orchestrator", job_id, payload))
    job_id = instance_id or job_id

    base_url = (req.url.split("?")[0]).rsplit("/", 1)[0]
    return JsonResponse(
        {
            "jobId": job_id,
            "statusUrl": f"{base_url}/status?jobId={job_id}",
            "resultUrl": f"{base_url}/result?jobId={job_id}",
        },
        status_code=202,
        headers=_cors_headers(req),
    )


@app.route(
    route="avaliar-data-mapping/status",
    methods=[func.HttpMethod.GET, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
@app.durable_client_input(client_name="client")
async def avaliar_data_mapping_status(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    job_id = req.params.get("jobId", None)
    if not job_id:
        return JsonErrorResponse("jobId is required", 400, headers=_cors_headers(req))

    status = await _maybe_await(client.get_status(job_id))
    if not status:
        return JsonErrorResponse("Job not found", 404, headers=_cors_headers(req))

    status_json = status.to_json() if hasattr(status, "to_json") else {}
    runtime_status = status_json.get("runtimeStatus") or str(getattr(status, "runtime_status", ""))
    custom_status = status_json.get("customStatus") or getattr(status, "custom_status", None) or {}

    payload = _build_status_payload(job_id, runtime_status, custom_status)
    return JsonResponse(payload, status_code=200, headers=_cors_headers(req))


@app.route(
    route="avaliar-data-mapping/result",
    methods=[func.HttpMethod.GET, func.HttpMethod.OPTIONS],
    auth_level=func.AuthLevel.ANONYMOUS,
)
@app.durable_client_input(client_name="client")
async def avaliar_data_mapping_result(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    # Preflight (CORS)
    if req.method == "OPTIONS":
        return EmptyResponse(status_code=204, headers=_cors_headers(req))

    job_id = req.params.get("jobId", None)
    if not job_id:
        return JsonErrorResponse("jobId is required", 400, headers=_cors_headers(req))

    status = await _maybe_await(client.get_status(job_id))
    if not status:
        return JsonErrorResponse("Job not found", 404, headers=_cors_headers(req))

    status_json = status.to_json() if hasattr(status, "to_json") else {}
    runtime_status = status_json.get("runtimeStatus") or str(getattr(status, "runtime_status", ""))
    custom_status = status_json.get("customStatus") or getattr(status, "custom_status", None) or {}
    mapped = _map_runtime_status(runtime_status)

    if mapped != "done":
        if mapped == "error":
            err = _download_json(DM_RESULT_CONTAINER, f"{job_id}/error.json") or {"error": "Job failed"}
            return JsonResponse(err, status_code=500, headers=_cors_headers(req))
        payload = _build_status_payload(job_id, runtime_status, custom_status)
        return JsonResponse(payload, status_code=202, headers=_cors_headers(req))

    result = _download_json(DM_RESULT_CONTAINER, f"{job_id}/result.json")
    if result is None:
        return JsonErrorResponse("Resultado ainda nao disponivel.", 404, headers=_cors_headers(req))

    return JsonResponse(result, status_code=200, headers=_cors_headers(req))


@app.orchestration_trigger(context_name="context")
def avaliar_data_mapping_orchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    context.set_custom_status({"state": "processing", "step": "avaliacao"})
    try:
        result = yield context.call_activity("avaliar_data_mapping_activity", payload)
    except Exception as ex:
        context.set_custom_status({"state": "error", "message": str(ex)})
        raise
    context.set_custom_status({"state": "done"})
    return result


@app.activity_trigger(input_name="payload")
def avaliar_data_mapping_activity(payload: dict) -> dict:
    job_id = (payload or {}).get("jobId") or "job"
    input_blob = (payload or {}).get("input_blob")
    filename = (payload or {}).get("filename") or "arquivo"
    organograma = (payload or {}).get("organograma")

    if not input_blob:
        raise ValueError("input_blob ausente")

    try:
        container = _ensure_container(DM_INPUT_CONTAINER)
        blob = container.get_blob_client(input_blob)
        file_bytes = blob.download_blob().readall()

        if (filename or "").lower().endswith(".doc"):
            raise ValueError("Formato .doc (Word antigo) nao suportado. Envie .docx, .pdf ou .xlsx.")

        result_payload = avaliacao_dm.evaluate_data_mapping_payload(organograma, filename, file_bytes)
        result_blob = f"{job_id}/result.json"
        _upload_json(DM_RESULT_CONTAINER, result_blob, result_payload)
        return {"jobId": job_id, "result_blob": result_blob}
    except Exception as ex:
        _upload_json(DM_RESULT_CONTAINER, f"{job_id}/error.json", {"error": str(ex)})
        raise
