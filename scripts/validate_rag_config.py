import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _load_local_settings(path: Path) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    values = data.get("Values", {}) if isinstance(data, dict) else {}
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if os.getenv(key) is None:
            os.environ[key] = str(value)


def _detect_provider() -> str:
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    azure_key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
    if endpoint and azure_key:
        return "azure_openai"
    if (os.getenv("OPENAI_API_KEY") or "").strip():
        return "openai"
    return "none"


def _status_error_info(ex: Exception) -> Tuple[int | None, str]:
    status_code = getattr(ex, "status_code", None)
    msg = str(ex)
    return status_code if isinstance(status_code, int) else None, msg


def _check_chat_azure(report: Dict[str, Any]) -> bool:
    try:
        from openai import AzureOpenAI
    except Exception as ex:
        report["chat_check"] = {"ok": False, "error": f"openai package not available: {ex}"}
        return False

    model = (os.getenv("RAG_EVAL_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or "").strip()
    client = AzureOpenAI(
        api_key=(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip(),
        azure_endpoint=(os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip(),
        api_version=(os.getenv("AZURE_OPENAI_API_VERSION") or "2024-06-01").strip(),
    )
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "responda apenas OK"}],
            max_tokens=5,
            temperature=0,
        )
        report["chat_check"] = {"ok": True, "model": model}
        return True
    except Exception as ex:
        status_code, msg = _status_error_info(ex)
        report["chat_check"] = {"ok": False, "model": model, "status_code": status_code, "error": msg}
        return False


def _check_embed_azure(report: Dict[str, Any]) -> bool:
    try:
        from openai import AzureOpenAI
    except Exception as ex:
        report["embedding_check"] = {"ok": False, "error": f"openai package not available: {ex}"}
        return False

    model = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or os.getenv("RAG_EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or ""
    ).strip()
    client = AzureOpenAI(
        api_key=(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip(),
        azure_endpoint=(os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip(),
        api_version=(os.getenv("AZURE_OPENAI_API_VERSION") or "2024-06-01").strip(),
    )
    try:
        client.embeddings.create(model=model, input=["healthcheck"])
        report["embedding_check"] = {"ok": True, "model": model}
        return True
    except Exception as ex:
        status_code, msg = _status_error_info(ex)
        report["embedding_check"] = {"ok": False, "model": model, "status_code": status_code, "error": msg}
        return False


def _check_chat_openai(report: Dict[str, Any]) -> bool:
    try:
        from openai import OpenAI
    except Exception as ex:
        report["chat_check"] = {"ok": False, "error": f"openai package not available: {ex}"}
        return False

    model = (os.getenv("RAG_EVAL_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "responda apenas OK"}],
            max_tokens=5,
            temperature=0,
        )
        report["chat_check"] = {"ok": True, "model": model}
        return True
    except Exception as ex:
        status_code, msg = _status_error_info(ex)
        report["chat_check"] = {"ok": False, "model": model, "status_code": status_code, "error": msg}
        return False


def _check_embed_openai(report: Dict[str, Any]) -> bool:
    try:
        from openai import OpenAI
    except Exception as ex:
        report["embedding_check"] = {"ok": False, "error": f"openai package not available: {ex}"}
        return False

    model = (
        os.getenv("OPENAI_EMBEDDING_MODEL")
        or os.getenv("RAG_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    ).strip()
    client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())
    try:
        client.embeddings.create(model=model, input=["healthcheck"])
        report["embedding_check"] = {"ok": True, "model": model}
        return True
    except Exception as ex:
        status_code, msg = _status_error_info(ex)
        report["embedding_check"] = {"ok": False, "model": model, "status_code": status_code, "error": msg}
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RAG env configuration and deployment connectivity.")
    parser.add_argument(
        "--local-settings",
        default="Backend/local.settings.json",
        help="Path to local.settings.json to preload env (default: Backend/local.settings.json)",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Validate config shape only (do not call chat/embeddings endpoints).",
    )
    parser.add_argument(
        "--allow-missing-credentials",
        action="store_true",
        help="Do not fail when credentials are missing (useful for template validation).",
    )
    args = parser.parse_args()

    _load_local_settings(Path(args.local_settings))

    report: Dict[str, Any] = {
        "rag_flags": {
            "RAG_ENABLED": os.getenv("RAG_ENABLED"),
            "RAG_ROLLOUT_STAGE": os.getenv("RAG_ROLLOUT_STAGE"),
            "RAG_TOP_K": os.getenv("RAG_TOP_K"),
            "RAG_EVAL_MODEL": os.getenv("RAG_EVAL_MODEL"),
            "RAG_FALLBACK_TO_OLD_LLM": os.getenv("RAG_FALLBACK_TO_OLD_LLM"),
            "RAG_ONLY_FILL_MISSING": os.getenv("RAG_ONLY_FILL_MISSING"),
        },
        "provider": _detect_provider(),
        "config_ok": True,
        "errors": [],
        "warnings": [],
    }

    if (os.getenv("RAG_ENABLED") or "").lower() != "true":
        report["warnings"].append("RAG_ENABLED is not true")

    provider = report["provider"]
    if provider == "none":
        msg = "No provider credentials configured (Azure OpenAI or OpenAI)"
        if args.allow_missing_credentials:
            report["warnings"].append(msg)
        else:
            report["errors"].append(msg)

    if provider == "azure_openai":
        required = {
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("RAG_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL"),
            "RAG_EVAL_MODEL": os.getenv("RAG_EVAL_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        }
        missing = [k for k, v in required.items() if not (v or "").strip()]
        if missing:
            report["errors"].append(f"Missing Azure settings: {missing}")

    if provider == "openai":
        if not (os.getenv("OPENAI_API_KEY") or "").strip():
            report["errors"].append("OPENAI_API_KEY is missing")
        if not (os.getenv("RAG_EVAL_MODEL") or os.getenv("OPENAI_MODEL") or "").strip():
            report["errors"].append("RAG_EVAL_MODEL/OPENAI_MODEL is missing")

    if report["errors"]:
        report["config_ok"] = False

    connectivity_ok = True
    if not args.skip_connectivity and report["config_ok"]:
        if provider == "azure_openai":
            chat_ok = _check_chat_azure(report)
            emb_ok = _check_embed_azure(report)
            connectivity_ok = bool(chat_ok and emb_ok)
        elif provider == "openai":
            chat_ok = _check_chat_openai(report)
            emb_ok = _check_embed_openai(report)
            connectivity_ok = bool(chat_ok and emb_ok)

    report["connectivity_ok"] = connectivity_ok if not args.skip_connectivity else None
    if not args.skip_connectivity and not connectivity_ok:
        report["errors"].append("Connectivity check failed (verify 401/403/404 details above)")
        report["config_ok"] = False

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["config_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
