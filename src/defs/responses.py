import azure.functions as func
import json


class EmptyResponse(func.HttpResponse):
    def __init__(self, status_code: int = 200, headers: dict | None = None):
        super().__init__(status_code=status_code, headers=headers)


class JsonResponse(func.HttpResponse):
    def __init__(self, data: dict, status_code: int = 200, headers: dict | None = None):
        super().__init__(
            body=json.dumps(data),
            status_code=status_code,
            mimetype="application/json",
            headers=headers,
        )


class JsonErrorResponse(JsonResponse):
    def __init__(self, message: str | Exception, status_code: int = 500, headers: dict | None = None):
        if isinstance(message, Exception):
            message = str(message)
        super().__init__(data={"error": message}, status_code=status_code, headers=headers)


__all__ = ["EmptyResponse", "JsonResponse", "JsonErrorResponse"]
