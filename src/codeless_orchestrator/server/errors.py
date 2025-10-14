from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from ..engine.errors import GraphCompileError, OrchestrationUnsupportedError, ToolExecutionError
from ..providers.exceptions import ProviderError


def to_http_error(err: Exception) -> HTTPException:
    """Map internal exceptions to HTTPException with appropriate status codes."""
    if isinstance(err, (ValueError,)):  # Validation/parse error
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err))
    if isinstance(err, (GraphCompileError, OrchestrationUnsupportedError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    if isinstance(err, (ToolExecutionError, ProviderError)):
        # Upstream/tooling failure
        code = status.HTTP_502_BAD_GATEWAY
        detail: Any = {"message": str(err)}
        if isinstance(err, ProviderError):
            if err.status_code is not None:
                detail = {"message": err.message, "status_code": err.status_code, "code": err.code}
        return HTTPException(status_code=code, detail=detail)
    # Fallback
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))

