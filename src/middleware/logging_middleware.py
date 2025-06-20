import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time = time.perf_counter() - start_time
        handler = request.url.path
        method = request.method
        status_code = response.status_code

        logger.info(f"{method} {handler} [{status_code}] took {process_time:.4f} seconds")

        return response
