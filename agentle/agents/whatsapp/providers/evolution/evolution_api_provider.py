# agentle/agents/whatsapp/providers/evolution.py
"""
Evolution API implementation for WhatsApp with enhanced resilience.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime
from typing import Any, override
from urllib.parse import urljoin

import aiohttp

from agentle.agents.whatsapp.models.downloaded_media import DownloadedMedia
from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage
from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message_status import WhatsAppMessageStatus
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import (
    EvolutionAPIConfig,
)
from agentle.resilience.circuit_breaker.in_memory_circuit_breaker import (
    InMemoryCircuitBreaker,
)
from agentle.resilience.rate_limiting.in_memory_rate_limiter import (
    InMemoryRateLimiter,
)
from agentle.sessions.session_manager import SessionManager
from agentle.sessions.in_memory_session_store import InMemorySessionStore

logger = logging.getLogger(__name__)


class EvolutionAPIError(Exception):
    """Exception raised for Evolution API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Mapping[str, Any] | None = None,
        request_url: str | None = None,
        request_data: Mapping[str, Any] | None = None,
        is_retriable: bool = True,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        self.request_url = request_url
        self.request_data = request_data
        self.is_retriable = is_retriable

    def __str__(self) -> str:
        """Enhanced string representation with context."""
        base_message = super().__str__()
        details: list[str] = []

        if self.status_code:
            details.append(f"status={self.status_code}")
        if self.request_url:
            details.append(f"url={self.request_url}")
        if self.response_data:
            details.append(f"response={self.response_data}")

        if details:
            return f"{base_message} ({', '.join(details)})"
        return base_message


class EvolutionAPIProvider(WhatsAppProvider):
    """
    Evolution API implementation for WhatsApp messaging with enhanced resilience.

    This provider implements the WhatsApp interface using Evolution API,
    which provides a REST API for WhatsApp Web.

    Enhanced Features:
    - Circuit breaker pattern for fault tolerance
    - Rate limiting to prevent API abuse
    - Automatic retry with exponential backoff
    - Comprehensive error handling and recovery
    - Request/response monitoring and metrics
    - Connection pooling and timeout management
    - Automatic session management with TTL
    - Memory-efficient caching
    """

    config: EvolutionAPIConfig
    session_manager: SessionManager[WhatsAppSession]
    session_ttl_seconds: int
    _http_session: aiohttp.ClientSession | None
    _circuit_breaker: InMemoryCircuitBreaker | None
    _rate_limiter: InMemoryRateLimiter | None
    _request_metrics: MutableMapping[str, Any]
    _max_retries: int
    _base_retry_delay: float
    _connection_pool_size: int

    def __init__(
        self,
        config: EvolutionAPIConfig,
        session_manager: SessionManager[WhatsAppSession] | None = None,
        session_ttl_seconds: int = 3600,
        enable_circuit_breaker: bool = True,
        enable_rate_limiting: bool = True,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        connection_pool_size: int = 100,
    ):
        """
        Initialize Evolution API provider with enhanced resilience.

        Args:
            config: Evolution API configuration
            session_manager: Optional session manager (creates in-memory if not provided)
            session_ttl_seconds: Default TTL for sessions in seconds
            enable_circuit_breaker: Whether to enable circuit breaker
            enable_rate_limiting: Whether to enable rate limiting
            max_retries: Maximum number of retry attempts
            base_retry_delay: Base delay for exponential backoff
            connection_pool_size: HTTP connection pool size
        """
        logger.info(
            "Initializing Evolution API provider with instance '%s' at %s, "
            + "session_ttl=%ss, circuit_breaker=%s, rate_limiting=%s",
            config.instance_name,
            config.base_url,
            session_ttl_seconds,
            enable_circuit_breaker,
            enable_rate_limiting,
        )

        self.config = config
        self.session_ttl_seconds = session_ttl_seconds
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._connection_pool_size = connection_pool_size
        self._http_session = None

        # Initialize session manager
        if session_manager is None:
            logger.debug("Creating in-memory session store for Evolution API provider")
            session_store = InMemorySessionStore[WhatsAppSession]()
            self.session_manager = SessionManager(
                session_store=session_store,
                default_ttl_seconds=session_ttl_seconds,
                enable_metrics=True,
                max_retry_attempts=3,
            )
        else:
            logger.debug("Using provided session manager for Evolution API provider")
            self.session_manager = session_manager

        # Initialize circuit breaker
        if enable_circuit_breaker:
            self._circuit_breaker = InMemoryCircuitBreaker(
                failure_threshold=5,
                recovery_timeout=300.0,  # 5 minutes
                half_open_max_calls=3,
                enable_metrics=True,
            )
            logger.debug("Circuit breaker enabled for Evolution API provider")
        else:
            self._circuit_breaker = None

        # Initialize rate limiter
        if enable_rate_limiting:
            self._rate_limiter = InMemoryRateLimiter(
                default_config={
                    "max_requests_per_minute": 120,  # Conservative limit
                    "max_requests_per_hour": 5000,
                },
                enable_metrics=True,
                cleanup_interval_seconds=300,
            )
            logger.debug("Rate limiter enabled for Evolution API provider")
        else:
            self._rate_limiter = None

        # Initialize metrics
        self._request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_blocks": 0,
            "rate_limit_blocks": 0,
            "retry_attempts": 0,
            "average_response_time_ms": 0.0,
            "last_error_time": None,
            "last_error_message": None,
        }

        logger.info(
            f"Evolution API provider initialized successfully for instance '{config.instance_name}'"
        )

    def change_instance(self, instance_name: str) -> None:
        """Change the instance of the Evolution API provider."""
        self.config.instance_name = instance_name

    def clone(
        self,
        config: EvolutionAPIConfig | None = None,
        session_manager: SessionManager[WhatsAppSession] | None = None,
        session_ttl_seconds: int = 3600,
        enable_circuit_breaker: bool = True,
        enable_rate_limiting: bool = True,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        connection_pool_size: int = 100,
    ) -> EvolutionAPIProvider:
        return EvolutionAPIProvider(
            config=self.config.clone(
                new_base_url=config.base_url,
                new_instance_name=config.instance_name,
                new_api_key=config.api_key,
                new_webhook_url=config.webhook_url,
                new_timeout=config.timeout,
            )
            if config
            else self.config,
            session_manager=session_manager,
            session_ttl_seconds=session_ttl_seconds,
            enable_circuit_breaker=enable_circuit_breaker,
            enable_rate_limiting=enable_rate_limiting,
            max_retries=max_retries,
            base_retry_delay=base_retry_delay,
            connection_pool_size=connection_pool_size,
        )

    @override
    def get_instance_identifier(self) -> str:
        """Get the instance identifier for the WhatsApp provider."""
        return self.config.instance_name

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with optimized configuration."""
        if self._http_session is None:
            logger.debug("Creating new aiohttp session for Evolution API")
            headers = {
                "apikey": self.config.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Agentle-WhatsApp-Bot/1.0",
            }

            # Configure connection pooling and timeouts
            connector = aiohttp.TCPConnector(
                limit=self._connection_pool_size,
                limit_per_host=min(self._connection_pool_size // 2, 50),
                ttl_dns_cache=300,  # 5 minutes DNS cache
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=10,  # 10 seconds for connection
                sock_read=self.config.timeout - 10,  # Remaining time for reading
            )

            self._http_session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=connector,
                raise_for_status=False,  # We handle status codes manually
            )
            logger.debug(
                f"HTTP session created with timeout={self.config.timeout}s, pool_size={self._connection_pool_size}"
            )
        return self._http_session

    def _build_url(self, endpoint: str, use_message_prefix: bool = True) -> str:
        """
        Build full URL for API endpoint.

        Args:
            endpoint: The API endpoint
            use_message_prefix: Whether to prefix with /message/ (default: True)
        """
        if use_message_prefix:
            url = urljoin(self.config.base_url, f"/message/{endpoint}")
        else:
            url = urljoin(self.config.base_url, f"/{endpoint}")

        logger.debug(f"Built API URL: {url}")
        return url

    async def _make_request_with_resilience(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int | Sequence[int] = 200,
    ) -> Mapping[str, Any]:
        """
        Make HTTP request with comprehensive resilience mechanisms.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Full URL for the request
            data: Optional JSON data to send
            expected_status: Expected HTTP status code

        Returns:
            Response data as dictionary

        Raises:
            EvolutionAPIError: If the request fails after all resilience mechanisms
        """
        circuit_id = f"evolution_api_{self.config.instance_name}"
        rate_limit_id = f"api_{self.config.instance_name}"

        # Check circuit breaker
        if self._circuit_breaker and await self._circuit_breaker.is_open(circuit_id):
            self._request_metrics["circuit_breaker_blocks"] += 1
            logger.warning(
                f"Circuit breaker is open for {circuit_id}, blocking request to {url}"
            )
            raise EvolutionAPIError(
                "Circuit breaker is open, request blocked",
                request_url=url,
                is_retriable=False,
            )

        # Check rate limiting
        if self._rate_limiter and not await self._rate_limiter.can_proceed(
            rate_limit_id
        ):
            self._request_metrics["rate_limit_blocks"] += 1
            logger.warning(
                f"Rate limit exceeded for {rate_limit_id}, blocking request to {url}"
            )
            raise EvolutionAPIError(
                "Rate limit exceeded, request blocked",
                request_url=url,
                is_retriable=True,
            )

        # Record rate limit usage
        if self._rate_limiter:
            await self._rate_limiter.record_request(rate_limit_id)

        # Attempt request with retries
        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                response_data = await self._make_request(
                    method, url, data, expected_status
                )

                # Record success
                if self._circuit_breaker:
                    await self._circuit_breaker.record_success(circuit_id)

                self._request_metrics["successful_requests"] += 1
                return response_data

            except EvolutionAPIError as e:
                last_exception = e

                # Record failure
                if self._circuit_breaker:
                    await self._circuit_breaker.record_failure(circuit_id)

                self._request_metrics["failed_requests"] += 1
                self._request_metrics["last_error_time"] = time.time()
                self._request_metrics["last_error_message"] = str(e)

                # Check if error is retriable
                if not e.is_retriable or attempt >= self._max_retries:
                    break

                # Calculate delay with exponential backoff and jitter
                delay = self._base_retry_delay * (2**attempt)
                jitter = delay * 0.1 * (hash(url) % 10) / 10  # Deterministic jitter
                total_delay = delay + jitter

                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries + 1}), "
                    + f"retrying in {total_delay:.2f}s: {e}"
                )

                self._request_metrics["retry_attempts"] += 1

                import asyncio

                await asyncio.sleep(total_delay)

        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise EvolutionAPIError(
                "Request failed after all retries with no exception recorded"
            )

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int | Sequence[int] = 200,
    ) -> Mapping[str, Any]:
        """
        Make HTTP request with proper error handling and metrics.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Full URL for the request
            data: Optional JSON data to send
            expected_status: Expected HTTP status code

        Returns:
            Response data as dictionary

        Raises:
            EvolutionAPIError: If the request fails
        """
        start_time = time.time()
        self._request_metrics["total_requests"] += 1

        # Log request details (excluding sensitive data)
        safe_data = self._sanitize_request_data(data) if data else None
        logger.debug(f"Making {method} request to {url}")
        if safe_data:
            logger.debug(f"Request payload: {safe_data}")

        try:
            match method.upper():
                case "GET":
                    async with self.session.get(url) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case "POST":
                    async with self.session.post(url, json=data) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case "PUT":
                    async with self.session.put(url, json=data) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case "DELETE":
                    async with self.session.delete(url) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case _:
                    duration = time.time() - start_time
                    logger.error(
                        f"Unsupported HTTP method '{method}' for {url} (duration: {duration:.3f}s)"
                    )
                    raise ValueError(f"Unsupported HTTP method: {method}")

        except aiohttp.ClientError as e:
            duration = time.time() - start_time
            logger.error(
                f"HTTP client error for {method} {url} (duration: {duration:.3f}s): {type(e).__name__}: {e}",
                extra={
                    "method": method,
                    "url": url,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "request_data": self._sanitize_request_data(data) if data else None,
                },
            )
            raise EvolutionAPIError(
                f"Network error: {e}",
                request_url=url,
                request_data=self._sanitize_request_data(data) if data else None,
                is_retriable=True,
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Unexpected error for {method} {url} (duration: {duration:.3f}s): {type(e).__name__}: {e}",
                extra={
                    "method": method,
                    "url": url,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "request_data": self._sanitize_request_data(data) if data else None,
                },
            )
            raise EvolutionAPIError(
                f"Unexpected error: {e}",
                request_url=url,
                request_data=self._sanitize_request_data(data) if data else None,
                is_retriable=False,
            )

    def _sanitize_request_data(
        self, data: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Remove sensitive information from request data for logging."""
        if not data:
            return data

        # Create a copy and remove sensitive fields
        sanitized = dict(data)

        # Remove API keys and tokens
        for key in list(sanitized.keys()):
            if any(
                sensitive in key.lower()
                for sensitive in ["key", "token", "secret", "password"]
            ):
                sanitized[key] = "***REDACTED***"

        return sanitized

    async def _handle_response(
        self,
        response: aiohttp.ClientResponse,
        expected_status: int | Sequence[int],
        request_url: str,
        request_data: Mapping[str, Any] | None,
        start_time: float,
    ) -> Mapping[str, Any]:
        """
        Handle HTTP response with proper error handling and metrics.

        Args:
            response: aiohttp response object
            expected_status: Expected HTTP status code
            request_url: The URL that was requested
            request_data: The data that was sent with the request
            start_time: When the request was started

        Returns:
            Response data as dictionary

        Raises:
            EvolutionAPIError: If the response indicates an error
        """
        duration = time.time() - start_time

        # Update average response time
        current_avg = self._request_metrics["average_response_time_ms"]
        total_requests = self._request_metrics["total_requests"]
        self._request_metrics["average_response_time_ms"] = (
            current_avg * (total_requests - 1) + duration * 1000
        ) / total_requests

        logger.debug(
            f"Received response {response.status} for {request_url} (duration: {duration:.3f}s)",
            extra={
                "status_code": response.status,
                "url": request_url,
                "duration_seconds": duration,
                "expected_status": expected_status,
            },
        )

        if isinstance(expected_status, int):
            expected_status = [expected_status]

        if response.status in expected_status:
            try:
                response_data = await response.json()
                logger.debug(f"Response data received: {response_data}")
                return response_data
            except Exception as e:
                logger.warning(f"Response is not valid JSON: {e}, returning empty dict")
                # If response is not JSON, return empty dict
                return {}

        # Handle error responses
        try:
            error_data = await response.json()
            logger.debug(f"Error response data: {error_data}")
        except Exception as e:
            logger.warning(f"Failed to parse error response as JSON: {e}")
            error_text = await response.text()
            error_data = {"error": error_text}
            logger.debug(f"Error response text: {error_text}")

        error_message = f"Evolution API error: {response.status}"
        if "error" in error_data:
            error_message += f" - {error_data['error']}"
        elif "message" in error_data:
            error_message += f" - {error_data['message']}"

        # Determine if error is retriable
        is_retriable = self._is_retriable_error(response.status)

        logger.error(
            f"API request failed: {error_message} (duration: {duration:.3f}s, retriable: {is_retriable})",
            extra={
                "status_code": response.status,
                "url": request_url,
                "duration_seconds": duration,
                "error_data": error_data,
                "request_data": self._sanitize_request_data(request_data)
                if request_data
                else None,
                "is_retriable": is_retriable,
            },
        )

        raise EvolutionAPIError(
            error_message,
            status_code=response.status,
            response_data=error_data,
            request_url=request_url,
            request_data=self._sanitize_request_data(request_data)
            if request_data
            else None,
            is_retriable=is_retriable,
        )

    def _is_retriable_error(self, status_code: int) -> bool:
        """Determine if an HTTP error status code indicates a retriable error."""
        # Retriable errors: 5xx server errors, 429 rate limit, 408 timeout
        retriable_codes = {408, 429, 500, 502, 503, 504}
        return status_code in retriable_codes

    async def initialize(self) -> None:
        """Initialize the Evolution API connection with enhanced validation."""
        return
        # logger.info(
        #     f"Initializing Evolution API connection for instance '{self.config.instance_name}'"
        # )

        # try:
        #     # Check instance status
        #     url = self._build_url("instance/fetchInstances", use_message_prefix=False)
        #     logger.debug(f"Fetching instances from {url}")
        #     response_data = await self._make_request_with_resilience("GET", url)

        #     # Look for our instance in the response
        #     instances = (
        #         response_data if isinstance(response_data, list) else [response_data]
        #     )
        #     instance_found = False
        #     available_instances: list[str] = []

        #     logger.debug("Processing %d instances from API response", len(instances))

        #     for instance_data in instances:
        #         if isinstance(instance_data, dict):
        #             instance_name = instance_data.get("name")

        #             if instance_name and isinstance(instance_name, str):
        #                 available_instances.append(instance_name)
        #                 logger.debug(f"Found instance: {instance_name}")

        #                 if instance_name == self.config.instance_name:
        #                     instance_found = True
        #                     logger.info(
        #                         f"Target instance '{self.config.instance_name}' found and accessible"
        #                     )

        #                     # Log additional instance details if available
        #                     if "connectionStatus" in instance_data:
        #                         logger.info(
        #                             f"Instance connection status: {instance_data['connectionStatus']}"
        #                         )
        #                     if "profilePictureUrl" in instance_data:
        #                         logger.debug("Instance has profile picture configured")

        #     if not instance_found:
        #         error_msg = (
        #             f"Instance '{self.config.instance_name}' not found. "
        #             f"Available instances: {available_instances}"
        #         )
        #         logger.error(
        #             error_msg,
        #             extra={
        #                 "target_instance": self.config.instance_name,
        #                 "available_instances": available_instances,
        #                 "total_instances": len(available_instances),
        #             },
        #         )
        #         raise EvolutionAPIError(error_msg, is_retriable=False)

        #     logger.info(
        #         f"Evolution API provider initialized successfully for instance: {self.config.instance_name}"
        #     )

        # except EvolutionAPIError:
        #     logger.error("Failed to initialize Evolution API provider due to API error")
        #     raise
        # except Exception as e:
        #     logger.error(
        #         f"Failed to initialize Evolution API provider: {type(e).__name__}: {e}",
        #         extra={
        #             "instance_name": self.config.instance_name,
        #             "base_url": self.config.base_url,
        #             "error_type": type(e).__name__,
        #         },
        #     )
        #     raise EvolutionAPIError(f"Initialization failed: {e}", is_retriable=True)

    async def shutdown(self) -> None:
        """Shutdown the Evolution API connection and clean up resources."""
        logger.info("Shutting down Evolution API provider")

        try:
            # Close HTTP session
            if self._http_session:
                logger.debug("Closing aiohttp session")
                await self._http_session.close()
                self._http_session = None

            # Close resilience components
            if self._circuit_breaker:
                await self._circuit_breaker.close()
                logger.debug("Circuit breaker closed")

            if self._rate_limiter:
                await self._rate_limiter.close()
                logger.debug("Rate limiter closed")

            # Close session manager
            logger.debug("Closing session manager")
            await self.session_manager.close()

            logger.info("Evolution API provider shutdown complete")

        except Exception as e:
            logger.error(
                f"Error during Evolution API provider shutdown: {type(e).__name__}: {e}",
                extra={"error_type": type(e).__name__},
            )

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        """Send a text message via Evolution API with enhanced error handling."""
        logger.info(f"Sending text message to {to} (length: {len(text)} chars)")
        if quoted_message_id:
            logger.debug(f"Message is quoting message ID: {quoted_message_id}")

        try:
            # Check if there's a stored remoteJid for this contact
            session = await self.get_session(to)
            remote_jid = session.context_data.get("remote_jid") if session else None

            if remote_jid:
                logger.info(f"🔑 Using stored remoteJid for {to}: {remote_jid}")
                normalized_to = remote_jid
            else:
                normalized_to = self._normalize_phone(to)
                logger.debug(f"Normalized phone number: {to} -> {normalized_to}")

            payload: Mapping[str, Any] = {
                "number": normalized_to,
                "text": text,
            }

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"sendText/{self.config.instance_name}")
            response_data = await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            # Validate response structure - API can return 201 with error payload
            if "key" not in response_data:
                # Check if this is an error response disguised as success
                error_msg = "Invalid response structure from Evolution API"
                if "message" in response_data:
                    error_msg = f"Evolution API error: {response_data['message']}"
                elif "error" in response_data:
                    error_msg = f"Evolution API error: {response_data['error']}"
                
                logger.error(
                    f"API returned success status but error payload: {response_data}",
                    extra={
                        "to_number": to,
                        "response_data": response_data,
                    }
                )
                raise EvolutionAPIError(
                    error_msg,
                    response_data=response_data,
                    request_url=url,
                    is_retriable=False,
                )

            message_id = response_data["key"]["id"]
            from_jid = response_data["key"]["remoteJid"]

            message = WhatsAppTextMessage(
                id=message_id,
                from_number=from_jid,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                text=text,
                quoted_message_id=quoted_message_id,
            )

            logger.info(
                f"Text message sent successfully to {to}: {message_id}",
                extra={
                    "message_id": message_id,
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "from_jid": from_jid,
                    "text_length": len(text),
                    "has_quote": quoted_message_id is not None,
                },
            )
            return message

        except EvolutionAPIError:
            logger.error(f"Evolution API error while sending text message to {to}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to send text message to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                    "has_quote": quoted_message_id is not None,
                },
            )
            raise EvolutionAPIError(f"Failed to send text message: {e}")

    # [Continue with other methods using the same pattern...]
    # For brevity, I'll include key methods with the enhanced resilience pattern

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ) -> WhatsAppMediaMessage:
        """Send a media message via Evolution API with enhanced error handling."""
        logger.info(
            f"Sending {media_type} media message to {to}",
            extra={
                "to_number": to,
                "media_type": media_type,
                "media_url": media_url,
                "has_caption": caption is not None,
                "has_filename": filename is not None,
                "has_quote": quoted_message_id is not None,
            },
        )

        try:
            # Determine endpoint based on media type
            endpoint_map = {
                "image": "sendMedia",
                "document": "sendMedia",
                "audio": "sendWhatsappAudio",
                "video": "sendMedia",
            }

            endpoint = endpoint_map.get(media_type)
            if not endpoint:
                logger.error(
                    f"Unsupported media type: {media_type}. Supported types: {list(endpoint_map.keys())}"
                )
                raise EvolutionAPIError(
                    f"Unsupported media type: {media_type}", is_retriable=False
                )

            normalized_to = self._normalize_phone(to)
            logger.debug(f"Normalized phone number: {to} -> {normalized_to}")
            logger.debug(f"Using endpoint: {endpoint}")

            payload: MutableMapping[str, Any] = {
                "number": normalized_to,
                "mediatype": media_type,
                "mimetype": f"{media_type}/*",
                "caption": caption or "",
                "mediaMessage": {
                    "mediaurl": media_url
                },  # Note: Evolution API uses "mediaurl" not "mediaUrl"
            }

            if caption:
                payload["mediaMessage"]["caption"] = caption
                logger.debug(f"Added caption (length: {len(caption)} chars)")

            if filename and media_type == "document":
                payload["mediaMessage"]["fileName"] = filename
                logger.debug(f"Added filename: {filename}")

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"{endpoint}/{self.config.instance_name}")
            response_data = await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            # Validate response structure - API can return 201 with error payload
            if "key" not in response_data:
                error_msg = "Invalid response structure from Evolution API"
                if "message" in response_data:
                    error_msg = f"Evolution API error: {response_data['message']}"
                elif "error" in response_data:
                    error_msg = f"Evolution API error: {response_data['error']}"
                
                logger.error(
                    f"API returned success status but error payload: {response_data}",
                    extra={"to_number": to, "response_data": response_data}
                )
                raise EvolutionAPIError(
                    error_msg, response_data=response_data, request_url=url, is_retriable=False
                )

            message_id = response_data["key"]["id"]
            from_jid = response_data["key"]["remoteJid"]

            # Create appropriate media message type
            message_class_map = {
                "image": WhatsAppImageMessage,
                "document": WhatsAppDocumentMessage,
                "audio": WhatsAppAudioMessage,
                "video": WhatsAppVideoMessage,
            }

            message_class = message_class_map[media_type]
            message = message_class(
                id=message_id,
                from_number=from_jid,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                media_url=media_url,
                media_mime_type=f"{media_type}/*",
                caption=caption,
                filename=filename,
                quoted_message_id=quoted_message_id,
            )

            logger.info(
                f"{media_type.title()} media message sent successfully to {to}: {message_id}",
                extra={
                    "message_id": message_id,
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "from_jid": from_jid,
                    "media_type": media_type,
                    "media_url": media_url,
                    "has_caption": caption is not None,
                    "has_filename": filename is not None,
                },
            )
            return message

        except EvolutionAPIError:
            logger.error(
                f"Evolution API error while sending {media_type} media message to {to}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to send {media_type} media message to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "media_type": media_type,
                    "media_url": media_url,
                    "error_type": type(e).__name__,
                },
            )
            raise EvolutionAPIError(f"Failed to send media message: {e}")

    # [Continue with remaining methods using enhanced patterns...]

    async def send_audio_message(
        self,
        to: str,
        audio_base64: str,
        quoted_message_id: str | None = None,
    ) -> WhatsAppMediaMessage:
        """Send an audio message via Evolution API with enhanced error handling."""
        logger.info(f"Sending audio message to {to}")
        if quoted_message_id:
            logger.debug(f"Audio message is quoting message ID: {quoted_message_id}")

        try:
            # CRITICAL FIX: Check if there's a stored remoteJid for this contact
            session = await self.get_session(to)
            remote_jid = session.context_data.get("remote_jid") if session else None

            if remote_jid:
                logger.info(
                    f"🔑 Using stored remoteJid for audio to {to}: {remote_jid}"
                )
                normalized_to = remote_jid
            else:
                normalized_to = self._normalize_phone(to)
                logger.debug(f"Normalized phone number: {to} -> {normalized_to}")

            payload: MutableMapping[str, Any] = {
                "number": normalized_to,
                "audio": audio_base64,
            }

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"sendWhatsAppAudio/{self.config.instance_name}")
            response_data = await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            # Validate response structure - API can return 201 with error payload
            if "key" not in response_data:
                error_msg = "Invalid response structure from Evolution API"
                if "message" in response_data:
                    error_msg = f"Evolution API error: {response_data['message']}"
                elif "error" in response_data:
                    error_msg = f"Evolution API error: {response_data['error']}"
                
                logger.error(
                    f"API returned success status but error payload: {response_data}",
                    extra={"to_number": to, "response_data": response_data}
                )
                raise EvolutionAPIError(
                    error_msg, response_data=response_data, request_url=url, is_retriable=False
                )

            message_id = response_data["key"]["id"]
            from_jid = response_data["key"]["remoteJid"]

            message = WhatsAppAudioMessage(
                id=message_id,
                from_number=from_jid,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                media_url="",  # Base64 audio doesn't have a URL
                media_mime_type="audio/ogg",
                quoted_message_id=quoted_message_id,
                is_voice_note=True,
            )

            logger.info(
                f"Audio message sent successfully to {to}: {message_id}",
                extra={
                    "message_id": message_id,
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "from_jid": from_jid,
                    "has_quote": quoted_message_id is not None,
                },
            )
            return message

        except EvolutionAPIError:
            logger.error(f"Evolution API error while sending audio message to {to}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to send audio message to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "error_type": type(e).__name__,
                    "has_quote": quoted_message_id is not None,
                },
            )
            raise EvolutionAPIError(f"Failed to send audio message: {e}")

    async def send_audio_message_by_url(
        self,
        to: str,
        audio_url: str,
        quoted_message_id: str | None = None,
    ) -> WhatsAppMediaMessage:
        """Send an audio message via URL using Evolution API."""
        logger.info(f"Sending audio message via URL to {to}: {audio_url}")
        if quoted_message_id:
            logger.debug(f"Audio message is quoting message ID: {quoted_message_id}")

        try:
            # CRITICAL FIX: Check if there's a stored remoteJid for this contact
            session = await self.get_session(to)
            remote_jid = session.context_data.get("remote_jid") if session else None

            if remote_jid:
                logger.info(
                    f"🔑 Using stored remoteJid for audio URL to {to}: {remote_jid}"
                )
                normalized_to = remote_jid
            else:
                normalized_to = self._normalize_phone(to)
                logger.debug(f"Normalized phone number: {to} -> {normalized_to}")

            payload: MutableMapping[str, Any] = {
                "number": normalized_to,
                "audioUrl": audio_url,  # Use URL instead of base64
            }

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"sendWhatsAppAudio/{self.config.instance_name}")
            response_data = await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            # Validate response structure - API can return 201 with error payload
            if "key" not in response_data:
                error_msg = "Invalid response structure from Evolution API"
                if "message" in response_data:
                    error_msg = f"Evolution API error: {response_data['message']}"
                elif "error" in response_data:
                    error_msg = f"Evolution API error: {response_data['error']}"
                
                logger.error(
                    f"API returned success status but error payload: {response_data}",
                    extra={"to_number": to, "response_data": response_data}
                )
                raise EvolutionAPIError(
                    error_msg, response_data=response_data, request_url=url, is_retriable=False
                )

            message_id = response_data["key"]["id"]
            from_jid = response_data["key"]["remoteJid"]

            message = WhatsAppAudioMessage(
                id=message_id,
                from_number=from_jid,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                media_url=audio_url,  # Store the URL
                media_mime_type="audio/ogg",
                quoted_message_id=quoted_message_id,
                is_voice_note=True,
            )

            logger.info(
                f"Audio message sent successfully via URL to {to}: {message_id}",
                extra={
                    "message_id": message_id,
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "from_jid": from_jid,
                    "audio_url": audio_url,
                    "has_quote": quoted_message_id is not None,
                },
            )
            return message

        except EvolutionAPIError:
            logger.error(
                f"Evolution API error while sending audio message via URL to {to}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to send audio message via URL to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "audio_url": audio_url,
                    "error_type": type(e).__name__,
                    "has_quote": quoted_message_id is not None,
                },
            )
            raise EvolutionAPIError(f"Failed to send audio message via URL: {e}")

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """Send typing indicator via Evolution API."""
        logger.debug(f"Sending typing indicator to {to} for {duration}s")

        try:
            # Check if there's a stored remoteJid for this contact
            session = await self.get_session(to)
            remote_jid = session.context_data.get("remote_jid") if session else None

            if remote_jid:
                logger.debug(
                    f"🔑 Using stored remoteJid for typing indicator to {to}: {remote_jid}"
                )
                normalized_to = remote_jid
            else:
                normalized_to = self._normalize_phone(to)

            payload = {
                "number": normalized_to,
                "presence": "composing",
                "delay": duration * 1000,
                "options": {
                    "delay": duration * 1000,
                    "presence": "composing",
                    "number": normalized_to,
                },  # Evolution API expects milliseconds
            }

            url = self._build_url(
                f"chat/sendPresence/{self.config.instance_name}",
                use_message_prefix=False,
            )
            await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            logger.debug(
                f"Typing indicator sent successfully to {to} for {duration}s",
                extra={
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "duration_seconds": duration,
                },
            )

        except EvolutionAPIError as e:
            # Typing indicator failures are non-critical
            logger.warning(
                f"Failed to send typing indicator to {to}: {e}",
                extra={"to_number": to, "duration_seconds": duration, "error": str(e)},
            )
        except Exception as e:
            logger.warning(
                f"Failed to send typing indicator to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                },
            )

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        """Send recording indicator via Evolution API."""
        logger.debug(f"Sending recording indicator to {to} for {duration}s")

        try:
            # Check if there's a stored remoteJid for this contact
            session = await self.get_session(to)
            remote_jid = session.context_data.get("remote_jid") if session else None

            if remote_jid:
                logger.debug(
                    f"🔑 Using stored remoteJid for recording indicator to {to}: {remote_jid}"
                )
                normalized_to = remote_jid
            else:
                normalized_to = self._normalize_phone(to)

            payload = {
                "number": normalized_to,
                "presence": "recording",
                "delay": duration * 1000,
                "options": {
                    "delay": duration * 1000,
                    "presence": "recording",
                    "number": normalized_to,
                },  # Evolution API expects milliseconds
            }

            url = self._build_url(
                f"chat/sendPresence/{self.config.instance_name}",
                use_message_prefix=False,
            )
            await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            logger.debug(
                f"Recording indicator sent successfully to {to} for {duration}s",
                extra={
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "duration_seconds": duration,
                },
            )

        except EvolutionAPIError as e:
            # Recording indicator failures are non-critical
            logger.warning(
                f"Failed to send recording indicator to {to}: {e}",
                extra={"to_number": to, "duration_seconds": duration, "error": str(e)},
            )
        except Exception as e:
            logger.warning(
                f"Failed to send recording indicator to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                },
            )

    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read via Evolution API."""
        logger.debug(f"Marking message as read: {message_id}")

        try:
            # Extract the phone number from message_id if it's in Evolution format
            if "@" in message_id:
                msg_id, phone = message_id.split("@", 1)
                logger.debug(f"Extracted message ID: {msg_id}, phone: {phone}")
            else:
                msg_id = message_id
                phone = ""
                logger.debug(f"Using message ID as-is (no phone extraction): {msg_id}")

            payload = {
                "readMessages": [{"id": msg_id, "remoteJid": phone, "fromMe": False}]
            }

            url = self._build_url(
                f"chat/markMessageAsRead/{self.config.instance_name}",
                use_message_prefix=False,
            )
            await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            logger.debug(
                f"Message marked as read successfully: {message_id}",
                extra={
                    "message_id": message_id,
                    "extracted_id": msg_id,
                    "phone": phone,
                },
            )

        except EvolutionAPIError as e:
            # Read receipt failures are non-critical
            logger.warning(
                f"Failed to mark message as read: {message_id}: {e}",
                extra={"message_id": message_id, "error": str(e)},
            )
        except Exception as e:
            logger.warning(
                f"Failed to mark message as read: {message_id}: {type(e).__name__}: {e}",
                extra={"message_id": message_id, "error_type": type(e).__name__},
            )

    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        """Get contact information via Evolution API with enhanced error handling."""
        logger.debug(f"Fetching contact info for {phone}")

        try:
            normalized_phone = self._normalize_phone(phone)
            logger.debug(
                f"Normalized phone for contact fetch: {phone} -> {normalized_phone}"
            )

            url = self._build_url(
                f"chat/fetchProfile/{self.config.instance_name}",
                use_message_prefix=False,
            )

            # Extract digits only for the API call
            normalized_phone_digits = re.match(r"\d+", normalized_phone)
            normalized_phone_digits = (
                normalized_phone_digits.group() if normalized_phone_digits else ""
            )
            payload = {"number": normalized_phone_digits}

            response_data = await self._make_request_with_resilience(
                "POST", url, payload
            )

            if not response_data:
                logger.debug(f"No contact data returned for {phone}")
                return None

            contact = WhatsAppContact(
                phone=normalized_phone_digits,
                name=response_data.get("name"),
                push_name=response_data.get("pushName"),
                profile_picture_url=response_data.get("profilePictureUrl"),
            )

            logger.info(
                f"Contact info retrieved successfully for {phone}",
                extra={
                    "phone": phone,
                    "normalized_phone": normalized_phone_digits,
                    "has_name": contact.name is not None,
                    "has_push_name": contact.push_name is not None,
                    "has_profile_picture": contact.profile_picture_url is not None,
                },
            )
            return contact

        except EvolutionAPIError as e:
            logger.warning(
                f"Evolution API error while fetching contact info for {phone}: {e}",
                extra={"phone": phone, "error": str(e)},
            )
            return None
        except Exception as e:
            logger.warning(
                f"Failed to get contact info for {phone}: {type(e).__name__}: {e}",
                extra={"phone": phone, "error_type": type(e).__name__},
            )
            return None

    async def get_session(self, phone: str) -> WhatsAppSession | None:
        """Get or create a session for a phone number with enhanced error handling."""
        logger.debug(f"Getting/creating session for {phone}")

        try:
            # CRITICAL FIX: Do NOT normalize phone for session management
            # Sessions should use the original phone number to avoid duplicates
            # Normalization should ONLY happen when making Evolution API calls
            clean_phone = phone.split("@")[0] if "@" in phone else phone
            session_id = f"{self.config.instance_name}_{clean_phone}"

            logger.debug(f"Session ID: {session_id}")

            # Try to get existing session
            session = await self.session_manager.get_session(
                session_id, refresh_ttl=True
            )

            if session:
                # Update last activity
                session.last_activity = datetime.now()
                await self.session_manager.update_session(session_id, session)
                logger.debug(
                    f"Retrieved existing session for {phone}",
                    extra={
                        "phone": phone,
                        "session_id": session_id,
                        "last_activity": session.last_activity,
                    },
                )
                return session

            # Create new session
            logger.debug(f"Creating new session for {phone}")
            contact = await self.get_contact_info(phone)
            if not contact:
                logger.debug(
                    f"No contact info available, creating minimal contact for {phone}"
                )
                contact = WhatsAppContact(phone=clean_phone)

            new_session = WhatsAppSession(
                session_id=session_id,
                phone_number=clean_phone,
                contact=contact,
            )

            # Store the session
            success = await self.session_manager.create_session(
                session_id, new_session, ttl_seconds=self.session_ttl_seconds
            )

            if success:
                return new_session
            else:
                logger.warning(
                    f"Failed to create session for {phone}, session may already exist"
                )
                # Try to get the existing session again
                return await self.session_manager.get_session(session_id)

        except Exception as e:
            logger.error(
                f"Failed to get/create session for {phone}: {type(e).__name__}: {e}",
                extra={"phone": phone, "error_type": type(e).__name__},
            )
            return None

    async def update_session(self, session: WhatsAppSession) -> None:
        """Update session data with enhanced error handling."""
        logger.debug(f"Updating session: {session.session_id}")

        try:
            session.last_activity = datetime.now()
            success = await self.session_manager.update_session(
                session.session_id, session, ttl_seconds=self.session_ttl_seconds
            )

            if success:
                logger.debug(
                    f"Session updated successfully: {session.session_id}",
                    extra={
                        "session_id": session.session_id,
                        "phone_number": session.phone_number,
                        "last_activity": session.last_activity,
                    },
                )
            else:
                logger.warning(f"Failed to update session: {session.session_id}")

        except Exception as e:
            logger.error(
                f"Failed to update session {session.session_id}: {type(e).__name__}: {e}",
                extra={
                    "session_id": session.session_id,
                    "phone_number": session.phone_number,
                    "error_type": type(e).__name__,
                },
            )

    @override
    async def validate_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        """Process incoming webhook data from Evolution API with enhanced validation."""
        logger.info(f"Validating webhook payload with event: {payload.event}")

        try:
            # Evolution API webhook structure validation
            event_type = payload.event
            if event_type is None:
                logger.error("Webhook validation failed: Event type is missing")
                raise EvolutionAPIError(
                    "Event type is required in webhook payload", is_retriable=False
                )

            instance_name = payload.instance
            if instance_name is None:
                logger.error("Webhook validation failed: Instance name is missing")
                raise EvolutionAPIError(
                    "Instance name is required in webhook payload", is_retriable=False
                )

            if instance_name != self.config.instance_name:
                logger.error(
                    f"Webhook validation failed: Instance mismatch - expected '{self.config.instance_name}', got '{instance_name}'"
                )
                raise EvolutionAPIError(
                    f"Webhook for wrong instance: expected {self.config.instance_name}, got {instance_name}",
                    is_retriable=False,
                )

            logger.info(
                f"Webhook validated successfully: {event_type} for instance {instance_name}",
                extra={
                    "event_type": event_type,
                    "instance_name": instance_name,
                    "expected_instance": self.config.instance_name,
                },
            )

        except EvolutionAPIError:
            logger.error("Webhook validation failed due to Evolution API error")
            raise
        except Exception as e:
            logger.error(
                f"Failed to validate webhook: {type(e).__name__}: {e}",
                extra={"error_type": type(e).__name__},
            )
            raise EvolutionAPIError(
                f"Failed to process webhook: {e}", is_retriable=False
            )

    async def download_media(self, media_id: str) -> DownloadedMedia:
        """Download media content by ID with enhanced error handling."""
        logger.info(f"Downloading media with ID: {media_id}")

        try:
            url = self._build_url(
                f"chat/getBase64FromMediaMessage/{self.config.instance_name}",
                use_message_prefix=False,
            )

            payload = {"message": {"key": {"id": media_id}}, "convertToMp4": True}

            response_data = await self._make_request_with_resilience(
                "POST", url, payload, expected_status=[200, 201]
            )

            nested_data = response_data.get("data")
            nested_mapping = nested_data if isinstance(nested_data, Mapping) else {}
            response_keys = sorted(str(key) for key in response_data.keys())
            nested_data_keys = sorted(str(key) for key in nested_mapping.keys())

            base64_payload = response_data.get("base64") or nested_mapping.get("base64")
            mimetype_payload = (
                response_data.get("mimetype")
                or nested_mapping.get("mimetype")
                or "application/octet-stream"
            )
            detected_media_type = str(mimetype_payload).split("/", 1)[0].strip()

            if not base64_payload:
                logger.error(
                    "Media download failed: No base64 data in response for media %s "
                    + "(media_type=%s, response_keys=%s, nested_data_keys=%s)",
                    media_id,
                    detected_media_type or "unknown",
                    response_keys,
                    nested_data_keys,
                )
                raise EvolutionAPIError(
                    "No base64 data in media response", is_retriable=False
                )

            import base64

            normalized_base64_payload = str(base64_payload)
            if ";base64," in normalized_base64_payload:
                normalized_base64_payload = normalized_base64_payload.split(
                    ";base64,", maxsplit=1
                )[1]

            media_data = base64.b64decode(normalized_base64_payload)
            media_size = len(media_data)

            logger.info(
                f"Media downloaded successfully: {media_id} ({media_size} bytes)",
                extra={"media_id": media_id, "size_bytes": media_size},
            )

            mimetype = str(mimetype_payload).split(";")[0].strip()
            if not mimetype:
                mimetype = "application/octet-stream"

            return DownloadedMedia(data=media_data, mime_type=mimetype)

        except EvolutionAPIError:
            logger.error(f"Evolution API error while downloading media {media_id}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to download media {media_id}: {type(e).__name__}: {e}",
                extra={"media_id": media_id, "error_type": type(e).__name__},
            )
            raise EvolutionAPIError(f"Failed to download media: {e}")

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        webhook_url = self.config.webhook_url or ""
        logger.debug(f"Retrieved webhook URL: {webhook_url}")
        return webhook_url

    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages with enhanced error handling."""
        logger.info(f"Setting webhook URL: {url}")

        try:
            webhook_config = {
                "webhook": {
                    "url": url,
                    "webhook_by_events": True,
                    "events": [
                        "messages.upsert",
                        "messages.update",
                        "send.message",
                        "connection.update",
                    ],
                }
            }

            api_url = self._build_url(
                f"webhook/set/{self.config.instance_name}", use_message_prefix=False
            )
            await self._make_request_with_resilience("PUT", api_url, webhook_config)

            self.config.webhook_url = url
            logger.info(
                f"Webhook URL set successfully: {url}",
                extra={
                    "webhook_url": url,
                    "instance_name": self.config.instance_name,
                    "events": webhook_config["webhook"]["events"],
                },
            )

        except EvolutionAPIError:
            logger.error(f"Evolution API error while setting webhook URL: {url}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to set webhook URL: {url}: {type(e).__name__}: {e}",
                extra={"webhook_url": url, "error_type": type(e).__name__},
            )
            raise EvolutionAPIError(f"Failed to set webhook URL: {e}")

    def _normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to Evolution API format.

        Evolution API expects phone numbers in the format: countrycode+number@s.whatsapp.net
        
        This function:
        - Removes any existing suffix (@s.whatsapp.net, @lid, etc.)
        - Strips non-numeric characters
        - Adds country code 55 if not present
        - Appends @s.whatsapp.net suffix
        
        NOTE: This function does NOT modify the number itself (no '9' insertion).
        The number is sent exactly as provided by the user/webhook.
        """
        original_phone = phone

        # Remove any @ suffix if present
        if "@" in phone:
            phone = phone.split("@")[0]

        # Remove non-numeric characters
        phone = "".join(c for c in phone if c.isdigit())

        # Add country code 55 if not present
        if not phone.startswith("55"):
            phone = "55" + phone
            logger.debug(
                f"Added country code 55 to phone: {original_phone} -> {phone}"
            )

        # Always use @s.whatsapp.net for normal numbers
        phone = phone + "@s.whatsapp.net"

        if original_phone != phone:
            logger.info(f"Phone number normalized: {original_phone} -> {phone}")

        return phone

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get comprehensive statistics about the Evolution API provider.

        Returns:
            Dictionary with provider statistics including resilience metrics
        """
        logger.debug("Retrieving provider statistics")

        base_stats: MutableMapping[str, Any] = {
            "instance_name": self.config.instance_name,
            "base_url": self.config.base_url,
            "webhook_url": self.config.webhook_url,
            "timeout": self.config.timeout,
            "session_ttl_seconds": self.session_ttl_seconds,
            "has_active_session": self._http_session is not None,
            "max_retries": self._max_retries,
            "base_retry_delay": self._base_retry_delay,
            "connection_pool_size": self._connection_pool_size,
        }

        # Add request metrics
        base_stats["request_metrics"] = dict(self._request_metrics)

        # Add session manager stats
        session_stats = self.session_manager.get_stats()
        base_stats["session_stats"] = session_stats

        # Add circuit breaker stats if enabled
        if self._circuit_breaker:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We can't await in a sync method, so we create a task
                    # This is not ideal but provides some stats
                    base_stats["circuit_breaker_enabled"] = True
                else:
                    base_stats["circuit_breaker_enabled"] = True
            except Exception:
                base_stats["circuit_breaker_enabled"] = True
        else:
            base_stats["circuit_breaker_enabled"] = False

        # Add rate limiter stats if enabled
        if self._rate_limiter:
            base_stats["rate_limiter_enabled"] = True
        else:
            base_stats["rate_limiter_enabled"] = False

        logger.debug(f"Provider statistics: {base_stats}")
        return base_stats

    async def get_detailed_stats(self) -> Mapping[str, Any]:
        """
        Get detailed statistics including async data from resilience components.

        Returns:
            Dictionary with comprehensive statistics
        """
        base_stats = dict(self.get_stats())

        # Add detailed circuit breaker metrics
        if self._circuit_breaker:
            cb_metrics = await self._circuit_breaker.get_metrics()
            base_stats["circuit_breaker_metrics"] = cb_metrics

        # Add detailed rate limiter metrics
        if self._rate_limiter:
            rl_metrics = await self._rate_limiter.get_metrics()
            base_stats["rate_limiter_metrics"] = rl_metrics

        return base_stats

    async def reset_stats(self) -> None:
        """Reset all statistics and metrics."""
        # Reset request metrics
        self._request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_blocks": 0,
            "rate_limit_blocks": 0,
            "retry_attempts": 0,
            "average_response_time_ms": 0.0,
            "last_error_time": None,
            "last_error_message": None,
        }

        # Reset resilience component metrics
        if self._circuit_breaker:
            await self._circuit_breaker.reset_metrics()

        if self._rate_limiter:
            await self._rate_limiter.reset_metrics()

        # Reset session manager metrics
        self.session_manager.reset_metrics()

        logger.info("All Evolution API provider statistics reset")
