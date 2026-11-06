"""HTTP GET for user-supplied URLs, guarded against server-side request forgery.

Every hop, including each redirect, is checked before connecting:
- only http and https on their default ports, with no credentials in the URL;
- the hostname must resolve only to public addresses, and the connection goes to
  the address that was checked, so DNS can't be swapped in between (rebinding).
  TLS certificates are still verified against the real hostname via SNI;
- bodies are streamed and abandoned past a size limit, under an overall deadline.
Environment proxy settings are ignored so requests can't be routed elsewhere.
"""
import ipaddress
import socket
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from backend.errors import BlockedDestination, FetchFailed

MAX_URL_LENGTH = 2048
MAX_REDIRECTS = 5
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
DEFAULT_PORTS = {"http": 80, "https": 443}


def system_resolver(host, port):
    """Every address `host` resolves to, as strings."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchFailed("That website's address couldn't be found.", code="unknown_host") from None
    return [info[4][0] for info in infos]


def is_public_address(address):
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6:
        # Addresses that embed an IPv4 address are judged by that address.
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is not None:
            ip = embedded
    return ip.is_global and not ip.is_multicast


def check_url(url):
    """Parsed URL if it's an acceptable http(s) address, else FetchFailed."""
    if not isinstance(url, str) or len(url) > MAX_URL_LENGTH:
        raise FetchFailed("That link is too long.", code="invalid_url")
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        raise FetchFailed("That doesn't look like a valid link.", code="invalid_url") from None
    if parts.scheme not in DEFAULT_PORTS:
        raise FetchFailed("Only http:// and https:// links are supported.", code="invalid_url")
    if not parts.hostname:
        raise FetchFailed("That link has no website address.", code="invalid_url")
    if parts.username is not None or parts.password is not None:
        raise FetchFailed("Links with a username or password aren't supported.", code="invalid_url")
    if port is not None and port != DEFAULT_PORTS[parts.scheme]:
        raise BlockedDestination("Links to non-standard ports aren't supported.")
    return parts


@dataclass
class FetchResult:
    url: str            # final URL after redirects
    status: int
    content_type: str   # lower-case media type without parameters
    charset: str | None
    body: bytes


class SafeFetcher:
    def __init__(self, user_agent, resolver=system_resolver, transport=None,
                 timeout=10.0, deadline=25.0):
        self.user_agent = user_agent
        self.resolver = resolver
        self.deadline = deadline
        self.client = httpx.Client(transport=transport, follow_redirects=False, trust_env=False,
                                   timeout=httpx.Timeout(timeout, connect=5.0))

    def close(self):
        self.client.close()

    def _pinned_request(self, parts, accept):
        """Request for `parts` that connects to a vetted address of its host."""
        host = parts.hostname
        port = DEFAULT_PORTS[parts.scheme]
        try:
            literal = ipaddress.ip_address(host)
            addresses = [str(literal)]
        except ValueError:
            addresses = self.resolver(host, port)
        if not addresses or not all(is_public_address(address) for address in addresses):
            raise BlockedDestination("That link points to a private or local network address.")

        address = addresses[0]
        connect_host = f"[{address}]" if ":" in address else address
        target = urlunsplit((parts.scheme, connect_host, parts.path or "/", parts.query, ""))
        headers = {"Host": host if parts.port is None else f"{host}:{parts.port}",
                   "User-Agent": self.user_agent, "Accept": accept,
                   "Accept-Encoding": "gzip, deflate"}
        extensions = {"sni_hostname": host} if parts.scheme == "https" else {}
        return self.client.build_request("GET", target, headers=headers, extensions=extensions)

    def get(self, url, max_bytes, accept="*/*"):
        """Follow up to MAX_REDIRECTS safe redirects; returns any final status."""
        started = time.monotonic()
        for _ in range(MAX_REDIRECTS + 1):
            parts = check_url(url)
            request = self._pinned_request(parts, accept)
            try:
                response = self.client.send(request, stream=True)
            except httpx.TimeoutException:
                raise FetchFailed("The website took too long to respond.", code="timeout") from None
            except httpx.HTTPError:
                raise FetchFailed("The website couldn't be reached.", code="unreachable") from None
            try:
                location = response.headers.get("location")
                if response.status_code in REDIRECT_STATUSES and location:
                    url = urljoin(urlunsplit(parts), location)
                    continue
                body = self._read_limited(response, max_bytes, started)
            finally:
                response.close()
            media_type, _, params = response.headers.get("content-type", "").partition(";")
            charset = None
            for param in params.split(";"):
                key, _, value = param.strip().partition("=")
                if key.lower() == "charset" and value:
                    charset = value.strip("\"' ")
            return FetchResult(url=urlunsplit(parts), status=response.status_code,
                               content_type=media_type.strip().lower(), charset=charset, body=body)
        raise FetchFailed("The link redirected too many times.", code="too_many_redirects")

    def _read_limited(self, response, max_bytes, started):
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise FetchFailed("The file is too large.", code="too_large")
        chunks, total = [], 0
        try:
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise FetchFailed("The file is too large.", code="too_large")
                if time.monotonic() - started > self.deadline:
                    raise FetchFailed("The website took too long to respond.", code="timeout")
                chunks.append(chunk)
        except httpx.TimeoutException:
            raise FetchFailed("The website took too long to respond.", code="timeout") from None
        except httpx.HTTPError:
            raise FetchFailed("The download was interrupted.", code="unreachable") from None
        return b"".join(chunks)
