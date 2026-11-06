"""Fetch an explicitly selected public PDF without following arbitrary network URLs.

This service does not parse/index a paper for RAG. A caller must require matched
identity or explicit human review before invoking its parsing pipeline.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import re
import socket
import ssl
import tempfile
import threading
import time
import unicodedata
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .paper_discovery import PaperCandidate, normalize_doi

_DNS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="refmind-pdf-dns")
_DNS_SLOTS = threading.BoundedSemaphore(4)


class FulltextError(RuntimeError):
    pass


class FulltextIdentityError(FulltextError):
    pass


@dataclass(frozen=True)
class FetchedPaper:
    path: Path
    sha256: str
    size_bytes: int
    source_url: str
    resolved_url: str
    version: str
    license: str
    identity_status: str
    warnings: tuple[str, ...] = ()
    detected_title: str = ""
    detected_dois: tuple[str, ...] = ()

    def to_dict(self):
        return {**asdict(self), "path": str(self.path)}


def _public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
        return address.is_global and not address.is_multicast and not address.is_unspecified
    except ValueError:
        return False


def _public_url(url: str, timeout: float = 10):
    if not isinstance(url, str) or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise FulltextError("invalid PDF URL")
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in {80, 443}:
            raise ValueError()
        if not _DNS_SLOTS.acquire(blocking=False):
            raise FulltextError("PDF address resolver is busy")
        try:
            future = _DNS_EXECUTOR.submit(socket.getaddrinfo, parsed.hostname, port, type=socket.SOCK_STREAM)
        except BaseException:
            _DNS_SLOTS.release()
            raise
        future.add_done_callback(lambda _: _DNS_SLOTS.release())
        addresses = future.result(timeout=max(.01, timeout))
    except (ValueError, OSError, FutureTimeout):
        raise FulltextError("PDF address cannot be resolved safely") from None
    if not addresses or any(not _public_address(record[4][0]) for record in addresses):
        raise FulltextError("non-public PDF address rejected")
    return parsed, port, tuple(dict.fromkeys(record[4][0] for record in addresses))


@contextmanager
def _open_public(url: str, timeout: float):
    """Pin the validated public IP, retaining the original TLS SNI/Host name."""
    started = time.monotonic()
    parsed, port, addresses = _public_url(url, timeout)
    timeout -= time.monotonic() - started
    if timeout <= 0:
        raise FulltextError("PDF download deadline exceeded")
    if parsed.scheme == "https":
        connection = http.client.HTTPSConnection(parsed.hostname, port, timeout=timeout, context=ssl.create_default_context())
    else:
        connection = http.client.HTTPConnection(parsed.hostname, port, timeout=timeout)
    # Avoid a second DNS resolution after validation (DNS rebinding).
    connection._create_connection = lambda address, timeout=timeout, source_address=None, **kw: socket.create_connection(
        (addresses[0], port), timeout, source_address
    )
    live_socket = [None]
    def abort():
        current = live_socket[0] or connection.sock
        if current is not None:
            try:
                current.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(timeout, abort)
    timer.daemon = True
    timer.start()
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection.request("GET", path, headers={"Accept": "application/pdf", "User-Agent": "RefMind/1.0 public-paper-reader", "Accept-Encoding": "identity"})
        live_socket[0] = connection.sock
        if connection.sock is not None and not _public_address(connection.sock.getpeername()[0]):
            raise FulltextError("non-public PDF peer rejected")
        yield connection.getresponse()
    finally:
        timer.cancel()
        connection.close()


def _normalized_text(value: str) -> str:
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", value).casefold())


def inspect_pdf_identity(path: Path, candidate: PaperCandidate):
    """Conservative first-page/metadata check, not a full scientific parser."""
    import fitz
    try:
        with fitz.open(path) as document:
            if document.needs_pass or len(document) < 1:
                raise FulltextError("PDF is encrypted or contains no pages")
            metadata = document.metadata or {}
            first_page = document[0].get_text()[:12000]
            title = str(metadata.get("title") or "").strip()
            header = first_page[:2500]
            # Only explicit DOI declarations in the document header can reject identity.
            declared = re.findall(r"(?:\bdoi\s*:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[^\s<>\"]+)", header, re.I)
            dois = tuple(dict.fromkeys(v for raw in declared if (v := normalize_doi(raw.rstrip(".,;")))))
            expected_doi = normalize_doi(candidate.doi)
            if expected_doi and dois and expected_doi not in dois:
                raise FulltextIdentityError("downloaded PDF declares another DOI")
            expected = _normalized_text(candidate.title)
            title_match = len(expected) >= 20 and (expected in _normalized_text(first_page) or expected == _normalized_text(title))
            doi_match = bool(expected_doi and expected_doi in dois)
            if doi_match or title_match:
                return "matched", title, dois
            return "unknown", title, dois
    except (FulltextError, FulltextIdentityError):
        raise
    except Exception:
        raise FulltextError("downloaded bytes are not a readable PDF") from None


def fetch_public_pdf(candidate: PaperCandidate, destination_dir: str | Path, *, location_index: int = 0,
                     timeout_seconds: float = 60, max_bytes: int = 50 * 1024 * 1024,
                     expected_sha256: str | None = None) -> FetchedPaper:
    if not isinstance(candidate, PaperCandidate):
        raise ValueError("candidate must be PaperCandidate")
    if type(location_index) is not int or not 0 <= location_index < len(candidate.locations):
        raise ValueError("invalid location index")
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise ValueError("invalid download timeout")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 200 * 1024 * 1024:
        raise ValueError("invalid PDF size limit")
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("invalid expected PDF hash")
    location = candidate.locations[location_index]
    if location.is_oa is not True or not location.url:
        raise FulltextError("select a known public PDF location before fetching")
    root = Path(destination_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    current_url, original_url = location.url, location.url
    deadline = time.monotonic() + timeout_seconds
    temporary = None
    try:
        for hop in range(6):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FulltextError("PDF download deadline exceeded")
            # _open_public validates and pins the address on EVERY redirect hop.
            with _open_public(current_url, remaining) as response:
                # Closing the connection bounds even a slow-drip response's wall time.
                timer = threading.Timer(remaining, response.close)
                timer.daemon = True
                timer.start()
                try:
                    if response.status in {301, 302, 303, 307, 308}:
                        target = response.getheader("Location")
                        if not target or hop >= 5:
                            raise FulltextError("PDF redirect limit exceeded or target missing")
                        current_url = urljoin(current_url, target)
                        continue
                    if response.status != 200:
                        raise FulltextError(f"public PDF returned HTTP {response.status}")
                    content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().casefold()
                    if content_type not in {"application/pdf", "application/octet-stream", "binary/octet-stream", ""}:
                        raise FulltextError("public URL did not return a PDF")
                    encoding = (response.getheader("Content-Encoding") or "identity").casefold()
                    if encoding not in {"", "identity"}:
                        raise FulltextError("encoded PDF downloads are not accepted")
                    length = response.getheader("Content-Length")
                    if length:
                        try:
                            declared_length = int(length)
                        except ValueError:
                            raise FulltextError("invalid PDF content length") from None
                        if declared_length < 1 or declared_length > max_bytes:
                            raise FulltextError("PDF exceeds size limit")
                    digest, size, prefix = hashlib.sha256(), 0, b""
                    with tempfile.NamedTemporaryFile(prefix=".paper-", suffix=".downloading", dir=root, delete=False) as handle:
                        temporary = Path(handle.name)
                        while True:
                            if time.monotonic() >= deadline:
                                raise FulltextError("PDF download deadline exceeded")
                            block = response.read(min(65536, max_bytes - size + 1))
                            if not block:
                                break
                            if not prefix:
                                prefix = block[:1024]
                            size += len(block)
                            if size > max_bytes:
                                raise FulltextError("PDF exceeds size limit")
                            digest.update(block)
                            handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if size < 5 or not prefix.lstrip().startswith(b"%PDF-"):
                        raise FulltextError("downloaded bytes have no PDF signature")
                    if length and size != declared_length:
                        raise FulltextError("PDF download is incomplete")
                    sha256 = digest.hexdigest()
                    if expected_sha256 and sha256 != expected_sha256.casefold():
                        raise FulltextIdentityError("PDF SHA256 does not match the expected version")
                    status, title, dois = inspect_pdf_identity(temporary, candidate)
                    if expected_sha256:
                        status = "matched"
                    target = root / f"{sha256}.pdf"
                    if target.exists():
                        if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
                            raise FulltextError("existing cached PDF is not the expected regular file")
                        temporary.unlink()
                    else:
                        os.replace(temporary, target)
                    temporary = None
                    warnings = ("PDF身份尚未确认；必须人工核验后再解析或入库",) if status == "unknown" else ()
                    return FetchedPaper(target, sha256, size, original_url, current_url, location.version,
                                        location.license, status, warnings, title, dois)
                finally:
                    timer.cancel()
        raise FulltextError("PDF redirect limit exceeded")
    except (OSError, http.client.HTTPException) as exc:
        raise FulltextError("public PDF transfer failed") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["FetchedPaper", "FulltextError", "FulltextIdentityError", "fetch_public_pdf", "inspect_pdf_identity"]
