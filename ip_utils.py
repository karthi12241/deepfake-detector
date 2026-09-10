"""Safely and reliably resolve the public IP of an uploader in Streamlit.

Supports:
- Reverse proxy / CDN environments (Cloudflare, AWS, Nginx, Streamlit Cloud)
  via forwarded headers (X-Forwarded-For, CF-Connecting-IP, X-Real-IP, etc.).
- Direct / local development runs (localhost, LAN) via external public IP
  discovery services (ipify, ifconfig.me, my-ip.io).
"""

from __future__ import annotations

import ipaddress
import json
import urllib.request
from collections.abc import Mapping


def is_public_ip(value: object) -> bool:
    """Return True if candidate string represents a valid public/routable IP."""
    if not value:
        return False
    candidate = str(value).strip().strip('"').strip("'").strip("[]")
    if candidate.lower().startswith("for="):
        candidate = candidate[4:].strip().strip('"').strip("'").strip("[]")
    # Strip port if present (e.g. 203.0.113.195:443)
    if ":" in candidate and "." in candidate:
        candidate = candidate.split(":")[0].strip()
    try:
        addr = ipaddress.ip_address(candidate)
        return addr.is_global and not (
            addr.is_private
            or addr.is_loopback
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_link_local
        )
    except ValueError:
        return False


def _canonical_public_ip(value: object) -> str:
    """Return canonical public IP string, or empty string."""
    if is_public_ip(value):
        candidate = str(value).strip().strip('"').strip("'").strip("[]")
        if candidate.lower().startswith("for="):
            candidate = candidate[4:].strip().strip('"').strip("'").strip("[]")
        if ":" in candidate and "." in candidate:
            candidate = candidate.split(":")[0].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            return ""
    return ""


def get_ip_from_headers(
    headers: Mapping[str, object] | None,
    peer_ip: object = "",
) -> str:
    """Inspect request headers and peer IP for a valid public IP."""
    if headers:
        normalized = {str(k).lower(): str(v) for k, v in headers.items()}

        # 1. Direct client headers injected by edge reverse proxies/CDNs
        for header_name in ("cf-connecting-ip", "true-client-ip", "x-real-ip", "x-client-ip"):
            if header_name in normalized:
                resolved = _canonical_public_ip(normalized[header_name])
                if resolved:
                    return resolved

        # 2. X-Forwarded-For chain: client IP is the first non-private IP
        forwarded = normalized.get("x-forwarded-for", "")
        if forwarded:
            for item in str(forwarded).split(","):
                resolved = _canonical_public_ip(item)
                if resolved:
                    return resolved

        # 3. RFC 7239 Forwarded header: e.g. for=192.0.2.60;proto=http;by=203.0.113.43
        rfc_forwarded = normalized.get("forwarded", "")
        if rfc_forwarded:
            for part in str(rfc_forwarded).split(";"):
                for sub in part.split(","):
                    resolved = _canonical_public_ip(sub)
                    if resolved:
                        return resolved

    # 4. Peer IP reported directly by server
    if peer_ip:
        resolved = _canonical_public_ip(peer_ip)
        if resolved:
            return resolved

    return ""


def fetch_external_public_ip(timeout: float = 3.0) -> str:
    """Query fast public IP services when running locally or on private network."""
    services = [
        "https://api.ipify.org?format=json",
        "https://api.my-ip.io/v2/ip.json",
        "https://ifconfig.me/ip",
        "https://ipapi.co/ip/",
    ]
    for url in services:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "deepfake-detector/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode("utf-8", errors="ignore").strip()
                if "{" in content:
                    try:
                        data = json.loads(content)
                        cand = data.get("ip") or data.get("ip_address") or ""
                    except Exception:
                        cand = ""
                else:
                    cand = content
                resolved = _canonical_public_ip(cand)
                if resolved:
                    return resolved
        except Exception:
            continue
    return ""


def get_uploader_ip(
    headers: Mapping[str, object] | None = None,
    peer_ip: object = "",
    *,
    trust_forwarded_headers: bool = True,
) -> str:
    """Identify the real public IP of the user who uploaded the image.

    Checks:
    1. HTTP headers (X-Forwarded-For, CF-Connecting-IP, X-Real-IP, etc.)
    2. Streamlit peer connection IP
    3. External public IP resolver (for local dev / localhost where client is the host)
    """
    if trust_forwarded_headers:
        header_ip = get_ip_from_headers(headers, peer_ip)
        if header_ip:
            return header_ip
    elif peer_ip:
        peer = _canonical_public_ip(peer_ip)
        if peer:
            return peer

    # Fallback to external resolver (handles localhost and local execution)
    external_ip = fetch_external_public_ip()
    if external_ip:
        return external_ip

    return "127.0.0.1"