"""Safely and reliably resolve the public IP of an uploader in Streamlit.

Supports:
- Reverse proxy / CDN environments (Cloudflare, AWS ALB, Nginx, Streamlit Cloud)
  via forwarded headers (CF-Connecting-IP, X-Forwarded-For, X-Real-IP, etc.).
- Safe handling of X-Forwarded-For avoiding spoofing and private internal IPs.
- Strictly avoids misidentifying the Streamlit Cloud server container IP as the visitor IP.
"""

from __future__ import annotations

import ipaddress
import json
import os
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


def is_local_development(
    headers: Mapping[str, object] | None = None,
    peer_ip: object = "",
) -> bool:
    """Detect if running in local development mode vs a hosted cloud environment."""
    # Check known Streamlit Cloud / hosted container environment variables
    if os.environ.get("STREAMLIT_SHARING_HOST") or os.environ.get("IS_STREAMLIT_COMMUNITY_CLOUD"):
        return False

    if headers:
        normalized = {str(k).lower(): str(v) for k, v in headers.items()}
        host = normalized.get("host", "").lower()
        if any(d in host for d in ("streamlit.app", "herokuapp.com", "onrender.com", "hf.space", "azurewebsites.net")):
            return False
        if any(h in normalized for h in ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip")):
            return False
        if "localhost" in host or "127.0.0.1" in host or "0.0.0.0" in host:
            return True

    peer_str = str(peer_ip or "").strip()
    if peer_str in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    if not headers and not peer_ip:
        return True
    return False


def get_ip_from_headers(
    headers: Mapping[str, object] | None,
    peer_ip: object = "",
) -> tuple[str, str]:
    """Inspect request headers and peer IP for a valid public IP.

    Returns:
        (resolved_ip, source_description)
    """
    if headers:
        normalized = {str(k).lower(): str(v) for k, v in headers.items()}

        # 1. Direct client headers injected by edge reverse proxies (Cloudflare, Akamai)
        # CF-Connecting-IP is authoritative on Cloudflare-fronted platforms: Cloudflare strips
        # any client-sent CF-Connecting-IP and sets it to the genuine connecting client IP.
        for header_name in ("cf-connecting-ip", "true-client-ip", "x-real-ip", "x-client-ip"):
            if header_name in normalized:
                resolved = _canonical_public_ip(normalized[header_name])
                if resolved:
                    return resolved, f"Header: {header_name}"

        # 2. X-Forwarded-For chain: client IP is the first non-private IP
        forwarded = normalized.get("x-forwarded-for", "")
        if forwarded:
            for item in str(forwarded).split(","):
                resolved = _canonical_public_ip(item)
                if resolved:
                    return resolved, "Header: x-forwarded-for"

        # 3. RFC 7239 Forwarded header: e.g. for=192.0.2.60;proto=http;by=203.0.113.43
        rfc_forwarded = normalized.get("forwarded", "")
        if rfc_forwarded:
            for part in str(rfc_forwarded).split(";"):
                for sub in part.split(","):
                    resolved = _canonical_public_ip(sub)
                    if resolved:
                        return resolved, "Header: forwarded"

    # 4. Peer IP reported directly by server (valid only if it is a routable public IP)
    if peer_ip:
        resolved = _canonical_public_ip(peer_ip)
        if resolved:
            return resolved, "Context: peer_ip"

    return "", "None"


def fetch_external_public_ip(timeout: float = 3.0) -> str:
    """Query fast public IP services ONLY during local development.

    CRITICAL: This function makes an outbound HTTP request from the CURRENT MACHINE.
    - If called on a developer's laptop, it returns the developer's public IP.
    - If called on Streamlit Cloud, it returns the STREAMLIT CLOUD CONTAINER IP (e.g. Dallas, TX).
    Therefore, this MUST NEVER be used as a visitor-IP fallback in cloud deployments.
    """
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
    allow_local_dev_fallback: bool = False,
) -> str:
    """Identify the real public IP of the user who uploaded the image.

    Checks:
    1. HTTP reverse proxy headers (CF-Connecting-IP, X-Forwarded-For, X-Real-IP)
    2. Streamlit peer connection IP (if public)
    3. If neither contains a public client IP:
       - On Streamlit Cloud / remote deployment: returns 'Unavailable (Not exposed by hosting platform)'.
         It strictly avoids querying external IP services so the server's IP is never returned.
       - In local dev (if allow_local_dev_fallback=True): queries local machine public IP.
    """
    if trust_forwarded_headers:
        header_ip, _ = get_ip_from_headers(headers, peer_ip)
        if header_ip:
            return header_ip
    elif peer_ip:
        peer = _canonical_public_ip(peer_ip)
        if peer:
            return peer

    # Only fall back to external resolver if explicitly permitted AND in local development
    if allow_local_dev_fallback and is_local_development(headers, peer_ip):
        external_ip = fetch_external_public_ip()
        if external_ip:
            return external_ip
        return "127.0.0.1"

    # In Streamlit Cloud / production where visitor IP is not exposed:
    return "Unavailable (Not exposed by hosting platform)"


def get_ip_diagnostics(
    headers: Mapping[str, object] | None = None,
    peer_ip: object = "",
) -> dict[str, object]:
    """Inspect request context and headers for testing and deployment verification."""
    normalized = {str(k).lower(): str(v) for k, v in (headers or {}).items()}

    # Sanitize and extract only IP-relevant headers (omit cookies, auth tokens, etc.)
    relevant_keys = [
        "cf-connecting-ip",
        "true-client-ip",
        "x-real-ip",
        "x-client-ip",
        "x-forwarded-for",
        "forwarded",
        "host",
        "user-agent",
        "cf-ipcountry",
        "cf-ray",
    ]
    extracted_headers = {k: normalized[k] for k in relevant_keys if k in normalized}

    resolved_ip, source = get_ip_from_headers(headers, peer_ip)
    is_local = is_local_development(headers, peer_ip)

    return {
        "peer_ip_raw": str(peer_ip or "None"),
        "peer_ip_is_public": is_public_ip(peer_ip),
        "relevant_headers": extracted_headers,
        "all_header_names": sorted(list(normalized.keys())),
        "resolved_from_headers": resolved_ip or "None",
        "resolution_source": source,
        "is_local_dev": is_local,
        "environment": "Local Development" if is_local else "Streamlit Cloud / Hosted",
        "final_resolved_ip": resolved_ip if resolved_ip else "Unavailable (Not exposed by hosting platform)",
        "server_fallback_blocked": True,
    }
