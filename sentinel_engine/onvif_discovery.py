"""
Phase 4: ONVIF camera discovery via WS-Discovery multicast.

Sends a UDP Probe to 239.255.255.250:3702 (WS-Discovery multicast address).
Devices that support ONVIF respond with their service endpoints.
We then query each endpoint for its RTSP stream URL using GetStreamUri.

No onvif library required — uses raw UDP + simple XML parsing.

Usage:
    cameras = asyncio.run(discover_onvif_cameras(timeout=5))
    for cam in cameras:
        print(cam["rtsp_url"])
"""

import asyncio
import logging
import re
import socket
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# WS-Discovery multicast address and port
WS_DISCOVERY_ADDR = "239.255.255.250"
WS_DISCOVERY_PORT = 3702

# Probe message (ONVIF WS-Discovery)
_PROBE_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope"
          xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <Header>
    <wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
      urn:uuid:{message_id}
    </wsa:MessageID>
    <wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
      urn:schemas-xmlsoap-org:ws:2005:04:discovery
    </wsa:To>
    <wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
      http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe
    </wsa:Action>
  </Header>
  <Body>
    <dn:Probe>
      <dn:Types>dn:NetworkVideoTransmitter</dn:Types>
    </dn:Probe>
  </Body>
</Envelope>"""

# GetStreamUri SOAP request
_GET_STREAM_URI_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope"
          xmlns:media="http://www.onvif.org/ver10/media/wsdl"
          xmlns:tt="http://www.onvif.org/ver10/schema">
  <Body>
    <media:GetStreamUri>
      <media:StreamSetup>
        <tt:Stream>RTP-Unicast</tt:Stream>
        <tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>
      </media:StreamSetup>
      <media:ProfileToken>{profile_token}</media:ProfileToken>
    </media:GetStreamUri>
  </Body>
</Envelope>"""

_GET_PROFILES_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope"
          xmlns:media="http://www.onvif.org/ver10/media/wsdl">
  <Body>
    <media:GetProfiles/>
  </Body>
</Envelope>"""


def _extract_tag(xml: str, tag: str) -> Optional[str]:
    """Extract first text content of a tag from XML string."""
    pattern = rf"<[^>]*{re.escape(tag)}[^>]*>(.*?)</"
    m = re.search(pattern, xml, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_all(xml: str, tag: str) -> list[str]:
    pattern = rf"<[^>]*{re.escape(tag)}[^>]*>(.*?)</"
    return [m.group(1).strip() for m in re.finditer(pattern, xml, re.DOTALL)]


def _extract_xaddrs(xml: str) -> list[str]:
    """Extract XAddrs (service endpoints) from WS-Discovery ProbeMatch."""
    addrs = _extract_tag(xml, "XAddrs") or ""
    return [a for a in addrs.split() if a.startswith("http")]


async def _probe_udp(timeout: float) -> list[str]:
    """
    Send WS-Discovery Probe and collect ProbeMatch responses.
    Returns list of XAddrs (device service endpoint URLs).
    """
    message_id = str(uuid.uuid4())
    probe = _PROBE_TEMPLATE.format(message_id=message_id).encode()

    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.setblocking(False)

    endpoints: list[str] = []
    try:
        sock.sendto(probe, (WS_DISCOVERY_ADDR, WS_DISCOVERY_PORT))

        deadline = loop.time() + timeout
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, 65535), timeout=remaining
                )
                xml = data.decode(errors="replace")
                for addr in _extract_xaddrs(xml):
                    if addr not in endpoints:
                        endpoints.append(addr)
                        logger.debug(f"ONVIF: found device at {addr}")
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.debug(f"ONVIF UDP recv error: {e}")
    finally:
        sock.close()

    return endpoints


async def _get_rtsp_url(
    endpoint: str,
    username: str = "",
    password: str = "",
    timeout: float = 5.0,
) -> Optional[str]:
    """
    Query an ONVIF device endpoint for its primary RTSP stream URL.
    Uses GetProfiles then GetStreamUri.
    """
    import aiohttp

    headers = {"Content-Type": "application/soap+xml; charset=utf-8"}
    auth = aiohttp.BasicAuth(username, password) if username else None

    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            # 1. Get profiles
            async with session.post(
                endpoint,
                data=_GET_PROFILES_TEMPLATE.encode(),
                headers=headers,
                auth=auth,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                profiles_xml = await resp.text()

            tokens = _extract_all(profiles_xml, "token")
            if not tokens:
                logger.debug(f"ONVIF: no profiles at {endpoint}")
                return None

            profile_token = tokens[0]

            # 2. Get stream URI for first profile
            body = _GET_STREAM_URI_TEMPLATE.format(profile_token=profile_token)
            async with session.post(
                endpoint,
                data=body.encode(),
                headers=headers,
                auth=auth,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                stream_xml = await resp.text()

            rtsp_url = _extract_tag(stream_xml, "Uri")
            if rtsp_url:
                logger.info(f"ONVIF: stream URI at {endpoint}: {rtsp_url}")
                return rtsp_url

    except Exception as e:
        logger.debug(f"ONVIF: failed to get stream URI from {endpoint}: {e}")

    return None


async def discover_onvif_cameras(
    timeout: float = 5.0,
    username: str = "",
    password: str = "",
) -> list[dict]:
    """
    Discover ONVIF cameras on the local network.

    Args:
        timeout: UDP probe listen window in seconds
        username: ONVIF device username (optional for discovery, required for stream URI)
        password: ONVIF device password

    Returns:
        List of dicts: [{endpoint, rtsp_url, host, path}]
    """
    logger.info(f"Starting ONVIF WS-Discovery probe (timeout={timeout}s)...")
    endpoints = await _probe_udp(timeout=timeout)
    logger.info(f"Found {len(endpoints)} ONVIF device(s)")

    cameras = []
    tasks = [_get_rtsp_url(ep, username, password) for ep in endpoints]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for endpoint, result in zip(endpoints, results):
        if isinstance(result, Exception) or not result:
            cameras.append({
                "endpoint": endpoint,
                "rtsp_url": None,
                "host": _host_from_url(endpoint),
                "path": None,
                "reachable": False,
            })
        else:
            host, path = _parse_rtsp(result)
            cameras.append({
                "endpoint": endpoint,
                "rtsp_url": result,
                "host": host,
                "path": path,
                "reachable": True,
            })

    return cameras


def _host_from_url(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else url


def _parse_rtsp(rtsp_url: str) -> tuple[str, str]:
    """Split rtsp://host:port/path into (host:port, /path)."""
    m = re.match(r"rtsp://([^/]+)(/.*)$", rtsp_url)
    if m:
        return m.group(1), m.group(2)
    return rtsp_url, "/"
