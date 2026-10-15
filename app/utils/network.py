"""
Network Utilities for Tamil OCR API Server.
Handles dynamic detection of local and LAN IPv4 addresses across network adapters,
startup banner formatting, and network diagnostic information.
"""

import socket
import logging
from typing import Dict, List, Optional, Tuple
import psutil

logger = logging.getLogger("tamil_ocr.network")


def is_valid_lan_ipv4(ip: str) -> bool:
    """Check if an IPv4 address is a valid, routable LAN address."""
    if not ip or not isinstance(ip, str):
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False

    # Exclude loopback
    if nums[0] == 127:
        return False
    # Exclude APIPA / Link-local (169.254.x.x)
    if nums[0] == 169 and nums[1] == 254:
        return False
    # Exclude broadcast and multicast
    if nums[0] >= 224:
        return False
    if nums[0] == 0:
        return False

    return True


def get_primary_lan_ip() -> str:
    """
    Dynamically determine the system's primary LAN IPv4 address.
    Uses outbound UDP routing probe which selects the operating system's default route.
    """
    candidate_ip = None

    # Method 1: Outbound socket routing probe (does not send packets)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if is_valid_lan_ipv4(ip):
                candidate_ip = ip
    except Exception as e:
        logger.debug(f"Routing probe failed: {e}")

    # Method 2: Inspect network interfaces via psutil
    if not candidate_ip:
        try:
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        if is_valid_lan_ipv4(ip):
                            candidate_ip = ip
                            break
                if candidate_ip:
                    break
        except Exception as e:
            logger.debug(f"psutil if_addrs inspection failed: {e}")

    # Method 3: Standard gethostbyname
    if not candidate_ip:
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if is_valid_lan_ipv4(ip):
                candidate_ip = ip
        except Exception as e:
            logger.debug(f"gethostbyname failed: {e}")

    return candidate_ip or "127.0.0.1"


def get_all_interfaces_info() -> List[Dict[str, str]]:
    """
    List all network interfaces and their IPv4 addresses,
    annotating whether each interface is recommended for LAN connections.
    """
    interfaces = []
    primary_ip = get_primary_lan_ip()

    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    is_lan = is_valid_lan_ipv4(ip)
                    is_primary = (ip == primary_ip)

                    status = "Not recommended (Link-local/APIPA)" if ip.startswith("169.254.") else (
                        "Loopback (This device only)" if ip.startswith("127.") else (
                            "Primary LAN (Recommended for mobile/web devices)" if is_primary else (
                                "Secondary LAN" if is_lan else "Other"
                            )
                        )
                    )

                    interfaces.append({
                        "interface": iface,
                        "ip": ip,
                        "netmask": addr.netmask or "N/A",
                        "is_primary": is_primary,
                        "is_lan": is_lan,
                        "status": status,
                    })
    except Exception as e:
        logger.warning(f"Could not enumerate interfaces: {e}")

    return interfaces


def get_network_summary(host: str, port: int) -> Dict:
    """Return structured network metadata for API info and startup output."""
    lan_ip = get_primary_lan_ip()
    is_all_interfaces = (host in ("0.0.0.0", "::"))

    effective_host = lan_ip if is_all_interfaces else host
    local_url = f"http://127.0.0.1:{port}"
    lan_url = f"http://{lan_ip}:{port}"

    return {
        "configured_host": host,
        "configured_port": port,
        "is_all_interfaces": is_all_interfaces,
        "primary_lan_ip": lan_ip,
        "local_url": local_url,
        "lan_url": lan_url,
        "docs_url": f"{lan_url}/docs",
        "openapi_url": f"{lan_url}/openapi.json",
        "health_url": f"{lan_url}/health",
        "interfaces": get_all_interfaces_info(),
    }


def print_startup_banner(host: str, port: int):
    """Display an informative, clear console banner at server launch."""
    summary = get_network_summary(host, port)
    primary_ip = summary["primary_lan_ip"]
    local_url = summary["local_url"]
    lan_url = summary["lan_url"]
    docs_url = summary["docs_url"]
    health_url = summary["health_url"]

    banner = f"""
================================================================================
           TAMIL OCR REST API SERVER - PRODUCTION READY
================================================================================
Engine:        ocr-tamil v0.1.4 (PARSeq Tamil + English, CRAFT Detector)
Server Host:   {host}
Server Port:   {port}
--------------------------------------------------------------------------------
ACCESSIBILITY URLS:
  * Localhost:         {local_url}
  * LAN / Wi-Fi URL:   {lan_url}
  * API Documentation: {docs_url}
  * Health Endpoint:   {health_url}

NETWORK ADAPTERS DETECTED:"""

    for iface in summary["interfaces"]:
        marker = " [PRIMARY]" if iface["is_primary"] else ""
        banner += f"\n  - {iface['interface']}: {iface['ip']}{marker} ({iface['status']})"

    banner += f"""

FOR CLIENT APPLICATIONS (React Native, Expo, Web, Desktop):
  * Use the LAN URL: {lan_url}
  * Ensure devices are on the same Wi-Fi / Local Network.
  * Windows Firewall: Run in admin PowerShell if blocked:
      New-NetFirewallRule -DisplayName "Tamil-OCR-Port-{port}" -Direction Inbound -LocalPort {port} -Protocol TCP -Action Allow
  * Linux Firewall:
      sudo ufw allow {port}/tcp
================================================================================
"""
    print(banner, flush=True)
