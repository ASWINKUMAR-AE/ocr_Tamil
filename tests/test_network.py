"""
Tests for Network IP utilities and startup diagnostics.
"""

from app.utils.network import is_valid_lan_ipv4, get_primary_lan_ip, get_network_summary


def test_is_valid_lan_ipv4():
    """Test classification of LAN vs loopback/link-local IPv4 addresses."""
    # Valid private / LAN IPs
    assert is_valid_lan_ipv4("192.168.1.100") is True
    assert is_valid_lan_ipv4("10.0.0.15") is True
    assert is_valid_lan_ipv4("172.16.0.5") is True

    # Invalid / Loopback / APIPA
    assert is_valid_lan_ipv4("127.0.0.1") is False
    assert is_valid_lan_ipv4("169.254.1.1") is False
    assert is_valid_lan_ipv4("") is False
    assert is_valid_lan_ipv4("invalid.ip") is False
    assert is_valid_lan_ipv4("256.0.0.1") is False


def test_get_primary_lan_ip():
    """Test dynamic detection returns a valid IPv4 address."""
    ip = get_primary_lan_ip()
    assert isinstance(ip, str)
    parts = ip.split(".")
    assert len(parts) == 4
    for p in parts:
        assert 0 <= int(p) <= 255


def test_get_network_summary():
    """Test network summary returns expected keys and URLs."""
    summary = get_network_summary("0.0.0.0", 8000)
    assert summary["configured_host"] == "0.0.0.0"
    assert summary["configured_port"] == 8000
    assert summary["is_all_interfaces"] is True
    assert "http://127.0.0.1:8000" in summary["local_url"]
    assert ":8000" in summary["lan_url"]
    assert ":8000/docs" in summary["docs_url"]
    assert ":8000/health" in summary["health_url"]
    assert isinstance(summary["interfaces"], list)
