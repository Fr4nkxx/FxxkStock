"""Tests for CNINFO announcement fetcher."""

import copy
import json
from unittest.mock import patch

import pytest

import fxxkstock.default_config as default_config
from fxxkstock.dataflows.cninfo import (
    _load_cn_name_map,
    _load_orgid_map,
    _normalize_org_id,
    _query_announcements,
    _resolve_org_id_via_browser,
    fetch_cninfo_announcements,
    get_cninfo_insider,
)
from fxxkstock.dataflows.config import set_config
from fxxkstock.dataflows.errors import NoMarketDataError


@pytest.mark.unit
def test_fetch_cninfo_announcements(tmp_path):
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"data_cache_dir": str(tmp_path), "market_region": "cn_a"})

    fake_anns = [
        {
            "announcementTitle": "2025年第一次临时股东大会决议公告",
            "announcementTime": 1717200000000,
            "adjunctUrl": "/finalpage/2025-06-01/1200000001.PDF",
        }
    ]

    with patch(
        "fxxkstock.dataflows.cninfo._query_announcements",
        return_value=fake_anns,
    ):
        out = fetch_cninfo_announcements("600519.SS", "2025-05-25", "2025-06-05")

    assert "股东大会" in out
    assert "cninfo.com.cn" in out


@pytest.mark.unit
def test_get_cninfo_insider_raises_for_hk():
    set_config({"market_region": "cn_hk"})
    with pytest.raises(NoMarketDataError):
        get_cninfo_insider("0700.HK")


@pytest.mark.unit
def test_load_orgid_map_from_cache(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "cninfo_cache_ttl_hours": 24})
    cache = tmp_path / "cninfo" / "orgid_map.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"600519": "gssh0600519"}), encoding="utf-8")

    mapping = _load_orgid_map()
    assert mapping["600519"] == "gssh0600519"


@pytest.mark.unit
def test_load_cn_name_map_refreshes_when_name_cache_missing(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "cninfo_cache_ttl_hours": 24})
    cn_dir = tmp_path / "cninfo"
    cn_dir.mkdir(parents=True)
    (cn_dir / "orgid_map.json").write_text(
        json.dumps({"603629": "gssh0603629"}), encoding="utf-8"
    )

    fake_items = [{"code": "603629", "orgId": "gssh0603629", "zwjc": "利通电子"}]

    with patch(
        "fxxkstock.dataflows.cninfo._http_get_json",
        return_value={"stockList": fake_items},
    ):
        mapping = _load_cn_name_map()

    assert mapping["603629"] == "利通电子"
    assert (cn_dir / "cn_name_map.json").exists()


@pytest.mark.unit
def test_normalize_org_id_does_not_duplicate_exchange_prefix():
    assert _normalize_org_id("gssh0600667", "gssh") == "gssh0600667"
    assert _normalize_org_id("gssz0000001", "gssz") == "gssz0000001"
    assert _normalize_org_id("jjjl0000041", "gssz") == "jjjl0000041"
    assert _normalize_org_id("9900023660", "gssh") == "9900023660"
    assert _normalize_org_id("0600667", "gssh") == "gssh0600667"


@pytest.mark.unit
def test_query_announcements_sends_normalized_org_id():
    captured = {}

    def fake_post(url, payload):
        captured.update(payload)
        return {"announcements": [{"announcementTitle": "测试公告"}]}

    with (
        patch(
            "fxxkstock.dataflows.cninfo._resolve_org_id",
            return_value="gssh0600667",
        ),
        patch("fxxkstock.dataflows.cninfo._http_post", side_effect=fake_post),
    ):
        announcements = _query_announcements(
            "600667",
            "2026-06-01",
            "2026-06-30",
        )

    assert captured["stock"] == "600667,gssh0600667"
    assert announcements[0]["announcementTitle"] == "测试公告"


@pytest.mark.unit
def test_query_announcements_preserves_numeric_institution_org_id():
    captured = {}

    def fake_post(url, payload):
        captured.update(payload)
        return {"announcements": [{"announcementTitle": "火炬电子公告"}]}

    with (
        patch(
            "fxxkstock.dataflows.cninfo._resolve_org_id",
            return_value="9900023660",
        ),
        patch("fxxkstock.dataflows.cninfo._http_post", side_effect=fake_post),
    ):
        _query_announcements("603678", "2026-06-23", "2026-06-30")

    assert captured["stock"] == "603678,9900023660"


@pytest.mark.unit
def test_query_announcements_uses_fund_column_for_etf_org_id():
    captured = {}

    def fake_post(url, payload):
        captured.update(payload)
        return {"announcements": [{"announcementTitle": "ETF公告"}]}

    with (
        patch(
            "fxxkstock.dataflows.cninfo._resolve_org_id",
            return_value="jjjl0000041",
        ),
        patch("fxxkstock.dataflows.cninfo._http_post", side_effect=fake_post),
    ):
        _query_announcements("159819", "2026-06-01", "2026-06-30")

    assert captured["stock"] == "159819,jjjl0000041"
    assert captured["column"] == "fund"
    assert captured["plate"] == ""


@pytest.mark.unit
def test_query_announcements_uses_browser_etf_org_id_fallback():
    captured = {}

    def fake_post(url, payload):
        captured.update(payload)
        return {"announcements": [{"announcementTitle": "ETF公告"}]}

    with (
        patch("fxxkstock.dataflows.cninfo._resolve_org_id", return_value=None),
        patch(
            "fxxkstock.dataflows.cninfo._resolve_org_id_via_browser",
            return_value="jjjl0000041",
        ),
        patch("fxxkstock.dataflows.cninfo._http_post", side_effect=fake_post),
    ):
        _query_announcements("159819", "2026-06-01", "2026-06-30")

    assert captured["stock"] == "159819,jjjl0000041"
    assert captured["column"] == "fund"
