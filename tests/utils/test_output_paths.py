"""Tests for campaign output path helpers."""

from pathlib import Path

from scgo.utils.output_paths import (
    formula_searches_dir,
    formula_ts_results_dir,
    resolve_campaign_root,
    resolve_go_campaign_searches_dir,
    resolve_go_searches_dir,
    resolve_go_ts_pipeline_paths,
    resolve_minima_dir,
    resolve_ts_campaign_paths,
)


def test_formula_dirs():
    root = Path("/tmp/campaign")
    assert formula_searches_dir(root, "Pt5") == Path("/tmp/campaign/Pt5_searches")
    assert formula_ts_results_dir(root, "Pt5") == Path("/tmp/campaign/Pt5_ts_results")


def test_resolve_campaign_root_from_searches_path(tmp_path):
    searches = tmp_path / "Pt5_searches"
    searches.mkdir()
    assert resolve_campaign_root(searches) == tmp_path.resolve()


def test_resolve_ts_campaign_paths_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    campaign, minima, ts = resolve_ts_campaign_paths(None, "Pt5")
    assert campaign == tmp_path.resolve()
    assert minima == tmp_path / "Pt5_searches"
    assert ts == tmp_path / "Pt5_ts_results"


def test_resolve_ts_campaign_paths_explicit_searches(tmp_path):
    campaign_root = tmp_path / "pt5_gas_mace"
    searches = campaign_root / "Pt5_searches"
    searches.mkdir(parents=True)
    campaign, minima, ts = resolve_ts_campaign_paths(
        campaign_root,
        "Pt5",
        searches_dir=searches,
    )
    assert campaign == campaign_root.resolve()
    assert minima == searches.resolve()
    assert ts == (campaign_root / "Pt5_ts_results").resolve()


def test_resolve_ts_campaign_paths_output_dir_is_searches(tmp_path):
    campaign_root = tmp_path / "pt5_gas_mace"
    searches = campaign_root / "Pt5_searches"
    searches.mkdir(parents=True)
    campaign, minima, ts = resolve_ts_campaign_paths(searches, "Pt5")
    assert campaign == campaign_root.resolve()
    assert minima == searches.resolve()
    assert ts == (campaign_root / "Pt5_ts_results").resolve()


def test_resolve_minima_dir_override(tmp_path):
    searches = tmp_path / "custom" / "Pt5_searches"
    searches.mkdir(parents=True)
    got = resolve_minima_dir(tmp_path / "custom", "Pt5", searches_dir=searches)
    assert got == searches.resolve()


def test_resolve_go_searches_dir_explicit(tmp_path):
    explicit = tmp_path / "custom_searches"
    assert resolve_go_searches_dir(explicit, "Pt5") == explicit.resolve()


def test_resolve_go_searches_dir_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_go_searches_dir(None, "Pt5") == (tmp_path / "Pt5_searches").resolve()


def test_resolve_go_campaign_searches_dir(tmp_path):
    parent = tmp_path / "benchmark" / "results"
    parent.mkdir(parents=True)
    got = resolve_go_campaign_searches_dir(parent, "Pt5")
    assert got == parent / "Pt5_searches"


def test_resolve_go_campaign_searches_dir_none():
    assert resolve_go_campaign_searches_dir(None, "Pt5") is None


def test_resolve_go_ts_pipeline_paths(tmp_path):
    campaign = tmp_path / "Pt5_campaign"
    searches, ts = resolve_go_ts_pipeline_paths(campaign, "Pt5")
    assert searches == campaign / "Pt5_searches"
    assert ts == campaign / "Pt5_ts_results"


def test_resolve_campaign_root_none_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_campaign_root(None) == tmp_path.resolve()


def test_resolve_campaign_root_plain_campaign_path(tmp_path):
    campaign = tmp_path / "benchmark" / "results"
    campaign.mkdir(parents=True)
    assert resolve_campaign_root(campaign) == campaign.resolve()


def test_resolve_ts_campaign_paths_from_campaign_root(tmp_path):
    campaign_root = tmp_path / "pt5_mace_mace_matpes_0"
    campaign_root.mkdir(parents=True)
    campaign, minima, ts = resolve_ts_campaign_paths(campaign_root, "Pt5")
    assert campaign == campaign_root.resolve()
    assert minima == campaign_root / "Pt5_searches"
    assert ts == campaign_root / "Pt5_ts_results"
