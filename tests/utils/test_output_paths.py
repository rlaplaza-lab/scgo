"""Tests for campaign output path helpers."""

from pathlib import Path

from scgo.utils.output_paths import (
    formula_searches_dir,
    formula_ts_results_dir,
    resolve_campaign_root,
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
