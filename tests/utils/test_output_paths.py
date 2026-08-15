"""Tests for campaign output path helpers."""

from pathlib import Path

from scgo.utils.output_paths import (
    calculator_slug_from_go_params,
    formula_searches_dir,
    formula_ts_results_dir,
    resolve_campaign_root_from_args,
    resolve_go_campaign_searches_dir,
    resolve_go_searches_dir,
    resolve_go_ts_pipeline_paths,
    resolve_ts_campaign_paths,
)


def test_formula_dirs():
    root = Path("/tmp/campaign")
    assert formula_searches_dir(root, "Pt5") == Path("/tmp/campaign/Pt5_searches")
    assert formula_ts_results_dir(root, "Pt5") == Path("/tmp/campaign/Pt5_ts_results")


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


def test_resolve_ts_campaign_paths_from_campaign_root(tmp_path):
    campaign_root = tmp_path / "pt5_mace_mace_matpes_0"
    campaign_root.mkdir(parents=True)
    campaign, minima, ts = resolve_ts_campaign_paths(campaign_root, "Pt5")
    assert campaign == campaign_root.resolve()
    assert minima == campaign_root / "Pt5_searches"
    assert ts == campaign_root / "Pt5_ts_results"


# --------------------------------------------------------------------------
# Single campaign-root model
# --------------------------------------------------------------------------


def test_slug_from_go_params():
    assert calculator_slug_from_go_params({"calculator": "MACE"}) == "mace"
    assert calculator_slug_from_go_params({"calculator": "mace"}) == "mace"
    assert calculator_slug_from_go_params({"calculator": "UMA"}) == "uma"
    assert calculator_slug_from_go_params({"calculator": "Other"}) == "other"
    # Unknown calculators keep their lowercase slug, not a "calc" fallback.
    assert calculator_slug_from_go_params({"calculator": "EMT"}) == "emt"
    # Default when no calculator is given.
    assert calculator_slug_from_go_params(None) == "mace"
    assert calculator_slug_from_go_params({}) == "mace"


def test_root_from_searches_infers_parent(tmp_path):
    searches = tmp_path / "campaign" / "Pt5_searches"
    searches.mkdir(parents=True)
    got = resolve_campaign_root_from_args(searches, path_key="Pt5")
    assert got == (tmp_path / "campaign").resolve()


def test_root_from_ts_results_infers_parent(tmp_path):
    ts_results = tmp_path / "campaign" / "Pt5_ts_results"
    ts_results.mkdir(parents=True)
    got = resolve_campaign_root_from_args(ts_results, path_key="Pt5")
    assert got == (tmp_path / "campaign").resolve()


def test_root_from_plain_output_dir_is_itself(tmp_path):
    campaign = tmp_path / "benchmark" / "results"
    campaign.mkdir(parents=True)
    assert (
        resolve_campaign_root_from_args(campaign, path_key="Pt5") == campaign.resolve()
    )


def test_root_from_output_root_and_stem(tmp_path):
    got = resolve_campaign_root_from_args(
        None,
        output_root=tmp_path / "results",
        output_stem="pt5_gas",
        path_key="Pt5",
        calc_slug="mace",
    )
    assert got == (tmp_path / "results" / "pt5_gas_mace").resolve()


def test_root_from_output_root_defaults_stem_to_path_key(tmp_path):
    got = resolve_campaign_root_from_args(
        None,
        output_root=tmp_path / "results",
        path_key="Pt5",
        calc_slug="uma",
    )
    assert got == (tmp_path / "results" / "Pt5_uma").resolve()


def test_root_from_output_stem_only_uses_scgo_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = resolve_campaign_root_from_args(
        None,
        output_stem="pt5_gas",
        path_key="Pt5",
        calc_slug="mace",
    )
    assert got == (tmp_path / "scgo_runs" / "pt5_gas_mace").resolve()


def test_root_without_any_args_is_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_campaign_root_from_args(None, path_key="Pt5") == tmp_path.resolve()


def test_campaign_root_equivalence(tmp_path):
    """All six runners resolve the same sibling layout from the same campaign root.

    ``run_go`` / ``run_go_campaign`` pass a ``*_searches`` dir or the root, GO+TS
    runners pass the root, and TS runners pass the root; every one must land on
    ``{root}/{path_key}_searches`` + ``{root}/{path_key}_ts_results``. This
    includes ``run_go_ts_campaign``, which no longer inserts a
    ``{path_key}_campaign/`` wrapper.
    """
    root = (tmp_path / "pt5_gas_mace").resolve()
    root.mkdir(parents=True)
    path_key = "Pt5"
    expected_searches = root / f"{path_key}_searches"
    expected_ts = root / f"{path_key}_ts_results"

    # run_go: output_dir is the searches dir itself -> root is its parent.
    run_go_root = resolve_campaign_root_from_args(expected_searches, path_key=path_key)

    # run_go_campaign: output_dir is the campaign parent (the root).
    run_go_campaign_root = resolve_campaign_root_from_args(root, path_key=path_key)

    # run_go_ts: campaign root, either explicit or built from root/stem.
    run_go_ts_root = resolve_campaign_root_from_args(root, path_key=path_key)
    run_go_ts_default_root = resolve_campaign_root_from_args(
        None,
        output_root=tmp_path,
        output_stem="pt5_gas",
        path_key=path_key,
        calc_slug="mace",
    )

    # run_go_ts_campaign: sibling shape (no `_campaign` wrapper).
    run_go_ts_campaign_root = resolve_campaign_root_from_args(root, path_key=path_key)

    # run_ts_search / run_ts_campaign: campaign root passed through unchanged.
    ts_search_root, ts_search_minima, ts_search_results = resolve_ts_campaign_paths(
        root, path_key
    )
    run_ts_campaign_root = resolve_campaign_root_from_args(root, path_key=path_key)

    roots = [
        run_go_root,
        run_go_campaign_root,
        run_go_ts_root,
        run_go_ts_default_root,
        run_go_ts_campaign_root,
        ts_search_root,
        run_ts_campaign_root,
    ]
    assert all(r == root for r in roots), roots

    for r in roots:
        assert formula_searches_dir(r, path_key) == expected_searches
        assert formula_ts_results_dir(r, path_key) == expected_ts

    assert ts_search_minima == expected_searches
    assert ts_search_results == expected_ts
