from ase.calculators.emt import EMT

from scgo.algorithms import ga_go
from tests.test_utils import MockRelaxer


def test_ga_go_offspring_logging_levels(tmp_path, rng, capfd):
    """Verbosity 1 shows phase summaries; verbosity 2 adds per-offspring detail."""
    from scgo.utils.logging import configure_logging

    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    outdir_v1 = tmp_path / "ga_go_log_v1"

    configure_logging(1)
    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(outdir_v1),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        rng=rng,
        verbosity=1,
    )
    assert isinstance(minima, list)
    captured_v1 = capfd.readouterr().out
    assert "Population initialization" in captured_v1
    assert "--- Generation 0 ---" in captured_v1
    assert "Crossover:" in captured_v1
    assert "Mutation:" in captured_v1
    assert "Offspring:" in captured_v1
    assert "Relaxation:" in captured_v1
    assert "Offspring rejected by system_type validation" not in captured_v1
    assert "Selected parents for pairing" not in captured_v1
    assert "Pairing produced child" not in captured_v1

    configure_logging(2)
    outdir_v2 = tmp_path / "ga_go_log_v2"
    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(outdir_v2),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        rng=rng,
        verbosity=2,
    )
    assert isinstance(minima, list)
    captured_v2 = capfd.readouterr().out
    assert "--- Generation 0 ---" in captured_v2
    assert "Crossover:" in captured_v2
    assert "Offspring 1:" in captured_v2 or "Offspring 2:" in captured_v2
    assert "Generation 0 offspring loop: n_offspring=" not in captured_v2
    assert "Offspring rejected by system_type validation" not in captured_v2
