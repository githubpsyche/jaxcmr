"""Check the solver contract, sampling geometry and restart budgets."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import OptimizeResult
import jaxcmr.fitting.gradient as gradient_module
from jaxcmr.fitting import ParameterSpace, ScipyTrustConstr
from jaxcmr.helpers import make_dataset


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


class QuadraticLoss:
    """Known optimum for testing the optimizer independently of CMR."""

    def __init__(self, factory, dataset, features):
        pass

    def __call__(self, indices, base, names, parameters):
        target = jnp.array([2.0, 0.3])[:, None]
        return jnp.sum((parameters - target) ** 2, axis=0)


def fitter(**options):
    data = make_dataset(jnp.array([[1, 0]]), pres_itemnos=jnp.array([[1, 2]]))
    return ScipyTrustConstr(
        data,
        None,
        {},
        object,
        QuadraticLoss,
        {
            "bounds": {"amplitude": [0.0, 10.0], "drift": [0.1, 0.9]},
            "screening_size": 0,
            "best_of": 2,
            "seed": 27,
            "initial_params": {"amplitude": 9.0, "drift": 0.8},
            **options,
        },
    )


def test_log1p_roundtrip_and_derivative_near_zero():
    space = ParameterSpace({"amplitude": [0.0, 100.0]}, {"amplitude": "log1p"})
    physical = jnp.array([[0.0], [1e-08], [1.0], [100.0]])
    unit = jax.jit(space.to_unit)(physical)
    np.testing.assert_allclose(jax.jit(space.to_physical)(unit), physical, rtol=1e-13, atol=1e-14)
    derivative = jax.jit(jax.jacfwd(space.to_physical))(unit[1])
    np.testing.assert_allclose(derivative[0, 0], (1 + 1e-08) * np.log1p(100), rtol=1e-13)
    with pytest.raises(ValueError, match="above -1"):
        ParameterSpace({"amplitude": [-1.0, 10.0]}, {"amplitude": "log1p"})


def test_trust_solver_finds_known_optimum_and_reports_objective_gradient():
    algorithm = fitter(best_of=1, gradient_tolerance=1e-10)
    algorithm.warmup(jnp.array([True]))
    result = algorithm.fit(jnp.array([True]), subject_id=8)
    assert result["fitness"][0] < 1e-10
    np.testing.assert_allclose(
        [result["fits"]["amplitude"][0], result["fits"]["drift"][0]], [2.0, 0.3], atol=2e-05
    )
    assert result["fits"]["subject"] == [8]
    record = result["diagnostics"]["restarts"][0]
    assert record["converged"]
    assert record["constraint_violation"] == 0
    assert np.isfinite(record["projected_gradient_norm"])
    assert result["hyperparameters"]["method"] == "trust-constr"


def test_sampling_geometry_does_not_change_optimization_coordinates_or_explicit_start():
    algorithm = fitter(
        screening_size=8,
        screening_batch_size=4,
        num_steps=1,
        sampling_transforms={"amplitude": "log1p"},
    )
    result = algorithm.fit(jnp.array([True]))
    screen = result["diagnostics"]["screening"]
    bank = np.asarray(screen["candidate_unit"])
    physical = np.asarray(algorithm.sampling_space.to_physical(jnp.asarray(bank)))
    np.testing.assert_allclose(screen["scores"], ((physical - [2.0, 0.3]) ** 2).sum(axis=1))
    np.testing.assert_allclose(physical[0], [9.0, 0.8])
    assert screen["selected_indices"][0] == 0
    np.testing.assert_allclose(result["diagnostics"]["restarts"][0]["initial_unit"], [0.9, 0.875])
    assert algorithm.parameter_space.transforms == {}
    assert result["hyperparameters"]["sampling_transforms"] == {"amplitude": "log1p"}


@pytest.mark.parametrize(
    "total_limit, budgets", [(4.0, ["local", "local"]), (0.45, ["local", "global"])]
)
def test_local_deadline_allows_another_restart_and_never_credits_late_improvements(
    monkeypatch, total_limit, budgets
):
    algorithm = fitter(local_time_limit_seconds=0.25, time_limit_seconds=total_limit)
    algorithm.warmup(jnp.array([True]))
    now = [0.0]
    evaluate = algorithm._value_and_grad

    def timed_evaluation(*args):
        result = jax.block_until_ready(evaluate(*args))
        now[0] += 0.2
        return result

    monkeypatch.setattr(gradient_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(algorithm, "_value_and_grad", timed_evaluation)
    result = algorithm.fit(jnp.array([True]))
    records = result["diagnostics"]["restarts"]
    assert [record["budget"] for record in records] == budgets
    assert all((record["late_evaluations"] == 1 for record in records))
    assert all((row["seconds"] <= total_limit for row in result["diagnostics"]["history"]))
    assert result["converged"] == [False]
    parameters = jnp.array([[result["fits"]["amplitude"][0]], [result["fits"]["drift"][0]]])
    checked = QuadraticLoss(None, None, None)(None, None, None, parameters)[0]
    np.testing.assert_allclose(result["fitness"][0], checked)


def test_nonfinite_gradient_is_not_reported_as_a_valid_trust_fit(monkeypatch):
    algorithm = fitter()
    monkeypatch.setattr(
        algorithm, "_value_and_grad", lambda *args: (jnp.array(1.0), jnp.full(2, jnp.nan))
    )
    with pytest.raises(ValueError, match="No finite loss/gradient pair"):
        algorithm.fit(jnp.array([True]))


def test_restart_callbacks_keep_their_own_record_and_deadline(monkeypatch):
    """Retaining an earlier callback must not redirect it to a later restart."""
    algorithm = fitter(local_time_limit_seconds=1.0)
    algorithm.warmup(jnp.array([True]))
    now = [0.0]
    callbacks = []

    def minimize_once(evaluate, initial, **options):
        value, gradient = evaluate(initial)
        callbacks.append((evaluate, options["callback"]))
        options["callback"](initial)
        now[0] += 0.25
        return OptimizeResult(
            x=initial,
            fun=value,
            grad=gradient,
            nit=1,
            success=False,
            message="Stopped after the initial evaluation.",
            optimality=float(np.max(np.abs(gradient))),
            constr_violation=0.0,
        )

    monkeypatch.setattr(gradient_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(gradient_module, "minimize", minimize_once)
    result = algorithm.fit(jnp.array([True]))
    records = result["diagnostics"]["restarts"]
    assert len(callbacks) == 2
    callbacks[0][1](np.zeros(2))
    assert [record["nit"] for record in records] == [2, 1]
    now[0] = 1.1
    with pytest.raises(gradient_module._BudgetExpired):
        callbacks[0][0](np.array(records[0]["initial_unit"]))
    callbacks[1][0](np.array(records[1]["initial_unit"]))
    assert [record["nfev"] for record in records] == [1, 2]


@pytest.mark.parametrize(
    "options",
    [
        {"local_time_limit_seconds": 0},
        {"local_time_limit_seconds": np.inf},
        {"initial_trust_radius": -1},
        {"step_tolerance": np.nan},
        {"relative_tolerance": 1e-09},
        {"max_fun": 100},
    ],
)
def test_invalid_trust_options_fail_clearly(options):
    with pytest.raises(ValueError):
        fitter(**options)
