"""Joint analysis losses: scoring semantics, shared simulations, and JAX use."""

from collections.abc import Mapping
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.loss import mse
from jaxcmr.typing import Array, Float_, Int_, RecallDataset


class FixedRecallModel(Pytree):
    """Recall one designated item, so expected analysis errors are exact."""

    def __init__(self, choice: Array):
        self.item_count = 3
        self.is_active = jnp.asarray(True)
        self.studied = jnp.ones(3, dtype=bool)
        self.recallable = self.studied
        self.recall_total = jnp.asarray(0)
        self.study_index = jnp.asarray(3)
        self.probabilities = jax.nn.one_hot(choice, 4)

    def experience(self, choice: Int_) -> "FixedRecallModel":
        return self

    def start_retrieving(self) -> "FixedRecallModel":
        return self

    def retrieve(self, choice: Int_) -> "FixedRecallModel":
        return self.replace(is_active=jnp.asarray(False), recall_total=jnp.asarray(1))

    def activations(self) -> Array:
        return self.probabilities[1:]

    def outcome_probabilities(self) -> Array:
        return self.probabilities

    def outcome_probability(self, choice: Int_) -> Array:
        return self.probabilities[choice]


class FixedRecallFactory:
    def __init__(self, dataset: RecallDataset, features):
        self.choices = dataset["choice"][:, 0]

    def create_trial_model(self, trial_index: Array, parameters: Mapping[str, Float_]):
        return FixedRecallModel(self.choices[trial_index])

    def create_model(self, parameters: Mapping[str, Float_]):
        return self.create_trial_model(jnp.asarray(0), parameters)


def dataset() -> RecallDataset:
    return {
        "subject": jnp.array([[1], [2]]),
        "listLength": jnp.array([[3], [3]]),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]]),
        "recalls": jnp.array([[1, 0, 0], [1, 0, 0]]),
        "choice": jnp.array([[2], [2]]),
    }


def long_summary(recalls, *, list_length, dataset, trial_indices):
    return jnp.full(3, jnp.mean(recalls[..., 0]))


def short_summary(recalls, *, list_length, dataset, trial_indices):
    return 3 * jnp.mean(recalls[..., 0])


def evaluate(loss, indices=None):
    if indices is None:
        indices = jnp.array([0, 1])
    return jax.jit(lambda selected: loss(selected, {}, (), jnp.empty((0, 2))))(indices)


@pytest.mark.parametrize(
    ("reduction", "expected", "long_part", "short_part"),
    [("pooled", 5 / 14, 3 / 14, 1 / 7), ("per_analysis", 0.5, 1 / 6, 1 / 3)],
)
def test_weighted_scaled_components_have_exact_expected_values(
    reduction, expected, long_part, short_part
):
    Loss = mse.make_factory(
        {"long": long_summary, "short": short_summary},
        simulation_count=2,
        weights={"long": 2, "short": 1},
        scales={"long": jnp.full(3, 2.0), "short": 3.0},
        reduction=reduction,
    )
    loss = Loss(FixedRecallFactory, dataset(), None)
    parts = jax.jit(lambda x: loss.component_scores(jnp.array([0, 1]), {}, (), x))(
        jnp.empty((0, 2))
    )
    np.testing.assert_allclose(parts["long"], [long_part] * 2, rtol=1e-06)
    np.testing.assert_allclose(parts["short"], [short_part] * 2, rtol=1e-06)
    np.testing.assert_allclose(evaluate(loss), [expected] * 2, rtol=1e-06)
    np.testing.assert_allclose(sum(parts.values()), evaluate(loss), rtol=1e-06)


def test_trial_specific_models_are_used_even_for_identical_presentations():
    data = dataset()
    data["recalls"] = jnp.array([[1, 0, 0], [3, 0, 0]])
    data["choice"] = jnp.array([[1], [3]])

    def by_trial(recalls, *, list_length, dataset, trial_indices):
        return jnp.mean(recalls[..., 0], axis=1)

    loss = mse.make_factory({"by_trial": by_trial}, simulation_count=2)(
        FixedRecallFactory, data, None
    )
    np.testing.assert_array_equal(evaluate(loss), [0, 0])
    score = jax.jit(lambda indices: loss(indices, {}, (), jnp.empty((0, 1))))
    np.testing.assert_array_equal(score(jnp.array([1])), [0])
    np.testing.assert_array_equal(score(jnp.array([0])), [0])
    np.testing.assert_array_equal(score(jnp.array([1, 0])), [0])


def test_analysis_metadata_and_observations_follow_dynamic_subsets():
    data = dataset()
    data["recalls"] = jnp.array([[1, 0, 0], [2, 0, 0]])
    data["choice"] = jnp.array([[2], [3]])
    data["condition"] = jnp.array([[1.0], [2.0]])

    def conditioned(recalls, *, list_length, dataset, trial_indices):
        return jnp.mean(recalls[..., 0] * dataset["condition"][trial_indices])

    loss = mse.make_factory({"conditioned": conditioned}, simulation_count=2)(
        FixedRecallFactory, data, None
    )
    score = jax.jit(lambda indices: loss(indices, {}, (), jnp.empty((0, 1))))
    np.testing.assert_allclose(score(jnp.array([0])), [1])
    np.testing.assert_allclose(score(jnp.array([1])), [4])
    np.testing.assert_allclose(score(jnp.array([1, 0])), [2.25])


def test_observed_missing_bins_are_fixed_and_do_not_hide_missing_predictions():

    def with_missing_bin(recalls, *, list_length, dataset, trial_indices):
        return jnp.array([jnp.mean(recalls[..., 0]), jnp.nan])

    def missing_prediction(recalls, *, list_length, dataset, trial_indices):
        mean = jnp.mean(recalls[..., 0])
        return jnp.array([jnp.where(mean > 1, jnp.nan, mean), jnp.nan])

    valid_loss = mse.make_factory({"summary": with_missing_bin}, simulation_count=2)(
        FixedRecallFactory, dataset(), None
    )
    invalid_loss = mse.make_factory({"summary": missing_prediction}, simulation_count=2)(
        FixedRecallFactory, dataset(), None
    )
    np.testing.assert_allclose(evaluate(valid_loss), [1, 1])
    assert np.all(np.isposinf(evaluate(invalid_loss)))


def test_no_valid_targets_cannot_report_zero_error_and_zero_weight_disables_analysis():

    def undefined(recalls, *, list_length, dataset, trial_indices):
        return jnp.array([jnp.nan])

    Loss = mse.make_factory({"valid": long_summary, "undefined": undefined}, simulation_count=2)
    assert np.all(np.isposinf(evaluate(Loss(FixedRecallFactory, dataset(), None))))
    DisabledLoss = mse.make_factory(
        {"valid": long_summary, "undefined": undefined},
        simulation_count=2,
        weights={"undefined": 0},
    )
    np.testing.assert_allclose(evaluate(DisabledLoss(FixedRecallFactory, dataset(), None)), [1, 1])


def test_shape_mismatches_are_rejected_instead_of_broadcasting_predictions():

    def mismatched(recalls, *, list_length, dataset, trial_indices):
        return jnp.ones(recalls.shape[1])

    with pytest.raises(ValueError, match="observed shape.*simulated shape"):
        evaluate(
            mse.make_factory({"summary": mismatched}, simulation_count=2)(
                FixedRecallFactory, dataset(), None
            )
        )
    with pytest.raises(ValueError, match="cannot broadcast"):
        evaluate(
            mse.make_factory(
                {"summary": long_summary}, simulation_count=2, scales={"summary": jnp.ones(2)}
            )(FixedRecallFactory, dataset(), None)
        )


def test_empty_trial_selection_is_rejected():
    loss = mse.make_factory({"summary": long_summary})(FixedRecallFactory, dataset(), None)
    with pytest.raises(ValueError, match="at least one trial"):
        evaluate(loss, jnp.array([], dtype=int))


def test_factory_copies_configuration_mappings_and_scale_arrays():
    analyses = {"summary": long_summary}
    weights = {"summary": 1.0}
    scales = {"summary": np.full(3, 2.0)}
    Loss = mse.make_factory(analyses, weights=weights, scales=scales, simulation_count=2)
    analyses.clear()
    weights["summary"] = 0
    scales["summary"][:] = 0
    np.testing.assert_allclose(evaluate(Loss(FixedRecallFactory, dataset(), None)), [0.25, 0.25])
