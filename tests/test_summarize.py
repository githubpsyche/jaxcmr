import numpy as np
from jax import numpy as jnp
from scipy.stats import t as t_dist, ttest_rel

from jaxcmr.summarize import (
    add_summary_lines,
    aicc_winner_comparison_matrix,
    bound_params,
    calculate_aic,
    calculate_aicc,
    calculate_aicc_weights,
    calculate_aic_weights,
    calculate_bic_scores,
    calculate_ci,
    generate_t_p_matrices,
    pairwise_aic_differences,
    pairwise_aicc_differences,
    pairwise_median_aicc_differences,
    raw_winner_comparison_matrix,
    summarize_parameters,
    winner_comparison_matrix,
)


def _model_results():
    return [
        {
            "name": "ModelA",
            "fitness": [10.0, 12.0, 11.0],
            "free": ["p1", "p2"],
            "fixed": {"p3": 0.5},
            "fits": {
                "p1": [0.1, 0.2, 0.3],
                "p2": [0.4, 0.5, 0.6],
                "p3": [0.5, 0.5, 0.5],
            },
        },
        {
            "name": "ModelB",
            "fitness": [9.0, 11.0, 10.0],
            "free": ["p1"],
            "fixed": {"p2": 0.5, "p3": 0.5},
            "fits": {
                "p1": [0.1, 0.2, 0.3],
                "p2": [0.5, 0.5, 0.5],
                "p3": [0.5, 0.5, 0.5],
            },
        },
    ]


def test_bound_params_clips_above_upper():
    """Behavior: ``bound_params`` clips a value exceeding the upper bound.

    Given:
      - A parameter with value 5.0 and bounds [0.0, 1.0].
    When:
      - ``bound_params`` is called.
    Then:
      - The parameter is clipped to the upper bound 1.0.
    Why this matters:
      - Prevents optimizer-proposed values from leaving the feasible region.
    """
    # Arrange / Given
    params = {"a": jnp.array(5.0)}
    bounds = {"a": [0.0, 1.0]}

    # Act / When
    result = bound_params(params, bounds)

    # Assert / Then
    assert jnp.isclose(result["a"], 1.0).item()


def test_bound_params_clips_below_lower():
    """Behavior: ``bound_params`` clips a value below the lower bound.

    Given:
      - A parameter with value -3.0 and bounds [0.0, 1.0].
    When:
      - ``bound_params`` is called.
    Then:
      - The parameter is clipped to the lower bound 0.0.
    Why this matters:
      - Prevents negative values when the parameter must be non-negative.
    """
    # Arrange / Given
    params = {"a": jnp.array(-3.0)}
    bounds = {"a": [0.0, 1.0]}

    # Act / When
    result = bound_params(params, bounds)

    # Assert / Then
    assert jnp.isclose(result["a"], 0.0).item()


def test_calculate_ci_exact_halfwidth():
    """Behavior: CI half-width matches hand-calculated t * stderr.

    Given:
      - data = [1, 2, 3, 4, 5], n = 5.
      - std(ddof=1) = sqrt(2.5), stderr = sqrt(2.5)/sqrt(5) = sqrt(0.5).
      - t_crit = t.ppf(0.975, df=4).
    When:
      - ``calculate_ci`` is called with default 95% confidence.
    Then:
      - CI = stderr * t_crit ≈ 1.9632.
    Why this matters:
      - Verifies the exact confidence interval formula against hand
        calculation using the Student-t distribution.
    """
    # Arrange / Given
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stderr = np.std(data, ddof=1) / np.sqrt(5)
    expected_ci = stderr * t_dist.ppf(0.975, 4)

    # Act / When
    ci = calculate_ci(data)

    # Assert / Then
    assert np.isclose(ci, expected_ci, atol=1e-10)
    assert np.isclose(ci, 1.96324, atol=1e-4)


def test_calculate_ci_wider_at_higher_confidence():
    """Behavior: ``calculate_ci`` grows with confidence level.

    Given:
      - The same data evaluated at 90% and 99% confidence.
    When:
      - ``calculate_ci`` is called for both levels.
    Then:
      - The 99% interval is wider than the 90% interval.
    Why this matters:
      - Higher confidence requires a wider interval to maintain coverage.
    """
    # Arrange / Given
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Act / When
    ci_90 = calculate_ci(data, confidence=0.90)
    ci_99 = calculate_ci(data, confidence=0.99)

    # Assert / Then
    assert ci_99 > ci_90


def test_calculate_aic_exact_values():
    """Behavior: AIC = 2k - 2*LL for each model, sorted ascending.

    Given:
      - ModelA: k=2, LL=33 → AIC = 4 - 66 = -62.
      - ModelB: k=1, LL=30 → AIC = 2 - 60 = -58.
    When:
      - ``calculate_aic`` is called.
    Then:
      - ModelA first (AIC = -62 < -58), ModelB second.
    Why this matters:
      - Verifies the exact AIC formula and ascending sort order.
    """
    # Arrange / Given
    results = _model_results()

    # Act / When
    df = calculate_aic(results)

    # Assert / Then
    assert df.iloc[0]["AIC"] <= df.iloc[1]["AIC"]
    aic_a = df[df["Model"] == "ModelA"]["AIC"].values[0]  # type: ignore[union-attr]
    aic_b = df[df["Model"] == "ModelB"]["AIC"].values[0]  # type: ignore[union-attr]
    assert np.isclose(aic_a, 2 * 2 - 2 * 33.0)  # -62
    assert np.isclose(aic_b, 2 * 1 - 2 * 30.0)  # -58
    assert df.iloc[0]["Model"] == "ModelA"  # lower AIC first


def test_calculate_bic_exact_values():
    """Behavior: BIC = k*ln(n) - 2*LL for each model, sorted descending.

    Given:
      - n=3 subjects for both models.
      - ModelA: k=2, LL=33 → BIC = 2*ln(3) - 66.
      - ModelB: k=1, LL=30 → BIC = ln(3) - 60.
    When:
      - ``calculate_bic_scores`` is called.
    Then:
      - BIC values match the formula and are sorted descending.
    Why this matters:
      - Verifies the exact BIC formula with the log(n) penalty term.
    """
    # Arrange / Given
    results = _model_results()
    n = 3

    # Act / When
    df = calculate_bic_scores(results)

    # Assert / Then
    bic_a = df[df["Model"] == "ModelA"]["BIC"].values[0]  # type: ignore[union-attr]
    bic_b = df[df["Model"] == "ModelB"]["BIC"].values[0]  # type: ignore[union-attr]
    assert np.isclose(bic_a, 2 * np.log(n) - 2 * 33.0)
    assert np.isclose(bic_b, 1 * np.log(n) - 2 * 30.0)
    assert df.iloc[0]["BIC"] >= df.iloc[1]["BIC"]


def test_calculate_aic_weights_exact_values():
    """Behavior: AIC weights match exp(-0.5 * delta) / sum for known AICs.

    Given:
      - n_subjects = 3.
      - ModelA: AIC = 2*2*3 + 2*33 = 78. ModelB: AIC = 2*1*3 + 2*30 = 66.
      - min AIC = 66, deltas = [12, 0].
      - wt_A = exp(-6)/(1+exp(-6)), wt_B = 1/(1+exp(-6)).
    When:
      - ``calculate_aic_weights`` is called.
    Then:
      - Exact weight values match and sum to 1.0. ModelB has higher weight.
    Why this matters:
      - Verifies the exact AIC weight computation including delta
        calculation and softmax normalization. Uses summed per-subject
        AIC formula (2*k*n_subjects + 2*total_nll) consistent with
        ``_subjectwise_aic``.
    """
    # Arrange / Given
    results = _model_results()
    exp_neg6 = np.exp(-6.0)
    expected_wt_b = 1.0 / (1.0 + exp_neg6)
    expected_wt_a = exp_neg6 / (1.0 + exp_neg6)

    # Act / When
    df = calculate_aic_weights(results)

    # Assert / Then
    assert np.isclose(df["AICw"].sum(), 1.0)
    wt_a = df[df["Model"] == "ModelA"]["AICw"].values[0]  # type: ignore[union-attr]
    wt_b = df[df["Model"] == "ModelB"]["AICw"].values[0]  # type: ignore[union-attr]
    assert np.isclose(wt_a, expected_wt_a, atol=1e-10)
    assert np.isclose(wt_b, expected_wt_b, atol=1e-10)
    assert df.iloc[0]["Model"] == "ModelB"  # higher weight first


def test_pairwise_aic_differences_exact_values():
    """Behavior: Mean delta-AIC matches hand-calculated per-subject diffs.

    Given:
      - Per-subject AIC_A = 2*fitness_A + 2*k_A = 2*fitness_A + 4,
        Per-subject AIC_B = 2*fitness_B + 2*k_B = 2*fitness_B + 2.
      - Per-subject diffs: 2*(fitness_A - fitness_B) + 2 = 4 for all subjects.
    When:
      - ``pairwise_aic_differences`` is called.
    Then:
      - mean_delta[A,B] = 4.0 (A has higher penalized fitness).
      - CI = 0.0 (constant differences → zero variance).
      - Diagonal is NaN. Not equivalent (|4.0| > margin 2.0).
    Why this matters:
      - Verifies exact per-subject AIC penalty allocation and mean
        difference computation.
    """
    # Arrange / Given
    results = _model_results()

    # Act / When
    mean_df, ci_df, equiv_df = pairwise_aic_differences(results)

    # Assert / Then — diagonal
    for name in ["ModelA", "ModelB"]:
        assert np.isnan(mean_df.loc[name, name])
        assert np.isnan(ci_df.loc[name, name])
        assert equiv_df.loc[name, name] == False  # noqa: E712

    # Off-diagonal exact values
    assert np.isclose(mean_df.loc["ModelA", "ModelB"], 4.0, atol=1e-10)
    assert np.isclose(mean_df.loc["ModelB", "ModelA"], -4.0, atol=1e-10)
    assert np.isclose(ci_df.loc["ModelA", "ModelB"], 0.0, atol=1e-10)
    # |4.0| > equivalence_margin 2.0 → not equivalent
    assert equiv_df.loc["ModelA", "ModelB"] == False  # noqa: E712


def test_winner_comparison_exact_fractions():
    """Behavior: Penalized fractions match hand-calculated comparisons.

    Given:
      - penalized_A = 2*fitness_A + 4 = [24, 28, 26].
      - penalized_B = 2*fitness_B + 2 = [20, 24, 22].
      - B < A on all 3 subjects.
    When:
      - ``winner_comparison_matrix`` is called.
    Then:
      - fraction(A<B) = 0.0, fraction(B<A) = 1.0. Diagonal is NaN.
    Why this matters:
      - Verifies the AIC-style penalty and per-subject comparison logic.
    """
    # Arrange / Given
    results = _model_results()

    # Act / When
    df = winner_comparison_matrix(results)

    # Assert / Then
    for name in ["ModelA", "ModelB"]:
        assert np.isnan(df.loc[name, name])
    assert np.isclose(df.loc["ModelA", "ModelB"], 0.0)
    assert np.isclose(df.loc["ModelB", "ModelA"], 1.0)


def test_raw_winner_comparison_exact_fractions():
    """Behavior: Raw (unpenalized) comparison yields ModelB winning 100%.

    Given:
      - ModelB fitness [9, 11, 10] < ModelA fitness [10, 12, 11] on all
        subjects.
    When:
      - ``raw_winner_comparison_matrix`` is called.
    Then:
      - fraction(B<A) = 1.0, fraction(A<B) = 0.0.
    Why this matters:
      - Without penalty, strictly lower raw fitness should always win.
    """
    # Arrange / Given
    results = _model_results()

    # Act / When
    df = raw_winner_comparison_matrix(results)

    # Assert / Then
    assert np.isnan(df.loc["ModelA", "ModelA"])
    assert np.isclose(df.loc["ModelB", "ModelA"], 1.0)
    assert np.isclose(df.loc["ModelA", "ModelB"], 0.0)


def test_generate_t_p_matrices_exact_t_stat():
    """Behavior: Off-diagonal t-stat matches scipy ttest_rel.

    Given:
      - ModelX fitness = [10, 13, 11], ModelY fitness = [9, 11, 10].
      - diffs (Y-X) = [-1, -2, -1], mean=-4/3, stderr=1/3, t=-4.0.
    When:
      - ``generate_t_p_matrices`` is called.
    Then:
      - t[Y,X] = -4.0 (one-sided "less"). Diagonal is empty string.
    Why this matters:
      - Verifies paired t-test computation against scipy reference with
        non-degenerate (varying) differences.
    """
    # Arrange / Given
    results = [
        {"name": "ModelX", "fitness": [10.0, 13.0, 11.0], "free": ["p1"]},
        {"name": "ModelY", "fitness": [9.0, 11.0, 10.0], "free": ["p1"]},
    ]
    expected_t, expected_p = ttest_rel(
        [9.0, 11.0, 10.0], [10.0, 13.0, 11.0], alternative="less"
    )

    # Act / When
    t_df, p_df = generate_t_p_matrices(results)

    # Assert / Then — diagonal is empty string
    for name in ["ModelX", "ModelY"]:
        assert t_df.loc[name, name] == ""
        assert p_df.loc[name, name] == ""

    # Off-diagonal: t(Y,X) = -4.0
    assert np.isclose(t_df.loc["ModelY", "ModelX"], expected_t, atol=1e-10)
    assert np.isclose(p_df.loc["ModelY", "ModelX"], expected_p, atol=1e-10)
    assert np.isclose(expected_t, -4.0, atol=1e-10)


def test_generate_t_p_matrices_lower_fitness_gives_negative_t():
    """Behavior: Negative t-stat when row model has lower fitness.

    Given:
      - ModelX fitness = [10, 13, 11], ModelY fitness = [9, 11, 10].
      - ModelY < ModelX on average.
    When:
      - ``generate_t_p_matrices`` is called with alternative="less".
    Then:
      - t[Y,X] < 0 and p[Y,X] < 0.05.
    Why this matters:
      - Confirms the direction convention: negative t means the row
        model has lower fitness (better fit) than the column model.
    """
    # Arrange / Given
    results = [
        {"name": "ModelX", "fitness": [10.0, 13.0, 11.0], "free": ["p1"]},
        {"name": "ModelY", "fitness": [9.0, 11.0, 10.0], "free": ["p1"]},
    ]

    # Act / When
    t_df, p_df = generate_t_p_matrices(results)

    # Assert / Then
    assert t_df.loc["ModelY", "ModelX"] < 0.0
    assert p_df.loc["ModelY", "ModelX"] < 0.05


def test_summarize_parameters_exact_mean_values():
    """Behavior: Markdown table contains exact formatted mean values.

    Given:
      - ModelA fitness mean = (10+12+11)/3 = 11.00.
      - ModelB fitness mean = (9+11+10)/3 = 10.00.
      - Fixed param p3 mean = 0.50 for both models.
    When:
      - ``summarize_parameters`` is called.
    Then:
      - Table contains "11.00" and "10.00" for fitness means.
      - Table contains "0.50" for p3 means.
    Why this matters:
      - Verifies that parameter summary correctly computes and formats
        per-model mean statistics in the Markdown table.
    """
    # Arrange / Given
    results = _model_results()

    # Act / When
    table = summarize_parameters(results)  # type: ignore[arg-type]

    # Assert / Then — structure
    assert "| Parameter |" in table
    assert "|---|---" in table
    # Exact mean values for fitness
    assert "11.00" in table  # ModelA mean fitness
    assert "10.00" in table  # ModelB mean fitness
    # Exact mean values for p3
    assert "0.50" in table
    # Min/max for fitness
    assert "12.00" in table  # ModelA max fitness
    assert "9.00" in table   # ModelB min fitness


def test_add_summary_lines_exact_formatted_values():
    """Behavior: Summary rows contain exact formatted statistics.

    Given:
      - Variant 0: [1, 2, 3] → mean=2.00, min=1.00, max=3.00.
      - Variant 1: [4, 5, 6] → mean=5.00, min=4.00, max=6.00.
    When:
      - ``add_summary_lines`` is called with label "test_metric".
    Then:
      - Result contains rows with exact formatted values and label
        underscores replaced with spaces.
    Why this matters:
      - Verifies that each statistic is correctly computed and formatted
        for the Markdown table output.
    """
    # Arrange / Given
    md_table = ""
    errors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    # Act / When
    result = add_summary_lines(md_table, errors, "test_metric")

    # Assert / Then — label formatting
    assert "test metric" in result
    # Exact formatted values
    assert "2.00" in result   # variant 0 mean
    assert "5.00" in result   # variant 1 mean
    assert "1.00" in result   # variant 0 min
    assert "4.00" in result   # variant 1 min
    assert "3.00" in result   # variant 0 max
    assert "6.00" in result   # variant 1 max
    # Three rows: mean, min, max
    lines = result.strip().split("\n")
    assert len(lines) == 3
    assert "mean" in lines[0]
    assert "min" in lines[1]
    assert "max" in lines[2]


# ---------------------------------------------------------------------------
# AICc tests
# ---------------------------------------------------------------------------

def test_calculate_aicc_exact_values():
    """Behavior: AICc = AIC + 2k(k+1)/(n_s - k - 1) summed across subjects.

    Given:
      - ModelA: k=2, fitness=[10,12,11], n_obs=[20,20,20].
        Per-subject AICc_s = 2*NLL + 4 + 2*2*3/(20-2-1) = 2*NLL + 4 + 12/17.
        Total = 2*33 + 3*4 + 3*12/17 = 66 + 12 + 36/17 = 78 + 36/17.
      - ModelB: k=1, fitness=[9,11,10], n_obs=[20,20,20].
        Per-subject AICc_s = 2*NLL + 2 + 2*1*2/(20-1-1) = 2*NLL + 2 + 4/18.
        Total = 2*30 + 3*2 + 3*4/18 = 60 + 6 + 12/18 = 66 + 2/3.
    When:
      - ``calculate_aicc`` is called.
    Then:
      - Values match the formula. ModelB (lower AICc) is first.
    """
    results = _model_results()
    n_obs = np.array([20.0, 20.0, 20.0])

    df = calculate_aicc(results, n_obs)

    expected_a = 2 * 33.0 + 3 * 4.0 + 3 * (2 * 2 * 3) / (20 - 2 - 1)
    expected_b = 2 * 30.0 + 3 * 2.0 + 3 * (2 * 1 * 2) / (20 - 1 - 1)

    aicc_a = df[df["Model"] == "ModelA"]["AICc"].values[0]
    aicc_b = df[df["Model"] == "ModelB"]["AICc"].values[0]
    assert np.isclose(aicc_a, expected_a, atol=1e-10)
    assert np.isclose(aicc_b, expected_b, atol=1e-10)
    assert df.iloc[0]["Model"] == "ModelB"  # lower AICc first


def test_calculate_aicc_weights_sum_to_one():
    """Behavior: AICc weights sum to 1.0 and ModelB has higher weight.

    Given:
      - Same fixture as AIC weight test but with AICc correction.
    When:
      - ``calculate_aicc_weights`` is called.
    Then:
      - Weights sum to 1.0 and ModelB (fewer params, lower NLL) dominates.
    """
    results = _model_results()
    n_obs = np.array([20.0, 20.0, 20.0])

    df = calculate_aicc_weights(results, n_obs)

    assert np.isclose(df["AICcw"].sum(), 1.0)
    assert df.iloc[0]["Model"] == "ModelB"  # higher weight first


def test_aicc_correction_increases_with_k():
    """Behavior: AICc penalty exceeds AIC penalty, more so for larger k.

    Given:
      - ModelA (k=2) and ModelB (k=1) with n_obs=20.
    When:
      - AICc and AIC are computed.
    Then:
      - AICc > AIC for both models.
      - The correction gap is larger for ModelA (k=2) than ModelB (k=1).
    """
    results = _model_results()
    n_obs = np.array([20.0, 20.0, 20.0])

    df_aic = calculate_aic(results)
    df_aicc = calculate_aicc(results, n_obs)

    aic_a = df_aic[df_aic["Model"] == "ModelA"]["AIC"].values[0]
    aic_b = df_aic[df_aic["Model"] == "ModelB"]["AIC"].values[0]
    aicc_a = df_aicc[df_aicc["Model"] == "ModelA"]["AICc"].values[0]
    aicc_b = df_aicc[df_aicc["Model"] == "ModelB"]["AICc"].values[0]

    # AICc sums per-subject (so AIC total is 2k*N + 2*NLL, not 2k - 2*LL)
    # Compare corrections: aicc - per-subject-aic-total
    aic_total_a = 2 * 2 * 3 + 2 * 33.0  # 78
    aic_total_b = 2 * 1 * 3 + 2 * 30.0  # 66
    correction_a = aicc_a - aic_total_a
    correction_b = aicc_b - aic_total_b

    assert correction_a > 0
    assert correction_b > 0
    assert correction_a > correction_b  # larger k → larger correction


def test_aicc_raises_when_n_obs_too_small():
    """Behavior: ValueError when n_obs <= k + 1 for any subject.

    Given:
      - ModelA has k=2, and one subject has n_obs=3 (= k+1).
    When:
      - ``calculate_aicc`` is called.
    Then:
      - ValueError is raised.
    """
    import pytest

    results = _model_results()
    n_obs = np.array([3.0, 20.0, 20.0])  # first subject: n=3, k=2, n-k-1=0

    with pytest.raises(ValueError, match="AICc undefined"):
        calculate_aicc(results, n_obs)


def test_pairwise_aicc_differences_exact_values():
    """Behavior: Pairwise ΔAICc matches hand-calculated per-subject diffs.

    Given:
      - Constant n_obs=20 for all subjects.
      - Per-subject ΔAICc = (2*NLL_A + 4 + 12/17) - (2*NLL_B + 2 + 4/18).
        = 2*(fitness_A - fitness_B) + 2 + 12/17 - 4/18.
        fitness diffs are all 1.0, so Δ = 2 + 2 + 12/17 - 4/18 (constant).
    When:
      - ``pairwise_aicc_differences`` is called.
    Then:
      - mean_delta[A,B] matches the constant difference.
      - CI = 0.0 (constant → zero variance).
    """
    results = _model_results()
    n_obs = np.array([20.0, 20.0, 20.0])

    mean_df, ci_df, equiv_df = pairwise_aicc_differences(results, n_obs)

    # Per-subject difference is constant
    k_a, k_b = 2, 1
    n = 20.0
    correction_a = 2 * k_a * (k_a + 1) / (n - k_a - 1)
    correction_b = 2 * k_b * (k_b + 1) / (n - k_b - 1)
    # Per-subject: (2*NLL_A + 2*k_a + corr_a) - (2*NLL_B + 2*k_b + corr_b)
    # NLL diffs are all 1.0
    expected_diff = 2.0 * 1.0 + 2 * (k_a - k_b) + (correction_a - correction_b)

    assert np.isclose(mean_df.loc["ModelA", "ModelB"], expected_diff, atol=1e-10)
    assert np.isclose(ci_df.loc["ModelA", "ModelB"], 0.0, atol=1e-10)


def test_pairwise_median_aicc_differences_matches_mean_for_constant_diffs():
    """Behavior: Median equals mean when per-subject diffs are constant.

    Given:
      - Same constant-diff fixture as pairwise_aicc_differences test.
    When:
      - ``pairwise_median_aicc_differences`` is called.
    Then:
      - Median matches mean exactly.
    """
    results = _model_results()
    n_obs = np.array([20.0, 20.0, 20.0])

    mean_df, _, _ = pairwise_aicc_differences(results, n_obs)
    median_df, _ = pairwise_median_aicc_differences(results, n_obs)

    assert np.isclose(
        median_df.loc["ModelA", "ModelB"],
        mean_df.loc["ModelA", "ModelB"],
        atol=1e-10,
    )


def test_aicc_winner_comparison_exact_fractions():
    """Behavior: AICc-penalized fractions match hand-calculated comparisons.

    Given:
      - With n_obs=20, ModelB has lower AICc on all 3 subjects (lower NLL
        and fewer parameters).
    When:
      - ``aicc_winner_comparison_matrix`` is called.
    Then:
      - fraction(A<B) = 0.0, fraction(B<A) = 1.0.
    """
    results = _model_results()
    n_obs = np.array([20.0, 20.0, 20.0])

    df = aicc_winner_comparison_matrix(results, n_obs)

    for name in ["ModelA", "ModelB"]:
        assert np.isnan(df.loc[name, name])
    assert np.isclose(df.loc["ModelA", "ModelB"], 0.0)
    assert np.isclose(df.loc["ModelB", "ModelA"], 1.0)
