"""
All tests common for Weng-Lin models are located here.
"""

import math
import random
from typing import Any

import pytest

from openskill.models import MODELS
from openskill.models.weng_lin import (
    BradleyTerryFull,
    BradleyTerryPart,
    PlackettLuce,
    ThurstoneMostellerFull,
    ThurstoneMostellerPart,
)
from openskill.models.weng_lin.common import _ladder_pairs, _unwind, v, vt, w, wt


@pytest.mark.parametrize("model", MODELS)
def test_calculate_team_ratings(model) -> None:
    """
    Tests the :code:`_calculate_team_ratings` function.
    """

    model = model()
    r = model.rating
    team_1 = [r()]
    team_2 = [r(), r()]

    # Aggregates all players in a team
    result = model._calculate_team_ratings([team_1, team_2])
    assert result[0].mu == pytest.approx(25)
    assert result[1].mu == pytest.approx(50)
    assert result[0].sigma_squared == pytest.approx(69.44444)
    assert result[1].sigma_squared == pytest.approx(138.88888)
    assert result[0].team == team_1
    assert result[1].team == team_2
    assert result[0].rank == 0
    assert result[1].rank == 1

    # 5 v 5
    result = model._calculate_team_ratings(
        [[r(), r(), r(), r(), r()], [r(), r(), r(), r(), r()]]
    )
    assert result[0].mu == pytest.approx(125)
    assert result[1].mu == pytest.approx(125)
    assert result[0].sigma_squared == pytest.approx(347.2222222)
    assert result[1].sigma_squared == pytest.approx(347.2222222)

    # 5 v 5 v 5 with ranks
    result = model._calculate_team_ratings(
        game=[
            [r(), r(), r(), r(), r()],
            [r(), r(), r(), r(), r()],
            [r(), r(), r(), r(), r()],
        ],
        ranks=[3, 1, 2],
    )
    assert result[0].mu == pytest.approx(125)
    assert result[1].mu == pytest.approx(125)
    assert result[2].mu == pytest.approx(125)
    assert result[0].sigma_squared == pytest.approx(347.2222222)
    assert result[1].sigma_squared == pytest.approx(347.2222222)
    assert result[2].sigma_squared == pytest.approx(347.2222222)


@pytest.mark.parametrize("model", MODELS)
def test_calculate_rankings(model) -> None:
    """
    Tests the :code:`_calculate_rankings` function.
    """
    model = model()
    r = model.rating

    # Generate 5 players overall for single player teams
    a = [r()]
    b = [r()]
    c = [r()]
    d = [r()]
    e = [r()]

    # Conduct Tests
    assert model._calculate_rankings([]) == []
    assert model._calculate_rankings([], []) == []
    assert model._calculate_rankings([a, b, c, d]) == [0, 1, 2, 3]
    assert model._calculate_rankings([a, b], [0, 0]) == [0, 0]
    assert model._calculate_rankings([a, b, c, d], [1, 2, 3, 4]) == [0, 1, 2, 3]
    assert model._calculate_rankings([a, b, c, d], [1, 1, 3, 4]) == [0, 0, 2, 3]
    assert model._calculate_rankings([a, b, c, d], [1, 2, 3, 3]) == [0, 1, 2, 2]
    assert model._calculate_rankings([a, b, c, d], [1, 2, 2, 4]) == [0, 1, 1, 3]
    assert model._calculate_rankings([a, b, c, d, e], [14, 32, 47, 47, 48]) == [
        0,
        1,
        2,
        2,
        4,
    ]


def test_unwind() -> None:
    """
    Tests the :code:`_unwind` function.
    """
    # Zero Items
    source: list[Any] = []
    rank: list[Any] = []
    output, tenet = _unwind(rank, source)
    assert output == []
    assert tenet == []

    # Accepts 1 Item
    source = ["a"]
    rank = [0]
    output, tenet = _unwind(rank, source)
    assert output == ["a"]
    assert tenet == [0]

    # Accepts 2 Items
    source = ["b", "a"]
    rank = [1, 0]
    output, tenet = _unwind(rank, source)
    assert output == ["a", "b"]
    assert tenet == [1, 0]

    # Accepts 3 Items
    source = ["b", "c", "a"]
    rank = [1, 2, 0]
    output, tenet = _unwind(rank, source)
    assert output == ["a", "b", "c"]
    assert tenet == [2, 0, 1]

    # Accepts 4 Items
    source = ["b", "d", "c", "a"]
    rank = [1, 3, 2, 0]
    output, tenet = _unwind(rank, source)
    assert output == ["a", "b", "c", "d"]
    assert tenet == [3, 0, 2, 1]

    # Can undo the ranking
    source = [random.random() for _ in range(100)]
    random.shuffle(source)
    rank = [i for i in range(100)]
    trans, tenet = _unwind(rank, source)
    output, de_de_rank = _unwind(tenet, trans)
    assert source == output
    assert de_de_rank == rank

    # Allows ranks that are not zero-indexed integers
    source = ["a", "b", "c", "d", "e", "f"]
    rank = [0.28591, 0.42682, 0.35912, 0.21237, 0.60619, 0.47078]
    output, tenet = _unwind(rank, source)
    assert output == ["d", "a", "c", "b", "f", "e"]


def test_v() -> None:
    """
    Test the v function
    """
    assert v(1, 2) == pytest.approx(1.525135276160981, 0.00001)
    assert v(0, 2) == pytest.approx(2.373215532822843, 0.00001)
    assert v(0, -1) == pytest.approx(0.2875999709391784, 0.00001)
    assert v(0, 10) == 10


def test_vt() -> None:
    """
    Test the vt function
    """
    assert vt(-1000, -100) == 1100
    assert vt(1000, -100) == -1100
    assert vt(-1000, 1000) == pytest.approx(0.79788, 0.00001)
    assert vt(0, 1000) == 0


def test_w():
    """
    Test the w function
    """
    assert w(1, 2) == pytest.approx(0.800902334429651, 0.00001)
    assert w(0, 2) == pytest.approx(0.885720899585924, 0.00001)
    assert w(0, -1) == pytest.approx(0.3703137142233946, 0.00001)
    assert w(0, 10) == 0
    assert w(-1, 10) == 1


def test_wt():
    """
    Test the wt function
    """
    assert wt(1, 2) == pytest.approx(0.3838582646421707, 0.00001)
    assert wt(0, 2) == pytest.approx(0.2262586964500768, 0.00001)
    assert wt(0, -1) == 1
    assert wt(0, 0) == 1.0
    assert wt(0, 10) == pytest.approx(0)


def test_ladder_pairs():
    """
    Test the ladder_pairs function
    """
    assert _ladder_pairs([]) == [[]]
    assert _ladder_pairs([1]) == [[]]
    assert _ladder_pairs([1, 2]) == [[2], [1]]
    assert _ladder_pairs([1, 2, 3]) == [[2], [1, 3], [2]]
    assert _ladder_pairs([1, 2, 3, 4]) == [[2], [1, 3], [2, 4], [3]]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tie_score", [-1, 0, 0.1, 10, 13.4])
@pytest.mark.parametrize("num_teams", [2, 5, 10])
@pytest.mark.parametrize("team_size", [1, 2, 5, 10])
@pytest.mark.parametrize("tie_type", ["score", "rank"])
def test_ties(model, tie_score, num_teams, team_size, tie_type) -> None:
    model_instance = model()
    teams = [
        [model_instance.rating() for _ in range(team_size)] for _ in range(num_teams)
    ]
    player_mu_before = [player.mu for team in teams for player in team]
    assert all(
        mu == player_mu_before[0] for mu in player_mu_before
    ), f"Model {model.__name__} with score {tie_score}: All players should start with equal mu"

    player_sigma_before = [player.sigma for team in teams for player in team]
    assert all(
        sigma == player_sigma_before[0] for sigma in player_sigma_before
    ), f"Model {model.__name__} with score {tie_score}: All players should start with equal sigma"

    if tie_type == "score":
        scores = [tie_score for _ in range(num_teams)]
        new_teams = model_instance.rate(teams, scores=scores)
    else:  # rank
        ranks = [tie_score for _ in range(num_teams)]
        new_teams = model_instance.rate(teams, ranks=ranks)

    player_mu_after = [player.mu for team in new_teams for player in team]
    assert all(
        mu_after == mu_before
        for mu_after, mu_before in zip(player_mu_after, player_mu_before)
    ), f"Model {model.__name__} with score {tie_score}: All players should end with equal mu"
    player_sigma_after = [player.sigma for team in new_teams for player in team]
    assert all(
        sigma_after <= sigma_before
        for sigma_after, sigma_before in zip(player_sigma_after, player_sigma_before)
    ), f"Model {model.__name__} with score {tie_score}: All players should end with lower or equal sigma"


@pytest.mark.parametrize("model", MODELS)
def test_ties_with_close_ratings(model) -> None:
    model_instance = model()

    player_1 = model_instance.rating(mu=30)
    player_2 = model_instance.rating(mu=20)

    new_teams = model_instance.rate([[player_1], [player_2]], ranks=[0, 0])

    if model in (PlackettLuce, BradleyTerryFull, ThurstoneMostellerFull):
        # Full-pairing models apply tie-group mu averaging.
        assert new_teams[0][0].mu == pytest.approx(30)
        assert new_teams[1][0].mu == pytest.approx(20)
    else:
        # Partial-pairing models still converge on ties.
        assert new_teams[0][0].mu < 30
        assert new_teams[1][0].mu > 20


@pytest.mark.parametrize("model", MODELS)
def test_rate_tau_override_honors_zero(model) -> None:
    """A per-call tau of 0.0 should bypass model-level tau inflation."""
    configured_tau = 1.25
    model_with_override = model(tau=configured_tau)
    override_sigmas_seen: list[float] = []

    def capture_override_compute(
        *,
        teams,
        ranks=None,
        scores=None,
        weights=None,
    ):
        override_sigmas_seen.extend([player.sigma for team in teams for player in team])
        return [list(team) for team in teams]

    model_with_override._compute = capture_override_compute  # type: ignore[method-assign]
    p1_override = model_with_override.rating(mu=30.0, sigma=7.0)
    p2_override = model_with_override.rating(mu=20.0, sigma=9.0)
    model_with_override.rate([[p1_override], [p2_override]], ranks=[1, 2], tau=0.0)
    assert override_sigmas_seen == pytest.approx([7.0, 9.0])

    # Without an override, model-level tau should still be applied.
    model_default = model(tau=configured_tau)
    default_sigmas_seen: list[float] = []

    def capture_default_compute(
        *,
        teams,
        ranks=None,
        scores=None,
        weights=None,
    ):
        default_sigmas_seen.extend([player.sigma for team in teams for player in team])
        return [list(team) for team in teams]

    model_default._compute = capture_default_compute  # type: ignore[method-assign]
    p1_default = model_default.rating(mu=30.0, sigma=7.0)
    p2_default = model_default.rating(mu=20.0, sigma=9.0)
    model_default.rate([[p1_default], [p2_default]], ranks=[1, 2])
    assert default_sigmas_seen == pytest.approx(
        [
            math.sqrt(7.0 * 7.0 + configured_tau * configured_tau),
            math.sqrt(9.0 * 9.0 + configured_tau * configured_tau),
        ]
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bad_weight", [0.0, -1.0])
def test_rate_rejects_non_positive_weights(model, bad_weight) -> None:
    """Per-player weights must be strictly positive to keep updates stable."""
    model_instance = model(weight_bounds=None)
    a = model_instance.rating()
    b = model_instance.rating()

    with pytest.raises(ValueError, match=r"weights.*> 0"):
        model_instance.rate(
            [[a], [b]],
            ranks=[1, 2],
            weights=[[1.0], [bad_weight]],
        )


@pytest.mark.parametrize(
    "model",
    [PlackettLuce, BradleyTerryFull, ThurstoneMostellerFull],
)
def test_tie_rank_groups_apply_shared_mu_change(model) -> None:
    """Tied teams should receive the same tie-adjusted mu change."""
    model_instance = model()
    teams = [
        [
            model_instance.rating(mu=35.0, sigma=7.0),
            model_instance.rating(mu=25.0, sigma=8.0),
        ],
        [
            model_instance.rating(mu=20.0, sigma=6.0),
            model_instance.rating(mu=30.0, sigma=5.0),
        ],
        [model_instance.rating(mu=28.0, sigma=7.0)],
    ]
    before = [[player.mu for player in team] for team in teams]

    new_teams = model_instance.rate(teams, ranks=[1, 1, 3])
    delta_team_0 = [new_teams[0][i].mu - before[0][i] for i in range(len(new_teams[0]))]
    delta_team_1 = [new_teams[1][i].mu - before[1][i] for i in range(len(new_teams[1]))]

    for d0, d1 in zip(delta_team_0, delta_team_1):
        assert d0 == pytest.approx(d1)


# ---------------------------------------------------------------------------
# Geometric mean likelihood draw tests
# ---------------------------------------------------------------------------

# Models where we can compare draw vs win vs loss updates directly.
# PlackettLuce uses a different mechanism (exp/softmax, naturally bounded).
_PAIRWISE_MODELS = [
    BradleyTerryFull,
    BradleyTerryPart,
    ThurstoneMostellerFull,
    ThurstoneMostellerPart,
]


@pytest.mark.parametrize("model", _PAIRWISE_MODELS)
@pytest.mark.parametrize(
    "mu_a, mu_b",
    [
        (25.0, 25.0),  # equal skill
        (30.0, 20.0),  # moderate gap
        (40.0, 10.0),  # large gap
        (50.0, 5.0),  # extreme gap
    ],
)
def test_draw_mu_bounded_between_win_and_loss(model, mu_a: float, mu_b: float) -> None:
    """Draw mu change must be strictly between win and loss mu changes.

    This is the core property of the geometric mean likelihood: a draw is
    always an intermediate outcome, never more extreme than a decisive result.
    """
    m = model()
    sigma = m.sigma

    # Win for A
    a_win = m.rating(mu=mu_a, sigma=sigma)
    b_win = m.rating(mu=mu_b, sigma=sigma)
    win_result = m.rate([[a_win], [b_win]], ranks=[1, 2])
    mu_after_win = win_result[0][0].mu

    # Loss for A
    a_loss = m.rating(mu=mu_a, sigma=sigma)
    b_loss = m.rating(mu=mu_b, sigma=sigma)
    loss_result = m.rate([[a_loss], [b_loss]], ranks=[2, 1])
    mu_after_loss = loss_result[0][0].mu

    # Draw
    a_draw = m.rating(mu=mu_a, sigma=sigma)
    b_draw = m.rating(mu=mu_b, sigma=sigma)
    draw_result = m.rate([[a_draw], [b_draw]], ranks=[1, 1])
    mu_after_draw = draw_result[0][0].mu

    delta_win = mu_after_win - mu_a
    delta_loss = mu_after_loss - mu_a
    delta_draw = mu_after_draw - mu_a

    lo = min(delta_win, delta_loss)
    hi = max(delta_win, delta_loss)

    assert lo < delta_draw < hi or delta_draw == pytest.approx(0.0, abs=1e-10), (
        f"Draw delta {delta_draw:.6f} not between win {delta_win:.6f} "
        f"and loss {delta_loss:.6f} for {model.__name__}"
    )


@pytest.mark.parametrize("model", _PAIRWISE_MODELS)
def test_draw_sigma_bounded_between_win_and_loss(model) -> None:
    """Draw sigma reduction should not exceed that of a decisive outcome."""
    m = model()
    sigma = m.sigma
    mu_a, mu_b = 40.0, 10.0  # large gap to stress the old pathology

    results = {}
    for label, ranks in [("win", [1, 2]), ("loss", [2, 1]), ("draw", [1, 1])]:
        a = m.rating(mu=mu_a, sigma=sigma)
        b = m.rating(mu=mu_b, sigma=sigma)
        res = m.rate([[a], [b]], ranks=ranks)
        results[label] = res[0][0].sigma

    # Sigma after draw should be >= the minimum of win and loss sigmas
    # (i.e., draw should not reduce uncertainty more than a decisive outcome).
    min_decisive_sigma = min(results["win"], results["loss"])
    assert results["draw"] >= min_decisive_sigma - 1e-10, (
        f"Draw sigma {results['draw']:.6f} < min decisive sigma "
        f"{min_decisive_sigma:.6f} for {model.__name__}"
    )


@pytest.mark.parametrize("model", _PAIRWISE_MODELS)
def test_draw_equal_players_no_mu_change(model) -> None:
    """A draw between equal players should produce no mu change."""
    m = model()
    mu = 25.0
    a = m.rating(mu=mu, sigma=m.sigma)
    b = m.rating(mu=mu, sigma=m.sigma)
    result = m.rate([[a], [b]], ranks=[1, 1])
    assert result[0][0].mu == pytest.approx(mu, abs=1e-10)
    assert result[1][0].mu == pytest.approx(mu, abs=1e-10)


@pytest.mark.parametrize("model", _PAIRWISE_MODELS)
def test_draw_equal_players_sigma_decreases(model) -> None:
    """A draw between equal players should still reduce sigma (we observed a game)."""
    m = model()
    sigma = m.sigma
    a = m.rating(mu=25.0, sigma=sigma)
    b = m.rating(mu=25.0, sigma=sigma)
    result = m.rate([[a], [b]], ranks=[1, 1])
    assert result[0][0].sigma < sigma
    assert result[1][0].sigma < sigma


@pytest.mark.parametrize("model", MODELS)
def test_alpha_parameter_accepted(model) -> None:
    """All models should accept the alpha parameter."""
    m = model(alpha=0.5)
    assert m.alpha == 0.5
    m2 = model(alpha=0.7)
    assert m2.alpha == pytest.approx(0.7)


@pytest.mark.parametrize("model", _PAIRWISE_MODELS)
def test_alpha_default_symmetric(model) -> None:
    """With default alpha=0.5, draw updates should be symmetric.

    If A draws B, the mu changes should be equal and opposite.
    """
    m = model()
    a = m.rating(mu=30.0, sigma=m.sigma)
    b = m.rating(mu=20.0, sigma=m.sigma)
    result = m.rate([[a], [b]], ranks=[1, 1])
    delta_a = result[0][0].mu - 30.0
    delta_b = result[1][0].mu - 20.0
    assert delta_a == pytest.approx(-delta_b, abs=1e-8)


@pytest.mark.parametrize(
    "model",
    [
        BradleyTerryFull,
        BradleyTerryPart,
        ThurstoneMostellerFull,
        ThurstoneMostellerPart,
    ],
)
def test_alpha_affects_draw_direction(model) -> None:
    """Non-default alpha should shift draw updates toward win or loss."""
    mu_a, mu_b = 30.0, 20.0

    # alpha > 0.5: draw treated more like a win for favoured player
    m_high = model(alpha=0.8)
    a = m_high.rating(mu=mu_a, sigma=m_high.sigma)
    b = m_high.rating(mu=mu_b, sigma=m_high.sigma)
    result_high = m_high.rate([[a], [b]], ranks=[1, 1])
    delta_high = result_high[0][0].mu - mu_a

    # alpha < 0.5: draw treated more like a loss for favoured player
    m_low = model(alpha=0.2)
    a = m_low.rating(mu=mu_a, sigma=m_low.sigma)
    b = m_low.rating(mu=mu_b, sigma=m_low.sigma)
    result_low = m_low.rate([[a], [b]], ranks=[1, 1])
    delta_low = result_low[0][0].mu - mu_a

    # Higher alpha should give a more positive (or less negative) mu change
    assert delta_high > delta_low, (
        f"alpha=0.8 delta {delta_high:.6f} should exceed "
        f"alpha=0.2 delta {delta_low:.6f} for {model.__name__}"
    )


@pytest.mark.parametrize("model", _PAIRWISE_MODELS)
@pytest.mark.parametrize("mu_a", [35.0, 45.0, 60.0])
def test_draw_never_worse_than_loss_for_favourite(model, mu_a: float) -> None:
    """Regression test: a draw must never penalise the favourite more than a loss.

    This was the core pathology of the old interval-based draw model (vt/wt)
    in Thurstone-Mosteller, where large skill gaps caused the draw update to
    exceed the loss update in magnitude.
    """
    mu_b = 10.0
    m = model()

    # Loss for A (A ranked 2nd)
    a_loss = m.rating(mu=mu_a, sigma=m.sigma)
    b_loss = m.rating(mu=mu_b, sigma=m.sigma)
    loss_result = m.rate([[a_loss], [b_loss]], ranks=[2, 1])
    loss_penalty = mu_a - loss_result[0][0].mu  # positive = mu went down

    # Draw
    a_draw = m.rating(mu=mu_a, sigma=m.sigma)
    b_draw = m.rating(mu=mu_b, sigma=m.sigma)
    draw_result = m.rate([[a_draw], [b_draw]], ranks=[1, 1])
    draw_penalty = mu_a - draw_result[0][0].mu  # positive = mu went down

    assert draw_penalty <= loss_penalty + 1e-10, (
        f"Draw penalty {draw_penalty:.6f} exceeds loss penalty "
        f"{loss_penalty:.6f} for {model.__name__} at mu_a={mu_a}"
    )
