"""Batch processing for openskill rating models.

Enables parallel processing of thousands of games with automatic
wave-based partitioning for entity safety and reproducible ordering.

Architecture:
    Games are partitioned into "waves" where no entity (player) appears
    in more than one game within the same wave. This guarantees:

    1. **Safety**: No concurrent writes to the same entity's ratings.
    2. **Reproducibility**: Games are processed in input order across waves.
    3. **Parallelism**: All games within a wave can run simultaneously.

    A background thread builds waves ahead of time (producer) while
    worker processes/threads consume and process them.

Threading Model:
    - Free-threaded Python (3.13t/3.14t): Uses ``ThreadPoolExecutor``
      for true parallel execution with shared memory (zero-copy).
    - GIL Python: Uses ``ProcessPoolExecutor`` with per-worker model
      initialization and serialized results. Wave partitioning ensures
      no shared-state conflicts.

Usage::

    from openskill.models import PlackettLuce
    from openskill.batch import BatchProcessor, Game

    model = PlackettLuce()
    processor = BatchProcessor(model, n_workers=10)
    games = [
        Game(teams=[["alice", "bob"], ["carol", "dave"]], ranks=[1, 2]),
        Game(teams=[["alice", "eve"], ["frank", "grace"]], scores=[10, 20]),
    ]

    ratings = processor.process(games)
    # {"alice": (mu, sigma), "bob": (mu, sigma), ...}
"""

import importlib
import multiprocessing
import queue
import sys
import sysconfig
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

__all__ = ["Game", "BatchProcessor", "partition_waves"]


def _is_free_threaded() -> bool:
    """Check if running on free-threaded Python with GIL disabled."""
    build_supports = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    gil_disabled = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
    return build_supports and gil_disabled


_FREE_THREADED: bool = _is_free_threaded()

# Global model reference for worker processes (set by _init_worker).
_worker_model: Any = None


@dataclass
class Game:
    """
    Descriptor for a single game in batch processing.

    :param teams: List of teams, each a list of entity ID strings.
    :param ranks: Optional ranks (lower = better). Mutually exclusive
                  with scores.
    :param scores: Optional scores (higher = better). Mutually exclusive
                   with ranks.
    :param weights: Optional per-player contribution weights.
    """

    teams: list[list[str]]
    ranks: list[float] | None = None
    scores: list[float] | None = None
    weights: list[list[float]] | None = None


def partition_waves(games: list[Game]) -> list[list[tuple[int, Game]]]:
    """
    Partition games into conflict-free waves.

    Uses a greedy algorithm: each game is assigned to the earliest
    wave that has no entity overlap. This preserves input ordering
    for reproducibility.

    :param games: Games in processing order.
    :return: List of waves. Each wave is a list of
             ``(original_index, game)`` tuples.
    """
    waves: list[list[tuple[int, Game]]] = []
    wave_entities: list[set[str]] = []

    for idx, game in enumerate(games):
        game_ents: set[str] = set()
        for team in game.teams:
            game_ents.update(team)

        placed = False
        for w_idx in range(len(waves)):
            if game_ents.isdisjoint(wave_entities[w_idx]):
                waves[w_idx].append((idx, game))
                wave_entities[w_idx].update(game_ents)
                placed = True
                break

        if not placed:
            waves.append([(idx, game)])
            wave_entities.append(game_ents.copy())

    return waves


def _partition_waves_generator(
    games: list[Game],
) -> "list[list[tuple[int, Game]]]":
    """
    Yield waves incrementally as a generator.

    Builds one maximal independent set at a time by scanning remaining
    games. Enables pipelined processing where computation starts before
    all waves are determined.

    :param games: Games in processing order.
    :yields: Waves as lists of ``(original_index, game)`` tuples.
    """
    remaining = list(range(len(games)))

    result: list[list[tuple[int, Game]]] = []
    while remaining:
        wave: list[tuple[int, Game]] = []
        wave_entities: set[str] = set()
        still_remaining: list[int] = []

        for idx in remaining:
            game = games[idx]
            game_ents: set[str] = set()
            for team in game.teams:
                game_ents.update(team)

            if game_ents.isdisjoint(wave_entities):
                wave.append((idx, game))
                wave_entities.update(game_ents)
            else:
                still_remaining.append(idx)

        result.append(wave)
        remaining = still_remaining

    return result


def _extract_model_config(model: Any) -> tuple[str, str, dict[str, Any]]:
    """
    Extract picklable model constructor arguments.

    Used to reconstruct the model in worker processes.
    Custom gamma functions must be picklable (module-level functions
    work; lambdas will raise ``PickleError`` at submission time).
    """
    model_class = type(model)
    module = model_class.__module__
    class_name = model_class.__name__

    kwargs: dict[str, Any] = {
        "mu": model.mu,
        "sigma": model.sigma,
        "beta": model.beta,
        "kappa": model.kappa,
        "tau": model.tau,
        "limit_sigma": model.limit_sigma,
    }

    if hasattr(model, "gamma"):
        kwargs["gamma"] = model.gamma
    if hasattr(model, "margin"):
        kwargs["margin"] = model.margin
    if hasattr(model, "balance"):
        kwargs["balance"] = model.balance
    if hasattr(model, "weight_bounds"):
        kwargs["weight_bounds"] = model.weight_bounds

    return module, class_name, kwargs


def _init_worker(
    module_name: str, class_name: str, model_kwargs: dict[str, Any]
) -> None:
    """Initialize model in worker process (called once per process)."""
    global _worker_model
    mod = importlib.import_module(module_name)
    model_class = getattr(mod, class_name)
    _worker_model = model_class(**model_kwargs)


def _worker_rate_game(
    args: tuple[
        list[list[int]],
        list[list[float]],
        list[list[float]],
        list[float] | None,
        list[float] | None,
        list[list[float]] | None,
    ],
) -> list[tuple[int, float, float]]:
    """
    Rate a single game in a worker process.

    Uses the global ``_worker_model`` set by :func:`_init_worker`.

    :return: List of ``(entity_array_index, new_mu, new_sigma)``.
    """
    team_indices, team_mus, team_sigmas, ranks, scores, weights = args

    teams = []
    for t_mus, t_sigs in zip(team_mus, team_sigmas):
        team = [
            _worker_model.rating(mu=m, sigma=s) for m, s in zip(t_mus, t_sigs)
        ]
        teams.append(team)

    rate_kwargs: dict[str, Any] = {}
    if ranks is not None:
        rate_kwargs["ranks"] = list(ranks)
    if scores is not None:
        rate_kwargs["scores"] = list(scores)
    if weights is not None:
        rate_kwargs["weights"] = [list(w) for w in weights]

    result = _worker_model.rate(teams, **rate_kwargs)

    updates: list[tuple[int, float, float]] = []
    for team_idx, team in enumerate(result):
        for player_idx, player in enumerate(team):
            updates.append(
                (team_indices[team_idx][player_idx], player.mu, player.sigma)
            )

    return updates


class BatchProcessor:
    """
    Parallel batch processor for openskill rating models.

    Partitions games into conflict-free waves and processes each
    wave in parallel. A background thread builds waves ahead of
    computation so workers never wait for partitioning.

    :param model: An openskill model instance (e.g. ``PlackettLuce()``).
    :param n_workers: Number of parallel workers. Defaults to CPU count.
    :param pipeline: If ``True``, build waves in a background thread
                     while workers process earlier waves.
    """

    def __init__(
        self,
        model: Any,
        n_workers: int | None = None,
        pipeline: bool = True,
    ):
        self.model = model
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.pipeline = pipeline
        self._use_threads: bool = _FREE_THREADED
        self._model_config = _extract_model_config(model)

    def process(
        self,
        games: list[Game],
        initial_ratings: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Process all games and return final ratings.

        Games are processed in input order. Within each wave, games
        that share no entities run in parallel. Between waves, updates
        are committed so later games see earlier results.

        :param games: Games in chronological/processing order.
        :param initial_ratings: Pre-existing ``{entity_id: (mu, sigma)}``.
        :return: Final ``{entity_id: (mu, sigma)}`` for all entities.
        """
        if not games:
            return dict(initial_ratings) if initial_ratings else {}

        # Build entity registry
        entity_to_idx: dict[str, int] = {}
        for game in games:
            for team in game.teams:
                for eid in team:
                    if eid not in entity_to_idx:
                        entity_to_idx[eid] = len(entity_to_idx)

        n = len(entity_to_idx)
        mus: list[float] = [self.model.mu] * n
        sigmas: list[float] = [self.model.sigma] * n

        if initial_ratings:
            for eid, (mu, sigma) in initial_ratings.items():
                if eid in entity_to_idx:
                    idx = entity_to_idx[eid]
                    mus[idx] = mu
                    sigmas[idx] = sigma

        # Dispatch to processing strategy
        if self.n_workers <= 1:
            self._process_sequential(games, entity_to_idx, mus, sigmas)
        elif self.pipeline:
            self._process_pipelined(games, entity_to_idx, mus, sigmas)
        else:
            waves = partition_waves(games)
            self._process_waves(waves, entity_to_idx, mus, sigmas)

        # Build result
        idx_to_eid = {v: k for k, v in entity_to_idx.items()}
        return {idx_to_eid[i]: (mus[i], sigmas[i]) for i in range(n)}

    def _process_sequential(
        self,
        games: list[Game],
        entity_to_idx: dict[str, int],
        mus: list[float],
        sigmas: list[float],
    ) -> None:
        """Process all games sequentially (single worker)."""
        for game in games:
            self._rate_game_inplace(game, entity_to_idx, mus, sigmas)

    def _process_pipelined(
        self,
        games: list[Game],
        entity_to_idx: dict[str, int],
        mus: list[float],
        sigmas: list[float],
    ) -> None:
        """
        Producer-consumer pipeline.

        A background thread builds waves via the incremental generator
        while workers process each wave as it arrives. The builder is
        lightweight (set operations) and runs in the main process. On
        GIL Python the worker processes run independently so the builder
        thread has near-zero impact on compute throughput.
        """
        wave_q: queue.Queue[list[tuple[int, Game]] | None] = queue.Queue(
            maxsize=self.n_workers * 2
        )

        def build_waves_ahead() -> None:
            for wave in _partition_waves_generator(games):
                wave_q.put(wave)
            wave_q.put(None)  # sentinel

        builder = threading.Thread(target=build_waves_ahead, daemon=True)
        builder.start()

        try:
            if self._use_threads:
                with ThreadPoolExecutor(
                    max_workers=self.n_workers
                ) as executor:
                    while True:
                        wave = wave_q.get()
                        if wave is None:
                            break
                        self._execute_wave_threaded(
                            wave, entity_to_idx, mus, sigmas, executor
                        )
            else:
                module, class_name, model_kwargs = self._model_config
                with ProcessPoolExecutor(
                    max_workers=self.n_workers,
                    initializer=_init_worker,
                    initargs=(module, class_name, model_kwargs),
                ) as executor:
                    while True:
                        wave = wave_q.get()
                        if wave is None:
                            break
                        self._execute_wave_multiprocess(
                            wave, entity_to_idx, mus, sigmas, executor
                        )
        finally:
            builder.join()

    def _process_waves(
        self,
        waves: list[list[tuple[int, Game]]],
        entity_to_idx: dict[str, int],
        mus: list[float],
        sigmas: list[float],
    ) -> None:
        """Process pre-built waves (non-pipelined)."""
        if self._use_threads:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                for wave in waves:
                    self._execute_wave_threaded(
                        wave, entity_to_idx, mus, sigmas, executor
                    )
        else:
            module, class_name, model_kwargs = self._model_config
            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_init_worker,
                initargs=(module, class_name, model_kwargs),
            ) as executor:
                for wave in waves:
                    self._execute_wave_multiprocess(
                        wave, entity_to_idx, mus, sigmas, executor
                    )

    def _execute_wave_threaded(
        self,
        wave: list[tuple[int, Game]],
        entity_to_idx: dict[str, int],
        mus: list[float],
        sigmas: list[float],
        executor: ThreadPoolExecutor,
    ) -> None:
        """Process a wave with ThreadPoolExecutor (free-threaded Python)."""
        if len(wave) <= 2:
            for _, game in wave:
                self._rate_game_inplace(game, entity_to_idx, mus, sigmas)
            return

        futures = [
            executor.submit(
                self._rate_game_inplace, game, entity_to_idx, mus, sigmas
            )
            for _, game in wave
        ]
        for f in futures:
            f.result()

    def _execute_wave_multiprocess(
        self,
        wave: list[tuple[int, Game]],
        entity_to_idx: dict[str, int],
        mus: list[float],
        sigmas: list[float],
        executor: ProcessPoolExecutor,
    ) -> None:
        """Process a wave with ProcessPoolExecutor (GIL Python)."""
        if len(wave) <= 2:
            for _, game in wave:
                self._rate_game_inplace(game, entity_to_idx, mus, sigmas)
            return

        work_items = []
        for _, game in wave:
            team_indices: list[list[int]] = []
            team_mus: list[list[float]] = []
            team_sigmas: list[list[float]] = []

            for team_ids in game.teams:
                indices = [entity_to_idx[eid] for eid in team_ids]
                team_indices.append(indices)
                team_mus.append([mus[i] for i in indices])
                team_sigmas.append([sigmas[i] for i in indices])

            work_items.append(
                (
                    team_indices,
                    team_mus,
                    team_sigmas,
                    game.ranks,
                    game.scores,
                    game.weights,
                )
            )

        chunksize = max(1, len(work_items) // (self.n_workers * 4))
        for updates in executor.map(
            _worker_rate_game, work_items, chunksize=chunksize
        ):
            for idx, new_mu, new_sigma in updates:
                mus[idx] = new_mu
                sigmas[idx] = new_sigma

    def _rate_game_inplace(
        self,
        game: Game,
        entity_to_idx: dict[str, int],
        mus: list[float],
        sigmas: list[float],
    ) -> None:
        """Rate a single game, updating mus/sigmas arrays in place."""
        teams = []
        team_indices: list[list[int]] = []
        for team_ids in game.teams:
            team = []
            indices = []
            for eid in team_ids:
                idx = entity_to_idx[eid]
                team.append(self.model.rating(mu=mus[idx], sigma=sigmas[idx]))
                indices.append(idx)
            teams.append(team)
            team_indices.append(indices)

        rate_kwargs: dict[str, Any] = {}
        if game.ranks is not None:
            rate_kwargs["ranks"] = list(game.ranks)
        if game.scores is not None:
            rate_kwargs["scores"] = list(game.scores)
        if game.weights is not None:
            rate_kwargs["weights"] = [list(w) for w in game.weights]

        result = self.model.rate(teams, **rate_kwargs)

        for team_idx, team in enumerate(result):
            for player_idx, player in enumerate(team):
                idx = team_indices[team_idx][player_idx]
                mus[idx] = player.mu
                sigmas[idx] = player.sigma
