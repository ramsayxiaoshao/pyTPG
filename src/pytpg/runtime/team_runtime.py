"""Deterministic bidding and action selection for one team."""

from __future__ import annotations

from collections.abc import Sequence

from pytpg.runtime._model import LearnerLike, TeamLike
from pytpg.runtime.config import RuntimeConfig
from pytpg.runtime.errors import TeamReferenceRequiresGraphError
from pytpg.runtime.executor import ProgramExecutor
from pytpg.runtime.inference import infer_register_count
from pytpg.runtime.operators import OperatorRegistry


class DeterministicRuntime:
    """Reference semantics for independent learner bids and stable selection."""

    __slots__ = ("executor",)

    def __init__(
        self,
        config: RuntimeConfig,
        operators: OperatorRegistry | None = None,
    ) -> None:
        self.executor = ProgramExecutor(config, operators)

    @classmethod
    def infer_for_team(
        cls,
        team: TeamLike,
        observation: Sequence[float],
        operators: OperatorRegistry | None = None,
    ) -> DeterministicRuntime:
        """Infer only shape configuration for the `Team.act` convenience API."""

        return cls(
            RuntimeConfig(
                input_size=len(observation),
                register_count=infer_register_count((team,)),
            ),
            operators,
        )

    def bids(
        self,
        team: TeamLike,
        observation: Sequence[float],
    ) -> tuple[float, ...]:
        """Evaluate every learner independently in stable team order."""

        return tuple(
            self.executor.execute(learner.program, observation).output
            for learner in team.learners
        )

    def select(self, team: TeamLike, observation: Sequence[float]) -> LearnerLike:
        """Select the first learner having the maximum finite raw bid."""

        bids = self.bids(team, observation)
        winner_index = max(range(len(team.learners)), key=bids.__getitem__)
        return team.learners[winner_index]

    def act(self, team: TeamLike, observation: Sequence[float]) -> int:
        """Return an atomic action or reject graph traversal in Milestone 1."""

        winner = self.select(team, observation)
        if winner.action.kind != "atomic":
            msg = (
                f"learner {winner.id} selected team {winner.action.team_id}; "
                "use GraphRuntime for team-reference traversal"
            )
            raise TeamReferenceRequiresGraphError(msg)
        return winner.action.action_id


__all__ = ["DeterministicRuntime"]
