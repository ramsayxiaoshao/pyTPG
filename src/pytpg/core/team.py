"""Team value object."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pytpg.core._validation import require_non_negative_integer
from pytpg.core.identifiers import ActionID, LearnerID, TeamID
from pytpg.core.learner import Learner

if TYPE_CHECKING:
    from pytpg.runtime.team_runtime import DeterministicRuntime


@dataclass(frozen=True, slots=True)
class Team:
    """A stable ID and non-empty ordered collection of learners."""

    id: TeamID
    learners: tuple[Learner, ...]

    def __post_init__(self) -> None:
        require_non_negative_integer(self.id, "team ID")
        if not isinstance(self.learners, tuple):
            raise TypeError("team learners must be a tuple")
        if not self.learners:
            raise ValueError("team must contain at least one learner")
        if not all(isinstance(item, Learner) for item in self.learners):
            raise TypeError("team learners contain an unsupported value")

        learner_ids: set[LearnerID] = set()
        for learner in self.learners:
            if learner.id in learner_ids:
                msg = f"team {self.id} repeats learner ID {learner.id}"
                raise ValueError(msg)
            learner_ids.add(learner.id)

    def act(
        self,
        observation: Sequence[float],
        *,
        runtime: DeterministicRuntime | None = None,
    ) -> ActionID:
        """Select an atomic action using deterministic Milestone 1 semantics.

        Omitting `runtime` infers the smallest compatible register file. Research
        experiments should pass an explicit runtime to make shape configuration
        part of their recorded setup.
        """

        if runtime is None:
            from pytpg.runtime.team_runtime import DeterministicRuntime

            runtime = DeterministicRuntime.infer_for_team(self, observation)
        return ActionID(runtime.act(self, observation))


__all__ = ["Team"]
