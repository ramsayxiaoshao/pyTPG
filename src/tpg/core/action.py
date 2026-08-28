"""Environment-independent action value objects."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

from tpg.core._validation import require_non_negative_integer
from tpg.core.identifiers import ActionID, TeamID


@dataclass(frozen=True, slots=True)
class AtomicAction:
    """An opaque action ID that terminates graph traversal."""

    action_id: ActionID

    @property
    def kind(self) -> Literal["atomic"]:
        """Return the stable action discriminator used by the runtime."""

        return "atomic"

    def __post_init__(self) -> None:
        require_non_negative_integer(self.action_id, "action ID")


@dataclass(frozen=True, slots=True)
class TeamReference:
    """An edge to another team in the containing graph."""

    team_id: TeamID

    @property
    def kind(self) -> Literal["team_reference"]:
        """Return the stable action discriminator used by the runtime."""

        return "team_reference"

    def __post_init__(self) -> None:
        require_non_negative_integer(self.team_id, "referenced team ID")


Action: TypeAlias = AtomicAction | TeamReference

__all__ = ["Action", "AtomicAction", "TeamReference"]
