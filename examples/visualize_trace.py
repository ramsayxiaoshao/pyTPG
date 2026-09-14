"""Record a deterministic cyclic policy and render an offline replay."""

from pathlib import Path

from pytpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
    InputIndex,
    InputOperand,
    Instruction,
    Learner,
    LearnerID,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    Team,
    TeamID,
    TeamReference,
    TPGGraph,
)
from pytpg.runtime import GraphRuntime, RuntimeConfig
from pytpg.visualization import TraceSession, render_html


def example_graph() -> TPGGraph:
    def learner(
        identifier: int, value: float | None, action: AtomicAction | TeamReference
    ) -> Learner:
        operand = (
            InputOperand(InputIndex(0)) if value is None else ConstantOperand(value)
        )
        return Learner(
            LearnerID(identifier),
            Program(
                ProgramID(identifier + 10),
                (Instruction(OperatorName("identity"), RegisterIndex(0), (operand,)),),
            ),
            action,
        )

    root = Team(
        TeamID(0),
        (
            learner(0, None, TeamReference(TeamID(1))),
            learner(1, 0.0, AtomicAction(ActionID(0))),
        ),
    )
    child = Team(
        TeamID(1),
        (
            learner(2, 100.0, TeamReference(TeamID(0))),
            learner(3, 0.5, AtomicAction(ActionID(7))),
            learner(4, None, AtomicAction(ActionID(8))),
        ),
    )
    return TPGGraph((root, child), (root.id,))


def main() -> None:
    graph = example_graph()
    runtime = GraphRuntime(RuntimeConfig(input_size=1, register_count=1))
    session = TraceSession.from_graph(graph)
    for step, value in enumerate((-1.0, 0.25, 1.0, 0.0, 0.4, 2.0)):
        result = runtime.traverse_detailed(graph, (value,))
        session.append(result, step=step, agent_id="demo", metadata={"signal": value})
    json_path = Path("example_trace.json").resolve()
    session.to_json(json_path)
    print(json_path)
    print(render_html(session, "example_trace.html"))


if __name__ == "__main__":
    main()
