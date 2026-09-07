"""Recall a signal after three neutral observations using composed memory."""

from tpg.core import (
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
    TPGGraph,
)
from tpg.memory import ObservationHistoryMemory, StatefulGraphRuntime
from tpg.runtime import GraphRuntime, RuntimeConfig


def recall_graph() -> TPGGraph:
    """Choose action 0 for a negative remembered signal and 1 for positive."""

    remembered_signal = InputOperand(InputIndex(1))
    negative_bid = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("subtract"),
                RegisterIndex(0),
                (ConstantOperand(0.0), remembered_signal),
            ),
        ),
    )
    positive_bid = Program(
        ProgramID(1),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (remembered_signal,),
            ),
        ),
    )
    team = Team(
        TeamID(0),
        (
            Learner(LearnerID(0), negative_bid, AtomicAction(ActionID(0))),
            Learner(LearnerID(1), positive_bid, AtomicAction(ActionID(1))),
        ),
    )
    return TPGGraph((team,), (team.id,))


graph = recall_graph()
controller = StatefulGraphRuntime(
    GraphRuntime(RuntimeConfig(input_size=4, register_count=1)),
    ObservationHistoryMemory(observation_size=1, depth=3),
    observation_size=1,
)

for signal in (-1.0, 1.0):
    controller.reset_episode()
    controller.act(graph, (signal,))
    controller.act(graph, (0.0,))
    controller.act(graph, (0.0,))
    recalled_action = controller.act(graph, (0.0,))
    controller.end_episode()
    expected_action = 0 if signal < 0.0 else 1
    print(
        f"signal={signal:+.0f}, recalled_action={recalled_action}, "
        f"expected={expected_action}"
    )
