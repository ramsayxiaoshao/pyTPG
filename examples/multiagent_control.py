"""Run heterogeneous independent and centralized shared TPG control."""

from pytpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
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
from pytpg.multiagent import (
    AgentID,
    AgentSpec,
    IndependentMultiAgentRuntime,
    SharedControlRuntime,
    TPGAgentController,
)


def constant_action_graph(action_id: int) -> TPGGraph:
    program = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (ConstantOperand(1.0),),
            ),
        ),
    )
    team = Team(
        TeamID(0),
        (
            Learner(
                LearnerID(0),
                program,
                AtomicAction(ActionID(action_id)),
            ),
        ),
    )
    return TPGGraph((team,), (team.id,))


specs = (
    AgentSpec(AgentID("scout"), observation_size=1, action_count=2),
    AgentSpec(AgentID("arm"), observation_size=2, action_count=3),
)
observations = {
    AgentID("scout"): (0.5,),
    AgentID("arm"): (1.0, -1.0),
}

independent = IndependentMultiAgentRuntime(
    (
        TPGAgentController.create(specs[0], constant_action_graph(1)),
        TPGAgentController.create(specs[1], constant_action_graph(2)),
    )
)
independent.reset_episode()
independent_actions = independent.step(observations).actions_by_agent
independent.end_episode()

# With action counts (2, 3), encoded action 5 decodes to (1, 2).
shared = SharedControlRuntime.create(specs, constant_action_graph(5))
shared.reset_episode()
shared_step = shared.step(observations)
shared.end_episode()

print(f"independent={independent_actions}")
print(
    f"shared_joint={shared_step.joint_action_id}, "
    f"decoded={shared_step.actions_by_agent}"
)
