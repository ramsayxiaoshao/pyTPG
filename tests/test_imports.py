"""Smoke tests for the intentionally small public package namespaces."""


def test_top_level_package_imports() -> None:
    import tpg

    assert tpg.__version__ == "0.3.0"
    assert tpg.__all__ == ["__version__"]


def test_core_public_api_imports() -> None:
    from tpg import core

    expected = {
        "Action",
        "ActionID",
        "AtomicAction",
        "ConstantOperand",
        "InputIndex",
        "InputOperand",
        "Instruction",
        "Learner",
        "LearnerID",
        "Operand",
        "OperatorName",
        "Program",
        "ProgramID",
        "RegisterIndex",
        "RegisterOperand",
        "TPGGraph",
        "Team",
        "TeamID",
        "TeamReference",
    }

    assert set(core.__all__) == expected


def test_runtime_public_api_imports() -> None:
    from tpg import runtime

    expected = {
        "DeterministicRuntime",
        "ExecutionResult",
        "GraphRuntime",
        "GraphSummary",
        "GraphValidationError",
        "GraphValidationIssue",
        "GraphValidationReport",
        "GraphValidator",
        "InvalidInstructionError",
        "InvalidObservationError",
        "MissingTeamError",
        "NoEligibleLearnerError",
        "Operator",
        "OperatorArityError",
        "OperatorExecutionError",
        "OperatorFunction",
        "OperatorRegistry",
        "ProgramExecutor",
        "RegisterAccessError",
        "RegisterFile",
        "RootSelectionError",
        "RuntimeConfig",
        "RuntimeConfigurationError",
        "TPGExecutionError",
        "TeamReferenceRequiresGraphError",
        "TraversalActionKind",
        "TraversalLimitExceededError",
        "TraversalResult",
        "TraversalStep",
        "UnknownOperatorError",
        "ValidationSeverity",
        "cyclic_team_ids",
        "default_operator_registry",
        "reachable_team_ids",
        "summarize_graph",
    }

    assert set(runtime.__all__) == expected


def test_evaluation_can_be_imported_before_evolution() -> None:
    from tpg import evaluation, evolution

    assert set(evaluation.__all__) == {
        "ContextualBandit",
        "ContextualBanditCase",
        "Evaluator",
        "FitnessFunction",
        "GenerationStatistics",
        "OperatorStatistics",
        "RunStatistics",
        "SequentialEvaluator",
    }
    assert set(evolution.__all__) == {
        "AddLearnerMutation",
        "AddTeamMutation",
        "ChildRecord",
        "DeleteInstructionMutation",
        "DeleteLearnerMutation",
        "DeleteTeamMutation",
        "EvaluatedIndividual",
        "EvaluatedPopulation",
        "EvolutionConfigurationError",
        "EvolutionEngine",
        "EvolutionError",
        "EvolutionRunResult",
        "GenomeConfig",
        "GenomeFactory",
        "GraphInitializer",
        "InitializationConfig",
        "InsertInstructionMutation",
        "InvalidFitnessError",
        "ModifyInstructionMutation",
        "MutateLearnerAction",
        "MutationConfig",
        "MutationInvariantError",
        "MutationOperator",
        "MutationOutcome",
        "Population",
        "PopulationInitializer",
        "RandomGenerator",
        "Reproducer",
        "ReproductionConfig",
        "ReproductionResult",
        "SelectionStrategy",
        "TournamentSelection",
        "WeightedMutation",
        "create_rng",
        "default_mutation",
    }


def test_research_infrastructure_public_apis() -> None:
    from tpg import callbacks, serialization
    from tpg.config import TPGConfig
    from tpg.experiment import ExperimentResult, resume_experiment, run_experiment
    from tpg.metadata import ExperimentMetadata
    from tpg.seed import SeedManager

    assert set(callbacks.__all__) == {
        "EventRecorder",
        "EvolutionCallback",
        "EvolutionEvent",
        "EvolutionEventKind",
        "LoggingCallback",
    }
    assert set(serialization.__all__) == {
        "CHECKPOINT_FORMAT",
        "CHECKPOINT_FORMAT_VERSION",
        "GRAPH_FORMAT",
        "GRAPH_FORMAT_VERSION",
        "Checkpoint",
        "CheckpointCompatibilityError",
        "InvalidSerializedDataError",
        "RNGSnapshot",
        "SerializationError",
        "UnsupportedFormatVersionError",
        "capture_rng",
        "checkpoint_from_dict",
        "checkpoint_to_dict",
        "dumps_checkpoint",
        "dumps_graph",
        "graph_from_dict",
        "graph_to_dict",
        "load_checkpoint",
        "load_graph",
        "loads_checkpoint",
        "loads_graph",
        "restore_rng",
        "save_checkpoint",
        "save_graph",
    }
    assert TPGConfig.__name__ == "TPGConfig"
    assert SeedManager.__name__ == "SeedManager"
    assert ExperimentMetadata.__name__ == "ExperimentMetadata"
    assert ExperimentResult.__name__ == "ExperimentResult"
    assert callable(run_experiment)
    assert callable(resume_experiment)


def test_gymnasium_adapter_public_api_imports_without_optional_dependency() -> None:
    from tpg import adapters

    assert set(adapters.__all__) == {
        "GymnasiumAdapter",
        "GymnasiumDependencyError",
        "GymnasiumEnvironment",
        "GymnasiumEnvironmentFactory",
        "GymnasiumEpisodeResult",
        "GymnasiumEpisodeStateError",
        "GymnasiumEvaluationConfig",
        "GymnasiumEvaluationError",
        "GymnasiumFitness",
        "GymnasiumIntegrationError",
        "GymnasiumReset",
        "GymnasiumStep",
        "InvalidGymnasiumActionError",
        "InvalidGymnasiumTransitionError",
        "UnsupportedGymnasiumSpaceError",
        "make_gymnasium_environment",
    }
