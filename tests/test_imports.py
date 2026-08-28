"""Smoke tests for the intentionally small public package namespaces."""


def test_top_level_package_imports() -> None:
    import tpg

    assert tpg.__version__ == "0.1.0"
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
        "InvalidInstructionError",
        "InvalidObservationError",
        "Operator",
        "OperatorArityError",
        "OperatorExecutionError",
        "OperatorFunction",
        "OperatorRegistry",
        "ProgramExecutor",
        "RegisterAccessError",
        "RegisterFile",
        "RuntimeConfig",
        "RuntimeConfigurationError",
        "TPGExecutionError",
        "TeamReferenceRequiresGraphError",
        "UnknownOperatorError",
        "default_operator_registry",
    }

    assert set(runtime.__all__) == expected
