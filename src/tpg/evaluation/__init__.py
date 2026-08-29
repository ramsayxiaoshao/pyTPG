"""Fitness evaluation protocols, reference evaluator, and toy tasks."""

from tpg.evaluation.evaluator import Evaluator, FitnessFunction, SequentialEvaluator
from tpg.evaluation.statistics import (
    GenerationStatistics,
    OperatorStatistics,
    RunStatistics,
)
from tpg.evaluation.toy import ContextualBandit, ContextualBanditCase

__all__ = [
    "ContextualBandit",
    "ContextualBanditCase",
    "Evaluator",
    "FitnessFunction",
    "GenerationStatistics",
    "OperatorStatistics",
    "RunStatistics",
    "SequentialEvaluator",
]
