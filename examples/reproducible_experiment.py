"""Run, checkpoint, load, and resume one deterministic toy experiment."""

from pathlib import Path

from tpg.config import TPGConfig
from tpg.evaluation import ContextualBandit
from tpg.evolution import GenomeConfig, InitializationConfig, ReproductionConfig
from tpg.experiment import resume_experiment, run_experiment
from tpg.serialization import load_checkpoint, save_checkpoint

config = TPGConfig(
    initialization=InitializationConfig(
        GenomeConfig(input_size=1, register_count=4, n_actions=2),
        population_size=8,
        learners_per_team=2,
        program_length=2,
    ),
    tournament_size=2,
    reproduction=ReproductionConfig(elite_count=1, mutation_steps=2),
)
task = ContextualBandit.signed_binary()

initial = run_experiment(
    config,
    task,
    generations=5,
    master_seed=42,
    name="signed-binary-bandit",
    tags=(("purpose", "example"),),
)
checkpoint_path = Path("bandit-generation-5.tpg.json")
save_checkpoint(initial.checkpoint(), checkpoint_path)

resumed = resume_experiment(load_checkpoint(checkpoint_path), task, generations=5)
print(f"checkpoint: {checkpoint_path}")
print(f"final generation: {resumed.evolution.final_population.generation}")
print(f"best fitness: {resumed.evolution.best.fitness:.3f}")
print(f"config digest: {resumed.config.digest}")
