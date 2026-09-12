"""Run a tiny deterministic evolution on a two-case contextual bandit."""

from pytpg.evaluation import ContextualBandit, SequentialEvaluator
from pytpg.evolution import (
    EvolutionEngine,
    GenomeConfig,
    GraphInitializer,
    InitializationConfig,
    PopulationInitializer,
    Reproducer,
    ReproductionConfig,
    TournamentSelection,
    create_rng,
    default_mutation,
)

seed = 0
rng = create_rng(seed)
initializer = GraphInitializer(
    InitializationConfig(
        GenomeConfig(input_size=1, register_count=4, n_actions=2),
        population_size=4,
        learners_per_team=2,
        program_length=2,
    )
)
population = PopulationInitializer(initializer).initialize(rng)
reproducer = Reproducer(
    TournamentSelection(tournament_size=2),
    default_mutation(initializer.factory),
    ReproductionConfig(elite_count=1, mutation_steps=2),
)
result = EvolutionEngine(SequentialEvaluator(), reproducer).run(
    population,
    ContextualBandit.signed_binary(),
    generations=20,
    rng=rng,
)

print(f"seed={seed}")
print(f"best fitness by generation: {result.best_fitness_history}")
print(f"final best fitness: {result.best.fitness:.3f}")
