"""Run a small fixed-seed TPG evolution on Gymnasium CartPole."""

from tpg.adapters import (
    GymnasiumAdapter,
    GymnasiumEvaluationConfig,
    GymnasiumFitness,
    make_gymnasium_environment,
)
from tpg.evaluation import SequentialEvaluator
from tpg.evolution import (
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

seed = 42
rng = create_rng(seed)

with GymnasiumAdapter(make_gymnasium_environment("CartPole-v1")) as environment:
    input_size = environment.input_size
    n_actions = environment.n_actions

initializer = GraphInitializer(
    InitializationConfig(
        GenomeConfig(input_size, register_count=4, n_actions=n_actions),
        population_size=12,
        learners_per_team=3,
        program_length=4,
    )
)
population = PopulationInitializer(initializer).initialize(rng)
reproducer = Reproducer(
    TournamentSelection(tournament_size=3),
    default_mutation(initializer.factory),
    ReproductionConfig(elite_count=1, mutation_steps=2),
)
fitness = GymnasiumFitness(
    lambda: make_gymnasium_environment("CartPole-v1"),
    GymnasiumEvaluationConfig.from_seed(
        seed,
        episodes=3,
        max_episode_steps=500,
    ),
)
result = EvolutionEngine(SequentialEvaluator(), reproducer).run(
    population,
    fitness,
    generations=5,
    rng=rng,
)

print(f"seed={seed}")
print(f"best mean return by generation: {result.best_fitness_history}")
print(f"final best mean return: {result.best.fitness:.1f}")
