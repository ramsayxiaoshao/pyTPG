"""Render the best policy topology from a saved evaluated checkpoint."""

import argparse
from pathlib import Path

from pytpg.serialization import load_checkpoint
from pytpg.visualization import TraceSession, render_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoint_topology.html"))
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    best = checkpoint.evaluated.best
    session = TraceSession.from_graph(best.graph)
    print(f"Generation: {checkpoint.evaluated.generation}")
    print(f"Best individual position: {best.position}; fitness: {best.fitness}")
    print(f"Teams: {len(best.graph.teams)}; roots: {best.graph.root_team_ids}")
    print("Topology only: checkpoint does not contain decision observations or bids.")
    print(render_html(session, args.output))


if __name__ == "__main__":
    main()
