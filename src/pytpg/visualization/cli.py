"""Command-line offline trace rendering."""

import argparse
import webbrowser
from collections.abc import Sequence

from pytpg.visualization.model import TraceSession
from pytpg.visualization.render import VisualizationDependencyError, render_html


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a pyTPG execution trace offline"
    )
    parser.add_argument("trace", help="TraceSession JSON file")
    parser.add_argument("--output", default="trace.html", help="Output HTML path")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    try:
        path = render_html(TraceSession.from_json(args.trace), args.output)
    except (VisualizationDependencyError, ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(path)
    if args.open_browser:
        webbrowser.open(path.as_uri())
    return 0
