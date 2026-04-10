#!/usr/bin/env python3
"""
Real-time viewer for Station Concordia simulation progress.

Displays agent decisions and LLM responses as they happen.

Usage:
    python tools/view_concordia_live.py [--output-file PATH]
"""

import argparse
import json
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")
    print("Falling back to simple text output.\n")


class ConcordiaViewer:
    """Real-time viewer for Concordia simulation."""

    def __init__(self, output_file: str):
        """Initialize viewer."""
        self.output_file = Path(output_file)
        self.last_size = 0
        self.decisions_seen = set()

        if RICH_AVAILABLE:
            self.console = Console()

    def parse_concordia_action(self, action: str) -> dict:
        """
        Parse Concordia's multi-question action format.

        Returns dict with questions and answers extracted.
        """
        result = {
            "self_perception": "",
            "situation": "",
            "risk": "",
            "social": "",
            "strategy": "",
            "final_action": "",
        }

        lines = action.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if "What kind of person is" in line:
                current_section = "self_perception"
            elif "What situation is" in line:
                current_section = "situation"
            elif "How dangerous is" in line:
                current_section = "risk"
            elif "What are other people doing" in line:
                current_section = "social"
            elif "What would a person like" in line:
                current_section = "strategy"
            elif "What will you do next" in line or "Exercise:" in line:
                current_section = "final_action"
            elif line.startswith("Answer:") and current_section:
                answer = line.replace("Answer:", "").strip()
                if answer:
                    result[current_section] = answer
            elif current_section and line and not line.startswith("-"):
                # Continuation of previous answer
                if result[current_section]:
                    result[current_section] += " " + line
                else:
                    result[current_section] = line

        return result

    def format_decision_rich(self, agent_id: str, decision: dict) -> Panel:
        """Format a decision using rich formatting."""
        time_val = decision.get("time", 0.0)
        action = decision.get("action", "")
        reasoning = decision.get("reasoning", {})
        translated = decision.get("translated", {})

        # Create content
        content = []

        # Time
        content.append(f"[bold cyan]Time:[/bold cyan] {time_val:.1f}s\n")

        # LLM Responses
        content.append("[bold yellow]═══ LLM REASONING ═══[/bold yellow]\n")

        if reasoning.get("self_perception"):
            content.append(f"[bold]Self:[/bold] {reasoning['self_perception']}\n")

        if reasoning.get("situation"):
            content.append(f"[bold]Situation:[/bold] {reasoning['situation']}\n")

        if reasoning.get("risk"):
            content.append(f"[bold]Risk:[/bold] {reasoning['risk']}\n")

        if reasoning.get("social"):
            content.append(f"[bold]Social:[/bold] {reasoning['social']}\n")

        if reasoning.get("strategy"):
            content.append(f"[bold]Strategy:[/bold] {reasoning['strategy']}\n")

        # Final action
        content.append(f"[bold green]Action:[/bold green] {action}\n")

        # Game Master Translation
        content.append("\n[bold magenta]═══ GAME MASTER ═══[/bold magenta]\n")
        content.append(f"[bold]Type:[/bold] {translated.get('action_type', 'unknown')}\n")
        content.append(f"[bold]Target:[/bold] {translated.get('target', 'none')}\n")
        content.append(f"[bold]Confidence:[/bold] {translated.get('confidence', 0.0):.1%}\n")
        content.append(f"[bold]Reasoning:[/bold] {translated.get('reasoning', 'none')}")

        panel = Panel(
            "".join(content),
            title=f"[bold white]{agent_id}[/bold white]",
            border_style="blue",
            expand=False,
        )

        return panel

    def format_decision_simple(self, agent_id: str, decision: dict) -> str:
        """Format a decision using simple text."""
        time_val = decision.get("time", 0.0)
        action = decision.get("action", "")
        reasoning = decision.get("reasoning", {})
        translated = decision.get("translated", {})

        output = []
        output.append("=" * 80)
        output.append(f"AGENT: {agent_id} | TIME: {time_val:.1f}s")
        output.append("=" * 80)

        output.append("\n--- LLM REASONING ---")
        if reasoning.get("self_perception"):
            output.append(f"Self: {reasoning['self_perception']}")
        if reasoning.get("situation"):
            output.append(f"Situation: {reasoning['situation']}")
        if reasoning.get("risk"):
            output.append(f"Risk: {reasoning['risk']}")
        if reasoning.get("social"):
            output.append(f"Social: {reasoning['social']}")
        if reasoning.get("strategy"):
            output.append(f"Strategy: {reasoning['strategy']}")
        output.append(f"Action: {action}")

        output.append("\n--- GAME MASTER ---")
        output.append(f"Type: {translated.get('action_type', 'unknown')}")
        output.append(f"Target: {translated.get('target', 'none')}")
        output.append(f"Confidence: {translated.get('confidence', 0.0):.1%}")
        output.append(f"Reasoning: {translated.get('reasoning', 'none')}")
        output.append("")

        return "\n".join(output)

    def watch(self):
        """Watch the output file for changes."""
        print(f"Watching: {self.output_file}")
        print("Waiting for simulation to start...\n")

        try:
            while True:
                if not self.output_file.exists():
                    time.sleep(1)
                    continue

                try:
                    with open(self.output_file) as f:
                        data = json.load(f)

                    agent_decisions = data.get("agent_decisions", {})

                    # Check for new decisions
                    for agent_id, agent_data in agent_decisions.items():
                        decisions = agent_data.get("decisions", [])

                        for decision in decisions:
                            # Create unique key
                            key = f"{agent_id}_{decision.get('time', 0.0)}"

                            if key not in self.decisions_seen:
                                self.decisions_seen.add(key)

                                # Display decision
                                if RICH_AVAILABLE:
                                    panel = self.format_decision_rich(agent_id, decision)
                                    self.console.print(panel)
                                    self.console.print()
                                else:
                                    output = self.format_decision_simple(agent_id, decision)
                                    print(output)

                except json.JSONDecodeError:
                    # File being written, try again
                    pass

                time.sleep(2)  # Check every 2 seconds

        except KeyboardInterrupt:
            print("\n\nViewer stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time viewer for Station Concordia simulation"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="scenarios/station_concordia/output/agent_decisions.json",
        help="Path to output JSON file",
    )

    args = parser.parse_args()

    viewer = ConcordiaViewer(args.output_file)
    viewer.watch()


if __name__ == "__main__":
    main()
