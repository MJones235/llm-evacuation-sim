#!/usr/bin/env python3
"""
GUI viewer for Station Concordia simulation progress.

Real-time window that displays agent decisions and LLM responses.
"""

import argparse
import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext


class ConcordiaGUIViewer:
    """GUI viewer for Concordia simulation."""

    def __init__(self, output_file: str, run_id: str):
        """Initialize GUI viewer."""
        self.output_file = Path(output_file)
        self.run_id = run_id
        self.decisions_seen = set()
        self.running = True

        # Create main window
        self.root = tk.Tk()
        self.root.title(f"Concordia Simulation Viewer - Run {run_id}")
        self.root.geometry("1000x700")

        # Configure colors
        self.bg_color = "#1e1e1e"
        self.text_color = "#d4d4d4"
        self.header_color = "#569cd6"
        self.success_color = "#4ec9b0"
        self.warning_color = "#ce9178"
        self.error_color = "#f48771"

        self.root.configure(bg=self.bg_color)

        self._setup_ui()
        self._start_monitoring()

    def _setup_ui(self):
        """Setup the user interface."""
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        title = tk.Label(
            header_frame,
            text=f"Station Concordia - Run {self.run_id}",
            font=("Consolas", 16, "bold"),
            bg=self.bg_color,
            fg=self.header_color,
        )
        title.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header_frame,
            text="● Waiting for simulation...",
            font=("Consolas", 10),
            bg=self.bg_color,
            fg=self.warning_color,
        )
        self.status_label.pack(side=tk.RIGHT)

        # Main scrolled text area
        self.text_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#252526",
            fg=self.text_color,
            insertbackground=self.text_color,
            selectbackground="#264f78",
            relief=tk.FLAT,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure text tags for colors
        self.text_area.tag_config(
            "header", foreground=self.header_color, font=("Consolas", 11, "bold")
        )
        self.text_area.tag_config("agent", foreground="#4ec9b0", font=("Consolas", 12, "bold"))
        self.text_area.tag_config("section", foreground="#dcdcaa", font=("Consolas", 10, "bold"))
        self.text_area.tag_config("label", foreground="#9cdcfe")
        self.text_area.tag_config("value", foreground="#ce9178")
        self.text_area.tag_config("action", foreground="#4ec9b0")
        self.text_area.tag_config("separator", foreground="#3e3e42")

        # Button frame
        button_frame = tk.Frame(self.root, bg=self.bg_color)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        clear_btn = tk.Button(
            button_frame,
            text="Clear Display",
            command=self._clear_display,
            bg="#0e639c",
            fg="white",
            font=("Consolas", 10),
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = tk.Checkbutton(
            button_frame,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor="#3e3e42",
            font=("Consolas", 10),
        )
        auto_scroll_check.pack(side=tk.LEFT, padx=10)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _clear_display(self):
        """Clear the text display."""
        self.text_area.delete(1.0, tk.END)

    def _on_closing(self):
        """Handle window closing."""
        self.running = False
        self.root.destroy()

    def _append_text(self, text, tag=None):
        """Append text to the display."""
        self.text_area.insert(tk.END, text, tag)
        if self.auto_scroll_var.get():
            self.text_area.see(tk.END)

    def _display_decision(self, agent_id: str, decision: dict):
        """Display a decision in the GUI."""
        time_val = decision.get("time", 0.0)
        action = decision.get("action", "")
        reasoning = decision.get("reasoning", {})
        translated = decision.get("translated", {})

        # Agent header
        self._append_text("\n")
        self._append_text("═" * 80 + "\n", "separator")
        self._append_text(f"  {agent_id.upper()}  ", "agent")
        self._append_text(f"│ Time: {time_val:.1f}s\n", "header")
        self._append_text("═" * 80 + "\n", "separator")

        # LLM Reasoning
        self._append_text("\n▼ LLM REASONING\n", "section")

        if reasoning.get("self_perception"):
            self._append_text("  Self: ", "label")
            self._append_text(f"{reasoning['self_perception']}\n", "value")

        if reasoning.get("situation"):
            self._append_text("  Situation: ", "label")
            self._append_text(f"{reasoning['situation']}\n", "value")

        if reasoning.get("risk"):
            self._append_text("  Risk: ", "label")
            self._append_text(f"{reasoning['risk']}\n", "value")

        if reasoning.get("social"):
            self._append_text("  Social: ", "label")
            self._append_text(f"{reasoning['social']}\n", "value")

        if reasoning.get("strategy"):
            self._append_text("  Strategy: ", "label")
            self._append_text(f"{reasoning['strategy']}\n", "value")

        # Final action
        self._append_text("  Action: ", "label")
        self._append_text(f"{action}\n", "action")

        # Game Master
        self._append_text("\n▼ GAME MASTER\n", "section")
        self._append_text("  Type: ", "label")
        self._append_text(f"{translated.get('action_type', 'unknown')}\n", "value")
        self._append_text("  Target: ", "label")
        self._append_text(f"{translated.get('target', 'none')}\n", "value")
        self._append_text("  Confidence: ", "label")
        self._append_text(f"{translated.get('confidence', 0.0):.1%}\n", "value")
        self._append_text("  Reasoning: ", "label")
        self._append_text(f"{translated.get('reasoning', 'none')}\n", "value")

        self._append_text("\n")

    def _monitor_file(self):
        """Monitor the output file for changes."""
        while self.running:
            if not self.output_file.exists():
                time.sleep(1)
                continue

            try:
                with open(self.output_file) as f:
                    data = json.load(f)

                agent_decisions = data.get("agent_decisions", {})

                # Update status
                self.status_label.config(text="● Simulation running", fg=self.success_color)

                # Check for new decisions
                for agent_id, agent_data in agent_decisions.items():
                    decisions = agent_data.get("decisions", [])

                    for decision in decisions:
                        key = f"{agent_id}_{decision.get('time', 0.0)}"

                        if key not in self.decisions_seen:
                            self.decisions_seen.add(key)
                            self._display_decision(agent_id, decision)

            except json.JSONDecodeError:
                # File being written
                pass
            except Exception as e:
                print(f"Error monitoring file: {e}")

            time.sleep(2)

    def _start_monitoring(self):
        """Start monitoring in a background thread."""
        monitor_thread = threading.Thread(target=self._monitor_file, daemon=True)
        monitor_thread.start()

    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GUI viewer for Station Concordia simulation")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--run-id", type=str, required=True, help="Simulation run ID")

    args = parser.parse_args()

    viewer = ConcordiaGUIViewer(args.output_file, args.run_id)
    viewer.run()


if __name__ == "__main__":
    main()
