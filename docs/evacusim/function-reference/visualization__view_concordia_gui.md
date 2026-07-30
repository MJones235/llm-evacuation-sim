# Function Reference: evacusim/visualization/view_concordia_gui.py

- Source file: [evacusim/visualization/view_concordia_gui.py](../../../evacusim/visualization/view_concordia_gui.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

GUI viewer for Station Concordia simulation progress.

## Top-Level Functions

### main

- Signature: def main()
- Location: [evacusim/visualization/view_concordia_gui.py#L239](../../../evacusim/visualization/view_concordia_gui.py#L239)
- Summary: Main entry point.
- Key calls: argparse.ArgumentParser, parser.add_argument, parser.parse_args, ConcordiaGUIViewer, viewer.run

## Classes and Methods

### Class ConcordiaGUIViewer

- Location: [evacusim/visualization/view_concordia_gui.py#L17](../../../evacusim/visualization/view_concordia_gui.py#L17)
- Summary: GUI viewer for Concordia simulation.
- Method count: 9

#### __init__

- Signature: def __init__(self, output_file, run_id)
- Location: [evacusim/visualization/view_concordia_gui.py#L20](../../../evacusim/visualization/view_concordia_gui.py#L20)
- Summary: Initialize GUI viewer.
- Key calls: Path, set, tk.Tk, self.root.title, self.root.geometry, self.root.configure, self._setup_ui, self._start_monitoring

#### _setup_ui

- Signature: def _setup_ui(self)
- Location: [evacusim/visualization/view_concordia_gui.py#L45](../../../evacusim/visualization/view_concordia_gui.py#L45)
- Summary: Setup the user interface.
- Key calls: tk.Frame, header_frame.pack, tk.Label, title.pack, self.status_label.pack, scrolledtext.ScrolledText, self.text_area.pack, self.text_area.tag_config, button_frame.pack, tk.Button

#### _clear_display

- Signature: def _clear_display(self)
- Location: [evacusim/visualization/view_concordia_gui.py#L125](../../../evacusim/visualization/view_concordia_gui.py#L125)
- Summary: Clear the text display.
- Key calls: self.text_area.delete

#### _on_closing

- Signature: def _on_closing(self)
- Location: [evacusim/visualization/view_concordia_gui.py#L129](../../../evacusim/visualization/view_concordia_gui.py#L129)
- Summary: Handle window closing.
- Key calls: self.root.destroy

#### _append_text

- Signature: def _append_text(self, text, tag=None)
- Location: [evacusim/visualization/view_concordia_gui.py#L134](../../../evacusim/visualization/view_concordia_gui.py#L134)
- Summary: Append text to the display.
- Key calls: self.text_area.insert, self.auto_scroll_var.get, self.text_area.see

#### _display_decision

- Signature: def _display_decision(self, agent_id, decision)
- Location: [evacusim/visualization/view_concordia_gui.py#L140](../../../evacusim/visualization/view_concordia_gui.py#L140)
- Summary: Display a decision in the GUI.
- Key calls: decision.get, self._append_text, reasoning.get, agent_id.upper, translated.get

#### _monitor_file

- Signature: def _monitor_file(self)
- Location: [evacusim/visualization/view_concordia_gui.py#L194](../../../evacusim/visualization/view_concordia_gui.py#L194)
- Summary: Monitor the output file for changes.
- Key calls: time.sleep, self.output_file.exists, data.get, self.status_label.config, agent_decisions.items, open, json.load, agent_data.get, print, self.decisions_seen.add

#### _start_monitoring

- Signature: def _start_monitoring(self)
- Location: [evacusim/visualization/view_concordia_gui.py#L229](../../../evacusim/visualization/view_concordia_gui.py#L229)
- Summary: Start monitoring in a background thread.
- Key calls: threading.Thread, monitor_thread.start

#### run

- Signature: def run(self)
- Location: [evacusim/visualization/view_concordia_gui.py#L234](../../../evacusim/visualization/view_concordia_gui.py#L234)
- Summary: Run the GUI.
- Key calls: self.root.mainloop

## Coverage Notes

- Nested helper functions documented in this file: 0
