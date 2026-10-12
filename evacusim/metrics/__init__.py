"""Run outputs and measurement.

- :mod:`~evacusim.metrics.manifest`: ``manifest.json``, what produced a run.
- :class:`ResultsWriter` and :class:`RunRecord`: every output file.
- :class:`PopulationMonitor`: zone occupancy over time.
- :class:`LLMCostReporter`: LLM token use and estimated cost (GBP).
- :mod:`~evacusim.metrics.regression_harness`: compare two runs on quality KPIs.
"""
