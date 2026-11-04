"""The timing benchmark: the time per event of every simulator on the inputs of
every scenario, on CPU and GPU nodes, measured in jobs of its own.

It is a workflow beside the pipeline's (ddml_bench/), with its own command
line (scripts/ddml-timing, which slurm_timing.sh, timing.sh and
timing_summary.sh run) and its own results, under <paths.data_dir>/timing. It
takes everything else from the pipeline: the configuration (configs/*.yaml,
whose timing section is its own), the scenarios and their inputs, the models
in models/, the environment from setup.sh, and the ddsim command each
simulator runs, so that a timing run simulates exactly as the pipeline does.

  settings.py            the configuration's timing section, checked
  plan.py                the tasks, one per kind of node, scenario input and
                         simulator, and what of each is done (in the
                         environment: what each model's inference can run on)
  runner.py              one task, in a job: ddsim on the input's first events,
                         each event's wall-clock and CPU time taken from the
                         moment its first line comes out, and the results
  events.py              reading ddsim's output, event by event, and the
                         statistics of the times
  inference.py           inside ddsim: the models' embedded-Python inference
                         timed, and run on the GPU
  ddsim_steer_timing.py  the steering file that sets that up, then runs the
                         simulator's own steering file
  executors.py           running the tasks here, one after another, or
                         submitting a slurm job for each, on its kind of node
  summary.py             the results gathered into one table, with each mean
                         time per event, its uncertainty and the speed-up on
                         Geant4
  cli.py                 the command line

docs/timing.md describes how it works and what it measures.
"""
