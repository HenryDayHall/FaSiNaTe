"""The configuration's timing section (configs/default.yaml), checked.

  events            events timed of each input
  warmup_events     events before them, simulated but not counted
  pin_cpu           run ddsim on one CPU of the job's
  keep_simulations  keep the timing runs' simulations beside their results
  job_name_prefix   the slurm jobs are named <prefix>-<device>-<simulator>
  devices           the kinds of node, each with
      inference             where the models' inference runs: cpu, or cuda
      simulators            all, models, or a list of simulators' names
      also_cpu_only_models  (cuda only) time the models whose inference can
                            only run on the CPU too, on this node's CPU
      partition, constraint, time, sbatch_args
                            the slurm settings of its jobs

This needs only the standard library.
"""

from ddml_bench.layout import TIMING_SUBMISSIONS, LayoutError, check_name

INFERENCE = ("cpu", "cuda")
# The simulators a kind of node times: every one, or every model
ALL, MODELS = "all", "models"
DEVICE_KEYS = ("inference", "simulators", "also_cpu_only_models", "partition", "constraint",
               "time", "sbatch_args")
# The folder of the timing folder that holds the slurm submissions, beside
# those of the kinds of node
SUBMISSIONS = TIMING_SUBMISSIONS


class SettingsError(Exception):
    pass


def _whole(value, least):
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _text_or_none(value):
    return value is None or (isinstance(value, str) and bool(value.strip()))


def device_settings(name, settings):
    """One kind of node's settings, checked, with the defaults filled in."""
    where = "timing.devices.%s" % name
    try:
        check_name(name, "the kinds of node in timing.devices")
    except LayoutError as error:
        raise SettingsError(str(error))
    if name == SUBMISSIONS:
        raise SettingsError("%s: %s is the folder of the timing's slurm submissions, so it can't "
                            "name a kind of node" % (where, name))
    if not isinstance(settings, dict):
        raise SettingsError("%s should be the settings of a kind of node: %s"
                            % (where, ", ".join(DEVICE_KEYS)))
    unknown = sorted(set(settings) - set(DEVICE_KEYS))
    if unknown:
        raise SettingsError("%s has %s; a kind of node has %s" % (where, ", ".join(unknown),
                                                                   ", ".join(DEVICE_KEYS)))
    inference = settings.get("inference")
    if inference not in INFERENCE:
        raise SettingsError("%s.inference should be where the models' inference runs: %s, not %r"
                            % (where, " or ".join(INFERENCE), inference))
    simulators = settings.get("simulators", ALL)
    if simulators not in (ALL, MODELS) and not (
            isinstance(simulators, list) and simulators
            and all(isinstance(s, str) and s for s in simulators)):
        raise SettingsError("%s.simulators should be all (geant4 and every model), models (every "
                            "model) or a list of simulators, not %r" % (where, simulators))
    also = settings.get("also_cpu_only_models", False)
    if not isinstance(also, bool):
        raise SettingsError("%s.also_cpu_only_models should be true or false" % where)
    for key in ("partition", "constraint", "time"):
        if not _text_or_none(settings.get(key)):
            raise SettingsError("%s.%s should be text (quoted, for a time) or null" % (where, key))
    args = settings.get("sbatch_args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise SettingsError("%s.sbatch_args should be a list of sbatch options" % where)
    return {"inference": inference, "simulators": simulators, "also_cpu_only_models": also,
            "partition": settings.get("partition"), "constraint": settings.get("constraint"),
            "time": settings.get("time"), "sbatch_args": list(args)}


def timing_settings(config):
    """The configuration's timing section, checked: a dict of its settings,
    devices that of each kind of node (device_settings())."""
    section = config.get("timing")
    if not isinstance(section, dict):
        raise SettingsError("the configuration has no timing section (see configs/default.yaml)")
    events = section.get("events")
    if not _whole(events, 1):
        raise SettingsError("timing.events should be the number of events timed of each input, "
                            "at least 1, not %r" % (events,))
    warmup = section.get("warmup_events")
    if not _whole(warmup, 0):
        raise SettingsError("timing.warmup_events should be a whole number, at least 0, not %r"
                            % (warmup,))
    for key in ("pin_cpu", "keep_simulations"):
        if not isinstance(section.get(key), bool):
            raise SettingsError("timing.%s should be true or false" % key)
    prefix = section.get("job_name_prefix") or "fasinate-timing"
    if not isinstance(prefix, str):
        raise SettingsError("timing.job_name_prefix should be text")
    devices = section.get("devices")
    if not isinstance(devices, dict) or not devices:
        raise SettingsError("timing.devices should be the kinds of node to time on, e.g. cpu and "
                            "gpu")
    return {"events": events, "warmup_events": warmup, "pin_cpu": section["pin_cpu"],
            "keep_simulations": section["keep_simulations"], "job_name_prefix": prefix,
            "devices": dict((name, device_settings(name, settings))
                            for name, settings in devices.items())}
