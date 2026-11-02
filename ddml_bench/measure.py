"""Measure the benchmark observables on reconstructed events.

The observables stage (ddml_bench/stages/observables.py) gathers, for each
simulator (each model in models/, and geant4) whose reconstructions of the
scenarios the observables use are finished, the numbers every plot of the
observables in the configuration needs, into the .npz file the layout names:

  <paths.data_dir>/observables/<simulator><sample>.npz

The observables known (the keys of `observables` in the configuration):

  radial_profile        energy of the reconstructed hits against the distance
                        from the incident particle's line of flight (through its
                        vertex, along its momentum) to the hit's cell centre,
                        one profile per max_radius; stored as the energy in
                        each ring around the line of flight, which the figures
                        divide by the ring's area (ddml_bench/plot.py)
  longitudinal_profile  energy of the reconstructed hits per calorimeter layer,
                        from the layer field of each hit's cellID
  energy                the reconstructed energy of each event, the sum of the
                        energies of its reconstructed particles (PandoraPFA's,
                        in particle_collections; only those of the PDG codes in
                        pdg, if given), with the incident particle's energy;
                        the figures take the energy linearity and resolution
                        from the central 90% of the events at each incident
                        energy (ddml_bench/plot.py)
  intrinsic_angle       the intrinsic angles of each event's shower, theta and
                        phi of its axis, less those of the incident particle's
                        direction: the residuals, reconstructed - true. The
                        axis is the first principal component of an
                        energy-weighted principal component analysis (PCA) of
                        the reconstructed hits (principal_axes()) in the
                        bounding box, if there is one: a box about the
                        incident particle's line of flight, centred on it in
                        every layer (in_bounding_box()). There is a PCA for
                        each fraction of those hits in hit_fractions, the
                        most energetic first: all of them, and the top 4%,
                        which leaves out the peripheral hits that tilt the
                        axis (most_energetic()). The figures histogram the
                        residuals (ddml_bench/plot.py)
  photon_separation     of an event of two incident photons (a di-photon
                        scenario): their separation on the front face of the
                        ECAL barrel, the distance between the points where their
                        lines of flight cross it (separations_on_surface()), the
                        mean of their energies, and the number of the event's
                        reconstructed photons (its reconstructed particles in
                        particle_collections of the PDG codes in pdg, [22]; all
                        of them without pdg). The figures take the mean number
                        of reconstructed photons in bins of the separation

The incident particle is the first MC particle with generator status 1 (the
particle gun's), from the MC truth collection: MCParticles, as ddsim names it in
EDM4hep and the reconstruction passes it on, or MCParticle, its LCIO name (the
separation takes both of an event that has exactly two, the photons of a
di-photon event). Each
profile is stored per input file of the scenario, as the sum over events of each
bin's energy and of its square, with the number of events; the mean energy per
event in a bin, and its statistical uncertainty, follow from those. The energy
and the intrinsic angles are stored per event (the central 90% needs every
event, and the residuals can be binned anew), in the order of the files, each
file's events in turn. Keys, with "__" between the parts:

  <observable>__<scenario>__<profile>__edges        bin edges
  <observable>__<scenario>__<profile>__sum          [file, bin] sum of energies, GeV
  <observable>__<scenario>__<profile>__sum_sq       [file, bin] sum of squares, GeV^2
  <observable>__<scenario>__<profile>__settings     the settings, as JSON
  energy__<scenario>__reconstructed                 [event] reconstructed energy, GeV
  energy__<scenario>__true                          [event] incident energy, GeV
  energy__<scenario>__settings                      the settings, as JSON
  intrinsic_angle__<scenario>__dtheta_<hits>        [event] theta and phi of the axis
  intrinsic_angle__<scenario>__dphi_<hits>          from the PCA of those hits, less the
                                                    incident particle's, degrees (phi's
                                                    from -180 to 180); NaN for a shower
                                                    without an axis (fewer than 2 hits,
                                                    in the bounding box if there is one)
  intrinsic_angle__<scenario>__theta_true           [event] theta and phi of the incident
  intrinsic_angle__<scenario>__phi_true             particle's direction, degrees
  intrinsic_angle__<scenario>__settings             the settings, as JSON
  photon_separation__<scenario>__separation         [event] separation of the two incident
                                                    photons on the ECAL face, mm; NaN for
                                                    an event without exactly two, or with
                                                    one not heading into the face
  photon_separation__<scenario>__energy             [event] mean energy of the two, GeV
  photon_separation__<scenario>__n_photons          [event] reconstructed photons
  photon_separation__<scenario>__settings           the settings, as JSON
  <scenario>__files                                 the reconstructed files, relative
                                                    to paths.data_dir
  <scenario>__seeds                                 [file] the ddsim seed each was
                                                    simulated with, -1 if ddsim chose it
  <scenario>__n_events                              [file] events used
  <scenario>__n_no_primary                          [file] events left out
  <scenario>__true_energy                           [file] mean incident energy, GeV
  meta__simulator, meta__label, meta__color, meta__linestyle

where <hits> is all, for all the hits, or top<percent> for a fraction of them,
e.g. top4 for the most energetic 4%. Theta is the polar angle, from the z axis
(the beam), and phi the azimuth, atan2(y, x), in the global frame.

A .npz file is up to date when it has every key the configuration asks for,
made with the same settings from the same reconstructed files, none of them
reconstructed again since. The file's own record of what it was made from
decides this, so the reconstructed files needn't be there any more, or ever
have been in this data folder (the .npz files of the references can be
fetched rather than made): only those that are there are checked for being
newer. (Files measured before the paths were recorded relative to
paths.data_dir, with full paths, still count, from any data folder.) Settings
that only change the figures (see DRAWING) don't count, nor do settings added
since a measurement was made, at the value that measures as before (see
ADDED_SETTINGS: e.g. bounding_box null, every hit).
"""

import json
import os
import sys

from .layout import SEP

# Events read from a file at a time
CHUNK = 2000


class ObservableError(Exception):
    pass


def say(message):
    print("observables: " + message, file=sys.stderr)


def profile_key(observable, scenario, profile, what):
    return SEP.join((observable, scenario, profile, what))


def event_key(observable, scenario, what):
    """The key of an observable stored per event."""
    return SEP.join((observable, scenario, what))


# ---------------------------------------------------------------------------
# The observables: their profiles (binning) and what each hit contributes


def _numbers(settings, key, observable, kind=float):
    value = settings.get(key)
    values = value if isinstance(value, list) else [value]
    try:
        return [kind(v) for v in values]
    except (TypeError, ValueError):
        raise ObservableError("observables.%s.%s should be a number or a list of numbers"
                              % (observable, key))


# Each returns [(profile name, (first edge, last edge, number of bins))]; the
# binning is plain numbers, so the plan can be checked without numpy.


def radial_profiles(observable, settings):
    radii = _numbers(settings, "max_radius", observable)
    n_bins = _numbers(settings, "n_bins", observable, int)
    if len(n_bins) == 1:
        n_bins = n_bins * len(radii)
    if len(n_bins) != len(radii) or min(radii) <= 0 or min(n_bins) < 1:
        raise ObservableError("observables.%s: n_bins needs a positive number of bins for "
                              "each max_radius" % observable)
    return [("r%g" % radius, (0.0, radius, bins)) for radius, bins in zip(radii, n_bins)]


def longitudinal_profiles(observable, settings):
    max_layer = _numbers(settings, "max_layer", observable, int)[0]
    if max_layer < 0:
        raise ObservableError("observables.%s.max_layer should be at least 0" % observable)
    # One bin per layer number, centred on it
    return [("layers", (-0.5, max_layer + 0.5, max_layer + 1))]


def bin_edges(binning):
    import numpy as np
    first, last, n_bins = binning
    return np.linspace(first, last, n_bins + 1)


# observable: (its profiles, what it needs from each hit)
KINDS = {
    "radial_profile": (radial_profiles, "radius"),
    "longitudinal_profile": (longitudinal_profiles, "layer"),
}


# ---------------------------------------------------------------------------
# The observables of each event as a whole, stored per event rather than in
# bins (PER_EVENT, below): their settings, what is stored of each event, and
# how it is measured on a chunk of events. Each measuring function takes the
# observable, its settings, the file's Branches, the chunk (start, stop), the
# incident particles (read_primary()) and the hits of the collections the
# observables use (read_hits()), and gives {what: [event of the chunk]}.


def check_particles(observable, settings):
    """The settings of an observable of the reconstructed particles."""
    collections = settings.get("particle_collections")
    if not isinstance(collections, list) or not collections \
            or not all(isinstance(c, str) and c for c in collections):
        raise ObservableError("observables.%s needs particle_collections, a list of the "
                              "collections of reconstructed particles, e.g. [PandoraPFOs]"
                              % observable)
    codes = settings.get("pdg")
    if codes is not None and (not isinstance(codes, list) or not all(
            isinstance(c, int) and not isinstance(c, bool) for c in codes)):
        raise ObservableError("observables.%s.pdg should be a list of PDG codes, e.g. [22], "
                              "or null for every reconstructed particle" % observable)


def energy_items(observable, settings):
    return ("reconstructed", "true")


def measure_energy(observable, settings, branches, start, stop, primary, hits):
    """The reconstructed energy of each event, and the incident particle's."""
    return {"reconstructed": read_reconstructed_energy(branches, settings, observable, start,
                                                       stop),
            "true": primary[2]}


# The intrinsic angles: theta and phi of the axis of each shower, from the PCA
# of its hits, against those of the incident particle's direction

ANGLES = ("theta", "phi")
# The residuals' histograms in the figures when the settings don't say: from
# -5 to 5 degrees, in 100 bins
MAX_RESIDUAL = 5.0
RESIDUAL_BINS = 100
# Hits closer together than this (mm) have no axis between them; the centres
# of any two cells are much further apart
SAME_POINT = 1e-6


def hit_selections(observable, settings):
    """[(name, fraction)]: the PCAs of the intrinsic angle, one for each
    fraction of the hits in hit_fractions, named all for all of them and
    top<percent> for the rest, e.g. top4 for the most energetic 4%."""
    fractions = _numbers(settings, "hit_fractions", observable)
    if not fractions or not all(0 < f <= 1 for f in fractions):
        raise ObservableError("observables.%s.hit_fractions should be fractions of the hits, "
                              "above 0 and at most 1, e.g. [1.0, 0.04]" % observable)
    names = ["all" if f == 1 else "top%g" % (100 * f) for f in fractions]
    if len(set(names)) != len(names):
        raise ObservableError("observables.%s.hit_fractions has a fraction twice: %s"
                              % (observable, ", ".join("%g" % f for f in fractions)))
    return list(zip(names, fractions))


def residual_binning(observable, settings):
    """{PCA name: (first edge, last edge, number of bins)}: how the figures
    histogram the residuals of each PCA of hit_selections(), from
    -max_residual to max_residual degrees in n_bins bins, each a number for
    every PCA or a list with one for each (MAX_RESIDUAL and RESIDUAL_BINS
    if not given)."""
    selections = hit_selections(observable, settings)
    given = dict(settings)
    for key, default in (("max_residual", MAX_RESIDUAL), ("n_bins", RESIDUAL_BINS)):
        if given.get(key) is None:
            given[key] = default
    limits = _numbers(given, "max_residual", observable)
    n_bins = _numbers(given, "n_bins", observable, int)
    limits = limits * len(selections) if len(limits) == 1 else limits
    n_bins = n_bins * len(selections) if len(n_bins) == 1 else n_bins
    if len(limits) != len(selections) or len(n_bins) != len(selections) \
            or min(limits) <= 0 or min(n_bins) < 1:
        raise ObservableError("observables.%s: max_residual and n_bins need a positive number "
                              "each, or one for each of hit_fractions" % observable)
    return dict((name, (-limit, limit, bins))
                for (name, _), limit, bins in zip(selections, limits, n_bins))


def bounding_box(observable, settings):
    """The side (mm) of the bounding box, the box about each shower that its
    hits must be in to go into the PCAs (in_bounding_box()): bounding_box in
    the settings, or None for every hit (null, or not given)."""
    side = settings.get("bounding_box")
    if side is None:
        return None
    if isinstance(side, bool) or not isinstance(side, (int, float)) or not side > 0:
        raise ObservableError("observables.%s.bounding_box should be the side, in mm, of the box "
                              "about each shower that its hits must be in (e.g. 800), or null "
                              "for all of them" % observable)
    return float(side)


def check_angles(observable, settings):
    """The settings of the intrinsic angle."""
    collections = settings.get("hit_collections")
    if not isinstance(collections, list) or not collections \
            or not all(isinstance(c, str) and c for c in collections):
        raise ObservableError("observables.%s needs hit_collections, a list of the collections "
                              "of reconstructed calorimeter hits, e.g. [EcalBarrelCollectionRec]"
                              % observable)
    bounding_box(observable, settings)
    residual_binning(observable, settings)  # and hit_fractions with it


def angle_items(observable, settings):
    """dtheta_<PCA> and dphi_<PCA> for each PCA, then theta_true and phi_true."""
    items = []
    for name, _ in hit_selections(observable, settings):
        items += ["d%s_%s" % (angle, name) for angle in ANGLES]
    return tuple(items) + tuple(angle + "_true" for angle in ANGLES)


def most_energetic(event, energy, n_events, fraction):
    """Which hits are among the most energetic fraction of their event's hits:
    as many as the fraction of them, rounded up, but at least two, the fewest
    an axis needs (or all, for a fraction of 1). Equal energies go in the
    order of the hits."""
    import numpy as np
    if fraction >= 1:
        return np.ones(len(event), dtype=bool)
    counts = np.bincount(event, minlength=n_events)
    # The hits by event, the most energetic first, and the place of each in its event
    order = np.lexsort((-energy, event))
    rank = np.empty(len(event), dtype=np.int64)
    rank[order] = np.arange(len(event)) - (np.cumsum(counts) - counts)[event[order]]
    # (less a little, so that a product that should be whole isn't rounded up past it)
    keep = np.maximum(np.ceil(fraction * counts - 1e-9), 2)
    return rank < keep[event]


def in_bounding_box(event, position, vertex, direction, side):
    """Which of the hits (the index of each one's event, and its position
    [hit, xyz], mm) are in their event's bounding box, about the line of
    flight of its incident particle (from its vertex, along its direction:
    [event, xyz]). The layers of the ECAL barrel's top stave are planes of
    constant y, and in each the box is a square of that side, its edges along
    x and z, centred on where the line of flight crosses the layer: each layer
    shifted along the shower's axis, so that the axis is at its centre (as
    CaloClouds3's preprocessing shifts them). In depth it reaches half the side
    in front of the vertex (on the ECAL's face) and half the side behind it.
    A hit on an edge is in it; of a particle not heading across the layers (a
    direction without y), no hit is."""
    import numpy as np
    half = side / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        # How far the line of flight goes in x and in z for each mm in y
        slope = direction[:, [0, 2]] / direction[:, [1]]
        depth = position[:, 1] - vertex[event, 1]
        inside = np.abs(depth) <= half
        # x, then z, to keep the memory a chunk's hits need low
        for i, c in enumerate((0, 2)):
            across = position[:, c] - vertex[event, c] - depth * slope[event, i]
            inside &= np.abs(across) <= half
    return inside


def principal_axes(event, energy, position, n_events):
    """The axis of the hits of each event: the first principal component of an
    energy-weighted PCA, i.e. the direction in which the hits spread the most
    about their centre, the spread and the centre both weighted by the hits'
    energies (the eigenvector of the largest eigenvalue of the weighted
    covariance of their positions). Unit vectors [event, 3] of either sign;
    NaN for an event without hits with energy at two points apart."""
    import numpy as np
    keep = energy > 0
    event, energy, position = event[keep], energy[keep], position[keep]
    weight = np.bincount(event, weights=energy, minlength=n_events)
    covariance = np.empty((n_events, 3, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = np.stack([np.bincount(event, weights=energy * position[:, c],
                                       minlength=n_events) for c in range(3)], axis=1)
        centre /= weight[:, None]
        offset = position - centre[event]
        for a in range(3):
            for b in range(a, 3):
                covariance[:, a, b] = covariance[:, b, a] = np.bincount(
                    event, weights=energy * offset[:, a] * offset[:, b],
                    minlength=n_events) / weight
    has = (weight > 0) & np.isfinite(covariance).all(axis=(1, 2))
    covariance[~has] = np.eye(3)  # anything that eigh takes; these have no axis
    spread, vectors = np.linalg.eigh(covariance)  # eigenvalues from the smallest
    axes = vectors[:, :, -1].copy()
    axes[~has | ~(spread[:, -1] > SAME_POINT ** 2)] = np.nan
    return axes


def polar_angles(vectors):
    """theta, from the z axis, and phi, atan2(y, x), of each of the vectors
    [n, 3], in degrees."""
    import numpy as np
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = vectors[:, 2] / np.linalg.norm(vectors, axis=1)
    return (np.degrees(np.arccos(np.clip(cos_theta, -1, 1))),
            np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])))


def azimuth_residual(phi, phi_true):
    """phi less phi_true, in degrees, from -180 to 180: the shorter way round,
    so that 179 less -179 is -2, not 358."""
    return (phi - phi_true + 180) % 360 - 180


def angle_residuals(axes, direction):
    """theta and phi of each axis less those of the direction, in degrees, phi's
    from -180 to 180; the axis taken along the direction (a principal axis has
    no sign of its own)."""
    import numpy as np
    axes = np.where((np.einsum("ij,ij->i", axes, direction) < 0)[:, None], -axes, axes)
    theta, phi = polar_angles(axes)
    theta_true, phi_true = polar_angles(direction)
    return theta - theta_true, azimuth_residual(phi, phi_true)


def measure_angles(observable, settings, branches, start, stop, primary, hits):
    """The residuals of the intrinsic angles of each event, from each PCA of
    hit_selections() of the hits of hit_collections in the bounding box (if
    the settings give one, bounding_box()), and the incident particle's theta
    and phi."""
    import numpy as np
    vertex, direction, _, has = primary
    n_events = stop - start
    parts = [hit for hit in hits if hit[0] in settings["hit_collections"]]
    event = np.concatenate([p[1] for p in parts] + [np.zeros(0, dtype=np.int64)])
    energy = np.concatenate([p[2] for p in parts] + [np.zeros(0)])
    position = np.concatenate([p[3] for p in parts] + [np.zeros((0, 3))])
    # Hits without energy have no weight in a PCA, so they aren't counted
    with_energy = energy > 0
    event, energy, position = event[with_energy], energy[with_energy], position[with_energy]
    side = bounding_box(observable, settings)
    if side is not None:
        inside = in_bounding_box(event, position, vertex, direction, side)
        # A shower with none of its hits in its box (as when its particle
        # starts far from where it enters the calorimeter) has no axis
        with_hits = has & (np.bincount(event, minlength=n_events) > 0)
        emptied = with_hits & (np.bincount(event[inside], minlength=n_events) == 0)
        if emptied.any():
            say("    warning: %d of the %d showers with hits have none in the bounding box (%g mm "
                "about the incident particle's line of flight), so they have no axis; does the "
                "particle start on the ECAL's front face, heading into the top stave, as the "
                "particle gun's do?" % (emptied.sum(), with_hits.sum(), side))
        event, energy, position = event[inside], energy[inside], position[inside]
    values = dict(zip(("theta_true", "phi_true"), polar_angles(direction)))
    for name, fraction in hit_selections(observable, settings):
        chosen = most_energetic(event, energy, n_events, fraction)
        axes = principal_axes(event[chosen], energy[chosen], position[chosen], n_events)
        values["dtheta_" + name], values["dphi_" + name] = angle_residuals(axes, direction)
    return values


# The separation of the two photons of a di-photon event on the front face of
# the ECAL barrel, against the number of reconstructed photons

# The front face of the ECAL barrel when the settings don't say: the distance
# of the planes of its staves from the beam axis (mm), where the guns start
# their photons in ILD_l5_o1_v02, and the number of staves, one of them the
# top one, centred at phi = 90 degrees (ILD's octagon)
SURFACE_DISTANCE = 1804.7
BARREL_SYMMETRY = 8
# The separation's bins in the figures when the settings don't say: from 0 to
# 90 mm in 18 bins of 5 mm, as in the earlier di-photon analysis
MAX_SEPARATION = 90.0
SEPARATION_BINS = 18


def surface(observable, settings):
    """(distance, staves) of the ECAL barrel's front face that the separation is
    measured on: surface_distance and barrel_symmetry, or their defaults."""
    distance = settings.get("surface_distance")
    staves = settings.get("barrel_symmetry")
    distance = SURFACE_DISTANCE if distance is None else distance
    staves = BARREL_SYMMETRY if staves is None else staves
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not distance > 0:
        raise ObservableError("observables.%s.surface_distance should be the distance of the "
                              "front face of the ECAL barrel's staves from the beam axis, in mm "
                              "(%g in ILD_l5_o1_v02)" % (observable, SURFACE_DISTANCE))
    if isinstance(staves, bool) or not isinstance(staves, int) or staves < 3:
        raise ObservableError("observables.%s.barrel_symmetry should be the number of staves of "
                              "the ECAL barrel (%d in ILD), at least 3" % (observable,
                                                                           BARREL_SYMMETRY))
    return float(distance), staves


def separation_binning(observable, settings):
    """(0, max_separation, n_bins): how the figures bin the separation (mm),
    MAX_SEPARATION and SEPARATION_BINS if not given."""
    given = dict(settings)
    for key, default in (("max_separation", MAX_SEPARATION), ("n_bins", SEPARATION_BINS)):
        if given.get(key) is None:
            given[key] = default
    limits = _numbers(given, "max_separation", observable)
    n_bins = _numbers(given, "n_bins", observable, int)
    if len(limits) != 1 or len(n_bins) != 1 or not limits[0] > 0 or n_bins[0] < 1:
        raise ObservableError("observables.%s: max_separation and n_bins should be a positive "
                              "number each" % observable)
    return (0.0, limits[0], n_bins[0])


def check_separation(observable, settings):
    """The settings of the photon separation."""
    check_particles(observable, settings)
    surface(observable, settings)
    separation_binning(observable, settings)


def separation_items(observable, settings):
    return ("separation", "energy", "n_photons")


def separations_on_surface(vertex, momentum, distance, staves):
    """The separation of each pair of photons ([event, photon, xyz], two
    photons each) on the front face of the ECAL barrel: the distance between
    the points where their lines of flight (from their vertices, along their
    momenta) cross the plane of the face of the stave they head into. The
    barrel has that many staves, one centred at phi = 90 degrees, their planes
    at that distance from the beam axis; the pair's stave is the one nearest in
    phi to the sum of the two photons' directions. A photon that starts on the
    face crosses it where it starts. NaN for a pair with a photon heading away
    from the face (or without two photons: NaN vertices or momenta)."""
    import numpy as np
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = momentum / np.linalg.norm(momentum, axis=2, keepdims=True)
        total = direction.sum(axis=1)
        step = 2 * np.pi / staves
        phi = np.arctan2(total[:, 1], total[:, 0])
        stave = np.pi / 2 + step * np.round((phi - np.pi / 2) / step)
        normal = np.stack([np.cos(stave), np.sin(stave), np.zeros_like(stave)], axis=1)
        along = np.einsum("epc,ec->ep", direction, normal)
        reach = (distance - np.einsum("epc,ec->ep", vertex, normal)) / along
        crossing = vertex + reach[:, :, None] * direction
        separation = np.linalg.norm(crossing[:, 0] - crossing[:, 1], axis=1)
    heads_in = np.all(along > 0, axis=1)
    return np.where(heads_in & np.isfinite(separation), separation, np.nan)


def read_incident_pairs(branches, start, stop):
    """Vertices and momenta [event, photon, xyz] and masses [event, photon] of
    the MC particles of generator status 1 of each event that has two of them,
    the incident photons of a di-photon event; NaN for the other events."""
    import awkward as ak
    import numpy as np
    mc = branches.collection_with(MC_COLLECTIONS, "generatorStatus")
    mask = branches.jagged(mc + ".generatorStatus", start, stop) == 1
    two = ak.to_numpy(ak.sum(mask, axis=1)) == 2

    def pair(name):
        values = ak.pad_none(branches.jagged(name, start, stop)[mask], 2, axis=1, clip=True)
        values = ak.to_numpy(ak.fill_none(values, np.nan)).astype(np.float64).reshape(-1, 2)
        values[~two] = np.nan
        return values

    vertex = np.stack([pair(mc + ".vertex." + c) for c in "xyz"], axis=2)
    momentum = np.stack([pair(mc + ".momentum." + c) for c in "xyz"], axis=2)
    return vertex, momentum, pair(mc + ".mass")


def measure_separation(observable, settings, branches, start, stop, primary, hits):
    """The separation of each event's two incident photons on the ECAL face,
    the mean of their energies, and the number of its reconstructed photons."""
    import numpy as np
    vertex, momentum, mass = read_incident_pairs(branches, start, stop)
    distance, staves = surface(observable, settings)
    energy = np.sqrt(np.sum(momentum ** 2, axis=2) + np.nan_to_num(mass) ** 2).mean(axis=1)
    return {"separation": separations_on_surface(vertex, momentum, distance, staves),
            "energy": energy,
            "n_photons": count_reconstructed(branches, settings, observable, start, stop)}


# The neutral pions of the tau decays: the pi0 candidates of the
# reconstruction, each matched to the MC truth through its two photons

# What is stored of each event (the number of its pi0 candidates, and of its MC
# pi0s matched to), and of each candidate, in the order of the events, each
# event's candidates in turn: the invariant mass of its two photons, its
# (fitted) energy, the sum of its photons' energies, theta and phi of its
# momentum, its match (PI0_MATCHES), and the energy, theta and phi of the MC
# pi0 it is matched to, with that pi0's place among its event's (NaN, and -1,
# for a fake one)
PI0_EVENT_ITEMS = ("n_candidates", "n_true")
PI0_CANDIDATE_ITEMS = ("mass", "energy", "photon_energy", "theta", "phi", "match",
                       "true_energy", "true_theta", "true_phi", "true_index")
# A candidate's match: neither photon correct, only one, or both, of one MC pi0
FAKE, CONFUSED, GOOD = 0, 1, 2
PI0_MATCHES = {FAKE: "fake", CONFUSED: "confused", GOOD: "good"}
# The bins of the figures when the settings don't say: the mass from 90 to
# 180 MeV in bins of 5 MeV, around GammaGammaCandidateFinder's window of 40 MeV
# either side of the pi0's mass; the energy less the MC pi0's from -6 to 6 GeV
# in bins of 250 MeV; and theta and phi of the direction less the MC pi0's
# from -0.3 to 0.3 degrees in bins of 0.015 degrees (an ECAL cell, 5 mm wide
# and 1.8 m from the beam, is 0.16 degrees across; the directions, from the
# centres of the photons' clusters, are finer than that)
PI0_BINNING = {"mass": (0.09, 0.18, 18), "energy": (-6.0, 6.0, 48),
               "theta": (-0.3, 0.3, 40), "phi": (-0.3, 0.3, 40)}
# The units of each figure's bins
PI0_UNITS = {"mass": "GeV", "energy": "GeV", "theta": "degrees", "phi": "degrees"}
# A candidate's reconstructed energy in the figures: its own, from the fit, or
# the sum of its two photons'
RECONSTRUCTED_ENERGIES = {"fitted": "energy", "photons": "photon_energy"}
# The links of the reconstructed particles to the MC particles: the members
# with the indices of each end, from podio's links (EDM4hep 0.99 on) or the
# associations before them
LINK_ENDS = (("_from", "_to"), ("_rec", "_sim"))
PI0 = 111
PHOTON = 22


def _collection_name(observable, settings, key, optional=False):
    value = settings.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ObservableError("observables.%s.%s should be the name of a collection%s"
                              % (observable, key, ", or null" if optional else ""))
    return value


def pi0_binning(observable, settings):
    """{"mass": (first, last, bins), "energy": (...), "theta": (...), "phi":
    (...)}: how the figures bin the mass of the good pi0s, their energy less
    the MC pi0s', and theta and phi of their direction less the MC pi0s', from
    the first to the last of <figure>_range (mass_range, energy_range: GeV;
    theta_range, phi_range: degrees) in <figure>_bins bins (PI0_BINNING where
    not given)."""
    binning = {}
    for figure, default in PI0_BINNING.items():
        given = settings.get(figure + "_range")
        bins = settings.get(figure + "_bins")
        if given is None:
            first, last = default[:2]
        elif isinstance(given, list) and len(given) == 2:
            first, last = given
        else:
            first = last = None
        bins = default[2] if bins is None else bins
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in (first, last)) or not first < last:
            raise ObservableError("observables.%s.%s_range should be [first, last], the lowest "
                                  "and highest edges of the bins in %s"
                                  % (observable, figure, PI0_UNITS[figure]))
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise ObservableError("observables.%s.%s_bins should be a number of bins, at least 1"
                                  % (observable, figure))
        binning[figure] = (float(first), float(last), bins)
    return binning


def check_pi0(observable, settings):
    """The settings of the pi0s."""
    for key in ("candidate_collection", "particle_collection", "link_collection"):
        _collection_name(observable, settings, key)
    _collection_name(observable, settings, "reverse_link_collection", optional=True)
    codes = settings.get("ancestor_pdg")
    if codes is not None and (not isinstance(codes, list) or not codes or not all(
            isinstance(c, int) and not isinstance(c, bool) for c in codes)):
        raise ObservableError("observables.%s.ancestor_pdg should be a list of PDG codes, e.g. "
                              "[15] for the taus, or null for every MC pi0" % observable)
    energy = settings.get("reconstructed_energy")
    if energy is not None and energy not in RECONSTRUCTED_ENERGIES:
        raise ObservableError("observables.%s.reconstructed_energy should be %s, not %r"
                              % (observable, " or ".join(RECONSTRUCTED_ENERGIES), energy))
    pi0_binning(observable, settings)


def pi0_items(observable, settings):
    return PI0_EVENT_ITEMS + PI0_CANDIDATE_ITEMS


def _flat(branches, name, start, stop):
    """A member of a collection, flattened over the events, with the offset of
    each event's first object (one more than the events)."""
    import awkward as ak
    import numpy as np
    values = branches.jagged(name, start, stop)
    counts = ak.to_numpy(ak.num(values)).astype(np.int64)
    return ak.to_numpy(ak.flatten(values)), np.concatenate([[0], np.cumsum(counts)])


def _relation(branches, collection, relation, start, stop, observable):
    """A relation of the objects of a collection (podio's), e.g. the parents
    of the MC particles: each object's begin and end in its event's list of
    related objects (flattened over the events), the indices in that list of
    the objects related to (within their event, flattened), with the offset of
    each event's list, and their collection IDs."""
    import numpy as np
    begin = "%s.%s_begin" % (collection, relation)
    index = "_%s_%s.index" % (collection, relation)
    for name in (begin, index):
        if name not in branches.keys:
            raise ObservableError("%s has no %s (the %s of the %s), which observables.%s needs"
                                  % (branches.path, name, relation, collection, observable))
    starts, _ = _flat(branches, begin, start, stop)
    ends, _ = _flat(branches, "%s.%s_end" % (collection, relation), start, stop)
    indices, offsets = _flat(branches, index, start, stop)
    ids = (_flat(branches, "_%s_%s.collectionID" % (collection, relation), start, stop)[0]
           if "_%s_%s.collectionID" % (collection, relation) in branches.keys else None)
    return starts.astype(np.int64), ends.astype(np.int64), indices.astype(np.int64), offsets, ids


def _links(branches, collection, start, stop, observable, setting):
    """The links of a link collection: weight, the indices of the
    reconstructed and of the MC particle of each (within their events), the
    offset of each event's links, and the collection IDs of either end."""
    import numpy as np
    if collection + ".weight" not in branches.keys:
        raise ObservableError("%s has no %s.weight (does it have the links %s? set "
                              "observables.%s.%s)" % (branches.path, collection, collection,
                                                      observable, setting))
    weight, offsets = _flat(branches, collection + ".weight", start, stop)
    for reco, mc in LINK_ENDS:
        names = ["_%s%s.index" % (collection, end) for end in (reco, mc)]
        if all(name in branches.keys for name in names):
            ends = [_flat(branches, name, start, stop)[0].astype(np.int64) for name in names]
            ids = [_flat(branches, name[:-len("index")] + "collectionID", start, stop)[0]
                   if name[:-len("index")] + "collectionID" in branches.keys else None
                   for name in names]
            return weight.astype(np.float64), ends[0], ends[1], offsets, ids
    raise ObservableError("the links %s in %s have no ends (_%s_from and _to, or _rec and _sim)"
                          % (collection, branches.path, collection))


def _points_to(branches, ids, target, what, observable, setting):
    """Check that the collection IDs a relation or link records are all of the
    target collection, when the file's metadata says which that is (podio's
    table of collections; ids None or the table empty: nothing to check)."""
    import numpy as np
    table = branches.collection_ids()
    if ids is None or not table or target not in table or not len(ids):
        return
    others = set(int(i) for i in np.unique(ids)) - {table[target]}
    if others:
        names = dict((value, name) for name, value in table.items())
        raise ObservableError("%s in %s points to %s, not to %s; set observables.%s.%s"
                              % (what, branches.path, ", ".join(sorted(
                                  names.get(i, "collection %d" % i) for i in others)), target,
                                 observable, setting))


def best_links(event_links, by_reco):
    """{the index of a particle at one end: the index at the other end of its
    link of the highest weight} of an event's links [(weight, reco index, MC
    index)]: for each reconstructed particle (by_reco) or each MC particle. Of
    links of the same weight, the first."""
    best = {}
    for weight, reco, mc in event_links:
        key, other = (reco, mc) if by_reco else (mc, reco)
        if key not in best or weight > best[key][0]:
            best[key] = (weight, other)
    return dict((key, other) for key, (_, other) in best.items())


def match_pi0(photons, energies, best_mc, best_reco, slots):
    """(match, the place of the MC pi0) of a candidate of two photons (indices
    of reconstructed particles, with their energies): a photon is correct when
    its best MC particle is a photon of a selected MC pi0 (slots: {MC photon:
    the place of its pi0}) and, with the reverse links (best_reco not None),
    that MC photon's best reconstructed particle is it. GOOD when both are
    correct and the two photons of one pi0, CONFUSED when only one is (or
    they are of different pi0s: the place is then that of the more energetic
    photon's), FAKE (-1) otherwise."""
    correct = []
    for photon, energy in zip(photons, energies):
        mc = best_mc.get(photon)
        if mc is None or mc not in slots:
            continue
        if best_reco is not None and best_reco.get(mc) != photon:
            continue
        correct.append((energy, slots[mc], mc))
    if len(correct) == 2 and correct[0][1] == correct[1][1] and correct[0][2] != correct[1][2]:
        return GOOD, correct[0][1]
    if correct:
        return CONFUSED, max(correct, key=lambda item: item[0])[1]
    return FAKE, -1


def measure_pi0(observable, settings, branches, start, stop, primary, hits):
    """Each event's pi0 candidates (candidate_collection), with the invariant
    mass and energies of their two photons (particle_collection), matched to
    the event's MC pi0s through the links (match_pi0()): the MC pi0s from the
    generator that decay into two photons, with an ancestor of the PDG codes
    of ancestor_pdg (the taus) if given."""
    import numpy as np
    n = stop - start
    mc = branches.collection_with(MC_COLLECTIONS, "PDG")
    candidates = settings["candidate_collection"]
    particles = settings["particle_collection"]
    for collection, setting in ((candidates, "candidate_collection"),
                                (particles, "particle_collection")):
        if collection + ".energy" not in branches.keys:
            raise ObservableError("%s has no %s.energy (does it have the collection %s? set "
                                  "observables.%s.%s)" % (branches.path, collection, collection,
                                                          observable, setting))
    ancestors = set(abs(code) for code in settings.get("ancestor_pdg") or [])

    def momenta(collection):
        return np.stack([_flat(branches, "%s.momentum.%s" % (collection, c), start, stop)[0]
                         for c in "xyz"], axis=1).astype(np.float64)

    # The MC particles, with their daughters (and parents, for their ancestors)
    pdg, mc_offsets = _flat(branches, mc + ".PDG", start, stop)
    status = _flat(branches, mc + ".generatorStatus", start, stop)[0]
    mc_momentum = momenta(mc)
    mc_mass = _flat(branches, mc + ".mass", start, stop)[0].astype(np.float64)
    daughters = _relation(branches, mc, "daughters", start, stop, observable)
    parents = _relation(branches, mc, "parents", start, stop, observable) if ancestors else None

    # The candidates, and the reconstructed particles they are made of
    energy, candidate_offsets = _flat(branches, candidates + ".energy", start, stop)
    momentum = momenta(candidates)
    made_of = _relation(branches, candidates, "particles", start, stop, observable)
    _points_to(branches, made_of[4], particles, "the particles of %s" % candidates, observable,
               "particle_collection")
    particle_energy, particle_offsets = _flat(branches, particles + ".energy", start, stop)
    particle_momentum = momenta(particles)

    # The links between them, each way
    links = {}
    for key in ("link_collection", "reverse_link_collection"):
        if settings.get(key):
            weight, reco, truth, offsets, ids = _links(branches, settings[key], start, stop,
                                                       observable, key)
            for end, target in zip(ids, (particles, mc)):
                _points_to(branches, end, target, "the links %s" % settings[key], observable, key)
            links[key] = (weight, reco, truth, offsets)

    def related(relation, event, particle):
        """The indices (within the event) of the objects one is related to."""
        starts, ends, indices, offsets, _ = relation
        return indices[offsets[event] + starts[particle]:offsets[event] + ends[particle]]

    def descends(event, first, local):
        """Whether an MC particle has an ancestor of the PDG codes asked for."""
        seen, todo = set(), [local]
        while todo:
            for parent in related(parents, event, first + todo.pop()):
                parent = int(parent)
                if parent not in seen:
                    seen.add(parent)
                    if abs(int(pdg[first + parent])) in ancestors:
                        return True
                    todo.append(parent)
        return False

    def event_links(key, event):
        weight, reco, truth, offsets = links[key]
        span = slice(offsets[event], offsets[event + 1])
        return zip(weight[span].tolist(), reco[span].tolist(), truth[span].tolist())

    values = dict((what, []) for what in PI0_CANDIDATE_ITEMS if what not in ("theta", "phi"))
    true_directions = []
    counts = np.diff(candidate_offsets)
    n_true = np.zeros(n, dtype=np.int64)
    for event in range(n):
        # The MC pi0s of the event: their energies and momenta, and the place
        # of the pi0 of each of their photons
        first = int(mc_offsets[event])
        slots, truth = {}, []
        for local in np.flatnonzero(np.abs(pdg[first:mc_offsets[event + 1]]) == PI0):
            particle = first + int(local)
            if status[particle] == 0:  # made by the simulation, not the generator
                continue
            photons = [int(d) for d in related(daughters, event, particle)]
            if len(photons) != 2 or any(int(pdg[first + d]) != PHOTON for d in photons):
                continue
            if ancestors and not descends(event, first, int(local)):
                continue
            for photon in photons:
                slots[photon] = len(truth)
            p = mc_momentum[particle]
            truth.append((float(np.sqrt(p @ p + mc_mass[particle] ** 2)), p))
        n_true[event] = len(truth)
        best_mc = best_links(event_links("link_collection", event), by_reco=True)
        best_reco = (best_links(event_links("reverse_link_collection", event), by_reco=False)
                     if "reverse_link_collection" in links else None)

        # Its candidates
        base = int(particle_offsets[event])
        for candidate in range(int(candidate_offsets[event]), int(candidate_offsets[event + 1])):
            photons = [int(p) for p in related(made_of, event, candidate)]
            mass = photon_energy = float("nan")
            match, slot = FAKE, -1
            if len(photons) == 2:
                e1, e2 = (float(particle_energy[base + p]) for p in photons)
                p1, p2 = (particle_momentum[base + p] for p in photons)
                norms = float(np.linalg.norm(p1) * np.linalg.norm(p2))
                if norms > 0:
                    cos = float(p1 @ p2) / norms
                    mass = float(np.sqrt(max(2 * e1 * e2 * (1 - cos), 0.0)))
                photon_energy = e1 + e2
                match, slot = match_pi0(photons, (e1, e2), best_mc, best_reco, slots)
            values["mass"].append(mass)
            values["energy"].append(float(energy[candidate]))
            values["photon_energy"].append(photon_energy)
            values["match"].append(match)
            values["true_index"].append(slot)
            values["true_energy"].append(truth[slot][0] if slot >= 0 else float("nan"))
            true_directions.append(truth[slot][1] if slot >= 0 else np.full(3, np.nan))

    result = {"n_candidates": counts.astype(np.int64), "n_true": n_true}
    for what, stored in values.items():
        result[what] = np.array(stored, dtype=np.int64 if what in ("match", "true_index")
                                else np.float64)
    result["theta"], result["phi"] = polar_angles(momentum.reshape(-1, 3))
    result["true_theta"], result["true_phi"] = polar_angles(
        np.array(true_directions, dtype=np.float64).reshape(-1, 3))
    return result


# The observables of each event as a whole, stored per event rather than in
# bins, as the figures need every event: observable: (what checks its
# settings, what is stored of each event (given the settings), and what
# measures that)
PER_EVENT = {
    "energy": (check_particles, energy_items, measure_energy),
    "intrinsic_angle": (check_angles, angle_items, measure_angles),
    "photon_separation": (check_separation, separation_items, measure_separation),
    "pi0": (check_pi0, pi0_items, measure_pi0),
}
# Of those, the observables that store some of their items per object of an
# event rather than per event (e.g. per pi0 candidate): observable: (the item
# per event giving each event's number of them, the items stored per object)
PER_CANDIDATE = {"pi0": ("n_candidates", PI0_CANDIDATE_ITEMS)}


def event_items(name, settings):
    """What an observable stored per event stores of each event."""
    return tuple(PER_EVENT[name][1](name, settings))


# The settings an observable gained after it was first measured, each with the
# value that measures it as before: with that value, the setting is left out
# of the settings a measurement records, so measurements made before it was
# there (here, or published) stay up to date
ADDED_SETTINGS = {"intrinsic_angle": {"bounding_box": None}}


# The settings of each observable that only change its figures, not what is
# measured, so changing them only needs plotting again (./plot_observables.sh)
DRAWING = {"intrinsic_angle": ("max_residual", "n_bins"),
           "photon_separation": ("max_separation", "n_bins"),
           "pi0": ("reconstructed_energy", "mass_range", "mass_bins", "energy_range",
                   "energy_bins", "theta_range", "theta_bins", "phi_range", "phi_bins")}


def observable_settings(name, settings):
    """The settings that decide an observable's numbers (not how it is
    compared, or drawn), without those added since that measure as before
    (ADDED_SETTINGS)."""
    drawn = ("metric",) + DRAWING.get(name, ())
    before = ADDED_SETTINGS.get(name, {})
    return {k: v for k, v in sorted(settings.items())
            if k not in drawn and not (k in before and v == before[k])}


def check_observables(config):
    """The observables in the configuration, checked, and the scenarios they are
    measured on, in order."""
    observables = config.get("observables") or {}
    if not observables:
        raise ObservableError("the configuration has no observables")
    scenarios = []
    for name, settings in observables.items():
        if name not in KINDS and name not in PER_EVENT:
            raise ObservableError("there's no observable '%s'; the known ones are: %s"
                                  % (name, ", ".join(list(KINDS) + list(PER_EVENT))))
        if not isinstance(settings, dict) or not settings.get("scenarios"):
            raise ObservableError("observables.%s needs a list of scenarios" % name)
        if name in PER_EVENT:
            PER_EVENT[name][0](name, settings)
        else:
            KINDS[name][0](name, settings)  # check the binning early
            if not settings.get("hit_collections"):
                raise ObservableError("observables.%s needs hit_collections" % name)
        for scenario in settings["scenarios"]:
            if scenario not in scenarios:
                scenarios.append(scenario)
    return observables, scenarios


def restrict(observables, scenarios):
    """The observables measured on these scenarios alone, and the scenarios
    they are then measured on, in order: each observable with only those of
    its scenarios, and without those left with none (the observables of a
    scenario that is disabled, or whose inputs aren't there to run it)."""
    kept, measured = {}, []
    for name, settings in observables.items():
        on = [scenario for scenario in settings["scenarios"] if scenario in scenarios]
        if not on:
            continue
        kept[name] = dict(settings, scenarios=on)
        measured += [scenario for scenario in on if scenario not in measured]
    return kept, measured


# ---------------------------------------------------------------------------
# Reading the reconstructed files


def parse_encoding(description):
    """{field: (offset, width, signed)} from a DD4hep cellID description such as
    "system:5,module:3,...,cellX:32:-16,cellY:-16"."""
    fields = {}
    offset = 0
    for part in description.split(","):
        bits = [b.strip() for b in part.split(":")]
        try:
            if len(bits) == 2:
                name, width = bits[0], int(bits[1])
            elif len(bits) == 3:
                name, offset, width = bits[0], int(bits[1]), int(bits[2])
            else:
                raise ValueError
        except ValueError:
            raise ObservableError("can't read the cellID encoding %r" % description)
        fields[name] = (offset, abs(width), width < 0)
        offset += abs(width)
    return fields


def decode_field(cell_ids, field):
    import numpy as np
    offset, width, signed = field
    values = ((cell_ids.astype(np.uint64) >> np.uint64(offset))
              & np.uint64((1 << width) - 1)).astype(np.int64)
    if signed:
        values = np.where(values >= (1 << (width - 1)), values - (1 << width), values)
    return values


def encoding_from_file(path, collection):
    """The cellID encoding podio stored for a collection, or None."""
    try:
        from podio.root_io import Reader
    except ImportError:
        return None
    key = collection + "__CellIDEncoding"
    try:
        reader = Reader(path)
        categories = list(getattr(reader, "categories", ["metadata", "events"]))
    except Exception:
        return None
    for category in ["metadata"] + [c for c in categories if c != "metadata"]:
        try:
            frames = reader.get(category)
            for frame in frames:
                try:
                    value = frame.get_parameter(key)
                except Exception:
                    value = None
                if value:
                    return str(value)
                break  # parameters are in the first frame of a category, if anywhere
        except Exception:
            continue
    return None


def encoding_from_geometry(compact, collection):
    """The cellID encoding of the readout of a collection, from the detector."""
    if not compact:
        return None
    if not os.path.isabs(compact):
        k4geo = os.environ.get("k4geo_DIR") or os.environ.get("K4GEO")
        if not k4geo:
            return None
        compact = os.path.join(k4geo, compact)
    try:
        import dd4hep
        detector = dd4hep.Detector.getInstance()
        if not getattr(encoding_from_geometry, "loaded", None) == compact:
            detector.fromXML(compact)
            encoding_from_geometry.loaded = compact
        names = [collection[:-len("Rec")]] if collection.endswith("Rec") else []
        for name in names + [collection]:
            readout = detector.readout(name)
            if readout.isValid():
                return str(readout.idSpec().fieldDescription())
    except Exception as error:  # DD4hep missing, or the geometry can't load
        say("could not read the cellID encoding from %s: %s" % (compact, error))
    return None


# The MC truth collection, by the names it can have: ddsim names it MCParticles
# when it writes EDM4hep (and the reconstruction keeps the input's collections),
# and MCParticle is its LCIO name, which a conversion from LCIO keeps
MC_COLLECTIONS = ("MCParticles", "MCParticle")


class Branches:
    """Columns of a podio events tree, found by name (e.g. "MCParticles.vertex.x")."""

    def __init__(self, path):
        import uproot
        self.path = path
        self.file = uproot.open(path)
        if "events" not in self.file:
            raise ObservableError("%s has no 'events' tree; is it an EDM4hep ROOT file?" % path)
        self.tree = self.file["events"]
        self.keys = {}
        for key in self.tree.keys(recursive=True):
            self.keys.setdefault(key.split("/")[-1], key)
        self.entries = self.tree.num_entries
        self._ids = None

    def collection_ids(self):
        """{collection: its collection ID} from podio's table of the events'
        collections in the file's metadata, or {} if it has none that can be
        read (files not written by podio, e.g. the tests')."""
        if self._ids is None:
            self._ids = {}
            try:
                tree = self.file["podio_metadata"]
                keys = dict((key.split("/")[-1], key) for key in tree.keys(recursive=True))
                ids = keys.get("events___CollectionTypeInfo.collectionID")
                names = keys.get("events___CollectionTypeInfo.name")
                if ids and names:
                    found = tree.arrays([ids, names], library="np", entry_stop=1)
                    self._ids = dict((str(name), int(value)) for name, value
                                     in zip(found[names][0], found[ids][0]))
            except Exception:  # no metadata, or another podio's
                self._ids = {}
        return self._ids

    def jagged(self, name, start, stop):
        if name not in self.keys:
            collection = name.split(".")[0]
            raise ObservableError("%s has no %s (does it have the collection %s?)"
                                  % (self.path, name, collection))
        return self.tree[self.keys[name]].array(entry_start=start, entry_stop=stop,
                                                library="ak")

    def collection_with(self, candidates, member):
        """The first of the candidate collections that has the member."""
        for name in candidates:
            if "%s.%s" % (name, member) in self.keys:
                return name
        others = sorted({key.split(".")[0] for key in self.keys
                         if key.endswith("." + member)})
        raise ObservableError("%s has no MC truth collection (%s with a %s); %s" % (
            self.path, " or ".join(candidates), member,
            "collections with a %s: %s" % (member, ", ".join(others)) if others
            else "no collection in it has a %s" % member))

    def close(self):
        self.file.close()


def read_primary(branches, start, stop):
    """Vertex, unit direction and energy of each event's incident particle (the
    first MC particle of generator status 1), and which events have one."""
    import awkward as ak
    import numpy as np
    mc = branches.collection_with(MC_COLLECTIONS, "generatorStatus")
    status = branches.jagged(mc + ".generatorStatus", start, stop)
    mask = status == 1

    def first(name):
        values = ak.firsts(branches.jagged(name, start, stop)[mask])
        return ak.to_numpy(ak.fill_none(values, np.nan)).astype(np.float64)

    vertex = np.stack([first(mc + ".vertex." + c) for c in "xyz"], axis=1)
    momentum = np.stack([first(mc + ".momentum." + c) for c in "xyz"], axis=1)
    mass = first(mc + ".mass")
    p = np.linalg.norm(momentum, axis=1)
    has = np.isfinite(p) & (p > 0)
    direction = np.zeros_like(momentum)
    direction[has] = momentum[has] / p[has, None]
    return vertex, direction, np.sqrt(p ** 2 + np.nan_to_num(mass) ** 2), has


def read_hits(branches, collections, start, stop, need_cell_id):
    """The hits of the collections, flattened: event index, energy, position
    and cellID (or None)."""
    import awkward as ak
    import numpy as np
    parts = []
    for collection in collections:
        energy = branches.jagged(collection + ".energy", start, stop)
        counts = ak.to_numpy(ak.num(energy))
        event = np.repeat(np.arange(len(counts)), counts)
        position = np.stack([ak.to_numpy(ak.flatten(
            branches.jagged("%s.position.%s" % (collection, c), start, stop)))
            for c in "xyz"], axis=1).astype(np.float64)
        cell_id = (ak.to_numpy(ak.flatten(branches.jagged(collection + ".cellID", start, stop)))
                   if need_cell_id else None)
        parts.append((collection, event, ak.to_numpy(ak.flatten(energy)).astype(np.float64),
                      position, cell_id))
    return parts


# The member of a reconstructed particle holding its PDG code: PDG since
# EDM4hep 0.99, type before
PDG_MEMBERS = ("PDG", "type")


def selected_particles(branches, settings, observable, start, stop):
    """The energies of each event's reconstructed particles, jagged, for each of
    the particle collections: all of them, or those whose PDG code (either
    sign) is in pdg, if given."""
    import awkward as ak
    import numpy as np
    codes = np.abs(np.array(settings.get("pdg") or [], dtype=np.int64))
    selected = []
    for collection in settings["particle_collections"]:
        name = collection + ".energy"
        if name not in branches.keys:
            raise ObservableError("%s has no %s (does it have the collection %s? set "
                                  "observables.%s.particle_collections)"
                                  % (branches.path, name, collection, observable))
        energy = ak.values_astype(branches.jagged(name, start, stop), np.float64)
        if codes.size:
            member = next((m for m in PDG_MEMBERS if "%s.%s" % (collection, m) in branches.keys),
                          None)
            if member is None:
                raise ObservableError("the reconstructed particles of %s in %s have no PDG "
                                      "code (%s), which observables.%s.pdg needs"
                                      % (collection, branches.path, " or ".join(PDG_MEMBERS),
                                         observable))
            pdg = branches.jagged("%s.%s" % (collection, member), start, stop)
            keep = np.isin(np.abs(ak.to_numpy(ak.flatten(pdg)).astype(np.int64)), codes)
            energy = energy[ak.unflatten(keep, ak.num(pdg))]
        selected.append(energy)
    return selected


def read_reconstructed_energy(branches, settings, observable, start, stop):
    """The reconstructed energy of each event: the sum of the energies of its
    reconstructed particles in the particle collections, or of those whose PDG
    code (either sign) is in pdg, if given. 0 for an event with none."""
    import awkward as ak
    import numpy as np
    total = np.zeros(stop - start)
    for energy in selected_particles(branches, settings, observable, start, stop):
        total += ak.to_numpy(ak.sum(energy, axis=1))
    return total


def count_reconstructed(branches, settings, observable, start, stop):
    """The number of each event's reconstructed particles in the particle
    collections, or of those whose PDG code (either sign) is in pdg, if given."""
    import awkward as ak
    import numpy as np
    count = np.zeros(stop - start, dtype=np.int64)
    for energy in selected_particles(branches, settings, observable, start, stop):
        count += ak.to_numpy(ak.num(energy, axis=1)).astype(np.int64)
    return count


def accumulate(edges, values, energy, event, n_events, keep_events):
    """Per bin, the sum over the kept events of the bin's energy, and of its square."""
    import numpy as np
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, values, side="right") - 1
    valid = (index >= 0) & (index < n_bins) & np.isfinite(values)
    per_event = np.bincount(event[valid] * n_bins + index[valid], weights=energy[valid],
                            minlength=n_events * n_bins).reshape(n_events, n_bins)
    per_event = per_event[keep_events]
    return per_event.sum(axis=0), (per_event ** 2).sum(axis=0)


# ---------------------------------------------------------------------------
# Computing


def wanted(observables):
    """[(observable, scenario, profile, edges, settings JSON)] that the observables
    in the configuration ask for; profile and edges are None for those stored
    per event."""
    result = []
    for name, settings in observables.items():
        settings_json = json.dumps(observable_settings(name, settings), sort_keys=True)
        if name in PER_EVENT:
            for scenario in settings["scenarios"]:
                result.append((name, scenario, None, None, settings_json))
            continue
        for profile, binning in KINDS[name][0](name, settings):
            for scenario in settings["scenarios"]:
                result.append((name, scenario, profile, bin_edges(binning), settings_json))
    return result


def same_files(stored, paths, relative=None):
    """Whether the reconstructed files a .npz file records (stored) are paths:
    recorded as relative(path), relative to the data folder; or, as files
    measured before the records were relative are, by their full paths, in
    this data folder or in another one (the same files below it)."""
    recorded = [relative(p) for p in paths] if relative else list(paths)
    if stored == recorded or stored == list(paths):
        return True
    return len(stored) == len(recorded) and all(
        old == new or (os.path.isabs(old) and not os.path.isabs(new)
                       and old.endswith(os.sep + new))
        for old, new in zip(stored, recorded))


def is_up_to_date(path, profiles, files, relative=None):
    """The .npz file has every key asked for, with the same binning and settings,
    from the same reconstructed files ({scenario: [path]}), none of them
    reconstructed again since. relative(path) is how the files are recorded.

    What the .npz file records decides it: the reconstructed files needn't be
    there (e.g. deleted since, or the .npz file fetched with the inputs rather
    than measured here); only those that are must not be newer than it."""
    import numpy as np
    if not os.path.isfile(path):
        return False, "no %s yet" % os.path.basename(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            for name, scenario, profile, edges, settings_json in profiles:
                if profile is None:
                    for what in event_items(name, json.loads(settings_json)) + ("settings",):
                        if event_key(name, scenario, what) not in keys:
                            return False, "%s has no %s" % (
                                os.path.basename(path), event_key(name, scenario, what))
                    if str(data[event_key(name, scenario, "settings")]) != settings_json:
                        return False, "the settings of %s changed" % name
                    continue
                for what in ("edges", "sum", "sum_sq", "settings"):
                    if profile_key(name, scenario, profile, what) not in keys:
                        return False, "%s has no %s" % (
                            os.path.basename(path), profile_key(name, scenario, profile, what))
                stored = data[profile_key(name, scenario, profile, "edges")]
                if stored.shape != edges.shape or not np.allclose(stored, edges):
                    return False, "the binning of %s/%s changed" % (name, profile)
                if str(data[profile_key(name, scenario, profile, "settings")]) != settings_json:
                    return False, "the settings of %s changed" % name
            for scenario, paths in files.items():
                for what in ("files", "seeds", "n_events", "n_no_primary", "true_energy"):
                    if SEP.join((scenario, what)) not in keys:
                        return False, "%s has no %s" % (os.path.basename(path),
                                                       SEP.join((scenario, what)))
                stored = [str(f) for f in data[SEP.join((scenario, "files"))]]
                if not same_files(stored, paths, relative):
                    return False, "the reconstructed files of %s changed" % scenario
                there = [p for p in paths if os.path.isfile(p)]
                if there and max(os.path.getmtime(p) for p in there) > os.path.getmtime(path):
                    return False, "files of %s were reconstructed again since" % scenario
    except (OSError, ValueError) as error:
        return False, "can't read %s: %s" % (path, error)
    return True, ""


def cell_id_fields(settings, observable, path, collection, compact, cache):
    key = (collection, compact)
    if key not in cache:
        description = (settings.get("cell_id_encoding") or encoding_from_file(path, collection)
                       or encoding_from_geometry(compact, collection))
        if not description:
            raise ObservableError(
                "can't tell how to decode the cellIDs of %s: it isn't in %s, and the detector "
                "geometry couldn't give it; set observables.%s.cell_id_encoding"
                % (collection, path, observable))
        cache[key] = parse_encoding(description)
    fields = cache[key]
    field = settings.get("layer_field") or "layer"
    if field not in fields:
        raise ObservableError("the cellID encoding of %s has no field '%s' (it has: %s); "
                              "set observables.%s.layer_field"
                              % (collection, field, ", ".join(fields), observable))
    return fields[field]


def measure_file(path, observables, profiles, simulator, cache):
    """Sums for every profile of one reconstructed file, what is stored of each
    event for the observables stored per event ({observable: {what: [event]}}),
    and its event counts."""
    import numpy as np
    profiled = dict((name, settings) for name, settings in observables.items()
                    if name in KINDS)
    by_event = dict((name, settings) for name, settings in observables.items()
                    if name in PER_EVENT)
    # The hit collections to read, with what the profiles need from each (those
    # of the observables stored per event, e.g. the intrinsic angle, too)
    by_collection = {}
    for name, settings in profiled.items():
        for collection in settings["hit_collections"]:
            by_collection.setdefault(collection, set()).add(KINDS[name][1])
    for name, settings in by_event.items():
        for collection in settings.get("hit_collections") or ():
            by_collection.setdefault(collection, set())
    sums = {key: [np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)]
            for key, edges in profiles.items()}
    events = dict((name, dict((what, []) for what in event_items(name, settings)))
                  for name, settings in by_event.items())
    n_events = n_no_primary = 0
    energy_total = 0.0
    branches = Branches(path)
    try:
        for start in range(0, branches.entries, CHUNK):
            stop = min(start + CHUNK, branches.entries)
            n = stop - start
            primary = read_primary(branches, start, stop)
            vertex, direction, true_energy, has = primary
            n_events += int(has.sum())
            n_no_primary += int(n - has.sum())
            energy_total += float(true_energy[has].sum())
            hits = read_hits(branches, list(by_collection), start, stop,
                             any("layer" in v for v in by_collection.values()))
            for name, settings in by_event.items():
                measured = PER_EVENT[name][2](name, settings, branches, start, stop, primary,
                                              hits)
                # What is stored per object of an event (e.g. per pi0
                # candidate) is kept of the events kept, as the rest is
                count, per_object = PER_CANDIDATE.get(name, (None, ()))
                kept = np.repeat(has, measured[count]) if count else None
                for what, stored in events[name].items():
                    stored.append(measured[what][kept if what in per_object else has])
            for collection, event, energy, position, cell_id in hits:
                values = {}
                if "radius" in by_collection[collection]:
                    offset = position - vertex[event]
                    along = np.einsum("ij,ij->i", offset, direction[event])
                    values["radius"] = np.linalg.norm(
                        offset - along[:, None] * direction[event], axis=1)
                for name, settings in profiled.items():
                    if collection not in settings["hit_collections"]:
                        continue
                    need = KINDS[name][1]
                    if need == "layer" and "layer" not in values:
                        field = cell_id_fields(settings, name, path, collection,
                                               simulator.get("compact_file"), cache)
                        values["layer"] = decode_field(cell_id, field).astype(np.float64)
                        # A wrong cellID encoding gives layers outside the
                        # binning, which would otherwise drop out silently
                        max_layer = _numbers(settings, "max_layer", name, int)[0]
                        outside = (values["layer"] < 0) | (values["layer"] > max_layer)
                        if outside.mean() > 0.01 and (path, collection) not in cache:
                            cache[(path, collection)] = True
                            say("    warning: %.0f%% of the hits of %s have a layer outside "
                                "0 to %d; check observables.%s.cell_id_encoding and "
                                "max_layer" % (100 * outside.mean(), collection, max_layer,
                                               name))
                    for (obs, profile), edges in profiles.items():
                        if obs != name:
                            continue
                        total, total_sq = accumulate(edges, values[need], energy, event, n, has)
                        sums[(obs, profile)][0] += total
                        sums[(obs, profile)][1] += total_sq
    finally:
        branches.close()
    mean_energy = energy_total / n_events if n_events else float("nan")
    events = dict((name, dict((what, np.concatenate(parts) if parts else np.zeros(0))
                              for what, parts in stored.items()))
                  for name, stored in events.items())
    return sums, events, n_events, n_no_primary, mean_energy


def compute_simulator(observables, simulator, files, profiles, out_path, relative=None,
                      seeds=None):
    """Measure one simulator: every profile, and observable stored per event
    (from wanted()), on its reconstructed files ({scenario: [path]}), into
    out_path. The files are recorded as relative(path), with the ddsim seed
    each was simulated with ({scenario: [seed]}, None where ddsim chose it)."""
    import numpy as np
    arrays = {
        "meta__simulator": np.array(simulator["name"]),
        "meta__label": np.array(simulator["label"]),
        "meta__color": np.array(simulator["color"]),
        "meta__linestyle": np.array(json.dumps(simulator["linestyle"])),
    }
    cache = {}
    for scenario, paths in files.items():
        measured = {name: settings for name, settings in observables.items()
                    if scenario in settings["scenarios"]}
        scenario_profiles = {(name, profile): edges
                             for name, scen, profile, edges, _ in profiles
                             if scen == scenario and profile is not None}
        per_file = {key: ([], []) for key in scenario_profiles}
        per_event = dict((name, dict((what, []) for what in event_items(name, settings)))
                         for name, settings in measured.items() if name in PER_EVENT)
        counts, no_primary, energies = [], [], []
        for i, path in enumerate(paths, 1):
            say("  %s [%d/%d] %s" % (scenario, i, len(paths), os.path.basename(path)))
            sums, events, n, missing, energy = measure_file(path, measured, scenario_profiles,
                                                            simulator, cache)
            if missing:
                say("    %d event(s) without an incident particle (generator status 1) "
                    "were left out" % missing)
            for key in scenario_profiles:
                per_file[key][0].append(sums[key][0])
                per_file[key][1].append(sums[key][1])
            for name, stored in per_event.items():
                for what in stored:
                    stored[what].append(events[name][what])
            counts.append(n)
            no_primary.append(missing)
            energies.append(energy)
        arrays[SEP.join((scenario, "files"))] = np.array(
            [relative(p) for p in paths] if relative else paths)
        file_seeds = list((seeds or {}).get(scenario) or [None] * len(paths))
        if len(file_seeds) != len(paths):
            raise ObservableError("%d seeds for the %d reconstructed files of %s"
                                  % (len(file_seeds), len(paths), scenario))
        arrays[SEP.join((scenario, "seeds"))] = np.array(
            [-1 if seed is None else seed for seed in file_seeds], dtype=np.int64)
        arrays[SEP.join((scenario, "n_events"))] = np.array(counts, dtype=np.int64)
        arrays[SEP.join((scenario, "n_no_primary"))] = np.array(no_primary, dtype=np.int64)
        arrays[SEP.join((scenario, "true_energy"))] = np.array(energies)
        for name, scen, profile, edges, settings_json in profiles:
            if scen != scenario:
                continue
            if profile is None:
                for what, parts in per_event[name].items():
                    arrays[event_key(name, scenario, what)] = (
                        np.concatenate(parts) if parts else np.zeros(0))
                arrays[event_key(name, scenario, "settings")] = np.array(settings_json)
                continue
            arrays[profile_key(name, scenario, profile, "edges")] = edges
            arrays[profile_key(name, scenario, profile, "sum")] = np.array(
                per_file[(name, profile)][0])
            arrays[profile_key(name, scenario, profile, "sum_sq")] = np.array(
                per_file[(name, profile)][1])
            arrays[profile_key(name, scenario, profile, "settings")] = np.array(settings_json)

    # Written to a hidden file first, so the .npz file is always complete
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = os.path.join(os.path.dirname(out_path),
                           ".%s.%d.npz" % (os.path.basename(out_path), os.getpid()))
    try:
        with open(partial, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(partial, out_path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
