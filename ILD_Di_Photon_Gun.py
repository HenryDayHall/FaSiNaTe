"""
Pairs of photons into the front face of the top stave of the ECAL barrel, for
the di-photon separation benchmark of "A First Full Physics Benchmark for
Highly Granular Calorimeter Surrogates".

Each event has two photons of the same energy, both starting on the front face
of the stave (the plane y = surface_y, 1804.7 mm in ILD_l5_o1_v02, as for
ILD_Particle_Gun.py and the models' training showers) and each flying along
the line from the IP (the origin) through its starting point, so that they
look like photons from the IP, without the tracker's material on the way.

The pairs are placed as the earlier di-photon generator placed them
(di_photon_old/Di_Photon_Input_slcio.py): the pair's midpoint uniform over a
disc of centre_radius about (centre_x, centre_z) on the face, the separation
between the two photons uniform in [0, max_separation], and the line between
them at a uniform random angle in the plane of the face. (That generator fired
both photons along +y, normal to the stave, rather than from the IP.)

Each run writes <name>.slcio, two MC particles of generator status 1 per
event, and <name>.h5 of the truth: energy [event] (of each photon, GeV),
input_p_global [event, photon, xyz] (GeV), input_gun_position [event, photon,
xyz] (mm) and separation [event] (mm, on the face).
"""

import argparse
import os
import sys
from array import array

# What generating events needs (numpy, pyLCIO and h5py) is imported where it is
# used, so that the file names (get_parser and get_file_name) can be worked out
# with any Python, as ddml_bench does when it plans the pipeline.
from ILD_Particle_Gun import default_save_dir

# The front face of the top stave of ILD_l5_o1_v02's ECAL barrel, mm
SURFACE_Y = 1804.7
PHOTON_PDG = 22


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate pairs of photons from the IP into the ECAL barrel's top stave",
    )
    options = [
        ("n_events", int, 15000, "Number of events (photon pairs)"),
        ("energy", float, 20.0, "Energy of each photon, GeV"),
        ("seed", int, 0, "Random seed"),
        ("max_separation", float, 90.0,
         "The separations of the pairs are uniform from 0 to this, mm"),
        ("centre_radius", float, 10.0,
         "Radius of the disc on the face that the pairs' midpoints are spread over, mm"),
        ("centre_x", float, 0.0, "x of the centre of that disc, mm"),
        ("centre_z", float, 70.0, "z of the centre of that disc, mm"),
        ("surface_y", float, SURFACE_Y,
         "y of the front face of the top stave, where the photons start, mm"),
        ("save_dir", str, None,
         "Directory to save the LCIO file (default: input_single_particle_gun in the "
         "configured paths.data_dir)"),
    ]
    for name, kind, default, help_text in options:
        parser.add_argument("--" + name, action="store", type=kind, default=default,
                            help=help_text)
    return parser


def check_args(parsed_args):
    """Raise ValueError for options that can't make the pairs."""
    if parsed_args.n_events < 1:
        raise ValueError("n_events should be at least 1")
    if not parsed_args.energy > 0:
        raise ValueError("energy should be above 0 GeV")
    if not parsed_args.max_separation > 0:
        raise ValueError("max_separation should be above 0 mm")
    if not parsed_args.centre_radius >= 0:
        raise ValueError("centre_radius should be at least 0 mm")
    if not parsed_args.surface_y > 0:
        raise ValueError("surface_y should be above 0 mm, so that photons from the IP reach it")


def get_file_name(parsed_args):
    base_name = "diphoton_en%g_sep%g_x%g_z%g_r%g" % (
        parsed_args.energy, parsed_args.max_separation, parsed_args.centre_x,
        parsed_args.centre_z, parsed_args.centre_radius)
    if parsed_args.surface_y != SURFACE_Y:
        base_name += "_y%g" % parsed_args.surface_y
    base_name += "_seed%d" % parsed_args.seed
    save_dir = parsed_args.save_dir or default_save_dir()
    return os.path.join(save_dir, base_name + ".slcio")


def pick_pairs(n_events, energy, max_separation, centre_radius, centre_x, centre_z, surface_y,
               rng):
    """
    The starting points and momenta of n_events pairs of photons.

    Returns
    -------
    vertex : array (n_events, 2, 3)
        Where each photon starts, on the plane y = surface_y, mm.
    momentum : array (n_events, 2, 3)
        Each photon's momentum, along the line from the IP through its vertex, GeV.
    separation : array (n_events,)
        The distance between the two vertices, mm.
    """
    import numpy as np

    # The midpoint, uniform over the disc, then the separation, then the
    # angle of the line between the two photons, as the earlier generator
    # drew them
    angle = rng.uniform(0, 2 * np.pi, n_events)
    distance = np.sqrt(rng.uniform(0, 1, n_events)) * centre_radius
    mid_x = centre_x + distance * np.cos(angle)
    mid_z = centre_z + distance * np.sin(angle)
    separation = rng.uniform(0, max_separation, n_events)
    orientation = rng.uniform(0, 2 * np.pi, n_events)
    half_x = separation / 2 * np.cos(orientation)
    half_z = separation / 2 * np.sin(orientation)

    vertex = np.empty((n_events, 2, 3))
    vertex[:, 0] = np.stack([mid_x + half_x, np.full(n_events, surface_y), mid_z + half_z], axis=1)
    vertex[:, 1] = np.stack([mid_x - half_x, np.full(n_events, surface_y), mid_z - half_z], axis=1)
    momentum = energy * vertex / np.linalg.norm(vertex, axis=2, keepdims=True)
    return vertex, momentum, separation


def pairs_to_lcio(outfile, energy, vertex, momentum, separation):
    """Write the pairs to an LCIO file, two MC particles per event, and their
    truth to an .h5 file next to it."""
    import h5py
    import numpy as np
    from pyLCIO import EVENT, IOIMPL, IMPL

    writer = IOIMPL.LCFactory.getInstance().createLCWriter()
    writer.open(outfile, EVENT.LCIO.WRITE_NEW)
    print(" opened outfile: ", outfile)

    run = IMPL.LCRunHeaderImpl()
    run.parameters().setValue("Generator", os.path.basename(__file__))
    run.parameters().setValue("PDG", PHOTON_PDG)
    run.parameters().setValue("Charge", 0)
    run.parameters().setValue("Mass", 0.0)
    writer.writeRunHeader(run)

    for event_n in range(len(vertex)):
        col = IMPL.LCCollectionVec(EVENT.LCIO.MCPARTICLE)
        evt = IMPL.LCEventImpl()
        evt.setEventNumber(event_n)
        evt.addCollection(col, "MCParticle")
        for photon in range(2):
            mcp = IMPL.MCParticleImpl()
            mcp.setGeneratorStatus(1)
            mcp.setMass(0.0)
            mcp.setPDG(PHOTON_PDG)
            mcp.setMomentum(array("f", momentum[event_n, photon]))
            mcp.setCharge(0.0)
            # each photon starts and ends where it is placed
            mcp.setVertex(array("d", vertex[event_n, photon]))
            mcp.setEndpoint(array("d", vertex[event_n, photon]))
            col.addElement(mcp)
        writer.writeEvent(evt)
    writer.close()

    out_h5_file = os.path.splitext(outfile)[0] + ".h5"
    with h5py.File(out_h5_file, "w") as file:
        file.create_dataset("energy", data=np.full(len(vertex), energy))
        file.create_dataset("input_p_global", data=momentum)
        file.create_dataset("input_gun_position", data=vertex)
        file.create_dataset("separation", data=separation)


def run(parse_args):
    import numpy as np

    check_args(parse_args)
    rng = np.random.default_rng(parse_args.seed)
    vertex, momentum, separation = pick_pairs(
        parse_args.n_events, parse_args.energy, parse_args.max_separation,
        parse_args.centre_radius, parse_args.centre_x, parse_args.centre_z,
        parse_args.surface_y, rng)
    outfile = get_file_name(parse_args)
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    print(f"Writing to {outfile}")
    pairs_to_lcio(outfile, parse_args.energy, vertex, momentum, separation)


if __name__ == "__main__":
    parser = get_parser()
    print(sys.argv)
    parse_args = parser.parse_args()
    print(parse_args)
    try:
        run(parse_args)
    except ValueError as error:
        parser.error(str(error))
