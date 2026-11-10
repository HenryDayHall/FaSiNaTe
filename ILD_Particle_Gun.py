"""
Particle gun to fire straight into the detector, with a small offset in the z direction.

With --segments N, the events are in N files, and a run writes the one that
--segment says (from 1), <name>_seg<i>of<N>: it makes all n_events events, as
without segments, and writes those of the segment, numbered as they are in the
whole set (ddml_bench/layout.py's segment_bounds()).
"""

import sys
import os
from array import array
import argparse

# What generating events needs (numpy, pyLCIO, h5py and particle) is imported
# where it is used, so that the file names (get_parser and get_file_name) can be
# worked out with any Python, as ddml_bench does when it plans the pipeline.
from ddml_bench.layout import INPUT_DIR, segment_bounds, segment_suffix

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
# Subfolder of paths.data_dir (see configs/default.yaml) that the gun writes to
OUTPUT_SUBDIR = INPUT_DIR
# How the directions can be spread within the theta and phi ranges (see --distribution)
DISTRIBUTIONS = ("spherical", "angular", "surface")


def str_to_bool(text):
    """Parse a boolean command line value; argparse's type=bool treats "False" as True."""
    lowered = text.strip().lower()
    if lowered in ("1", "true", "t", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, not '{text}'")


def spherical_flag(text):
    """The older --spherical_distribution flag, as a --distribution value."""
    return "spherical" if str_to_bool(text) else "angular"


def default_save_dir():
    """
    The input_single_particle_gun folder inside the configured data directory.

    Inside a container started by the run_*.sh scripts the configuration is passed
    in as FASINATE_* environment variables (as setup.sh does), so those win;
    otherwise the configuration files are read with scripts/fasinate_config.py.
    """
    data_dir = os.environ.get("FASINATE_PATHS_DATA_DIR")
    if not data_dir:
        from ddml_bench.config import load_config

        data_dir = load_config()["paths"]["data_dir"]
    return os.path.join(data_dir, OUTPUT_SUBDIR)


class ParticleProperties:
    def __init__(self, pid, mass, charge):
        self.pid = pid
        self.mass = mass  # GeV
        self.charge = charge

    @classmethod
    def from_parser(cls, parsed_args):
        import particle

        pid = parsed_args.pid
        library_particle = particle.Particle.from_pdgid(pid)
        if library_particle.mass is None or library_particle.charge is None:
            raise ValueError(f"The particle package has no mass or charge for PDG ID {pid}")
        return cls(
            pid=pid,
            mass=library_particle.mass / 1000,
            charge=library_particle.charge,
        )

    def __repr__(self):
        return f"ParticleProperties(pid={self.pid}, mass={self.mass}, charge={self.charge})"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generation with a particle gun",
    )
    simple_args = [
        ("n_events", int, 100, "Number of events to generate"),
        ("pid", int, 22, "Particle Type"),
        ("seed", int, 0, "Random seed"),
        (
            "save_dir",
            str,
            None,
            f"Directory to save the LCIO file (default: {OUTPUT_SUBDIR} in the "
            "configured paths.data_dir)",
        ),
        ("gun_from_ip", str_to_bool, True, "If true, gun emulates particle from the IP"),
        ("segment", int, 1, "Write only this segment of the events, from 1"),
        ("segments", int, 1, "The number of segments the events are split into, each a file"),
    ]
    # optionally, all of the simple_args can be specified postionally
    for arg, arg_type, default, help_str in simple_args:
        parser.add_argument(
            f"--{arg}",
            action="store",
            type=arg_type,
            default=default,
            help=help_str,
        )
    parser.add_argument(
        "--distribution",
        choices=DISTRIBUTIONS,
        default="spherical",
        help="How the directions are spread within the theta and phi ranges: "
        "spherical, uniform in solid angle; angular, uniform in theta and phi; "
        "surface, so that the gun positions are uniform over the calorimeter "
        "surface (needs gun_from_ip)",
    )
    parser.add_argument(
        "--spherical_distribution",
        dest="distribution",
        type=spherical_flag,
        help="Older form of --distribution: true for spherical, false for angular",
    )

    # ranged args have to be specified with flags
    # or we wouldn't know if you were specifying a range or a single value
    range_properties = [
        ("theta", 90, "angle of incident particle in degrees"),
        ("phi", 90, "angle of incident particle in degrees"),
        ("energy", 10, "of incident particle in GeV"),
    ]
    for range_property, default, description in range_properties:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            f"--{range_property}",
            action="store",
            default=default,
            type=float,
            help=f"Fixed {range_property} {description}",
        )
        group.add_argument(
            f"--{range_property}_min_max",
            action="store",
            type=float,
            nargs=2,
            help=f"Minimum and maximum {range_property} {description}",
        )

    return parser


def pick_momenta(parsed_args, mass):
    """
    Pick the momenta for the particle gun.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The result of parsing the user args.

    Returns
    -------
    ParticleProperties
        The particle properties.
    theta : array-like (n_events,) or float
        The theta angles in radians.
    phi : array-like (n_events,) or float
        The phi angles in radians.
    energy : array-like (n_events,)
        The energies.
    """
    import numpy as np

    np.random.seed(parsed_args.seed)
    n_events = parsed_args.n_events

    has_distribution = (
        parsed_args.theta_min_max is not None or parsed_args.phi_min_max is not None
    )

    # pick the momenta
    if parsed_args.distribution == "surface":
        if not parsed_args.gun_from_ip:
            raise ValueError(
                "The surface distribution needs gun_from_ip, so that the gun is where "
                "the direction from the IP crosses the calorimeter surface"
            )
        theta, phi = gen_surface_momenta(
            n_events,
            parsed_args.theta_min_max,
            parsed_args.phi_min_max,
            parsed_args.theta,
            parsed_args.phi,
        )
    elif has_distribution and parsed_args.distribution == "spherical":
        theta, phi = gen_spherical_momenta(
            n_events,
            parsed_args.theta_min_max,
            parsed_args.phi_min_max,
            parsed_args.theta,
            parsed_args.phi,
        )
    else:
        if parsed_args.theta_min_max is not None:
            theta = np.random.uniform(*parsed_args.theta_min_max, size=n_events)
        else:
            theta = parsed_args.theta
        if parsed_args.phi_min_max is not None:
            phi = np.random.uniform(*parsed_args.phi_min_max, size=n_events)
        else:
            phi = parsed_args.phi
        theta = np.radians(theta)
        phi = np.radians(phi)

    energy = parsed_args.energy
    if parsed_args.energy_min_max is not None:
        if parsed_args.energy_min_max[0] < mass:
            raise ValueError(
                "The minimum energy must be greater than the particle mass"
            )
        energy = np.random.uniform(*parsed_args.energy_min_max, size=n_events)
    else:
        if energy < mass:
            raise ValueError("The energy must be greater than the particle mass")
        energy = np.full(n_events, energy)

    return theta, phi, energy


def gen_spherical_momenta(n_points, theta_min_max, phi_min_max, theta=90.0, phi=90.0):
    """
    Generate momenta that conform to theta and phi ranges,
    but are sampled evenly on the surface of a sphere.

    Sampling cos(theta) and phi uniformly gives a uniform density on the sphere,
    so the patch between the limits is sampled exactly, with no rejection loop.

    Parameters
    ----------
    n_points : int
        Number of directions to generate.
    theta_min_max : tuple or None
        The minimum and maximum theta angles in degrees.
        If None, every point has the fixed `theta`.
    phi_min_max : tuple or None
        The minimum and maximum phi angles in degrees.
        If None, every point has the fixed `phi`.
    theta, phi : float
        Fixed angles in degrees, used when the matching range is None.

    Returns
    -------
    theta : array-like (n_events,)
        The theta angles in radians.
    phi : array-like (n_events,)
        The phi angles in radians.
    """
    import numpy as np

    if theta_min_max is None:
        thetas = np.full(n_points, np.radians(theta))
    else:
        cos_limits = np.cos(np.radians(theta_min_max))
        thetas = np.arccos(np.random.uniform(cos_limits[0], cos_limits[1], size=n_points))
    if phi_min_max is None:
        phis = np.full(n_points, np.radians(phi))
    else:
        phis = np.radians(np.random.uniform(*phi_min_max, size=n_points))
    return thetas, phis


def gen_surface_momenta(n_points, theta_min_max, phi_min_max, theta=90.0, phi=90.0):
    """
    Generate directions from the IP that cross the calorimeter surface (the
    plane y = const that pick_gun_position puts the gun on, the front face of
    the top barrel stave) at points spread evenly over the part of it within
    the theta and phi ranges.

    A direction crosses the plane y = 1 at x = cot(phi), z = rho cot(theta),
    with rho = sqrt(1 + x^2), so the area element there is
    dx dz = rho dx d(cot(theta)): cot(theta) is uniform, and x = cot(phi) has a
    density proportional to rho. The distance of the plane only scales this.

    Parameters and returns are as for gen_spherical_momenta; the angles must be
    strictly between 0 and 180 degrees.
    """
    import numpy as np

    theta_limits = np.radians(theta_min_max if theta_min_max is not None else (theta, theta))
    phi_limits = np.radians(phi_min_max if phi_min_max is not None else (phi, phi))
    for limits in (theta_limits, phi_limits):
        if not np.all((limits > 0) & (limits < np.pi)):
            raise ValueError(
                "The surface distribution needs theta and phi strictly between "
                "0 and 180 degrees, where the directions reach the top of the barrel"
            )
    cot_thetas = np.random.uniform(*np.sort(1 / np.tan(theta_limits)), size=n_points)
    x_limits = np.sort(1 / np.tan(phi_limits))
    rho_max = np.sqrt(1 + np.max(x_limits**2))
    xs = np.empty(0)
    while len(xs) < n_points:  # rejection sampling: keep x with probability rho / rho_max
        proposed = np.random.uniform(*x_limits, size=n_points)
        keep = np.random.uniform(0, rho_max, size=n_points) < np.sqrt(1 + proposed**2)
        xs = np.concatenate([xs, proposed[keep]])
    return np.arctan2(1, cot_thetas), np.arctan2(1, xs[:n_points])


def momenta_to_pxyz(
    mass,
    thetas,
    phis,
    energies,
):
    """
    Convert angles and energies to momenta in the x, y, z directions.

    Parameters
    ----------
    mass : float
        The mass of the particle in GeV.
    thetas : array-like (n_events,) or float
        The theta angles of the particles in radians.
        If a float, all particles have the same theta.
    phis : array-like (n_events,) or float
        The phi angles of the particles in radians.
        If a float, all particles have the same phi.
    energies : array-like (n_events,)
        The energies of the particles units of GeV.
    """
    import numpy as np

    p = np.sqrt(energies**2 - mass**2)
    pxys = np.empty((len(p), 3))
    pxys[:, 0] = p * np.sin(thetas) * np.cos(phis)
    pxys[:, 1] = p * np.sin(thetas) * np.sin(phis)
    pxys[:, 2] = p * np.cos(thetas)
    return pxys


def get_file_name(parsed_args):
    ranges = ["theta", "phi", "energy"]
    summaries = []
    for range_name in ranges:
        min_max_name = f"{range_name}_min_max"
        if getattr(parsed_args, min_max_name) is not None:
            min_max = getattr(parsed_args, min_max_name)
            summary = f"{range_name[:2]}{min_max[0]:g}-{min_max[1]:g}"
        else:
            summary = f"{range_name[:2]}{getattr(parsed_args, range_name):g}"
        summaries.append(summary)
    base_name = (
        f"p{parsed_args.pid}_" + "_".join(summaries) + f"_seed{parsed_args.seed}"
    )
    if parsed_args.gun_from_ip:
        base_name += "_ip"
    if parsed_args.distribution != "spherical":
        base_name += f"_{parsed_args.distribution}"
    base_name += segment_suffix(parsed_args.segment, parsed_args.segments)
    save_dir = parsed_args.save_dir or default_save_dir()
    file_path = os.path.join(save_dir, base_name + ".slcio")
    return file_path


def pick_gun_position(gun_from_ip, pxyz, calorimeter_surface_y=1804.7):
    """
    The gun may be a fixed point, or it may be at the IP
    """
    import numpy as np

    if gun_from_ip:
        if np.any(pxyz[:, 1] <= 0):
            raise ValueError(
                "With gun_from_ip, every particle needs a positive y momentum to reach "
                f"the calorimeter surface at y={calorimeter_surface_y} mm; "
                "keep phi strictly between 0 and 180 degrees and theta away from 0 and 180"
            )
        distance_to_surface = calorimeter_surface_y / pxyz[:, 1]
        gun_pos = distance_to_surface[:, None] * pxyz
    else:
        gun_pos = np.array([(0.0, calorimeter_surface_y, -50.0)] * len(pxyz))

    return gun_pos


def pxyz_to_lcio(
    outfile,
    particle_properties,
    energy,
    pxyzs,
    gun_position,
    first_event=0,
):
    """
    For one particle PID, mass and charge, and a set
    of momenta, write n_events to an LCIO file, numbered from first_event.

    Parameters
    ----------
    outfile : str
        The name of the output file.
    particle_properties : ParticleProperties
        A ParticleProperties object which has attributes pid, mass, and charge.
    pxyzs : array-like (n_events, 3)
        The momenta of the particles in the x, y, z directions.
        Units are GeV.
    gun_position : array-like (n_events, 3)
        The position of the gun in the x, y, z directions.
    """
    import h5py
    from pyLCIO import EVENT, IOIMPL, IMPL

    writer = IOIMPL.LCFactory.getInstance().createLCWriter()
    writer.open(outfile, EVENT.LCIO.WRITE_NEW)
    print(" opened outfile: ", outfile)

    # ========== particle properties ===================
    genstat = 1

    # decayLen = 1.e32
    # =================================================

    # write a RunHeader
    run = IMPL.LCRunHeaderImpl()
    run.parameters().setValue("Generator", os.path.basename(__file__))
    run.parameters().setValue("PDG", particle_properties.pid)
    run.parameters().setValue("Charge", particle_properties.charge)
    run.parameters().setValue("Mass", particle_properties.mass)
    writer.writeRunHeader(run)
    # ================================================

    for event_n, momentum in enumerate(pxyzs):
        col = IMPL.LCCollectionVec(EVENT.LCIO.MCPARTICLE)
        evt = IMPL.LCEventImpl()

        evt.setEventNumber(first_event + event_n)

        evt.addCollection(col, "MCParticle")

        momentum = array("f", momentum)

        # particle starts and ends at the gun position
        vertex = array("d", gun_position[event_n])
        endpoint = array("d", gun_position[event_n])

        # --------------- create MCParticle -------------------

        mcp = IMPL.MCParticleImpl()

        mcp.setGeneratorStatus(genstat)
        mcp.setMass(particle_properties.mass)
        mcp.setPDG(particle_properties.pid)
        mcp.setMomentum(momentum)
        mcp.setCharge(particle_properties.charge)
        mcp.setVertex(vertex)
        mcp.setEndpoint(endpoint)

        # -------------------------------------------------------

        col.addElement(mcp)

        writer.writeEvent(evt)

    writer.close()

    out_h5_file = outfile.replace(".slcio", ".h5")
    with h5py.File(out_h5_file, "w") as file:
        file.create_dataset("energy", data=energy)
        file.create_dataset("input_p_global", data=pxyzs)
        file.create_dataset("input_gun_position", data=gun_position)


def run(parse_args):
    if not 1 <= parse_args.segments <= parse_args.n_events:
        raise ValueError("segments should be at least 1, and at most n_events")
    if not 1 <= parse_args.segment <= parse_args.segments:
        raise ValueError("segment should be from 1 to segments")
    particle_properties = ParticleProperties.from_parser(parse_args)
    print(particle_properties)

    theta, phi, energy = pick_momenta(parse_args, particle_properties.mass)
    pxyzs = momenta_to_pxyz(
        particle_properties.mass,
        theta,
        phi,
        energy,
    )
    gun_position = pick_gun_position(parse_args.gun_from_ip, pxyzs)

    # Of all the events, those of the segment
    start, stop = segment_bounds(parse_args.n_events, parse_args.segments)[parse_args.segment - 1]
    outfile = get_file_name(parse_args)
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    print(f"Writing to {outfile}")
    pxyz_to_lcio(
        outfile,
        particle_properties,
        energy[start:stop],
        pxyzs[start:stop],
        gun_position[start:stop],
        first_event=start,
    )


if __name__ == "__main__":
    parser = get_parser()
    print(parser)
    print(sys.argv)
    parse_args = parser.parse_args()
    print(parse_args)
    run(parse_args)
