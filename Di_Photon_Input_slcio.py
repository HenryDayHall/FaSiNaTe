from pyLCIO import EVENT, UTIL, IOIMPL, IMPL
import math
import random
from array import array
import argparse

import csv  ### For logging info about each event
import numpy as np

### Author: P.McKeown, CERN, Aug 2024
### Adapted File to create MC Particles for passing to ddsim
### Currently desgined to create two particles (eg photons) orthogonal to the face of the ECAL Barrel, in this case of ILD.
### The separation between the two particles is varied uniformly in the range [0, maxSep], with the midpoint between the two particles along the global Z axis being set by centralPosZ
### The energy of each particle is varied uniformly (and independently) between Emax and Emin


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--pdg", action="store", type=int, default=22, help="Particle Type"
    )

    parser.add_argument("--mass", action="store", type=int, default=0, help="Mass")

    parser.add_argument("--charge", action="store", type=int, default=0, help="Charge")

    """
    
    parser.add_argument('--angleMin_theta', action='store',
                        type=int, default=30,
                        help='Minimum theta angle of incident particle input: deg, coverted to [Rad]')
    
    parser.add_argument('--angleMax_theta', action='store',
                        type=int, default=90,
                        help='Maximum theta angle of incident particle input:deg, converted to [Rad]')

    parser.add_argument('--angleMin_phi', action='store',
                        type=int, default=30,
                        help='Minimum phi angle of incident particle input: deg, coverted to [Rad]')

    parser.add_argument('--angleMax_phi', action='store',
                        type=int, default=90,
                        help='Maximum phi angle of incident particle input:deg, converted to [Rad]')
    """

    parser.add_argument(
        "--maxSep",
        action="store",
        type=int,
        default=90,
        help="Maximum separation between the two photons created [mm]",
    )

    parser.add_argument(
        "--centralPosZ",
        action="store",
        type=int,
        default=70,
        help="Central position (i.e mid-way point) between two incident photons along the global z axis [mm]",
    )

    parser.add_argument(
        "--Emax",
        action="store",
        type=int,
        default=20,
        help="Maximum energy of incident particle [GeV]",
    )

    parser.add_argument(
        "--Emin",
        action="store",
        type=int,
        default=20,
        help="Minumum energy of incident particle [GeV]",
    )

    parser.add_argument(
        "--outdir",
        action="store",
        type=str,
        default=".",
        help="Output directory for the LCIO file and log",
    )

    return parser


def initialize_writer(outfile):
    wrt = IOIMPL.LCFactory.getInstance().createLCWriter()
    wrt.open(outfile, EVENT.LCIO.WRITE_NEW)
    print("Opened outfile:", outfile)
    return wrt


def write_run_header(wrt, pdg, charge, mass):
    run = IMPL.LCRunHeaderImpl()
    run.setRunNumber(0)
    run.parameters().setValue("Generator", "${lcgeo}_DIR/examples/lcio_particle_gun.py")
    run.parameters().setValue("PDG", pdg)
    run.parameters().setValue("Charge", charge)
    run.parameters().setValue("Mass", mass)
    wrt.writeRunHeader(run)


def initialize_event(j):
    col = IMPL.LCCollectionVec(EVENT.LCIO.MCPARTICLE)
    evt = IMPL.LCEventImpl()
    evt.setEventNumber(j)
    evt.addCollection(col, "MCParticle")
    return evt, col


def calculate_energy_and_momentum(mass, p):
    energy = math.sqrt(mass * mass + p * p)
    momentum = array("f", [0, p, 0])  # Fixed orientation
    return energy, momentum


def sample_points(radius, n_points, sep):
    points = []
    for _ in range(n_points):
        # Sample a random center within the radius
        angle = np.random.uniform(0, 2 * np.pi)
        center_dist = np.sqrt(np.random.uniform(0, 1)) * radius
        center_x = center_dist * np.cos(angle)
        center_y = center_dist * np.sin(angle)
        center = np.array([center_x, center_y])

        # Sample a random distance between 0 and sep
        dist = np.random.uniform(0, sep)

        # Calculate the two points equidistant from the center along a line
        dx = dist / 2
        point1 = np.array([center_x + dx, center_y])
        point2 = np.array([center_x - dx, center_y])

        # Rotate the points around the center by a random angle
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        point1 = center + np.dot(rotation_matrix, point1 - center)
        point2 = center + np.dot(rotation_matrix, point2 - center)

        points.append((point1, point2))

    return np.array(points)


def create_vertex_and_endpoint(positions, centralPosZ):
    vy = 1804.7

    vx_1, vz_1 = positions[0]
    vx_2, vz_2 = positions[1]

    vertex_1 = array("d", [vx_1, vy, vz_1 + centralPosZ])
    endpoint_1 = array("d", [vx_1, vy, vz_1 + centralPosZ])

    vertex_2 = array("d", [vx_2, vy, vz_2 + centralPosZ])
    endpoint_2 = array("d", [vx_2, vy, vz_2 + centralPosZ])

    return vertex_1, endpoint_1, vertex_2, endpoint_2


def create_mcparticle(genstat, mass, pdg, momentum, charge, vertex, endpoint):
    mcp = IMPL.MCParticleImpl()
    mcp.setGeneratorStatus(genstat)
    mcp.setMass(mass)
    mcp.setPDG(pdg)
    mcp.setMomentum(momentum)
    mcp.setCharge(charge)
    mcp.setVertex(vertex)
    mcp.setEndpoint(endpoint)
    return mcp


def write_to_lcio(outfile, nevt, pdg, mass, charge, maxSep, centralPosZ, Emax, Emin):
    wrt = initialize_writer(outfile)
    random.seed()

    genstat = 1
    write_run_header(wrt, pdg, charge, mass)

    evt_no_arr, photon_sep_arr, pos_z_1_arr, pos_z_2_arr = [], [], [], []

    positions = sample_points(10, nevt, maxSep)

    for j in range(nevt):
        evt, col = initialize_event(j)

        photon_sep = random.uniform(0, maxSep)
        p1, p2 = random.uniform(Emin, Emax), random.uniform(Emin, Emax)

        _, momentum1 = calculate_energy_and_momentum(mass, p1)
        _, momentum2 = calculate_energy_and_momentum(mass, p2)

        vertint = create_vertex_and_endpoint(positions[j], centralPosZ)
        vertex1, endpoint1, vertex2, endpoint2 = vertint

        mcp1 = create_mcparticle(
            genstat, mass, pdg, momentum1, charge, vertex1, endpoint1
        )
        mcp2 = create_mcparticle(
            genstat, mass, pdg, momentum2, charge, vertex2, endpoint2
        )

        col.addElement(mcp1)
        col.addElement(mcp2)

        wrt.writeEvent(evt)

        evt_no_arr.append(j)
        photon_sep_arr.append(photon_sep)
        pos_z_1_arr.append(vertex1[2])
        pos_z_2_arr.append(vertex2[2])

    wrt.close()
    return evt_no_arr, photon_sep_arr, pos_z_1_arr, pos_z_2_arr


if __name__ == "__main__":

    parser = get_parser()
    parse_args = parser.parse_args()

    pdgid = parse_args.pdg
    mass = parse_args.mass
    charge = parse_args.charge

    """
    aMax_theta = parse_args.angleMax_theta
    aMin_theta = parse_args.angleMin_theta

    aMax_phi = parse_args.angleMax_phi
    aMin_phi = parse_args.angleMin_phi
    """

    mSep = parse_args.maxSep
    cPosZ = parse_args.centralPosZ

    engrMax = parse_args.Emax
    engrMin = parse_args.Emin

    outdir = parse_args.outdir.rstrip("/")

    for i in range(1, 2):
        evt_no_arr, photon_sep_arr, pos_z_1_arr, pos_z_2_arr = write_to_lcio(
            f"{outdir}/ILD-gen-E{engrMax}{engrMin}pdg{pdgid}maxSep{mSep}centralPosZ{cPosZ}-p{i}.slcio",
            15000,
            pdgid,
            mass,
            charge,
            mSep,
            cPosZ,
            engrMax,
            engrMin,
        )

        ##--- write file purely for logging purposes
        # --- Array for logging
        evt_no_np = np.asarray(evt_no_arr)
        photon_sep_np = np.asarray(photon_sep_arr)
        pos_z_1_np = np.asarray(pos_z_1_arr)
        pos_z_2_np = np.asarray(pos_z_2_arr)

        print(np.shape(evt_no_np))

        stacked_out = np.column_stack(
            (evt_no_np, photon_sep_np, pos_z_1_np, pos_z_1_np, pos_z_2_np)
        )
        print(np.shape(stacked_out))
        np.savetxt(
            f"{outdir}/gen-E{engrMax}{engrMin}pdg{pdgid}maxSep{mSep}centralPosZ{cPosZ}-p{i}.csv",
            stacked_out.reshape(-1, 5),
            fmt="%.6f",
            delimiter=",",
            header="evt, photon sep (mm), pos z1 (mm), pos z2 (mm)",
        )
