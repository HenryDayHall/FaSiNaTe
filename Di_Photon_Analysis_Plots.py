#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

import pandas as pd


# Geant4 input paths
paths = {}

# these are on the straight line (sep to 30 mm)
paths["G4"] = [
    f"/gpfs/dust/ilc/user/mckeownp/DiPhotonAnalysis/DDML_DiPhoton_Analysis/output/G4_Di_Photon_E2020_SEED{i}.csv"
    for i in range(1, 4)
]

# 1, 2, 3
path_format = "/home/akorol/projects_gpfs/PC_version/ddml_diphoton_analysis/output/CD_ClusteredDS_Pretrained2024_03_27__00_23_43_ckpt_0.000000_2150000_ShowerFlow_alt1_nb2_calib_cog_seed{}_REC.csv"
# 1
path_format = "/gpfs/dust/maxwell/user/dayhallh/physical_recos/ana_output/di_photons_g4_output_files/ild-ana-e2020pdg22maxsep30centralposz50-p1-seed{}_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv"
paths["CaloClouds"] = [
    path_format.format(1) for i in range(1, 4)
]

#paths["L2LFlows"] = [
#    f"/home/akorol/projects_gpfs/PC_version/ddml_diphoton_analysis/output/BiBAE_TwoAngles_seed{i}_REC.csv"
#    for i in range(1, 4)
#]
#
#paths["L2LFlowsx9"] = [
#    f"/home/akorol/projects_gpfs/PC_version/ddml_diphoton_analysis/output/L2LFlowsx9_Di_Photon_E2020_SEED{i}.csv"
#    for i in range(1, 4)
#]

models = list(paths.keys())  # ensure fixed order
# Create Pandas dataframes


# Geant4
dataframes = {name: [pd.read_csv(path) for path in paths[name]] for name in models}

print("The keys of the first G4 dataframe are: ")
print(dataframes["G4"][0].keys())


# Functions- separate to other notebook
def ratio_correlated_error(
    error_numerator, numerator, error_denominator, denominator, correlation_coef=1
):
    # calculate errors for ratio of two correlated points.
    # Automatically assumes perfect correlation
    ratio = numerator / denominator
    error_ratio = ratio * np.sqrt(
        (error_numerator / numerator) ** 2
        + (error_denominator / denominator) ** 2
        - 2
        * correlation_coef
        * (error_numerator / numerator)
        * (error_denominator / denominator)
    )
    return error_ratio


def get_True_diphoton_sep(data_frame_arr):
    # Extract true diphoton separation.
    # Input: dataframe = array of dataframes for each seed
    # Output: array of each seed true photon separations

    keys = data_frame_arr[0].keys()
    sep_key = next(key for key in keys if "true_photon_separation" in key.lower())
    True_sep = [
        data_frame[sep_key] for data_frame in data_frame_arr
    ]
    return True_sep


def get_nPFOs_diphoton_sep(data_frame_arr):
    # Extract nPFOs
    # Input: dataframe = array of dataframes for each seed
    # Output: array of each seed nPFOs

    keys = data_frame_arr[0].keys()
    pfo_key = next(key for key in keys if "npfos" in key.lower())
    nPFOS = [data_frame[pfo_key] for data_frame in data_frame_arr]

    return nPFOS


def get_nPFOPhotons_diphoton_sep(data_frame_arr):
    # Extract nPFOPhotons
    # Input: dataframe = array of dataframes for each seed
    # Output: array of each seed nPFOs

    keys = data_frame_arr[0].keys()
    nPFOPhotons_key = next(
        key for key in keys if "npfophotons" in key.lower()
    )
    nPFOPhotons = [data_frame[nPFOPhotons_key] for data_frame in data_frame_arr]

    return nPFOPhotons


def get_fraction_fragment_energy_diphoton_sep(data_frame_arr):
    # Extract fraction_fragment_energy
    # Input: dataframe = array of dataframes for each seed
    # Output: array of each seed nPFOs
    keys = data_frame_arr[0].keys()
    fraction_fragment_energy_key = next(
        key for key in keys if "fraction_fragment_energy" in key.lower()
    )

    fraction_fragment_energy = [
        data_frame[fraction_fragment_energy_key] for data_frame in data_frame_arr
    ]

    return fraction_fragment_energy


def get_PFO_separation_diphoton_sep(data_frame_arr):
    # Extract fraction_fragment_energy
    # Input: dataframe = array of dataframes for each seed
    # Output: array of each seed nPFOs
    keys = data_frame_arr[0].keys()
    pfo_sep_key = next(
        key for key in keys if "pfo_separation" in key.lower()
    )
    PFO_separation = [
        data_frame[pfo_sep_key] for data_frame in data_frame_arr
    ]

    return PFO_separation


# Extract True separations
di_photon_True_sep = {
    model: get_True_diphoton_sep(dataframes[model]) for model in models
}


# Extract nPFOs
di_photon_nPFOs = {model: get_nPFOs_diphoton_sep(dataframes[model]) for model in models}


# Extract nPFOPhotons
di_photon_nPFOPhotons = {
    model: get_nPFOPhotons_diphoton_sep(dataframes[model]) for model in models
}
print("nPFOPhotons")
for model in models:
    for seed, nPFOPhotons in enumerate(di_photon_nPFOPhotons[model]):
        print(f"\tSeed {seed}: {nPFOPhotons.value_counts(normalize=False)}")

# Extract fraction_fragment_energy
di_photon_fragment_energy = {
    model: get_fraction_fragment_energy_diphoton_sep(dataframes[model])
    for model in models
}


# Extract PFO_separation
di_photon_PFO_separation = {
    model: get_PFO_separation_diphoton_sep(dataframes[model]) for model in models
}


# Plotting config
if True:
    font = {"family": "serif", "size": 19}
    mpl.rc("font", **font)
    plt.style.use("classic")
    mpl.rc("font", **font)
    lightcyan = (0.1, 0.9, 0.9, 0.2)
    lightred = (1, 0.1, 0.1, 0.3)
    lightgreen = (0.1, 1, 0.3, 0.3)
    lightblue = (0.1, 0.1, 0.9, 0.2)
    lightgrey = (0.1, 0.1, 0.1, 0.3)
    lightorange = (1.0, 0.5, 0.25, 0.3)

    very_dark_grey = (0.1, 0.1, 0.1, 0.8)

    color_list = []
    """
    color_list.append('dimgrey')
    color_list.append('blue')
    color_list.append('green')
    """

    # color_list.append('blue')
    color_list.append(very_dark_grey)  # 'dimgrey')
    color_list.append("darkorange")
    # color_list.append('red')
    color_list.append("tab:blue")
    color_list.append("darkviolet")
    color_list.append("green")
    # color_list.append('tab:blue')
    # color_list.append('tab:orange')
    # color_list.append('darkviolet')
    # color_list.append('green')

    fillcolor_list = []
    fillcolor_list.append(lightgrey)
    fillcolor_list.append(lightorange)
    fillcolor_list.append(lightblue)
    fillcolor_list.append(lightorange)
    fillcolor_list.append(lightgrey)
    fillcolor_list.append(lightblue)

    linewidth_list = []
    linewidth_list.append(2)
    linewidth_list.append(2)
    linewidth_list.append(2)
    linewidth_list.append(2)
    linewidth_list.append(2)
    linewidth_list.append(2)

    linestyle_list = []
    linestyle_list.append("-")
    linestyle_list.append("--")
    linestyle_list.append("--")
    linestyle_list.append("--")
    linestyle_list.append("-")
    linestyle_list.append(
        (0, (3, 1, 1, 1))
    )  # ('densely dashdotted',    (0, (3, 1, 1, 1))),
    linestyle_list.append("--")
    linestyle_list.append(
        (0, (3, 1, 1, 1))
    )  # ('densely dashdotted',    (0, (3, 1, 1, 1))),
    linestyle_list.append("--")
    linestyle_list.append(
        (0, (3, 1, 1, 1))
    )  # ('densely dashdotted',    (0, (3, 1, 1, 1))),


model_title = []
model_title.append("CaloClouds2A")
# model_title.append('L2LFlows')
model_title.append("BiBAE")
model_title.append("L2LFlowsx9")
model_title.append("EDM")


# For using training G4 version in ratio
def plot_Weighted_hist_errors(
    data_real,
    data_fake,
    weights_real,
    weights_fake,
    bins,
    minE,
    maxE,
    ymin,
    ymax,
    name,
    xtitle,
    ytitle,
    typ,
    p_typ,
    title_1,
    model_title,
    min_loc=5,
    maj_loc=10,
    log=False,
    norm=False,
    sub_min=0.0,
    sub_max=2.0,
    save_title=None,
):

    figSE, (axSE, axSub) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
    )

    if norm:
        pSEa = axSE.hist(
            data_real,
            bins=bins,
            range=[minE, maxE],
            density=None,
            weights=np.ones_like(data_real) / (float(len(data_real))),
            edgecolor=color_list[0],
            label="orig",
            linewidth=0,  # linewidth_list[0],
            color=fillcolor_list[0],
            linestyle=linestyle_list[0],
            histtype="step",
        )  # 'stepfilled')

    else:
        pSEa = [
            np.histogram(data, bins=bins, range=[minE, maxE], density=None)
            for data in data_real
        ]

        # Now calculate the weighted histograms
        pSEa_weighted = [
            np.histogram(
                data, bins=bins, range=[minE, maxE], density=None, weights=weights
            )
            for data, weights in zip(data_real, weights_real)
        ]

        print(np.shape(pSEa[0][0]))

        # average = weighted_hist/(hist)
        # array of histogramsm, shape: (nbins, nrepeats)
        pSEa_arr = np.concatenate(
            [
                np.asarray(weighted[0] / hist[0]).reshape((-1, 1))
                for hist, weighted in zip(pSEa, pSEa_weighted)
            ],
            axis=1,
        )

        pSEa_arr_avg = np.mean(pSEa_arr, axis=1)
        pSEa_arr_std = np.std(pSEa_arr, axis=1)

        #             print("G4 mean: ", pSEa_arr_avg)
        #             print("G4 std: ", pSEa_arr_std)
        #             return pSEa_arr_avg, pSEa_arr_std

        # Histogram of the mean
        # Append extra element at the end to fill entire histogram bar (i.e. add horizontal step to vertical step for bin)
        xs = np.append(pSEa[0][1][:-1], pSEa[0][1][-1] + 1)
        pSEa_mean = axSE.step(
            x=xs,
            y=np.append(pSEa_arr_avg, pSEa_arr_avg[-1]),
            color=very_dark_grey,
            linewidth=linewidth_list[0],
            where="post",
        )
        print(pSEa[0][1][:-1])
        print(pSEa[0][1][-1])

        # Error bars in the main plot
        pSEa_err = pSEa_arr_std  # np.sqrt(pSEa[0])
        pSEa_err_mean = pSEa_err  # /(np.sqrt(3))

        pSErr = axSE.bar(
            x=xs[:-1],
            height=2 * pSEa_err,
            bottom=(pSEa_arr_avg - pSEa_err),
            width=np.diff(pSEa[0][1]),
            align="edge",
            linewidth=0,
            color="slategrey",
            alpha=0.34,
            zorder=-1,
        )
        # width=np.diff(pSEa1[1]), align='edge', linewidth=0, color=very_dark_grey, alpha=0.15, zorder=-1)

        eps = 1e-5  # to avoid numerical error

        # ratio for real is straight line at 1
        pSuba = axSub.plot(
            np.linspace(minE, maxE, 1000),
            np.ones(1000),
            linewidth=linewidth_list[0],
            color=very_dark_grey,
        )

        # mask elements equal to zero
        pSEa_err[pSEa_err == 0] = eps
        pSEa_err_mean[pSEa_err_mean == 0] = eps
        pSEa_arr_avg[pSEa_arr_avg == 0] = eps

    axSE.plot(
        (0.32, 0.38),
        (0.86 - 0.02, 0.86 - 0.02),
        linewidth=linewidth_list[3],
        linestyle=linestyle_list[0],
        transform=axSE.transAxes,
        color=color_list[0],
    )
    axSE.text(
        0.40,
        0.86,
        title_1,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axSE.transAxes,
        color=color_list[0],
    )

    #         axSE.axhline(2, color='red')

    fake_err_ratio_list = []
    for j in range(len(data_fake[0])):
        if norm:
            pSpnEb = axSE.hist(
                data_fake[0][j],
                bins=pSEa[0][1],
                range=None,
                density=None,
                weights=np.ones_like(data_fake[j]) / (float(len(data_fake[j]))),
                edgecolor=color_list[j + 1],
                label="orig",
                linewidth=linewidth_list[
                    j + 1
                ],  # 1.0, #linewidth_list[0], #linewidth_list[j+1],
                linestyle=linestyle_list[
                    j + 1
                ],  # linestyle_list[0], #linestyle_list[j+1],
                histtype="step",
            )

        else:
            pSEb = [
                np.histogram(data[j], bins=pSEa[0][1], range=[minE, maxE], density=None)
                for data in data_fake
            ]

            pSEb_weighted = [
                np.histogram(
                    data[j],
                    bins=pSEa[0][1],
                    range=[minE, maxE],
                    density=None,
                    weights=weights[j],
                )
                for data, weights in zip(data_fake, weights_fake)
            ]
            # average = weighted_hist/(hist)
            # array of histogramsm, shape: (nbins, nrepeats)
            pSEb_arr = np.concatenate(
                [
                    np.asarray(pSEb_weighted[i][0] / pSEb[i][0]).reshape((-1, 1))
                    for i in range(len(data_fake))
                ],
                axis=1,
            )

            pSEb_arr_avg = np.mean(pSEb_arr, axis=1)
            pSEb_arr_std = np.std(pSEb_arr, axis=1)

            # Histogram of the mean
            pSEb_mean = axSE.step(
                x=xs,
                y=np.append(pSEb_arr_avg, pSEb_arr_avg[-1]),
                color=color_list[j + 1],
                linestyle=linestyle_list[j + 1],
                linewidth=linewidth_list[j + 1],
                where="post",
            )

            # Error bars in the main plot
            pSEb_err = pSEb_arr_std  # np.sqrt(pSEa[0])
            pSEb_err_mean = pSEb_err  # /(np.sqrt(3))

            colour = {0: "darkorange", 1: "tab:blue", 2: "darkviolet"}
            alpha = {0: 0.34, 1: 0.28, 2: 0.28}
            pSErr = axSE.bar(
                x=xs[:-1],
                height=2 * pSEb_err,
                bottom=(pSEb_arr_avg - pSEb_err),
                width=np.diff(pSEa[0][1]),
                align="edge",
                linewidth=0,
                color=colour[j],
                alpha=alpha[j],
                zorder=-1,
            )

            # sub plot
            pSubb = axSub.step(
                x=xs,
                y=np.append(
                    pSEb_arr_avg / pSEa_arr_avg, pSEb_arr_avg[-1] / pSEa_arr_avg[-1]
                ),
                color=color_list[j + 1],
                linewidth=linewidth_list[j + 1],
                where="post",
                linestyle=linestyle_list[j + 1],
            )

            pSEb_arr_avg[pSEb_arr_avg == 0] = eps
            pSEb_err_mean[pSEb_err_mean == 0] = eps

            # get model with correlated uncertainties
            # using the standard error on the mean! 3 measurements assumed
            fake_err_ratio = ratio_correlated_error(
                error_numerator=pSEb_err_mean,
                numerator=pSEb_arr_avg,
                error_denominator=pSEa_err_mean,
                denominator=pSEa_arr_avg,
                correlation_coef=0,
            )  # 1)
            # print(pSEb_arr_avg/pSEa_arr_avg)
            # print(fake_err_ratio)

            fake_err_ratio_clip = np.clip(fake_err_ratio, sub_min, sub_max)

            # mask out ratios that are almost zero for the error (i.e close to numerical precision)
            ratio = pSEb_arr_avg / pSEa_arr_avg
            fake_err_ratio[ratio < 1e-4] = 1e4

            fake_err_ratio_list.append(
                (pSEb_err_mean, pSEb_arr_avg, pSEa_err_mean, pSEa_arr_avg)
            )

            pSpnEb_subErr = axSub.bar(
                x=xs[:-1],
                height=2 * fake_err_ratio_clip,
                bottom=(ratio - fake_err_ratio_clip),
                width=np.diff(pSEa[0][1]),
                align="edge",
                linewidth=0,
                color=colour[j],
                alpha=alpha[j],
                zorder=-1,
            )

        axSE.plot(
            (0.32, 0.38),
            (0.86 - 0.02 - (j + 1) * 0.06, 0.86 - 0.02 - (j + 1) * 0.06),
            linewidth=linewidth_list[j + 1],
            linestyle=linestyle_list[j + 1],
            transform=axSE.transAxes,
            color=color_list[j + 1],
        )
        axSE.text(
            0.40,
            0.86 - (j + 1) * 0.06,
            model_title[j],
            horizontalalignment="left",
            verticalalignment="top",
            transform=axSE.transAxes,
            color=color_list[j + 1],
        )

    axSE.text(
        0.23,  # 0.10,#0.6,
        0.98,
        typ,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axSE.transAxes,
    )

    axSE.text(
        0.4,  # 0.62,#0.6,
        0.92,
        p_typ,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axSE.transAxes,
    )

    figSE.patch.set_facecolor("white")

    axSE.xaxis.set_minor_locator(MultipleLocator(min_loc))
    axSE.xaxis.set_major_locator(MultipleLocator(maj_loc))
    #     axSE.set_xticks([0, 10, 20, 30])
    axSE.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # axSE.set_xlim([minE, (maxE-(minE+maxE)/bins)])
    axSE.set_xlim([minE, maxE])
    print((maxE - (minE + maxE) / bins))

    axSE.set_ylabel(ytitle, fontsize=20)

    if log:
        axSE.set_yscale("log")
    axSub.set_xlabel(xtitle, family="serif", fontsize=34)
    axSub.set_ylabel("Ratio to G4", family="serif", fontsize=20)
    axSE.set_ylim([ymin, ymax])

    #     axSub.set_ylim([0, 2])
    #     axSub.set_ylim([0.2, 1.8])
    axSub.set_ylim([0.8, 1.2])
    axSub.xaxis.set_minor_locator(MultipleLocator(0.1))
    axSub.yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tight_layout()
    if save_title is not None:
        figSE.savefig(
            "./plots/" + save_title + ".png", bbox_inches="tight"
        )


#     return fake_err_ratio_list


# Some comments!
# - All these histograms will be weighted
# - Need to also account for this in the ratio plot...


n_seeds = len(di_photon_True_sep["G4"])
True_sep_diPhoton_arr = [
    [di_photon_True_sep[model][i] for model in models if model != "G4"]
    for i in range(n_seeds)
]


nPFOs_diPhoton_arr = [
    [di_photon_nPFOs[model][i] for model in models if model != "G4"]
    for i in range(n_seeds)
]

os.makedirs("plots", exist_ok=True)

plot_Weighted_hist_errors(
    di_photon_True_sep["G4"],
    True_sep_diPhoton_arr,
    di_photon_nPFOs["G4"],
    nPFOs_diPhoton_arr,
    15,
    0,
    30,
    0.5,
    3.0,
    "Photon Energy",
    "$d_{\\gamma \\gamma sep}$ (mm)",
    "Avg. No. Reconstructed Particles",
    "ILD Di-Photons",
    "$E_{\\gamma1}$=20GeV, $E_{\\gamma2}$=20GeV",
    "Geant4",
    model_title,
    min_loc=0.8,
    maj_loc=5,
    log=False,
    norm=False,
    save_title="a",
)

plt.close()

plot_Weighted_hist_errors(
    di_photon_True_sep["G4"],
    True_sep_diPhoton_arr,
    di_photon_nPFOs["G4"],
    nPFOs_diPhoton_arr,
    15,
    0,
    30,
    0.5,
    3.0,
    "Photon Energy",
    "$d_{\\gamma \\gamma sep}$ (mm)",
    "Avg. No. Reconstructed Photons",
    "ILD Di-Photons",
    "$E_{\\gamma1}$=20GeV, $E_{\\gamma2}$=20GeV",
    "Geant4",
    model_title,
    min_loc=0.5,
    maj_loc=4,
    log=False,
    norm=False,
    save_title="b",
)

plt.close()

plot_Weighted_hist_errors(
    di_photon_True_sep["G4"],
    True_sep_diPhoton_arr,
    di_photon_nPFOs["G4"],
    nPFOs_diPhoton_arr,
    15,
    0,
    30,
    0.5,
    3.0,
    "Photon Energy",
    "$d_{\\gamma \\gamma sep}$ (mm)",
    "Avg. No. Reconstructed Photons",
    "ILD Di-Photons",
    "$E_{\\gamma1}$=20GeV, $E_{\\gamma2}$=20GeV",
    "Geant4",
    model_title,
    min_loc=0.5,
    maj_loc=4,
    log=False,
    norm=False,
    save_title="c",
)

plt.close()

fraction_fragment_energy_diPhoton_arr = [
    [di_photon_fragment_energy[model][i] for model in models if model != "G4"]
    for i in range(n_seeds)
]

plot_Weighted_hist_errors(
    di_photon_True_sep["G4"],
    True_sep_diPhoton_arr,
    di_photon_fragment_energy["G4"],
    fraction_fragment_energy_diPhoton_arr,
    15,
    0,
    90,
    0,
    0.3,
    "Photon Energy",
    "$d_{\\gamma \\gamma sep}$ (mm)",
    "Avg. Fractional Fragment Energy",
    "ILD Di-Photons",
    "$E_{\\gamma1}$=20GeV, $E_{\\gamma2}$=20GeV",
    "Geant4",
    model_title,
    min_loc=0.5,
    maj_loc=4,
    log=False,
    norm=False,
    save_title="d",
)

plt.close()

fake_err_ratio_list = plot_Weighted_hist_errors(
    di_photon_True_sep["G4"],
    True_sep_diPhoton_arr,
    di_photon_fragment_energy["G4"],
    fraction_fragment_energy_diPhoton_arr,
    15,
    0,
    30,
    0,
    0.3,
    "Photon Energy",
    "$d_{\\gamma \\gamma sep}$ (mm)",
    "Avg. Fractional Fragment Energy",
    "ILD Di-Photons",
    "$E_{\\gamma1}$=20GeV, $E_{\\gamma2}$=20GeV",
    "Geant4",
    model_title,
    min_loc=0.5,
    maj_loc=4,
    log=False,
    norm=False,
    save_title="e",
)
