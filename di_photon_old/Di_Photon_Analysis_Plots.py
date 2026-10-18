#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import string

fig_num = 0
title = ""
out_path = "./analysis_plots"
if not os.path.exists(out_path):
    os.makedirs(out_path)


def save_fig(xlabel="", ylabel=""):
    global fig_num
    global title
    if title:
        safe_title = title.translate(dict(zip(string.punctuation, ""))).replace(
            " ", "_"
        )
        save_name = f"{fig_num}_{safe_title}.png"
    else:
        save_name = f"{fig_num}.png"
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(os.path.join(out_path, save_name))
    plt.close()
    fig_num += 1


mckeownp_base_path = "/data/dust/user/mckeownp/af-ilc.merged/DiPhotonAnalysis/DDML_DiPhoton_Analysis/output"
akorol_base_path_2 = "/data/dust/user/akorol/projects/analysis_chain/output"
henry_base_path = "/data/dust/user/dayhallh/physical_recos/rec_output"


paths = {
    "30_mm": {
        "G4": {
            #"SEED1": f"{mckeownp_base_path}/G4_Di_Photon_E2020_SEED1.csv",
            #"SEED2": f"{mckeownp_base_path}/G4_Di_Photon_E2020_SEED2.csv",
            #"SEED3": f"{mckeownp_base_path}/G4_Di_Photon_E2020_SEED3.csv",
            # "SEED1": f"{henry_base_path}/g4_20250112_200139_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
            "SEED1": f"{henry_base_path}/g4_output_files/g4_20250113_170817_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
        },
        #"G4O": {
        #    "SEED1": f"{henry_base_path}/g4_output_files/g4_20250113_170817_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
        #    },

        "G4_RB": {
            "SEED2": f"{mckeownp_base_path}/G4_Di_Photon_E2020_SEED2.csv",
        },
        "CaloClouds_clone": {
            "SEED1": f"{henry_base_path}/ccO_output_files/ccO_20250111_064152_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
        },
        "CaloClouds_A2": {
            #"SEED1": f"{henry_base_path}/ccA2_output_files/ccA2_20250111_050025_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
            "SEED1": f"{henry_base_path}/../output/ILD-gen-E2020pdg22maxSep30centralPosZ50-p1/cciiA2_20250122_180621_Analysis.csv"
        },
        "CaloClouds_A4": {
            "SEED1": f"{henry_base_path}/ccA4_output_files/ccA4_20250110_180746_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
        },
        "CaloClouds_A10": {
            #"SEED1": f"{henry_base_path}/ccA10_output_files/ccA10_20250114_210118_ILD_l5_o1_v02_REC_diPhotonAnalysis.csv",
            "SEED1": f"{henry_base_path}/../output/ILD-gen-E2020pdg22maxSep30centralPosZ50-p1/cciiA10_20250122_210243_Analysis.csv"
        },
        "CaloClouds_O10": {
            "SEED1": f"{henry_base_path}/../output/ILD-gen-E2020pdg22maxSep30centralPosZ50-p1/cciiO10_20250122_210258_Analysis.csv"

        },

        "BiBAE": {},
        "L2LFlows": {
            "SEED1": f"{mckeownp_base_path}/L2LFlows_Di_Photon_E2020_SEED1.csv",
            "SEED2": f"{mckeownp_base_path}/L2LFlows_Di_Photon_E2020_SEED2.csv",
            "SEED3": f"{mckeownp_base_path}/L2LFlows_Di_Photon_E2020_SEED3.csv",
        },
        "L2LFlows_x9": {
            "SEED1": f"{mckeownp_base_path}/L2LFlowsx9_Di_Photon_E2020_SEED1.csv",
            "SEED2": f"{mckeownp_base_path}/L2LFlowsx9_Di_Photon_E2020_SEED2.csv",
            "SEED3": f"{mckeownp_base_path}/L2LFlowsx9_Di_Photon_E2020_SEED3.csv",
        },
    },
    "90_mm": {
        "G4": {
            "SEED1": f"{akorol_base_path_2}/ILD-gen-E2020pdg22maxSep90centralPosZ70-p1.slcio/g4_20250108_221926_Analysis.csv"
        },
        "CaloClouds": {
            "SEED1": f"{akorol_base_path_2}/ILD-gen-E2020pdg22maxSep90centralPosZ70-p1.slcio/cc_20250113_113930_Analysis.csv"
        },
        "BiBAE": {
            "SEED1": f"{akorol_base_path_2}/ILD-gen-E2020pdg22maxSep90centralPosZ70-p1.slcio/cc_20250108_204626_Analysis.csv",  # CaloClouds3_CD_small_2025_01_07__08_26_02_ckpt_6135000.pt CM
        },
        "L2LFlows": {},
        "L2LFlows_x9": {},
        "G4_RB": {},
    },
    "90_mm_endcap": {
        "G4": {
            "SEED1": f"{akorol_base_path_2}/ILD-endcap-angled-gen-E2020pdg22maxSep90centralXYZ-0-740-2411.7_p1.slcio/g4_20250110_190531_Analysis.csv",
        },
        "CaloClouds": {
            "SEED1": f"{akorol_base_path_2}/ILD-endcap-angled-gen-E2020pdg22maxSep90centralXYZ-0-740-2411.7_p1.slcio/cc_20250110_191738_Analysis.csv",  # 20 steps small
        },
        "BiBAE": {
            "SEED1": f"{akorol_base_path_2}/ILD-endcap-angled-gen-E2020pdg22maxSep90centralXYZ-0-740-2411.7_p1.slcio/bibae_20250110_191808_Analysis.csv",
        },
    },
}


linestyle = {
    "G4": "-",
    "G4_RB": "--",
    "CaloClouds": "--",
    "CaloClouds_clone": "--",
    "CaloClouds_A2": "--",
    "CaloClouds_A4": "--",
    "CaloClouds_A10": "--",
    "CaloClouds_O10": "--",
    "CaloClouds_II_A2": "--",
    "CaloClouds_II_A10": "--",
    "CaloClouds_II_O10": "--",
    "BiBAE": "--",
    "L2LFlows": "--",
    "L2LFlows_x9": "--",
}
colors = {
    "G4": "gray",
    "G4_RB": "tab:red",
    "CaloClouds": "tab:orange",
    "CaloClouds_clone": "tab:orange",
    "CaloClouds_A2": "tab:green",
    "CaloClouds_A4": "turquoise",
    "CaloClouds_A10": "steelblue",
    "CaloClouds_O10": "hotpink",
    "CaloClouds_II_A2": "red",
    "CaloClouds_II_A10": "purple",
    "CaloClouds_II_O10": "orange",
    #"CaloClouds_II_A2": "tab:green",
    #"CaloClouds_II_A10": "steelblue",
    #"CaloClouds_II_O10": "steelblue",
    "BiBAE": "tab:blue",
    "L2LFlows": "violet",
    "L2LFlows_x9": "darkviolet",
}
labels = {
    "G4": "Geant4",
    "G4_RB": "Geant4 R_B-field",
    "CaloClouds": "CaloClouds3",
    "CaloClouds_clone": "CaloClouds3 (clone)",
    "CaloClouds_A2": "CaloClouds3 (alt1, 2 blocks)",
    "CaloClouds_A4": "CaloClouds3 (alt1, 4 blocks)",
    "CaloClouds_A10": "CaloClouds3 (alt1, 10 blocks)",
    "CaloClouds_O10": "CaloClouds3 (alt1, 10 blocks)",
    "CaloClouds_II_A2": "CaloClouds3 (alt1, 2 blocks)",
    "CaloClouds_II_A10": "CaloClouds3 (alt1, 10 blocks)",
    "CaloClouds_II_O10": "CaloClouds3 (alt1, 10 blocks)",
    "BiBAE": "BiBAE",
    "L2LFlows": "L2LFlows",
    "L2LFlows_x9": "L2LFlows (x9)",
}
hatch = {
    "G4": None,
    "G4_RB": None,
    "CaloClouds": None,
    "CaloClouds_clone": None,
    "CaloClouds_A2": None,
    "CaloClouds_A4": None,
    "CaloClouds_A10": None,
    "CaloClouds_O10": None,
    "CaloClouds_II_A2": None,
    "CaloClouds_II_A10": None,
    "CaloClouds_II_O10": None,
    "BiBAE": None,
    "L2LFlows": None,
    "L2LFlows_x9": None,
}


sep = "30_mm"
# sep = "90_mm_endcap"
# sep = "90_mm"
# sep = "90_mm_endcap"
# models = ["G4", "CaloClouds", "BiBAE", "L2LFlows_x9"]
models = ["G4", "CaloClouds_A2", "CaloClouds_A10", "CaloClouds_O10"]
data = {}
for model in models:
    data[model] = {}
    for seed in paths[sep][model]:
        data[model][seed] = pd.read_csv(paths[sep][model][seed])[:]
        print(data[model][seed]["True_photon_separation (mm)"][0])
    if not data[model]:
        del data[model]


def plot_type_A(g4_kwargs=None, model_kwargs=None):
    fig = plt.figure(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    if g4_kwargs is None:
        g4_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}

    g4_kwargs["bins"] = g4_kwargs.get("bins", 100)
    g4_kwargs["range"] = g4_kwargs.get("range", (-40, 40))
    g4_kwargs["color"] = g4_kwargs.get("color", colors["G4"])
    g4_kwargs["histtype"] = g4_kwargs.get("histtype", "stepfilled")

    h = plt.hist(
        data["G4"]["SEED1"].PFO_1_E + data["G4"]["SEED1"].PFO_2_E - 40, **g4_kwargs
    )

    model_kwargs["histtype"] = model_kwargs.get("histtype", "step")
    model_kwargs["lw"] = model_kwargs.get("lw", 1.5)
    model_kwargs["linestyle"] = model_kwargs.get("linestyle", "-")
    for model in models:
        if model == "G4":
            continue
        plt.hist(
            data[model]["SEED1"].PFO_1_E + data[model]["SEED1"].PFO_2_E - 40,
            bins=h[1],
            color=colors[model],
            **model_kwargs,
        )
    return fig


try:
    title = "Plot 1"
    fig = plot_type_A()
    save_fig(xlabel="Sum particles - 40", ylabel="Counts")
    do_type_A = True
except Exception as e:
    print(e)
    do_type_A = False
    pass


if do_type_A:
    fig = plot_type_A()
    plt.yscale("log")
    save_fig(xlabel="Sum particles - 40", ylabel="Counts")


def get_stats(df, bins=15, bins_range=(0, int(sep.split("_")[0]))):
    if "nPFOs" in df.columns:
        true_sep = df["True_photon_separation (mm)"].values
        nPFOs = df["nPFOs"].values
        nPFOPhotons = df["nPFOPhotons"].values
    else:
        true_sep = df["True_photon_separation (mm)"].values
        nPFOs = df[" nPFOs"].values
        nPFOPhotons = df[" nPFOPhotons"].values

    photon_counts, bin_edges = np.histogram(
        true_sep, weights=nPFOPhotons, bins=bins, range=bins_range
    )
    particles_counts, bin_edges = np.histogram(
        true_sep, weights=nPFOs, bins=bins, range=bins_range
    )
    event_counts, _ = np.histogram(true_sep, bins=bin_edges)

    avg_num_photons = photon_counts / event_counts
    avg_num_particles = particles_counts / event_counts

    photon_counts, bin_edges = np.histogram(
        true_sep, weights=nPFOPhotons == 2, bins=bins, range=bins_range
    )

    return avg_num_photons, avg_num_particles, bin_edges


bins = 18
bins_range = (0, int(sep.split("_")[0]))


num_rec_photons = {}
num_rec_particles = {}
for model in data:
    num_rec_photons[model] = []
    num_rec_particles[model] = []
    for seed in data[model]:
        avg_num_photons, avg_num_particles, bin_edges = get_stats(
            data[model][seed], bins, bins_range
        )
        num_rec_photons[model].append(avg_num_photons)
        num_rec_particles[model].append(avg_num_particles)


# mean = np.mean(num_rec_photons['G4'], axis=0)
# std = np.std(num_rec_photons['G4'], axis=0)
width = np.diff(bin_edges)[0]
bin_centers = bin_edges[:-1] + width / 2


avg_rec_photons = {}
avg_rec_particles = {}

for model in num_rec_photons:

    mean_photons = np.mean(num_rec_photons[model], axis=0)
    std_phtons = np.std(num_rec_photons[model], axis=0)
    avg_rec_photons[model] = (mean_photons, std_phtons)

    mean_particles = np.mean(num_rec_particles[model], axis=0)
    std_particles = np.std(num_rec_particles[model], axis=0)
    avg_rec_particles[model] = (mean_particles, std_particles)


avg_rec_photons = {}
avg_rec_particles = {}

for model in num_rec_photons:

    mean_photons = np.mean(num_rec_photons[model], axis=0)
    std_phtons = np.std(num_rec_photons[model], axis=0)
    avg_rec_photons[model] = (mean_photons, std_phtons)

    mean_particles = np.mean(num_rec_particles[model], axis=0)
    std_particles = np.std(num_rec_particles[model], axis=0)
    avg_rec_particles[model] = (mean_particles, std_particles)


def get_ratio_correlated_error(
    error_numerator, numerator, error_denominator, denominator, correlation_coef=1
):
    # calculate errors for ratio of two correlated points. Automatically assumes perfect correlation
    ratio = numerator / denominator
    error_ratio = ratio * np.sqrt(
        (error_numerator / numerator) ** 2
        + (error_denominator / denominator) ** 2
        - 2
        * correlation_coef
        * (error_numerator / numerator)
        * (error_denominator / denominator)
    )
    # error_ratio = np.sqrt((error_numerator/denominator)**2 - numerator*error_denominator**2/denominator**2)
    return error_ratio


def get_ratio_uncorrelated_error(a, sigma_a, b, sigma_b):
    f = a / b
    sigma_f = np.sqrt((sigma_a / a) ** 2 + (sigma_b / b) ** 2)
    return sigma_f


def plot(data, ylabel="Avg. No. Reconstructed Particles"):

    mean_ref, std_ref = data["G4"]

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
    )
    fig.patch.set_facecolor("white")
    axes[0].set_ylim(0.5, 3.5)
    # axes[0].set_ylim(-0.3, 2.5)
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylim(0.9, 1.1)
    axes[1].set_xlabel("$d_{\\gamma \\gamma sep}$ (mm)", fontsize=35)
    axes[1].set_ylabel("Ratio to G4")

    axes[0].hist(
        bin_centers,
        bins=bin_edges,
        weights=mean_ref,
        histtype="step",
        linestyle=linestyle["G4"],
        color=colors["G4"],
        lw=1.5,
    )
    axes[0].bar(
        bin_centers,
        height=std_ref * 2,
        bottom=mean_ref - std_ref,
        alpha=0.3,
        width=width,
        color=colors["G4"],
        linewidth=0,
    )
    axes[1].axhline(1, color="black", lw=1.5)

    for model in data:
        if model == "G4":
            continue

        # main plot
        mean, std = data[model]
        try:
            axes[0].hist(
                bin_centers,
                bins=bin_edges,
                weights=mean,
                histtype="step",
                linestyle=linestyle[model],
                color=colors[model],
                lw=1.5,
            )
        except Exception as e:
            import ipdb

            ipdb.set_trace()
            print(e)
        axes[0].bar(
            bin_centers,
            height=std * 2,
            bottom=mean - std,
            alpha=0.3,
            width=width,
            color=colors[model],
            linewidth=0,
        )

        # sub pplot

        # error_R = get_ratio_correlated_error(mean, std, mean_ref, std_ref)
        error_R = get_ratio_correlated_error(std, mean, std_ref, mean_ref, 0)

        axes[1].hist(
            bin_centers,
            bins=bin_edges,
            weights=mean / mean_ref,
            histtype="step",
            linestyle=linestyle[model],
            color=colors[model],
            lw=1.5,
        )
        axes[1].bar(
            bin_centers,
            height=error_R * 2,
            bottom=mean / mean_ref - error_R,
            alpha=0.3,
            width=width,
            color=colors[model],
            linewidth=0,
        )

        print(model, abs(mean - mean_ref).sum())


title = "Avg. No. Reconstructed Particles"
fig = plot(avg_rec_photons, ylabel="Avg. No. Reconstructed Particles")
save_fig()


# # Refactored
#


def get_stats(df, bins=15, bins_range=(0, int(sep.split("_")[0]))):
    if "nPFOs" in df.columns:
        true_sep = df["True_photon_separation (mm)"].values
        nPFOs = df["nPFOs"].values
        nPFOPhotons = df["nPFOPhotons"].values
    else:
        true_sep = df["True_photon_separation (mm)"].values
        nPFOs = df[" nPFOs"].values
        nPFOPhotons = df[" nPFOPhotons"].values

    photon_counts, bin_edges = np.histogram(
        true_sep, weights=nPFOPhotons, bins=bins, range=bins_range
    )
    particles_counts, bin_edges = np.histogram(
        true_sep, weights=nPFOs, bins=bins, range=bins_range
    )
    event_counts, _ = np.histogram(true_sep, bins=bin_edges)

    avg_num_photons = photon_counts / event_counts
    avg_num_particles = particles_counts / event_counts

    successe_counts, bin_edges = np.histogram(
        true_sep, weights=(nPFOPhotons == 2).astype(int), bins=bins, range=bins_range
    )

    n = event_counts
    p = successe_counts / n
    q = 1 - p

    var = event_counts * p * q
    std = np.sqrt(var)

    return photon_counts / n, std / n


eff_rec_photons = {}
eff_rec_particles = {}

for model in data:
    eff_rec_photons[model] = {"mean": [], "std": []}
    for seed in data[model]:
        expected_counts, std = get_stats(data[model][seed], bins, bins_range)
        eff_rec_photons[model]["mean"].append(expected_counts)
        eff_rec_photons[model]["std"].append(std)

        # num_rec_particles[model].append(avg_num_particles)


df_photons = {}
for model in data:
    df_photons[model] = []
    for seed in data[model]:
        df_photons[model].append(data[model][seed])


df_combined = {}
for model in df_photons:
    df_combined[model] = pd.concat(df_photons[model]).reset_index(drop=True)


eff_rec_photons = {}
eff_rec_particles = {}

for model in data:
    eff_rec_photons[model] = {"eff": [], "std": []}
    expected_counts, std = get_stats(df_combined[model], bins, bins_range)
    eff_rec_photons[model]["eff"] = expected_counts
    eff_rec_photons[model]["std"] = std


data = {}
for model in eff_rec_photons:
    mean_rec_photons = eff_rec_photons[model]["eff"]
    std = eff_rec_photons[model]["std"]
    data[model] = (mean_rec_photons, std)


def plot(data, ylabel="Avg. No. Reconstructed Particles"):

    mean_ref, std_ref = data["G4"]

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
    )
    fig.patch.set_facecolor("white")
    # axes[0].set_ylim(0.5,3)
    # axes[0].set_ylim(0, 100)
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylim(0.9, 1.1)
    axes[1].set_xlabel("$d_{\\gamma \\gamma sep}$ (mm)", fontsize=35)
    axes[1].set_ylabel("Ratio to G4")

    axes[0].hist(
        bin_centers,
        bins=bin_edges,
        weights=mean_ref,
        histtype="step",
        linestyle=linestyle["G4"],
        color="black",
        lw=1,
    )
    axes[0].bar(
        bin_centers,
        height=std_ref * 2,
        bottom=mean_ref - std_ref,
        alpha=0.3,
        width=width,
        color=colors["G4"],
        linewidth=0,
    )
    axes[1].axhline(1, color="black", lw=1.5)

    for model in data:
        if model == "G4":
            continue

        # main plot
        mean, std = data[model]
        axes[0].hist(
            bin_centers,
            bins=bin_edges,
            weights=mean,
            histtype="step",
            linestyle=linestyle[model],
            color=colors[model],
            lw=2,
        )
        axes[0].bar(
            bin_centers,
            height=std * 2,
            bottom=mean - std,
            alpha=0.3,
            width=width,
            color=colors[model],
            linewidth=0,
            hatch=hatch[model],
        )

        # sub pplot

        error_R = get_ratio_uncorrelated_error(mean_ref, std_ref, mean, std)
        # error_R = get_ratio_correlated_error(std, mean, std_ref, mean_ref, 1)

        axes[1].hist(
            bin_centers,
            bins=bin_edges,
            weights=mean / mean_ref,
            histtype="step",
            linestyle=linestyle[model],
            color=colors[model],
            lw=2,
        )
        axes[1].bar(
            bin_centers,
            height=error_R * 2,
            bottom=mean / mean_ref - error_R,
            alpha=0.3,
            width=width,
            color=colors[model],
            linewidth=0,
            hatch=hatch[model],
        )

        kl = calculate_kl_divergence(mean_ref, mean) * 10000
        print(
            model,
            f"KL: {round(kl, 3)} * 1e-4",
        )
        # print(model, abs(mean_ref - mean).sum())

    axes[0].set_ylim(0.5, 3.5)
    axes[0].axhline(2, color="red")
    return fig


def get_ratio_uncorrelated_error(a, sigma_a, b, sigma_b):
    # f = a / b
    f = b / a
    sigma_f = f * np.sqrt((sigma_a / a) ** 2 + (sigma_b / b) ** 2)
    return sigma_f


def calculate_kl_divergence(hist1, hist2):

    # Normalize histograms to get probability distributions
    p = hist1 / np.sum(hist1)
    q = hist2 / np.sum(hist2)

    # Avoid division by zero and log of zero
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # Calculate KL divergence
    kl_divergence = np.sum(p * np.log(p / q))

    return kl_divergence


title = "Avg. No. Reconstructed Photons"
fig = plot(data, ylabel="Avg. No. Reconstructed Photons")
save_fig()


std_runs = []
for n in range(100):
    runs = []
    for i in range(n):
        n_photons = np.random.choice([1, 2], 500, p=[0.5, 0.5])
        runs.append(np.mean(n_photons))
    std_runs.append(np.std(runs) / np.sqrt(n))


title = "Standard Error"
plt.plot(std_runs, lw=0.5)
save_fig(ylabel="Standard Error")


n_events = 1000
bins = 15
positions = np.random.random(n_events)
positions_2 = np.random.random(n_events)
positions_2 = positions


def get_photons(positions):
    photons_in_event = []
    for p in positions:
        n_photons = np.random.choice([1, 2], 1, p=[(1 - p), p])
        photons_in_event.append(n_photons)
    photons_in_event = np.concatenate(photons_in_event)
    return photons_in_event


photons_in_event = get_photons(positions)

photons_in_event = []
positions_list = []
for i in range(100):
    positions = np.random.random(n_events)
    photons_in_event.append(get_photons(positions))
    positions_list.append(positions)


def get_stats(positions, photons_in_event, bins):
    photon_counts, bin_edges = np.histogram(
        positions, weights=photons_in_event, bins=bins, range=(0, 1)
    )
    event_counts, _ = np.histogram(positions, bins=bin_edges)

    n = event_counts
    successe_counts, _ = np.histogram(
        positions, weights=(photons_in_event == 2).astype(int), bins=bin_edges
    )
    p = successe_counts / event_counts
    q = 1 - p
    std = np.sqrt(n * p * q) / event_counts

    return photon_counts / event_counts, std, bin_edges


avg_photons_list = []
for i in range(100):
    avg_photons, std, bin_edges = get_stats(
        positions_list[i], photons_in_event[i], bins
    )
    avg_photons_list.append(avg_photons)


avg_photons = np.mean([avg_photons_list[i] for i in range(50)], axis=0)
avg_photons_2 = np.mean([avg_photons_list[i] for i in range(50, 100)], axis=0)

std = np.std([avg_photons_list[i] for i in range(3)], axis=0)
std_2 = np.std([avg_photons_list[i] for i in range(3, 6)], axis=0)


width = np.diff(bin_edges)[0]
bin_centers = bin_edges[:-1] + width / 2


fig, axes = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(7, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
)

axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons,
    histtype="step",
    color="black",
    lw=1,
)
axes[0].bar(
    bin_centers,
    height=std * 2,
    bottom=avg_photons - std,
    alpha=0.3,
    width=width,
    color="black",
    linewidth=0,
)

axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons_2,
    histtype="step",
    color="tab:orange",
    lw=1,
)
axes[0].bar(
    bin_centers,
    height=std_2 * 2,
    bottom=avg_photons_2 - std_2,
    alpha=0.3,
    width=width,
    color="tab:orange",
    linewidth=0,
)

true_mean = bin_centers + 1
axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=true_mean,
    histtype="step",
    color="blue",
    lw=1,
    linestyle="--",
)


error_R = get_ratio_uncorrelated_error(avg_photons_2, std_2, avg_photons, std)

axes[1].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons / avg_photons_2,
    histtype="step",
    color="tab:orange",
    lw=1,
)
axes[1].bar(
    bin_centers,
    height=error_R * 2,
    bottom=avg_photons / avg_photons_2 - error_R,
    alpha=0.3,
    width=width,
    color="tab:orange",
    linewidth=0,
)

n_overlap_bins = sum(abs(avg_photons / avg_photons_2 - 1) < error_R)
print(n_overlap_bins)

axes[1].axhline(1, color="black", lw=1.5)
axes[0].set_ylim(0.7, 2.5)
axes[1].set_ylim(0.95, 1.05)

plt.show()


fig, axes = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(7, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
)

axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons,
    histtype="step",
    color="black",
    lw=1,
)
axes[0].bar(
    bin_centers,
    height=std * 2,
    bottom=avg_photons - std,
    alpha=0.3,
    width=width,
    color="black",
    linewidth=0,
)

axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons_2,
    histtype="step",
    color="tab:orange",
    lw=1,
)
axes[0].bar(
    bin_centers,
    height=std_2 * 2,
    bottom=avg_photons_2 - std_2,
    alpha=0.3,
    width=width,
    color="tab:orange",
    linewidth=0,
)

true_mean = bin_centers + 1
# axes[0].hist(bin_centers, bins=bin_edges, weights=true_mean, histtype='step', color='blue', lw=1, linestyle='--')


error_R = get_ratio_uncorrelated_error(avg_photons_2, std_2, avg_photons, std)

axes[1].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons / avg_photons_2,
    histtype="step",
    color="tab:orange",
    lw=1,
)
axes[1].bar(
    bin_centers,
    height=error_R * 2,
    bottom=avg_photons / avg_photons_2 - error_R,
    alpha=0.3,
    width=width,
    color="tab:orange",
    linewidth=0,
)

n_overlap_bins = sum(abs(avg_photons / avg_photons_2 - 1) < error_R)
print(n_overlap_bins)

axes[1].axhline(1, color="black", lw=1.5)
axes[0].set_ylim(0.7, 2.5)
axes[1].set_ylim(0.95, 1.05)

title = ""
save_fig()


fig, axes = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(7, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
)

axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons,
    histtype="step",
    color="black",
    lw=1,
)
axes[0].bar(
    bin_centers,
    height=std * 2,
    bottom=avg_photons - std,
    alpha=0.3,
    width=width,
    color="black",
    linewidth=0,
)

axes[0].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons_2,
    histtype="step",
    color="tab:orange",
    lw=1,
)
axes[0].bar(
    bin_centers,
    height=std_2 * 2,
    bottom=avg_photons_2 - std_2,
    alpha=0.3,
    width=width,
    color="tab:orange",
    linewidth=0,
)

true_mean = bin_centers + 1
# axes[0].hist(bin_centers, bins=bin_edges, weights=true_mean, histtype='step', color='blue', lw=1, linestyle='--')


error_R = get_ratio_uncorrelated_error(avg_photons_2, std_2, avg_photons, std)

axes[1].hist(
    bin_centers,
    bins=bin_edges,
    weights=avg_photons / avg_photons_2,
    histtype="step",
    color="tab:orange",
    lw=1,
)
axes[1].bar(
    bin_centers,
    height=error_R * 2,
    bottom=avg_photons / avg_photons_2 - error_R,
    alpha=0.3,
    width=width,
    color="tab:orange",
    linewidth=0,
)

n_overlap_bins = sum(abs(avg_photons / avg_photons_2 - 1) < error_R)
print(n_overlap_bins)

axes[1].axhline(1, color="black", lw=1.5)
axes[0].set_ylim(0.7, 2.5)
axes[1].set_ylim(0.95, 1.05)

title = "Plot 6"
save_fig()
