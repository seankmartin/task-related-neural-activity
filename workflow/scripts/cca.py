import logging
import numpy as np

import pandas as pd
from trna.common import load_config, split_spikes_into_trials
from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.dimension_reduction import scikit_cca
from trna.plot import plot_cca_correlation, plot_cca_correlation_features
from trna.common import (
    regions_to_string,
    name_from_recording,
    load_data,
    save_info_to_file,
)

from simuran.bridges.ibl_wide_bridge import IBLWideBridge
from simuran.bridges.allen_vbn_bridge import AllenVBNBridge
from simuran.plot.figure import SimuranFigure
from simuran.loaders.allen_loader import BaseAllenLoader
from simuran import set_only_log_to_file
from simuran.analysis.unit import bin_spike_train

module_logger = logging.getLogger("simuran.custom.cca")


def analyse_single_recording(recording, gpfa_window, out_dir, base_dir, brain_regions):
    print("Analysing recording: " + recording.get_name_for_save(base_dir))
    rel_dir = base_dir
    is_allen = isinstance(recording.loader, BaseAllenLoader)
    bridge = AllenVBNBridge() if is_allen else IBLWideBridge()
    region1, region2 = brain_regions
    unit_table1, spike_train1 = bridge.spike_train(recording, brain_regions=[region1])
    unit_table2, spike_train2 = bridge.spike_train(recording, brain_regions=[region2])
    unit_table = pd.concat([unit_table1, unit_table2])
    if len(unit_table) < 10:
        return None
    regions_as_str = regions_to_string(brain_regions)
    trial_info = bridge.trial_info(recording)

    out_dir.mkdir(parents=True, exist_ok=True)
    unit_table.to_csv(
        out_dir
        / name_from_recording(
            recording, f"unit_table_{regions_as_str}.csv", rel_dir=rel_dir
        )
    )
    for t in range(-20, 21, 2):
        per_trial_spikes1 = split_spikes_into_trials(
            spike_train1, trial_info["trial_times"], end_time=gpfa_window
        )
        per_trial_spikes2 = split_spikes_into_trials(
            spike_train2, trial_info["trial_times"], end_time=gpfa_window, delay=t
        )
        correct = []
        incorrect = []
        for trial1, trial2, correct_ in zip(
            per_trial_spikes1, per_trial_spikes2, trial_info["trial_correct"]
        ):
            binned_spikes1 = bin_spike_train(trial1, 0.02, t_stop=gpfa_window)
            binned_spikes2 = bin_spike_train(trial2, 0.02, t_stop=gpfa_window)
            cca, X, Y = scikit_cca(binned_spikes1, binned_spikes2)
            if correct_:
                correct.append([t, [X, Y]])
            else:
                incorrect.append([t, [X, Y]])
        correct = np.array(correct)
        incorrect = np.array(incorrect)

    info = {
        "scikit": {"correct": correct, "incorrect": incorrect},
    }
    save_info_to_file(info, recording, out_dir, brain_regions, rel_dir, bit="cca")
    with open(out_dir / f"cca_{regions_as_str}.txt", "w") as f:
        f.write(
            "Finished analysing: "
            + recording.get_name_for_save(rel_dir)
            + f" with {len(correct)} correct and {len(incorrect)} incorrect trials and {len(unit_table)} units"
        )
    return info


def plot_data(recording, info, out_dir, brain_regions, rel_dir=None):
    regions_as_str = regions_to_string(brain_regions)
    for key, value in info.items():
        correct = value["correct"]
        incorrect = value["incorrect"]
        fig = plot_cca_correlation_features(correct, incorrect, brain_regions)
        out_name = name_from_recording(
            recording, f"cca_{key}_{regions_as_str}.png", rel_dir=rel_dir
        )
        fig = SimuranFigure(fig, str(out_dir / out_name))
        fig.save()


def analyse_container(overwrite, config, recording_container, brain_regions):
    is_allen = isinstance(recording_container[0].loader, BaseAllenLoader)
    regions = []
    for region in brain_regions:
        if isinstance(region, str):
            regions.append(region)
        else:
            for sub_region in region:
                regions.append(sub_region)
    if is_allen:
        rel_dir_path = "allen_data_dir"
        brain_regions = config["allen_brain_regions"]
        n = "allen"
    else:
        rel_dir_path = "ibl_data_dir"
        brain_regions = config["ibl_brain_regions"]
        n = "ibl"
    for i, recording in enumerate(recording_container):
        output_dir = (
            config["output_dir"]
            / "cca"
            / recording.get_name_for_save(rel_dir=rel_dir_path)
        )
        info = load_data(
            recording, output_dir, regions, rel_dir=config[rel_dir_path], bit="cca"
        )
        if info is None or overwrite:
            recording = recording_container.load(i)
            info = analyse_single_recording(
                recording,
                config["gpfa_window"],
                output_dir,
                config[rel_dir_path],
                regions,
            )
        if info is not None:
            plot_data(
                recording, info, output_dir, regions, rel_dir=config[rel_dir_path]
            )
    all_info = []
    for i, recording in enumerate(recording_container):
        output_dir = (
            config["output_dir"]
            / "cca"
            / recording.get_name_for_save(rel_dir=rel_dir_path)
        )
        info = load_data(recording, output_dir, regions, rel_dir=config[rel_dir_path])
        if info is not None:
            all_info.append(info)
    output_dir = config["output_dir"] / "cca"
    fig = plot_cca_correlation(all_info, output_dir)
    regions = regions_to_string(brain_regions)
    sm_fig = SimuranFigure(fig, str(output_dir / f"{n}_{regions}_cca_correlation.png"))
    sm_fig.save()


def main(main_config, brain_table_location, overwrite=False):
    config = load_config(config=main_config)
    brain_table = pd.read_csv(brain_table_location)
    for brain_region_pair in config["allen_cca_regions"]:
        allen_recording_container, allen_loader = load_allen(
            config["allen_data_dir"], config["allen_manifest"], brain_region_pair
        )
        print(
            f"Loaded {len(allen_recording_container)} recordings from Allen with brain regions {brain_region_pair}"
        )
        analyse_container(
            overwrite, config, allen_recording_container, brain_region_pair
        )
        for brain_region in brain_region_pair:
            analyse_container(
                overwrite=overwrite,
                config=config,
                recording_container=allen_recording_container,
                brain_regions=brain_region,
            )
    for brain_region_pair in config["ibl_cca_regions"]:
        ibl_recording_container, ibl_loader = load_ibl(
            config["ibl_data_dir"], brain_table, brain_region_pair
        )
        print(
            f"Loaded {len(ibl_recording_container)} recordings from IBL with brain regions {config['ibl_brain_regions']}"
        )
        analyse_container(overwrite, config, ibl_recording_container, brain_region_pair)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        set_only_log_to_file(snakemake.log[0])
        main(snakemake.config, snakemake.input[0], snakemake.params.overwrite)
    else:
        main(None, r"G:/OpenData/OpenDataResults/tables/ibl_brain_regions.csv", False)
