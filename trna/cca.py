import logging
import numpy as np
import pandas as pd

from trna.dimension_reduction import scikit_cca
from trna.common import (
    save_info_to_file,
    ensure_enough_units,
    split_spikes_into_trials,
    regions_to_string,
    name_from_recording,
)

from simuran.bridges.ibl_wide_bridge import IBLWideBridge
from simuran.bridges.allen_vbn_bridge import AllenVBNBridge
from simuran.loaders.allen_loader import BaseAllenLoader
from simuran.analysis.unit import bin_spike_train

module_logger = logging.getLogger("simuran.custom.cca")


def analyse_single_recording(
    recording,
    gpfa_window,
    out_dir,
    base_dir,
    brain_regions,
    t_range=20,
    stack_method="hstack",
    filter_prop=None,
):
    print("Analysing recording: " + recording.get_name_for_save(base_dir))
    rel_dir = base_dir
    is_allen = isinstance(recording.loader, BaseAllenLoader)
    br_str = "structure_acronym" if is_allen else "acronym"
    bridge = (
        AllenVBNBridge(good_unit_properties=filter_prop)
        if is_allen
        else IBLWideBridge(good_unit_properties=filter_prop)
    )
    region1, region2 = brain_regions
    unit_table1, spike_train1 = bridge.spike_train(recording, brain_regions=[region1])
    unit_table2, spike_train2 = bridge.spike_train(recording, brain_regions=[region2])
    unit_table = pd.concat([unit_table1, unit_table2])
    regions_as_str = regions_to_string(brain_regions)
    trial_info = bridge.trial_info(recording)

    out_dir.mkdir(parents=True, exist_ok=True)
    unit_table_name = name_from_recording(
        recording, f"unit_table_{regions_as_str}.csv", rel_dir
    )
    unit_table_name = "--".join(unit_table_name.split("--")[-2:])
    unit_table.to_csv(out_dir / unit_table_name)
    if not ensure_enough_units(unit_table, 15, br_str):
        module_logger.warning(
            "Not enough units for {} in each brain region".format(
                recording.get_name_for_save()
            )
        )
        save_info_to_file(None, recording, out_dir, brain_regions, rel_dir, bit="cca")
        return None

    if t_range == 0:
        r = [0]
    else:
        r = range(-t_range, t_range, 2)

    correct = []
    incorrect = []
    for t in r:
        per_trial_spikes1 = split_spikes_into_trials(
            spike_train1, trial_info["trial_times"], end_time=gpfa_window
        )
        per_trial_spikes2 = split_spikes_into_trials(
            spike_train2, trial_info["trial_times"], end_time=gpfa_window, delay=t
        )
        if stack_method == "vstack":
            full_binned_spikes1 = []
            full_binned_spikes2 = []
            corrects = []
        for trial1, trial2, correct_ in zip(
            per_trial_spikes1, per_trial_spikes2, trial_info["trial_correct"]
        ):
            binned_spikes1 = bin_spike_train(trial1, 0.1, t_stop=gpfa_window)
            binned_spikes2 = bin_spike_train(trial2, 0.1, t_stop=gpfa_window)
            s1 = binned_spikes1.sum()
            s2 = binned_spikes2.sum()
            if s1 == 0 or s2 == 0:
                module_logger.warning(
                    f"Skipping trial with no spikes in one of the regions at delay {t}"
                )
                continue
            if stack_method == "hstack":
                cca, X, Y = scikit_cca(binned_spikes1.T, binned_spikes2.T)
                if correct_:
                    correct.append([t, [X, Y], [binned_spikes1, binned_spikes2]])
                else:
                    incorrect.append([t, [X, Y], [binned_spikes1, binned_spikes2]])
            else:
                corrects.append(correct_)
                full_binned_spikes1.append(binned_spikes1)
                full_binned_spikes2.append(binned_spikes2)
        if stack_method == "vstack":
            cca, X, Y = scikit_cca(
                np.concatenate(full_binned_spikes1, axis=1).T,
                np.concatenate(full_binned_spikes2, axis=1).T,
            )
            start_size = full_binned_spikes1[0].shape[1]
            for i, c in enumerate(corrects):
                x = X[i * start_size : (i + 1) * start_size]
                y = Y[i * start_size : (i + 1) * start_size]
                binned_spikes1 = full_binned_spikes1[i]
                binned_spikes2 = full_binned_spikes2[i]
                if c:
                    correct.append([t, [x, y], [binned_spikes1, binned_spikes2]])
                else:
                    incorrect.append([t, [x, y], [binned_spikes1, binned_spikes2]])

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
