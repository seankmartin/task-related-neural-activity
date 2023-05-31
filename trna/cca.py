import logging
import numpy as np
import pandas as pd

from trna.dimension_reduction import scikit_cca, find_correlations, find_correlation
from trna.common import (
    save_info_to_file,
    ensure_enough_units,
    split_spikes_into_trials,
    regions_to_string,
    name_from_recording,
    average_firing_rate,
)

from simuran.bridges.ibl_wide_bridge import IBLWideBridge
from simuran.bridges.allen_vbn_bridge import AllenVBNBridge
from simuran.loaders.allen_loader import BaseAllenLoader
from simuran.analysis.unit import bin_spike_train

module_logger = logging.getLogger("simuran.custom.cca")


def analyse_single_recording(
    recording,
    cca_params,
    out_dir,
    base_dir,
    brain_regions,
    t_range=20,
    filter_prop=None,
):
    print("Analysing recording: " + recording.get_name_for_save(base_dir))
    cca_window = cca_params["cca_window"]
    cca_bin_size = cca_params["cca_binsize"] / 1000
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
        r = range(-t_range, t_range + 1, 2)

    correct = []
    incorrect = []
    per_trial_corr_info_correct = []
    per_trial_corr_info_incorrect = []
    correct_rates = []
    incorrect_rates = []
    correct_corr_info = []
    incorrect_corr_info = []
    unsplit_corr_info_correct = []
    unsplit_corr_info_incorrect = []
    for_concat = [[], []]
    per_trial_spikes1 = split_spikes_into_trials(
        spike_train1, trial_info["trial_times"], end_time=cca_window
    )
    for t in r:
        per_trial_spikes2 = split_spikes_into_trials(
            spike_train2, trial_info["trial_times"], end_time=cca_window, delay=t
        )
        full_binned_spikes1 = []
        full_binned_spikes2 = []
        corrects = []
        for trial1, trial2, correct_ in zip(
            per_trial_spikes1, per_trial_spikes2, trial_info["trial_correct"]
        ):
            # Method 1 - per trial CCA
            binned_spikes1 = bin_spike_train(trial1, cca_bin_size, t_stop=cca_window)
            binned_spikes2 = bin_spike_train(trial2, cca_bin_size, t_stop=cca_window)
            s1 = binned_spikes1.sum()
            s2 = binned_spikes2.sum()
            if s1 == 0 or s2 == 0:
                module_logger.warning(
                    f"Skipping trial with no spikes in one of the regions at delay {t}"
                )
                continue
            cca, X, Y = scikit_cca(binned_spikes1.T, binned_spikes2.T)
            if correct_:
                correct.append([t, [X, Y], [binned_spikes1, binned_spikes2]])
                per_trial_corr_info_correct.append([t, find_correlation(X, Y)])
            else:
                incorrect.append([t, [X, Y], [binned_spikes1, binned_spikes2]])
                per_trial_corr_info_incorrect.append([t, find_correlation(X, Y)])
            for_concat[0].append(binned_spikes1.T)
            for_concat[1].append(binned_spikes2.T)

            # Method 3 - average firing rate CCA
            average_firing_rate1 = average_firing_rate(trial1, cca_window)
            average_firing_rate2 = average_firing_rate(trial2, cca_window)
            corrects.append(correct_)
            full_binned_spikes1.append(average_firing_rate1)
            full_binned_spikes2.append(average_firing_rate2)

        # Method 3 - average firing rate CCA
        if len(full_binned_spikes1) == 0:
            continue
        if len(full_binned_spikes2) == 0:
            continue
        cca, X, Y = scikit_cca(
            np.stack(full_binned_spikes1, axis=0),
            np.stack(full_binned_spikes2, axis=0),
        )
        for i, c in enumerate(corrects):
            binned_spikes1 = full_binned_spikes1[i]
            binned_spikes2 = full_binned_spikes2[i]
            if c:
                correct_rates.append(
                    [t, [X[i], Y[i]], [binned_spikes1, binned_spikes2]]
                )
            else:
                incorrect_rates.append(
                    [t, [X[i], Y[i]], [binned_spikes1, binned_spikes2]]
                )

        # Method 2 - stacked CCA
        full_stacked = np.concatenate(for_concat[0], axis=0)
        full_stacked2 = np.concatenate(for_concat[1], axis=0)
        data_per_trial = int(cca_window / cca_bin_size)
        num_trials = int(len(full_stacked) / data_per_trial)
        cca, X, Y = scikit_cca(full_stacked, full_stacked2)
        corr = find_correlations(X, Y, num_trials, data_per_trial)
        for c, correct_ in zip(corr, corrects):
            if correct_:
                unsplit_corr_info_correct.append([t, c])
            else:
                unsplit_corr_info_incorrect.append([t, c])

        correct_concat1 = np.concatenate(
            [c[2][0].T for c in correct if c[0] == t], axis=0
        )
        correct_concat2 = np.concatenate(
            [c[2][1].T for c in correct if c[0] == t], axis=0
        )
        incorrect_concat1 = np.concatenate(
            [c[2][0].T for c in incorrect if c[0] == t], axis=0
        )
        incorrect_concat2 = np.concatenate(
            [c[2][1].T for c in incorrect if c[0] == t], axis=0
        )
        data_per_trial = int(cca_window / cca_bin_size)
        num_correct_trials = int(len(correct_concat1) / data_per_trial)
        num_incorrect_trials = int(len(incorrect_concat1) / data_per_trial)
        cca, Xc, Yc = scikit_cca(correct_concat1, correct_concat2)
        corr = find_correlations(Xc, Yc, num_correct_trials, data_per_trial)
        for c in corr:
            correct_corr_info.append([t, c])
        cca, Xi, Yi = scikit_cca(incorrect_concat1, incorrect_concat2)
        corr = find_correlations(Xi, Yi, num_incorrect_trials, data_per_trial)
        for c in corr:
            incorrect_corr_info.append([t, c])

    per_trial_spikes1 = split_spikes_into_trials(
        spike_train1, trial_info["trial_times"], end_time=cca_window
    )
    per_trial_spikes2 = split_spikes_into_trials(
        spike_train2, trial_info["trial_times"], end_time=cca_window
    )
    info = {
        "correct": correct,
        "incorrect": incorrect,
        "correct_rates": correct_rates,
        "incorrect_rates": incorrect_rates,
        "per_trial_corr_info_correct": per_trial_corr_info_correct,
        "per_trial_corr_info_incorrect": per_trial_corr_info_incorrect,
        "concat_correct_corr_info": correct_corr_info,
        "concat_incorrect_corr_info": incorrect_corr_info,
        "unsplit_corr_info_correct": unsplit_corr_info_correct,
        "unsplit_corr_info_incorrect": unsplit_corr_info_incorrect,
        "per_trial_spikes": [per_trial_spikes1, per_trial_spikes2],
        "trial_info": trial_info,
    }
    save_info_to_file(info, recording, out_dir, brain_regions, rel_dir, bit="cca")
    with open(out_dir / f"cca_{regions_as_str}.txt", "w") as f:
        f.write(
            "Finished analysing: "
            + recording.get_name_for_save(rel_dir)
            + f" with {len(correct) / len(r)} correct and {len(incorrect) / len(r)} incorrect trials and {len(unit_table)} units"
        )
    return info
