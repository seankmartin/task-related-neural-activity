import logging

import numpy as np
import quantities as pq

from trna.dimension_reduction import elephant_gpfa
from trna.common import (
    split_spikes_into_trials,
    split_trajectories,
    ensure_enough_units,
    regions_to_string,
    name_from_recording,
    save_info_to_file,
    decimate_train_to_min,
)

from simuran.bridges.ibl_wide_bridge import IBLWideBridge
from simuran.bridges.allen_vbn_bridge import AllenVBNBridge
from simuran.loaders.allen_loader import BaseAllenLoader

module_logger = logging.getLogger("simuran.custom.gpfa")


def analyse_single_recording(
    recording, gpfa_params, out_dir, base_dir, brain_regions, filter_prop
):
    np.random.seed(42)
    gpfa_window = gpfa_params["gpfa_window"]
    gpfa_binsize = int(gpfa_params["gpfa_binsize"])
    print("Analysing recording: " + recording.get_name_for_save(base_dir))
    rel_dir = base_dir
    is_allen = isinstance(recording.loader, BaseAllenLoader)
    br_str = "structure_acronym" if is_allen else "acronym"
    bridge = (
        AllenVBNBridge(good_unit_properties=filter_prop)
        if is_allen
        else IBLWideBridge(good_unit_properties=filter_prop)
    )
    unit_table, spike_train = bridge.spike_train(recording, brain_regions=brain_regions)
    regions_as_str = regions_to_string(brain_regions)
    trial_info = bridge.trial_info(recording)

    out_dir.mkdir(parents=True, exist_ok=True)
    unit_table_name = name_from_recording(
        recording, f"unit_table_{regions_as_str}.csv", rel_dir
    )
    unit_table_name = "--".join(unit_table_name.split("--")[-2:])
    unit_table.to_csv(out_dir / unit_table_name)
    if not ensure_enough_units(unit_table, 10, br_str):
        module_logger.warning(
            "Not enough units for {} in each brain region".format(
                recording.get_name_for_save()
            )
        )
        save_info_to_file(None, recording, out_dir, brain_regions, rel_dir, bit="gpfa")
        return None
    unit_table, spike_train = decimate_train_to_min(unit_table, spike_train, br_str)
    unit_table_name = name_from_recording(
        recording, f"unit_table_{regions_as_str}_decimated.csv", rel_dir
    )
    unit_table_name = "--".join(unit_table_name.split("--")[-2:])
    unit_table.to_csv(out_dir / unit_table_name)

    per_trial_spikes = split_spikes_into_trials(
        spike_train, trial_info["trial_times"], end_time=gpfa_window
    )
    # binned = []
    # for trial in per_trial_spikes:
    #     binned_spikes = bin_spike_train(trial, 0.02, t_stop=gpfa_window)
    #     binned.append(binned_spikes)
    # binned_spikes = np.array(binned)
    try:
        gpfa_result, trajectories = elephant_gpfa(
            per_trial_spikes, gpfa_window, num_dim=3, bin_size=gpfa_binsize * pq.ms
        )
    # scikit_fa_result, fa_trajectories = scikit_fa(binned_spikes, n_components=3)
    except ValueError:
        module_logger.warning(
            "Not enough spikes for GPFA for {}".format(recording.get_name_for_save())
        )
        return None
    correct, incorrect = split_trajectories(trajectories, trial_info["trial_correct"])
    # fa_correct, fa_incorrect = split_trajectories(
    #     fa_trajectories, trial_info["trial_correct"]
    # )

    info = {
        "elephant": {"correct": correct, "incorrect": incorrect},
        # "scikit_fa": {"correct": fa_correct, "incorrect": fa_incorrect},
    }
    save_info_to_file(info, recording, out_dir, brain_regions, rel_dir, bit="gpfa")
    with open(out_dir / f"gpfa_{regions_as_str}.txt", "w") as f:
        f.write(
            "Finished analysing: "
            + recording.get_name_for_save(rel_dir)
            + f" with {len(correct)} correct and {len(incorrect)} incorrect trials and {len(unit_table)} units"
        )
    return info
