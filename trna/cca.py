import logging
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


def analyse_single_recording(recording, gpfa_window, out_dir, base_dir, brain_regions):
    print("Analysing recording: " + recording.get_name_for_save(base_dir))
    rel_dir = base_dir
    is_allen = isinstance(recording.loader, BaseAllenLoader)
    br_str = "structure_acronym" if is_allen else "acronym"
    bridge = AllenVBNBridge() if is_allen else IBLWideBridge()
    region1, region2 = brain_regions
    unit_table1, spike_train1 = bridge.spike_train(recording, brain_regions=[region1])
    unit_table2, spike_train2 = bridge.spike_train(recording, brain_regions=[region2])
    unit_table = pd.concat([unit_table1, unit_table2])
    if not ensure_enough_units(unit_table, 15, br_str):
        module_logger.warning(
            "Not enough units for {} in each brain region".format(
                recording.get_name_for_save()
            )
        )
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
            cca, X, Y = scikit_cca(binned_spikes1.T, binned_spikes2.T)
            print(X.shape, Y.shape)
            if correct_:
                correct.append([t, [X, Y]])
            else:
                incorrect.append([t, [X, Y]])

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
