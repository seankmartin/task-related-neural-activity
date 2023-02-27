import numpy as np
from skm_pyutils.table import list_to_df


def simple_behaviour_compare(spike_train, passes, trial_times):
    """
    Compare matrix of spike times in a window around trials to compare pass and fail.

    """
    result_list = []

    for p, t in zip(passes, trial_times):
        start_time, end_time = t
        for k, v in spike_train.items():
            part_of_interest = np.nonzero(
                np.logical_and((v >= start_time), (v < end_time))
            )[0]
            piece = v[part_of_interest]
            firing_rate_in_window = len(piece) / (end_time - start_time)
            result_list.append([p, k, firing_rate_in_window])

    result_df = list_to_df(result_list, ["Passed", "Unit", "Num spikes"])

    return result_df.groupby(["Passed", "Unit"]).mean()
