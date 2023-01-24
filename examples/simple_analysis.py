import numpy as np
from skm_pyutils.table import list_to_df

def extract_useful_allen(recording):
    def filter_units(unit_channels):
        unit_channels = unit_channels.sort_values(
            "probe_vertical_position", ascending=False
        )
        good_unit_filter = (
            (unit_channels["snr"] > 1)
            & (unit_channels["isi_violations"] < 1)
            & (unit_channels["firing_rate"] > 0.01)
            & (unit_channels["quality"])
        )
        return unit_channels[good_unit_filter]

    session = recording.data
    units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels, left_on="peak_channel_id", right_index=True)
    unit_channels.to_csv("units.csv")

    stimulus_presentations = session.stimulus_presentations
    change_times = stimulus_presentations[
        stimulus_presentations["active"] & stimulus_presentations["is_change"]
    ]["start_time"].values

    good_units = filter_units(unit_channels)

    return change_times, good_units, session.spike_times

def allen_to_general_bridge(recording):
    change_times, good_units, spike_times = extract_useful_allen(recording)


def one_to_general_bridge(recording):
    pass


def smooth_spike_train():
    pass


def simple_behaviour_compare(spike_trains, passes, trial_times, spike_train_rate):
    """
    Compare matrix of spike times in a window around trials to compare pass and fail.
    
    """
    result_list = []

    for p, t in zip(passes, trial_times):
        start_time, end_time = t
        start_time = start_time * spike_train_rate
        end_time = end_time * spike_train_rate
        piece = spike_trains[start_time:end_time]
        firing_rate_in_window = np.mean(piece)
        result_list.append([p, firing_rate_in_window])
    
    result_df = list_to_df(result_list, ["Passed", "Average value"])

    return result_df




    
