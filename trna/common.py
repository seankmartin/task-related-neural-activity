import numpy as np


def split_spikes_into_trials(spike_train, trial_info, end_time=None, num_trials=None):
    """
    Split the spike train into trials.

    Parameters:
    -----------
    spike_train: dict
        Dictionary of spike trains for each neuron.
    trial_info: List of tuples
        Trial start and end information.
    end_time: float
        End time of each trial.
        If None, uses 1.2 * median of the trial lengths.
    num_trials: int
        Number of trials to use. If None, uses all trials.

    Returns:
    --------
    new_spike_train: list
        List of lists of spike trains for each neuron in each trial.

    """
    end_time = (
        end_time
        if end_time is not None
        else np.median([t[1] - t[0] for t in trial_info]) * 1.2
    )
    new_spike_train = []
    for i, (start, end) in enumerate(trial_info):
        if (num_trials is not None) and (i == num_trials):
            break
        trial_spike_train = []
        for k, v in spike_train.items():
            to_use = v[(v >= start) & (v <= start + end_time)]
            trial_spike_train.append(to_use - start)
        new_spike_train.append(trial_spike_train)
    return new_spike_train


def split_trajectories(trajectories, trial_correct):
    """
    Split the trajectories into correct and incorrect trials.

    Parameters:
    -----------
    trajectories: np.ndarray
        The trajectories of the neurons with GPFA applied.
    trial_correct: np.ndarray
        Whether each trial was correct or not.

    Returns:
    --------
    correct: np.ndarray
        The trajectories of the neurons with GPFA applied for correct trials.
    incorrect: np.ndarray
        The trajectories of the neurons with GPFA applied for incorrect trials.

    """
    correct = trajectories[trial_correct == 1]
    incorrect = trajectories[trial_correct == 0]
    return correct, incorrect
