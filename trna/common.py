import shutil
import pickle
import numpy as np
from pathlib import Path
from simuran import ParamHandler


def save_info_to_file(info, recording, out_dir, regions, rel_dir=None, bit="gpfa"):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    regions_as_str = regions_to_string(regions)
    save_name = out_dir / "pickles" / (f"{name}_{regions_as_str}_{bit}.pkl")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "wb") as f:
        pickle.dump(info, f)


def average_firing_rate(spike_train, trial_length=None):
    """
    Calculate the average firing rate for a spike train.

    Parameters:
    -----------
    spike_train: Union[List, Dict]
        The spike train to calculate the average firing rate for.

    Returns:
    --------
    avg_firing_rates : np.ndarray
        The average firing rate for the spike train.

    """
    average_firing_rates = np.zeros(len(spike_train))
    if isinstance(spike_train, dict):
        to_iter = spike_train.values()
    else:
        to_iter = spike_train
    end_time = 0
    start_time = 10000000000
    if trial_length is None:
        for st in to_iter:
            start_time = min(start_time, st[0])
            end_time = max(end_time, st[-1])
        trial_length = end_time - start_time
    for i, st in enumerate(to_iter):
        average_firing_rates[i] = len(st) / trial_length
    return average_firing_rates


def load_data(recording, out_dir: "Path", regions, rel_dir=None, bit="gpfa"):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    regions_as_str = regions_to_string(regions)
    save_name = out_dir / "pickles" / (f"{name}_{regions_as_str}_{bit}.pkl")
    if save_name.is_file():
        print(
            "Loading pickle data for: "
            + recording.get_name_for_save(rel_dir=rel_dir)
            + regions_as_str
        )
        with open(save_name, "rb") as f:
            info = pickle.load(f)
        return info
    else:
        return "No pickle data found"


def name_from_recording(recording, filename, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    name = name + "--" + filename
    return name


def regions_to_string(brain_regions):
    s = ""
    for r in brain_regions:
        if isinstance(r, str):
            s += r + "_"
        else:
            s += "_".join(r) + "_"
    return s[:-1].replace("/", "-")


def ensure_enough_units(unit_table, min_num_units, brain_region_str):
    has_enough = True
    if len(unit_table) < min_num_units:
        has_enough = False
    brain_regions = unit_table[brain_region_str].unique()
    for region in brain_regions:
        if isinstance(region, str):
            if len(unit_table[unit_table[brain_region_str] == region]) < min_num_units:
                has_enough = False
                break
        else:
            num_units = 0
            for sub_region in region:
                num_units += len(unit_table[unit_table[brain_region_str] == sub_region])
            if num_units < min_num_units:
                has_enough = False
                break
    return has_enough


def decimate_train_to_min(unit_table, spike_train, br_str):
    brain_regions = unit_table[br_str].unique()
    new_spike_train = {}
    new_unit_table = unit_table.copy()
    min_units = 100000
    for region in brain_regions:
        region_units = unit_table[unit_table[br_str] == region]
        min_units = min(min_units, len(region_units))
    for region in brain_regions:
        region_units = unit_table[unit_table[br_str] == region]
        if len(region_units) > min_units:
            random_units = np.random.choice(
                range(len(region_units)), min_units, replace=False
            )
        else:
            random_units = range(len(region_units))
        for i in random_units:
            new_spike_train[region_units.index[i]] = spike_train[region_units.index[i]]
    new_unit_table = new_unit_table.loc[list(new_spike_train.keys())]
    return new_unit_table, new_spike_train


def split_spikes_into_trials(
    spike_train, trial_start_ends, end_time=None, num_trials=None, delay=0
):
    """
    Split the spike train into trials.

    Parameters:
    -----------
    spike_train: dict
        Dictionary of spike trains for each neuron.
    trial_start_ends: List of tuples
        Trial start and end information.
    end_time: float
        End time of each trial.
        If None, uses 1.2 * median of the trial lengths.
    num_trials: int
        Number of trials to use. If None, uses all trials.
    delay: int
        The delay in milliseconds to apply to the spike train.

    Returns:
    --------
    new_spike_train: list
        List of lists of spike trains for each neuron in each trial.

    """
    delay = delay / 1000
    end_time = (
        end_time
        if end_time is not None
        else np.median([t[1] - t[0] for t in trial_start_ends]) * 1.2
    )
    new_spike_train = []
    for i, (start, end) in enumerate(trial_start_ends):
        if (num_trials is not None) and (i == num_trials):
            break
        start_time = start + delay
        end_time_final = start + end_time + delay
        if start_time < 0:
            start_time = start
        trial_spike_train = []
        for k, v in spike_train.items():
            to_use = v[(v >= start_time) & (v <= end_time_final)]
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
    correct = trajectories[np.array(trial_correct).astype(bool)]
    incorrect = trajectories[~np.array(trial_correct).astype(bool)]
    return correct, incorrect


def load_config(config_path=None, config=None):
    """
    Establish the config for the project.

    Parameters:
    -----------
    config_path: str
        The path to the config file. If None, uses the default.
        The default is the config file config.yml in the project.

    Returns:
    --------
    parameters: ParamHandler
        The parameters for the project. Like a dict.

    """
    if config is None:
        config_path = (
            config_path or Path(__file__).parent.parent / "config" / "config.yaml"
        )
        parameters = ParamHandler(source_file=config_path)
    else:
        parameters = ParamHandler(attrs=config)

    try:
        from google.colab import drive

        drive.mount("/content/drive")
        data_directory = Path(parameters["drive_dir"])
    except ModuleNotFoundError:
        data_directory = Path(parameters["local_dir"])

    parameters["allen_data_dir"] = data_directory / parameters["allen_name"]
    parameters["ibl_data_dir"] = data_directory / parameters["ibl_name"]
    parameters["output_dir"] = data_directory / parameters["output_name"]

    return parameters


def write_config(cfg_file, cfg, output_location):
    if cfg_file is not None:
        shutil.copy(cfg_file, output_location)
    else:
        cfg.write(output_location)


def compute_distance(correct_curves, incorrect_curves):
    average_correct = np.mean(correct_curves, axis=0)
    average_incorrect = np.mean(incorrect_curves, axis=0)
    starting_distance = np.linalg.norm(average_correct[:, 0] - average_incorrect[:, 0])
    ending_distance = np.linalg.norm(average_correct[:, -1] - average_incorrect[:, -1])
    return starting_distance, ending_distance


def compute_variance(correct_curves, incorrect_curves, do_mean=True):
    correct_variance = np.var(correct_curves, axis=0)
    incorrect_variance = np.var(incorrect_curves, axis=0)
    if do_mean:
        correct_variance = np.mean(correct_variance)
        incorrect_variance = np.mean(incorrect_variance)
    return correct_variance, incorrect_variance


def scale_data(correct_curves, incorrect_curves, uniform=False):
    new_arr_correct = np.array([x for x in correct_curves])
    new_arr_incorrect = np.array([x for x in incorrect_curves])

    # Step 1, translate to 0
    total_min_x = min(np.min(new_arr_correct[:, 0]), np.min(new_arr_incorrect[:, 0]))
    total_min_y = min(np.min(new_arr_correct[:, 1]), np.min(new_arr_incorrect[:, 1]))
    total_min_z = min(np.min(new_arr_correct[:, 2]), np.min(new_arr_incorrect[:, 2]))

    new_arr_correct[:, 0] = new_arr_correct[:, 0] - total_min_x
    new_arr_correct[:, 1] = new_arr_correct[:, 1] - total_min_y
    new_arr_correct[:, 2] = new_arr_correct[:, 2] - total_min_z
    new_arr_incorrect[:, 0] = new_arr_incorrect[:, 0] - total_min_x
    new_arr_incorrect[:, 1] = new_arr_incorrect[:, 1] - total_min_y
    new_arr_incorrect[:, 2] = new_arr_incorrect[:, 2] - total_min_z

    # Step 2, scale to 1
    max_val_x = max(
        np.max(np.abs(new_arr_correct[:, 0])),
        np.max(np.abs(new_arr_incorrect[:, 0])),
    )
    max_val_y = max(
        np.max(np.abs(new_arr_correct[:, 1])),
        np.max(np.abs(new_arr_incorrect[:, 1])),
    )
    max_val_z = max(
        np.max(np.abs(new_arr_correct[:, 2])),
        np.max(np.abs(new_arr_incorrect[:, 2])),
    )
    if uniform:
        max_val_x = max_val_y = max_val_z = max(max_val_x, max_val_y, max_val_z)
    new_arr_correct[:, 0] = new_arr_correct[:, 0] / max_val_x
    new_arr_correct[:, 1] = new_arr_correct[:, 1] / max_val_y
    new_arr_correct[:, 2] = new_arr_correct[:, 2] / max_val_z
    new_arr_incorrect[:, 0] = new_arr_incorrect[:, 0] / max_val_x
    new_arr_incorrect[:, 1] = new_arr_incorrect[:, 1] / max_val_y
    new_arr_incorrect[:, 2] = new_arr_incorrect[:, 2] / max_val_z

    return new_arr_correct, new_arr_incorrect


def perform_scaling_and_stats(correct, incorrect, mean_only=True):
    distances = compute_distance(correct, incorrect)
    variances = compute_variance(correct, incorrect)
    scaled_correct, scaled_incorrect = scale_data(correct, incorrect)
    scaled_distances = compute_distance(scaled_correct, scaled_incorrect)
    scaled_variances = compute_variance(scaled_correct, scaled_incorrect)
    uniform_scaled_correct, uniform_scaled_incorrect = scale_data(
        correct, incorrect, uniform=True
    )
    uniform_scaled_distances = compute_distance(
        uniform_scaled_correct, uniform_scaled_incorrect
    )
    uniform_scaled_variances = compute_variance(
        uniform_scaled_correct, uniform_scaled_incorrect
    )
    if mean_only:
        distances = np.mean(distances)
        scaled_distances = np.mean(scaled_distances)
        uniform_scaled_distances = np.mean(uniform_scaled_distances)
    return (
        {
            "distances": distances,
            "variances": variances,
            "scaled_distances": scaled_distances,
            "scaled_variances": scaled_variances,
            "uniform_scaled_distances": uniform_scaled_distances,
            "uniform_scaled_variances": uniform_scaled_variances,
        },
        {
            "original": [correct, incorrect],
            "scaled": [scaled_correct, scaled_incorrect],
            "uniform_scaled": [uniform_scaled_correct, uniform_scaled_incorrect],
        },
    )
