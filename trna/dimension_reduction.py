"""Dimension reduction functions"""
import quantities as pq


def elephant_gpfa(per_trial_spike_train, trial_length, bin_size=20 * pq.ms, num_dim=3):
    """
    Perform GPFA on the spike trains.

    Parameters:
    -----------
    per_trial_spike_train: list
        List of lists of spike trains for each neuron in each trial.
    trial_length: float
        Length of each trial.
    bin_size: quantities.Quantity
        Size of the bins to use.
    num_dim: int
        Number of dimensions to use.

    Returns:
    --------
    gpfa_ndim: elephant.gpfa.GPFA
        The fitted GPFA object.
    trajectories: np.ndarray
        The trajectories of the neurons with GPFA applied.

    """
    from simuran.bridges.neo_bridge import convert_spikes_to_train
    from elephant.gpfa import GPFA

    neo_trains = [
        convert_spikes_to_train(small_train, custom_t_stop=trial_length)
        for small_train in per_trial_spike_train
    ]

    gpfa_ndim = GPFA(x_dim=num_dim, bin_size=bin_size)

    # Axis 0 is trials
    # Axis 1 is neurons
    # Axis 2 is the times
    trajectories = gpfa_ndim.fit_transform(neo_trains)

    return gpfa_ndim, trajectories


def scikit_cca(trial_rates1, trial_rates2, n_components=1):
    """
    Perform CCA on the spike trains.

    Parameters:
    -----------
    trial_rates1: np.ndarray
        The spike rates for the first set of neurons.
    trial_rates2: np.ndarray
        The spike rates for the second set of neurons.
    n_components: int
        Number of dimensions to use.

    Returns:
    --------
    cca: sklearn.cross_decomposition.CCA
        The fitted CCA object.
    X: np.ndarray
        The transformed data for the first set of neurons.
    Y: np.ndarray
        The transformed data for the second set of neurons.

    """
    from sklearn.cross_decomposition import CCA

    cca = CCA(n_components=n_components)
    X, Y = cca.fit_transform(trial_rates1, trial_rates2)

    return cca, X, Y
