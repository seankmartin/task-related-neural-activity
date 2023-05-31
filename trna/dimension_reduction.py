"""Dimension reduction functions"""
import numpy as np
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
    from simuran.bridges.neo_bridge import NeoBridge
    from elephant.gpfa import GPFA

    neo_trains = [
        NeoBridge.convert_spikes(small_train, custom_t_stop=trial_length)
        for small_train in per_trial_spike_train
    ]

    gpfa_ndim = GPFA(x_dim=num_dim, bin_size=bin_size)

    # Axis 0 is trials
    # Axis 1 is neurons
    # Axis 2 is the times
    trajectories = gpfa_ndim.fit_transform(neo_trains)

    return gpfa_ndim, trajectories


def scikit_fa(trial_rates, n_components=3):
    """
    Perform FA on the spike trains.

    Parameters:
    -----------
    trial_rates: np.ndarray
        The spike rates for the neurons.
    n_components: int
        Number of dimensions to use.

    Returns:
    --------
    fa: sklearn.decomposition.FactorAnalysis
        The fitted FA object.
    X: np.ndarray
        The transformed data.

    """
    from sklearn.decomposition import FactorAnalysis

    fa = FactorAnalysis(n_components=n_components, random_state=0)
    X = fa.fit_transform(trial_rates)

    return fa, X


def scikit_cca(trial_rates1, trial_rates2, cca=None, fit=True):
    """
    Perform CCA on the spike trains.

    Parameters:
    -----------
    trial_rates1: np.ndarray
        The spike rates for the first set of neurons.
    trial_rates2: np.ndarray
        The spike rates for the second set of neurons.
    cca: sklearn.cross_decomposition.CCA
        The CCA object to use.
    fit: bool
        Whether to fit the CCA object.

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

    if cca is None:
        cca = CCA(n_components=1)
    if fit:
        cca.fit(trial_rates1, trial_rates2)
    X, Y = cca.transform(trial_rates1, trial_rates2)
    X = X.squeeze()
    Y = Y.squeeze()

    return cca, X, Y


def find_correlation(X, Y):
    if len(X.shape) == 1:
        return np.corrcoef(X, Y)[0, 1]
    else:
        return np.array(
            [np.corrcoef(X[:, v], Y[:, v])[0, 1] for v in range(X.shape[1])]
        )


def manual_cca_transform(X_, Y_, cca):
    rotation_x = cca.x_rotations_
    rotation_y = cca.y_rotations_

    scale_x = X_.std(axis=0, ddof=1)
    scale_y = Y_.std(axis=0, ddof=1)
    scale_x[scale_x == 0.0] = 1
    scale_y[scale_y == 0.0] = 1
    scaled_X = (X_ - X_.mean(axis=0)) / scale_x
    scaled_Y = (Y_ - Y_.mean(axis=0)) / scale_y

    Xt = np.dot(scaled_X, rotation_x)
    Yt = np.dot(scaled_Y, rotation_y)

    return Xt.squeeze(), Yt.squeeze()


def find_correlations(X, Y, num_trials, data_per_trial):
    correlations = []
    for i in range(num_trials):
        start = i * data_per_trial
        end = start + data_per_trial
        correlations.append(find_correlation(X[start:end], Y[start:end]))
    return correlations
