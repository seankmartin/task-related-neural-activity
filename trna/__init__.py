"""Package to analyse neural activity related to a task."""

from trna.dimension_reduction import elephant_gpfa, scikit_cca
from trna.plot import simple_trajectory_plot
from trna.common import split_spikes_into_trials, split_trajectories
