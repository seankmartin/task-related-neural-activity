import simuran as smr
from simuran.loaders.allen_loader import AllenVisualBehaviorLoader

def load_allen(allen_cache_dir, allen_manifest):
    """
    Load the Allen data.
    
    Parameters
    ----------
    allen_cache_dir : str
        The directory to cache the data in.
    allen_manifest : str
        The path to the manifest file.
    
    Returns
    -------
    allen_recording_container : RecordingContainer
        The recording container for the Allen data.
    allen_loader : AllenVisualBehaviorLoader
        The loader for the Allen data.
    
    """
    allen_loader = AllenVisualBehaviorLoader(
        cache_directory=allen_cache_dir, manifest=allen_manifest
    )
    allen_sessions = allen_loader.get_sessions_table()
    allen_recording_container = smr.RecordingContainer.from_table(allen_sessions, allen_loader)

    return allen_recording_container, allen_loader