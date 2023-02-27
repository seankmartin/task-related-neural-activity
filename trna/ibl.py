import simuran as smr
from simuran.loaders.one_loader import OneAlyxLoader

def ibl_load(ibl_cache_dir):
    """
    Load the IBL data.
    
    Parameters
    ----------
    ibl_cache_dir : str
        The directory to cache the data in.
    
    Returns
    -------
    one_recording_container : RecordingContainer
        The recording container for the IBL data.
    one_loader : OneAlyxLoader
        The loader for the IBL data.
    
    """
    one_loader = OneAlyxLoader(cache_directory=ibl_cache_dir)
    one_sessions = one_loader.get_sessions_table()
    one_recording_container = smr.RecordingContainer.from_table(one_sessions, one_loader)

    return one_recording_container, one_loader