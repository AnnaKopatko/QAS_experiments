import numpy as np


# ======================================================
# Helper functions for safe Aim logging
# ======================================================
def safe_log_metric(run, key, value, step=None, context=None):
    """Safely log metric with error handling"""
    try:
        run.track(float(value), name=key, step=step, context=context or {})
    except Exception as e:
        print(f"⚠️ Failed to log metric {key}={value}: {e}")


def safe_log_metrics(run, metrics, step=None, context=None):
    """Safely log multiple metrics"""
    try:
        for key, value in metrics.items():
            run.track(float(value), name=key, step=step, context=context or {})
    except Exception as e:
        print(f"⚠️ Failed to log metrics: {e}")


# =============================================================================
# Helper: architecture array <-> string key conversion
# These mirror how pymoo stringifies arrays: str(np.array([1., 3., 2.]))
# BUT we use a cleaner dash-separated format for readability.
# =============================================================================

def arch_to_key(arch: np.ndarray) -> str:
    """Convert architecture array to a string cache key.
    
    Example: [1, 3, 2] -> "1-3-2"
    
    Args:
        arch: 1D integer array of block indices
        
    Returns:
        Dash-separated string key
    """
    return "-".join(str(int(x)) for x in arch)


def key_to_arch(key: str, n_layers: int) -> np.ndarray:
    """Convert a string cache key back to an architecture array.
    
    Example: "1-3-2" -> np.array([1, 3, 2])
    
    Args:
        key: Dash-separated string key
        n_layers: Expected number of layers (used for validation)
        
    Returns:
        1D integer array of block indices
    """
    return np.array([int(x) for x in key.split("-")])

