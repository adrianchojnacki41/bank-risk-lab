
import numpy as np


def check_count(value, name:str):
    """Check that value is a non-negative integer (or an array of them)."""
    
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be an integer, got {arr.dtype}")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value

def check_probability(value, name: str):
    """Check that value is a number (or an array of numbers) in [0, 1]. """
    
    arr = np.asarray(value)
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(f"{name} must be numeric, got {arr.dtype}")
    if not np.all((arr >= 0) & (arr <= 1)):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value

def check_non_negative(value, name: str):
    """Check that value is a finite non-negative number (e.g. an amount of money)."""
    
    arr = np.asarray(value)
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(f"{name} must be numeric, got {arr.dtype}")
    if not np.all(np.isfinite(arr) & (arr >= 0)):
        raise ValueError(f"{name} must be a finite non-negative number, got {value}")
    return value
