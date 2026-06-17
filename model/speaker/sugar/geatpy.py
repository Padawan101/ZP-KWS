"""
Mock geatpy module to avoid dependency issues
geatpy is used for genetic algorithm-based NAS but not needed for inference
"""

# Mock Problem class
class Problem:
    def __init__(self, *args, **kwargs):
        pass

# Mock Algorithm class
class Algorithm:
    def __init__(self, *args, **kwargs):
        pass

# Mock other commonly used components
def crtfld(*args, **kwargs):
    """Create field (used for NAS search space)"""
    pass
