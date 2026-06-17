"""
Mock thop module to avoid dependency issues
thop is used for computing model parameters/FLOPs but not needed for inference
"""

class BasicHooks:
    """Mock basic hooks for thop.vision"""
    @staticmethod
    def count_relu(*args, **kwargs):
        pass

    @staticmethod
    def count_convNd(*args, **kwargs):
        pass

    @staticmethod
    def count_linear(*args, **kwargs):
        pass

    @staticmethod
    def count_bn(*args, **kwargs):
        pass

    @staticmethod
    def count_avgpool(*args, **kwargs):
        pass

    @staticmethod
    def zero_ops(*args, **kwargs):
        pass

class Vision:
    """Mock vision module for thop"""
    basic_hooks = BasicHooks()

# Create vision attribute
vision = Vision()

def profile(model, inputs, verbose=False):
    """
    Mock implementation of thop.profile
    Returns dummy values since we only need it for model definition, not actual profiling
    """
    # Return dummy MACs and parameters
    macs = 0
    params = sum(p.numel() for p in model.parameters())
    return macs, params

def clever_format(nums, format="%.2f"):
    """
    Mock implementation of thop.clever_format
    """
    return [f"{num:,}" for num in nums]
