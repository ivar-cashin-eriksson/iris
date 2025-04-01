import torch

def get_device(device_str: str) -> torch.device:
    """Convert a string to a torch.device.
    
    Args:
        device_str: String representing the device ('cuda', 'cpu', or 'mps')
        
    Returns:
        torch.device: The appropriate device
        
    Raises:
        ValueError: If device_str is not one of the supported devices
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Device {device_str} is not available")