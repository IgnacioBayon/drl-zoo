import torch


def get_device(device_cfg: str) -> torch.device:
    """Return the appropriate torch device based on a config string.

    Args:
        device_cfg: ``"auto"`` selects CUDA when available, otherwise any
            explicit device string (e.g. ``"cpu"``, ``"cuda:1"``) is forwarded
            directly to :class:`torch.device`.
    """
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)
