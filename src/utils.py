import torch


def get_device(device_cfg: str) -> torch.device:
    """Return the appropriate torch device based on a config string."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def get_loss_fn(loss_fn_cfg: str):
    """Return the appropriate loss function based on a config string."""
    if loss_fn_cfg == "mse":
        return torch.nn.MSELoss()
    elif loss_fn_cfg == "huber":
        return torch.nn.SmoothL1Loss()
    else:
        raise ValueError(
            f"Unknown loss function '{loss_fn_cfg}'. Choose from: mse, huber"
        )
