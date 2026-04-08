import torch.nn as nn

class LossFactory:
    @staticmethod
    def get_loss(config_dict):
        loss_type = config_dict.get("loss_type", "mse").lower()
        if loss_type == "mse": return nn.MSELoss()
        elif loss_type == "l1": return nn.L1Loss()
        elif loss_type == "huber": return nn.HuberLoss(delta=config_dict.get("huber_delta", 1.0))
        raise ValueError(f"Unsupported loss: {loss_type}")