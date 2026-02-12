import torch
import copy

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        
        # Create a deep copy of the model for EMA
        self.ema_model = copy.deepcopy(model)
        
        # Ensure EMA model assumes eval mode (for inference)
        self.ema_model.eval()
        
        # Detach parameters so gradients aren't tracked
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA weights and copy buffers.
        """
        # 1. Update Parameters (with decay)
        # Note: We loop via zip to avoid creating large intermediate dicts
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            # ema_param = decay * ema_param + (1 - decay) * model_param
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=(1 - self.decay)
            )

        # 2. Update Buffers (exact copy)
        # We copy buffers (e.g., BatchNorm stats) directly from source
        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.data.copy_(model_buffer.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)

    # Helper method to easily access the model for inference
    def get_model(self):
        return self.ema_model