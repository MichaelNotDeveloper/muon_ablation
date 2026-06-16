import torch


class MultiOptimizer(torch.optim.Optimizer):
    """
    Wrapper that steps multiple optimizers together (e.g. Muon for 2D + Adam for rest).
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        param_groups = []
        defaults = {}
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
            defaults.update(optimizer.defaults)
        super().__init__(param_groups, defaults=defaults)

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss

    def state_dict(self):
        return {
            f"optimizer_{idx}": optimizer.state_dict()
            for idx, optimizer in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict):
        for idx, optimizer in enumerate(self.optimizers):
            key = f"optimizer_{idx}"
            if key in state_dict:
                optimizer.load_state_dict(state_dict[key])
