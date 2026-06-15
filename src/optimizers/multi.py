class Multioptimizers:
    """
    Wrapper that steps multiple optimizers together (e.g. Muon for 2D + Adam for rest).
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

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
