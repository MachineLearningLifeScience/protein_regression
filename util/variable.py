import torch

class Variable:
    def __init__(self, v, lower, upper):
        self.unconstrained = self.inverse(v, lower, upper)
        # TODO: make sure unconstrained requires grad
        self.lower = lower
        self.upper = upper

    def get_unconstrained(self):
        return self.unconstrained

    def get_value(self):
        return self.constrain(self.unconstrained, self.lower, self.upper)

    @staticmethod
    def inverse(val, lower, upper):
        inverse = -torch.log((upper - lower) / (val - lower) - 1)
        inverse.type(torch.float64)
        inverse.requires_grad_(True)
        return inverse

    @staticmethod
    def constrain(val, lower, upper):
        """
        constrain through σ function
        """
        constrained = lower + (upper - lower) * (1 / (1 + torch.exp(-val)))
        constrained.type(torch.float64)
        constrained.requires_grad_(True)
        return constrained

