class FrictionModel:
    def __init__(self, zeta=0.3, decay=0.01, min_theta=0.35, max_theta=0.85, adapt_rate=0.02):
        self.zeta = zeta
        self.decay = decay
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.adapt_rate = adapt_rate

    def adapt_theta(self, theta, bound_fraction):
        # Simple feedback: if too many bound, raise threshold; if too few, lower.
        target = 0.33  # desired bound fraction
        delta = (target - bound_fraction)
        theta += -self.adapt_rate * delta * (1.0 + self.zeta)
        theta = max(self.min_theta, min(self.max_theta, theta))
        return theta
