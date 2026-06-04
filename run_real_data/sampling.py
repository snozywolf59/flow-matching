import torch


class SamplingConfig:

    def __init__(self, noise_sampling: str = "gaussian", ode_steps: int = 6):
        self.noise_sampling = noise_sampling
        self.ode_steps = ode_steps

    def sample_noise(self, shape: tuple, sigma: float = 1) -> torch.Tensor:
        if self.noise_sampling == "gaussian":
            return torch.randn(shape) * sigma
        elif self.noise_sampling == "laplace":
            return torch.distributions.Laplace(0, sigma).sample(shape)
        else:
            raise ValueError(
                f"Unknown noise sampling method: {self.noise_sampling}"
            )

    def sample_noise_like(
        self, x: torch.Tensor, sigma: float = 1
    ) -> torch.Tensor:
        return self.sample_noise(x.shape, sigma).to(x.device)

    def __repr__(self) -> str:
        return f"SamplingConfig(noise_sampling={self.noise_sampling}, ode_steps={self.ode_steps})"
