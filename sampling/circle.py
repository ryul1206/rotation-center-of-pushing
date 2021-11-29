from .base_sampler import BaseSampler
import numpy as np


class CircleSampler(BaseSampler):
    def __init__(self, radius, center=None, **kwargs):
        super().__init__(**kwargs)
        self._radius = radius
        if center is None:
            center = np.array([radius / 2.0, 0])
        self._center = center

    def get_samples(self):
        pass
        # return (
        #     np.random.uniform(low=-self.radius, high=self.radius, size=(n, 2))
        #     + self.center
        # )
