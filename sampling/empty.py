from .base_sampler import BaseSampler
import numpy as np


class EmptySampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_samples(self):
        return np.array([])
