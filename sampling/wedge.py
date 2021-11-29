from .base_sampler import BaseSampler
import numpy as np


class WedgeSampler(BaseSampler):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self._length = length

    def get_samples(self):
        """
        Return: [[x, y, heading], ...]
        """
        samples = np.zeros((num_samples, 2))