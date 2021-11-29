from abc import *


class Sampler(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, n):
        pass
