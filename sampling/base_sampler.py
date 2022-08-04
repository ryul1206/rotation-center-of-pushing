from abc import *


class BaseSampler(metaclass=ABCMeta):
    @abstractmethod
    def get_samples(self):
        """
        Return:
            - next_poses [[x, y, heading], ...]
            - ICRs [[x, y], ...]
        """
        pass
