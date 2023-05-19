from abc import abstractmethod


class InverseOptimalControl:

    @abstractmethod
    def loglikelihood(self, x, params):
        pass
