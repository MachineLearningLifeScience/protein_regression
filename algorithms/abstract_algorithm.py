from numpy import ndarray


class AbstractAlgorithm:
    def get_name(self):
        raise NotImplementedError("abstract method")

    def predict(self, X: ndarray):
        raise NotImplementedError("abstract method")

    def predict_f(self, X: ndarray):
        raise NotImplementedError("abstract method")

    def train(self, X: ndarray, Y: ndarray):
        """

        :param X: NxD array
        :param Y: Nx1 array
        :return:
        """
        raise NotImplementedError("abstract method")
