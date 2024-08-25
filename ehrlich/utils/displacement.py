from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from multipledispatch import dispatch


class Displacement(ABC):

    @abstractmethod
    def displace(self, coords):
        ...


class ParallelTranslation(Displacement):

    @dispatch(float, np.ndarray)
    def __init__(self, k: Union[float, int], translation: np.ndarray):
        self.k = k
        self.translation = translation

    @dispatch(np.ndarray)
    def __init__(self, translation):
        self.__init__(1., translation)

    def displace(self, coords):
        if self.k == 1:
            return coords + self.translation
        return self.k * coords + self.translation


class Rotation(Displacement):

    def __init__(self, rotation_matrix, second_place=False):
        self.rotation_matrix = rotation_matrix
        self.second_place = second_place

    def displace(self, coords):
        if self.second_place:
            return np.dot(self.rotation_matrix, coords)
        return np.dot(coords, self.rotation_matrix)


class Transpose(Displacement):
    def displace(self, coords):
        return coords.T
