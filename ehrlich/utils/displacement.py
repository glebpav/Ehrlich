from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from multipledispatch import dispatch


class Displacement(ABC):

    """
    Abstract base class for all types displacements that could be done this
    coordinates matrix witch shape is (N, 3)

    Subclasses must implement the `displace` method, which defines the specific
    transformation to be applied to the input coordinates.
    """

    @abstractmethod
    def displace(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply a displacement/transformation to the input coordinates.
        :param coords: (np.ndarray): A 2D numpy array of shape (N, 3) representing the coordinates of N points in 3D space.
        :return: np.ndarray: A 2D numpy array of shape (N, 3) representing the transformed coordinates.
        """
        ...


class ParallelTranslation(Displacement):

    """
    Applies a parallel translation to a set of 3D coordinates. The translation
    can be scaled by a factor `k`.

    There are two ways to initialize this class:
    1. By specifying a scaling factor `k` and a translation vector.
    2. By specifying just the translation vector (in this case, `k` defaults to 1.0).
    """

    @dispatch(float, np.ndarray)
    def __init__(self, k: Union[float], translation: np.ndarray):
        self.k = k
        self.translation = translation

    @dispatch(np.ndarray)
    def __init__(self, translation: np.ndarray):
        self.__init__(1., translation)

    def displace(self, coords: np.ndarray) -> np.ndarray:
        if self.k == 1:
            return coords + self.translation
        return self.k * coords + self.translation


class Rotation(Displacement):

    """
    Applies a rotation to a set of 3D coordinates using a specified rotation matrix.

    The rotation can be applied in two ways:
    1. If `second_place` is False (default), the rotation is applied as: `coords @ rotation_matrix`.
    2. If `second_place` is True, the rotation is applied as: `rotation_matrix @ coords`.
    """

    def __init__(self, rotation_matrix: np.ndarray, second_place: bool = False):
        self.rotation_matrix = rotation_matrix
        self.second_place = second_place

    def displace(self, coords: np.ndarray) -> np.ndarray:
        if self.second_place:
            return np.dot(self.rotation_matrix, coords)
        return np.dot(coords, self.rotation_matrix)


class Transpose(Displacement):

    """
    Applies a transpose operation to the coordinates matrix.
    """

    def displace(self, coords: np.ndarray) -> np.ndarray:
        return coords.T
