import numpy as np

from copy import deepcopy
from typing import Callable

from ndt.Voxel import Voxel

from utils.mat_ops import *

class LabeledPoint:
    def __init__( self, pos: np.ndarray, label:str, weighted_v: Callable[[np.ndarray, str], list[tuple[Voxel | None, float]]] ):

        assert pos.shape == (3, 1), f"x_i must have shape (3, 1), not {pos.shape}"

        self.pos = pos
        self.label = label
        self.orig_pos = deepcopy( pos )
        self.__weighted_v = weighted_v
    
    def __call__( self ) -> np.ndarray:
        return self.pos
    
    def get_nearest_weighted_voxels( self ) -> list[tuple[Voxel | None, float]]:
        return self.__weighted_v( self.pos, self.label )
    
    def move_relative( self, delta_vec6: np.ndarray ) -> None:

        assert delta_vec6.shape == (6, 1), f"delta_vec6 must be shape (6, 1) of (x, y, z, r_x, r_y, r_z), not {delta_vec6.shape}"

        R = get_dcm( np.rad2deg( delta_vec6[3].squeeze() ), np.rad2deg( delta_vec6[4].squeeze() ), np.rad2deg( delta_vec6[5].squeeze() ) )
        self.pos = R @ self.pos + delta_vec6[:3]

    def set_position( self, vec6: np.ndarray ) -> None:

        assert vec6.shape == (6, 1), f"vec6 must be shape (6, 1) of (x, y, z, r_x, r_y, r_z), not {vec6.shape}"

        R = get_dcm( np.rad2deg( vec6[3].squeeze() ), np.rad2deg( vec6[4].squeeze() ), np.rad2deg( vec6[5].squeeze() ) )
        self.pos = R @ self.orig_pos + vec6[:3]