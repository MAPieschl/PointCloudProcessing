import numpy as np

from typing import Callable

from ndt.Voxel import Voxel
from utils import mat_ops as mat

class Point:
    def __init__( self, pos: np.ndarray, v: Voxel | None ):

        assert pos.shape == (3, 1), f"x_i must have shape (3, 1), not {pos.shape}"

        self.rel_pos = pos
        self.pos = pos
        self.voxel = v
    
    def __call__( self ) -> np.ndarray:
        return self.pos
    
    def is_contained_by( self ) -> Voxel | None:
        return self.voxel
    
    def move_relative( self, delta_vec6: np.ndarray, v: Callable[[np.ndarray], Voxel | None] ) -> None:

        assert delta_vec6.shape == (6, 1), f"delta_vec6 must be shape (6, 1) of (x, y, z, r_x, r_y, r_z), not {delta_vec6.shape}"

        R = mat.get_dcm( np.rad2deg( delta_vec6[3].squeeze() ), np.rad2deg( delta_vec6[4].squeeze() ), np.rad2deg( delta_vec6[5].squeeze() ) )
        self.pos = R @ self.pos + delta_vec6[:3]

        self.voxel = v( self.pos )

    def set_position( self, vec6: np.ndarray, v: Callable[[np.ndarray], Voxel | None] ) -> None:

        assert vec6.shape == (6, 1), f"vec6 must be shape (6, 1) of (x, y, z, r_x, r_y, r_z), not {vec6.shape}"

        R = mat.get_dcm( np.rad2deg( vec6[3].squeeze() ), np.rad2deg( vec6[4].squeeze() ), np.rad2deg( vec6[5].squeeze() ) )
        self.pos = R @ self.rel_pos + vec6[:3]

        self.voxel = v( self.pos )