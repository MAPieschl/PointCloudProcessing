import numpy as np

from ndt.Voxel import VoxelGrid, Voxel

class ReferencePointCloud:
    def __init__( self, y: np.ndarray, voxel_size: float = 2.0, voxel_conditioning_epsilon: float = 1e-3 ):

        assert y.ndim == 2, f"y must be an (n, 3) vector of points, not {y.shape}"
        assert y.shape[1] == 3, f"y must be an (n, 3) vector of points, not {y.shape}"

        self.voxel_grid = VoxelGrid( y, voxel_size, voxel_conditioning_epsilon )
    
    def get_voxel( self, pt: np.ndarray ) -> Voxel | None:
        return self.voxel_grid.get_voxel_containing( pt )
    
    def get_pc_list( self ) -> list[np.ndarray]:
        return self.voxel_grid.get_list_of_points()