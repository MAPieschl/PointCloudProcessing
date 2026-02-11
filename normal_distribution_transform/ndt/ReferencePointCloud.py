import numpy as np

from ndt.Voxel import Voxel

class ReferencePointCloud:
    def __init__( self, y: np.ndarray ):

        assert y.ndim == 2, f"y must be an (n, 3) vector of points, not {y.shape}"
        assert y.shape[1] == 3, f"y must be an (n, 3) vector of points, not {y.shape}"

        # For ease, divide into 8 voxels
        self._max = np.array( [ np.max( y[:, 0] ), np.max( y[:, 1] ), np.max( y[:, 2] ) ] )
        self._min = np.array( [ np.min( y[:, 0] ), np.min( y[:, 1] ), np.min( y[:, 2] ) ] )
        self._mid = np.array( ( self._max + self._min ) / 2 )

        y_div: list[list[np.ndarray]] = [[], [], [], [], [], [], [], []]

        for pt in y:
            idx: int | None = self.get_voxel_idx( pt )
            if( type( idx ) == int ):   y_div[idx].append( pt )
            else:
                raise ValueError( "Reference cloud voxelization failed." )
        
        self.voxels: list[Voxel] = []
        self.y: list[np.ndarray] = []
        for div in y_div:
            self.y.append( np.array( div ) )
            self.voxels.append( Voxel( np.array( div ) ) )

    def get_voxel_list( self ) -> list[Voxel]:
        return self.voxels
    
    def get_voxel( self, pt: np.ndarray ) -> Voxel | None:

        assert pt.shape == (3,) or pt.shape == (3, 1), f"pt shape must be (3,) or (3, 1), not {pt.shape}"

        idx = self.get_voxel_idx( pt )

        return self.voxels[idx] if type( idx ) == int else None
    
    def get_idx_of( self, voxel: Voxel ) -> int | None:
        try:
            return self.voxels.index( voxel )
        except ValueError:
            return None
    
    def get_pc_list( self ) -> list[np.ndarray]:
        return self.y
    
    def get_pc( self, idx: int ):
        return self.y[idx]
    
    def get_voxel_idx( self, pt: np.ndarray ) -> int | None:

        assert pt.shape == (3,) or pt.shape == (3, 1), f"pt shape must be (3,) or (3, 1), not {pt.shape}"

        if( pt[0] <= self._mid[0] and pt[0] >= self._min[0] and pt[1] <= self._mid[1] and pt[1] >= self._min[1] and pt[2] <= self._mid[2] and pt[2] >= self._min[2] ):
            return 0

        elif( pt[0] <= self._mid[0] and pt[0] >= self._min[0] and pt[1] <= self._mid[1] and pt[1] >= self._min[1] and pt[2] > self._mid[2] and pt[2] <= self._max[2]):
            return 1

        elif( pt[0] <= self._mid[0] and pt[0] >= self._min[0] and pt[1] > self._mid[1] and pt[1] <= self._max[1] and pt[2] <= self._mid[2] and pt[2] >= self._min[2]):
            return 2

        elif( pt[0] <= self._mid[0] and pt[0] >= self._min[0] and pt[1] > self._mid[1] and pt[1] <= self._max[1] and pt[2] > self._mid[2] and pt[2] <= self._max[2]):
            return 3

        elif( pt[0] > self._mid[0] and pt[0] <= self._max[0] and pt[1] <= self._mid[1] and pt[1] >= self._min[1] and pt[2] <= self._mid[2] and pt[2] >= self._min[2]):
            return 4

        elif( pt[0] > self._mid[0] and pt[0] <= self._max[0] and pt[1] <= self._mid[1] and pt[1] >= self._min[1] and pt[2] > self._mid[2] and pt[2] <= self._max[2]):
            return 5

        elif( pt[0] > self._mid[0] and pt[0] <= self._max[0] and pt[1] > self._mid[1] and pt[1] <= self._max[1] and pt[2] <= self._mid[2] and pt[2] >= self._min[2]):
            return 6

        elif( pt[0] > self._mid[0] and pt[0] <= self._max[0] and pt[1] > self._mid[1] and pt[1] <= self._max[1] and pt[2] > self._mid[2] and pt[2] <= self._max[2]):
            return 7
        
        else:
            return None