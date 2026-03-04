import numpy as np

from ndt.Parameters import Parameters
from typing import Callable
from utils.mat_ops import *

class Voxel:
    def __init__( self, y: np.ndarray, conditioning_epsilon: float ):
        
        if( y.shape[0] < 5 ): 
            self.is_empty = True
            return
        
        else: self.is_empty = False

        if( y.shape[1] != 3 or y.ndim != 2 ):
            raise ValueError( f"y must be shape (N, 3), not {y.shape}" )

        self.__y = y ## self.__y should be const! .transform always applies a transformation based on the original pose
        self.__epsilon = conditioning_epsilon

        self.mu = np.mean( y, axis = 0 )
        self.sigma = 1 / ( y.shape[0] - 1 ) * ( ( y - self.mu ).transpose() @ ( y - self.mu ) ) + np.eye( 3 ) * self.__epsilon
        self.info_matrix = np.linalg.inv( self.sigma )
        self.determinant = np.linalg.det( self.sigma )
        self.se3 = np.eye( 4, 4 )

    def get_score_P2D( self, x_i: np.ndarray, d1: Callable[[float], float], d2: Callable[[], float] ) -> float:

        if( self.is_empty ): return 0.0

        if( x_i.shape != (3, 1) ):
            raise ValueError( f"x_i must be shape (3, 1), not {x_i.shape}" )

        return float( d1( self.determinant ) * np.exp( -d2() * ( ( x_i - self.mu.reshape( ( 3, 1 ) ) ).transpose() @ self.info_matrix @ ( x_i - self.mu.reshape( ( 3, 1 ) ) ) ) / 2 ).squeeze() )

    def get_score_D2D( self, v_i, d1: Callable[[float], float], d2: Callable[[float], float] ) -> float:
        '''
        The reference voxel should be called with the target voxel as parameter to match Magnusson's implementation.
        '''

        if( self.is_empty ): return 0.0

        if( v_i.is_empty ): return 0.0

        # mu_ij = R @ mu_i + t_i - mu_j --> the equation below assumes the rotatation has already been applied to v_i
        mu_ij = ( v_i.mu - self.mu ).reshape( ( 3, 1 ) )

        M = self.se3[:3, :3].T @ v_i.sigma @ self.se3[:3, :3] + self.sigma
        if( not np.all( np.linalg.eigvals( M ) > 1e-6 ) ):
            print( np.linalg.eigvals( M ) )
        
        return float( d1( self.determinant ) * np.exp( -( d2( 1 ) * mu_ij.T @ np.linalg.inv( M ) @ mu_ij ) / 2 ).squeeze() )

    def get_points( self ) -> np.ndarray:
        if( self.is_empty ):    return np.array( [] )
        else:                   return self.__y

    def transform( self, p: Parameters ):
        if( not self.is_empty ):

            self.se3 = p.se3

            R = self.se3[:3, :3]
            t = self.se3[:3, 3:]

            new_y = ( R @ self.__y.T + t ).T

            self.mu = np.mean( new_y, axis = 0 )
            self.sigma = 1 / ( new_y.shape[0] - 1 ) * ( ( new_y - self.mu ).transpose() @ ( new_y - self.mu ) ) + np.eye( 3 ) * self.__epsilon
            self.info_matrix = np.linalg.inv( self.sigma )
            self.determinant = np.linalg.det( self.sigma )

class VoxelGrid:
    def __init__( self, pts: np.ndarray, voxel_size: float, conditioning_eps: float = 1e-3 ):
        self.__validate_points( pts )

        self.__voxel_size = voxel_size
        self.__eps = conditioning_eps
        self.__pts = pts
        self.__voxels: dict[tuple[int, int, int], Voxel] = {}

        self.build_voxel_grid( self.__voxel_size )

    def get_voxel_containing( self, pt: np.ndarray ) -> Voxel:
        self.__validate_point( pt )
        return self.__voxels[ self.__get_voxel_idx( pt ) ]
    
    def get_list_of_points( self ) -> list[np.ndarray]:
        list_of_points = []
        for idx, vox in self.__voxels.items():
            list_of_points.append( vox.get_points() )

        return list_of_points
    
    def get_list_of_voxel_means( self ) -> list[np.ndarray]:
        list_of_means = []
        for idx, vox in self.__voxels.items():
            list_of_means.append( vox.mu )
        
        return list_of_means
    
    def get_voxel_size( self ):
        return self.__voxel_size
    
    def get_weighted_8_nearest_voxels( self, pt: np.ndarray ) -> list[tuple[Voxel | None, float]]:
        self.__validate_point( pt )

        pt = pt.flatten()

        vox_idx_half_shift = ( pt - ( self.__voxel_size / 2 ) ) / self.__voxel_size
        base_idx = np.floor( vox_idx_half_shift ).astype( int )
        frac = vox_idx_half_shift - base_idx

        weighted_voxels: list[tuple[Voxel | None, float]] = []

        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):

                    wx = frac[0] if dx == 1 else ( 1 - frac[0] )
                    wy = frac[1] if dy == 1 else ( 1 - frac[1] )
                    wz = frac[2] if dz == 1 else ( 1 - frac[2] )

                    weight = float( wx * wy * wz )

                    if( weight > 0 ):
                        query_idx: tuple[int, int, int] = ( int( base_idx[0] + dx), int( base_idx[1] + dy ), int( base_idx[2] + dz ) )

                        if( query_idx in self.__voxels.keys() ):    weighted_voxels.append( ( self.__voxels[query_idx], weight ) )
                        else:                                       weighted_voxels.append( ( None, weight ) )

        return weighted_voxels
    
    def build_voxel_grid( self, voxel_size: float ):
        self.__voxel_size = voxel_size
        self.__voxels.clear()
        
        point_bins = {}
        
        for pt in self.__pts:
            pt = pt.flatten()

            idx = self.__get_voxel_idx( pt )

            if( idx not in point_bins ):
                point_bins[idx] = []

            point_bins[idx].append( pt )

        for idx, pt_l in point_bins.items():
            self.__voxels[idx] = Voxel( np.array( pt_l ), self.__eps )
    
    def __get_voxel_idx( self, pt: np.ndarray ) -> tuple[int, int, int]:
        self.__validate_point( pt )

        pt = pt.flatten()

        ix = int( np.floor( pt[0] / self.__voxel_size ) )
        iy = int( np.floor( pt[1] / self.__voxel_size ) )
        iz = int( np.floor( pt[2] / self.__voxel_size ) )
        
        return ( ix, iy, iz )

    def __validate_point( self, pt: np.ndarray ) -> None:

        assert pt.shape == ( 3, ) or pt.shape == ( 3, 1 ), f'pt must be shape ( 3, ) or ( 3, 1 ), not {pt.shape}'

    def __validate_points( self, pts: np.ndarray ) -> None:

        assert pts.ndim == 2, f'pts must be shape ( N, 3 ), not {pts.shape}'
        assert pts.shape[1] == 3, f'pts must be shape ( N, 3 ), not {pts.shape}'