import numpy as np

from typing import Callable
from utils.mat_ops import *

class Voxel:
    def __init__( self, y: np.ndarray, initial_vec6: np.ndarray = np.zeros( ( 6, 1 ) ) ):
        
        if( y.shape[0] < 5 ): 
            self.is_empty = True
            return
        
        else: self.is_empty = False

        if( y.shape[1] != 3 or y.ndim != 2 ):
            raise ValueError( f"y must be shape (N, 3), not {y.shape}" )

        self.mu = np.mean( y, axis = 0 )
        self.sigma = 1 / ( y.shape[0] - 1 ) * ( ( y - self.mu ).transpose() @ ( y - self.mu ) )
        self.info_matrix = np.linalg.inv( self.sigma )
        self.determinant = np.linalg.det( self.sigma )
        self.vec6 = initial_vec6

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
        R = get_se3_from_vec6( v_i.vec6, is_in_degrees = False )[:3, :3]
        
        return float( d1( self.determinant ) * np.exp( -( d2() * mu_ij.T @ np.linalg.inv( R.T @ v_i.sigma @ R + self.sigma ) @ mu_ij ) / 2 ).squeeze() )

    def transform( self, delta_vec6: np.ndarray ):
        if( not self.is_empty ):
            se3 = get_se3_from_vec6( delta_vec6, is_in_degrees = False )
            R = se3[:3, :3]
            t = se3[:3, 3:]

            self.mu = R @ self.mu + t.reshape( ( 3, ) )
            self.sigma = R @ self.sigma
            self.info_matrix = np.linalg.inv( self.sigma )
            self.determinant = np.linalg.det( self.sigma )
            self.vec6 += delta_vec6