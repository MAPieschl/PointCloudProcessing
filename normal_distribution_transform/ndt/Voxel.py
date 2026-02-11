import numpy as np

from typing import Callable

class Voxel:
    def __init__( self, y: np.ndarray ):
        
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

    def get_score( self, x_i: np.ndarray, d1: Callable[[float], float], d2: Callable[[], float] ) -> float:

        if( self.is_empty ): return 0.0

        if( x_i.shape != (3, 1) ):
            raise ValueError( f"x_i must be shape (3, 1), not {x_i.shape}" )

        return float( d1( self.determinant ) * np.exp( -d2() * ( ( x_i - self.mu.reshape( ( 3, 1 ) ) ).transpose() @ self.info_matrix @ ( x_i - self.mu.reshape( ( 3, 1 ) ) ) ) / 2 ).squeeze() )