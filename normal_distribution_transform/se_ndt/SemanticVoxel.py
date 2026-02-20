import numpy as np

from typing import Callable

class SemanticVoxel:
    def __init__( self, y: np.ndarray, labels: list[str] ):

        if( y.shape[1] != 3 or y.ndim != 2 ):
            raise ValueError( f"y must be shape (N, 3), not {y.shape}" )

        # Separate by label
        labels_np = np.array( labels, dtype = str )

        self.mu: dict[str, np.ndarray] = {}
        self.sigma: dict[str, np.ndarray] = {}
        self.info_matrix: dict[str, np.ndarray] = {}
        self.determinant: dict[str, float] = {}
        self.is_empty: dict[str, bool] = {}

        print( 'Initializing SemanticVoxel...' )
        for l in labels:
            if( l not in self.mu.keys() ):
                pts = y[ np.where( labels_np == l ) ]
                
                print( f'\t{l}:  {len(pts)} points found' )

                if( pts.shape[0] < 5 ): 
                    self.is_empty[l] = True
                    print( f'\t\t- insufficient number for estimation - omitting...' )
                    continue
                
                else: self.is_empty[l] = False

                self.mu[l] = np.mean( pts, axis = 0 )
                self.sigma[l] = 1 / ( pts.shape[0] - 1 ) * ( ( pts - self.mu[l] ).transpose() @ ( pts - self.mu[l] ) )
                self.info_matrix[l] = np.linalg.inv( self.sigma[l] )
                self.determinant[l] = np.linalg.det( self.sigma[l] )

    def get_score( self, x_i: np.ndarray, label: str, d1: Callable[[float], float], d2: Callable[[], float] ) -> float:

        if( label not in self.is_empty.keys() ):  return 0.0

        if( self.is_empty[label] ): return 0.0

        if( x_i.shape != (3, 1) ):
            raise ValueError( f"x_i must be shape (3, 1), not {x_i.shape}" )

        return float( d1( self.determinant[label] ) * np.exp( -d2() * ( ( x_i - self.mu[label].reshape( ( 3, 1 ) ) ).transpose() @ self.info_matrix[label] @ ( x_i - self.mu[label].reshape( ( 3, 1 ) ) ) ) / 2 ).squeeze() )