import numpy as np

class Octree:
    def __init__( self, pts: np.ndarray, parent: Octree | None = None, conditioning_eps: float = 0.001 ):
        self.__validate_points( pts )

        self.__pts = pts
        self.__parent: Octree | None = parent
        self.__children: list[Octree] = []
        self.__max_corner: np.ndarray = np.max( pts, axis = 0 )
        self.__min_corner: np.ndarray = np.min( pts, axis = 0 )
        self.__geo_center: np.ndarray = ( self.__max_corner - self.__min_corner ) / 2 + self.__min_corner

        self._eps = conditioning_eps
        self._mu = np.mean( pts, axis = 0 )
        self._sigma = 1 / ( pts.shape[0] - 1 ) * ( ( pts - self._mu ).transpose() @ ( pts - self._mu ) ) + np.eye( 3 ) * self._eps
        self._info_matrix = np.linalg.inv( self._sigma )
        self._determinant = np.linalg.det( self._sigma )

    def is_leaf( self ) -> bool:
        return len( self.__children ) == 0
    
    def get_parent( self ) -> Octree | None:
        return self.__parent
    
    def get_children( self ) -> list[Octree]:
        return self.__children
    
    def get_leaf_containing( self, pt: np.ndarray ) -> Octree | None:
        self.__validate_point( pt )
        if( self.is_leaf() ):

            if( pt[0] <= self.__max_corner[0] and pt[1] <= self.__max_corner[1] and pt[2] <= self.__max_corner[2] \
               and pt[0] >= self.__min_corner[0] and pt[1] >= self.__min_corner[1] and pt[2] >= self.__min_corner[2] ):
                
                return self
        
        else:
            for ch in self.__children:
                if( type( ch.get_leaf_containing( pt ) ) == Octree ):
                    return ch
            
        return None
    
    def get_geometric_center( self ):
        return self.__geo_center
    
    def get_mu_for_leaf_containing_point( self, pt: np.ndarray ) -> np.ndarray | None:
        self.__validate_point( pt )

        leaf = self.get_leaf_containing( pt )
        if( leaf is not None ): return leaf._mu
        else:                   return None
    
    def get_sigma_for_leaf_containing_point( self, pt: np.ndarray ) -> np.ndarray | None:
        self.__validate_point( pt )

        leaf = self.get_leaf_containing( pt )
        if( leaf is not None ): return leaf._sigma
        else:                   return None
    
    def get_info_matrix_for_leaf_containing_point( self, pt: np.ndarray ) -> np.ndarray | None:
        self.__validate_point( pt )

        leaf = self.get_leaf_containing( pt )
        if( leaf is not None ): return leaf._info_matrix
        else:                   return None
    
    def get_determinant_for_leaf_containing_point( self, pt: np.ndarray ) -> np.ndarray | None:
        self.__validate_point( pt )

        leaf = self.get_leaf_containing( pt )
        if( leaf is not None ): return leaf._determinant
        else:                   return None

    def get_all_leaf_nodes( self ) -> list[Octree]:

        if( self.is_leaf() ):
            return [ self ]
        
        else:
            leaves = []
            for child in self.__children:
                leaves += child.get_all_leaf_nodes()

            return leaves

    def get_my_k_nearest_leaf_nodes( self, k: int ) -> list[Octree] | None:
        if( self.__parent is not None ):
            return self.__parent.get_k_nearest_leaf_nodes( self.__geo_center, k )

        else:
            return None

    def get_k_nearest_leaf_nodes( self, pt: np.ndarray, k: int ) -> list[Octree]:
        self.__validate_point( pt )

        if( self.__parent is None ):
            leaf_distances = {}
            leaves = self.get_all_leaf_nodes()
            for leaf in leaves:
                leaf_distances[leaf.get_L2_distance_to_geometric_center( pt )] = leaf

            leaves_sorted = list( leaf_distances.keys() )
            leaves_sorted.sort()

            return [leaf_distances[leaves_sorted[n]] for n in range( k )]

        else:
            return self.__parent.get_k_nearest_leaf_nodes( pt, k )
        
    def get_L2_distance_to_geometric_center( self, pt: np.ndarray ) -> float:
        self.__validate_point( pt )

        return np.abs( np.linalg.norm( pt - self.__geo_center ) )
    
    def subdivide( self ):
        if( len( self.__children ) == 0 ):

            _max = self.__max_corner
            _mid = self.__geo_center
            _min = self.__min_corner

            new_divs = [[] for i in range( 8 )]

            for pt in self.__pts:

                if( pt[0] <= _mid[0] and pt[0] >= _min[0] and pt[1] <= _mid[1] and pt[1] >= _min[1] and pt[2] <= _mid[2] and pt[2] >= _min[2] ):
                    new_divs[0].append( pt )

                elif( pt[0] <= _mid[0] and pt[0] >= _min[0] and pt[1] <= _mid[1] and pt[1] >= _min[1] and pt[2] > _mid[2] and pt[2] <= _max[2]):
                    new_divs[1].append( pt )

                elif( pt[0] <= _mid[0] and pt[0] >= _min[0] and pt[1] > _mid[1] and pt[1] <= _max[1] and pt[2] <= _mid[2] and pt[2] >= _min[2]):
                    new_divs[2].append( pt )

                elif( pt[0] <= _mid[0] and pt[0] >= _min[0] and pt[1] > _mid[1] and pt[1] <= _max[1] and pt[2] > _mid[2] and pt[2] <= _max[2]):
                    new_divs[3].append( pt )

                elif( pt[0] > _mid[0] and pt[0] <= _max[0] and pt[1] <= _mid[1] and pt[1] >= _min[1] and pt[2] <= _mid[2] and pt[2] >= _min[2]):
                    new_divs[4].append( pt )

                elif( pt[0] > _mid[0] and pt[0] <= _max[0] and pt[1] <= _mid[1] and pt[1] >= _min[1] and pt[2] > _mid[2] and pt[2] <= _max[2]):
                    new_divs[5].append( pt )

                elif( pt[0] > _mid[0] and pt[0] <= _max[0] and pt[1] > _mid[1] and pt[1] <= _max[1] and pt[2] <= _mid[2] and pt[2] >= _min[2]):
                    new_divs[6].append( pt )

                elif( pt[0] > _mid[0] and pt[0] <= _max[0] and pt[1] > _mid[1] and pt[1] <= _max[1] and pt[2] > _mid[2] and pt[2] <= _max[2]):
                    new_divs[7].append( pt )
                else:
                    raise ValueError( f"Octree:  point {pt} did not fall into an octree node" )
                
            self.__children = [Octree( np.array( new_divs[i] ), parent = self, conditioning_eps = self._eps ) for i in range( len( new_divs ) )]

    def __validate_point( self, pt: np.ndarray ) -> None:

        assert pt.shape == ( 3, ) or pt.shape == ( 3, 1 ), f'pt must be shape ( 3, ) or ( 3, 1 ), not {pt.shape}'

    def __validate_points( self, pts: np.ndarray ) -> None:

        assert pts.ndim == 2, f'pts must be shape ( N, 3 ), not {pts.shape}'
        assert pts.shape[1] == 3, f'pts must be shape ( N, 3 ), not {pts.shape}'