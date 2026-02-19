import numpy as np

from typing import Callable

from ndt.Parameters import Parameters
from ndt.Voxel import Voxel
from ndt.Point import Point
from utils.mat_ops import *

class TargetPointCloudP2D:
    def __init__( self, p: Parameters, v: Callable[[np.ndarray], Voxel | None] ):

        self.p: Parameters = p
        self.points: list[Point] = []
        self.get_voxel = v

        self.next_idx: int = 0

    def __call__( self ) -> Parameters: 
        return self.p
    
    def __iter__( self ):
        while self.next_idx < len( self.points ):
            yield self.points[self.next_idx]
    
    def get_points( self ) -> list[Point]:
        return self.points
    
    def get_voxels( self ) -> list[Voxel | None]:
        voxels : list[Voxel | None] = []
        for pt in self.points:
            voxels.append( pt.is_contained_by() )

        return voxels
    
    def add( self, pt: Point ):
        self.points.append( pt )

    def move_relative( self, delta_vec6: np.ndarray ):

        self.p.update( delta_vec6 )

        for pt in self.points:
            pt.move_relative( delta_vec6, self.get_voxel )

    def set_pose( self, vec6: np.ndarray ):

        self.p.set_vec6( vec6 )

        for pt in self.points:
            pt.set_position( vec6, self.get_voxel )

    def J_E( self, x_i: np.ndarray ) -> np.ndarray:

        p = self.p

        a: np.ndarray = x_i[0] * ( -p.sx * p.sz + p.cx * p.sy * p.cz )  + x_i[1] * ( -p.sx * p.cz - p.cx * p.sy * p.sz )    + x_i[2] * ( -p.cx * p.cy )
        b: np.ndarray = x_i[0] * ( p.cx * p.sz + p.sx * p.sy * p.cz )   + x_i[1] * ( -p.sx * p.sy * p.sz + p.cx * p.cz )    + x_i[2] * ( -p.sx * p.cy )
        c: np.ndarray = x_i[0] * ( -p.sy * p.cz )                       + x_i[1] * ( p.sy * p.sz )                          + x_i[2] * ( p.cy )
        d: np.ndarray = x_i[0] * ( p.sx * p.cy * p.cz )                 + x_i[1] * ( -p.sx * p.cy * p.sz )                  + x_i[2] * ( p.sx * p.sy )
        e: np.ndarray = x_i[0] * ( -p.cx * p.cy * p.cz )                + x_i[1] * ( p.cx * p.cy * p.sz )                   + x_i[2] * ( -p.cx * p.sy )
        f: np.ndarray = x_i[0] * ( -p.cy * p.sz )                       + x_i[1] * ( -p.cy * p.cz )
        g: np.ndarray = x_i[0] * ( p.cx * p.cz - p.sx * p.sy * p.sz )   + x_i[1] * ( -p.cx * p.sz - p.sx * p.sy * p.cz )
        h: np.ndarray = x_i[0] * ( p.sx * p.cz + p.cx * p.sy * p.sz )   + x_i[1] * ( p.cx * p.sy * p.cz - p.sx * p.sz )

        return np.array(
            [[1,    0,  0,  0,              c.squeeze(),    f.squeeze()], 
             [0,    1,  0,  a.squeeze(),    d.squeeze(),    g.squeeze()], 
             [0,    0,  1,  b.squeeze(),    e.squeeze(),    h.squeeze()]]
        )
    
    def H_E( self, x_i: np.ndarray ) -> np.ndarray:

        p = self.p

        a = np.array(
            [
                0,
                ( x_i[0] * ( -p.cx * p.sz - p.sx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.cz + p.sx * p.sy * p.sz )    + x_i[2] * ( p.sx * p.cy ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.sz + p.cx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.sy * p.sz - p.sx * p.cz )    + x_i[2] * ( -p.cx * p.cy ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        b = np.array(
            [
                0,
                ( x_i[0] * ( p.cx * p.cy * p.cz )                 + x_i[1] * ( -p.cx * p.cy * p.sz )                  + x_i[2] * ( p.cx * p.sy ) ).squeeze(),
                ( x_i[0] * ( p.sx * p.cy * p.cz )                 + x_i[1] * ( -p.sx * p.cy * p.sz )                  + x_i[2] * ( p.sx * p.sy ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        c = np.array(
            [
                0,
                ( x_i[0] * ( -p.sx * p.cz - p.cx * p.sy * p.sz )  + x_i[1] * ( -p.sx * p.sz - p.cx * p.sy * p.cz ) ).squeeze(),
                ( x_i[0] * ( p.cx * p.cz - p.sx * p.sy * p.sz )   + x_i[1] * ( -p.sx * p.sy * p.cz - p.cx * p.sz ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        d = np.array(
            [
                ( x_i[0] * ( -p.cy * p.cz )                       + x_i[1] * ( p.cy * p.sz )                          + x_i[2] * ( -p.sy ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.sy * p.cz )                + x_i[1] * ( p.sx * p.sy * p.sz )                   + x_i[2] * ( p.sx * p.cy ) ).squeeze(),
                ( x_i[0] * ( p.cx * p.sy * p.cz )                 + x_i[1] * ( -p.cx * p.sy * p.sz )                  + x_i[2] * ( -p.cx * p.cy ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        e = np.array(
            [
                ( x_i[0] * ( p.sy * p.sz )                        + x_i[1] * ( p.sy * p.cz ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.cy * p.sz )                + x_i[1] * ( -p.sx * p.cy * p.cz ) ).squeeze(),
                ( x_i[0] * ( p.cx * p.cy * p.sz )                 + x_i[1] * ( p.cx * p.cy * p.cz ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        f = np.array(
            [
                ( x_i[0] * ( -p.cy * p.cz )                       + x_i[1] * ( p.cy * p.sz ) ).squeeze(),
                ( x_i[0] * ( -p.cx * p.sz - p.sx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.cz + p.sx * p.sy * p.sz ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.sz + p.cx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.sy * p.sz - p.sx * p.cz ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )


        return np.array([
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), np.zeros( ( 3, 1 ) ),  np.zeros( ( 3, 1 ) ),   np.zeros( ( 3, 1 ) )    ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), np.zeros( ( 3, 1 ) ),  np.zeros( ( 3, 1 ) ),   np.zeros( ( 3, 1 ) )    ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), np.zeros( ( 3, 1 ) ),  np.zeros( ( 3, 1 ) ),   np.zeros( ( 3, 1 ) )    ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), a,                     b,                      c                       ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), b,                     d,                      e                       ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), c,                     e,                      f                       ]
        ])
    
class TargetPointCloudD2D:
    def __init__( self, p: Parameters, x: list[np.ndarray] ):

        assert len( x ) > 0, f"TargetPointCloudD2D failed to initialize - x is empty"
        assert x[0].ndim == 2 and x[0].shape[1] == 3, f"x must be shape (N, 3), not {x.shape}"

        self.p: Parameters = p

        pts = np.vstack( x )

        # Create depth-1 octree
        self._max = np.array( [ np.max( pts[:, 0] ), np.max( pts[:, 1] ), np.max( pts[:, 2] ) ] )
        self._min = np.array( [ np.min( pts[:, 0] ), np.min( pts[:, 1] ), np.min( pts[:, 2] ) ] )
        self._mid = np.array( ( self._max + self._min ) / 2 )

        x_div: list[list[np.ndarray]] = [[], [], [], [], [], [], [], []]

        for pt in pts:
            idx: int | None = self.get_voxel_idx( pt )
            if( type( idx ) == int ):   x_div[idx].append( pt )
            else:
                raise ValueError( "Target cloud voxelization failed." )
        
        self.voxels: list[Voxel] = []
        self.x: list[np.ndarray] = []
        for div in x_div:
            self.x.append( np.array( div ) )
            self.voxels.append( Voxel( np.array( div ) ) )

    def __call__( self ) -> Parameters: 
        return self.p

    def transform( self, delta_vec6: np.ndarray ):

        self.p.update( delta_vec6 )
        for vx in self.voxels:
            vx.transform( delta_vec6 )

    def get_transformed_points( final_vec6 ):
        se3 = get_se3_from_vec6( final_vec6, is_in_degrees = False )
        R = se3[:3, :3]
        t = se3[:3, 3:]

        # TODO  finish

    def get_voxel_list( self ):
        return self.voxels

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

    def J_E( self, x_i: np.ndarray ) -> np.ndarray:

        p = self.p

        a: np.ndarray = x_i[0] * ( -p.sx * p.sz + p.cx * p.sy * p.cz )  + x_i[1] * ( -p.sx * p.cz - p.cx * p.sy * p.sz )    + x_i[2] * ( -p.cx * p.cy )
        b: np.ndarray = x_i[0] * ( p.cx * p.sz + p.sx * p.sy * p.cz )   + x_i[1] * ( -p.sx * p.sy * p.sz + p.cx * p.cz )    + x_i[2] * ( -p.sx * p.cy )
        c: np.ndarray = x_i[0] * ( -p.sy * p.cz )                       + x_i[1] * ( p.sy * p.sz )                          + x_i[2] * ( p.cy )
        d: np.ndarray = x_i[0] * ( p.sx * p.cy * p.cz )                 + x_i[1] * ( -p.sx * p.cy * p.sz )                  + x_i[2] * ( p.sx * p.sy )
        e: np.ndarray = x_i[0] * ( -p.cx * p.cy * p.cz )                + x_i[1] * ( p.cx * p.cy * p.sz )                   + x_i[2] * ( -p.cx * p.sy )
        f: np.ndarray = x_i[0] * ( -p.cy * p.sz )                       + x_i[1] * ( -p.cy * p.cz )
        g: np.ndarray = x_i[0] * ( p.cx * p.cz - p.sx * p.sy * p.sz )   + x_i[1] * ( -p.cx * p.sz - p.sx * p.sy * p.cz )
        h: np.ndarray = x_i[0] * ( p.sx * p.cz + p.cx * p.sy * p.sz )   + x_i[1] * ( p.cx * p.sy * p.cz - p.sx * p.sz )

        return np.array(
            [[1,    0,  0,  0,              c.squeeze(),    f.squeeze()], 
             [0,    1,  0,  a.squeeze(),    d.squeeze(),    g.squeeze()], 
             [0,    0,  1,  b.squeeze(),    e.squeeze(),    h.squeeze()]]
        )
    
    def H_E( self, x_i: np.ndarray ) -> np.ndarray:

        p = self.p

        a = np.array(
            [
                0,
                ( x_i[0] * ( -p.cx * p.sz - p.sx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.cz + p.sx * p.sy * p.sz )    + x_i[2] * ( p.sx * p.cy ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.sz + p.cx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.sy * p.sz - p.sx * p.cz )    + x_i[2] * ( -p.cx * p.cy ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        b = np.array(
            [
                0,
                ( x_i[0] * ( p.cx * p.cy * p.cz )                 + x_i[1] * ( -p.cx * p.cy * p.sz )                  + x_i[2] * ( p.cx * p.sy ) ).squeeze(),
                ( x_i[0] * ( p.sx * p.cy * p.cz )                 + x_i[1] * ( -p.sx * p.cy * p.sz )                  + x_i[2] * ( p.sx * p.sy ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        c = np.array(
            [
                0,
                ( x_i[0] * ( -p.sx * p.cz - p.cx * p.sy * p.sz )  + x_i[1] * ( -p.sx * p.sz - p.cx * p.sy * p.cz ) ).squeeze(),
                ( x_i[0] * ( p.cx * p.cz - p.sx * p.sy * p.sz )   + x_i[1] * ( -p.sx * p.sy * p.cz - p.cx * p.sz ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        d = np.array(
            [
                ( x_i[0] * ( -p.cy * p.cz )                       + x_i[1] * ( p.cy * p.sz )                          + x_i[2] * ( -p.sy ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.sy * p.cz )                + x_i[1] * ( p.sx * p.sy * p.sz )                   + x_i[2] * ( p.sx * p.cy ) ).squeeze(),
                ( x_i[0] * ( p.cx * p.sy * p.cz )                 + x_i[1] * ( -p.cx * p.sy * p.sz )                  + x_i[2] * ( -p.cx * p.cy ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        e = np.array(
            [
                ( x_i[0] * ( p.sy * p.sz )                        + x_i[1] * ( p.sy * p.cz ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.cy * p.sz )                + x_i[1] * ( -p.sx * p.cy * p.cz ) ).squeeze(),
                ( x_i[0] * ( p.cx * p.cy * p.sz )                 + x_i[1] * ( p.cx * p.cy * p.cz ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )

        f = np.array(
            [
                ( x_i[0] * ( -p.cy * p.cz )                       + x_i[1] * ( p.cy * p.sz ) ).squeeze(),
                ( x_i[0] * ( -p.cx * p.sz - p.sx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.cz + p.sx * p.sy * p.sz ) ).squeeze(),
                ( x_i[0] * ( -p.sx * p.sz + p.cx * p.sy * p.cz )  + x_i[1] * ( -p.cx * p.sy * p.sz - p.sx * p.cz ) ).squeeze()
            ]
        ).reshape( ( 3, 1 ) )


        return np.array([
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), np.zeros( ( 3, 1 ) ),  np.zeros( ( 3, 1 ) ),   np.zeros( ( 3, 1 ) )    ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), np.zeros( ( 3, 1 ) ),  np.zeros( ( 3, 1 ) ),   np.zeros( ( 3, 1 ) )    ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), np.zeros( ( 3, 1 ) ),  np.zeros( ( 3, 1 ) ),   np.zeros( ( 3, 1 ) )    ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), a,                     b,                      c                       ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), b,                     d,                      e                       ],
            [np.zeros( ( 3, 1 ) ),    np.zeros( ( 3, 1 ) ), np.zeros( ( 3, 1) ), c,                     e,                      f                       ]
        ])