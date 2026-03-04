import numpy as np

from typing import Callable, cast

from ndt.TargetPointCloud import TargetPointCloudP2D, TargetPointCloudD2D
from ndt.Voxel import VoxelGrid, Voxel
from ndt.Point import Point

from utils.mat_ops import *

class OptimizationP2D:
    def __init__( self, initial_lambda: float = 0.001, lambda_step: float = 10, zaganidis_d1: bool = False ):
        
        self.d1: Callable[[float], float] = cast( Callable[[float], float], lambda det: 1 / np.sqrt( ((2 * np.pi) ** 3) * det ) ) if zaganidis_d1 else cast( Callable[[float], float], lambda det: 1 )
        self.d2 = lambda : 1

        self.lbda: float = initial_lambda
        self.lbda_step: float = lambda_step

    def coarse_align( self, target_pc: TargetPointCloudP2D, initial_se3: np.ndarray = np.eye( 4 ) ) -> list[np.ndarray]:

        coarse_se3 = np.eye( 4 )

        initial_pos = -np.mean( np.array( [p.pos for p in target_pc.get_points()] ), axis = 0 )
        t = ( initial_se3[:3, :3] @ initial_pos + initial_se3[:3, 3:].reshape( (3, 1) ) ).reshape( ( 3, 1 ) )
        
        coarse_se3[:3, :3] = initial_se3[:3, :3]
        coarse_se3[:3, 3:] = t

        target_pc.set_pose( get_vec6_from_se3( coarse_se3, get_degrees = False ) )
        print( f'Course alignment set an initial pose of {target_pc.p.to_string()}' )

        return [ coarse_se3 ]

    def gradient_descent( self, target_pc: TargetPointCloudP2D, initial_se3: np.ndarray, learning_rate: float, epsilon: float = 0.0001, max_iterations: int = 100 ) -> list[np.ndarray]:

        target_pc.set_pose( get_vec6_from_se3( initial_se3, get_degrees = False ) )

        delta_s: float = epsilon * 10
        iterations: int = max_iterations

        step_se3 = []

        print( f'Initializing at {target_pc.p.to_string()}' )
        
        while( delta_s > epsilon and iterations > 0 ):

            g = self.f_d( target_pc.get_points(), target_pc.J_E )
            s_n = self.f( target_pc.get_points() )

            delta_p = -learning_rate * g

            target_pc.set_pose( target_pc.p.vec6 + delta_p )
            s_n1 = self.f( target_pc.get_points() )

            step_se3.append( target_pc.get_pose() )

            delta_s = abs( s_n - s_n1 )
            iterations -= 1
             
            print( f"Ending step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()} - score -> {s_n1}" )

        return step_se3

    def newtons_method( self, target_pc: TargetPointCloudP2D, epsilon: float = 0.00001, max_iterations: int = 100 ) -> list[np.ndarray]:

        delta_s: float = epsilon * 10
        iterations: int = max_iterations

        step_se3 = []

        print( f'Initializing Newton\'s Method at {target_pc.p.to_string()}' )
        
        while( delta_s > epsilon and iterations > 0 ):

            H = self.f_dd( target_pc.get_points(), target_pc.J_E, target_pc.H_E, self.validate_H_shift ) 
            if( H is not None ):

                g = self.f_d( target_pc.get_points(), target_pc.J_E )
                s_n = self.f( target_pc.get_points() )

                delta_p = -np.linalg.inv( H ) @ g

                target_pc.set_pose( target_pc.p.vec6 + delta_p )
                s_n1 = self.f( target_pc.get_points() )

                step_se3.append( target_pc.get_pose() )

                delta_s = abs( s_n - s_n1 )
                iterations -= 1

                print( f"Ending step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()} - score -> {s_n1}" )

        return step_se3
    
    def levenberg_marquardt( self, target_pc: TargetPointCloudP2D, epsilon: float = 0.00001, max_iterations: int = 10 ) -> list[np.ndarray]:

        delta_s: float = epsilon * 10
        iterations: int = max_iterations

        lbda = self.lbda
        lbda_step = self.lbda_step

        step_se3 = []
        total_steps = -1
        H = None
        g = np.zeros( ( 6, 1 ) )

        while( delta_s > epsilon ):

            if( total_steps < len( step_se3 ) ):
                H = self.f_dd( target_pc.get_points(), target_pc.J_E, target_pc.H_E, self.validate_H_scaled ) 

            if( H is not None ):

                print( f"Starting step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()}" )

                if( total_steps < len( step_se3 ) ):
                    g = self.f_d( target_pc.get_points(), target_pc.J_E )
                    total_steps += 1

                s_n: float = self.f( target_pc.get_points() )

                delta_p = -np.linalg.inv( H + lbda * np.diag( np.diag( H ) ) ) @ g

                current_vec6 = deepcopy( target_pc.p.vec6 )
                target_pc.set_pose( current_vec6 + delta_p )
                s_n1: float = self.f( target_pc.get_points() )

                iterations -= 1

                if( s_n1 > s_n ):
                    lbda *= lbda_step

                    target_pc.set_pose( current_vec6 )

                    print( f"Step {max_iterations - iterations} / {max_iterations}:  Score increased - increasing lambda to {lbda} to reverse direction - prediction remains {target_pc.p.to_string()} - score moved from {s_n:.6f} -> {s_n1:.6f}" )

                else:
                    lbda /= lbda_step

                    delta_s = abs( s_n1 - s_n )
                    step_se3.append( target_pc.get_pose() )

                    print( f"Step {max_iterations - iterations} / {max_iterations}:  Score decreased - decreasing lambda to {lbda} to speed up progress toward the minima - prediction is now {target_pc.p.to_string()} - score moved from {s_n:.6f} -> {s_n1:.6f}" )

        return step_se3

    def f( self, x: list[Point] ) -> float:

        s = 0
        for x_i in x:
            for v, weight in x_i.get_nearest_weighted_voxels():
                if( type( v ) == Voxel and not v.is_empty ):
                    s += weight * v.get_score_P2D( x_i.pos, self.d1, self.d2 )

        return -s
    
    def f_d( self, x: list[Point], J_E: Callable[[Point], np.ndarray] ) -> np.ndarray:

        g: np.ndarray = np.zeros((6, 1))

        for j in range( g.shape[0] ):
            for x_i in x:
                for v, weight in x_i.get_nearest_weighted_voxels():
                    if( type( v ) == Voxel and not v.is_empty ):
                        q: np.ndarray = x_i.pos - v.mu.reshape( ( 3, 1 ) )

                        g[j] += weight * ( self.d1( v.determinant ) * self.d2() * q.transpose() @ v.info_matrix @ J_E( x_i )[:, j].reshape( ( 3, 1 ) ) * np.exp( (-self.d2() / 2) * q.transpose() @ v.info_matrix @ q ) ).squeeze()

        return g

    def f_dd( self, x: list[Point], J_E: Callable[[Point], np.ndarray], H_E: Callable[[Point], np.ndarray], verify_H: Callable[[np.ndarray], np.ndarray | None] ) -> np.ndarray | None:

        H: np.ndarray = np.zeros( ( 6, 6 ) )

        for k in range( H.shape[0] ):
            for j in range( H.shape[1] ):
                for x_i in x:
                    for v, weight in x_i.get_nearest_weighted_voxels():
                        if( type( v ) == Voxel ):
                            if( not v.is_empty ):
                                q: np.ndarray = x_i.pos - v.mu.reshape( ( 3, 1 ) )
                                J = J_E( x_i )

                                term_1 = self.d1( v.determinant ) * self.d2() * np.exp( ( -self.d2() / 2 ) * q.transpose() @ v.info_matrix @ q )
                                term_2a = -self.d2() * ( q.transpose() @ v.info_matrix @ J[:, k] ) @ ( q.transpose() @ v.info_matrix @ J[:, j] )
                                term_2b = q.transpose() @ v.info_matrix @ H_E( x_i )[k, j, :]
                                term_2c = J[:, j].transpose() @ v.info_matrix @ J[:, k]

                                H[k, j] += weight * term_1.squeeze() * ( term_2a.squeeze() + term_2b.squeeze() + term_2c.squeeze() )

        return verify_H( H )
    
    def validate_H_shift( self, H: np.ndarray ) -> np.ndarray | None:

        if( H.ndim != 2 or H.shape[0] != H.shape[1] ):
            print( f"Hessian must be a square, 2D matrix, not {H.shape}. Escaping..." )
            return None
        
        eigvals = np.linalg.eigvals( H )
        
        # Check if any eigenvalues are non-positive
        if np.any( eigvals <= 0 ):
            lbda = -np.min( eigvals ) + 1e-3
            H = H + np.eye( H.shape[0] ) * lbda
        
        return H
    
    def validate_H_scaled( self, H: np.ndarray ) -> np.ndarray | None:

        if( H.ndim != 2 or H.shape[0] != H.shape[1] ):
            print( f"Hessian must be a square, 2D matrix, not {H.shape}. Escaping..." )
            return None
        
        eigval, eigvec = np.linalg.eigh( H )
        
        # If any eigenvalue is non-positive, clip it to a small value
        if np.any( eigval <= 0 ):
            clipped_eigenvalues = np.maximum( eigval, 1e-6 )
            
            # Reconstruct the Hessian
            H = eigvec @ np.diag( clipped_eigenvalues ) @ eigvec.T
        
        return H
    