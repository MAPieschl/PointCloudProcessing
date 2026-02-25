import numpy as np

from typing import Callable, cast

from ndt.TargetPointCloud import TargetPointCloudP2D, TargetPointCloudD2D
from ndt.Voxel import Voxel
from ndt.Point import Point

from utils.mat_ops import *

class OptimizationP2D:
    def __init__( self, initial_lambda: float = 0.001, lambda_step: float = 10, zaganidis_d1: bool = False ):
        
        self.d1: Callable[[float], float] = cast( Callable[[float], float], lambda det: 1 / np.sqrt( ((2 * np.pi) ** 3) * det ) ) if zaganidis_d1 else cast( Callable[[float], float], lambda det: 1 )
        self.d2 = lambda : 1

        self.lbda: float = initial_lambda
        self.lbda_step: float = lambda_step

    def course_align( self, target_pc: TargetPointCloudP2D, initial_se3: np.ndarray = np.eye( 4 ) ):

        course_se3 = np.eye( 4 )

        initial_pos = -np.mean( np.array( [p.pos for p in target_pc.get_points()] ), axis = 0 )
        t = ( initial_se3[:3, :3] @ initial_pos + initial_se3[:3, 3:].reshape( (3, 1) ) ).reshape( ( 3, 1 ) )
        
        course_se3[:3, :3] = initial_se3[:3, :3]
        course_se3[:3, 3:] = t

        return course_se3

    def gradient_descent( self, target_pc: TargetPointCloudP2D, initial_se3: np.ndarray, learning_rate: float, epsilon: float = 0.00001, max_iterations: int = 100 ):

        delta_s: float = epsilon * 10
        p_n: np.ndarray = initial_se3
        iterations: int = max_iterations
        
        while( delta_s > epsilon and iterations > 0 ):

            g = self.f_d( target_pc.get_points(), target_pc.J_E )
            s_n = self.f( target_pc.get_points() )

            delta_p = -learning_rate * g

            target_pc.move_relative( delta_p )
            s_n1 = self.f( target_pc.get_points() )

            delta_s = abs( s_n - s_n1 )
            iterations -= 1

            print( f"Ending step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()} - score -> {s_n1}" )

        return p_n

    def newtons_method( self, target_pc: TargetPointCloudP2D, initial_se3: np.ndarray, epsilon: float = 0.00001, max_iterations: int = 100 ):

        delta_s: float = epsilon * 10
        p_n: np.ndarray = initial_se3
        iterations: int = max_iterations
        
        while( delta_s > epsilon and iterations > 0 ):

            H = self.f_dd( target_pc.get_points(), target_pc.J_E, target_pc.H_E, self.__validate_H ) 
            if( H is not None ):

                g = self.f_d( target_pc.get_points(), target_pc.J_E )
                s_n = self.f( target_pc.get_points() )

                delta_p = -np.linalg.inv( H ) @ g

                target_pc.move_relative( delta_p )
                s_n1 = self.f( target_pc.get_points() )

                delta_s = abs( s_n - s_n1 )
                iterations -= 1

                print( f"Ending step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()} - score -> {s_n1}" )

        return p_n
    
    def levenberg_marquardt( self, target_pc: TargetPointCloudP2D, initial_se3: np.ndarray, epsilon: float = 0.00001, max_iterations: int = 10 ):

        delta_s: float = epsilon * 10
        p_n: np.ndarray = initial_se3
        iterations: int = max_iterations
        
        while( delta_s > epsilon and iterations > 0 ):

            H = self.f_dd( target_pc.get_points(), target_pc.J_E, target_pc.H_E, self.__validate_H ) 
            if( H is not None ):

                print( f"Starting step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()}" )

                g = self.f_d( target_pc.get_points(), target_pc.J_E )
                s_n: float = self.f( target_pc.get_points() )

                delta_p = np.linalg.inv( H ) @ g

                target_pc.move_relative( delta_p )
                s_n1: float = self.f( target_pc.get_points() )

                if( s_n1 < s_n ):
                    self.lbda *= self.lbda_step
                    target_pc.move_relative( -delta_p )
                    print( f"Step {max_iterations - iterations} / {max_iterations}:  Score increased - increasing lambda to {self.lbda} to reverse direction - prediction remains {target_pc.p.to_string()} - score -> {s_n1}" )

                else:
                    self.lbda /= self.lbda_step

                    delta_s = abs( s_n1 - s_n )
                    iterations -= 1
                    print( f"Step {max_iterations - iterations} / {max_iterations}:  Score decreased - decreasing lambda to {self.lbda} to speed up progress toward the minima - prediction is now {target_pc.p.to_string()} - score -> {s_n1}" )

        return p_n

    def f( self, x: list[Point] ) -> float:

        s = 0
        for x_i in x:
            v: Voxel | None = x_i.is_contained_by()
            if( type( v ) == Voxel ):
                s += v.get_score_P2D( x_i.pos, self.d1, self.d2 )

        return -s
    
    def f_d( self, x: list[Point], J_E: Callable[[np.ndarray], np.ndarray] ) -> np.ndarray:

        g: np.ndarray = np.zeros((6, 1))
        for i in range( g.shape[0] ):
            for x_i in x:
                v: Voxel | None = x_i.is_contained_by()
                if( type( v ) == Voxel ):
                    q: np.ndarray = x_i.pos - v.mu.reshape( ( 3, 1 ) )
                    
                    g[i] += ( self.d1( v.determinant ) * self.d2() * q.transpose() @ v.info_matrix @ J_E( q )[:, i].reshape( ( 3, 1 ) ) * np.exp( (-self.d2() / 2) * q.transpose() @ v.info_matrix @ q ) ).squeeze()
        
        return g

    def f_dd( self, x: list[Point], J_E: Callable[[np.ndarray], np.ndarray], H_E: Callable[[np.ndarray], np.ndarray], verify_H: Callable[[np.ndarray], np.ndarray | None] ) -> np.ndarray | None:

        H: np.ndarray = np.zeros( ( 6, 6 ) )

        for i in range( H.shape[0] ):
            for j in range( H.shape[1] ):
                for x_i in x:
                    v: Voxel | None = x_i.is_contained_by()
                    if( type( v ) == Voxel ):
                        q: np.ndarray = x_i.pos - v.mu.reshape( ( 3, 1 ) )

                        term_1 = self.d1( v.determinant ) * self.d2() * np.exp( ( -self.d2() / 2 ) * q.transpose() @ v.info_matrix @ q )
                        term_2a = -self.d2() * ( q.transpose() @ v.info_matrix @ J_E( q )[:, i] ) @ ( q.transpose() @ v.info_matrix @ J_E( q )[:, j] )
                        term_2b = q.transpose() @ v.info_matrix @ H_E( q )[i, j, :]
                        term_2c = J_E( q )[:, j].transpose() @ v.info_matrix @ J_E( q )[:, i]

                        H[i, j] += term_1.squeeze() * ( term_2a.squeeze() + term_2b.squeeze() + term_2c.squeeze() )

        return verify_H( H )
    
    def __validate_H( self, H: np.ndarray ) -> np.ndarray | None:

        if( H.ndim != 2 or H.shape[0] != H.shape[1] ):
            print( f"Hessian must be a square, 2D matrix, not {H.shape}. Escaping..." )
            return None
        
        # Check that Hessian is positive definite
        pos_def = False
        while( not pos_def ):
            try:
                _ = np.linalg.cholesky( H )
                pos_def = True

            except np.linalg.LinAlgError:
                H += self.lbda * np.eye( H.shape[0] )
                pos_def = False

            except Exception as e:
                print( f"Hessian validation failed with the following exception:\n\t{type(e)}: {e}" )

        # Check for singularity
        # if( np.isclose( np.linalg.det( H ), 0 ) ):
        #     print( "Hessian is singular and non-invertable. Escaping..." )
        #     return None
        
        return H
    
class OptimizationD2D:
    def __init__( self, initial_lambda: float = 0.001, lambda_step: float = 10, zaganidis_d1: bool = False ):
        
        self.d1: Callable[[float], float] = cast( Callable[[float], float], lambda det: 1 / np.sqrt( ((2 * np.pi) ** 3) * det ) ) if zaganidis_d1 else cast( Callable[[float], float], lambda det: 1 )
        self.d2 = lambda x : 1

        self.lbda: float = initial_lambda
        self.lbda_step: float = lambda_step

    def newtons_method( self, target_pc: TargetPointCloudD2D, initial_se3: np.ndarray, epsilon: float = 0.00001, max_iterations: int = 100 ):

        delta_s: float = epsilon * 10
        p_n: np.ndarray = initial_se3
        iterations: int = max_iterations
        
        while( delta_s > epsilon and iterations > 0 ):

            H = self.f_dd( target_pc.get_points(), target_pc.J_E, target_pc.H_E, self.__validate_H ) 
            if( H is not None ):

                g = self.f_d( target_pc.get_points(), target_pc.J_E )
                s_n = self.f( target_pc.get_points() )

                # delta_p = -np.linalg.inv( H ) @ g
                delta_p = 0.0001 * g

                target_pc.move_relative( delta_p )
                s_n1 = self.f( target_pc.get_points() )

                delta_s = abs( s_n - s_n1 )
                iterations -= 1

                print( f"Ending step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()} - score -> {s_n1}" )

        return p_n
    
    def levenberg_marquardt( self, target_pc: TargetPointCloudD2D, initial_se3: np.ndarray, epsilon: float = 0.00001, max_iterations: int = 10 ):

        delta_s: float = epsilon * 10
        p_n: np.ndarray = initial_se3
        iterations: int = max_iterations
        
        while( delta_s > epsilon and iterations > 0 ):

            H = self.f_dd( target_pc.get_points(), target_pc.J_E, target_pc.H_E, self.__validate_H ) 
            if( H is not None ):

                print( f"Starting step {max_iterations - iterations} / {max_iterations} at {target_pc.p.to_string()}" )

                g = self.f_d( target_pc.get_points(), target_pc.J_E )
                s_n: float = self.f( target_pc.get_points() )

                delta_p = np.linalg.inv( H ) @ g

                target_pc.move_relative( delta_p )
                s_n1: float = self.f( target_pc.get_points() )

                if( s_n1 < s_n ):
                    self.lbda *= self.lbda_step
                    target_pc.move_relative( -delta_p )
                    print( f"Step {max_iterations - iterations} / {max_iterations}:  Score increased - increasing lambda to {self.lbda} to reverse direction - prediction remains {target_pc.p.to_string()} - score -> {s_n1}" )

                else:
                    self.lbda /= self.lbda_step

                    delta_s = abs( s_n1 - s_n )
                    iterations -= 1
                    print( f"Step {max_iterations - iterations} / {max_iterations}:  Score decreased - decreasing lambda to {self.lbda} to speed up progress toward the minima - prediction is now {target_pc.p.to_string()} - score -> {s_n1}" )

        return p_n

    def f( self, ref_v: list[Voxel], tar_v: list[Voxel] ) -> float:

        s = 0
        for rv in ref_v:
            for tv in tar_v:
                s += rv.get_score_D2D( tv, self.d1, self.d2 )

        return -s
    
    def f_d( self, x: list[Point], J_E: Callable[[np.ndarray], np.ndarray] ) -> np.ndarray:

        print( "OptimizationD2D.f_d() is not yet implemented" )

        return 0.0

    def f_dd( self, x: list[Point], J_E: Callable[[np.ndarray], np.ndarray], H_E: Callable[[np.ndarray], np.ndarray], verify_H: Callable[[np.ndarray], np.ndarray | None] ) -> np.ndarray | None:

        print( "OptimizationD2D.f_dd() is not yet implemented" )

        return 0.0
    
    def __validate_H( self, H: np.ndarray ) -> np.ndarray | None:

        if( H.ndim != 2 or H.shape[0] != H.shape[1] ):
            print( f"Hessian must be a square, 2D matrix, not {H.shape}. Escaping..." )
            return None
        
        # Check that Hessian is positive definite
        pos_def = False
        while( not pos_def ):
            try:
                _ = np.linalg.cholesky( H )
                pos_def = True

            except np.linalg.LinAlgError:
                H += self.lbda * np.eye( H.shape[0] )
                pos_def = False

            except Exception as e:
                print( f"Hessian validation failed with the following exception:\n\t{type(e)}: {e}" )

        # Check for singularity
        # if( np.isclose( np.linalg.det( H ), 0 ) ):
        #     print( "Hessian is singular and non-invertable. Escaping..." )
        #     return None
        
        return H