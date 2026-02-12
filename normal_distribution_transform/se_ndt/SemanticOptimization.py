import numpy as np

from typing import Callable, cast

from se_ndt.SemanticTargetPointCloud import SemanticTargetPointCloud
from se_ndt.SemanticVoxel import SemanticVoxel
from se_ndt.LabeledPoint import LabeledPoint

class SemanticOptimization:
    def __init__( self, initial_lambda: float = 0.001, lambda_step: float = 10, zaganidis_d1: bool = False ):
        
        self.d1: Callable[[float], float] = cast( Callable[[float], float], lambda det: 1 / np.sqrt( ((2 * np.pi) ** 3) * det ) ) if zaganidis_d1 else cast( Callable[[float], float], lambda det: 1 )
        self.d2 = lambda : 1

        self.lbda: float = initial_lambda
        self.lbda_step: float = lambda_step

    def newtons_method( self, target_pc: SemanticTargetPointCloud, initial_se3: np.ndarray, epsilon: float = 0.00001, max_iterations: int = 100 ):

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
    
    def levenberg_marquardt( self, target_pc: SemanticTargetPointCloud, initial_se3: np.ndarray, epsilon: float = 0.00001, max_iterations: int = 10 ):

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

    def f( self, x: list[LabeledPoint] ) -> float:

        s = 0
        for x_i in x:
            v: SemanticVoxel | None = x_i.is_contained_by()
            if( type( v ) == SemanticVoxel ):
                s += v.get_score( x_i.pos, x_i.label, self.d1, self.d2 )

        return -s
    
    def f_d( self, x: list[LabeledPoint], J_E: Callable[[np.ndarray], np.ndarray] ) -> np.ndarray:

        g: np.ndarray = np.zeros((6, 1))
        for i in range( g.shape[0] ):
            for x_i in x:
                v: SemanticVoxel | None = x_i.is_contained_by()
                if( type( v ) == SemanticVoxel ):
                    q: np.ndarray = x_i.pos - v.mu[x_i.label].reshape( ( 3, 1 ) )
                    
                    g[i] += ( self.d1( v.determinant[x_i.label] ) * self.d2() * q.transpose() @ v.info_matrix[x_i.label] @ J_E( q )[:, i].reshape( ( 3, 1 ) ) * np.exp( (-self.d2() / 2) * q.transpose() @ v.info_matrix[x_i.label] @ q ) ).squeeze()
        
        return g

    def f_dd( self, x: list[LabeledPoint], J_E: Callable[[np.ndarray], np.ndarray], H_E: Callable[[np.ndarray], np.ndarray], verify_H: Callable[[np.ndarray], np.ndarray | None] ) -> np.ndarray | None:

        H: np.ndarray = np.zeros( ( 6, 6 ) )

        for i in range( H.shape[0] ):
            for j in range( H.shape[1] ):
                for x_i in x:
                    v: SemanticVoxel | None = x_i.is_contained_by()
                    if( type( v ) == SemanticVoxel ):
                        q: np.ndarray = x_i.pos - v.mu[x_i.label].reshape( ( 3, 1 ) )

                        term_1 = self.d1( v.determinant[x_i.label] ) * self.d2() * np.exp( ( -self.d2() / 2 ) * q.transpose() @ v.info_matrix[x_i.label] @ q )
                        term_2a = -self.d2() * ( q.transpose() @ v.info_matrix[x_i.label] @ J_E( q )[:, i] ) @ ( q.transpose() @ v.info_matrix[x_i.label] @ J_E( q )[:, j] )
                        term_2b = q.transpose() @ v.info_matrix[x_i.label] @ H_E( q )[i, j, :]
                        term_2c = J_E( q )[:, j].transpose() @ v.info_matrix[x_i.label] @ J_E( q )[:, i]

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