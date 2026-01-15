from dependencies import *

def solve_kabsch( truth_vecs: np.ndarray, measured_vecs: np.ndarray ):
    '''
    Uses the Kabsch algorithm to solve for the SE(3) to minimize error.
    
    @param truth_vecs       (np.ndarray) (N, 3) matrix with truth vectors from the sensor to the target
    @param measured_vecs    (np.ndarray) (N, 3) matrix with measured vectors from the sensor to the target

    @return SE(3)           (np.ndarray) (4, 4)
    '''

    truth_centered = truth_vecs - np.mean( truth_vecs, axis = 0 )
    meas_centered = measured_vecs - np.mean( measured_vecs, axis = 0 )

    H = meas_centered.T @ truth_centered

    U, S, Vt = np.linalg.svd( H )

    R = Vt.T @ U.T

    if( np.linalg.det( R ) < 0 ):
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    t = np.mean( truth_vecs, axis = 0 ) - R @ np.mean( measured_vecs, axis = 0 )

    aligned_vecs = ( R @ measured_vecs.T ).T + t
    errors = aligned_vecs - truth_vecs
    rmse = np.sqrt( np.mean( np.sum( errors ** 2, axis = 1 ) ) )

    return R, t, rmse