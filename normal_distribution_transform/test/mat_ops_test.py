import sys
sys.path.append( '' )

import numpy as np

from utils.mat_ops import *

'''
In the functions below, all manually-typed arrays are written in column-major format
for readability  
'''

def TEST__get_roll_pitch_yaw_deg():

    FNAME = 'get_roll_pitch_yaw_deg'

    ## Test no rotation
    assert np.allclose( get_roll_pitch_yaw_deg( np.eye( 3 ), True ), np.array( [0, 0, 0] ) ), f'{FNAME} test 1 failed.'

    ## Test roll only
    dcm = np.array( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] ).T
    computed = get_roll_pitch_yaw_deg( dcm, True )
    truth = np.array( [90, 0, 0] )
    assert np.allclose( computed, truth ), f'{FNAME} test 2 failed - computed = {computed} | truth = {truth}.'

    ## Test pitch only
    dcm = np.array( [[0, 0, -1], [0, 1, 0], [1, 0, 0]] ).T
    computed = get_roll_pitch_yaw_deg( dcm, True )
    truth = np.array( [0, 90, 0] )
    assert np.allclose( computed, truth ), f'{FNAME} test 3 failed - computed = {computed} | truth = {truth}.'

    ## Test yaw only
    dcm = np.array( [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] ).T
    computed = get_roll_pitch_yaw_deg( dcm, True )
    truth = np.array( [0, 0, 90] ) 
    assert np.allclose( computed, truth ), f'{FNAME} test 4 failed - computed = {computed} | truth = {truth}.'

    ## Test 3-axis
    roll = np.array( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] ).T
    pitch = np.array( [[0, 0, -1], [0, 1, 0], [1, 0, 0]] ).T
    yaw = np.array( [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] ).T
    
    computed = get_roll_pitch_yaw_deg( yaw @ pitch @ roll, True )
    truth = np.array( [0, 90, 0] )
    assert np.allclose( computed, truth ), f'{FNAME} test 4 failed - computed = {computed} | truth = {truth}.'

def TEST__get_dcm():

    FNAME = 'get_dcm'

    ## Test identity
    assert np.allclose( get_dcm( 0, 0, 0 ), np.eye( 3 ) ), f'{FNAME} test 1 failed.'

    ## Test roll
    truth = np.array( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] ).T
    computed = get_dcm( 90, 0, 0 )
    assert np.allclose( computed, truth ), f'{FNAME} test 2 failed - \ncomputed = \n{computed}\n\ntruth = \n{truth}\n.'

    ## Test pitch
    truth = np.array( [[0, 0, -1], [0, 1, 0], [1, 0, 0]] ).T
    computed = get_dcm( 0, 90, 0 )
    assert np.allclose( computed, truth ), f'{FNAME} test 3 failed - \ncomputed = \n{computed}\n\ntruth = \n{truth}\n.'

    ## Test yaw
    truth = np.array( [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] ).T
    computed = get_dcm( 0, 0, 90 )
    assert np.allclose( computed, truth ), f'{FNAME} test 4 failed - \ncomputed = \n{computed}\n\ntruth = \n{truth}\n.'

    ## Test 3-axis
    truth = np.array( [[0, 0, -1], [0, 1, 0], [1, 0, 0]] ).T
    computed = get_dcm( 90, 90, 90 )
    assert np.allclose( computed, truth ), f'{FNAME} test 5 failed - \ncomputed = \n{computed}\n\ntruth = \n{truth}\n.'

def TEST__get_transformation_error():

    FNAME = 'get_transformation_error'

    ## Test equal matrices
    assert get_transformation_error( np.eye( 4 ), np.eye( 4 ) ) == ( 0, 0 ), f'{FNAME} test 1 failed.'

    ## Test translation only
    t_est = np.array( [ 3, 4, 5 ] )

    est = np.eye( 4 )
    est[:3, 3:] = t_est.reshape( ( 3, 1 ) )
    assert get_transformation_error( np.eye( 4 ), est ) == ( 0, np.linalg.norm( t_est ) ), f'{FNAME} test 2 failed.'

    ## Test rotation only
    R_est = np.array( [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] ) # 90 deg rotation about R_z

    est = np.eye( 4 )
    est[:3, :3] = R_est
    R_error, t_error = get_transformation_error( np.eye( 4 ), est )
    assert np.isclose( R_error, np.pi / 2 ), f'{FNAME} test 3 (rotation) failed.'
    assert np.isclose( t_error, 0 ), f'{FNAME} test 3 (translation) failed.'

    ## Test rotation and translation
    R_est = np.array( [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] ) # 90 degree rotation about R_z
    t_est = np.array( [3, 4, 5] )

    est = np.eye( 4 )
    est[:3, :3] = R_est
    est[:3, 3:] = t_est.reshape( ( 3, 1 ) )
    R_error, t_error = get_transformation_error( np.eye( 4 ), est )
    assert np.isclose( R_error, np.pi / 2 ), f'{FNAME} test 4 (rotation) failed.'
    assert np.isclose( t_error, np.linalg.norm( t_est ) ), f'{FNAME} test 4 (translation) failed.'

def TEST__transform_pc():

    FNAME = 'transform_pc'

    pc = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])

    pc_translated = np.array([
        [10, 0, 0],
        [11, 0, 0],
        [10, 2, 0],
        [10, 0, 3]
    ])

    pc_rotated = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [-2, 0, 0],
        [0, 0, 3]
    ])

    pc_rot_trans = np.array([
        [10, 0, 0],
        [10, 1, 0],
        [8, 0, 0],
        [10, 0, 3]    
        ])

    tx_pc = transform_pc( pc, np.eye( 4 ) )
    assert np.allclose( tx_pc, pc ), f'{FNAME} test 1 failed -\ncomputed = \n{tx_pc}\n\ntruth - \n{pc}\n'

    tx_pc = transform_pc( pc, np.array( [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [10, 0, 0, 1]] ).T )
    assert np.allclose( tx_pc, pc_translated ), f'{FNAME} test 2 failed -\ncomputed = \n{tx_pc}\n\ntruth - \n{pc_translated}\n'

    tx_pc = transform_pc( pc, np.array( [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] ).T )
    assert np.allclose( tx_pc, pc_rotated ), f'{FNAME} test 3 failed -\ncomputed = \n{tx_pc}\n\ntruth - \n{pc_rotated}\n'

    tx_pc = transform_pc( pc, np.array( [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [10, 0, 0, 1]] ).T )
    assert np.allclose( tx_pc, pc_rot_trans ), f'{FNAME} test 4 failed -\ncomputed = \n{tx_pc}\n\ntruth - \n{pc_rot_trans}\n'

if __name__ == "__main__":

    TEST__get_roll_pitch_yaw_deg()
    TEST__get_dcm()
    TEST__get_transformation_error()
    TEST__transform_pc()

    print( 'All tests passed.' )