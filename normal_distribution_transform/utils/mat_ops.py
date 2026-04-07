import numpy as np
import plotly.graph_objects as go

from copy import deepcopy
from plotly.subplots import make_subplots

def _yaw( dcm: np.ndarray, yaw_rad: float ):

    R = np.array([[ np.cos(yaw_rad),    -np.sin(yaw_rad),    0.0 ], 
                  [ np.sin(yaw_rad),   np.cos(yaw_rad),    0.0 ],
                  [ 0.0,                0.0,                1.0 ]])
    
    return R @ dcm

def _pitch( dcm: np.ndarray, pitch_rad: float ):

    R = np.array([[ np.cos(pitch_rad),  0.0,    np.sin(pitch_rad)  ], 
                  [ 0.0,                1.0,    0.0                 ],
                  [ -np.sin(pitch_rad),  0.0,    np.cos(pitch_rad)   ]])
    
    return R @ dcm

def _roll( dcm: np.ndarray, roll_rad: float ):

    R = np.array([[ 1.0,    0.0,                0.0                 ], 
                  [ 0.0,    np.cos(roll_rad),   -np.sin(roll_rad)   ],
                  [ 0.0,    np.sin(roll_rad),   np.cos(roll_rad)    ]])
    
    return R @ dcm

def get_roll_pitch_yaw_deg( dcm: np.ndarray, vec_3: bool = False ):

    yaw = np.arctan2(dcm[1][0], dcm[0][0])
    pitch = -np.arcsin(dcm[2][0])
    roll = np.arctan2(dcm[2][1], dcm[2][2])

    if( vec_3 ):    return np.array( [np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)] )
    else:           return {'roll': np.rad2deg(roll), 'pitch': np.rad2deg(pitch), 'yaw': np.rad2deg(yaw)}

def get_dcm( roll_deg: float, pitch_deg: float, yaw_deg: float ):
    return _yaw( _pitch( _roll( np.eye(3), np.deg2rad(roll_deg) ), np.deg2rad(pitch_deg) ), np.deg2rad(yaw_deg) )

def get_vec6_from_se3( dcm: np.ndarray, get_degrees: bool ):

    eul_ang = get_roll_pitch_yaw_deg( dcm )

    if( get_degrees ):
        return np.array([
            dcm[0][3],
            dcm[1][3],
            dcm[2][3],
            eul_ang['roll'],
            eul_ang['pitch'],
            eul_ang['yaw']
        ]).reshape( ( 6, 1 ) )

    else:
        return np.array([
            dcm[0][3],
            dcm[1][3],
            dcm[2][3],
            np.deg2rad( eul_ang['roll'] ),
            np.deg2rad( eul_ang['pitch'] ),
            np.deg2rad( eul_ang['yaw'] )
        ]).reshape( ( 6, 1 ) )

def get_se3_from_vec6( vec6: np.ndarray, is_in_degrees: bool ):
    if( is_in_degrees ):
        R = get_dcm( vec6[3].squeeze(), vec6[4].squeeze(), vec6[5].squeeze() )

    else:
        R = get_dcm( np.rad2deg( vec6[3].squeeze() ), np.rad2deg( vec6[4].squeeze() ), np.rad2deg( vec6[5].squeeze() ) )


    se3 = np.zeros( ( 4, 4 ) )
    se3[:3, :3] = R
    se3[:3, 3:] = vec6[:3].reshape( ( 3, 1 ) )
    se3[3, 3]   = 1

    return se3

def get_DCM_positive_x_pointing_at_origin( pos: np.ndarray, roll_deg: float = 0.0 ):
    '''
    Computes a DCM for a point at (x, y, z) pointing toward the origin with roll_deg rotation about the x-axis (right = positive)
    '''

    pitch_rad = np.atan2(pos[2], np.sqrt(np.power(pos[0], 2) + np.power(pos[1], 2)))
    yaw_rad = np.pi + np.atan2(pos[1], pos[0])
    
    dcm = np.eye(3)
    dcm = _yaw( dcm, yaw_rad )
    dcm = _pitch( dcm, pitch_rad )
    dcm = _roll( dcm, np.deg2rad(roll_deg) )

    assert np.abs(np.linalg.norm(dcm[0]) - 1.0) < 0.001, "DCM not orthogonal"
    assert np.abs(np.linalg.norm(dcm[1]) - 1.0) < 0.001, "DCM not orthogonal"
    assert np.abs(np.linalg.norm(dcm[2]) - 1.0) < 0.001, "DCM not orthogonal"
    assert np.abs(np.linalg.norm(dcm.T[0]) - 1.0) < 0.001, "DCM not orthogonal"
    assert np.abs(np.linalg.norm(dcm.T[1]) - 1.0) < 0.001, "DCM not orthogonal"
    assert np.abs(np.linalg.norm(dcm.T[2]) - 1.0) < 0.001, "DCM not orthogonal"

    return dcm

def reorthogonalize( dcm: np.ndarray ):
    U, _, Vt = np.linalg.svd(dcm)
    return np.dot(U, Vt)

def plot_euler_angles(traces: np.ndarray, trace_labels: list, title: str):

    assert len(traces.shape) == 2, "`traces` must be a 2D np.ndarray"
    assert traces.shape[0] == len(trace_labels), "Number of trace labels must equal number of traces"

    x = np.arange(1, traces.shape[1])

    fig = make_subplots()

    for i, trace in enumerate(traces):
        fig.add_trace(go.Scatter(
            x = x,
            y = trace,
            mode = 'lines',
            name = trace_labels[i]
        ))

    fig.update_layout(
        title = title
    )

    fig.update_yaxes(title_text = 'Angle (deg)')

    fig.show()

def convert_radar_to_global(rg_az_el: np.ndarray, radar_pos: np.ndarray, radar_rpy: np.ndarray):
    
    g_R_r = _roll( _pitch( _yaw( np.eye(3), np.deg2rad(radar_rpy[2]) ), np.deg2rad(radar_rpy[1]) ), np.deg2rad(radar_rpy[0]) ).T

    point = np.array([
        rg_az_el[0] * np.cos(np.deg2rad(rg_az_el[1])) * np.sin(np.deg2rad(90 - rg_az_el[2])),
        rg_az_el[0] * np.sin(np.deg2rad(rg_az_el[1])) * np.sin(np.deg2rad(90 - rg_az_el[2])),
        rg_az_el[0] * np.cos(np.deg2rad(90 - rg_az_el[2]))
    ])

    return g_R_r @ point + radar_pos

def transform_to_target_P_sensor( target_P_global: np.ndarray, sensor_P_global: np.ndarray ) -> np.ndarray:

    if( target_P_global.shape != ( 4, 4 ) or sensor_P_global.shape != ( 4, 4 ) ):
        print( f"target_P_global and sensor_P_global must be shape (4, 4), not {target_P_global.shape} or {sensor_P_global.shape}" )
        return np.zeros( ( 4, 4 ) )
    
    target_P_sensor = np.zeros( ( 4, 4 ) )

    target_P_sensor[:3, :3] = sensor_P_global[:3, :3].T @ target_P_global[:3, :3]
    target_P_sensor[:3, 3:] = sensor_P_global[:3, :3].T @ ( target_P_global[:3, 3:] - sensor_P_global[:3, 3:] )
    target_P_sensor[3, 3] = 1

    return target_P_sensor

def get_transformation_error( truth_pose: np.ndarray, estimated_pose: np.ndarray, degrees: bool = False ) -> tuple[float, float]:
    '''
    This function returns a tuple of (rotation_error, translation_error). The rotation_error is the computed
    angle between the truth_pose and estimated pose using the axis-angle representation of the error. The
    translation error is simply the L2 translation error.
    '''

    T_error = np.linalg.inv( truth_pose ) @ estimated_pose

    cos_theta = ( np.trace( T_error[:3, :3] ) - 1 ) / 2
    error_R = np.arccos( np.clip( cos_theta, -1.0, 1.0 ) )
    error_t = float( np.linalg.norm( T_error[:3, 3:].reshape( ( 3, ) ) ) )
    
    if( degrees ): return ( np.rad2deg( error_R ), error_t )
    else: return ( error_R, error_t )

def transform_pc( point_cloud: np.ndarray, se3: np.ndarray ) -> np.ndarray:

    assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, f'point_cloud must have shape (N, 3), not {point_cloud.shape}'
    assert se3.shape == ( 4, 4 ), f'se3 must has shape ( 4, 4 ), not {se3.shape}'

    return ( se3[:3, :3] @ point_cloud.T + se3[:3, 3:] ).T