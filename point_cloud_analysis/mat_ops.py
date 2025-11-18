import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

def _yaw( dcm: np.ndarray, yaw_rad: float ):

    R = np.array([[ np.cos(yaw_rad),    np.sin(yaw_rad),    0.0 ], 
                  [ -np.sin(yaw_rad),   np.cos(yaw_rad),    0.0 ],
                  [ 0.0,                0.0,                1.0 ]])
    
    return R @ dcm

def _pitch( dcm: np.ndarray, pitch_rad: float ):

    R = np.array([[ np.cos(pitch_rad),  0.0,    -np.sin(pitch_rad)  ], 
                  [ 0.0,                1.0,    0.0                 ],
                  [ np.sin(pitch_rad),  0.0,    np.cos(pitch_rad)   ]])
    
    return R @ dcm

def _roll( dcm: np.ndarray, roll_rad: float ):

    R = np.array([[ 1.0,    0.0,                0.0                 ], 
                  [ 0.0,    np.cos(roll_rad),   np.sin(roll_rad)    ],
                  [ 0.0,    -np.sin(roll_rad),  np.cos(roll_rad)    ]])
    
    return R @ dcm

def get_roll_pitch_yaw_deg( dcm: np.ndarray ):

    yaw = np.arctan2(dcm[0][1], dcm[0][0])
    pitch = -np.arcsin(dcm[0][2])
    roll = np.arctan2(dcm[1][2], dcm[2][2])

    return {'roll': np.rad2deg(roll), 'pitch': np.rad2deg(pitch), 'yaw': np.rad2deg(yaw)}

def get_dcm( roll_deg: float, pitch_deg: float, yaw_deg: float ):
    return _roll( _pitch( _yaw( np.eye(3), np.deg2rad(yaw_deg) ), np.deg2rad(pitch_deg) ), np.deg2rad(roll_deg) ).T

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