import os
import re
import numpy as np

from datetime import datetime, timezone
from typing import Callable
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.plotting import *
from utils.mat_ops import *

OBJECT_P: dict[str, np.ndarray] = {
            'lidar': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            'f-15_model': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            'kc-46': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        }

CAMERA_P_LIDAR: np.ndarray = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

class ParsedAftrLog:
    def __init__( self, log_dir: str ):

        self.__optitrack_truth = {}
        self.__camera_est = {}
        self.__lidar_est = {}
        self.__lidar_pred = {
            'precision': {},
            'recall': {},
            'mIoU': {},
            'inference_time': {},
            'registration_time': {}
        }
        self.__lidar_num_pts = {}

        self.__import_logs( log_dir )

    def to_string( self ) -> str:
        out = "\n\nParsedAftrLog:\n"

        for time in list( self.__optitrack_truth.keys() ):

            out += f"\tOptitrack truth for {time}\n"

            for item in list( self.__optitrack_truth[time].keys() ):
                out += f"\t\t{item}:\n{self.__optitrack_truth[time][item]}\n\n"

            out += f"\tCamera estimation:\n{self.__camera_est[time]}\n\n"
            out += f"\tLiDAR estimation: \n{self.__lidar_est[time]}\n\n"
            out += f"\tLiDAR predictions:\n"
            for item in list( self.__lidar_pred['precision'][time] ):
                out += f"\t\t{item}:\tPrecision -> {self.__lidar_pred['precision'][time][item]:.3f} || Recall -> {self.__lidar_pred['recall'][time][item]:.3f}\n"

            out += '\n\n'

        return out
    
    def get_timestamps( self ) -> list[datetime]:
        return list( self.__optitrack_truth.keys() )
    
    def get_optitrack_data_at( self, timestamp: datetime ):
        return self.__optitrack_truth[timestamp]
    
    def get_camera_estimation_at( self, timestamp: datetime ):
        return self.__camera_est[timestamp]
    
    def get_lidar_estimation_at( self, timestamp: datetime ):
        return self.__lidar_est[timestamp]
    
    def get_lidar_precision_at( self, timestamp: datetime ):
        return self.__lidar_pred['precision'][timestamp]
    
    def get_lidar_recall_at( self, timestamp: datetime ):
        return self.__lidar_pred['recall'][timestamp]
    
    def get_lidar_mIoU_at( self, timestamp: datetime ):
        return self.__lidar_pred['mIoU'][timestamp]
    
    def get_lidar_inference_time_at( self, timestamp: datetime ):
        if( timestamp in self.__lidar_pred['inference_time'].keys() ):  return self.__lidar_pred['inference_time'][timestamp]
        else:                                                           return None
    
    def get_lidar_registration_time_at( self, timestamp: datetime ):
        if( timestamp in self.__lidar_pred['registration_time'].keys() ):   return self.__lidar_pred['registration_time'][timestamp]
        else:                                                               return None

    def get_lidar_num_points_at( self, timestamp ):
        return self.__lidar_num_pts[timestamp]
    
    def reprocess_using_estimates_from( self, log_dir: str, alternate_log: str ):

        alt_est: dict[str, np.ndarray] = {}
        self.__lidar_est.clear()
        self.__lidar_pred['registration_time'].clear()

        if( os.path.isfile( alternate_log ) ):

            with open( alternate_log, 'r' ) as f:
                for line in f.readlines():
                    line = line.strip()
                    if( '->lidar' in line ):
                        line_l = line.split( ' ' )

                        if( len( line_l ) == 21 ):
                            
                            R = []
                            for el in range( 5, 21 ):
                                R.append( float( line_l[el] ) )
                            R = np.array( R ).reshape( ( 4, 4 ) ).T

                            alt_est[line_l[4]] = R @ OBJECT_P['lidar']

        if( os.path.isdir( log_dir ) ):
            files = [f for f in os.listdir( log_dir ) if os.path.isfile( os.path.join( log_dir, f ) )]

            if( len( files ) > 2 ):
                print( f"Too many log files in {log_dir}, cannot determine the correct log to parse. Please reduce number of log files to 1." )
                return
            
            log_file = [ f for f in files if 'log_' in f ][0]

            with open( f"{log_dir}/{log_file}", 'r' ) as f:

                timestamp = None

                for line in f.readlines():
                    line = line.strip()

                    if( '->camera' in line ):
                        continue

                    elif( '->lidar' in line ):
                        if( timestamp != None ):
                            line_l = line.split( " " )

                            if( len( line_l ) == 21 ):      frame_filename = line_l[4]
                            elif( len( line_l ) == 19 ):    frame_filename = line_l[2]
                            else:
                                print( f'LiDAR line unrecognized {line}' )
                                return
                            
                            if( frame_filename in alt_est.keys() ):
                                self.__lidar_est[timestamp] = alt_est[frame_filename]

                            else:
                                self.__optitrack_truth.pop( timestamp, None )
                                self.__camera_est.pop( timestamp, None )
                                self.__lidar_pred['precision'].pop( timestamp, None )
                                self.__lidar_pred['recall'].pop( timestamp, None )
                                self.__lidar_pred['inference_time'].pop( timestamp, None )

                    else:
                        line = line.replace( '\t', ' ' ).split( ' ' )
                            
                        ## the magic re.sub() simply truncates the OptiTrack time to 6 digits
                        timestamp = datetime.strptime( re.sub(r'(\.\d{6})\d+', r'\1', line.pop( 0 )), "%Y.%b.%d_%H.%M.%S.%f.UTC" )
                        timestamp = timestamp.replace( tzinfo = timezone.utc )

    def __compute_mIoU( 
            self,
            truth_labels    :   list[str], 
            pred_labels     :   list[str]
        ) -> float | None:

        if( len( truth_labels ) != len( pred_labels ) ):
            print( f'truth_labels and pred_labels must be of equal length, not {len( truth_labels )} and {len( pred_labels )}' )
            return None
        
        labels = []
        for l in truth_labels:
            if( l not in labels ):  labels.append( l )

        part_ious = []
        for l in labels:
            truth = ( np.array( truth_labels ) == l )
            pred = ( np.array( pred_labels ) == l )

            intersection = np.sum( truth & pred )
            union = np.sum( truth | pred )

            if( union == 0 ):   part_ious.append( 1.0 )
            else:               part_ious.append( intersection / union )

        return float( np.mean( np.array( part_ious ) ) )    

    def __import_logs( self, log_dir: str ):

        if( os.path.isdir( log_dir ) ):
            files = [f for f in os.listdir( log_dir ) if os.path.isfile( os.path.join( log_dir, f ) )]
            dirs = [f for f in os.listdir( log_dir ) if os.path.isdir( os.path.join( log_dir, f ) )]

            results = {}

            if( len( files ) > 1 ):
                print( f"Too many log files in {log_dir}, cannot determine the correct log to parse. Please reduce number of log files to 1." )
                return

            if( len( dirs ) > 1 ):
                print( f"Too many directories in {log_dir}, cannot determine the correct lidar directory to parse. Please reduce number of subdirectories to 1." )
                return
            
            log_file = files[0]
            lidar_dir = dirs[0]

            with open( f"{log_dir}/{log_file}", 'r' ) as f:

                timestamp = None

                print( f'ParsedAftrLog is parsing {log_dir}/{log_file}...' )
                for line in tqdm( f.readlines() ):
                    line = line.strip()

                    if( line[0] == '#' ):  pass

                    if( '->camera' in line ):
                        if( timestamp != None ):
                            line_l = line.split( " " )

                            if( len( line_l ) != 18 ):
                                print( f'Unable to parse camera line:\n\t{line_l}' )
                                return
                            
                            R = []
                            for el in range( 2, 18 ):
                                R.append( float( line_l[el] ) )
                            R = np.array( R ).reshape( ( 4, 4 ) ).T

                            if( timestamp not in list( self.__optitrack_truth.keys() ) ):
                                print( f"Camera estimation at time {timestamp} provided without truth data." )
                                return

                            self.__camera_est[timestamp] = R
                            
                    elif( '->lidar' in line ):
                        if( timestamp != None ):
                            line_l = line.split( " " )

                            if( len( line_l ) == 21 ):
                            
                                self.__lidar_pred['inference_time'][timestamp] = float( line_l[2] )
                                self.__lidar_pred['registration_time'][timestamp] = float( line_l[3] )
                                
                                R = []
                                for el in range( 5, 21 ):
                                    R.append( float( line_l[el] ) )
                                R = np.array( R ).reshape( ( 4, 4 ) ).T

                                self.__lidar_est[timestamp] = R @ OBJECT_P['lidar']

                                frame_filename = line_l[4]
                            
                            elif( len( line_l ) == 19 ):
                                
                                R = []
                                for el in range( 3, 19 ):
                                    R.append( float( line_l[el] ) )
                                R = np.array( R ).reshape( ( 4, 4 ) ).T

                                self.__lidar_est[timestamp] = R @ OBJECT_P['lidar']

                                frame_filename = line_l[2]

                            else:

                                print( f'LiDAR line unrecognized {line}' )
                                return

                            true_pos = {}
                            false_pos = {}
                            false_neg = {}

                            truth_list = []
                            pred_list = []

                            with open( f'{log_dir}/{lidar_dir}/{frame_filename}', 'r' ) as l:
                                num_points = 0
                                for lidar_line in l.readlines():
                                    lidar_line = lidar_line.strip()

                                    try:
                                        label_start = lidar_line.index( ')' )

                                    except ValueError as ve:
                                        print( f'Lidar data is in an unexpected format:\n\t{lidar_line}' )
                                        return results

                                    lidar_line_l = lidar_line[label_start + 1:].split( " " )

                                    if( len( lidar_line_l ) < 3 ):
                                        print( f'Unable to parse lidar line:\n\t{lidar_line_l}' )
                                        return results
                                    
                                    ## NOTE:  [-1] -> truth label || [-2] -> predicted label
                                    
                                    if( lidar_line_l[-1] not in list( true_pos.keys() ) ):
                                        true_pos[lidar_line_l[-1]] = 0
                                        false_pos[lidar_line_l[-1]] = 0
                                        false_neg[lidar_line_l[-1]] = 0
                                    
                                    if( lidar_line_l[-2] not in list( true_pos.keys() ) ):
                                        true_pos[lidar_line_l[-2]] = 0
                                        false_pos[lidar_line_l[-2]] = 0
                                        false_neg[lidar_line_l[-2]] = 0

                                    true_pos[lidar_line_l[-1]] += 1 if lidar_line_l[-1] == lidar_line_l[-2] else 0
                                    false_pos[lidar_line_l[-2]] += 1 if lidar_line_l[-1] != lidar_line_l[-2] else 0
                                    false_neg[lidar_line_l[-1]] += 1 if lidar_line_l[-1] != lidar_line_l[-2] else 0

                                    truth_list.append( lidar_line_l[-1] )
                                    pred_list.append( lidar_line_l[-2] )

                                    num_points += 1

                            self.__lidar_num_pts[timestamp] = num_points

                            self.__lidar_pred['precision'][timestamp] = {}
                            self.__lidar_pred['recall'][timestamp] = {}
                            self.__lidar_pred['mIoU'][timestamp] = self.__compute_mIoU( truth_list, pred_list )

                            for key in list( true_pos.keys() ):

                                try:    self.__lidar_pred['precision'][timestamp][key] = true_pos[key] / ( true_pos[key] + false_pos[key] )
                                except: self.__lidar_pred['precision'][timestamp][key] = 0

                                try:    self.__lidar_pred['recall'][timestamp][key] = true_pos[key] / ( true_pos[key] + false_neg[key] )
                                except: self.__lidar_pred['recall'][timestamp][key] = 0

                    else:
                        line = line.replace( '\t', ' ' ).split( ' ' )
                        
                        ## the magic re.sub() simply truncates the OptiTrack time to 6 digits
                        timestamp = datetime.strptime( re.sub(r'(\.\d{6})\d+', r'\1', line.pop( 0 )), "%Y.%b.%d_%H.%M.%S.%f.UTC" )
                        timestamp = timestamp.replace( tzinfo = timezone.utc )
                        self.__optitrack_truth[timestamp] = {}
                        
                        num_items = int( line.pop( 0 ) )
                        for item in range( num_items ):
                            name = line[ 17 * item ]
                            R = []
                            for el in range( 16 ):
                                R.append( float( line[17 * item + ( el + 1 )] ) )
                            R = np.array( R ).reshape( ( 4, 4 ) ).T

                            if( name in OBJECT_P.keys() ):
                                self.__optitrack_truth[timestamp][name] = R

                        ## Artificially add the camera position in based on the known offset from the lidar
                        if( 'lidar' in self.__optitrack_truth[timestamp] ):
                            camera_P_global = np.zeros( ( 4, 4 ) )
                            camera_P_global[:3, :3] = CAMERA_P_LIDAR[:3, :3] @ self.__optitrack_truth[timestamp]['lidar'][:3, :3]
                            camera_P_global[:3, 3:] = self.__optitrack_truth[timestamp]['lidar'][:3, :3] @ CAMERA_P_LIDAR[:3, 3:] + self.__optitrack_truth[timestamp]['lidar'][:3, 3:]
                            camera_P_global[3, 3] = 1

                            self.__optitrack_truth[timestamp]['camera'] = camera_P_global

        else:
            print( f"{log_dir} does not exist." )
        
class AnalyzeAftrLog:
    def __init__( self, parsed_aftr_log: ParsedAftrLog, name: str, target_id: str, timestamps: list[datetime] | None = None ):

        self.__actual_pos_lidar_frame = {}
        self.__actual_pos_camera_frame = {}
        self.__est_pos_lidar_lidar_frame = {}
        self.__est_pos_camera_camera_frame = {}
        self.__actual_rpy_lidar_frame = {}
        self.__actual_rpy_camera_frame = {}
        self.__est_rpy_lidar_lidar_frame = {}
        self.__est_rpy_camera_camera_frame = {}
        self.__res_pos_lidar_lidar_frame = {}
        self.__res_pos_camera_camera_frame = {}
        self.__res_rpy_lidar_lidar_frame = {}
        self.__res_rpy_camera_camera_frame= {}
        self.__res_L2_lidar_lidar_frame = {}
        self.__res_rot_lidar_lidar_frame = {}
        self.__precision = {}
        self.__recall = {}
        self.__mIoU = {}
        self.__num_points = {}
        self.__inference_time = {}
        self.__registration_time = {}

        self.__timestamp_by_distance_lidar = {}
        self.__timestamp_by_distance_camera = {}
        self.__timestamp_by_num_points = {}
        self.__timestamp_by_initial_rotation_error = {}

        self.__parsed_aftr_log = parsed_aftr_log
        self.__name = name
        self.__target_id = target_id

        self.__organize_data( timestamps )

    def get_6DOF_residual_scatter_plots_by_distance( self, output_path: str, meter_range: tuple[float, float] = ( -25, 25 ), degree_range = ( -185, 185 ) ):

        if( os.path.isdir( output_path ) ):

            ## LiDAR

            dists = []
            pos_res = []
            rpy_res = []

            for dist in list( self.__timestamp_by_distance_lidar.keys() ):
                dists.append( dist )
                pos_res.append( self.__res_pos_lidar_lidar_frame[self.__timestamp_by_distance_lidar[dist]] )
                rpy_res.append( self.__res_rpy_lidar_lidar_frame[self.__timestamp_by_distance_lidar[dist]] )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( pos_res )[:, 0], 
                                                      f'{self.__name}:  x-translation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (m)',
                                                      y_range = meter_range ).write_image( f'{output_path}/res_x_lidar_by_dist.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( pos_res )[:, 1], 
                                                      f'{self.__name}:  y-translation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_y_lidar_by_dist.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( pos_res )[:, 2], 
                                                      f'{self.__name}:  z-translation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_z_lidar_by_dist.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( rpy_res )[:, 0], 
                                                      f'{self.__name}:  roll-rotation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (deg)',
                                                      y_range = degree_range  ).write_image( f'{output_path}/res_roll_lidar_by_dist.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( rpy_res )[:, 1], 
                                                      f'{self.__name}:  pitch-rotation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_pitch_lidar_by_dist.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( rpy_res )[:, 2], 
                                                      f'{self.__name}:  yaw-rotation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_yaw_lidar_by_dist.png', width = 1200, height = 400 )
            
            if( len( self.__timestamp_by_distance_camera.keys() ) > 0 ):
                ## Camera

                dists = []
                pos_res = []
                rpy_res = []

                for dist in list( self.__timestamp_by_distance_camera.keys() ):
                    dists.append( dist )
                    pos_res.append( self.__res_pos_camera_camera_frame[self.__timestamp_by_distance_camera[dist]] )
                    rpy_res.append( self.__res_rpy_camera_camera_frame[self.__timestamp_by_distance_camera[dist]] )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 0], 
                                                        f'{self.__name}:  x-translation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (m)',
                                                        y_range = meter_range  ).write_image( f'{output_path}/res_x_camera.png', width = 1200, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 1], 
                                                        f'{self.__name}:  y-translation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (m)',
                                                        y_range = meter_range  ).write_image( f'{output_path}/res_y_camera.png', width = 1200, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 2], 
                                                        f'{self.__name}:  z-translation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (m)',
                                                        y_range = meter_range  ).write_image( f'{output_path}/res_z_camera.png', width = 1200, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 0], 
                                                        f'{self.__name}:  roll-rotation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (deg)',
                                                        y_range = degree_range ).write_image( f'{output_path}/res_rl_camera.png', width = 1200, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 1], 
                                                        f'{self.__name}:  pitch-rotation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (deg)',
                                                        y_range = degree_range ).write_image( f'{output_path}/res_pt_camera.png', width = 1200, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 2], 
                                                        f'{self.__name}:  yaw-rotation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (deg)',
                                                        y_range = degree_range ).write_image( f'{output_path}/res_yw_camera.png', width = 1200, height = 400 )

        else:
            print( f"{output_path} does not exist" )

    def get_6DOF_residual_scatter_plots_by_initial_rotation_error( self, output_path: str, meter_range: tuple[float, float] = ( -25, 25 ), degree_range = ( -185, 185 ) ):

        if( os.path.isdir( output_path ) ):

            ## LiDAR

            err_r = []
            pos_res = []
            rpy_res = []

            for err in list( self.__timestamp_by_initial_rotation_error.keys() ):
                err_r.append( err )
                pos_res.append( self.__res_pos_lidar_lidar_frame[self.__timestamp_by_initial_rotation_error[err]] )
                rpy_res.append( self.__res_rpy_lidar_lidar_frame[self.__timestamp_by_initial_rotation_error[err]] )

            plot_2D_scatter_with_mean_and_std( np.array( err_r ), 
                                                      np.array( pos_res )[:, 0], 
                                                      f'{self.__name}:  x-translation residuals in LiDAR sensor frame',
                                                      'initial angle-off (deg)', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_x_lidar_by_rot_err.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( err_r ), 
                                                      np.array( pos_res )[:, 1], 
                                                      f'{self.__name}:  y-translation residuals in LiDAR sensor frame',
                                                      'initial angle-off (deg)', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_y_lidar_by_rot_err.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( err_r ), 
                                                      np.array( pos_res )[:, 2], 
                                                      f'{self.__name}:  z-translation residuals in LiDAR sensor frame',
                                                      'initial angle-off (deg)', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_z_lidar_by_rot_err.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( err_r ), 
                                                      np.array( rpy_res )[:, 0], 
                                                      f'{self.__name}:  roll-rotation residuals in LiDAR sensor frame',
                                                      'initial angle-off (deg)', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_roll_lidar_by_rot_err.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( err_r ), 
                                                      np.array( rpy_res )[:, 1], 
                                                      f'{self.__name}:  pitch-rotation residuals in LiDAR sensor frame',
                                                      'initial angle-off (deg)', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_pitch_lidar_by_rot_err.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( err_r ), 
                                                      np.array( rpy_res )[:, 2], 
                                                      f'{self.__name}:  yaw-rotation residuals in LiDAR sensor frame',
                                                      'initial angle-off (deg)', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_yaw_lidar_by_rot_err.png', width = 1200, height = 400 )

        else:
            print( f"{output_path} does not exist" )

    def get_6DOF_residual_scatter_plots_by_number_of_points( self, output_path: str, meter_range: tuple[float, float] = ( -25, 25 ), degree_range = ( -185, 185 ) ):

        if( os.path.isdir( output_path ) ):

            ## LiDAR

            num_points = []
            pos_res = []
            rpy_res = []

            for err in list( self.__timestamp_by_num_points.keys() ):
                num_points.append( err )
                pos_res.append( self.__res_pos_lidar_lidar_frame[self.__timestamp_by_num_points[err]] )
                rpy_res.append( self.__res_rpy_lidar_lidar_frame[self.__timestamp_by_num_points[err]] )

            plot_2D_scatter_with_mean_and_std( np.array( num_points ), 
                                                      np.array( pos_res )[:, 0], 
                                                      f'{self.__name}:  x-translation residuals in LiDAR sensor frame',
                                                      'number of points in frame', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_x_lidar_by_points.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( num_points ), 
                                                      np.array( pos_res )[:, 1], 
                                                      f'{self.__name}:  y-translation residuals in LiDAR sensor frame',
                                                      'number of points in frame', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_y_lidar_by_points.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( num_points ), 
                                                      np.array( pos_res )[:, 2], 
                                                      f'{self.__name}:  z-translation residuals in LiDAR sensor frame',
                                                      'number of points in frame', 
                                                      'residual (m)',
                                                      y_range = meter_range  ).write_image( f'{output_path}/res_z_lidar_by_points.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( num_points ), 
                                                      np.array( rpy_res )[:, 0], 
                                                      f'{self.__name}:  roll-rotation residuals in LiDAR sensor frame',
                                                      'number of points in frame', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_roll_lidar_by_points.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( num_points ), 
                                                      np.array( rpy_res )[:, 1], 
                                                      f'{self.__name}:  pitch-rotation residuals in LiDAR sensor frame',
                                                      'number of points in frame', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_pitch_lidar_by_points.png', width = 1200, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( num_points ), 
                                                      np.array( rpy_res )[:, 2], 
                                                      f'{self.__name}:  yaw-rotation residuals in LiDAR sensor frame',
                                                      'number of points in frame', 
                                                      'residual (deg)',
                                                      y_range = degree_range ).write_image( f'{output_path}/res_yaw_lidar_by_by_points.png', width = 1200, height = 400 )

        else:
            print( f"{output_path} does not exist" )

    def get_segmentation_performance_hist( self, output_path: str ):

        plot_class_precision_recall_hist( self.__precision, self.__recall, f'{self.__name}:  average part segmentation performance' ).write_image( f'{output_path}/seg_perf_hist.png', width = 800, height = 400 )

    def get_segmentation_performance_plots_by_range( self, output_path: str ):

        if( os.path.isdir( output_path ) ):

            dists = []
            precision = {}
            recall = {}

            for dist in list( self.__timestamp_by_distance_lidar.keys() ):
                dists.append( dist )

                for cl in self.__precision[self.__timestamp_by_distance_lidar[dist]]:
                    if( not( self.__precision[self.__timestamp_by_distance_lidar[dist]][cl] < 0.001 and self.__recall[self.__timestamp_by_distance_lidar[dist]][cl] < 0.001 ) ):
                        if( cl not in list( precision.keys() ) ):
                            precision[cl] = [[dist, self.__precision[self.__timestamp_by_distance_lidar[dist]][cl]]]
                            recall[cl] = [[dist, self.__recall[self.__timestamp_by_distance_lidar[dist]][cl]]]
                        else:
                            precision[cl].append( [dist, self.__precision[self.__timestamp_by_distance_lidar[dist]][cl]] )
                            recall[cl].append( [dist, self.__recall[self.__timestamp_by_distance_lidar[dist]][cl]] )

            for cl in list( precision.keys() ):
                precision[cl] = np.array( precision[cl] ).T
                recall[cl] = np.array( recall[cl] ).T

                plot_class_precision_recall_scatter( { cl: precision[cl] }, { cl: recall[cl] }, f'{self.__name}: part segmentation performance for {cl} by distance', 'actual distance - sensor to target (m)' ).write_image( f'{output_path}/seg_perf_dist_{cl}.png', width = 1200, height = 400 )

    def get_segmentation_performance_plots_by_number_of_points( self, output_path: str ):

        if( os.path.isdir( output_path ) ):

            dists = []
            precision = {}
            recall = {}

            for dist in list( self.__timestamp_by_num_points.keys() ):
                dists.append( dist )

                for cl in self.__precision[self.__timestamp_by_num_points[dist]]:
                    if( not( self.__precision[self.__timestamp_by_num_points[dist]][cl] < 0.001 and self.__recall[self.__timestamp_by_num_points[dist]][cl] < 0.001 ) ):
                        if( cl not in list( precision.keys() ) ):
                            precision[cl] = [[dist, self.__precision[self.__timestamp_by_num_points[dist]][cl]]]
                            recall[cl] = [[dist, self.__recall[self.__timestamp_by_num_points[dist]][cl]]]
                        else:
                            precision[cl].append( [dist, self.__precision[self.__timestamp_by_num_points[dist]][cl]] )
                            recall[cl].append( [dist, self.__recall[self.__timestamp_by_num_points[dist]][cl]] )

            for cl in list( precision.keys() ):
                precision[cl] = np.array( precision[cl] ).T
                recall[cl] = np.array( recall[cl] ).T

                plot_class_precision_recall_scatter( { cl: precision[cl] }, { cl: recall[cl] }, f'{self.__name}: part segmentation performance for {cl} by number of points in frame', 'number of points in frame' ).write_image( f'{output_path}/seg_perf_dist_{cl}.png', width = 1200, height = 400 )

    def get_mIoU_distribution( self ) -> np.ndarray:
        return np.array( [ self.__mIoU[timestamp] for timestamp in list( self.__mIoU.keys() ) ] )
    
    def get_L2_residual_distribution( self ) -> np.ndarray:
        return np.array( [ self.__res_L2_lidar_lidar_frame[timestamp] for timestamp in list( self.__res_L2_lidar_lidar_frame.keys() ) ] )

    def get_rot_residual_distribution( self ) -> np.ndarray:
        return np.array( [ self.__res_rot_lidar_lidar_frame[timestamp] for timestamp in list( self.__res_rot_lidar_lidar_frame.keys() ) ] )

    def get_timing_info( self, output_path: str ):

        if( os.path.isdir( output_path ) ):

            plot_histogram( np.array( list( self.__inference_time.items() ) )[:, 1], num_bins = 50, title = f'{self.__name}: inference time distribution', x_label = 'inference time (ms)' ).write_image( f'{output_path}/inference_time.png', width = 1200, height = 400 )
            plot_histogram( np.array( list( self.__registration_time.items() ) )[:, 1], num_bins = 50, title = f'{self.__name}: registration time distribution', x_label = 'registration time (ms)' ).write_image( f'{output_path}/registration_time.png', width = 1200, height = 400 )
        
    def get_inference_timing_distribution( self ) -> np.ndarray:
        return np.array( [ self.__inference_time[timestamp] for timestamp in list( self.__inference_time.keys() ) ] )
    
    def get_registration_timing_distribution( self ) -> np.ndarray:
        return np.array( [ self.__registration_time[timestamp] for timestamp in list( self.__registration_time.keys() ) ] )

    def produce_minimized_extrinsic_calibration_for_lidar( self ):

        '''
        Uses the Kabsch algorithm to refine the extrinsic calibration based of the sensor
        '''

        delta_P = np.zeros( ( 4, 4 ) )

        act_pos: np.ndarray = np.array( [ self.__actual_pos_lidar_frame[i] for i in list( self.__actual_pos_lidar_frame.keys() ) ] ).squeeze()
        est_pos: np.ndarray = np.array( [ self.__est_pos_lidar_lidar_frame[i] for i in list( self.__est_pos_lidar_lidar_frame.keys() ) ] ).squeeze()

        if( act_pos.shape[1] == 3 and est_pos.shape[1] == 3 and act_pos.shape[0] == act_pos.shape[0] ):

            centroid_act = np.mean( act_pos, axis = 0 )
            centroid_est = np.mean( est_pos, axis = 0 )

            act_res = act_pos - centroid_act
            est_res = est_pos - centroid_est

            R, rssd = Rotation.align_vectors( act_res, est_res, return_sensitivity = False )

            t_ext = centroid_act - R.as_matrix() @ centroid_est

            delta_P[:3, :3] = R.as_matrix()
            delta_P[:3, 3:] = t_ext.reshape( ( 3, 1 ) )
            delta_P[3, 3] = 1
        
        else:
            print( f"Unable to minimize error with act_pos shape of {act_pos.shape} and est_pos shape of {est_pos.shape}" )
        
        return delta_P

    def __organize_data( self, timestamps : list[datetime] | None = None ):

        ts_list = self.__parsed_aftr_log.get_timestamps() if timestamps is None else timestamps

        print( f'AnalyzeAftrLog is parsing timestamped log...' )
        for timestamp in tqdm( ts_list ):

            lidar_P_global_act = self.__parsed_aftr_log.get_optitrack_data_at( timestamp )['lidar']
            target_P_global_act = self.__parsed_aftr_log.get_optitrack_data_at( timestamp )[self.__target_id]

            target_P_global_est_lidar = self.__parsed_aftr_log.get_lidar_estimation_at( timestamp )

            self.__actual_pos_lidar_frame[timestamp] = transform_to_target_P_sensor( target_P_global_act, lidar_P_global_act )[:3, 3:]
            self.__est_pos_lidar_lidar_frame[timestamp] = transform_to_target_P_sensor( target_P_global_est_lidar, lidar_P_global_act )[:3, 3:]
            self.__actual_rpy_lidar_frame[timestamp] = get_roll_pitch_yaw_deg( transform_to_target_P_sensor( target_P_global_act, lidar_P_global_act ), True )
            self.__est_rpy_lidar_lidar_frame[timestamp] = get_roll_pitch_yaw_deg( transform_to_target_P_sensor( target_P_global_est_lidar, lidar_P_global_act ), True )
            self.__res_pos_lidar_lidar_frame[timestamp] = self.__est_pos_lidar_lidar_frame[timestamp] - self.__actual_pos_lidar_frame[timestamp]
            self.__res_rpy_lidar_lidar_frame[timestamp] = self.__est_rpy_lidar_lidar_frame[timestamp] - self.__actual_rpy_lidar_frame[timestamp]
            self.__res_rot_lidar_lidar_frame[timestamp], self.__res_L2_lidar_lidar_frame[timestamp] = get_transformation_error( 
                transform_to_target_P_sensor( target_P_global_act, lidar_P_global_act ),
                transform_to_target_P_sensor( target_P_global_est_lidar, lidar_P_global_act ),
                degrees = True
            )

            try:

                camera_P_global_act = self.__parsed_aftr_log.get_optitrack_data_at( timestamp )['camera']
                target_P_global_est_camera = self.__parsed_aftr_log.get_camera_estimation_at( timestamp )

                self.__actual_pos_camera_frame[timestamp] = transform_to_target_P_sensor( target_P_global_act, camera_P_global_act )[:3, 3:]
                self.__est_pos_camera_camera_frame[timestamp] = transform_to_target_P_sensor( target_P_global_est_camera, camera_P_global_act )[:3, 3:]
                self.__actual_rpy_camera_frame[timestamp] = get_roll_pitch_yaw_deg( transform_to_target_P_sensor( target_P_global_act, camera_P_global_act ), True )
                self.__est_rpy_camera_camera_frame[timestamp] = get_roll_pitch_yaw_deg( transform_to_target_P_sensor( target_P_global_est_camera, camera_P_global_act ), True )
                self.__res_pos_camera_camera_frame [timestamp] = self.__est_pos_camera_camera_frame[timestamp] - self.__actual_pos_camera_frame[timestamp]
                self.__res_rpy_camera_camera_frame[timestamp] = self.__est_rpy_camera_camera_frame[timestamp] - self.__actual_rpy_camera_frame[timestamp]
                
                self.__timestamp_by_distance_camera[np.linalg.norm( self.__actual_pos_camera_frame[timestamp] )] = timestamp

            except KeyError:
                ## Camera data not required - the above block will throw a KeyError if no camera data were provided
                pass

            self.__precision[timestamp] = self.__parsed_aftr_log.get_lidar_precision_at( timestamp )
            self.__recall[timestamp] = self.__parsed_aftr_log.get_lidar_recall_at( timestamp )
            self.__mIoU[timestamp] = self.__parsed_aftr_log.get_lidar_mIoU_at( timestamp )
            self.__inference_time[timestamp] = self.__parsed_aftr_log.get_lidar_inference_time_at( timestamp )
            self.__registration_time[timestamp] = self.__parsed_aftr_log.get_lidar_registration_time_at( timestamp )
            self.__num_points[timestamp] = self.__parsed_aftr_log.get_lidar_num_points_at( timestamp )

            self.__timestamp_by_distance_lidar[np.linalg.norm( self.__actual_pos_lidar_frame[timestamp] )] = timestamp
            self.__timestamp_by_num_points[self.__num_points[timestamp]] = timestamp

            err_r, _ = get_transformation_error( lidar_P_global_act, np.eye( 4 ) )
            self.__timestamp_by_initial_rotation_error[ np.rad2deg( abs( err_r ) ) ] = timestamp

###============ Free Functions ==================

def from_aftr_frame( filepath: str, print_func: Callable[[str], None] = print ) -> dict:

    frame = {
        'points': np.array([]),
        'class_labels': [],
        'part_labels': []
    }

    if( os.path.isfile( filepath ) ):

        with open( filepath, "r" ) as f:
            for l in f:
                l = l.strip()
                start_i = l.find( '(' )
                end_i = l.find( ')' )

                pos = np.array( l[start_i + 1 : end_i].replace( ',', '' ).split( ' ' ), dtype = float )
                labels = l[end_i + 1:].split( ' ' )

                if( frame['points'].size < 1 ): frame['points'] = pos
                else:                           frame['points'] = np.vstack( ( frame['points'], pos ), dtype = float )

                labels.remove( '' )

                if( len( labels ) > 1 ):
                    frame['class_labels'].append( labels[0] )
                    frame['part_labels'].append( labels[1] )
                else:
                    print_func( f"{filepath} is missing either class_labels or part_labels." )

    else:
        print_func( f"{filepath} is not a valid filename." )
    
    
    return frame

def organize_aftr_frame_by_part( aftr_frame: dict, print_func: Callable[[str], None] = print ) -> dict:

    frame = {
        'points': [],
        'part_labels': []
    }

    if( 'points' in aftr_frame.keys() and 'part_labels' in aftr_frame.keys() ):
        part_np = np.array( aftr_frame['part_labels'] )

        for lbl in aftr_frame['part_labels']:
            if( lbl not in frame['part_labels'] ):
                frame['part_labels'].append( lbl )
                ind = np.where( part_np == lbl )
                frame['points'].append( aftr_frame['points'][ind] )

    else:
        print_func( "aftr_frame should be the dictionary output from .from_aftr_frame()" )

    return frame


### FREE HELPER FUNCTIONS ###

def generate_pose_aligned_timestamps_from_aftr_frames(
        aftrLogs    : list[ParsedAftrLog],
        pose_of     : str                   = 'lidar'
    ) -> list[tuple]:

    ts_lists = [ l.get_timestamps() for l in aftrLogs ]

    last_pose = np.zeros( ( 4, 4 ) )
    last_index = [ 0 for i in range( len( aftrLogs ) ) ]

    aligned_ts = []

    print( 'Finding pose-aligned samples in Aftr frames...' )
    for ts in tqdm( ts_lists[0] ):

        next_pose = aftrLogs[0].get_optitrack_data_at( ts )[pose_of]

        if( not np.isclose( next_pose, last_pose ) ):

            new_set = [ ts ]

            for i in range( 1, len( aftrLogs ) ):
                for j in range( last_index[i], len( ts_lists[i] ) ):


    return [()]