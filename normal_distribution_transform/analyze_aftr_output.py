import sys
import os
import json
import re
import pandas as pd

from datetime import datetime, timezone

from ndt.Parameters import Parameters
from ndt.Point import Point
from ndt.Voxel import Voxel
from ndt.ReferencePointCloud import ReferencePointCloud
from ndt.TargetPointCloud import TargetPointCloudP2D
from ndt.Optimization import OptimizationP2D

from mesh.MeshSampler import MeshSampler

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *

OBJECT_R: dict[str, np.ndarray] = {
            'lidar': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]),
            'f15_cart': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
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
            'recall': {}
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
    
    def get_lidar_num_points_at( self, timestamp ):
        return self.__lidar_num_pts[timestamp]

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

                for line in f.readlines():
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

                            if( len( line_l ) != 19 ):
                                print( f'Unable to parse lidar line:\n\t{line_l}' )
                                return
                            
                            R = []
                            for el in range( 3, 19 ):
                                R.append( float( line_l[el] ) )
                            R = np.array( R ).reshape( ( 4, 4 ) ).T

                            self.__lidar_est[timestamp] = R

                            true_pos = {}
                            false_pos = {}
                            false_neg = {}

                            with open( f'{log_dir}/{lidar_dir}/{line_l[2]}', 'r' ) as l:
                                num_points = 0
                                for lidar_line in l.readlines():
                                    lidar_line = lidar_line.strip()

                                    try:
                                        label_start = lidar_line.index( ')' )

                                    except ValueError as ve:
                                        print( f'Lidar data is in an unexpected format:\n\t{lidar_line}' )
                                        return results

                                    lidar_line_l = lidar_line[label_start + 1:].split( " " )

                                    if( len( lidar_line_l ) != 3 ):
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

                                    num_points += 1

                            self.__lidar_num_pts[timestamp] = num_points

                            self.__lidar_pred['precision'][timestamp] = {}
                            self.__lidar_pred['recall'][timestamp] = {}

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

                            if( name in OBJECT_R.keys() ):
                                R[:3, :3] = R[:3, :3] @ OBJECT_R[name]

                                self.__optitrack_truth[timestamp][name] = R

                        ## Artificially add the camera position in based on the known offset from the lidar
                        if( 'lidar' in self.__optitrack_truth[timestamp] ):
                            camera_P_global = np.zeros( ( 4, 4 ) )
                            camera_P_global[:3, :3] = CAMERA_P_LIDAR[:3, :3] @ self.__optitrack_truth[timestamp]['lidar'][:3, :3]
                            camera_P_global[:3, 3:] = self.__optitrack_truth[timestamp]['lidar'][:3, :3] @ CAMERA_P_LIDAR[:3, 3:] + self.__optitrack_truth[timestamp]['lidar'][:3, 3:]
                            camera_P_global[3, 3] = 1

                            self.__optitrack_truth[timestamp]['camera'] = camera_P_global

            return
        
        else:
            print( f"{log_dir} does not exist." )
            return
        
class AnalyzeAftrLog:
    def __init__( self, parsed_aftr_log: ParsedAftrLog, target_id: str ):

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
        self.__precision = {}
        self.__recall = {}
        self.__num_points = {}

        self.__timestamp_by_distance_lidar = {}
        self.__timestamp_by_distance_camera = {}
        self.__timestamp_by_num_points = {}

        self.__parsed_aftr_log = parsed_aftr_log
        self.__target_id = target_id

        self.__organize_data()

    def get_6DOF_residual_scatter_plots_by_distance( self, output_path: str ):

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
                                                      'x-translation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (m)' ).write_image( f'{output_path}/res_x_lidar.png', width = 800, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( pos_res )[:, 1], 
                                                      'y-translation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (m)' ).write_image( f'{output_path}/res_y_lidar.png', width = 800, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( pos_res )[:, 2], 
                                                      'z-translation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (m)' ).write_image( f'{output_path}/res_z_lidar.png', width = 800, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( rpy_res )[:, 0], 
                                                      'roll-rotation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (deg)' ).write_image( f'{output_path}/res_roll_lidar.png', width = 800, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( rpy_res )[:, 1], 
                                                      'pitch-rotation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (deg)' ).write_image( f'{output_path}/res_pitch_lidar.png', width = 800, height = 400 )

            plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                      np.array( rpy_res )[:, 2], 
                                                      'yaw-rotation residuals in LiDAR sensor frame',
                                                      'actual distance - sensor to target (m)', 
                                                      'residual (deg)' ).write_image( f'{output_path}/res_yaw_lidar.png', width = 800, height = 400 )
            
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
                                                        'x-translation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (m)' ).write_image( f'{output_path}/res_x_camera.png', width = 800, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 1], 
                                                        'y-translation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (m)' ).write_image( f'{output_path}/res_y_camera.png', width = 800, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 2], 
                                                        'z-translation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (m)' ).write_image( f'{output_path}/res_z_camera.png', width = 800, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 0], 
                                                        'roll-rotation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (deg)' ).write_image( f'{output_path}/res_rl_camera.png', width = 800, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 1], 
                                                        'pitch-rotation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (deg)' ).write_image( f'{output_path}/res_pt_camera.png', width = 800, height = 400 )

                plot_2D_scatter_with_mean_and_std( np.array( dists ), 
                                                        np.array( pos_res )[:, 2], 
                                                        'yaw-rotation residuals in camera sensor frame',
                                                        'actual distance - sensor to target (m)', 
                                                        'residual (deg)' ).write_image( f'{output_path}/res_yw_camera.png', width = 800, height = 400 )

        else:
            print( f"{output_path} does not exist" )

    def get_segmentation_performance_hist( self, output_path: str ):

        plot_class_precision_recall_hist( self.__precision, self.__recall, 'average part segmentation performance' ).write_image( f'{output_path}/seg_perf_hist.png', width = 800, height = 400 )

    def get_segmentation_performance_plots_by_range( self, output_path: str ):

        if( os.path.isdir( output_path ) ):

            dists = []
            precision = {}
            recall = {}

            for dist in list( self.__timestamp_by_distance_lidar.keys() ):
                dists.append( dist )

                for cl in self.__precision[self.__timestamp_by_distance_lidar[dist]]:
                    if( cl not in list( precision.keys() ) ):
                        precision[cl] = [[dist, self.__precision[self.__timestamp_by_distance_lidar[dist]][cl]]]
                        recall[cl] = [[dist, self.__recall[self.__timestamp_by_distance_lidar[dist]][cl]]]
                    else:
                        precision[cl].append( [dist, self.__precision[self.__timestamp_by_distance_lidar[dist]][cl]] )
                        recall[cl].append( [dist, self.__recall[self.__timestamp_by_distance_lidar[dist]][cl]] )

            for cl in list( precision.keys() ):
                precision[cl] = np.array( precision[cl] ).T
                recall[cl] = np.array( recall[cl] ).T

                plot_class_precision_recall_scatter( { cl: precision[cl] }, { cl: recall[cl] }, 'part segmentation performance by distance', 'actual distance - sensor to target (m)' ).write_image( f'{output_path}/seg_perf_dist_{cl}.png', width = 800, height = 400 )

    def __organize_data( self ):

        for timestamp in self.__parsed_aftr_log.get_timestamps():

            lidar_P_global_act = self.__parsed_aftr_log.get_optitrack_data_at( timestamp )['lidar']
            camera_P_global_act = self.__parsed_aftr_log.get_optitrack_data_at( timestamp )['camera']
            target_P_global_act = self.__parsed_aftr_log.get_optitrack_data_at( timestamp )[self.__target_id]

            target_P_global_est_lidar = self.__parsed_aftr_log.get_lidar_estimation_at( timestamp )

            self.__actual_pos_lidar_frame[timestamp] = transform_to_target_P_sensor( target_P_global_act, lidar_P_global_act )[:3, 3:]
            self.__est_pos_lidar_lidar_frame[timestamp] = transform_to_target_P_sensor( target_P_global_est_lidar, lidar_P_global_act )[:3, 3:]
            self.__actual_rpy_lidar_frame[timestamp] = get_roll_pitch_yaw_deg( transform_to_target_P_sensor( target_P_global_act, lidar_P_global_act ), True )
            self.__est_rpy_lidar_lidar_frame[timestamp] = get_roll_pitch_yaw_deg( transform_to_target_P_sensor( target_P_global_est_lidar, lidar_P_global_act ), True )
            self.__res_pos_lidar_lidar_frame[timestamp] = self.__est_pos_lidar_lidar_frame[timestamp] - self.__actual_pos_lidar_frame[timestamp]
            self.__res_rpy_lidar_lidar_frame[timestamp] = self.__est_rpy_lidar_lidar_frame[timestamp] - self.__actual_rpy_lidar_frame[timestamp]

            try:
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
            self.__num_points[timestamp] = self.__parsed_aftr_log.get_lidar_num_points_at( timestamp )

            self.__timestamp_by_distance_lidar[np.linalg.norm( self.__actual_pos_lidar_frame[timestamp] )] = timestamp
            self.__timestamp_by_num_points[self.__num_points[timestamp]] = timestamp

def main( *args ) -> bool:
    
    if( not os.path.isdir( args[0][1] ) ): return False

    aftr_log = ParsedAftrLog( args[0][1] )
    analysis = AnalyzeAftrLog( aftr_log, 'f15_cart' )

    analysis.get_6DOF_residual_scatter_plots_by_distance( 'output/' )
    analysis.get_segmentation_performance_hist( 'output/' )
    analysis.get_segmentation_performance_plots_by_range( 'output/' )

    return True

if __name__ == "__main__":

    HELP_STR = 'python run_NDT.py name_of_your_test path_to_your_data_directory'

    if( sys.argv[1] == '-h' or sys.argv[1] == '--help' ):
        print( HELP_STR )
    
    else:
        if( not main( sys.argv[1:] ) ):
            print( HELP_STR )