import sys
import os
import json
import re
from datetime import datetime, timezone

from ndt.Parameters import Parameters
from ndt.Point import Point
from ndt.Voxel import Voxel
from ndt.ReferencePointCloud import ReferencePointCloud
from ndt.TargetPointCloud import TargetPointCloud
from ndt.Optimization import Optimization

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

def import_logs( log_dir: str, results: dict ) -> dict:

    if( os.path.isdir( log_dir ) ):
        files = [f for f in os.listdir( log_dir ) if os.path.isfile( os.path.join( log_dir, f ) )]
        dirs = [f for f in os.listdir( log_dir ) if os.path.isdir( os.path.join( log_dir, f ) )]

        if( len( files ) > 1 ):
            print( f"Too many log files in {log_dir}, cannot determine the correct log to parse. Please reduce number of log files to 1." )
            return  results

        if( len( dirs ) > 1 ):
            print( f"Too many directories in {log_dir}, cannot determine the correct lidar directory to parse. Please reduce number of subdirectories to 1." )
            return  results
        
        log_file = files[0]
        lidar_dir = dirs[0]

        with open( log_file, 'r' ) as f:

            optitrack_truth = {}
            camera_est = {}
            lidar_est = {}
            lidar_pred = {
                'precision': [],
                'recall': []
            }
            timestamp = None

            for line in f.readlines():
                line = line.strip()

                if( line[0] == '#' ):  pass

                if( '->camera' in line ):
                    if( timestamp != None ):
                        line_l = list( line )

                        if( len( line_l ) != 18 ):
                            print( f'Unable to parse camera line:\n\t{line_l}' )
                            return results
                        
                        R = []
                        for el in range( 2, 17 ):
                            R.append( float( line_l[el] ) )
                        R = np.array( R ).reshape( ( 4, 4 ) ).T

                        camera_est[timestamp] = R
                        
                if( '->lidar' in line ):
                    if( timestamp != None ):
                        line_l = list( line )

                        if( len( line_l ) != 18 ):
                            print( f'Unable to parse lidar line:\n\t{line_l}' )
                            return results
                        
                        R = []
                        for el in range( 2, 17 ):
                            R.append( float( line_l[el] ) )
                        R = np.array( R ).reshape( ( 4, 4 ) ).T

                        lidar_est[timestamp] = R

                        true_pos = {}
                        false_pos = {}
                        false_neg = {}

                        with open( f'{lidar_dir}/{line_l}', 'r' ) as l:
                            for lidar_line in l.readlines():
                                lidar_line = lidar_line.strip()

                                try:
                                    label_start = lidar_line.index( ')' )

                                except ValueError as ve:
                                    print( f'Lidar data is in an unexpected format:\n\t{lidar_line}' )
                                    return results

                                lidar_line_l = list( lidar_line[label_start + 1:] )

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

                                true_pos[lidar_line[-1]] += 1 if lidar_line_l[-1] == lidar_line_l[-2] else 0
                                false_pos[lidar_line[-2]] += 1 if lidar_line_l[-1] != lidar_line_l[-2] else 0
                                false_neg[lidar_line[-1]] += 1 if lidar_line_l[-1] != lidar_line_l[-2] else 0

                else:
                    line = line.replace( '\t', ' ' ).split( ' ' )
                    
                    # the magic re.sub() simply truncates the OptiTrack time to 6 digits
                    timestamp = datetime.strptime( re.sub(r'(\.\d{6})\d+', r'\1', line.pop( 0 )), "%Y.%b.%d_%H.%M.%S.%f.UTC" )
                    timestamp = timestamp.replace( tzinfo = timezone.utc )
                    optitrack_truth[timestamp] = {}
                    
                    num_items = int( line.pop( 0 ) )
                    for item in range( num_items ):
                        name = line[ 17 * item ]
                        R = []
                        for el in range( 16 ):
                            R.append( float( line[17 * item + ( el + 1 )] ) )
                        R = np.array( R ).reshape( ( 4, 4 ) ).T

                        if( name in OBJECT_R.keys() ):
                            R[:3, :3] = R[:3, :3] @ OBJECT_R[name]

                            optitrack_truth[timestamp][name] = R

        return results
    
    else:
        print( f"{log_dir} does not exist." )
        return results

def main( *args ) -> bool:
    
    if( not os.path.isdir( args[1] ) ): return False

    results = {
        'actual_pos': [],
        'estimated_pos': [],
        'actual_rpy': [],
        'estimated_rpy': [],
        'residual_pos': [],
        'residual_rpy': [],
        'initial_pos': [],
        'initial_rpy': [],
        'precision': [],
        'recall': [],
        'num_points': []
    }
    with open( f'{args[0]}.json', 'w' ) as w:
        json.dump( results, w )

    return True

if __name__ == "__main__":

    HELP_STR = 'python run_NDT.py name_of_your_test path_to_your_data_directory'

    if( sys.argv[1] == '-h' or sys.argv[1] == '--help' ):
        print( HELP_STR )
    
    else:
        if( not main( sys.argv[1:] ) ):
            print( HELP_STR )