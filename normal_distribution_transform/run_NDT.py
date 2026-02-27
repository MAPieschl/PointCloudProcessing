import sys
import os

from ndt.Parameters import Parameters
from ndt.Point import Point
from ndt.Voxel import Voxel, VoxelGrid
from ndt.TargetPointCloud import TargetPointCloudP2D
from ndt.Optimization import OptimizationP2D

from gui.registration.PointCloudRegistrationPlotter import PointCloudRegistrationPlotter
from PyQt6.QtWidgets import QApplication
from tqdm import tqdm

from mesh.MeshSampler import MeshSampler

from utils.aftr import *
from utils.mat_ops import *

FROM_MESH = 0
FROM_FILE = 1

TARGET_POINT_CLOUD = FROM_FILE

def main( *args ) -> bool:
    
    if( not os.path.isdir( args[0][1] ) ): return False
    if( not os.path.isfile( args[0][2] ) ): return False
    if( not os.path.isdir( args[0][4] ) ): return False

    NAME = args[0][0]
    DATA_PATH = args[0][1]
    MESH_PATH = args[0][2]
    CLASS_LABEL = args[0][3]
    OUTPUT_PATH = args[0][4]

    ### Create Reference Point Cloud ###
    mesh = MeshSampler( MESH_PATH, CLASS_LABEL, rotation_matrix = np.array( [[ 1, 0, 0 ], [0, 0, -1], [0, 1, 0]] ) )
    ref_pc = VoxelGrid( pts = np.asarray( mesh.mesh.vertices ), voxel_size = 10.0 )

    ref_pc_pts = ref_pc.get_list_of_points()
    ref_pc_vox = [f'Voxel {i}' for i in range( len( ref_pc_pts ) )]
    
    ### Create Target Point Cloud ###
    p = Parameters( se3 = np.eye( 4 ) )
    tar_pc = TargetPointCloudP2D( p, ref_pc.get_weighted_8_nearest_voxels )

    ### Instantiate Optimizer ###
    opt = OptimizationP2D()

    ### OPTION 1:  CREATE A UNIFORM SAMPLING OF THE REFERENCE MESH ###
    match( TARGET_POINT_CLOUD ):
        case 0:
            tar_pc_pts = []
            tar_pc_lbs = [ 'Target PC' ]

            for div in ref_pc_pts:
                for pt in div:
                    tar_pc_pts.append( pt )

            tar_pc_pts = np.unique( np.array( tar_pc_pts ), axis = 0 )

            rng = np.random.default_rng( seed = 42 )
            tar_pc_pts = tar_pc_pts[ rng.choice( tar_pc_pts.shape[0], size = 300, replace = False ) ]

            starting_pose = np.array( [50, 0, 8, 0, 0, 10] ).reshape( ( 6, 1 ) )
            tar_pc_pts = [ transform_pc( tar_pc_pts, get_se3_from_vec6( starting_pose, is_in_degrees = True ) ) ]

        case 1:
            starting_pose = None
            aftr_dict = from_aftr_frame( 'D:/kc46_sim_collect/full_pointnet/lidar_predictions/frame_0.txt' )

            tar_pc_pts = [ aftr_dict['points'] ]
            tar_pc_lbs = [ 'Target PC' ]

    # Set the position and load the TargetPointCloud
    for pt in tar_pc_pts[0]:  tar_pc.add( Point( pt.reshape( ( 3, 1 ) ), ref_pc.get_weighted_8_nearest_voxels ) )

    # Align point cloud
    target_se3 = [ tar_pc.get_pose() ]
    convergence_steps = opt.course_align( tar_pc )

    while( ref_pc.get_voxel_size() >= 1.0 ):
        print( f'Beginning iterations with voxel size set to {ref_pc.get_voxel_size()}' )
        convergence_steps += opt.levenberg_marquardt( tar_pc, max_iterations = 500 )
        ref_pc.build_voxel_grid( ref_pc.get_voxel_size() / 2 )

    target_se3 += convergence_steps

    if( starting_pose is not None ):
        trans_err = get_transformation_error( get_se3_from_vec6( starting_pose, is_in_degrees = False ), np.linalg.inv( target_se3[-1] ), degrees = True )
        print( f'Estimated Transformation:\n{np.linalg.inv( target_se3[-1] )}\n\nError:\n{trans_err[1]} & {trans_err[0]} degrees' )
    else:
        print( f'Estimated Transformation:\n{np.linalg.inv( target_se3[-1] )}' )

    ## Plot point cloud registration ###
    app = QApplication(sys.argv)
    plotter = PointCloudRegistrationPlotter( ref_pc_pts, ref_pc_vox, [ np.eye( 4 ) for i in range( len( target_se3 ) ) ],
                                             tar_pc_pts, tar_pc_lbs, target_se3 )
    
    plotter.show()
    sys.exit(app.exec())

    return True

if __name__ == "__main__":

    HELP_STR = 'python run_NDT.py name_of_your_test path_to_your_data_directory path_to_your_mesh target_class_label path_to_graph_target'

    if( sys.argv[1] == '-h' or sys.argv[1] == '--help' ):
        print( HELP_STR )
    
    else:
        if( not main( sys.argv[1:] ) ):
            print( HELP_STR )