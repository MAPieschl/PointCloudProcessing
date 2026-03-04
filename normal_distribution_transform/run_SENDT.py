import sys
import os

from ndt.Parameters import Parameters
from ndt.Voxel import Voxel, VoxelGrid
from se_ndt.LabeledPoint import LabeledPoint
from se_ndt.SemanticTargetPointCloud import SemanticTargetPointCloudP2D
from se_ndt.SemanticReferencePointCloud import SemanticReferencePointCloud
from se_ndt.SemanticOptimization import SemanticOptimizationP2D

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
    labels_vertices = mesh.get_labeled_vertices()
    ref_pc = SemanticReferencePointCloud( pts = np.asarray( labels_vertices['points'] ), labels = labels_vertices['labels'], voxel_size = 10 )

    ref_pc_pts = ref_pc.get_list_of_points()
    ref_pc_vox = [f'Voxel {i}' for i in range( len( ref_pc_pts ) )]
    
    ### Create Target Point Cloud ###
    p = Parameters( se3 = np.eye( 4 ) )
    tar_pc = SemanticTargetPointCloudP2D( p )

    current_voxel_size: float = 10
    ref_pc.resize_voxel_grids( current_voxel_size )

    # aftr_dict = organize_aftr_frame_by_part( from_aftr_frame( 'D:/sim_kc46_lidar/collect_2026.Jan.22_23.51.26.8719556.UTC/Lidar/frame_0.txt' ) )
    aftr_dict = organize_aftr_frame_by_part( from_aftr_frame( 'D:/kc46_sim_collect/full_pointnet/lidar_predictions/frame_2.txt' ) )
    for i in range( len( aftr_dict['part_labels'] ) ):  
        for pt in aftr_dict['points'][i]:
            tar_pc.add( LabeledPoint( pt.reshape( ( 3, 1 ) ), aftr_dict['part_labels'][i], ref_pc.get_weighted_8_nearest_voxels ) )

    ### Instantiate Optimizer ###
    opt = SemanticOptimizationP2D()

    # Align point cloud
    target_se3 = [ tar_pc.get_pose() ]
    target_se3 += opt.course_align_by_mean( tar_pc, np.eye( 4 ) )

    while( current_voxel_size >= 1.0 ):
        print( f'Beginning iterations with voxel size set to {ref_pc.get_voxel_size()}' )
        target_se3 += opt.levenberg_marquardt( tar_pc, max_iterations = 20 )

        next_voxel_size = ref_pc.get_voxel_size()
        if( type( next_voxel_size ) == float ):
            current_voxel_size = next_voxel_size / 2
            ref_pc.resize_voxel_grids( current_voxel_size )
        else:
            print( 'Algorithm does not yet support non-uniform voxel sizes' )
            break

    ## Plot point cloud registration ###
    app = QApplication(sys.argv)
    plotter = PointCloudRegistrationPlotter( ref_pc_pts, ref_pc_vox, [ np.eye( 4 ) for i in range( len( target_se3 ) ) ],
                                             aftr_dict['points'], aftr_dict['part_labels'], target_se3 )
    
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