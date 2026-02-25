import sys
import os

from ndt.Parameters import Parameters
from ndt.Point import Point
from ndt.Voxel import Voxel
from ndt.ReferencePointCloud import ReferencePointCloud
from ndt.TargetPointCloud import TargetPointCloudP2D
from ndt.Optimization import OptimizationP2D

from gui.registration.PointCloudRegistrationPlotter import PointCloudRegistrationPlotter
from PyQt6.QtWidgets import QApplication

from mesh.MeshSampler import MeshSampler

from utils.aftr import *
from utils.mat_ops import *

FROM_MESH = 0
FROM_FILE = 1

TARGET_POINT_CLOUD = FROM_MESH

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
    ref_pc = ReferencePointCloud( y = np.asarray( mesh.mesh.vertices ) )

    ref_pc_pts = ref_pc.get_pc_list()
    ref_pc_vox = [f'Voxel {i}' for i in range( len( ref_pc_pts ) )]
    
    ### Create Target Point Cloud ###
    p = Parameters( se3 = np.eye( 4 ) )
    tar_pc = TargetPointCloudP2D( p, ref_pc.get_voxel )

    ### Instantiate Optimizer ###
    opt = OptimizationP2D()

    ### OPTION 1:  CREATE A UNIFORM SAMPLING OF THE REFERENCE MESH ###
    match( TARGET_POINT_CLOUD ):
        case 0:
            tar_pc_pts, tar_pc_lbs, _, _ = mesh.create_full_sample_observations( n = 1, p = 300, pad = 300 )
            tar_pc_pts = tar_pc_pts.squeeze()

            tar_pc_pts = transform_pc( tar_pc_pts, get_se3_from_vec6( np.array( [10, 0, 5, 0, 0, 30] ), is_in_degrees = True ) )

            for pt in tar_pc_pts:   tar_pc.add( Point( pt.reshape(( 3, 1 )), ref_pc.get_voxel( pt.reshape(( 3, 1 )) ) ) )

            tar_pc_pts = [ tar_pc_pts ]
            tar_pc_lbs = [ 'Target PC' ] 

        case 1:
            aftr_dict = aftr.from_aftr_frame( 'D:/kc46_sim_collect/full_pointnet/lidar_predictions/frame_0.txt' )

            R = mat.get_dcm( 0, 0, 0 )
            aftr_dict['points'] = ( R @ aftr_dict['points'].T + np.array( [-70, 0, -10] ).reshape(( 3, 1 )) ).T

            aftr_dict = aftr.organize_aftr_frame_by_part( aftr_dict )
            tar_pc_pts = aftr_dict['points']
            tar_pc_lbs = aftr_dict['part_labels']
            for pt_set in tar_pc_pts:
                for pt in pt_set:   tar_pc.add( Point( pt.reshape(( 3, 1 )), ref_pc.get_voxel( pt.reshape(( 3, 1 )) ) ) )

    # Align point cloud
    target_se3 = [ np.eye( 4 ) ]
    target_se3.append( opt.course_align( tar_pc ) )
    target_se3.append( opt.gradient_descent( tar_pc, target_se3[-1], 0.001 ) )

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