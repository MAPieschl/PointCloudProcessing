import sys
import os

from ndt.Parameters import Parameters
from ndt.Voxel import VoxelGrid
from se_ndt.LabeledPoint import LabeledPoint
from se_ndt.SemanticTargetPointCloud import SemanticTargetPointCloudP2D
from se_ndt.SemanticReferencePointCloud import SemanticReferencePointCloud
from se_ndt.SemanticOptimization import SemanticOptimizationP2D

from mesh.MeshSampler import MeshSampler

from utils.aftr import *
from utils.mat_ops import *

def main() -> bool:

    NAME = 'kc46_sim_ndt_se'
    DATA_PATH = 'D:/kc46_sim_collect/full_pointnet/lidar_predictions/'
    MESH_PATH = 'mesh/meshes/segmented_kc46_small.obj'
    CLASS_LABEL = 'kc-46'
    OUTPUT_PATH = 'D:/kc46_sim_collect/full_pointnet/'

    ### Create Reference Point Cloud ###
    mesh = MeshSampler( MESH_PATH, CLASS_LABEL, rotation_matrix = np.array( [[ 1, 0, 0 ], [0, 0, -1], [0, 1, 0]] ) )
    labels_vertices = mesh.get_labeled_vertices()
    ref_pc = SemanticReferencePointCloud( pts = np.asarray( labels_vertices['points'] ), labels = labels_vertices['labels'], voxel_size = 10 )

    ### Instantiate Optimizer ###
    opt = SemanticOptimizationP2D()

    ### Create Target Point Cloud ###
    for frame in os.listdir( DATA_PATH ):

        p = Parameters( se3 = np.eye( 4 ) )
        tar_pc = SemanticTargetPointCloudP2D( p )
        ref_pc.resize_voxel_grids( 10 )

        aftr_dict = organize_aftr_frame_by_part( from_aftr_frame( f'{DATA_PATH}{frame}' ) )
        for i in range( len( aftr_dict['part_labels'] ) ):  
            for pt in aftr_dict['points'][i]:
                tar_pc.add( LabeledPoint( pt.reshape( ( 3, 1 ) ), aftr_dict['part_labels'][i], ref_pc.get_weighted_8_nearest_voxels ) )

    #     # Align point cloud
    #     target_se3 = [ tar_pc.get_pose() ]
    #     convergence_steps = opt.course_align( tar_pc )

    #     while( ref_pc.get_voxel_size() >= 1.0 ):
    #         print( f'Beginning iterations with voxel size set to {ref_pc.get_voxel_size()}' )
    #         convergence_steps += opt.levenberg_marquardt( tar_pc, max_iterations = 500 )
    #         ref_pc.build_voxel_grid( ref_pc.get_voxel_size() / 2 )

    #     target_se3 += convergence_steps
    #     ep = np.linalg.inv( target_se3[-1] )
    #     with open( f'{OUTPUT_PATH}{NAME}.txt', 'a' ) as f:

    #         line = f'->lidar virtual 0.00 0.00 {frame} '
    #         line += f'{ep[0][0]} {ep[1][0]} {ep[2][0]} {ep[3][0]} '
    #         line += f'{ep[0][1]} {ep[1][1]} {ep[2][1]} {ep[3][1]} '
    #         line += f'{ep[0][2]} {ep[1][2]} {ep[2][2]} {ep[3][2]} '
    #         line += f'{ep[0][3]} {ep[1][3]} {ep[2][3]} {ep[3][3]}\n'

    #         f.write( line )

    return True

if __name__ == "__main__":

    main()