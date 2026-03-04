import sys
import os

from ndt.Parameters import Parameters
from ndt.Point import Point
from ndt.Voxel import VoxelGrid
from ndt.TargetPointCloud import TargetPointCloudP2D
from ndt.Optimization import OptimizationP2D

from mesh.MeshSampler import MeshSampler

from utils.aftr import *
from utils.mat_ops import *

def main() -> bool:

    NAME = 'kc46_sim_ndt'
    DATA_PATH = 'D:/kc46_sim_collect/full_pointnet/lidar_predictions/'
    MESH_PATH = 'mesh/meshes/segmented_kc46_small.obj'
    CLASS_LABEL = 'kc-46'
    OUTPUT_PATH = 'D:/kc46_sim_collect/full_pointnet/'

    ### Create Reference Point Cloud ###
    mesh = MeshSampler( MESH_PATH, CLASS_LABEL, rotation_matrix = np.array( [[ 1, 0, 0 ], [0, 0, -1], [0, 1, 0]] ) )
    ref_pc = VoxelGrid( pts = np.asarray( mesh.mesh.vertices ), voxel_size = 10 )

    ### Instantiate Optimizer ###
    opt = OptimizationP2D()

    ### Create Target Point Cloud ###
    for frame in os.listdir( DATA_PATH ):

        p = Parameters( se3 = np.eye( 4 ) )
        tar_pc = TargetPointCloudP2D( p )
        ref_pc.build_voxel_grid( 10 )

        aftr_dict = from_aftr_frame( f'{DATA_PATH}{frame}' )
        tar_pc_pts = [ aftr_dict['points'] ]
        for pt in tar_pc_pts[0]:  tar_pc.add( Point( pt.reshape( ( 3, 1 ) ), ref_pc.get_weighted_8_nearest_voxels ) )

        # Align point cloud
        target_se3 = [ tar_pc.get_pose() ]
        convergence_steps = opt.coarse_align( tar_pc )

        while( ref_pc.get_voxel_size() >= 1.0 ):
            print( f'Beginning iterations with voxel size set to {ref_pc.get_voxel_size()}' )
            convergence_steps += opt.levenberg_marquardt( tar_pc, max_iterations = 500 )
            ref_pc.build_voxel_grid( ref_pc.get_voxel_size() / 2 )

        target_se3 += convergence_steps
        ep = np.linalg.inv( target_se3[-1] )
        with open( f'{OUTPUT_PATH}{NAME}.txt', 'a' ) as f:

            line = f'->lidar virtual 0.00 0.00 {frame} '
            line += f'{ep[0][0]} {ep[1][0]} {ep[2][0]} {ep[3][0]} '
            line += f'{ep[0][1]} {ep[1][1]} {ep[2][1]} {ep[3][1]} '
            line += f'{ep[0][2]} {ep[1][2]} {ep[2][2]} {ep[3][2]} '
            line += f'{ep[0][3]} {ep[1][3]} {ep[2][3]} {ep[3][3]}\n'

            f.write( line )

    return True

if __name__ == "__main__":

    main()