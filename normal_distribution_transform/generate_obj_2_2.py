import os
import sys

import numpy as np

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *
from utils.stats import *

SIM_PATH = 'D:/test_sets/obj_2/simulated/'
GT_PATH = 'D:/test_sets/obj_2/ground/'
SAVE_PATH = 'E:/AFIT/AAR/5_Thesis/doc/figures/obj_2_2/'

def main():

    sim    : ParsedAftrLog = ParsedAftrLog( SIM_PATH )
    ground : ParsedAftrLog = ParsedAftrLog( GT_PATH )

    paired_timestamps = generate_pose_aligned_timestamps_from_aftr_frames(
        aftrLogs        = [ sim, ground ],
        pose_of         = 'lidar',
        num_samples     = None
    )

    sim_analysis       : AnalyzeAftrLog = AnalyzeAftrLog( sim, 'Virtual LiDAR', 'f-15_model', paired_timestamps[0] )
    ground_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( ground, 'Aeva Atlas LiDAR', 'f-15_model', paired_timestamps[1] )

    paired_timestamps = verify_timestamps_and_filter(
        aftrLogs                    = [ sim_analysis, ground_analysis ],
        paired_timestamps           = paired_timestamps,
        ensure_target_origin_in_FoV = True
    )

    sim_analysis       : AnalyzeAftrLog = AnalyzeAftrLog( sim, 'Virtual LiDAR', 'f-15_model', paired_timestamps[0] )
    ground_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( ground, 'Aeva Atlas LiDAR', 'f-15_model', paired_timestamps[1] )

    sim_analysis.get_segmentation_performance_hist( f'{SAVE_PATH}simulated/hist/' )
    sim_analysis.get_segmentation_performance_plots_by_range( f'{SAVE_PATH}simulated/by_range/' )
    sim_analysis.get_segmentation_performance_plots_by_number_of_points( f'{SAVE_PATH}simulated/by_num_points/' )
    sim_analysis.get_confusion_matrix( f'{SAVE_PATH}simulated/', log_scale = True )
    sim_analysis.get_mIoU_by_dist_angle( f'{SAVE_PATH}simulated/' )

    ground_analysis.get_segmentation_performance_hist( f'{SAVE_PATH}ground/hist/' )
    ground_analysis.get_segmentation_performance_plots_by_range( f'{SAVE_PATH}ground/by_range/' )
    ground_analysis.get_segmentation_performance_plots_by_number_of_points( f'{SAVE_PATH}ground/by_num_points/' )
    ground_analysis.get_confusion_matrix( f'{SAVE_PATH}ground/', log_scale = True )
    ground_analysis.get_mIoU_by_dist_angle( f'{SAVE_PATH}ground/' )
    
    sim_mIoU       : np.ndarray = sim_analysis.get_mIoU_distribution()
    ground_mIoU    : np.ndarray = ground_analysis.get_mIoU_distribution()

    plot_distributions( 
        [ sim_mIoU, ground_mIoU ], 
        [ 'Virtual LiDAR', 'Aeva Atlas LiDAR' ],
        100,
        'Per-sample mIoU Distributions',
        'mIoU',
        'Number of Samples'    
    ).write_image( f'{SAVE_PATH}mIoU_dist.png', width = 1200, height = 600 )

    tab, sym = paired_wilcoxon_signed_rank_test( sim_mIoU, ground_mIoU, 'Virtual LiDAR', 'Aeva Atlas LiDAR' )
    
    tab.to_latex(
        f'{SAVE_PATH}miou_results.tex',
        index = True,
        caption = 'Segmentation results on the Boeing KC-46 using full and vanilla PointNet models.',
        label = 'tab:obj_2_2_wilcoxon',
        column_format = 'L{2in} R{3in}',
        escape = False
    )

    sym.write_image( f'{SAVE_PATH}mIoU_symmetry.png', width = 1200, height = 600 )

if __name__ == '__main__':

    if( not os.path.isdir( SAVE_PATH ) ):
        print( f'{SAVE_PATH} is invalid.' )
    
    elif( not os.path.isdir( SIM_PATH ) ):
        print( f'{SIM_PATH} is invalid.' )
    
    elif( not os.path.isdir( GT_PATH ) ):
        print( f'{GT_PATH} is invalid.' )

    else:

        os.makedirs( f'{SAVE_PATH}simulated/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}simulated/hist/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}simulated/by_range/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}simulated/by_num_points/', exist_ok = True )

        os.makedirs( f'{SAVE_PATH}ground/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}ground/hist/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}ground/by_range/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}ground/by_num_points/', exist_ok = True )

        main()