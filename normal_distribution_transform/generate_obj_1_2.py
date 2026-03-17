import os
import sys

import numpy as np

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *
from utils.stats import *

FULL_PATH = 'D:/test_sets/kc46_full_pointnet/seg_gicp/'
VANILLA_PATH = 'D:/test_sets/kc46_vanilla_pointnet/seg_ndt/'
SAVE_PATH = 'E:/AFIT/AAR/5_Thesis/doc/figures/obj_1_2/'

def main():

    full    : ParsedAftrLog = ParsedAftrLog( FULL_PATH )
    vanilla : ParsedAftrLog = ParsedAftrLog( VANILLA_PATH )

    full_analysis       : AnalyzeAftrLog = AnalyzeAftrLog( full, 'Full PointNet', 'kc-46' )
    vanilla_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( vanilla, 'Vanilla PointNet', 'kc-46' )

    full_analysis.get_segmentation_performance_hist( f'{SAVE_PATH}full/hist/' )
    full_analysis.get_segmentation_performance_plots_by_range( f'{SAVE_PATH}full/by_range/' )
    full_analysis.get_segmentation_performance_plots_by_number_of_points( f'{SAVE_PATH}full/by_num_points/' )

    vanilla_analysis.get_segmentation_performance_hist( f'{SAVE_PATH}vanilla/hist/' )
    vanilla_analysis.get_segmentation_performance_plots_by_range( f'{SAVE_PATH}vanilla/by_range/' )
    vanilla_analysis.get_segmentation_performance_plots_by_number_of_points( f'{SAVE_PATH}vanilla/by_num_points/' )
    
    full_mIoU       : np.ndarray = full_analysis.get_mIoU_distribution()
    vanilla_mIoU    : np.ndarray = vanilla_analysis.get_mIoU_distribution()

    plot_distributions( 
        [ full_mIoU, vanilla_mIoU ], 
        [ 'Full PointNet', 'Vanilla PointNet' ],
        100,
        'Per-sample mIoU Distributions',
        'mIoU',
        'Number of Samples'    
    ).write_image( f'{SAVE_PATH}mIoU_dist.png', width = 1200, height = 600 )

    wilcoxon_signed_rank_test( full_mIoU, vanilla_mIoU, 'Full PointNet', 'Vanilla PointNet' ).to_latex(
        f'{SAVE_PATH}miou_results.tex',
        index = False,
        caption = 'Segmentation results on the Boeing KC-46 using full and vanilla PointNet models.',
        label = 'tab:obj_1_2_wilcoxon',
        column_format = 'L{2in} R{3in}',
        escape = False
    )

    full_timing     : np.ndarray = full_analysis.get_inference_timing_distribution()
    vanilla_timing  : np.ndarray = vanilla_analysis.get_inference_timing_distribution()

    plot_distributions( 
        [ full_timing, vanilla_timing ], 
        [ 'Full PointNet', 'Vanilla PointNet' ],
        100,
        'Per-sample Inference Time Distributions',
        'time (ms)',
        'Number of Samples'    
    ).write_image( f'{SAVE_PATH}time_dist.png', width = 1200, height = 600 )

    wilcoxon_signed_rank_test( full_timing, vanilla_timing, 'Full PointNet', 'Vanilla PointNet' ).to_latex(
        f'{SAVE_PATH}timing_results.tex',
        index = False,
        caption = 'Inference time on the Boeing KC-46 using full and vanilla PointNet models.',
        label = 'tab:obj_1_2_timing',
        column_format = 'L{2in} R{3in}',
        escape = False
    )


if __name__ == '__main__':

    if( not os.path.isdir( SAVE_PATH ) ):
        print( f'{SAVE_PATH} is invalid.' )
    
    elif( not os.path.isdir( FULL_PATH ) ):
        print( f'{FULL_PATH} is invalid.' )
    
    elif( not os.path.isdir( VANILLA_PATH ) ):
        print( f'{VANILLA_PATH} is invalid.' )

    else:

        os.makedirs( f'{SAVE_PATH}full/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}full/hist/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}full/by_range/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}full/by_num_points/', exist_ok = True )

        os.makedirs( f'{SAVE_PATH}vanilla/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}vanilla/hist/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}vanilla/by_range/', exist_ok = True )
        os.makedirs( f'{SAVE_PATH}vanilla/by_num_points/', exist_ok = True )

        main()