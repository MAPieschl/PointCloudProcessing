import os
import sys

import numpy as np

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *
from utils.stats import *

GICP_PATH = 'D:/test_sets/obj_1/full/gicp/'
SE_PT2PL_ICP_PATH = 'D:/test_sets/obj_1/full/se_icp/'
NDT_PATH = 'D:/test_sets/obj_1/full/ndt/'
SE_P2D_NDT_PATH = 'D:/test_sets/obj_1/full/se_ndt/'

SAVE_PATH = 'E:/AFIT/AAR/5_Thesis/doc/figures/obj_1_3/'

NORMALIZE = True

def main():

    gicp    : ParsedAftrLog = ParsedAftrLog( GICP_PATH )
    seg_icp : ParsedAftrLog = ParsedAftrLog( SE_PT2PL_ICP_PATH )
    ndt     : ParsedAftrLog = ParsedAftrLog( NDT_PATH )
    seg_ndt : ParsedAftrLog = ParsedAftrLog( SE_P2D_NDT_PATH )

    paired_timestamps = generate_pose_aligned_timestamps_from_aftr_frames(
        aftrLogs        = [ gicp, seg_icp, ndt, seg_ndt ],
        pose_of         = 'lidar',
        num_samples     = 2000
    )

    gicp_analysis       : AnalyzeAftrLog = AnalyzeAftrLog( gicp, 'GICP', 'kc-46', paired_timestamps[0] )
    seg_icp_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( seg_icp, 'SE-Pt2Pl-ICP', 'kc-46', paired_timestamps[1] )
    ndt_analysis        : AnalyzeAftrLog = AnalyzeAftrLog( ndt, 'NDT', 'kc-46', paired_timestamps[2] )
    seg_ndt_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( seg_ndt, 'SE-P2D-NDT', 'kc-46', paired_timestamps[3] )
    
    gicp_L2_res         = gicp_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    gicp_rot_res        = gicp_analysis.get_rot_residual_distribution()
    seg_icp_L2_res      = seg_icp_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    seg_icp_rot_res     = seg_icp_analysis.get_rot_residual_distribution()
    ndt_L2_res          = ndt_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    ndt_rot_res         = ndt_analysis.get_rot_residual_distribution()
    seg_ndt_L2_res      = seg_ndt_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    seg_ndt_rot_res     = seg_ndt_analysis.get_rot_residual_distribution()

    plot_multi_cdf(
        data    = [ gicp_L2_res, seg_icp_L2_res, ndt_L2_res, seg_ndt_L2_res ],
        labels  = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        title   = 'Registration Translation Error CDF',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
        x_lim   = 0.3,
    ).write_image( f'{SAVE_PATH}l2_error.png', width = 1200, height = 400 )

    plot_distributions(
        data        = [ gicp_L2_res, seg_icp_L2_res, ndt_L2_res, seg_ndt_L2_res ],
        labels      = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        num_bins    = 50,
        title       = 'Registration Translation Error',
        x_label     = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
        x_lim       = ( 0, 0.3 )
    ).write_image( f'{SAVE_PATH}l2_dist.png' )

    plot_multi_cdf(
        data    = [ gicp_rot_res, seg_icp_rot_res, ndt_rot_res, seg_ndt_rot_res ],
        labels  = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        title   = 'Registration Rotation Error CDF',
        x_label = 'Rotation error (deg)',
        x_lim   = 30,
    ).write_image( f'{SAVE_PATH}rot_error.png', width = 1200, height = 400 )

    plot_distributions(
        data        = [ gicp_rot_res, seg_icp_rot_res, ndt_rot_res, seg_ndt_rot_res ],
        labels      = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        num_bins    = 50,
        title       = 'Registration Rotation Error',
        x_label     = 'Rotation error (deg)',
        x_lim       = ( 0, 30 )
    ).write_image( f'{SAVE_PATH}rot_dist.png' )

    get_cdf_percentiles_with_CI(
        data                = [ gicp_L2_res, seg_icp_L2_res, ndt_L2_res, seg_ndt_L2_res ],
        labels              = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        confidence_interval = 0.95,
        num_bootstrap       = 10000,
        percentiles         = [ 50, 75, 90, 95, 99 ],
        units               = '$l_{target}$',
        seed                = 42
    ).to_latex( 
        f'{SAVE_PATH}L2_confidence.tex',
        index = True,
        caption = 'Per-algorithm translation residuals.',
        label = 'tab:obj_1_3_l2_confidence',
        escape = False
    )

    get_cdf_percentiles_with_CI(
        data                = [ gicp_rot_res, seg_icp_rot_res, ndt_rot_res, seg_ndt_rot_res ],
        labels              = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        confidence_interval = 0.95,
        num_bootstrap       = 10000,
        percentiles         = [ 50, 75, 90, 95, 99 ],
        units               = 'deg',
        seed                = 42
    ).to_latex( 
        f'{SAVE_PATH}rot_confidence.tex',
        index = True,
        caption = 'Per-algorithm rotation residuals.',
        label = 'tab:obj_1_3_rot_confidence',
        escape = False
    )

    friedman_chi_square_test(
        data                = [ gicp_L2_res, seg_icp_L2_res, ndt_L2_res, seg_ndt_L2_res ],
        labels              = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        significance        = 0.05
    ).to_latex(
        f'{SAVE_PATH}L2_friedman.tex',
        index = False,
        caption = 'Friedman chi-square test on translation residuals.',
        label = 'tab:obj_1_3_l2_freidman',
        escape = False
    )

    friedman_chi_square_test(
        data                = [ gicp_rot_res, seg_icp_rot_res, ndt_rot_res, seg_ndt_rot_res ],
        labels              = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        significance        = 0.05
    ).to_latex(
        f'{SAVE_PATH}rot_friedman.tex',
        index = False,
        caption = 'Friedman chi-square test on rotation residuals.',
        label = 'tab:obj_1_3_rot_freidman',
        escape = False
    )

    tab, sym = pairwise_wilcoxon_signed_rank_test(
        data                = [ seg_icp_L2_res, gicp_L2_res, ndt_L2_res, seg_ndt_L2_res ],
        labels              = [ '(1)', '(2)', '(3)', '(4)' ],
        significance        = 0.05,
        name                = 'Translation',
        include_pairs       = [( 0, 1 ), ( 0, 2 ), (0, 3) ]
    )

    tab.to_latex(
        f'{SAVE_PATH}L2_wilcoxon.tex',
        index = True,
        caption = 'Paired Wilcoxon signed-rank test on translation residuals after the Holm-Bonferroni correction. Algorithms are (1) SE-Pt2Pl-ICP, (2) GICP, (3) NDT, (4) SE-P2D-NDT.',
        label = 'tab:obj_1_3_l2_wilcoxon',
        escape = False
    )

    sym.write_image( f'{SAVE_PATH}L2_wilcoxon.png', width = 1200, height = 400 )

    tab, sym = pairwise_wilcoxon_signed_rank_test(
        data                = [ seg_icp_rot_res, gicp_rot_res, ndt_rot_res, seg_ndt_rot_res ],
        labels              = [ '(1)', '(2)', '(3)', '(4)' ],
        significance        = 0.05,
        name                = 'Rotation',
        include_pairs       = [( 0, 1 ), ( 0, 2 ), (0, 3) ]
    )

    tab.to_latex(
        f'{SAVE_PATH}rot_wilcoxon.tex',
        index = True,
        caption = 'Paired Wilcoxon signed-rank test on rotation residuals after the Holm-Bonferroni correction. Algorithms are (1) SE-Pt2Pl-ICP, (2) GICP, (3) NDT, (4) SE-P2D-NDT.',
        label = 'tab:obj_1_3_rot_wilcoxon',
        escape = False
    )

    sym.write_image( f'{SAVE_PATH}rot_wilcoxon.png', width = 1200, height = 400 )

if __name__ == '__main__':

    if( not os.path.isdir( SAVE_PATH ) ):           print( f'{SAVE_PATH} is invalid.' )
    elif( not os.path.isdir( GICP_PATH ) ):         print( f'{GICP_PATH} is invalid.' )
    elif( not os.path.isdir( SE_PT2PL_ICP_PATH ) ): print( f'{SE_PT2PL_ICP_PATH} is invalid.' )
    elif( not os.path.isdir( NDT_PATH ) ):          print( f'{NDT_PATH} is invalid.' )
    elif( not os.path.isdir( SE_P2D_NDT_PATH ) ):   print( f'{SE_P2D_NDT_PATH} is invalid.' )

    else:                                           main()