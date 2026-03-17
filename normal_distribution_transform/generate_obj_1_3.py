import os
import sys

import numpy as np

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *
from utils.stats import *

GICP_PATH = 'D:/test_sets/kc46_full_pointnet/ndt/' ### FIX
SE_PT2PL_ICP_PATH = 'D:/test_sets/kc46_full_pointnet/seg_gicp/'
NDT_PATH = 'D:/test_sets/kc46_full_pointnet/ndt/'
SE_P2D_NDT_PATH = 'D:/test_sets/kc46_full_pointnet/seg_gicp/' ### FIX

SAVE_PATH = 'E:/AFIT/AAR/5_Thesis/doc/figures/obj_1_3/'

def main():

    gicp    : ParsedAftrLog = ParsedAftrLog( GICP_PATH )
    seg_icp : ParsedAftrLog = ParsedAftrLog( SE_PT2PL_ICP_PATH )
    ndt     : ParsedAftrLog = ParsedAftrLog( NDT_PATH )
    seg_ndt : ParsedAftrLog = ParsedAftrLog( SE_P2D_NDT_PATH )

    gicp_analysis       : AnalyzeAftrLog = AnalyzeAftrLog( gicp, 'GICP', 'kc-46' )
    seg_icp_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( seg_icp, 'SE-Pt2Pl-ICP', 'kc-46' )
    ndt_analysis        : AnalyzeAftrLog = AnalyzeAftrLog( ndt, 'NDT', 'kc-46' )
    seg_ndt_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( seg_ndt, 'SE-P2D-NDT', 'kc-46' )
    
    gicp_L2_res         = gicp_analysis.get_L2_residual_distribution()
    gicp_rot_res        = gicp_analysis.get_rot_residual_distribution()
    seg_icp_L2_res      = seg_icp_analysis.get_L2_residual_distribution()
    seg_icp_rot_res     = seg_icp_analysis.get_rot_residual_distribution()
    ndt_L2_res          = ndt_analysis.get_L2_residual_distribution()
    ndt_rot_res         = ndt_analysis.get_rot_residual_distribution()
    seg_ndt_L2_res      = seg_ndt_analysis.get_L2_residual_distribution()
    seg_ndt_rot_res     = seg_ndt_analysis.get_rot_residual_distribution()

    plot_multi_cdf(
        data    = [ gicp_L2_res, seg_icp_L2_res, ndt_L2_res, seg_ndt_L2_res ],
        labels  = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        title   = 'Registration Translation Error CDF',
        x_label = 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}l2_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gicp_rot_res, seg_icp_rot_res, ndt_rot_res, seg_ndt_rot_res ],
        labels  = [ 'GICP', 'SE-Pt2Pl-ICP', 'NDT', 'SE-P2D-NDT' ],
        title   = 'Registration Rotation Error CDF',
        x_label = 'Rotation error (rad)',
    ).write_image( f'{SAVE_PATH}rot_error.png', width = 1200, height = 400 )

if __name__ == '__main__':

    if( not os.path.isdir( SAVE_PATH ) ):           print( f'{SAVE_PATH} is invalid.' )
    elif( not os.path.isdir( GICP_PATH ) ):         print( f'{GICP_PATH} is invalid.' )
    elif( not os.path.isdir( SE_PT2PL_ICP_PATH ) ): print( f'{SE_PT2PL_ICP_PATH} is invalid.' )
    elif( not os.path.isdir( NDT_PATH ) ):          print( f'{NDT_PATH} is invalid.' )
    elif( not os.path.isdir( SE_P2D_NDT_PATH ) ):   print( f'{SE_P2D_NDT_PATH} is invalid.' )

    else:                                           main()