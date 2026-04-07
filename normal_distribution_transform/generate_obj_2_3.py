import os
import sys
import pickle

import numpy as np

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *
from utils.stats import *

SIM_PATH = 'D:/test_sets/obj_2/simulated/'
SIM_TRUTH_PATH = 'D:/test_sets/obj_2/simulated_truth/'
GT_PATH = 'D:/test_sets/obj_2/ground/'
GT_TRUTH_PATH = 'D:/test_sets/obj_2/ground_truth/'
GT_GICP_PATH = 'D:/test_sets/obj_2/ground_gicp/'
SAVE_PATH = 'E:/AFIT/AAR/5_Thesis/doc/figures/obj_2_3/'
TEMP_PATH = 'E:/repos/PointCloudProcessing/normal_distribution_transform/temp/'

NORMALIZE = True

def main():

    ### SETUP ###

    if( not os.path.isfile( f'{TEMP_PATH}obj_2_3_sets.pkl' ) ):
        sim         : ParsedAftrLog = ParsedAftrLog( SIM_PATH )
        sim_truth   : ParsedAftrLog = ParsedAftrLog( SIM_TRUTH_PATH )
        ground      : ParsedAftrLog = ParsedAftrLog( GT_PATH )
        gd_truth    : ParsedAftrLog = ParsedAftrLog( GT_TRUTH_PATH )
        gd_gicp     : ParsedAftrLog = ParsedAftrLog( GT_GICP_PATH )

        paired_timestamps = generate_pose_aligned_timestamps_from_aftr_frames(
            aftrLogs        = [ sim, sim_truth, ground, gd_truth, gd_gicp ],
            pose_of         = 'lidar',
            num_samples     = None
        )

        sim_analysis        : AnalyzeAftrLog = AnalyzeAftrLog( sim, 'Virtual LiDAR', 'f-15_model', paired_timestamps[0] )
        sim_truth_analysis  : AnalyzeAftrLog = AnalyzeAftrLog( sim_truth, 'Virtual LiDAR (Truth Labels)', 'f-15_model', paired_timestamps[1] )
        ground_analysis     : AnalyzeAftrLog = AnalyzeAftrLog( ground, 'Aeva Atlas LiDAR', 'f-15_model', paired_timestamps[2] )
        gd_truth_analysis   : AnalyzeAftrLog = AnalyzeAftrLog( gd_truth, 'Aeva Atlas (Truth Labels)', 'f-15_model', paired_timestamps[3] )
        gd_gicp_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( gd_gicp, 'Aeva Atlas (GICP)', 'f-15_model', paired_timestamps[4] )

        paired_timestamps = verify_timestamps_and_filter(
            aftrLogs                    = [ sim_analysis, sim_truth_analysis, ground_analysis, gd_truth_analysis, gd_gicp_analysis ],
            paired_timestamps           = paired_timestamps,
            ensure_target_origin_in_FoV = True
        )

        sim_analysis       : AnalyzeAftrLog = AnalyzeAftrLog( sim, 'Virtual LiDAR', 'f-15_model', paired_timestamps[0] )
        sim_truth_analysis : AnalyzeAftrLog = AnalyzeAftrLog( sim_truth, 'Virtual LiDAR (Truth Labels)', 'f-15_model', paired_timestamps[1] )
        ground_analysis    : AnalyzeAftrLog = AnalyzeAftrLog( ground, 'Aeva Atlas LiDAR', 'f-15_model', paired_timestamps[2] )
        gd_truth_analysis  : AnalyzeAftrLog = AnalyzeAftrLog( gd_truth, 'Aeva Atlas (Truth Labels)', 'f-15_model', paired_timestamps[3] )
        gd_gicp_analysis   : AnalyzeAftrLog = AnalyzeAftrLog( gd_gicp, 'Aeva Atlas (GICP)', 'f-15_model', paired_timestamps[4] )

        with open( f'{TEMP_PATH}obj_2_3_sets.pkl', 'wb' ) as p:
            pickle.dump( 
                obj     = { 
                    'sim':              sim_analysis, 
                    'sim_truth':        sim_truth_analysis, 
                    'ground':           ground_analysis, 
                    'ground_truth':     gd_truth_analysis, 
                    'ground_gicp':      gd_gicp_analysis,
                    'paired_ts':        paired_timestamps
                },
                file    = p 
            )

    else:
        with open( f'{TEMP_PATH}obj_2_3_sets.pkl', 'rb' ) as p:
            t : dict = pickle.load( p )

            sim_analysis        = t['sim']
            sim_truth_analysis  = t['sim_truth']
            ground_analysis     = t['ground']
            gd_truth_analysis   = t['ground_truth']
            gd_gicp_analysis    = t['ground_gicp']
            paired_timestamps   = t['paired_ts']

    paired_ts_geq75 = find_timestamps_parallel_to( 
        paired_timestamps       = paired_timestamps, 
        paired_timestamp_index  = 2,  
        timestamps              = ground_analysis.get_timestamps_above_mIoU( 0.75 )
    )

    paired_ts_geq50 = find_timestamps_parallel_to( 
        paired_timestamps       = paired_timestamps, 
        paired_timestamp_index  = 2,  
        timestamps              = ground_analysis.get_timestamps_above_mIoU( 0.50 )
    )

    paired_ts_geq25 = find_timestamps_parallel_to( 
        paired_timestamps       = paired_timestamps, 
        paired_timestamp_index  = 2,  
        timestamps              = ground_analysis.get_timestamps_above_mIoU( 0.25 )
    )
    
    ### BASIC RESIDUAL CDF ###

    sim_truth_L2_res         = sim_truth_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    sim_truth_rot_res        = sim_truth_analysis.get_rot_residual_distribution()
    ground_truth_L2_res      = gd_truth_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    ground_truth_rot_res     = gd_truth_analysis.get_rot_residual_distribution()
    ground_L2_res            = ground_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    ground_rot_res           = ground_analysis.get_rot_residual_distribution()

    plot_multi_cdf(
        data    = [ sim_truth_L2_res, ground_truth_L2_res ],
        labels  = [ 'Virtual', 'Aeva Atlas' ],
        title   = 'Registration Translation Error CDF using Truth Point Labels',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}L2_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ sim_truth_rot_res, ground_truth_rot_res ],
        labels  = [ 'Virtual', 'Aeva Atlas' ],
        title   = 'Registration Rotation Error CDF using Truth Point Labels',
        x_label = 'Rotation error (deg)',
    ).write_image( f'{SAVE_PATH}rot_error.png', width = 1200, height = 400 )

    ### PAIRED WILCOXON TESTS ###

    tab, sym = paired_translation_rotation_wilcoxon_signed_rank_test(
        sample_a_L2         = sim_truth_L2_res,
        sample_b_L2         = ground_truth_L2_res,
        sample_a_rot        = sim_truth_rot_res,
        sample_b_rot        = ground_truth_rot_res,
        name_a              = 'Virtual',
        name_b              = 'Aeva Atlas',
        significance        = 0.05
    )

    tab.to_latex(
        f'{SAVE_PATH}wilcoxon.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on residuals.',
        label = 'tab:obj_2_3_wilcoxon',
        escape = False
    )
    
    sym.write_image( f'{SAVE_PATH}wilcoxon.png', width = 1200, height = 400 )

    ### VARIABLE SEGMENTATION PERFORMANCE CDF ###
    gd_truth_75_L2_res     = gd_truth_analysis.get_L2_residual_distribution( paired_ts_geq75[3],  normalize = NORMALIZE )
    gd_truth_75_rot_res    = gd_truth_analysis.get_rot_residual_distribution( paired_ts_geq75[3] )
    gd_75_L2_res           = ground_analysis.get_L2_residual_distribution( paired_ts_geq75[2],  normalize = NORMALIZE )
    gd_75_rot_res          = ground_analysis.get_rot_residual_distribution( paired_ts_geq75[2] )
    gd_gicp_75_L2_res      = gd_gicp_analysis.get_L2_residual_distribution( paired_ts_geq75[4],  normalize = NORMALIZE )
    gd_gicp_75_rot_res     = gd_gicp_analysis.get_rot_residual_distribution( paired_ts_geq75[4] )
    
    gd_truth_50_L2_res     = gd_truth_analysis.get_L2_residual_distribution( paired_ts_geq50[3],  normalize = NORMALIZE )
    gd_truth_50_rot_res    = gd_truth_analysis.get_rot_residual_distribution( paired_ts_geq50[3] )
    gd_50_L2_res           = ground_analysis.get_L2_residual_distribution( paired_ts_geq50[2],  normalize = NORMALIZE )
    gd_50_rot_res          = ground_analysis.get_rot_residual_distribution( paired_ts_geq50[2] )
    gd_gicp_50_L2_res      = gd_gicp_analysis.get_L2_residual_distribution( paired_ts_geq50[4],  normalize = NORMALIZE )
    gd_gicp_50_rot_res     = gd_gicp_analysis.get_rot_residual_distribution( paired_ts_geq50[4] )
    
    gd_truth_25_L2_res     = gd_truth_analysis.get_L2_residual_distribution( paired_ts_geq25[3],  normalize = NORMALIZE )
    gd_truth_25_rot_res    = gd_truth_analysis.get_rot_residual_distribution( paired_ts_geq25[3] )
    gd_25_L2_res           = ground_analysis.get_L2_residual_distribution( paired_ts_geq25[2],  normalize = NORMALIZE )
    gd_25_rot_res          = ground_analysis.get_rot_residual_distribution( paired_ts_geq25[2] )
    gd_gicp_25_L2_res      = gd_gicp_analysis.get_L2_residual_distribution( paired_ts_geq25[4],  normalize = NORMALIZE )
    gd_gicp_25_rot_res     = gd_gicp_analysis.get_rot_residual_distribution( paired_ts_geq25[4] )

    gd_truth_L2_res        = gd_truth_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    gd_truth_rot_res       = gd_truth_analysis.get_rot_residual_distribution()
    gd_L2_res              = ground_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    gd_rot_res             = ground_analysis.get_rot_residual_distribution()
    gd_gicp_L2_res         = gd_gicp_analysis.get_L2_residual_distribution( normalize = NORMALIZE )
    gd_gicp_rot_res        = gd_gicp_analysis.get_rot_residual_distribution()

    tab, sym = paired_translation_rotation_wilcoxon_signed_rank_test( 
        sample_a_L2     = gd_truth_75_L2_res,
        sample_b_L2     = gd_75_L2_res,
        sample_a_rot    = gd_truth_75_rot_res,
        sample_b_rot    = gd_75_rot_res,
        name_a          = 'SE-Pt2Pl-ICP (Truth Labels)',
        name_b          = 'SE-Pt2Pl-ICP (PointNet)',
        significance    = 0.05
    )
    tab.to_latex(
        f'{SAVE_PATH}wilcoxon_75.tex',
        index = True,
        caption = f'Paired Wilcoxon Signed-Rank test on residuals when mIoU >= 0.75.',
        label = 'tab:obj_2_3_wilcoxon_75',
        escape = False
    )
    sym.write_image(f'{SAVE_PATH}wilcoxon_75_sym.png', width = 1200, height = 400 )

    tab, sym = paired_translation_rotation_wilcoxon_signed_rank_test( 
        sample_a_L2     = gd_truth_50_L2_res,
        sample_b_L2     = gd_50_L2_res,
        sample_a_rot    = gd_truth_50_rot_res,
        sample_b_rot    = gd_50_rot_res,
        name_a          = 'SE-Pt2Pl-ICP (Truth Labels)',
        name_b          = 'SE-Pt2Pl-ICP (PointNet)',
        significance    = 0.05
    )
    tab.to_latex(
        f'{SAVE_PATH}wilcoxon_50.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on residuals when mIoU >= 0.50.',
        label = 'tab:obj_2_3_wilcoxon_50',
        escape = False
    )
    sym.write_image(f'{SAVE_PATH}wilcoxon_50_sym.png', width = 1200, height = 400 )

    tab, sym = paired_translation_rotation_wilcoxon_signed_rank_test( 
        sample_a_L2     = gd_truth_25_L2_res,
        sample_b_L2     = gd_25_L2_res,
        sample_a_rot    = gd_truth_25_rot_res,
        sample_b_rot    = gd_25_rot_res,
        name_a          = 'SE-Pt2Pl-ICP (Truth Labels)',
        name_b          = 'SE-Pt2Pl-ICP (PointNet)',
        significance    = 0.05
    )
    tab.to_latex(
        f'{SAVE_PATH}wilcoxon_25.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on residuals when mIoU >= 0.25.',
        label = 'tab:obj_2_3_wilcoxon_25',
        escape = False
    )
    sym.write_image(f'{SAVE_PATH}wilcoxon_25_sym.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_75_L2_res, gd_gicp_75_L2_res, gd_75_L2_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration translation error CDF on samples where SE-Pt2Pl-ICP achieved >= 75% mIoU | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( paired_ts_geq75[2] ) ):.3f}',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}L2_miou_75_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_75_rot_res, gd_gicp_75_rot_res, gd_75_rot_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration rotation error CDF on samples where SE-Pt2Pl-ICP achieved >= 75% mIoU | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( paired_ts_geq75[2] ) ):.3f}',
        x_label = 'Rotation error (deg)',
    ).write_image( f'{SAVE_PATH}rot_miou_75_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_50_L2_res, gd_gicp_50_L2_res, gd_50_L2_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration translation error CDF on samples where SE-Pt2Pl-ICP achieved >= 50% mIoU | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( paired_ts_geq50[2] ) ):.3f}',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}L2_miou_50_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_50_rot_res, gd_gicp_50_rot_res, gd_50_rot_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration rotation error CDF on samples where SE-Pt2Pl-ICP achieved >= 50% mIoU | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( paired_ts_geq50[2] ) ):.3f}',
        x_label = 'Rotation error (deg)',
    ).write_image( f'{SAVE_PATH}rot_miou_50_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_25_L2_res, gd_gicp_25_L2_res, gd_25_L2_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration translation error CDF on samples where SE-Pt2Pl-ICP achieved >= 25% mIoU | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( paired_ts_geq25[2] ) ):.3f}',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}L2_miou_25_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_25_rot_res, gd_gicp_25_rot_res, gd_25_rot_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration rotation error CDF on samples where SE-Pt2Pl-ICP achieved >= 25% mIoU | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( paired_ts_geq25[2] ) ):.3f}',
        x_label = 'Rotation error (deg)',
    ).write_image( f'{SAVE_PATH}rot_miou_25_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_L2_res, gd_gicp_L2_res, gd_L2_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration translation error CDF on all samples | median mIoU = {np.median( ground_analysis.get_mIoU_distribution() ):.3f}',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}L2_miou_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_rot_res, gd_gicp_rot_res, gd_rot_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration rotation error CDF on all samples | median mIoU = {np.median( ground_analysis.get_mIoU_distribution() ):.3f}',
        x_label = 'Rotation error (deg)',
    ).write_image( f'{SAVE_PATH}rot_miou_error.png', width = 1200, height = 400 )

    filtered_by_dist = find_timestamps_parallel_to( 
        paired_timestamps       = paired_timestamps, 
        paired_timestamp_index  = 2,  
        timestamps              = ground_analysis.get_timestamps_between_range_inclusive( distance_range = ( 4.5, 100 ) )
    )

    gd_truth_filt_L2_res     = gd_truth_analysis.get_L2_residual_distribution( filtered_by_dist[3],  normalize = NORMALIZE )
    gd_truth_filt_rot_res    = gd_truth_analysis.get_rot_residual_distribution( filtered_by_dist[3] )
    gd_filt_L2_res           = ground_analysis.get_L2_residual_distribution( filtered_by_dist[2],  normalize = NORMALIZE )
    gd_filt_rot_res          = ground_analysis.get_rot_residual_distribution( filtered_by_dist[2] )
    gd_gicp_filt_L2_res      = gd_gicp_analysis.get_L2_residual_distribution( filtered_by_dist[4],  normalize = NORMALIZE )
    gd_gicp_filt_rot_res     = gd_gicp_analysis.get_rot_residual_distribution( filtered_by_dist[4] )

    plot_multi_cdf(
        data    = [ gd_truth_filt_L2_res, gd_gicp_filt_L2_res, gd_filt_L2_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration translation error CDF on all samples where LiDAR is >= 4.5m from target origin | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( filtered_by_dist[2] ) ):.3f}',
        x_label = '(L2 error) / (body length)' if NORMALIZE else 'L2 error (m)',
    ).write_image( f'{SAVE_PATH}L2_miou_filt_error.png', width = 1200, height = 400 )

    plot_multi_cdf(
        data    = [ gd_truth_filt_rot_res, gd_gicp_filt_rot_res, gd_filt_rot_res ],
        labels  = [ 'Truth Labels', 'GICP (No Labels)', 'Predicted Labels' ],
        title   = f'Registration rotation error CDF on all samples where LiDAR is >= 4.5m from target origin | median mIoU = {np.median( ground_analysis.get_mIoU_distribution( filtered_by_dist[2] ) ):.3f}',
        x_label = 'Rotation error (deg)',
    ).write_image( f'{SAVE_PATH}rot_miou_filt_error.png', width = 1200, height = 400 )

    tab, sym = pairwise_wilcoxon_signed_rank_test(
        data            = [ gd_filt_L2_res, gd_truth_filt_L2_res, gd_gicp_filt_L2_res ],
        labels          = [ 'Predicted Labels', 'Truth Labels', 'GICP (No Labels)' ],
        significance    = 0.05,
        name            = 'Filtered translation residuals',
        include_pairs   = [(0, 1), (0, 2)]
    )

    tab.to_latex(
        f'{SAVE_PATH}L2_wilcoxon_filt.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on translation residuals for the filtered data. Datasets are (1) SE-Pt2Pl-ICP (predicted labels), (2) SE-Pt2Pl-ICP (truth labels), (3) GICP',
        label = 'tab:obj_2_3_L2_wilcoxon_filt',
        escape = False
    )
    
    sym.write_image( f'{SAVE_PATH}L2_wilcoxon_filt.png', width = 1200, height = 400 )

    tab, sym = pairwise_wilcoxon_signed_rank_test(
        data            = [ gd_filt_rot_res, gd_truth_filt_rot_res, gd_gicp_filt_rot_res ],
        labels          = [ 'Predicted Labels', 'Truth Labels', 'GICP (No Labels)' ],
        significance    = 0.05,
        name            = 'Filtered rotation residuals',
        include_pairs   = [(0, 1), (0, 2)]
    )

    tab.to_latex(
        f'{SAVE_PATH}rot_wilcoxon_filt.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on rotation residuals for the filtered data. Datasets are (1) SE-Pt2Pl-ICP (predicted labels), (2) SE-Pt2Pl-ICP (truth labels), (3) GICP',
        label = 'tab:obj_2_3_rot_wilcoxon_filt',
        escape = False
    )
    
    sym.write_image( f'{SAVE_PATH}rot_wilcoxon_filt.png', width = 1200, height = 400 )

    
    tab, sym = pairwise_wilcoxon_signed_rank_test(
        data            = [ gd_L2_res, gd_truth_L2_res, gd_gicp_L2_res ],
        labels          = [ 'Predicted Labels', 'Truth Labels', 'GICP (No Labels)' ],
        significance    = 0.05,
        name            = 'Translation residuals',
        include_pairs   = [(0, 1), (0, 2)]
    )

    tab.to_latex(
        f'{SAVE_PATH}L2_wilcoxon_full.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on translation residuals for the full data set. Datasets are (1) SE-Pt2Pl-ICP (predicted labels), (2) SE-Pt2Pl-ICP (truth labels), (3) GICP',
        label = 'tab:obj_2_3_L2_wilcoxon_full',
        escape = False
    )
    
    sym.write_image( f'{SAVE_PATH}L2_wilcoxon_full.png', width = 1200, height = 400 )

    tab, sym = pairwise_wilcoxon_signed_rank_test(
        data            = [ gd_rot_res, gd_truth_rot_res, gd_gicp_rot_res ],
        labels          = [ 'Predicted Labels', 'Truth Labels', 'GICP (No Labels)' ],
        significance    = 0.05,
        name            = 'Rotation residuals',
        include_pairs   = [(0, 1), (0, 2)]
    )

    tab.to_latex(
        f'{SAVE_PATH}rot_wilcoxon_full.tex',
        index = True,
        caption = 'Paired Wilcoxon Signed-Rank test on rotation residuals for the full data set. Datasets are (1) SE-Pt2Pl-ICP (predicted labels), (2) SE-Pt2Pl-ICP (truth labels), (3) GICP',
        label = 'tab:obj_2_3_rot_wilcoxon_full',
        escape = False
    )
    
    sym.write_image( f'{SAVE_PATH}rot_wilcoxon_full.png', width = 1200, height = 400 )

    get_cdf_percentiles_with_CI(
        data                = [ gd_truth_L2_res, ground_L2_res, gd_filt_L2_res ],
        labels              = [ 'Aeva (truth labels)', 'Aeva', 'Aeva (range filtered)' ],
        confidence_interval = 0.95,
        num_bootstrap       = 10000,
        percentiles         = [ 50, 75, 90, 95, 99 ],
        units               = '$l_{target}$' if NORMALIZE else 'm',
        seed                = 42
    ).to_latex( 
        f'{SAVE_PATH}L2_confidence.tex',
        index = True,
        caption = 'Translation residuals by point cloud source.',
        label = 'tab:obj_2_3_l2_confidence',
        escape = False
    )

    get_cdf_percentiles_with_CI(
        data                = [ gd_truth_rot_res, ground_rot_res, gd_filt_rot_res ],
        labels              = [ 'Aeva (truth labels)', 'Aeva', 'Aeva (range filtered)' ],
        confidence_interval = 0.95,
        num_bootstrap       = 10000,
        percentiles         = [ 50, 75, 90, 95, 99 ],
        units               = 'deg',
        seed                = 42
    ).to_latex( 
        f'{SAVE_PATH}rot_confidence.tex',
        index = True,
        caption = 'Rotation residuals by point cloud source.',
        label = 'tab:obj_2_3_rot_confidence',
        escape = False
    )

    ground_analysis.get_translation_error_by_dist_angle( SAVE_PATH, normalize = True )
    ground_analysis.get_rotation_error_by_dist_angle( SAVE_PATH, normalize = True )

if __name__ == '__main__':

    if( not os.path.isdir( SAVE_PATH ) ):           print( f'{SAVE_PATH} is invalid.' )
    elif( not os.path.isdir( SIM_PATH ) ):          print( f'{SIM_PATH} is invalid.' )
    elif( not os.path.isdir( GT_PATH ) ):           print( f'{GT_PATH} is invalid.' )

    else:                                           main()