import sys
import os

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *

def main( *args ) -> bool:
    
    if( not os.path.isdir( args[0][1] ) ): return False
    if( not os.path.isdir( args[0][3] ) ): return False

    aftr_log = ParsedAftrLog( args[0][1] )
    analysis = AnalyzeAftrLog( aftr_log, args[0][0], args[0][2]  )

    analysis.get_6DOF_residual_scatter_plots_by_distance( args[0][3] )
    analysis.get_segmentation_performance_hist( args[0][3] )
    analysis.get_segmentation_performance_plots_by_range( args[0][3] )
    analysis.get_timing_info( args[0][3] )

    print( analysis.produce_minimized_extrinsic_calibration_for_lidar() )

    return True

if __name__ == "__main__":

    HELP_STR = 'python run_NDT.py name_of_your_test path_to_your_data_directory target_class_label path_to_graph_target'

    if( sys.argv[1] == '-h' or sys.argv[1] == '--help' ):
        print( HELP_STR )
    
    else:
        if( not main( sys.argv[1:] ) ):
            print( HELP_STR )