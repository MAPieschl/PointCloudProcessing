import sys
import os
import json

from ndt.Parameters import Parameters
from ndt.Point import Point
from ndt.Voxel import Voxel
from ndt.ReferencePointCloud import ReferencePointCloud
from ndt.TargetPointCloud import TargetPointCloud
from ndt.Optimization import Optimization

from mesh.MeshSampler import MeshSampler

from utils.aftr import *
from utils.mat_ops import *
from utils.plotting import *

def main( *args ) -> bool:
    
    if( not os.path.isdir( args[1] ) ): return False

    results = {
        'actual_pos': [],
        'estimated_pos': [],
        'actual_rpy': [],
        'estimated_pose': [],
        'residual_pos': [],
        'residual_rpy': [],
        'initial_pos': [],
        'initial_rpy': [],
        'num_points': []
    }

    

    with open( f'{args[0]}.json', 'w' ) as w:
        json.dump( results, w )

    return True

if __name__ == "__main__":

    HELP_STR = 'python run_NDT.py name_of_your_test path_to_your_data'

    if( sys.argv[1] == '-h' or sys.argv[1] == '--help' ):
        print( HELP_STR )
    
    else:
        if( not main( sys.argv[1:] ) ):
            print( HELP_STR )