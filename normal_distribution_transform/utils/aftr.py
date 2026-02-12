import os
import numpy as np

from typing import Callable

def from_aftr_frame( filepath: str, print_func: Callable[[str], None] = print ) -> dict:

    frame = {
        'points': np.array([]),
        'class_labels': [],
        'part_labels': []
    }

    if( os.path.isfile( filepath ) ):

        with open( filepath, "r" ) as f:
            for l in f:
                l = l.strip()
                start_i = l.find( '(' )
                end_i = l.find( ')' )

                pos = np.array( l[start_i + 1 : end_i].replace( ',', '' ).split( ' ' ), dtype = float )
                labels = l[end_i + 1:].split( ' ' )

                if( frame['points'].size < 1 ): frame['points'] = pos
                else:                           frame['points'] = np.vstack( ( frame['points'], pos ), dtype = float )

                labels.remove( '' )

                if( len( labels ) > 1 ):
                    frame['class_labels'].append( labels[0] )
                    frame['part_labels'].append( labels[1] )
                else:
                    print_func( f"{filepath} is missing either class_labels or part_labels." )

    else:
        print_func( f"{filepath} is not a valid filename." )
    
    
    return frame

def organize_aftr_frame_by_part( aftr_frame: dict, print_func: Callable[[str], None] = print ) -> dict:

    frame = {
        'points': [],
        'part_labels': []
    }

    if( 'points' in aftr_frame.keys() and 'part_labels' in aftr_frame.keys() ):
        part_np = np.array( aftr_frame['part_labels'] )

        for lbl in aftr_frame['part_labels']:
            if( lbl not in frame['part_labels'] ):
                frame['part_labels'].append( lbl )
                ind = np.where( part_np == lbl )
                frame['points'].append( aftr_frame['points'][ind] )

    else:
        print_func( "aftr_frame should be the dictionary output from .from_aftr_frame()" )

    return frame