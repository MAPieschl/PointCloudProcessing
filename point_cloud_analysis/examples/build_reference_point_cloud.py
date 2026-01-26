import sys
import os

import numpy as np

HELP = """Incorrect input arguments. To run program:
\tpython build_reference_point_cloud.py <point_cloud.txt> <offset>
Where point_cloud.txt should have a line-by-line format of:  '(x, y, z) class_label part_label'
and the offset is in the format \"(x,y,z)\"."""

def main( *args ) -> None:
    
    filepath: str = args[0][1]
    offset: str = args[0][2]
    
    try:
        offset = offset.replace( "(", "" )
        offset = offset.replace( ")", "" )
        offset_l: list[str] = offset.split( "," )

        offset_np: np.ndarray = np.array( [ float( offset_l[0] ), float( offset_l[1] ), float( offset_l[2] ) ] )

    except Exception as e:
        print( HELP )
        print( f"Error occurred while parsing offset:\n\t{type(e)}: {e}" )
        return

    if( not os.path.isfile( filepath ) ):
        print( HELP )
        return
    
    newlines: list[str] = []
    with open( filepath, "r" ) as f:
        for line in f:

            try:
                end_idx = line.find( ")" )
                pos_str = line[:end_idx]
                pos_str = pos_str.replace( "(", "" )
                pos_str = pos_str.replace( ")", "" )
                pos_l: list[str] = pos_str.split( "," )

                newlines.append( f"({float( pos_l[0] ) - offset_np[0]:.3f}, {float( pos_l[1] ) - offset_np[1]:.3f}, {float( pos_l[2] ) - offset_np[2]:.3f}{line[end_idx:]}" )

            except Exception as e:
                print( HELP )
                print( f"Error occurred while parsing offset:\n\t{type(e)}: {e}" )
                return

    with open( filepath, "w" ) as f:
        for line in newlines:
            f.write( line )

    print( f"Offset successfully applied to {filepath}." )

if __name__ == "__main__":
    if( len( sys.argv ) < 3 ):
        print( HELP )
    else:
        main( sys.argv )

