from dependencies import *
import utils.globals as globals

class OptiTrack:
    def __init__(
        self,
        object_R: dict[str, np.ndarray] = {
            'corner_reflector': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]),
            'mmwave': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]),
            'lidar': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ])
        },
        print_func: Callable[[str], None] = print
    ):

        self._print = print_func
        self._object_R = object_R

    def parse_log( self, path: str ) -> dict:

        if( os.path.isfile( path ) ):
            try:
                output = {}
                with open( path, 'r' ) as f:
                    for line in f.readlines():
                        line = line.strip()
                        if( line[0] == '#' ):  pass
                        else:
                            line = line.replace( '\t', ' ' ).split( ' ' )
                            
                            # the magic re.sub() simply truncates the OptiTrack time to 6 digits
                            timestamp = datetime.strptime( re.sub(r'(\.\d{6})\d+', r'\1', line.pop( 0 )), "%Y.%b.%d_%H.%M.%S.%f.UTC" )
                            timestamp = timestamp.replace( tzinfo = timezone.utc )
                            output[timestamp] = {}
                            
                            num_items = int( line.pop( 0 ) )
                            for item in range( num_items ):
                                name = line[ 17 * item ]
                                R = []
                                for el in range( 16 ):
                                    R.append( float( line[17 * item + ( el + 1 )] ) )
                                R = np.array( R ).reshape( ( 4, 4 ) ).T

                                if( name in self._object_R.keys() ):
                                    R[:3, :3] = R[:3, :3] @ self._object_R[name]

                                output[timestamp][name] = R

                return output

            except Exception as e:
                self._print( f"OptiTrack:  Error occured while parsing file:\n\t{type(e)}: {e}" )
                return {}
                            
        else:
            self._print( "OptiTrack log file failed to load." )
            return {}