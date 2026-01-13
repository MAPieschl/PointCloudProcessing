from dependencies import *
import utils.globals as globals

class OptiTrack:
    def __init__( self, print_func: Callable[[str], None] = print ):

        self._print = print_func

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
                            output[timestamp] = {}
                            
                            num_items = int( line.pop( 0 ) )
                            for item in range( num_items ):
                                name = line[ 17 * item ]
                                R = []
                                for el in range( 16 ):
                                    R.append( float( line[17 * item + ( el + 1 )] ) )
                                R = np.array( R ).reshape( ( 4, 4 ) )

                                output[timestamp][name] = R

                return output

            except Exception as e:
                self._print( f"OptiTrack:  Error occured while parsing file:\n\t{type(e)}: {e}" )
                return {}
                            
        else:
            self._print( "OptiTrack log file failed to load." )
            return {}