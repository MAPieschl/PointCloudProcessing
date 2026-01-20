from dependencies import *
import utils.globals as globals

from utils.TQDMCapture import TQDMCapture

class ROS:
    def __init__(  self, print_func: Callable[[str], None] ):

        self._print = print_func

        self._datatypes = {
            'PointField': {
                1: {
                    'dtype': np.int8,
                    'bytes': 1,
                    'le': '<i1',
                    'be': '>i1'
                },
                2: {
                    'dtype': np.uint8,
                    'bytes': 1,
                    'le': '<u1',
                    'be': '>u1'
                },
                3: {
                    'dtype': np.int16,
                    'bytes': 2,
                    'le': '<i2',
                    'be': '>i2'
                },
                4: {
                    'dtype': np.uint16,
                    'bytes': 2,
                    'le': '<u2',
                    'be': '>u2'
                },
                5: {
                    'dtype': np.int32,
                    'bytes': 4,
                    'le': '<i4',
                    'be': '>i4'
                },
                6: {
                    'dtype': np.uint32,
                    'bytes': 4,
                    'le': '<u4',
                    'be': '>u4'
                },
                7: {
                    'dtype': np.float32,
                    'bytes': 4,
                    'le': '<f4',
                    'be': '>f4'
                },
                8: {
                    'dtype': np.float64,
                    'bytes': 8,
                    'le': '<f8',
                    'be': '>f8'
                },
            }
        }

    def create_np_dtype_from( self, ros2_fields: list, is_bigendian: bool ):

        dtype_unordered = {}

        for field in ros2_fields:
            if( 'PointField' in str( field ) ):
                if( field.count != 1 ):     Exception( 'Error in PointField parsing - multiple values not currently handled.' )
                dtype_unordered[field.offset] = ( field.name, self._datatypes['PointField'][int( field.datatype )]['be' if is_bigendian else 'le'] )

        dtype_ordered = [dtype_unordered[key] for key in sorted( dtype_unordered.keys() )]
        
        return np.dtype(dtype_ordered)

class Provizio:
    def __init__( self, topics: list[str] = ['rt/provizio_radar_point_cloud'], print_func: Callable[[str], None] = print ):
        
        self._print = print_func
        self._topics = topics
        self._ROS = ROS( print_func )
    
    def parse_mcap( self, path: str, progress_capture: TQDMCapture | None = None ) -> dict:

        if( os.path.isfile( path ) ):
            with open( path, "rb" ) as f:

                reader = make_reader( f, decoder_factories = [DecoderFactory()] )
                frames = {}

                try:
                    for schema, channel, message, ros_msg in tqdm( reader.iter_decoded_messages( topics = self._topics ), file = progress_capture ):

                        msg = { k: getattr( ros_msg, k ) for k in ros_msg.__slots__ }

                        frames[message.sequence] = {
                            'name': schema.name,
                            'encoding': schema.encoding,
                            'topic': channel.topic,
                            'metadata': channel.metadata,
                            'channel_id': channel.id,
                            'log_time': datetime.fromtimestamp( message.log_time / 1e9, tz = timezone.utc ),
                            'publish_time': datetime.fromtimestamp( message.publish_time / 1e9, tz = timezone.utc ),
                            'sequence': message.sequence,
                            'height': msg['height'],
                            'width': msg['width'],
                            'point_step': msg['point_step'],
                            'row_step': msg['row_step'],
                            'is_dense': msg['is_dense'],
                            'is_bigendian': msg['is_bigendian'],
                            'fields': deque( [field.name for field in msg['fields']] ),
                            'data': np.frombuffer( msg['data'], self._ROS.create_np_dtype_from( msg['fields'], msg['is_bigendian'] ) ),
                            'dtype': self._ROS.create_np_dtype_from( msg['fields'], msg['is_bigendian'] )
                        }
                
                except Exception as e:
                    self._print( f'Unable to parse MCAP data -> {type(e)}: {e}' )

            return frames

        else:
            self._print( f"File {path} does not exist." )
            return {}
        
    def to_aftr_frame( self, path: str, points: np.ndarray, labels: np.ndarray = np.array( [] ) ) -> None:

        if( len( points.shape ) != 2 or points.shape[1] != 3 ):
            self._print( f"Unable to create aftr frame -> points vector must be shape (N, 3), not {points.shape}." )
            return
        
        if( points.shape[0] != labels.shape[0] and labels.shape[0] != 0):
            self._print( f"Unable to create aftr frame -> if labels are available, the number of labels much match the number of points. Currently there are {points.shape[0]} points and {labels.shape[0]} labels." )
            return
        
        if( os.path.isdir( os.path.dirname( path ) ) ):
            with open( path, 'w' ) as f:
                for i, pt in enumerate( points ):
                    f.write( f'({pt[0]}, {pt[1]}, {pt[2]})' )
                    if( labels.shape[0] > 0 ):
                        for lbl in labels[i]:
                            f.write( f' {lbl}' )
                    f.write( '\n' )

        else:
            self._print( "Unable to create aftr frame -> path does not exist." )