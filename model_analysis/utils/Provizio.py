from dependencies import *

class ROS:
    def __init__(  self, print_func: Callable[[str], None] ):

        self._print = print_func

        self._datatypes = {
            PointField: {
                'datatype': {
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
        }

    def create_np_dtype_from( self, ros2_fields: list, is_bigendian: bool ):

        dtype_list = []

        for field in ros2_fields:
            if( type( field ) == PointField ):
                if( field.count != 1 ):     Exception( 'Error in PointField parsing - multiple values not currently handled.' )
                dtype_list.append( ( field.name,  ) )

class Provizio:
    def __init__( self, topics: list[str] = ['rt/provizio_radar_point_cloud'], print_func: Callable[[str], None] = print ):
        
        self._print = print_func
        self._topics = topics
        self._dtype = np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('radar_relative_radial_velocity', '<f4'),
            ('signal_to_noise_ratio', '<f4'),
            ('ground_relative_radial_velocity', '<f4'),
        ])
    
    def parse_mcap( self, path: str ):

        if( os.path.isfile( path ) ):
            with open( path, "rb" ) as f:

                reader = make_reader( f, decoder_factories = [DecoderFactory()] )

                for schema, channel, message, ros_msg in reader.iter_decoded_messages( topics = self._topics ):

                    msg = { k: getattr( ros_msg, k ) for k in ros_msg.__slots__ }

                    self._print( f"Name:  {schema.name}" )
                    self._print( f"Encoding:  {schema.encoding}" )
                    self._print( f"Topic:  {channel.topic}" )
                    self._print( f"Metadata:" )
                    for key in list( channel.metadata.keys() ):
                        self._print( f"\t{key}:  {channel.metadata[key]}" )
                    self._print( f"Channel ID:  {channel.id}" )
                    self._print( f"Log Time:  {message.log_time}" )
                    self._print( f"Publish:  {message.publish_time}" )
                    self._print( f"Sequence:  {message.sequence}" )
                    for key in list( msg ):
                        if( key[0] != '_' ):
                            self._print( f"\t{key}:  {msg[key]} {type(msg[key])}" )

        else:
            self._print( f"File {path} does not exist." )

vizio = Provizio()
vizio.parse_mcap( "C:/Users/user/Downloads/20260108_212149_UTC_provizio_ROS2.mcap" )