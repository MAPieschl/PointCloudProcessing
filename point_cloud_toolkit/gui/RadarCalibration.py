from dependencies import *
import utils.globals as globals

import utils.mat_ops as mat_ops
import utils.corner_reflector as corner_reflector
from utils.Provizio import Provizio
from utils.OptiTrack import OptiTrack
from utils.custom_plotting import LineCanvas, PointCloudPlot

class LineItemRadiobuttonwithSlider(QWidget):
    def __init__( self, label: str, print_func: Callable[[str], None]):
        super().__init__()

        self._show_notification = print_func

        self.line_layout = QHBoxLayout( self )

        self.radiobutton = QRadioButton( label )
        self.line_layout.addWidget( self.radiobutton )

        self.slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.line_layout.addWidget( self.slider )

class RadarCalibration( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification

        # Provizio object
        self._vizio = Provizio( print_func = self._show_notification )
        self._optitrack = OptiTrack( print_func = self._show_notification )

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "Radar Calibration", False )

        # Build toolbar
        self.double_validator = QDoubleValidator( 0, 1000, 3 )

        self.radar_data_btn = QPushButton( "Select MCAP (ROS2) point cloud file" )
        self.radar_data_btn.clicked.connect( self.load_mcap_data )
        self.left_toolbar.addWidget( self.radar_data_btn )

        self.optitrack_btn = QPushButton( "Select OptiTrack file" )
        self.optitrack_btn.clicked.connect( self.load_optitrack_data )
        self.left_toolbar.addWidget( self.optitrack_btn )

        self.selection_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.selection_layout )

        self.loaded_frames_area = QScrollArea()
        self.left_toolbar.addWidget( self.loaded_frames_area )

        self.loaded_frames_container = QWidget()
        self.loaded_frames_area.setWidget( self.loaded_frames_container )
        self.loaded_frames_area.setWidgetResizable( True )

        self.loaded_frames_layout = QVBoxLayout( self.loaded_frames_container )
        self.loaded_frames_layout.setAlignment( Qt.AlignmentFlag.AlignTop )
        self.loaded_frames: dict[LineItemRadiobuttonwithSlider, dict] = {}
        self.truth_data: dict[datetime, dict]
        self.target_truth_position = None
        self.target_filter_center = None
        self.target_filter_radius = None

        self.loaded_frames_btn_group = QButtonGroup()

        self.current_color_field = None
        self.color_label = QLabel()
        self.left_toolbar.addWidget( self.color_label )
        self.change_color_btn = QPushButton( "Change Color Scheme" )
        self.change_color_btn.clicked.connect( lambda x, s = self.change_color_btn : self.update_radar_calibration( s, 'next' ) )
        self.left_toolbar.addWidget( self.change_color_btn )

        self.left_toolbar.addWidget( QLabel( "Filter by Color" ) )
        self.filter_color_slider = QSlider( Qt.Orientation.Horizontal )
        self.filter_color_slider.setRange( 0, 100 )
        self.filter_color_slider.setValue( 0 )
        self.filter_color_slider.sliderMoved.connect( lambda x, s = self.filter_color_slider : self.update_radar_calibration( s, x ) )
        self.left_toolbar.addWidget( self.filter_color_slider )

        self.shift_title_area = QHBoxLayout()
        self.left_toolbar.addLayout( self.shift_title_area )

        self.shift_title_area.addWidget( QLabel( "Shift filter center by " ) )
        self.shift_amount = QLineEdit()
        self.shift_amount.setValidator( self.double_validator )
        self.shift_title_area.addWidget( self.shift_amount )

        self.shift_pos_btn_area = QHBoxLayout()
        self.left_toolbar.addLayout( self.shift_pos_btn_area )

        self.shift_neg_btn_area = QHBoxLayout()
        self.left_toolbar.addLayout( self.shift_neg_btn_area )

        self.posx_btn = QPushButton( "+x" )
        self.posx_btn.clicked.connect( lambda : self.set_target_filter_center( np.array( [float( self.shift_amount.text() ), 0, 0] ), True ) )
        self.shift_pos_btn_area.addWidget( self.posx_btn )

        self.posx_btn = QPushButton( "+y" )
        self.posx_btn.clicked.connect( lambda : self.set_target_filter_center( np.array( [0, float( self.shift_amount.text() ), 0] ), True ) )
        self.shift_pos_btn_area.addWidget( self.posx_btn )

        self.posx_btn = QPushButton( "+z" )
        self.posx_btn.clicked.connect( lambda : self.set_target_filter_center( np.array( [0, 0, float( self.shift_amount.text() )] ), True ) )
        self.shift_pos_btn_area.addWidget( self.posx_btn )

        self.posx_btn = QPushButton( "-x" )
        self.posx_btn.clicked.connect( lambda : self.set_target_filter_center( np.array( [-float( self.shift_amount.text() ), 0, 0] ), True ) )
        self.shift_neg_btn_area.addWidget( self.posx_btn )

        self.posx_btn = QPushButton( "-y" )
        self.posx_btn.clicked.connect( lambda : self.set_target_filter_center( np.array( [0, -float( self.shift_amount.text() ), 0] ), True ) )
        self.shift_neg_btn_area.addWidget( self.posx_btn )

        self.posx_btn = QPushButton( "-z" )
        self.posx_btn.clicked.connect( lambda : self.set_target_filter_center( np.array( [0, 0, -float( self.shift_amount.text() )] ), True ) )
        self.shift_neg_btn_area.addWidget( self.posx_btn )

        self.reset_btn = QPushButton( "Reset to Target Truth Position" )
        self.reset_btn.clicked.connect( self.reset_target_filter_center )
        self.left_toolbar.addWidget( self.reset_btn )

        self.left_toolbar.addStretch()

        # Calibration tool
        self.calib_area = QVBoxLayout()
        self.main_area.addLayout( self.calib_area, 15 )

        self.pc_plot = PointCloudPlot( 
            title = "Calibration Data Display", 
            print_func = self._show_notification 
        )
        self.pc_plot_area = QWebEngineView()
        self.calib_area.addWidget( self.pc_plot_area )
        
        # Main area separator
        v_line = QFrame()
        v_line.setFrameShape( QFrame.Shape.VLine )
        self.main_area.addWidget( v_line, 1 )

        # Corner reflector analysis tool
        self.reflector_area = QVBoxLayout()
        self.main_area.addLayout( self.reflector_area, 5 )

        title_label = QLabel( "Corner Reflector Analysis" )
        title_label.setFont( QFont( "Arial", 18 ) )
        title_label.setAlignment( Qt.AlignmentFlag.AlignCenter )
        self.reflector_area.addWidget( title_label, 1 )

        # Build plots
        self.front_plot = LineCanvas(
            title = 'Corner Reflector - Front View',
            print_func = self._show_notification
        )
        self.front_plot_area = QWebEngineView()
        self.reflector_area.addWidget( self.front_plot_area, 3 )

        self.top_plot = LineCanvas(
            title = 'Corner Reflector - Top View',
            print_func = self._show_notification
        )
        self.top_plot_area = QWebEngineView()
        self.reflector_area.addWidget( self.top_plot_area, 3 )

        # Build options area
        self.options_layout = QVBoxLayout()
        self.reflector_area.addLayout( self.options_layout, 3 )

        self.options_layout.addWidget( QLabel( "Orient Corner Reflector" ) )

        self.yaw_layout = QHBoxLayout()
        self.options_layout.addLayout( self.yaw_layout, 1 )

        self.yaw_layout.addWidget( QLabel( "Yaw" ) )

        self.yaw_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.yaw_slider.setRange( -45, 45 )
        self.yaw_slider.sliderMoved.connect( self.update_corner_reflector )
        self.yaw_layout.addWidget( self.yaw_slider )

        self.pitch_layout = QHBoxLayout()
        self.options_layout.addLayout( self.pitch_layout, 1 )

        self.pitch_layout.addWidget( QLabel( "Pitch" ) )

        self.pitch_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.pitch_slider.setRange( -45, 45 )
        self.pitch_slider.sliderMoved.connect( self.update_corner_reflector )
        self.pitch_layout.addWidget( self.pitch_slider )

        self.roll_layout = QHBoxLayout()
        self.options_layout.addLayout( self.roll_layout )

        self.roll_layout.addWidget( QLabel( "Roll" ) )

        self.roll_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.roll_slider.setRange( -90, 90 )
        self.roll_slider.sliderMoved.connect( self.update_corner_reflector )
        self.roll_layout.addWidget( self.roll_slider )

        self.options_layout.addWidget( QLabel( "Move Entry Point" ) )
        self.SLIDER_SCALE = 1000

        self.x_layout = QHBoxLayout()
        self.options_layout.addLayout( self.x_layout , 1)

        self.x_layout.addWidget( QLabel( "X" ) )

        self.x_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.x_slider.sliderMoved.connect( self.update_corner_reflector )
        self.x_slider.sliderReleased.connect( self.update_corner_reflector )
        self.x_layout.addWidget( self.x_slider )

        self.y_layout = QHBoxLayout()
        self.options_layout.addLayout( self.y_layout , 1)

        self.y_layout.addWidget( QLabel( "Y" ) )

        self.y_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.y_slider.sliderMoved.connect( self.update_corner_reflector )
        self.y_slider.sliderReleased.connect( self.update_corner_reflector )
        self.y_layout.addWidget( self.y_slider )

        self.edge_entry_layout = QHBoxLayout()
        self.options_layout.addLayout( self.edge_entry_layout, 1 )

        self.edge_label = QLabel( "Edge Length" )
        self.edge_entry_layout.addWidget( self.edge_label )

        self.edge_entry = QLineEdit()
        self.edge_entry.setValidator( self.double_validator )
        self.edge_entry.setText( "0.1" )
        self.edge_entry.textChanged.connect( self.update_corner_reflector )
        self.edge_entry_layout.addWidget( self.edge_entry )

        self.edge_unit = QLabel( "m" )
        self.edge_entry_layout.addWidget( self.edge_unit )

        self.freq_entry_layout = QHBoxLayout()
        self.options_layout.addLayout( self.freq_entry_layout, 1 )
        
        self.freq_label = QLabel( "Carrier Frequency" )
        self.freq_entry_layout.addWidget( self.freq_label )

        self.freq_entry = QLineEdit()
        self.freq_entry.setValidator( self.double_validator )
        self.freq_entry.setText( "78.5" )
        self.freq_entry.textChanged.connect( self.update_corner_reflector )
        self.freq_entry_layout.addWidget( self.freq_entry )

        self.freq_unit = QLabel( "GHz" )
        self.freq_entry_layout.addWidget( self.freq_unit )
        
        self.dist_traveled_label = QLabel()
        self.options_layout.addWidget( self.dist_traveled_label )

        self.rcs_label = QLabel()
        self.options_layout.addWidget( self.rcs_label )

        self.update_()

    def update_( self, *args ) -> None:

        self.update_radar_calibration( args )
        self.update_corner_reflector( args )

    def update_radar_calibration( self, *args ) -> None:
        
        if( len( args ) >= 2 and type( self.pc_plot ) == PointCloudPlot ):
            if( type( args[0] ) == LineItemRadiobuttonwithSlider and type( args[1] ) == bool ):
                if( args[1] ):  
                    if( self.current_color_field is None): 
                        self.current_color_field = self.loaded_frames[args[0]]['fields']
                    self.pc_plot.clear()
                    self.pc_plot.add(
                        structured_to_unstructured( self.loaded_frames[args[0]]['data'][['x', 'y', 'z']], dtype = np.float32 ), 
                        np.array( self.loaded_frames[args[0]]['data'][self.current_color_field[0]], dtype = np.float32 ),
                        f"{self.loaded_frames[args[0]]['name']}_{self.loaded_frames[args[0]]['sequence']}"
                    )

                    try:
                        opti_ts = list( self.truth_data.keys() )
                        idx = bisect.bisect_left( opti_ts, self.loaded_frames[args[0]]['log_time'] )

                        R_radar = self.truth_data[opti_ts[idx]]['mmwave']
                        R_corner_reflector = self.truth_data[opti_ts[idx]]['corner_reflector']

                        self.target_truth_position = ( R_radar[:3, :3].T @ (R_corner_reflector[:3, 3:] - R_radar[:3, 3:]) ).T[0]
                        if( self.target_filter_center is None ):
                            self.target_filter_center = self.target_truth_position

                        self.pc_plot.add_red_point( self.target_truth_position, size = 5 )

                    except Exception as e:
                        self._show_notification( f"Unable to load truth position:\n\t{type(e)}: {e}" )

                else:           
                    self.pc_plot.remove( f"{self.loaded_frames[args[0]]['name']}_{self.loaded_frames[args[0]]['sequence']}" )
                    self.current_color_field = None

            elif( type( args[0] ) == QPushButton and args[1] == 'next' and type( self.current_color_field ) == deque ):
                self.current_color_field.rotate(1)
                self.loaded_frames_btn_group.checkedButton().toggled.emit( True )

            elif( type( args[0] ) == LineItemRadiobuttonwithSlider and type( args[1] ) == int ):
                if( self.target_filter_center is not None ):
                    max_radius = self.pc_plot.get_max_radius_from( self.target_filter_center )
                    self.target_filter_radius = max_radius * ( args[1] / 100 ) ** 2
                    self.set_target_filter_center( self.target_filter_center, False )

            elif( type( args[0] ) == QSlider and type( args[1] ) == int ):
                self.pc_plot.filter_by_color( args[1], True )
                self.loaded_frames_btn_group.checkedButton().toggled.emit( True )

        if( type( self.current_color_field ) == deque ):
            self.color_label.setText( f"Color Scheme:  {self.current_color_field[0]}" )
        else:
            self.color_label.setText( 'Color Scheme:  none' )

        if( type( self.pc_plot ) == PointCloudPlot ):
            
            html_plot = pio.to_html( self.pc_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.pc_plot_area.setHtml( html_plot )

    def update_corner_reflector( self, *args ) -> None:

        # Corner reflector analysis updates
        edge_length = float( self.edge_entry.text() )
        a = edge_length / np.sqrt(2)
        self.rcs_label.setText( f"RCS:  {self.compute_rcs( a, globals.C / ( float( self.freq_entry.text() ) * 1e9 ) ):.3f} m^2" )

        if( type( self.front_plot ) == LineCanvas and type( self.top_plot ) == LineCanvas ):
            aspect_ratio = self.front_plot_area.width() / self.front_plot_area.height()
            y_lims = [-edge_length / 1.5, edge_length / 1.5]
            x_lims = [y_lims[0] * aspect_ratio, y_lims[1] * aspect_ratio]

            self.x_slider.setRange( int( np.floor( x_lims[0] * self.SLIDER_SCALE ) ), int( np.ceil( x_lims[1] * self.SLIDER_SCALE ) ) )
            self.y_slider.setRange( int( np.floor( y_lims[0] * self.SLIDER_SCALE ) ), int( np.ceil( y_lims[1] * self.SLIDER_SCALE ) ) )

            reflector_info = self.draw_reflector_pose( edge_length, self.pitch_slider.value(), self.yaw_slider.value(), self.roll_slider.value() )
            reflection = self.draw_reflections( np.array( [self.x_slider.value() / self.SLIDER_SCALE, self.y_slider.value() / self.SLIDER_SCALE, y_lims[1]] ), reflector_info )

            self.dist_traveled_label.setText( f"Ray Distance Measured:  {reflection:0.3f} m | Actual Distance:  {2 * (y_lims[1] - reflector_info['apex'][2]):0.3f} | Error:  {reflection - 2 * (y_lims[1] - reflector_info['apex'][2]):0.3f}" )

            html_plot = pio.to_html( self.front_plot.get_fig( x_lims , y_lims ), full_html = False, include_plotlyjs = 'cdn' )
            self.front_plot_area.setHtml( html_plot )

            html_plot = pio.to_html( self.top_plot.get_fig( x_lims , y_lims ), full_html = False, include_plotlyjs = 'cdn' )
            self.top_plot_area.setHtml( html_plot )

    def load_mcap_data( self ):
        try:

            file_dialog = QFileDialog(self)
            file_path, _ = file_dialog.getOpenFileName( self, "Select MCAP (ROS2) point cloud file", "" )
            
            if( os.path.isfile( file_path ) ):
                frames = self._vizio.parse_mcap( file_path )

                # Clear out previous frames
                for line_item in self.loaded_frames.keys():
                    self.loaded_frames_layout.removeWidget( line_item )
                self.loaded_frames.clear()

                # Add new model
                for key in list( frames.keys() ):
                    li = LineItemRadiobuttonwithSlider( f"Frame {key}", self._show_notification )
                    self.loaded_frames[li] = frames[key]
                    li.radiobutton.toggled.connect( lambda x, s = li : self.update_radar_calibration( s, x ) )
                    li.slider.setRange( 0, 100 )
                    li.slider.sliderMoved.connect( lambda x, s = li : self.update_radar_calibration( s, x ) )
                    # li.slider.sliderReleased.connect( lambda x, s = li: self.update_radar_calibration( s, x ) )
                    self.loaded_frames_btn_group.addButton( li.radiobutton )
                    self.loaded_frames_layout.addWidget( li )

            else:
                self._show_notification( "MCAP file no longer exists." )

            self.update_()

        except Exception as e:
            self._show_notification( f"GUI:  Unable to load point cloud due to error:\n\t{type( e ).__name__}: {e}" )
    
    def load_optitrack_data( self ):
        try:

            file_dialog = QFileDialog( self )
            file_path, _ = file_dialog.getOpenFileName( self, "Select OptiTrack .log file" )

            if( os.path.isfile( file_path ) ):
                self.truth_data = self._optitrack.parse_log( file_path )

        except:
            self._show_notification( "OptiTrack file no longer exists." )

    def set_target_filter_center( self, value: np.ndarray, is_relative: bool = False ):

        if( value.shape[0] != 3 or len( value.shape ) != 1 ):
            self._show_notification( "Value provided to set_target_filter_center must be a (3,) numpy vector." )
            return

        if( is_relative and self.target_filter_center is not None ):
            self.target_filter_center += value
        else:
            self.target_filter_center = value
        
        if( self.target_filter_radius is not None ):
            self.pc_plot.filter_by_radius( self.target_filter_center, self.target_filter_radius )
            self.loaded_frames_btn_group.checkedButton().toggled.emit( True )

    def reset_target_filter_center( self ):
        self.target_filter_center = self.target_truth_position
        if( self.target_filter_center is not None and self.target_filter_radius is not None ):
            self.pc_plot.filter_by_radius( self.target_filter_center, self.target_filter_radius )
            self.loaded_frames_btn_group.checkedButton().toggled.emit( True )

    def compute_rcs( self, a: float, wavelength: float ) -> float:

        return ( 4 * np.pi * ( a ** 4 ) ) / ( 3 * ( wavelength ** 2 ) )
    
    def draw_reflector_pose( self, edge_length: float, roll: float, pitch: float, yaw: float ) -> dict[str, np.ndarray]:

        # Radius of incircle
        r = np.sqrt( 3 ) * edge_length / 6

        # Array of corners (CCW)
        corners = np.array( [[-0.5 * edge_length, -r, 0], [0.5 * edge_length, -r, 0], [0, 2 * r, 0], [-0.5 * edge_length, -r, 0]] )
        apex = np.array( [0, 0, -edge_length / np.sqrt( 6 )] )

        # Rotation matrix (3D)
        R = mat_ops.get_dcm( roll, pitch, yaw )

        # Rotate triangle
        corners = ( R @ corners.T ).T
        apex = ( R @ apex.T ).T

        # Front view lines
        front_lines = []
        for i in range( 3 ):
            front_lines.append( [corners[i][:2], corners[i + 1][:2]] )
            front_lines.append( [corners[i][:2], apex[:2]] )
        
        self.front_plot.clear()
        self.front_plot.add( np.array( front_lines ), np.array( ['black' for i in range( len( front_lines ) )] ) )
        
        # Top view lines
        top_lines = []
        for i in range( 3 ):
            top_lines.append( [corners[i][[0, 2]], corners[i + 1][[0, 2]]] )
            top_lines.append( [corners[i][[0, 2]], apex[[0, 2]]] )
        
        self.top_plot.clear()
        self.top_plot.add( np.array( top_lines ), np.array( ['black' for i in range ( len( top_lines ) )] ) )

        return {
            'corners': corners[:3],
            'apex': apex
        }

    def draw_reflections( self, ray_origin: np.ndarray, reflector_info: dict[str, np.ndarray] ) -> float:

        ray_vector = np.array( [0, 0, -1] )
        ultimate_origin = ray_origin
        rays = np.array( [] )
        while( True ):
            reflection = corner_reflector.get_reflection( ray_origin, ray_vector, reflector_info['corners'], reflector_info['apex'], self._show_notification )

            if( reflection == {} ):
                if( rays.shape[0] < 1 ):
                    rays = np.array( [[ ray_origin, ray_origin * np.array( [1, 1, -1] ) ]] )
                else:
                    scaling_factor = ( ultimate_origin[2] - ray_origin[2] ) / ray_vector[2]
                    if( not np.isfinite( scaling_factor ) ):  scaling_factor = 0
                    rays = np.concatenate( (rays, np.array( [[ray_origin, ray_origin + ray_vector * scaling_factor]] )), axis = 0 )
                break
            
            if( rays.shape[0] > 0 ):
                rays = np.concatenate( (rays, np.array( [[ray_origin, reflection['collision_point']]] )), axis = 0 )
            else:
                rays = np.array( [[ray_origin, reflection['collision_point']]] )
            
            ray_origin = reflection['collision_point']
            ray_vector = reflection['reflection_vector']

        total_dist = 0
        for i in range( rays.shape[0] ):
            total_dist += np.linalg.norm( rays[i][1] - rays[i][0] )

        if( rays.shape[0] >= 2 ):
            returned_to_source = True if np.linalg.norm( np.cross( np.array( [0, 0, 1] ), rays[-1][1] - rays[-1][0] ) ) < 0.00001 else False
        else:
            returned_to_source = False

        front_rays = rays[:, :, :2]
        self.front_plot.add( front_rays, np.array( ['green' if returned_to_source else 'red' for i in range( len( front_rays ) )] ) )

        top_rays = rays[:, :, [0, 2]]
        self.top_plot.add( top_rays, np.array( ['green' if returned_to_source else 'red' for i in range( len( top_rays ) )] ) )
            
        return float( total_dist )