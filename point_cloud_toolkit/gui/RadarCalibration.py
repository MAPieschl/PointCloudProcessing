from dependencies import *

from utils.custom_plotting import LineCanvas, PointCloudPlot

class RadarCalibration( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification

        # Provizio object
        self._vizio = vizio.Provizio( print_func = self._show_notification )

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "Radar Calibration", False )

        # Build toolbar
        self.radar_data_btn = QPushButton( "Select MCAP (ROS2) point cloud file" )
        self.radar_data_btn.clicked.connect( self.load_mcap_data )
        self.left_toolbar.addWidget( self.radar_data_btn )

        self.optitrack_btn = QPushButton( "Select OptiTrack file" )
        self.optitrack_btn.clicked.connect( self.load_optitrack_data )
        self.left_toolbar.addWidget( self.optitrack_btn )

        self.selection_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.selection_layout )

        self.select_all_btn = QPushButton( "Select All" )
        self.select_all_btn.clicked.connect( lambda : self.select_all( True ) )
        self.selection_layout.addWidget( self.select_all_btn )

        self.clear_all_btn = QPushButton( "Clear Selection" )
        self.clear_all_btn.clicked.connect( lambda : self.select_all( False ) )
        self.selection_layout.addWidget( self.clear_all_btn )

        self.loaded_frames_area = QScrollArea()
        self.left_toolbar.addWidget( self.loaded_frames_area )

        self.loaded_frames_container = QWidget()
        self.loaded_frames_area.setWidget( self.loaded_frames_container )
        self.loaded_frames_area.setWidgetResizable( True )

        self.loaded_frames_layout = QVBoxLayout( self.loaded_frames_container )
        self.loaded_frames_layout.setAlignment( Qt.AlignmentFlag.AlignTop )
        self.loaded_frames: dict[QCheckBox, dict] = {}

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

        self.double_validator = QDoubleValidator( 0, 1000, 3 )

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
            if( type( args[0] ) == QCheckBox and type( args[1] ) == bool ):
                if( args[1] ):  
                    print( type( self.loaded_frames[args[0]]['data'][0] ), self.loaded_frames[args[0]]['data'] )
                    self.pc_plot.add(
                        self.loaded_frames[args[0]]['data'], 
                        np.array( [self.loaded_frames[args[0]]['sequence'] for i in range( self.loaded_frames[args[0]]['data'].shape[0] )] ),
                        f"{self.loaded_frames[args[0]]['name']}_{self.loaded_frames[args[0]]['sequence']}"
                    )

                else:           
                    self.pc_plot.remove( f"{self.loaded_frames[args[0]]['name']}_{self.loaded_frames[args[0]]['sequence']}" )

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
                for cb in list( self.loaded_frames.keys() ):
                    self.loaded_frames_layout.removeWidget( cb )
                self.loaded_frames.clear()

                # Add new model
                for key in list( frames.keys() ):
                    cb = QCheckBox( f"Frame {key}" )
                    self.loaded_frames[cb] = frames[key]
                    cb.stateChanged.connect( lambda x, s = cb: self.update_radar_calibration( s, x == 2 ) )
                    self.loaded_frames_layout.addWidget( cb )

            else:
                self._show_notification( "Model directory no longer exists." )

            self.update_()

        except Exception as e:
            self._show_notification( f"GUI:  Unable to load point cloud due to error:\n\t{type( e ).__name__}: {e}" )
    
    def load_optitrack_data( self ):
        return
    
    def select_all( self, select: bool ):
        return

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