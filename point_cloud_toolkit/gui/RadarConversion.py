from dependencies import *
import utils.globals as globals

import utils.mat_ops as mat_ops
import utils.calibration as calibration
from utils.Provizio import Provizio
from utils.custom_plotting import PointCloudPlot
from utils.TQDMCapture import TQDMCapture

class RadarConversion( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification
        self._show_yes_no_query = parent.show_yes_no_query

        # Provizio object
        self._vizio = Provizio( print_func = self._show_notification )
        self._tqdm_capture = TQDMCapture()

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "Radar Conversion", False )

        # Build toolbar
        self.double_validator = QDoubleValidator( 0, 1000, 3 )

        self.radar_data_btn = QPushButton( "Select MCAP (ROS2) point cloud file" )
        self.radar_data_btn.clicked.connect( self.load_mcap_data )
        self.left_toolbar.addWidget( self.radar_data_btn )

        self.data_progress_bar = QProgressBar()
        self.data_progress_bar.setRange( 0, 100 )
        self.left_toolbar.addWidget( self.data_progress_bar )

        self.selection_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.selection_layout )

        self.loaded_frames_area = QScrollArea()
        self.left_toolbar.addWidget( self.loaded_frames_area )

        self.loaded_frames_container = QWidget()
        self.loaded_frames_area.setWidget( self.loaded_frames_container )
        self.loaded_frames_area.setWidgetResizable( True )

        self.loaded_frames_layout = QVBoxLayout( self.loaded_frames_container )
        self.loaded_frames_layout.setAlignment( Qt.AlignmentFlag.AlignTop )
        self.loaded_frames: dict[QRadioButton, dict] = {}

        self.loaded_frames_btn_group = QButtonGroup()

        self.current_color_field = None
        self.color_label = QLabel()
        self.left_toolbar.addWidget( self.color_label )
        self.change_color_btn = QPushButton( "Change Color Scheme" )
        self.change_color_btn.clicked.connect( lambda x, s = self.change_color_btn : self.update_( s, 'next' ) )
        self.left_toolbar.addWidget( self.change_color_btn )

        self.left_toolbar.addWidget( QLabel( "Filter by Color" ) )
        self.filter_color_slider = QSlider( Qt.Orientation.Horizontal )
        self.filter_color_slider.setRange( 0, 100 )
        self.filter_color_slider.setValue( 0 )
        self.filter_color_slider.sliderMoved.connect( lambda x, s = self.filter_color_slider : self.update_( s, x ) )
        self.left_toolbar.addWidget( self.filter_color_slider )

        self.target_dir_btn = QPushButton( "Select Target Directory" )
        self.target_dir_btn.clicked.connect( self.select_target_dir )
        self.left_toolbar.addWidget( self.target_dir_btn )

        self.target_dir_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.target_dir_layout )

        self.target_dir_layout.addWidget( QLabel( "Target Directory: " ) )
        self.target_dir_label = QLabel( "" )
        self.target_dir_layout.addWidget( self.target_dir_label )

        self.convert_frame_btn = QPushButton( "Convert Frame to AftrFrame" )
        self.convert_frame_btn.clicked.connect( self.convert )
        self.left_toolbar.addWidget( self.convert_frame_btn )

        self.convert_collect_btn = QPushButton( "Convert Collect to AftrFrames" )
        self.convert_collect_btn.clicked.connect( self.convert_all )
        self.left_toolbar.addWidget( self.convert_collect_btn )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange( 0, 100 )
        self.left_toolbar.addWidget( self.progress_bar )

        self.left_toolbar.addStretch()

        # Display area
        self.display_area = QVBoxLayout()
        self.main_area.addLayout( self.display_area, 15 )

        self.pc_plot = PointCloudPlot( 
            title = "Point Cloud Frame Display", 
            print_func = self._show_notification 
        )
        self.pc_plot_area = QWebEngineView()
        self.display_area.addWidget( self.pc_plot_area )

        self.update_()

    def update_( self, *args ) -> None:
        
        if( len( args ) >= 2 and type( self.pc_plot ) == PointCloudPlot ):
            if( type( args[0] ) == QRadioButton and type( args[1] ) == bool ):
                if( args[1] ):  

                    if( self.current_color_field is None): 
                        self.current_color_field = self.loaded_frames[args[0]]['fields']

                    if( self.loaded_frames[args[0]]['sequence'] != self.current_frame ):
                        self.pc_plot.clear_red_points()
                        self.pc_plot.clear_filter()
                        self.current_frame = self.loaded_frames[args[0]]['sequence']

                    self.pc_plot.clear()
                    self.pc_plot.add(
                        structured_to_unstructured( self.loaded_frames[args[0]]['data'][['x', 'y', 'z']], dtype = np.float32 ), 
                        np.array( self.loaded_frames[args[0]]['data'][self.current_color_field[0]], dtype = np.float32 ),
                        f"{self.loaded_frames[args[0]]['name']}_{self.loaded_frames[args[0]]['sequence']}"
                    )

                else:           
                    self.pc_plot.remove( f"{self.loaded_frames[args[0]]['name']}_{self.loaded_frames[args[0]]['sequence']}" )
                    self.current_color_field = None

            elif( type( args[0] ) == QPushButton and args[1] == 'next' and type( self.current_color_field ) == deque ):
                self.current_color_field.rotate(1)
                self.loaded_frames_btn_group.checkedButton().toggled.emit( True )

            elif( type( args[0] ) == QSlider and type( args[1] ) == int ):
                self.pc_plot.filter_by_color( args[1], True )
                self.loaded_frames_btn_group.checkedButton().toggled.emit( True )

        if( type( self.current_color_field ) == deque ):
            self.color_label.setText( f"Color Scheme:  {self.current_color_field[0]}" )
        else:
            self.color_label.setText( 'Color Scheme:  none' )

        # if( type( self.pc_plot ) == PointCloudPlot ):
        #     html_plot = pio.to_html( self.pc_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
        #     self.pc_plot_area.setHtml( html_plot )

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
                for key in tqdm( list( frames.keys() ), file = self._tqdm_capture ):
                    rb = QRadioButton()
                    self.loaded_frames[rb] = frames[key]
                    rb.toggled.connect( lambda x, s = rb : self.update_( s, x ) )
                    self.loaded_frames_btn_group.addButton( rb )
                    # self.loaded_frames_layout.addWidget( rb )
                    progress = self._tqdm_capture.get_percent()
                    if( type( progress ) == int ):
                        self.data_progress_bar.setValue( progress )

            else:
                self._show_notification( "MCAP file no longer exists." )

            self.update_()

        except Exception as e:
            self._show_notification( f"GUI:  Unable to load point cloud due to error:\n\t{type( e ).__name__}: {e}" )

    def select_target_dir( self ):
        try:

            file_dialog = QFileDialog(self)
            dir_path = file_dialog.getExistingDirectory( self, "Choose target directory", "" )
            
            if( os.path.isdir( dir_path ) ):
                self.target_dir_label.setText( dir_path )

            else:
                self._show_notification( "Directory no longer exists." )

            self.update_()

        except Exception as e:
            self._show_notification( f"GUI:  Error during selection of target directory:\n\t{type( e ).__name__}: {e}" )

    def convert( self, key = None ):
        try:
            if( type( key ) != QRadioButton ):
                key = self.loaded_frames_btn_group.checkedButton()

            if( type( key ) == QRadioButton ):
                frame = self.loaded_frames[key]

                with open( f"{self.target_dir_label.text()}/frame_{frame['sequence']}.txt", "w" ) as f:
                    for pt in frame['data']:
                        f.write( f"({pt['x']}, {pt['y']}, {pt['z']})\n" )

                if( not os.path.isfile( f"{self.target_dir_label.text()}/__index.log" ) ):
                    with open( f"{self.target_dir_label.text()}/__index.log", "w" ) as f:
                        f.write( "# UTC_Time\tFrameID\tfilename\n" )

                with open( f"{self.target_dir_label.text()}/index.txt", "a" ) as f:
                    f.write( f"{frame['log_time'].strftime( "%Y.%b.%d_%H.%M.%S.%f" )}.UTC {frame['sequence']} frame_{frame['sequence']}.txt\n" )

            else:
                self._show_notification( f"GUI:  Could not determine which QRadioButton was checked -- unable to convert point cloud." )
                
        except Exception as e:
            self._show_notification( f"GUI:  Error occured while attempting to convert frame:\n\t{type( e ).__name__}: {e}" )

    def convert_all( self ):
        try:
            for key in tqdm( self.loaded_frames.keys(), file = self._tqdm_capture ):
                self.convert( key )
                progress = self._tqdm_capture.get_percent()
                if( type( progress ) == int ):
                    self.progress_bar.setValue( progress )

        except Exception as e:
            self._show_notification( f"GUI:  Error occured while attempting to convert all frames:\n\t{type( e ).__name__}: {e}" )