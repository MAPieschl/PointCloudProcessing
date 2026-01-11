from dependencies import *

from utils.custom_plotting import LinePlot

class RadarCalibration( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "Radar Calibration", False )

        # Calibration tool
        self.calib_area = QVBoxLayout()
        self.main_area.addLayout( self.calib_area, 10 )
        
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
        self.front_plot = LinePlot(
            title = 'Corner Reflector - Front View',
            x_axis_title = '',
            y1_axis_title = '',
            y2_axis_title = '',
            print_func = self._show_notification
        )
        self.front_data = {}
        self.front_plot_area = QWebEngineView()
        self.reflector_area.addWidget( self.front_plot_area, 3 )

        self.top_plot = LinePlot(
            title = 'Corner Reflector - Top View',
            x_axis_title = '',
            y1_axis_title = '',
            y2_axis_title = '',
            print_func = self._show_notification
        )
        self.top_data = {}
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
        self.yaw_slider.sliderMoved.connect( self.update_ )
        self.yaw_layout.addWidget( self.yaw_slider )

        self.pitch_layout = QHBoxLayout()
        self.options_layout.addLayout( self.pitch_layout, 1 )

        self.pitch_layout.addWidget( QLabel( "Pitch" ) )

        self.pitch_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.pitch_slider.setRange( -45, 45 )
        self.pitch_slider.sliderMoved.connect( self.update_ )
        self.pitch_layout.addWidget( self.pitch_slider )

        self.roll_layout = QHBoxLayout()
        self.options_layout.addLayout( self.roll_layout )

        self.roll_layout.addWidget( QLabel( "Roll" ) )

        self.roll_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.roll_slider.setRange( -90, 90 )
        self.roll_slider.sliderMoved.connect( self.update_ )
        self.roll_layout.addWidget( self.roll_slider )

        self.options_layout.addWidget( QLabel( "Move Entry Point" ) )

        self.x_layout = QHBoxLayout()
        self.options_layout.addLayout( self.x_layout , 1)

        self.x_layout.addWidget( QLabel( "X" ) )

        self.x_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.x_slider.sliderMoved.connect( self.update_ )
        self.x_layout.addWidget( self.x_slider )

        self.y_layout = QHBoxLayout()
        self.options_layout.addLayout( self.y_layout , 1)

        self.y_layout.addWidget( QLabel( "Y" ) )

        self.y_slider = QSlider( orientation = Qt.Orientation.Horizontal )
        self.y_slider.sliderMoved.connect( self.update_ )
        self.y_layout.addWidget( self.y_slider )

        self.double_validator = QDoubleValidator( 0, 1000, 3 )

        self.edge_entry_layout = QHBoxLayout()
        self.options_layout.addLayout( self.edge_entry_layout, 1 )

        self.edge_label = QLabel( "Edge Length" )
        self.edge_entry_layout.addWidget( self.edge_label )

        self.edge_entry = QLineEdit()
        self.edge_entry.setValidator( self.double_validator )
        self.edge_entry.setText( "0.1" )
        self.edge_entry.textChanged.connect( self.update_ )
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
        self.freq_entry.textChanged.connect( self.update_ )
        self.freq_entry_layout.addWidget( self.freq_entry )

        self.freq_unit = QLabel( "GHz" )
        self.freq_entry_layout.addWidget( self.freq_unit )

        self.rcs_label = QLabel()
        self.options_layout.addWidget( self.rcs_label )

        self.update_()

    def update_( self, *args ) -> None:

        # Corner reflector analysis updates
        edge_length = float( self.edge_entry.text() )
        a = edge_length / np.sqrt(2)
        self.rcs_label.setText( f"RCS:  {self.compute_rcs( a, globals.C / ( float( self.freq_entry.text() ) * 1e9 ) ):.3f} m^2" )

        if( type( self.front_plot ) == LinePlot ):
            html_plot = pio.to_html( self.front_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.front_plot_area.setHtml( html_plot )

    def compute_rcs( self, a: float, wavelength: float ) -> float:

        return ( 4 * np.pi * ( a ** 4 ) ) / ( 3 * ( wavelength ** 2 ) )
    
    def draw_front_view( self, edge_length: float, roll: float, pitch: float, yaw: float ) -> None:

        # Radius of incircle
        r = np.sqrt( 3 ) * edge_length / 6

        corners = np.array( [[-0.5 * edge_length, -r], [0.5 * edge_length, -r], [0, 2 * r], [-0.5 * edge_length, -r]] )