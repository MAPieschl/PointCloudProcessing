from dependencies import *
import utils.globals as globals

from utils.OptiTrack import OptiTrack

class OptiTrackCalibration( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification
        self._optitrack = OptiTrack( self._show_notification )

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "OptiTrack Calibration" )

        # Build toolbar
        self.optitrack_btn = QPushButton( "Select OptiTrack file" )
        self.optitrack_btn.clicked.connect( self.load_optitrack_data )
        self.left_toolbar.addWidget( self.optitrack_btn )
        
        self.available_items_area = QScrollArea()
        self.left_toolbar.addWidget( self.available_items_area )

        self.available_items_container = QWidget()
        self.available_items_area.setWidget( self.available_items_container )
        self.available_items_area.setWidgetResizable( True )

        self.available_items_layout = QVBoxLayout( self.available_items_container )
        self.available_items_layout.setAlignment( Qt.AlignmentFlag.AlignTop )
        self.available_items: dict[QCheckBox, str] = {}

        self.left_toolbar.addStretch()
            
    def load_optitrack_data( self ):
        try:

            file_dialog = QFileDialog( self )
            file_path, _ = file_dialog.getOpenFileName( self, "Select OptiTrack .log file" )

            if( os.path.isfile( file_path ) ):
                self._optitrack.parse_log( file_path )

        except:
            self._show_notification( "OptiTrack file no longer exists." )