from dependencies import *
import utils.globals as globals
import utils.mat_ops as mat_ops

from utils.OptiTrack import OptiTrack
from utils.custom_plotting import LinePlot

class OptiTrackCalibration( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification
        self._optitrack = OptiTrack( self._show_notification )

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "OptiTrack Calibration" )
        
        # Data storage
        self.data: dict[QCheckBox, dict] = {}
        self.cb_by_item = {}

        # Build toolbar
        self.optitrack_btn = QPushButton( "Select OptiTrack file" )
        self.optitrack_btn.clicked.connect( self.load_optitrack_data )
        self.left_toolbar.addWidget( self.optitrack_btn )

        self.show_position_btn = QRadioButton( "Show Position" )
        self.show_position_btn.setChecked( True )
        self.show_position_btn.toggled.connect( self.update_ )
        self.left_toolbar.addWidget( self.show_position_btn )

        self.show_rotation_btn = QRadioButton( "Show Rotation" )
        self.show_rotation_btn.toggled.connect( self.update_ )
        self.left_toolbar.addWidget( self.show_rotation_btn )

        self.available_items_area = QScrollArea()
        self.left_toolbar.addWidget( self.available_items_area )

        self.available_items_container = QWidget()
        self.available_items_area.setWidget( self.available_items_container )
        self.available_items_area.setWidgetResizable( True )

        self.available_items_layout = QVBoxLayout( self.available_items_container )
        self.available_items_layout.setAlignment( Qt.AlignmentFlag.AlignTop )
        self.available_items: dict[QCheckBox, dict] = {}

        self.left_toolbar.addWidget( QLabel( "Set:" ) )

        self.affected = QComboBox()
        self.left_toolbar.addWidget( self.affected )

        self.left_toolbar.addWidget( QLabel( "...to match..." ) )

        self.affector = QComboBox()
        self.left_toolbar.addWidget( self.affector )

        self.sync_x_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.sync_x_layout )

        self.sync_x_btn = QPushButton( "Sync X" )
        self.sync_x_btn.clicked.connect( lambda : self.sync( 'x' ) )
        self.sync_x_layout.addWidget( self.sync_x_btn )

        self.sync_x_offset = QLabel()
        self.sync_x_layout.addWidget( self.sync_x_offset )

        self.sync_y_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.sync_y_layout )

        self.sync_y_btn = QPushButton( "Sync Y" )
        self.sync_y_btn.clicked.connect( lambda : self.sync( 'y' ) )
        self.sync_y_layout.addWidget( self.sync_y_btn )

        self.sync_y_offset = QLabel()
        self.sync_y_layout.addWidget( self.sync_y_offset )

        self.sync_z_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.sync_z_layout )

        self.sync_z_btn = QPushButton( "Sync Z" )
        self.sync_z_btn.clicked.connect( lambda : self.sync( 'z' ) )
        self.sync_z_layout.addWidget( self.sync_z_btn )

        self.sync_z_offset = QLabel()
        self.sync_z_layout.addWidget( self.sync_z_offset )

        self.sync_yw_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.sync_yw_layout )

        self.sync_yw_btn = QPushButton( "Sync Yaw" )
        self.sync_yw_btn.clicked.connect( lambda : self.sync( 'yaw' ) )
        self.sync_yw_layout.addWidget( self.sync_yw_btn )

        self.sync_yw_offset = QLabel()
        self.sync_yw_layout.addWidget( self.sync_yw_offset )

        self.sync_p_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.sync_p_layout )

        self.sync_p_btn = QPushButton( "Sync Pitch" )
        self.sync_p_btn.clicked.connect( lambda : self.sync( 'pitch' ) )
        self.sync_p_layout.addWidget( self.sync_p_btn )

        self.sync_p_offset = QLabel()
        self.sync_p_layout.addWidget( self.sync_p_offset )

        self.sync_r_layout = QHBoxLayout()
        self.left_toolbar.addLayout( self.sync_r_layout )

        self.sync_r_btn = QPushButton( "Sync Roll" )
        self.sync_r_btn.clicked.connect( lambda : self.sync( 'roll' ) )
        self.sync_r_layout.addWidget( self.sync_r_btn )

        self.sync_r_offset = QLabel()
        self.sync_r_layout.addWidget( self.sync_r_offset )

        self.hline = QFrame()
        self.hline.setFrameShape( QFrame.Shape.HLine )
        self.left_toolbar.addWidget( self.hline )

        self.final_values = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }

        self.xyz_final = QLabel()
        self.left_toolbar.addWidget( self.xyz_final )

        self.rpy_final = QLabel()
        self.left_toolbar.addWidget( self.rpy_final )

        self.left_toolbar.addStretch()

        # Build main plot area
        self.plot_area = QVBoxLayout()
        self.main_area.addLayout( self.plot_area )

        self.x_plot = LinePlot( 
            title = 'X-Position',
            x_axis_title = 'Sample',
            y1_axis_title = 'OptiTrack Position (m)',
            print_func = self._show_notification
        )
        self.x_plot_area = QWebEngineView()

        self.y_plot = LinePlot( 
            title = 'Y-Position',
            x_axis_title = 'Sample',
            y1_axis_title = 'OptiTrack Position (m)',
            print_func = self._show_notification
        )
        self.y_plot_area = QWebEngineView()

        self.z_plot = LinePlot( 
            title = 'Z-Position',
            x_axis_title = 'Sample',
            y1_axis_title = 'OptiTrack Position (m)',
            print_func = self._show_notification
        )
        self.z_plot_area = QWebEngineView()

        self.yaw_plot = LinePlot( 
            title = 'Yaw (Global OptiTrack)',
            x_axis_title = 'Sample',
            y1_axis_title = 'Yaw Angle (deg)',
            print_func = self._show_notification
        )
        self.yaw_plot_area = QWebEngineView()

        self.pitch_plot = LinePlot( 
            title = 'Pitch (Global OptiTrack)',
            x_axis_title = 'Sample',
            y1_axis_title = 'Pitch Angle (deg)',
            print_func = self._show_notification
        )
        self.pitch_plot_area = QWebEngineView()

        self.roll_plot = LinePlot( 
            title = 'Roll (Global OptiTrack)',
            x_axis_title = 'Sample',
            y1_axis_title = 'Roll (deg)',
            print_func = self._show_notification
        )
        self.roll_plot_area = QWebEngineView()

    def update_( self, *args ):

        self.update_plots()

        if( self.show_position_btn.isChecked() ):
            try:
                self.plot_area.removeWidget( self.roll_plot_area )
                self.plot_area.removeWidget( self.pitch_plot_area )
                self.plot_area.removeWidget( self.yaw_plot_area )
            except:
                pass

            self.plot_area.addWidget( self.x_plot_area )
            html_plot = pio.to_html( self.x_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.x_plot_area.setHtml( html_plot )

            self.plot_area.addWidget( self.y_plot_area )
            html_plot = pio.to_html( self.y_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.y_plot_area.setHtml( html_plot )

            self.plot_area.addWidget( self.z_plot_area )
            html_plot = pio.to_html( self.z_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.z_plot_area.setHtml( html_plot )

        else:
            try:
                self.plot_area.removeWidget( self.x_plot_area )
                self.plot_area.removeWidget( self.y_plot_area )
                self.plot_area.removeWidget( self.z_plot_area )
            except:
                pass
            
            self.plot_area.addWidget( self.roll_plot_area )
            html_plot = pio.to_html( self.roll_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.roll_plot_area.setHtml( html_plot )

            self.plot_area.addWidget( self.pitch_plot_area )
            html_plot = pio.to_html( self.pitch_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.pitch_plot_area.setHtml( html_plot )

            self.plot_area.addWidget( self.yaw_plot_area )
            html_plot = pio.to_html( self.yaw_plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.yaw_plot_area.setHtml( html_plot )

        self.xyz_final.setText(  f"( {self.final_values['x']:.3f}, {self.final_values['y']:.3f}, {self.final_values['z']:.3f} )"  )
        self.rpy_final.setText( f"( {self.final_values['roll']}, {self.final_values['pitch']}, {self.final_values['yaw']} )" )

    def update_plots( self ):

        for item in self.data.keys():
            if( item.isChecked() ):
                try:
                    self.x_plot.remove_y1_trace( item.text() )
                    self.y_plot.remove_y1_trace( item.text() )
                    self.z_plot.remove_y1_trace( item.text() )
                    self.yaw_plot.remove_y1_trace( item.text() )
                    self.pitch_plot.remove_y1_trace( item.text() )
                    self.roll_plot.remove_y1_trace( item.text() )
                except: pass

                self.x_plot.add_y1_trace( self.data[ item ]['x'], item.text() )                    
                self.y_plot.add_y1_trace( self.data[ item ]['y'], item.text() )                    
                self.z_plot.add_y1_trace( self.data[ item ]['z'], item.text() )                    
                self.yaw_plot.add_y1_trace( self.data[ item ]['yaw'], item.text() )                    
                self.pitch_plot.add_y1_trace( self.data[ item ]['pitch'], item.text() )                    
                self.roll_plot.add_y1_trace( self.data[ item ]['roll'], item.text() ) 

    def load_optitrack_data( self ):
        try:
            file_dialog = QFileDialog( self )
            file_path, _ = file_dialog.getOpenFileName( self, "Select OptiTrack .log file" )

            if( os.path.isfile( file_path ) ):
                data_by_time = self._optitrack.parse_log( file_path )

                # Clear old data
                self.cb_by_item.clear()
                self.data.clear()


                # Load new data
                for timestamp in data_by_time.keys():
                    for item in data_by_time[timestamp].keys():
                        if( item not in self.cb_by_item.keys() ):

                            self.cb_by_item[item] = QCheckBox( item )
                            self.cb_by_item[item].clicked.connect( lambda x, s = self.cb_by_item[item] : self.update_( s, x ) )
                            self.available_items_layout.addWidget( self.cb_by_item[item] )

                            se3 = data_by_time[timestamp][item]
                            pos = se3[3:][:3][0]
                            rpy = mat_ops.get_roll_pitch_yaw_deg( se3[:3, :3] )

                            self.data[self.cb_by_item[item]] = {
                                'x': [ pos[0] ],
                                'y': [ pos[1] ],
                                'z': [ pos[2] ],
                                'roll': [ rpy['roll'] ],
                                'pitch': [ rpy['pitch'] ],
                                'yaw': [ rpy['yaw'] ]
                            }
                        
                        else:

                            se3 = data_by_time[timestamp][item]
                            pos = se3[3:][:3][0]
                            rpy = mat_ops.get_roll_pitch_yaw_deg( se3[:3, :3] )

                            self.data[self.cb_by_item[item]]['x'].append( pos[0] )
                            self.data[self.cb_by_item[item]]['y'].append( pos[1] )
                            self.data[self.cb_by_item[item]]['z'].append( pos[2] )
                            self.data[self.cb_by_item[item]]['roll'].append( rpy['roll'] )
                            self.data[self.cb_by_item[item]]['pitch'].append( rpy['pitch'] )
                            self.data[self.cb_by_item[item]]['yaw'].append( rpy['yaw'] )

            for item in self.data.keys():
                
                self.data[item]['x'] = np.array( self.data[item]['x'] )
                self.data[item]['y'] = np.array( self.data[item]['y'] )
                self.data[item]['z'] = np.array( self.data[item]['z'] )
                self.data[item]['roll'] = np.array( self.data[item]['roll'] )
                self.data[item]['pitch'] = np.array( self.data[item]['pitch'] )
                self.data[item]['yaw'] = np.array( self.data[item]['yaw'] )
                
                self.affected.addItem( item.text() )
                self.affector.addItem( item.text() )

        except Exception as e:
            self._show_notification( f"Failed to load OptiTrack file:\n\t{type(e)}: {e}" )

    def sync( self, attribute: str ):

        affector_name = self.affector.currentText()
        affected_name = self.affected.currentText()

        try:
            if( attribute not in self.data[self.cb_by_item[affector_name]] or attribute not in self.data[self.cb_by_item[affected_name]] ):
                self._show_notification( f"{attribute} is not a valid attribute of the items selected." )

            offset_vector = self.data[self.cb_by_item[affected_name]][attribute] - self.data[self.cb_by_item[affector_name]][attribute]
            self._show_notification( str( offset_vector ) )

            match( attribute ):
                case 'x':
                    self.sync_x_offset.setText( f'Mean: {np.mean( offset_vector ):.3f} | Var: {np.var( offset_vector ):.3f}' )
                    self.final_values['x'] += np.mean( offset_vector )
                case 'y':
                    self.sync_y_offset.setText( f'Mean: {np.mean( offset_vector ):.3f} | Var: {np.var( offset_vector ):.3f}' )
                    self.final_values['y'] += np.mean( offset_vector )
                case 'z':
                    self.sync_z_offset.setText( f'Mean: {np.mean( offset_vector ):.3f} | Var: {np.var( offset_vector ):.3f}' )
                    self.final_values['z'] += np.mean( offset_vector )
                case 'yaw':
                    self.sync_yw_offset.setText( f'Mean: {np.mean( offset_vector ):.3f} | Var: {np.var( offset_vector ):.3f}' )
                    self.final_values['roll'] += np.mean( offset_vector )
                case 'pitch':
                    self.sync_p_offset.setText( f'Mean: {np.mean( offset_vector ):.3f} | Var: {np.var( offset_vector ):.3f}' )
                    self.final_values['pitch'] += np.mean( offset_vector )
                case 'roll':
                    self.sync_r_offset.setText( f'Mean: {np.mean( offset_vector ):.3f} | Var: {np.var( offset_vector ):.3f}' )
                    self.final_values['yaw'] += np.mean( offset_vector )

            
            self.data[self.cb_by_item[affected_name]][attribute] = self.data[self.cb_by_item[affected_name]][attribute] - np.mean( offset_vector )

            self.update_()

        except: pass