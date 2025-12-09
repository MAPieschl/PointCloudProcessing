from dependencies import *

from utils.custom_plotting import LinePlot

class TrainingPerformance( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "Training Performance" )

        # Build toolbar
        self.model_btn = QPushButton( "Select Model Training Directory" )
        self.model_btn.clicked.connect( self.load_training_history )
        self.left_toolbar.addWidget( self.model_btn )
        
        self.available_traces_area = QScrollArea()
        self.left_toolbar.addWidget( self.available_traces_area )

        self.available_traces_container = QWidget()
        self.available_traces_area.setWidget( self.available_traces_container )
        self.available_traces_area.setWidgetResizable( True )

        self.available_traces_layout = QVBoxLayout( self.available_traces_container )
        self.available_traces_layout.setAlignment( Qt.AlignmentFlag.AlignTop )
        self.available_traces: dict[QCheckBox, str] = {}

        self.left_toolbar.addStretch()

        # Build plot
        self.plot = None
        self.data = {}
        self.plot_area = QWebEngineView()
        self.main_area.addWidget( self.plot_area )

    def update_( self, *args ):
        
        if( len( args ) >= 2 and type( self.plot ) == LinePlot ):
            if( type( args[0] ) == str and type( args[1] ) == bool ):
                if( 'accuracy' in args[0] or 'error' in args[0] ):
                    if( args[1] ):  self.plot.add_y1_trace( np.array( self.data[args[0]] ), args[0] )
                    else:           self.plot.remove_y1_trace( args[0] )

        if( type( self.plot ) == LinePlot ):
            html_plot = pio.to_html( self.plot.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
            self.plot_area.setHtml( html_plot )
    
    def load_training_history( self ):
        # try:

        file_dialog = QFileDialog(self)
        model_path = file_dialog.getExistingDirectory( self, "Select model training directory", "" )
        
        if( os.path.isdir( model_path ) ):

            history = glob.glob( os.path.join( model_path, "*_history.json") )
            
            if( len( history ) < 1 ):
                self._show_notification( "Please select a directory that contains the model training ..._history.json." )
                return
            
            with open( history[0], 'r' ) as hist:
                self.data = json.load( hist )
            
            self.available_traces.clear()
            for key in list( self.data.keys() ):
                cb = QCheckBox( key )
                self.available_traces[cb] = key
                cb.stateChanged.connect( lambda x: self.update_( self.available_traces[cb], x == 2 ) )
                self.available_traces_layout.addWidget( cb )
                        
            model_name = model_path.split( '/' )[-1]

            self.plot = LinePlot( title = f'Training Performance for {model_name}',
                                    x_axis_title = 'Epochs',
                                    y1_axis_title = 'Loss',
                                    y2_axis_title = 'Metric',
                                    print_func = self._show_notification )

        else:
            self._show_notification( "Model directory no longer exists." )

        self.update_()

        # except Exception as e:
        #     self._show_notification( f"GUI:  Unable to load point cloud due to error:\n\t{type( e ).__name__}: {e}" )