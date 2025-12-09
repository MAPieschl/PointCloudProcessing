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

        self.plot_select = QComboBox()
        self.left_toolbar.addWidget( self.plot_select )
        
        self.left_toolbar.addStretch()

        # Build plot
        self.plot = None
        self.data = {}
        self.plot_area = QWebEngineView()
        self.main_area.addWidget( self.plot_area )

    def update_( self ):

        if( type( self.plot ) == LinePlot ):
            html_plot = pio.to_html( self.plot.get_fig(), full_html = False, include_plotlyjs = True )
            self.main_area.setHtml( html_plot )
    
    def load_training_history( self ):
        try:

            file_dialog = QFileDialog(self)
            model_path = file_dialog.getExistingDirectory( self, "Select model training directory", "" )
            
            if( os.path.isdir( model_path ) ):

                history = glob.glob( os.path.join( model_path, "*_history.json") )
                
                if( len( history ) < 1 ):
                    self._show_notification( "Please select a directory that contains the model training ..._history.json." )
                    return
                
                with open( history[0], 'r' ) as hist:
                    history = json.load( hist )
                
                for info in list( history.keys() ):
                    for output in globals.MODEL_OUTPUTS:
                        i = info.find( output )
                        if( i >= 0 ):
                            if( output not in self.data ):
                                self.data[output] = {}
                            


                model_name = model_path.split( '/' )[-1]

                self.plot = LinePlot( title = f'Training Performance for {model_name}',
                                      x_axis_title = 'Epochs',
                                      y1_axis_title = 'Loss',
                                      y2_axis_title = 'Metric' )

            else:
                self._show_notification( "Model directory no longer exists." )

            self.update_()

        except Exception as e:
            self._show_notification( f"GUI:  Unable to load point cloud due to error:\n\t{type( e ).__name__}: {e}" )