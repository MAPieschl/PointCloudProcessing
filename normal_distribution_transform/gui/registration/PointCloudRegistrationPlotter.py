import os

from gui.qt_dependencies import *
from utils.mat_ops import *
from utils.plotting import *
from copy import deepcopy

from mesh.MeshSampler import MeshSampler

import plotly.io as pio

class IterativeEvaluator:
    def __init__( self, 
                 reference_pc: list[np.ndarray], 
                 reference_labels: list[str], 
                 reference_se3_list: list[np.ndarray],
                 target_pc: list[np.ndarray], 
                 target_labels: list[str], 
                 target_se3_list: list[np.ndarray],
                 print_func: Callable[[str], None] = print ):

        assert len( reference_pc ) == len( reference_labels ), f'There must be one label for each group of reference points, not {len( reference_labels )} labels and {len( reference_pc )} points.'
        assert len( target_pc ) == len( target_labels ), f'There must be one label for each group of target points, not {len( reference_labels )} labels and {len( reference_pc )} points.'
        assert len( reference_se3_list ) == len( target_se3_list ), f'There must be an equal number of transformations listed for the reference and target point clouds.'
        for pc in reference_pc:
            if( pc.size > 0 ):
                assert pc.ndim == 2 and pc.shape[1] == 3, f'Listed point clouds must have shape (N, 3), not {pc.shape}'
        for pc in target_pc:
            if( pc.size > 0 ):
                assert pc.ndim == 2 and pc.shape[1] == 3, f'Listed point clouds must have shape (N, 3), not {pc.shape}'

        self.__ref_pc = []
        self.__ref_lb = []
        for i in range( len( reference_pc ) ):
            if( reference_pc[i].size > 0 ):
                self.__ref_pc.append( reference_pc[i] )
                self.__ref_lb.append( reference_labels[i] )

        self.__tar_pc = []
        self.__tar_lb = []
        for i in range( len( target_pc ) ):
            if( target_pc[i].size > 0 ):
                self.__tar_pc.append( target_pc[i] )
                self.__tar_lb.append( target_labels[i] )

        self.__num_iterations = len( reference_se3_list )
        self.__ref_se3 = reference_se3_list
        self.__tar_se3 = target_se3_list

        self.__show_notification = print_func

    def get_num_iterations( self ) -> int:
        return self.__num_iterations
    
    def get_plot( self, iteration: int,  title: str = '' ) -> go.Figure:
        if( iteration < self.__num_iterations ):
            ref_pc = self.__get_transformed_reference_point_cloud( self.__ref_se3[iteration] )
            tar_pc = self.__get_transformed_target_point_cloud( self.__tar_se3[iteration] )

            return display_point_clouds( ref_pc + tar_pc, self.__ref_lb + self.__tar_lb, title = title )
        
        else:
            self.__show_notification( f'Iteration {iteration} requested, but only {self.__num_iterations} exist.' )
            return go.Figure()
    
    def __get_transformed_reference_point_cloud( self, se3: np.ndarray ):
        new_ref_pc = []
        for pc in self.__ref_pc:
            new_ref_pc.append( transform_pc( pc, se3 ) )

        return new_ref_pc
    
    def __get_transformed_target_point_cloud( self, se3: np.ndarray ):
        new_tar_pc = []
        for pc in self.__tar_pc:
            new_tar_pc.append( transform_pc( pc, se3 ) )

        return new_tar_pc

class PointCloudRegistrationPlotter( QMainWindow ):
    def __init__( self, 
                 reference_pc: list[np.ndarray], 
                 reference_labels: list[str], 
                 reference_se3_list: list[np.ndarray],
                 target_pc: list[np.ndarray], 
                 target_labels: list[str],
                 target_se3_list: list[np.ndarray]  ):
        
        super().__init__()
        self.setWindowTitle( 'Point Cloud Registration' )
        
        # Central widget for managing different views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget( self.stacked_widget )
        
        # Add views to the stacked widget
        self.plotter = QWebEngineView()
        self.stacked_widget.addWidget( self.plotter )
        self.stacked_widget.setCurrentWidget( self.plotter )

        # Create Object
        self.it_eval = IterativeEvaluator( reference_pc, reference_labels, reference_se3_list, target_pc, target_labels, target_se3_list, self.show_notification )
        
        # Create and configure the toolbar
        self.toolbar = QToolBar( "Main Toolbar" )
        self.addToolBar( Qt.ToolBarArea.BottomToolBarArea, self.toolbar )
        self.toolbar.setMovable( False )
        self.toolbar.setFloatable( False )

        # Create a slider to show various iteration frames
        self.iteration_slider = QSlider()
        self.iteration_slider.setOrientation( Qt.Orientation.Horizontal )
        self.iteration_slider.setRange( 0, self.it_eval.get_num_iterations() - 1 )
        self.iteration_slider.valueChanged.connect( self.update_ )
        self.toolbar.addWidget( self.iteration_slider )

        self.plotter.resize( 800, 600 )

        self.__initialize_plot()
        self.update_()

    def update_( self ):
        fig = self.it_eval.get_plot( self.iteration_slider.value() ).to_json()

        js_cmd = f"""
            var new_fig = {fig}
            Plotly.react('it_eval_plot', new_fig.data, new_fig.layout)
        """

        pg = self.plotter.page()
        if( pg is not None ): pg.runJavaScript( js_cmd )

    def get_iterative_evaluator_object( self ):
        return self.it_eval

    def show_notification( self, msg: str ):
        popup = QMessageBox( self )
        popup.setWindowTitle( "Notification" )
        popup.setText( msg )
        popup.setIcon( QMessageBox.Icon.Information )
        popup.setStandardButtons( QMessageBox.StandardButton.Ok )

        response = popup.exec()

    def show_yes_no_query( self, msg: str ) -> int:
        popup = QMessageBox( self )
        popup.setWindowTitle( "Query" )
        popup.setText( msg )
        popup.setIcon( QMessageBox.Icon.Question )
        popup.setStandardButtons( QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No )

        return popup.exec()
    
    def __initialize_plot( self ):

        fig = self.it_eval.get_plot( self.iteration_slider.value() )
        html_plot = html_plot = pio.to_html( self.it_eval.get_plot( self.iteration_slider.value() ), full_html = True, include_plotlyjs = 'include', div_id = 'it_eval_plot' )
        
        html_file = os.path.abspath( 'temp/plotly_render.html' )
        with open( html_file, 'w', encoding = 'utf-8' ) as f:
            f.write( html_plot )
        
        local_url = QUrl.fromLocalFile( html_file )
        self.plotter.setUrl( local_url )