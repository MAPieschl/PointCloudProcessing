from gui.qt_dependencies import *
from utils.mat_ops import *
from utils.plotting import *
from copy import deepcopy

from mesh.MeshSampler import MeshSampler

import plotly.io as pio

class IterativeEvaluator:
    def __init__( self, reference_pc: list[np.ndarray], reference_labels: list[str], target_pc: list[np.ndarray], target_labels: list[str] ):

        assert len( reference_pc ) == len( reference_labels ), f'There must be one label for each group of reference points, not {len( reference_labels )} labels and {len( reference_pc )} points.'
        assert len( target_pc ) == len( target_labels ), f'There must be one label for each group of target points, not {len( reference_labels )} labels and {len( reference_pc )} points.'
        for pc in reference_pc:
            assert pc.ndim == 2 and pc.shape[1] == 3, f'Listed point clouds must have shape (N, 3), not {pc.shape}'
        for pc in target_pc:
            assert pc.ndim == 2 and pc.shape[1] == 3, f'Listed point clouds must have shape (N, 3), not {pc.shape}'

        self.__num_iterations = 0
        self.__ref_pc = reference_pc
        self.__ref_lb = reference_labels
        self.__tar_pc = target_pc
        self.__tar_lb = target_labels

    def get_num_iterations( self ):
        return self.__num_iterations
    
    def get_transformed_reference_point_cloud( self, se3: np.ndarray ):
        new_ref_pc = []
        for pc in self.__ref_pc:
            new_ref_pc.append( transform_pc( pc, se3 ) )

        return new_ref_pc
    
    def get_reference_labels( self ):
        return self.__ref_lb
    
    def get_transformed_target_point_cloud( self, se3: np.ndarray ):
        new_tar_pc = []
        for pc in self.__tar_pc:
            new_tar_pc.append( transform_pc( pc, se3 ) )

        return new_tar_pc
    
    def get_target_labels( self ):
        return self.__tar_lb

class PointCloudRegistrationPlotter( QMainWindow ):
    def __init__( self ):
        super().__init__()
        self.setWindowTitle( 'Point Cloud Registration' )
        
        # Central widget for managing different views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget( self.stacked_widget )
        
        # Add views to the stacked widget
        self.plotter = QWebEngineView()
        self.stacked_widget.addWidget( self.plotter )

        # Create Object
        self.it_eval = IterativeEvaluator()
        
        # Create and configure the toolbar
        self.toolbar = QToolBar( "Main Toolbar" )
        self.addToolBar( Qt.ToolBarArea.BottomToolBarArea, self.toolbar )
        self.toolbar.setMovable( False )
        self.toolbar.setFloatable( False )

    def update_( self ):
        html_plot = pio.to_html( self.plotter.get_fig(), full_html = False, include_plotlyjs = 'cdn' )
        self.pc_plot_area.setHtml( html_plot )

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