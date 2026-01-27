from dependencies import *

from utils.custom_plotting import MatplotlibCanvas, QuiverPlot

class SemanticMeshICP( QWidget ):
    def __init__( self, parent ):
        super().__init__( parent )

        # Define parent functors
        self._show_notification = parent.show_notification
        self._show_yes_no_query = parent.show_yes_no_query

        # Build GUI
        self.main_layout, self.left_toolbar, self.main_area = parent.get_left_toolbar_layout( self, "SemanticMeshICP", False )

        # Left toolbar build
        self.load_obj_btn = QPushButton( "Load Reference Mesh (.obj)" )
        self.load_obj_btn.clicked.connect( self.load_obj )
        self.left_toolbar.addWidget( self.load_obj_btn )

        self.quiver_plot = QuiverPlot( "SemanticMeshICP", self._show_notification )
        self.quiver_plot_area = MatplotlibCanvas( parent = self )
        self.main_area.addWidget( self.quiver_plot_area )

    def update_( self ):

        self.quiver_plot.draw_matplotlib_fig( self.quiver_plot_area )

    def load_obj( self ):
        try:

            file_dialog = QFileDialog( self )
            file_path, _ = file_dialog.getOpenFileName( self, "Select .obj file" )
            
            if( os.path.isfile( file_path ) and os.path.splitext( file_path )[1] == '.obj' ):
                scene: trimesh.Scene = trimesh.load_scene( file_path )

                for name, mesh in scene.geometry.items():

                    verts = np.array( mesh.vertices )
                    verts = np.concatenate( [verts, np.array( mesh.vertex_normals )], axis = 1 )

                    self.quiver_plot.add_data( verts, name )
                    
            self.update_()

        except Exception as e:
            self._show_notification( f"Unable to load reference mesh:\t\n{type( e )}: {e}" )