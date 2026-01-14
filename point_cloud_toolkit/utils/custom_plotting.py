from dependencies import *
import utils.globals as globals

class LinePlot:
    def __init__( self,
                   title: str = "", 
                   x_axis_title: str = "", 
                   y1_axis_title: str = "", 
                   y2_axis_title: str = "", 
                   print_func: Callable[[str], None] = print ):
          
        self._title = title
        self._x_axis_title = x_axis_title
        self._y1_axis_title = y1_axis_title
        self._y2_axis_title = y2_axis_title
        self._print = print_func

        self._data_y1: list[np.ndarray] = []
        self._labels_y1: list[str]      = []
        self._data_y2: list[np.ndarray] = []
        self._labels_y2:list[str]       = []

    def add_y1_trace( self, data: np.ndarray, label: str ) -> None:

        if( len( data.shape ) != 1 ):
            self._print( "ERROR IN LINE_PLOT:  'data' must be input as a 1D array." )
            return
        
        self._data_y1.append( data )
        self._labels_y1.append( label )

    def add_y2_trace( self, data: np.ndarray, label: str ) -> None:

        if( len( data.shape ) != 1 ):
            self._print( "ERROR IN LINE_PLOT:  'data' must be input as a 1D array." )
            return
        
        self._data_y2.append( data )
        self._labels_y2.append( label )

    def remove_y1_trace( self, label: str ) -> None:
        
        idx = self._labels_y1.index( label )

        if( idx >= 0 ):
            self._data_y1.pop( idx )
            self._labels_y1.pop( idx )

    def remove_y2_trace( self, label: str ) -> None:
        
        idx = self._labels_y2.index( label )

        if( idx >= 0 ):
            self._data_y2.pop( idx )
            self._labels_y2.pop( idx )

    def get_fig( self ) -> go.Figure:
        
        fig = go.Figure()

        y1_maxes = [i.shape[0] for i in self._data_y1] if len( self._data_y1 ) > 0 else [ 1 ]
        y2_maxes = [i.shape[0] for i in self._data_y2] if len( self._data_y2 ) > 0 else [ 1 ]

        x = max( max( y1_maxes ), max( y2_maxes ) )
        x = np.linspace( 1, x, x )

        if( len( self._data_y1 ) > 0 ):

            for i, trace in enumerate( self._data_y1 ):

                fig.add_trace(
                    go.Scatter(
                        x = x,
                        y = trace,
                        mode = 'lines',
                        name = self._labels_y1[i],
                        yaxis = 'y1'
                    )
                )

        if( len( self._data_y2 ) > 0 ):
             
            for i, trace in enumerate( self._data_y2 ):

                fig.add_trace(
                    go.Scatter(
                        x = x,
                        y = trace,
                        mode = 'lines',
                        name = self._labels_y2[i],
                        yaxis = 'y2'
                    )
                )

            fig.update_layout(
                title = self._title,
                xaxis_title = self._x_axis_title,
                yaxis = dict( title = self._y1_axis_title ),
                yaxis2 = dict(
                    title = self._y2_axis_title,
                    overlaying = 'y',
                    side = 'right'
                )
            )

        else:

            fig.update_layout(
                title = self._title,
                xaxis_title = self._x_axis_title,
                yaxis = dict( title = self._y1_axis_title ),
            ) 

        return fig
    
    def show( self ) -> None:

        fig = self.get_fig()
        fig.show()

class PointCloudPlot:
    def __init__( self,
                   title: str = "", 
                   print_func: Callable[[str], None] = print ):
          
        self._title = title
        self._print = print_func

        self._size_lims: list[int] = [1, 10]
        self._opacity_lims: list[float] = [0.0, 1.0]

        self._data = np.array( [] )
        self._colors = np.array( [] )
        self._tags = np.array( [] )
        self._size = np.array( [] )
        self._opacity = np.array( [] )

        self._red_points = np.array( [] )
        self._red_tags = np.array( [] )
        self._red_size = np.array( [] )

        self._filter = None

        self._color_filter = None
        self._radius_filter = None

    def add( self, data: np.ndarray, color: np.ndarray, tag: str, size: int = 5, opacity: float = 1.0 ) -> None:

        size = np.clip( size, self._size_lims[0], self._size_lims[1], dtype = type( self._size_lims[0] ) )
        opacity = np.clip( opacity, self._opacity_lims[0], self._opacity_lims[1], dtype = type( self._opacity_lims[0] ) )

        if( self._data.shape[0] < 1 or self._colors.shape[0] < 1 or self._tags.shape[0] < 1 ):
            self._data = data
            self._colors = color
            self._tags = np.array( [tag for i in range( data.shape[0] )] )
            self._size = np.array( [size for i in range( data.shape[0] )] )
            self._opacity = np.array( [opacity for i in range( data.shape[0] )] )

        else:
            self._data = np.concatenate( ( self._data, data ), axis = 0 )
            self._colors = np.concatenate( ( self._colors, color ), axis = 0 )
            self._tags = np.concatenate( ( self._tags, np.array( [tag for i in range( data.shape[0] )] ) ), axis = 0 )
            self._size = np.concatenate( ( self._size, np.array( [size for i in range( data.shape[0] )] ) ), axis = 0 )
            self._opacity = np.concatenate( ( self._opacity, np.array( [opacity for i in range( data.shape[0] )] ) ), axis = 0 )

    def add_red_point( self, pt: np.ndarray, tag: str = '', size: int = 5 ):

        size = np.clip( size, self._size_lims[0], self._size_lims[1], dtype = type( self._size_lims[0] ) )

        if( self._red_points.shape[0] < 1 or self._red_tags.shape[0] < 1 or self._red_size.shape[0] < 1 ):
            self._red_points = np.array( [pt] )
            self._red_tags = np.array( [tag] )
            self._red_size = np.array( [size] )

        else:
            self._red_points = np.concatenate( ( self._red_points, np.array( [pt] ) ), axis = 0 )
            self._red_tags = np.concatenate( ( self._red_tags, np.array( [tag] ) ), axis = 0 )
            self._size = np.concatenate( ( self._red_size, np.array( [size] ) ), axis = 0 )

    def clear_red_points( self ):
        self._red_points = np.array( [] )
        self._red_tags = np.array( [] )
        self._red_size = np.array( [] )

    def clear_filter( self ):
        self._filter = None
        self._radius_filter = None
        self._color_filter = None

    def clear( self ):

        self._data = np.array( [] )
        self._colors = np.array( [] )
        self._tags = np.array( [] )
        self._size = np.array( [] )
        self._opacity = np.array( [] )
        # self.clear_filter()
        self.clear_red_points()

    def remove( self, tag: str ):

        indices = np.where( self._tags == tag )

        self._data = np.delete( self._data, indices, axis = 0 )
        self._colors = np.delete( self._colors, indices, axis = 0 )
        self._tags = np.delete( self._tags, indices, axis = 0 )
        self._size = np.delete( self._size, indices, axis = 0 )
        self._opacity = np.delete( self._opacity, indices, axis = 0 )

    def set_size( self, tag: str, size: int ) -> None:

        indices = np.where( self._tags == tag )
        self._size[indices] = np.clip( size, self._size_lims[0], self._size_lims[1], dtype = type( self._size_lims[0] ) )

    def set_opacity( self, tag: str, opacity: float ) -> None:

        indices = np.where( self._tags == tag )
        self._opacity[indices] = np.clip( opacity, self._opacity_lims[0], self._opacity_lims[1], dtype = type( self._opacity_lims[0] ) )

    def filter_by_radius( self, center: np.ndarray, radius: float ):

        self._radius_filter = np.sum( ( self._data - center ) ** 2, axis = 1 ) < radius ** 2

    def filter_by_color( self, value: float, show_greater_than: bool = True ):

        if( 0 <= value <= 100 ):
            _min = min( self._colors )
            _max = max( self._colors )

            _threshold = ( ( value / 100 ) * ( _max - _min ) + _min )
            if( show_greater_than ):
                self._color_filter = self._colors > _threshold

            else:
                self._color_filter = self._colors <= _threshold

        else:
            self._print( f"PointCloudPlot:  filter_by_color requires a value in range [0, 100], not {value}" )

    def get_points( self ):

        return self._data[self._filter]
    
    def get_max_radius_from( self, value: np.ndarray ):

        return np.max( np.sqrt( np.sum( ( self._data - value ) ** 2, axis = 1 ) ) )

    def get_fig( self ) -> go.Figure:
        
        fig = go.Figure()

        # Preliminary scale for empty plot
        x_lims = [-10, 10]
        y_lims = [-10, 10]
        z_lims = [-10, 10]

        if( self._color_filter is not None and self._radius_filter is not None ):   self._filter = self._radius_filter & self._color_filter
        elif( self._color_filter is not None ):                                     self._filter = self._color_filter
        elif( self._radius_filter is not None ):                                    self._filter = self._radius_filter
        else:                                                                       self._filter = None

        # self._print( f"filter: {self._filter} {self._filter is not None}\nradius_filter: {self._radius_filter} {self._radius_filter is not None}\ncolor_filter: {self._color_filter} {self._color_filter is not None}" )

        if( self._data.shape[0] > 0 ):

            # Set limits before filtering to maintain plot scale
            x_lims = [np.min( self._data[:, 0] ), np.max( self._data[:, 0] )]
            y_lims = [np.min( self._data[:, 1] ), np.max( self._data[:, 1] )]
            z_lims = [np.min( self._data[:, 2] ), np.max( self._data[:, 2] )]

            fig.add_trace(
                go.Scatter3d(
                    x = self._data[:, 0] if self._filter is None else self._data[self._filter, 0],
                    y = self._data[:, 1] if self._filter is None else self._data[self._filter, 1],
                    z = self._data[:, 2] if self._filter is None else self._data[self._filter, 2],
                    mode = 'markers',
                    marker = dict(
                        size = self._size if self._filter is None else self._size[self._filter],
                        color = self._colors if self._filter is None else self._colors[self._filter],
                        colorscale = 'Viridis'
                    )
                )
            )

        if( self._red_points.shape[0] > 0 ):

            # Expand limits if required
            if( np.max( self._red_points[:, 0] ) > x_lims[1] ):   x_lims[1] = np.max( self._red_points[:, 0] )
            if( np.min( self._red_points[:, 0] ) < x_lims[0] ):   x_lims[0] = np.min( self._red_points[:, 0] )
            if( np.max( self._red_points[:, 0] ) > y_lims[1] ):   y_lims[1] = np.max( self._red_points[:, 1] )
            if( np.min( self._red_points[:, 0] ) < y_lims[0] ):   y_lims[0] = np.min( self._red_points[:, 1] )
            if( np.max( self._red_points[:, 0] ) > z_lims[1] ):   z_lims[1] = np.max( self._red_points[:, 2] )
            if( np.min( self._red_points[:, 0] ) < z_lims[0] ):   z_lims[0] = np.min( self._red_points[:, 2] )

            fig.add_trace(
                go.Scatter3d(
                    x = self._red_points[:, 0],
                    y = self._red_points[:, 1],
                    z = self._red_points[:, 2],
                    mode = 'markers',
                    marker = dict(
                        size = self._red_size,
                        color = 'red',
                    )
                )
            )

        fig.update_layout(
            title = self._title,
            # scene = dict(
            #     xaxis = dict( range = x_lims ),
            #     yaxis = dict( range = y_lims ),
            #     zaxis = dict( range = z_lims )
            # )
        ) 

        return fig
    
    def show( self ) -> None:
        
        fig = self.get_fig()
        fig.show()

class LineCanvas:
    def __init__( self, title: str = "", print_func: Callable[[str], None] = print ):

        self._title = title
        self._print = print_func

        self._lines = np.array( [] )
        self._colors = np.array( [] )

    def add( self, lines: np.ndarray, colors: np.ndarray ) -> None:

        if( len( lines.shape ) != 3 or len( colors.shape ) != 1 ):
            self._print( f"LineCanvas requires a 3D array for lines and 1D array for colors. Currently, lines has shape {lines.shape} and colors has shape {colors.shape}." )
            return
        
        if( lines.shape[0] != colors.shape[0] ):
            self._print( f"LineCanvas data requires that each line be provided a color. Currently, there are {lines.shape[1]} lines and {colors.shape[1]} colors." )
            return

        if( self._lines.shape[0] > 0 ):
            self._lines = np.concatenate( ( self._lines, lines ), axis = 0 )
            self._colors = np.concatenate( ( self._colors, colors ), axis = 0 )
            
        else:
            self._lines = lines
            self._colors = colors

    def clear( self ):

        self._lines = np.array( [] )
        self._colors = np.array( [] )

    def get_fig( self, x_lims: list[float], y_lims: list[float] ) -> go.Figure:
        
        fig = go.Figure()

        if( self._lines.shape[0] > 0 and self._lines.shape[0] == self._colors.shape[0] ):

            for i, line in enumerate( self._lines ):
                fig.add_shape( type = "line", x0 = line[0][0], y0 = line[0][1], x1 = line[1][0], y1 = line[1][1], line = dict( color = self._colors[i] ) )

            fig.update_layout(
                title = self._title,
                xaxis_range = x_lims,
                yaxis_range = y_lims
            ) 

        return fig