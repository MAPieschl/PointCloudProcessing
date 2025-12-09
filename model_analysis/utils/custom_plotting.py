from dependencies import *

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