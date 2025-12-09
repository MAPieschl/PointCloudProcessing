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

        self.data_y1 = np.ndarray( [[]] )
        self.labels_y1 = np.ndarray( [[]] )
        self.data_y2 = np.ndarray( [[]] )
        self.labels_y2 = np.ndarray( [[]] )

    def update_y1( self, data: np.ndarray, labels: np.ndarray ) -> None:

        if( len( data.shape ) != 2 or len( data.shape ) != 2 ):
            self._print( "ERROR IN LINE_PLOT:  'data' must be input as a 2D array, where the first dimension" \
                         "contains indiviudal line plots and the second dimension contains the data." )
            return
        
        if( len( labels.shape ) != 1 or data.shape[0] != labels.shape[0] or len( labels.shape ) != 1 or data.shape[0] != labels.shape[0] ):
             self._print( "ERROR IN LINE_PLOT:  'labels' must be a one dimensional np.ndarray containing a" \
                          "label for each line in 'data'." )
             return  
        
        self._data_y1 = data
        self._labels_y1 = labels

    def update_y2( self, data: np.ndarray, labels: np.ndarray ) -> None:

        if( len( data.shape ) != 2 or len( data.shape ) != 2 ):
            self._print( "ERROR IN LINE_PLOT:  'data' must be input as a 2D array, where the first dimension" \
                         "contains indiviudal line plots and the second dimension contains the data." )
            return
        
        if( len( labels.shape ) != 1 or data.shape[0] != labels.shape[0] or len( labels.shape ) != 1 or data.shape[0] != labels.shape[0] ):
             self._print( "ERROR IN LINE_PLOT:  'labels' must be a one dimensional np.ndarray containing a" \
                          "label for each line in 'data'." )
             return  
        
        self._data_y2 = data
        self._labels_y2 = labels

    def get_fig( self ) -> go.Figure:
        
        fig = go.Figure()
        x = np.linspace(0, np.max( self._data_y1.shape[1], self._data_y2.shape[1] ), np.max( self._data_y1.shape[1], self._data_y2.shape[1] ) )

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

        if( len( self._data_y2 ) == 0 or len( self._labels_y2 ) == 0 ):
             
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