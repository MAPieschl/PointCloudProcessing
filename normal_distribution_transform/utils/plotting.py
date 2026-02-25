import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go

from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, cast
from datetime import datetime

def plot_univariate_functions( funcs: list[Callable[[float], float]],
                              labels: list[str], x_range: tuple[float, float], 
                              title: str = '', xlabel: str = 'x', ylabel: str = 'y', 
                              print_func: Callable[[str], None] = print ) -> plt.Figure:
    
    fig, ax = plt.subplots()
    if( len( funcs ) != len( labels ) ):
        print_func( "Number of funcs must equal number of labels" )
        return fig

    x = np.linspace( x_range[0], x_range[1], 100 )
    y = np.array( [ [ f(x_) for x_ in x ] for f in funcs ] )

    for i, y_ in enumerate( y ):
        ax.plot( x, y_, label = labels[i] )

    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )
    ax.set_title( title )

    ax.legend()

    return fig


def plot_multivariate_functions( funcs: list[Callable[[float, float], float]],
                                labels: list[str], x_range: tuple[float, float],
                                title: str = '', x1_label: str = 'x_1', x2_label: str = 'x_2',
                                y_label: str = 'y', print_func: Callable[[str], None] = print ) -> plt.Figure:
    
    fig, ax = plt.subplots( subplot_kw = {"projection": "3d"} )
    ax = cast( Axes3D, ax ) # inconsequential in execution - for PyLance only

    if( len( funcs ) != len( labels ) ):
        print_func( "Number of funcs must equal number of labels" )
        return fig
    
    x1 = np.linspace( x_range[0], x_range[1], 100 )
    x2 = np.linspace( x_range[0], x_range[1], 100 )
    x1, x2 = np.meshgrid( x1, x2 )

    vec_f = [ np.vectorize( f ) for f in funcs ]

    y = np.array( [ f( x1, x2 ) for f in vec_f ] )

    for i, y_ in enumerate( y ):
        ax.plot_surface( x1, x2, y_, label = labels[i] )

    ax.set_title( title )
    ax.set_xlabel( x1_label )
    ax.set_ylabel( x2_label )
    ax.set_zlabel( y_label )
    ax.legend()

    return fig

def plot_sampled_surface( samples: np.ndarray, x_r: tuple[float, float],
                         y_r: tuple[float, float],
                         title: str = '', x_label: str = 'x', 
                         y_label: str = 'y', z_label: str = 'z',
                         print_func: Callable[[str], None] = print ) -> go.Figure:

    fig = go.Figure()

    x = np.linspace( x_r[0], x_r[1], samples.shape[1] )
    y = np.linspace( y_r[0], y_r[1], samples.shape[0] )

    fig.add_trace(
        go.Surface(
            x = x,
            y = y,
            z = samples
        )
    )

    fig.update_layout(
        title = title,
        scene = dict(
            xaxis_title = x_label,
            yaxis_title = y_label,
            zaxis_title = z_label
        )
    )

    return fig

def plot_2D_scatter_with_mean_and_std( x: np.ndarray,
                                      y: np.ndarray, 
                                      title: str, 
                                      x_label: str = 'x', 
                                      y_label: str = 'y', 
                                      print_func: Callable[[str], None] = print ) -> go.Figure:
    
    fig = go.Figure()

    ## Bin the values to create mean and std deviation data
    num_bins = int( x.shape[0] / 50 )
    bins = np.linspace( np.min( x ), np.max( x ), num_bins )

    x = x.squeeze()
    y = y.squeeze()

    means = []
    std = []
    x_val = []

    for i in range( bins.shape[0] ):
        if( i < num_bins - 1 ):
            bin_idx = np.where( ( x >= bins[i] ) & ( x < bins[i + 1] ) )

            if( len( bin_idx ) > 0 ):
                means.append( np.mean( y[bin_idx] ) )
                std.append( np.std( y[bin_idx] ) )
                x_val.append( ( bins[i + 1] - bins[i] ) / 2 + bins[i] )

    means = np.array( means )
    std = np.array( std )
    x_val = np.array( x_val )

    if( not ( np.all( np.isfinite( means ) ) and np.all( np.isfinite( std ) ) and np.all( np.isfinite( x_val ) ) ) ):
        print( 'Infinite values detected in mean and/or stdev calculations.' )
        return fig

    fig.add_trace( go.Scatter(
        x = x,
        y = y,
        mode = 'markers',
        marker = dict( color = 'rgba(0, 0, 191, 0.5)' ),
        name = 'residuals'
    ) )

    fig.add_trace( go.Scatter(
        x = x_val,
        y = means,
        mode = 'lines',
        line = dict( color = 'red' ),
        name = 'mean',
        connectgaps = True
    ) )

    fig.add_trace( go.Scatter(
        x = x_val,
        y = means + std,
        mode = 'lines',
        line = dict( color = 'blue' ),
        name = 'stdev - lower',
        connectgaps = True
    ) )

    fig.add_trace( go.Scatter(
        x = x_val,
        y = means - std,
        mode = 'lines',
        line = dict( color = 'blue' ),
        name = 'stdev - upper',
        connectgaps = True,
        fill = 'tonexty',
        fillcolor = 'rgba(0, 0, 255, 0.4)'
    ) )

    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label
    )

    return fig

def plot_class_precision_recall_hist( precision_data: dict[datetime, dict[str, float]],
                                     recall_data: dict[datetime, dict[str, float]],
                                     title:  str, print_func: Callable[[str], None] = print ) -> go.Figure:
    
    fig = go.Figure()

    classes = []
    precision_tuple = []
    recall_tuple = []

    for ts in list( precision_data.keys() ):
        for cl in list( precision_data[ts].keys() ):
            if( cl not in classes ):
                classes.append( cl )
                precision_tuple.append( [ 1, precision_data[ts][cl] ] )
                recall_tuple.append( [ 1, recall_data[ts][cl] ] )
            
            idx = classes.index( cl )

            precision_tuple[idx][0] += 1
            precision_tuple[idx][1] += precision_data[ts][cl]
            
            recall_tuple[idx][0] += 1
            recall_tuple[idx][1] += recall_data[ts][cl]

    precision = [ i[1] / i[0] for i in precision_tuple ]
    recall = [ i[1] / i[0] for i in recall_tuple ]

    fig.add_trace( go.Bar(
        x = classes,
        y = precision,
        name = 'precision',
        marker_color = 'blue',
        texttemplate = "%{y:.2f}",
        # textposition = 'outside',
        textfont_size = 16
    ) )

    fig.add_trace( go.Bar(
        x = classes,
        y = recall,
        name = 'recall',
        marker_color = 'red',
        texttemplate = "%{y:.2f}",
        # textposition = 'outside',
        textfont_size = 16
    ) )

    fig.update_layout(
        barmode = 'group',
        title = title,
        xaxis_title = 'part classes',
        yaxis_title = ''
    )

    return fig

def plot_class_precision_recall_scatter( precision_data: dict[str, np.ndarray],
                                         recall_data: dict[str, np.ndarray],
                                         title: str,
                                         x_label: str,
                                         print_func: Callable[[str], None] = print ) -> go.Figure:
    
    fig = go.Figure()

    for cl in list( precision_data.keys() ):

        fig.add_trace( go.Scatter(
            x = precision_data[cl][0, :],
            y = precision_data[cl][1, :],
            mode = 'markers',
            marker = dict( color = 'blue' ),
            name = f'{cl} precision'
        ))

        fig.add_trace( go.Scatter(
            x = recall_data[cl][0, :],
            y = recall_data[cl][1, :],
            mode = 'markers',
            marker = dict( color = 'red' ),
            name = f'{cl} recall'
        ))

    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = ''
    )

    return fig

def plot_histogram( data: np.ndarray,
                   num_bins: int,
                   title: str = '',
                   x_label: str = '',
                   y_label: str = '',
                   print_func: Callable[[str], None] = print ) -> go.Figure:
    
    fig = go.Figure()

    bins = np.linspace( np.min( data ), np.max( data ), num_bins )
    bin_count = np.zeros( bins.shape )
    for bin in range( len( bins ) - 1 ):
        for d in data:
            if( d >= bins[bin] and d < bins[bin + 1] ):
                bin_count[bin] += 1

    fig.add_trace( go.Bar(
        x = bins,
        y = bin_count,
        marker_color = 'blue'
     ) )
    
    fig.add_annotation(
        x = np.mean( data ),
        y = np.max( bin_count ),
        text = f"Mean: {np.mean( data )}\nStDev: {np.std( data )}"
    )
    
    fig.update_layout(
        barmode = 'group',
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label
    )

    return fig

def display_point_clouds( clouds: list, labels: list, title: str = 'Point Cloud' ) -> go.Figure:
        '''
        Displays a point cloud as a 3D scatter plot

        @param clouds    (list) of ndarrays     -> [cloud][points]
        @param labels    (list) parallel array  -> [cloud]
        @param title     (str)  plot title   
        '''

        assert len( clouds ) == len( labels ), "display_point_clouds:  ensure there is a label for each cloud"

        plots = []
        for i, cloud in enumerate( clouds ):

            if( cloud.shape[0] < 1 ): continue
            
            plots.append( go.Scatter3d(
                x = cloud[:, 0],
                y = cloud[:, 1],
                z = cloud[:, 2],
                mode = 'markers',
                marker = dict(
                    size = 2,
                    opacity = 1.0
                ),
                name = labels[i]
            ))

        fig = go.Figure( data = plots )

        fig.update_layout( scene = dict(
            xaxis_title = 'X',
            yaxis_title = 'Y',
            zaxis_title = 'Z',
            aspectmode = 'data'
        ),
        title = title,
        margin = dict( l = 0, r = 0, b = 0, t = 40 )
        )

        return fig