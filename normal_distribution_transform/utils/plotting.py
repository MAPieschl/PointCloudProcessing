import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go

from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, cast

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
                                      print_func: Callable[[str], None] = print ):
    
    fig = go.Figure()

    ## Bin the values to create mean and std deviation data
    num_bins = x.shape[0] / 10
    bins = np.linspace( np.min( x ), np.max( x ), num_bins )

    means = []
    std = []
    x_val = []

    for i in range( bins.shape[0] ):
        if( i < num_bins - 1 ):
            bin_idx = np.where( x >= bins[i] and x < bins[i + 1] )
            means.append( np.mean( y[bin_idx] ) )
            std.append( np.std( y[bin_idx] ) )
            x_val = ( bins[i + 1] - bins[i] ) / 2 + bins[i]

    means = np.array( means )
    std = np.array( std )
    x_val = np.array( x_val )

    fig.add_trace( go.Scatter(
        x = x,
        y = y,
        mode = 'markers'
    ) )

    fig.add_trace( go.Scatter(
        x = x_val,
        y = means,
        mode = 'lines',
        line = dict( color = 'red' )
    ) )

    fig.add_trace( go.Scatter(
        x = x_val,
        y = means + std,
        mode = 'lines',
        line = dict( color = 'blue' )
    ) )

    fig.add_trace( go.Scatter(
        x = x_val,
        y = means - std,
        mode = 'lines',
        line = dict( color = 'blue' )
    ) )

    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label
    )