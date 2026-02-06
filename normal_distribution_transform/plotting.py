import matplotlib.pyplot as plt
import numpy as np

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

    ax.set_xlabel( x1_label )
    ax.set_ylabel( x2_label )
    ax.set_zlabel( y_label )
    ax.legend()

    return fig