import numpy as np

from typing import Callable

def finite_difference_approximation( f: Callable[[float], float], at: tuple[float, float], h: float  = 0.001, print_func: Callable[[str], None] = print ) -> Callable[[float], float]:
    '''
    Solves the derivative of f(x) at x_0 using a finite difference approximation.

    @param  f   (Callable[[float], float]) takes the derivative of this function
    @param  at  (tuple[float, float]) the (x, y) coordinates of the point of differentiation
    @param  h   (float default = 0.001) small change in x used to generate the secant line
    @param  print_func  (Callable[[str], None], default = print) available for custom print/log functionality

    @return Callable[[float], float], the derivative of f
    '''

    x, y = at

    m = ( f( x + h ) - f( x - h ) ) / ( 2 * h )