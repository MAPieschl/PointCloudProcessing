import numpy as np
import pandas as pd

from typing import Callable
from scipy.stats import wilcoxon, friedmanchisquare
from itertools import combinations
from plotly.subplots import make_subplots
from statsmodels.stats.multitest import multipletests

from utils.plotting import *

def pairwise_wilcoxon_signed_rank_test(
        data            : list[np.ndarray],
        labels          : list[str],
        significance    : float                         = 0.05,
        name            : str                           = '',
        include_pairs   : list[tuple[int, int]] | None  = None
    ) -> tuple[pd.DataFrame, go.Figure]:

    df, pl, pl_nm = [], [], []
    combos = list( combinations( range( len( data ) ), 2 ) )

    for i, j in combos:
        if( include_pairs is None or ( i, j ) in include_pairs ):
            _df, _pl = paired_wilcoxon_signed_rank_test(
                sample_a        = data[i],
                sample_b        = data[j],
                name_a          = 'First',
                name_b          = 'Second',
                significance    = significance
            )

            _df = _df.rename( columns = { 'Value' : f'({i + 1}) vs. ({j + 1})' } )
            df.append( _df )
            pl.append( _pl )
            pl_nm.append( f'{labels[i]} v. {labels[j]}' )

    pvals = [ float( f.at['p-value', f.columns[0]] ) for f in df ]
    reject, corr_p, _, _ = multipletests( pvals, alpha = significance, method = 'holm' )

    for i, f in enumerate( df ):
        f.at['p-value', f.columns[0]] = f'{corr_p[i]:.3f}'
        f.at['Is Significant', f.columns[0]] = f'{reject[i]}'

    combined_df = df[0]
    for i in range( 1, len( df ) ):
        combined_df = pd.merge( combined_df, df[i], on = 'Metric', how = 'outer' )

    combined_df = combined_df.drop( 'Test Type' ).reindex( [
        f'First Median',
        f'Second Median',
        f'W-Statistic',
        f'p-value',
        f'Significance Level',
        f'Is Significant',
        f'Effect'
    ] )

    combined_fig = combine_plots(
        figs            = pl,
        subplot_titles  = pl_nm,
        shape           = ( int( np.ceil( len( combos ) / 2.0 ) ), 2 ),
        plot_title      = 'Pairwise Wilcoxon Symmetry Tests' if name == '' else f'Pairwise Wilcoxon Symmetry Tests: {name}'
    )

    return combined_df, combined_fig

def paired_translation_rotation_wilcoxon_signed_rank_test(
        sample_a_L2 : np.ndarray, 
        sample_b_L2 : np.ndarray,
        sample_a_rot: np.ndarray, 
        sample_b_rot: np.ndarray,
        name_a      : str,
        name_b      : str,
        significance: float         = 0.05
) -> tuple[pd.DataFrame, go.Figure]:
    
    df, pl = [], []
    cols = ['Translation', 'Rotation']

    for i, pair in enumerate( [ ( sample_a_L2, sample_b_L2 ), ( sample_a_rot, sample_b_rot ) ] ):
        _df, _pl = paired_wilcoxon_signed_rank_test(
            sample_a        = pair[0],
            sample_b        = pair[1],
            name_a          = name_a,
            name_b          = name_b,
            significance    = significance
        )

        _df = _df.rename( columns = { 'Value' : cols[i] } )
        df.append( _df )
        pl.append( _pl )

    combined_df = df[0]
    for i in range( 1, len( df ) ):
        combined_df = pd.merge( combined_df, df[i], on = 'Metric', how = 'outer' )

    combined_df = combined_df.drop( 'Test Type' ).reindex( [
        f'{name_a} Median',
        f'{name_b} Median',
        f'W-Statistic',
        f'p-value',
        f'Significance Level',
        f'Is Significant',
        f'Effect'
    ] )

    combined_fig = combine_plots(
        figs            = pl,
        subplot_titles  = cols,
        shape           = ( 1, 2 ),
        plot_title      = 'Pairwise Wilcoxon Symmetry Tests for Translation (left) and Rotation (right)'
    )

    return combined_df, combined_fig

def paired_wilcoxon_signed_rank_test( 
        sample_a    : np.ndarray, 
        sample_b    : np.ndarray,
        name_a      : str,
        name_b      : str,
        significance: float         = 0.05
    ) -> tuple[pd.DataFrame, go.Figure]:
    '''
    Assumptions:
    -  Symmetric distribution (does not rely on normality)
    -  Samples are paired
    '''
    
    if( len( sample_a ) != len( sample_b ) ):
        raise ValueError( "Arrays must be the same length for a paired test." )
        
    stat, p_value = wilcoxon( sample_a, sample_b, alternative = 'two-sided' )
    
    median_a = np.median( sample_a )
    median_b = np.median( sample_b )

    rc = 1.0 - ( 2.0 * float( stat ) ) / ( len( sample_a ) * ( len( sample_a ) + 1 ) / 2.0 )

    results = pd.DataFrame( [{
        'Test Type': 'Paired Wilcoxon Signed-Rank',
        f'{name_a} Median': f'{median_a:.3f}',
        f'{name_b} Median': f'{median_b:.3f}',
        f'W-Statistic': f'{stat}',
        f'p-value': f'{p_value:.3f}',
        f'Significance Level': f'{significance:.3f}',
        f'Is Significant': f'{float( p_value ) < significance}',
        f'Effect': f'{'large' if rc >= 0.43 else ( 'medium' if rc >= 0.28 else ( 'small' if rc >= 0.11 else 'insignificant' ) )} ({rc:.2f})'
    }] )

    # results = results.T.reset_index()
    results = results.T
    results.columns = ['Value']
    results.index.name = 'Metric'
        
    return ( results, test_wilcoxon_assumptions( sample_a, sample_b ) )

def test_wilcoxon_assumptions(
        sample_a    : np.ndarray, 
        sample_b    : np.ndarray,
    ) -> go.Figure:

    if( len(sample_a) != len(sample_b) ):
        raise ValueError("Arrays must be the same length for a paired test.")
    
    diffs = sample_a - sample_b
    diffs = diffs[diffs != 0]

    return plot_histogram( 
        data            = diffs, 
        num_bins        = 50, 
        title           = f'Wilcoxon Symmetry Test - Median at {np.median( diffs ):.3f}',
        x_label         = 'Paired Differences',
        add_annotations = False 
    )

def friedman_chi_square_test(
        data        : list[np.ndarray],
        labels      : list[str],
        significance: float             = 0.05
    ) -> pd.DataFrame:

    for i in range( 1, len( data ) ):  
        if( len( data[0] ) != len( data[i] ) ):  
            print( f'friedman_chi_square_test:  data has inhomogoneous data length -> {[ len(i) for i in data ]}' )
            return pd.DataFrame()
    
    if( len( data ) != len( labels ) ):
        print( f'friedman_chi_square_test:  There must be a label for each dataset, not {len( data )} datasets and {len( labels )} labels.' )
        return pd.DataFrame()
    
    stat, p = friedmanchisquare( *data )

    meds = np.median( np.array( data ), axis = 1 )

    df = pd.DataFrame( [{
        'Test Type':  'Friedman Chi-Square',
        'Paired samples':  len( data[0] ),
        'p-value': f'{p:.3f}',
        'Significance Level':  significance,
        'Is Significant':  f'{float( p ) < significance}',
        'Min Median Error':  labels[ int( np.argmin( meds ) ) ]
    }])

    df = df.T.reset_index()
    df.columns = ['Metric', 'Value']

    return df

def get_cdf_percentiles_with_CI(
        data                : list[np.ndarray],
        labels              : list[str],
        confidence_interval : float             = 0.95,
        num_bootstrap       : int               = 10000,
        percentiles         : list[float]       = [50, 75, 90, 95, 99],
        units               : str               = 'm',
        seed                : int | None        = None
    ) -> pd.DataFrame:

    if( len( data ) != len( labels ) ):
        print( 'There must be one label for each distribution provided to get_cdf_percentiles_with_CI().' )
        return pd.DataFrame()
    
    rng = np.random.default_rng( seed )
    
    X = [ np.sort( d ) for d in data ]

    results = {}

    ci_label = f'{int( confidence_interval * 100 )}\% CI'

    for p in percentiles:

        perc = f'{p:.1f}\%'

        val_key = ( perc, '$\epsilon_{max}$' )
        ci_key = ( perc, ci_label )

        results[ val_key ] = {}
        results[ ci_key ] = {}

        for x, l in zip( X, labels ):
            ci = get_bootstrap_CI( x, p, rng, confidence_interval, num_bootstrap )
            results[ val_key ][ l ] = f'{np.percentile( x, p ):.2f} {units}'
            results[ ci_key ][ l ] = f'({ci[0]:.2f}, {ci[1]:.2f}) {units}'

    results = pd.DataFrame.from_dict( results, orient = 'index' )
    results.index.names = [ 'Percentile', 'Metric' ]

    return results

def get_bootstrap_CI(
        data                : np.ndarray,
        percentile          : float,
        rng                 : np.random.Generator,
        confidence_interval : float                 = 0.95,
        num_bootstrap       : int                   = 10000
) -> tuple[float, float]:
    
    if( data.ndim != 1 ):
        raise ValueError( f'get_bootstrap_CI() only accepts data of size (N,)' )

    lb = ( 1.0 - confidence_interval ) / 2.0
    ub = 1.0 - lb

    i_sets = rng.integers( 0, data.shape[0], size = ( num_bootstrap, data.shape[0] ) )
    boot_data = [ np.percentile( data[i], percentile ) for i in i_sets ]

    return ( float( np.quantile( boot_data, lb ) ), float( np.quantile( boot_data, ub ) ) )