import numpy as np
import pandas as pd

from typing import Callable
from scipy.stats import wilcoxon

def wilcoxon_signed_rank_test( 
        sample_a    : np.ndarray, 
        sample_b    : np.ndarray,
        name_a      : str,
        name_b      : str,
        significance: float         = 0.05
    ) -> pd.DataFrame:
    '''
    Assumptions:
    -  Symmetric distribution (does not rely on normality)
    -  Samples are paired
    '''
    
    # Check that the arrays are perfectly paired (same length)
    if( len(sample_a) != len(sample_b) ):
        raise ValueError("Arrays must be the same length for a paired test.")
        
    # Run the Wilcoxon Signed-Rank Test
    # alternative='two-sided' checks for any difference (better or worse)
    stat, p_value = wilcoxon(sample_a, sample_b, alternative='two-sided')
    
    # Calculate medians to determine which model is generally performing better
    median_a = np.median(sample_a)
    median_b = np.median(sample_b)

    results = pd.DataFrame( [{
        'Test Type': 'Paired Wilcoxon Signed-Rank',
        f'{name_a} Median': f'{median_a:.3f}',
        f'{name_b} Median': f'{median_b:.3f}',
        f'W-Statistic': f'{stat}',
        f'p-value': f'{p_value:.3f}',
        f'Significance Level': f'{significance:.3f}',
        f'Is Significant': f'{float( p_value ) < significance}'
    }] )

    results = results.T.reset_index()
    results.columns = ['Metric', 'Value']
        
    return results