import pandas as pd
import numpy as np
import chainladder as cl
import sys
import matplotlib.pyplot as plt
from datetime import datetime


def analyze_triangle(claim_df, premium_df=None):
    """Apply multiple actuarial methodologies to estimate ultimate losses."""
    import chainladder as cl
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    results = {}

    # Test chain ladder assumptions
    print("Testing Chain Ladder assumptions...")
    results['val_corr'] = claim_df.valuation_correlation(p_critical=0.10)
    results['dev_corr'] = claim_df.development_correlation(p_critical=0.10)

    print(f"Valuation correlation test shows significant correlation: {results['val_corr'].z_critical.values[0][0]}")
    print(f"Development correlation test shows significant correlation: {results['dev_corr'].t_critical.values[0][0]}")

    # Method 1: Basic Chain Ladder with different averages
    print("\nApplying Chain Ladder with different averaging methods...")
    methods = {
        'volume_wtd': cl.Development(average='volume'),
        'simple_avg': cl.Development(average='simple'),
        'regression': cl.Development(average='regression'),
        'mixed': cl.Development(average=['volume', 'volume', 'simple', 'simple', 'simple',
                                         'regression', 'regression', 'regression', 'volume'])
    }

    for name, method in methods.items():
        results[name] = cl.Chainladder().fit(method.fit_transform(claim_df))

    # Method 2: Chain Ladder with different time periods
    print("Applying Chain Ladder with different time period selections...")
    periods = {
        'all_periods': cl.Development(n_periods=-1),
        'last_3': cl.Development(n_periods=3),
        'last_5': cl.Development(n_periods=5)
    }

    for name, method in periods.items():
        results[name] = cl.Chainladder().fit(method.fit_transform(claim_df))

    # Method 3: Chain Ladder with outlier treatment
    print("Applying Chain Ladder with outlier treatment...")
    outlier_methods = {
        'drop_high': cl.Development(drop_high=True),
        'olympic_avg': cl.Development(drop_high=True, drop_low=True),
        'drop_latest_diag': cl.Development(drop_valuation=claim_df.valuation.max().strftime('%Y'))
    }

    for name, method in outlier_methods.items():
        results[name] = cl.Chainladder().fit(method.fit_transform(claim_df))

    # Method 4: Bornhuetter-Ferguson Method
    print("Applying Bornhuetter-Ferguson Method...")
    if premium_df is not None:
        try:
            dev = cl.Development().fit(claim_df)

            # Calculate average loss ratio more safely
            latest_diag = claim_df.latest_diagonal.set_backend("numpy")
            premium_latest = premium_df.latest_diagonal.set_backend("numpy")

            # Make sure we have matching indices
            common_indices = list(set(latest_diag.index).intersection(set(premium_latest.index)))
            if common_indices:
                # Calculate using only common indices
                historical_lr = latest_diag.loc[common_indices] / premium_latest.loc[common_indices]

                # Get average LR more safely
                if len(historical_lr.values) > 0:
                    avg_lr = float(np.nanmean(historical_lr.values))

                    # BF with different expected loss ratios
                    apriori_els = {
                        'historical_avg': avg_lr,
                        'conservative': avg_lr * 1.10,
                        'optimistic': avg_lr * 0.90
                    }

                    for name, apriori in apriori_els.items():
                        bf = cl.BornhuetterFerguson(apriori=apriori)
                        results[f'bf_{name}'] = bf.fit(dev.transform(claim_df))

                    # Method 5: Cape Cod (Stanard-Bühlmann) Method
                    print("Applying Cape Cod Method...")
                    try:
                        cc = cl.CapeCod()
                        results['cape_cod'] = cc.fit(dev.transform(claim_df), sample_weight=premium_df)
                    except Exception as e:
                        print(f"Cape Cod method failed: {e}")
            else:
                print("No matching indices between claim and premium triangles for BF method")
        except Exception as e:
            print(f"Bornhuetter-Ferguson method failed: {e}")

    # Method 6: Mack Chainladder for stochastic analysis
    print("Applying Mack Chainladder Method...")
    try:
        mack = cl.MackChainladder()
        results['mack'] = mack.fit(cl.Development().fit_transform(claim_df))
    except Exception as e:
        print(f"Mack Chainladder method failed: {e}")

    # Summarize results
    print("\nSummarizing ultimate estimates...")
    ultimates = pd.DataFrame()

    for method, model in results.items():
        if hasattr(model, 'ultimate_'):
            try:
                ultimate = model.ultimate_.sum()
                if len(ultimate.index) > 0:  # Check if there's at least one row
                    ultimates[method] = ultimate.iloc[0]
            except Exception as e:
                print(f"Could not extract ultimates for {method}: {e}")

    if not ultimates.empty:
        results['summary'] = ultimates

        # Create comparison chart
        plt.figure(figsize=(12, 8))
        ultimates.T.plot(kind='bar')
        plt.title('Ultimate Estimates by Method')
        plt.ylabel('Ultimate Loss')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('ultimate_comparison.png')
        results['chart'] = 'ultimate_comparison.png'

        # Format as table
        table = ultimates.T.reset_index()
        table.columns = ['Method', 'Ultimate']
        table['Ultimate'] = table['Ultimate'].map('${:,.0f}'.format)
        print("\nUltimate estimates by method:")
        print(table)
    else:
        print("No ultimates could be calculated.")

    return results


# This function should be called from step_02.py
def import_triangle(claim_df, premium_df):
    """
    Import the claims triangle into chainladder and perform multiple methodologies

    Parameters:
    -----------
    claim_df : DataFrame
        DataFrame containing claims data
    premium_df : DataFrame
        DataFrame containing premium data

    Returns:
    --------
    dict
        Results from various reserving methods
    """

    # Print version information
    print("Package Versions:")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"chainladder: {cl.__version__}")
    print(f"Python {sys.version} on {sys.platform}")


    # Convert integer years to proper dates
    claim_df['origin'] = pd.to_datetime(claim_df['accident_year'], format='%Y')
    claim_df['valuation'] = pd.to_datetime(claim_df['calendar_year'], format='%Y') + pd.offsets.YearEnd(0)

    premium_df['origin'] = pd.to_datetime(premium_df['accident_year'], format='%Y')
    # Create the triangle
    triangle = cl.Triangle(
        claim_df,
        origin='origin',
        development='valuation',
        columns='paid_losses',
        cumulative=True  # Explicitly set this to avoid warnings
    )

    # And for premium triangle
    premium_triangle = None
    if premium_df is not None:
        premium_triangle = cl.Triangle(
            premium_df,
            origin='origin',
            columns='earned_premium',
            cumulative=False  # Premium is typically not cumulative
        )

    # Create the triangle
    triangle = cl.Triangle(
        claim_df,
        origin='origin',
        development='valuation',
        columns='paid_losses',
        cumulative=True
    )

    print("\nClaims Triangle:")
    print(triangle)

    print("\nPremium Triangle:")
    print(premium_triangle)

    # Apply various methodologies
    results = analyze_triangle(triangle, premium_triangle)

    return results, triangle, premium_triangle
