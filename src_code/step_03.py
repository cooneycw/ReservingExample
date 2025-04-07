import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chainladder as cl
from datetime import datetime
from src_code.step_01 import generate_claim_data
from src_code.step_02 import percentage_supplemental_required_method, plot_supplemental_factors


def prepare_triangle_data(claim_df, value_col='paid_losses'):
    """
    Convert the claim data to a format suitable for chainladder triangle.
    This approach formats development periods correctly for chainladder.

    Parameters:
    - claim_df: DataFrame with claim data
    - value_col: Column name to use for values in the triangle

    Returns:
    - Triangle data suitable for chainladder
    """
    import pandas as pd
    import numpy as np
    import chainladder as cl

    # Clean input
    clean_df = claim_df.dropna(subset=['accident_year', 'dev_year', value_col]).copy()
    clean_df['accident_year'] = clean_df['accident_year'].astype(int)
    clean_df['dev_year'] = clean_df['dev_year'].astype(int)

    # Pivot into triangle format (for display only)
    triangle_df = clean_df.pivot_table(
        index='accident_year',
        columns='dev_year',
        values=value_col,
        aggfunc='sum'
    )

    print(f"Pivoted triangle for {value_col}:")
    print(triangle_df)

    try:
        # Convert development periods to proper development lags
        # Format the development periods as '{development period}M' to indicate months
        # This helps chainladder correctly interpret the development lags
        dev_periods = {i: f"{i * 12}M" for i in triangle_df.columns}
        formatted_df = triangle_df.rename(columns=dev_periods)

        # Create triangle using chainladder's DataFrame constructor
        triangle = cl.Triangle(
            data=formatted_df,
            origin='index',
            development='columns',
            format='%Y'  # Format for origin period (year)
        )
        return triangle

    except Exception as e:
        print(f"Error creating triangle for {value_col}: {e}")

        # Try alternative approach using from_pandas
        try:
            # Try the from_pandas approach
            triangle = cl.Triangle.from_pandas(
                triangle_df,
                origin_format="%Y",  # Format for accident years
                development_format="%d"  # Format for development periods as integers
            )
            return triangle

        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")

            # Try one more approach using the long format
            try:
                # Prepare data in long format with properly formatted development periods
                cl_input_df = clean_df[['accident_year', 'dev_year', value_col]].copy()

                # Format development periods as developmentlag in months
                cl_input_df['development'] = cl_input_df['dev_year'].apply(lambda x: f"{x * 12}M")

                # Create the final input DataFrame
                final_df = pd.DataFrame({
                    'origin': cl_input_df['accident_year'],
                    'development': cl_input_df['development'],
                    'values': cl_input_df[value_col]
                })

                # Create triangle from the long format
                triangle = cl.Triangle(
                    data=final_df,
                    origin='origin',
                    development='development',
                    columns='values',
                    format='%Y'  # Format for origin period
                )
                return triangle

            except Exception as e3:
                print(f"Third approach also failed: {e3}")

                print("Triangle DataFrame info:")
                print(triangle_df.info())
                print("Triangle DataFrame dtypes:")
                print(triangle_df.dtypes)
                print("DataFrame column values:")
                print(triangle_df.columns.tolist())
                return None


def run_chainladder_method(triangle, method_type='paid'):
    """
    Run the Chain Ladder method on a triangle of data.

    Parameters:
    - triangle: chainladder Triangle object
    - method_type: 'paid' or 'incurred' (for reporting purposes)

    Returns:
    - Dictionary with chainladder result objects
    """
    # Calculate development factors
    cl_dev = cl.Development().fit(triangle)
    ldfs = cl_dev.ldf_

    # Apply Chain Ladder method
    ultimate = cl.Chainladder().fit(cl_dev.transform(triangle))

    # Calculate IBNR
    ibnr = ultimate.ibnr_

    return {
        'development': cl_dev,
        'ldfs': ldfs,
        'ultimate': ultimate.ultimate_,
        'ibnr': ibnr,
        'method_type': method_type
    }


def run_bf_method(triangle, premium_array, loss_ratio=0.65):
    """
    Run the Bornhuetter-Ferguson method on a triangle of data.

    Parameters:
    - triangle: chainladder Triangle object (post development fit)
    - premium_array: Series with earned premiums
    - loss_ratio: Expected loss ratio

    Returns:
    - Dictionary with BF result objects
    """
    # Calculate development factors first
    cl_dev = cl.Development().fit(triangle)

    # Ensure premium array index is aligned with triangle origins
    triangle_origins = triangle.origin.get_array()[0]
    aligned_premiums = pd.Series(
        index=triangle_origins,
        data=[premium_array.get(origin, 0) for origin in triangle_origins]
    )

    # Check for any missing premiums and fill with reasonable values if needed
    if aligned_premiums.isna().any() or (aligned_premiums == 0).any():
        # Use the average premium for any missing values
        avg_premium = premium_array.mean()
        aligned_premiums = aligned_premiums.replace(0, avg_premium)
        aligned_premiums = aligned_premiums.fillna(avg_premium)
        print(f"Warning: Some accident years were missing premiums. Using average: {avg_premium}")

    # Set up the Bornhuetter-Ferguson method
    bf = cl.BornhuetterFerguson(apriori=aligned_premiums * loss_ratio)

    # Apply BF method
    bf_result = bf.fit(cl_dev.transform(triangle))

    return {
        'development': cl_dev,
        'ultimate': bf_result.ultimate_,
        'ibnr': bf_result.ibnr_
    }


def run_benktander_method(triangle, premium_array, loss_ratio=0.65, n_iters=3):
    """
    Run the Benktander method on a triangle of data.

    Parameters:
    - triangle: chainladder Triangle object
    - premium_array: Series with earned premiums
    - loss_ratio: Expected loss ratio
    - n_iters: Number of iterations for the Benktander method

    Returns:
    - Dictionary with Benktander result objects
    """
    # Calculate development factors first
    cl_dev = cl.Development().fit(triangle)

    # Ensure premium array index is aligned with triangle origins
    triangle_origins = triangle.origin.get_array()[0]
    aligned_premiums = pd.Series(
        index=triangle_origins,
        data=[premium_array.get(origin, 0) for origin in triangle_origins]
    )

    # Check for any missing premiums and fill with reasonable values if needed
    if aligned_premiums.isna().any() or (aligned_premiums == 0).any():
        # Use the average premium for any missing values
        avg_premium = premium_array.mean()
        aligned_premiums = aligned_premiums.replace(0, avg_premium)
        aligned_premiums = aligned_premiums.fillna(avg_premium)
        print(f"Warning: Some accident years were missing premiums. Using average: {avg_premium}")

    # Apply Benktander method
    benktander = cl.Benktander(apriori=aligned_premiums * loss_ratio, n_iters=n_iters)
    benk_result = benktander.fit(cl_dev.transform(triangle))

    return {
        'development': cl_dev,
        'ultimate': benk_result.ultimate_,
        'ibnr': benk_result.ibnr_
    }


def run_cape_cod_method(triangle, premium_array):
    """
    Run the Cape Cod method on a triangle of data.

    Parameters:
    - triangle: chainladder Triangle object
    - premium_array: Series with earned premiums

    Returns:
    - Dictionary with Cape Cod result objects
    """
    # Calculate development factors first
    cl_dev = cl.Development().fit(triangle)

    # Ensure premium array index is aligned with triangle origins
    triangle_origins = triangle.origin.get_array()[0]
    aligned_premiums = pd.Series(
        index=triangle_origins,
        data=[premium_array.get(origin, 0) for origin in triangle_origins]
    )

    # Check for any missing premiums and fill with reasonable values if needed
    if aligned_premiums.isna().any() or (aligned_premiums == 0).any():
        # Use the average premium for any missing values
        avg_premium = premium_array.mean()
        aligned_premiums = aligned_premiums.replace(0, avg_premium)
        aligned_premiums = aligned_premiums.fillna(avg_premium)
        print(f"Warning: Some accident years were missing premiums. Using average: {avg_premium}")

    # Apply Cape Cod method
    cape_cod = cl.CapeCod(apriori=aligned_premiums)
    cc_result = cape_cod.fit(cl_dev.transform(triangle))

    return {
        'development': cl_dev,
        'ultimate': cc_result.ultimate_,
        'ibnr': cc_result.ibnr_
    }


def run_reserving_analysis(claim_df, premium_df, avg_loss_ratio=0.65):
    """
    Run a complete reserving analysis using multiple methods.

    Parameters:
    - claim_df: DataFrame with claim data
    - premium_df: DataFrame with premium data
    - avg_loss_ratio: Average loss ratio for BF and Benktander methods

    Returns:
    - Dictionary with results from different methods
    """
    import pandas as pd
    import numpy as np
    import traceback
    import sys

    results = {}

    # Make sure accident_year and dev_year are clean
    claim_df = claim_df.copy()
    premium_df = premium_df.copy()

    # Ensure that key columns are not None and are properly typed
    claim_df = claim_df.dropna(subset=['accident_year', 'dev_year', 'paid_losses', 'incd_losses'])
    claim_df['accident_year'] = claim_df['accident_year'].astype(int)
    claim_df['dev_year'] = claim_df['dev_year'].astype(int)

    premium_df = premium_df.dropna(subset=['accident_year', 'earned_premium'])
    premium_df['accident_year'] = premium_df['accident_year'].astype(int)

    # ----- Prepare triangle data for different methods -----

    try:
        # Paid losses triangle
        paid_triangle = prepare_triangle_data(claim_df, 'paid_losses')
        if paid_triangle is None:
            print("Could not create paid losses triangle, aborting analysis")
            return results

        print("Paid Losses Triangle:")
        print(paid_triangle)

        # Incurred losses triangle
        incurred_triangle = prepare_triangle_data(claim_df, 'incd_losses')
        if incurred_triangle is None:
            print("Could not create incurred losses triangle, aborting analysis")
            return results

        print("\nIncurred Losses Triangle:")
        print(incurred_triangle)
    except Exception as e:
        print(f"Error preparing triangles: {e}")
        traceback.print_exc(file=sys.stdout)
        return results

    # Prepare premium data for methods that require it
    try:
        premium_array = premium_df.set_index('accident_year')['earned_premium']

        # Get the accident years from the triangles for alignment
        accident_years = claim_df['accident_year'].unique()

        # Check if premium data covers all accident years
        missing_years = [ay for ay in accident_years if ay not in premium_array.index]
        if missing_years:
            print(f"Warning: Premium data missing for accident years: {missing_years}")
            # For missing years, use the average premium
            avg_premium = premium_array.mean()
            for year in missing_years:
                premium_array[year] = avg_premium
    except Exception as e:
        print(f"Error preparing premium data: {e}")
        traceback.print_exc(file=sys.stdout)
        return results

    # ----- Method 1: Chain Ladder Method on Paid Losses -----
    print("\n===== Chain Ladder Method (Paid) =====")

    try:
        # Run Chain Ladder on paid losses
        cl_paid_results = run_chainladder_method(paid_triangle, 'paid')

        print("Link Ratios (Development Factors):")
        print(cl_paid_results['ldfs'])

        print("\nPaid Ultimate Estimates:")
        print(cl_paid_results['ultimate'])

        print("\nPaid IBNR Estimates:")
        print(cl_paid_results['ibnr'])

        results['chain_ladder_paid'] = cl_paid_results
    except Exception as e:
        print(f"Error in Chain Ladder (Paid) method: {e}")
        traceback.print_exc(file=sys.stdout)

    # ----- Method 2: Chain Ladder Method on Incurred Losses -----
    print("\n===== Chain Ladder Method (Incurred) =====")

    try:
        # Run Chain Ladder on incurred losses
        cl_incurred_results = run_chainladder_method(incurred_triangle, 'incurred')

        print("Link Ratios (Development Factors):")
        print(cl_incurred_results['ldfs'])

        print("\nIncurred Ultimate Estimates:")
        print(cl_incurred_results['ultimate'])

        print("\nIncurred IBNR Estimates:")
        print(cl_incurred_results['ibnr'])

        results['chain_ladder_incurred'] = cl_incurred_results
    except Exception as e:
        print(f"Error in Chain Ladder (Incurred) method: {e}")
        traceback.print_exc(file=sys.stdout)

    # ----- Method 3: Bornhuetter-Ferguson Method on Paid Losses -----
    print("\n===== Bornhuetter-Ferguson Method (Paid) =====")

    try:
        # Run Bornhuetter-Ferguson on paid losses
        bf_paid_results = run_bf_method(paid_triangle, premium_array, avg_loss_ratio)

        print("BF Paid Ultimate Estimates:")
        print(bf_paid_results['ultimate'])

        print("\nBF Paid IBNR Estimates:")
        print(bf_paid_results['ibnr'])

        results['bf_paid'] = bf_paid_results
    except Exception as e:
        print(f"Error in Bornhuetter-Ferguson method: {e}")
        traceback.print_exc(file=sys.stdout)

    # ----- Method 4: Benktander Method -----
    print("\n===== Benktander Method =====")

    try:
        # Run Benktander on paid losses
        benk_results = run_benktander_method(paid_triangle, premium_array, avg_loss_ratio)

        print("Benktander Ultimate Estimates:")
        print(benk_results['ultimate'])

        print("\nBenktander IBNR Estimates:")
        print(benk_results['ibnr'])

        results['benktander'] = benk_results
    except Exception as e:
        print(f"Error in Benktander method: {e}")
        traceback.print_exc(file=sys.stdout)

    # ----- Method 5: Cape Cod Method -----
    print("\n===== Cape Cod Method =====")

    try:
        # Run Cape Cod on paid losses
        cc_results = run_cape_cod_method(paid_triangle, premium_array)

        print("Cape Cod Ultimate Estimates:")
        print(cc_results['ultimate'])

        print("\nCape Cod IBNR Estimates:")
        print(cc_results['ibnr'])

        results['cape_cod'] = cc_results
    except Exception as e:
        print(f"Error in Cape Cod method: {e}")
        traceback.print_exc(file=sys.stdout)

    # ----- Method 6: Percentage Supplemental Required Method -----
    print("\n===== Percentage Supplemental Required Method =====")

    try:
        # Determine mature years (those with relatively high development)
        dev_by_ay = claim_df.groupby('accident_year')['dev_year'].max()

        # Sort development years to find good candidates for mature years
        sorted_devs = dev_by_ay.sort_values(ascending=False)

        # Use the top 2-3 most developed years if they have at least 5 dev periods
        mature_candidates = sorted_devs[sorted_devs >= 5].index.tolist()

        # If we don't have enough years with 5+ dev periods, use the oldest 2-3 years
        if len(mature_candidates) < 2:
            oldest_years = sorted(claim_df['accident_year'].unique())[:min(3, len(claim_df['accident_year'].unique()))]
            mature_candidates = oldest_years

        # Take at most 3 mature years to avoid overweighting old data
        mature_years = mature_candidates[:min(3, len(mature_candidates))]

        print(f"Using mature years for PSR method: {mature_years}")

        # Apply Percentage Supplemental Required method
        psr_results = percentage_supplemental_required_method(
            claim_df,
            method_params={
                'mature_years': mature_years,
                'averaging_method': 'medial',
                'exclude_outliers': True
            }
        )

        print("Supplemental Factors by Development Age:")
        print(psr_results['supplemental_factors'])

        print("\nPercentage Supplemental Required Ultimate Estimates:")
        print(psr_results['ultimate'])

        print("\nPercentage Supplemental Required IBNR Estimates:")
        print(psr_results['ibnr'])

        # Plot the supplemental factors
        try:
            plot_supplemental_factors(psr_results['supplemental_factors'])
        except Exception as plot_e:
            print(f"Error plotting supplemental factors: {plot_e}")
            traceback.print_exc(file=sys.stdout)

        results['pct_supplemental'] = psr_results
    except Exception as e:
        print(f"Error in Percentage Supplemental Required method: {e}")
        traceback.print_exc(file=sys.stdout)

    return results


def compare_results(results, claim_df):
    """
    Compare results from different methods to the 'actual' ultimates in our simulated data.

    Parameters:
    - results: Dictionary with results from different methods
    - claim_df: DataFrame with the generated claim data, including actual ultimate losses

    Returns:
    - DataFrame with comparison of methods
    """
    # Get the actual ultimates from our generated data
    if 'ultimate_loss' in claim_df.columns:
        actual_ultimates = claim_df.groupby('accident_year')['ultimate_loss'].first()

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Actual Ultimate': actual_ultimates
        })
    else:
        # If no actual ultimates in the data, create an empty comparison
        comparison = pd.DataFrame(index=sorted(claim_df['accident_year'].unique()))
        print("Warning: No 'ultimate_loss' in claims data - comparison will not include actual ultimates")

    # Add results from each method
    methods = {
        'chain_ladder_paid': 'Chain Ladder (Paid)',
        'chain_ladder_incurred': 'Chain Ladder (Incurred)',
        'bf_paid': 'Bornhuetter-Ferguson',
        'benktander': 'Benktander',
        'cape_cod': 'Cape Cod',
        'pct_supplemental': 'Percentage Supplemental'
    }

    for method_key, method_name in methods.items():
        if method_key in results:
            try:
                if method_key == 'pct_supplemental':
                    # Handle pct_supplemental differently since it's not a chainladder object
                    for idx, value in results[method_key]['ultimate'].items():
                        comparison.loc[idx, method_name] = value
                else:
                    # Handle chainladder Triangle objects
                    ultimates = results[method_key]['ultimate'].to_frame()
                    for origin, values in ultimates.groupby('origin'):
                        comparison.loc[origin, method_name] = values.iloc[0, 0]
            except Exception as e:
                print(f"Error adding {method_name} to comparison: {e}")

    # Calculate percentage differences from actual if we have them
    if 'Actual Ultimate' in comparison.columns:
        for col in comparison.columns:
            if col != 'Actual Ultimate':
                comparison[f'{col} % Diff'] = (comparison[col] - comparison['Actual Ultimate']) / comparison[
                    'Actual Ultimate'] * 100

    return comparison


def create_reserve_summary(results, claim_df):
    """
    Create a summary of IBNR estimates from all methods.

    Parameters:
    - results: Dictionary with results from different methods
    - claim_df: DataFrame with the generated claim data

    Returns:
    - DataFrame with IBNR summary
    """
    # Create base summary DataFrame with latest incurred
    latest_data = claim_df.sort_values(['accident_year', 'dev_year'])
    latest_data = latest_data.groupby('accident_year').last()

    # Start with reported incurred from the latest diagonal
    summary = pd.DataFrame({
        'Reported Incurred': latest_data['incd_losses']
    })

    # Add actual ultimate and IBNR if available
    if 'ultimate_loss' in claim_df.columns:
        # Get the actual ultimates
        ultimates = claim_df.groupby('accident_year')['ultimate_loss'].first()

        # Add to summary
        summary['Actual Ultimate'] = ultimates
        summary['Actual IBNR'] = ultimates - latest_data['incd_losses']

    # Define method names for the summary
    methods = {
        'chain_ladder_paid': 'Chain Ladder (Paid)',
        'chain_ladder_incurred': 'Chain Ladder (Incurred)',
        'bf_paid': 'Bornhuetter-Ferguson',
        'benktander': 'Benktander',
        'cape_cod': 'Cape Cod',
        'pct_supplemental': 'Percentage Supplemental'
    }

    # Add IBNR estimates from each method
    for method_key, method_name in methods.items():
        if method_key in results:
            try:
                if method_key == 'pct_supplemental':
                    # Handle pct_supplemental differently
                    for idx, value in results[method_key]['ibnr'].items():
                        summary.loc[idx, f'{method_name} IBNR'] = value
                else:
                    # Handle chainladder Triangle objects
                    ibnr = results[method_key]['ibnr'].to_frame()
                    for origin, values in ibnr.groupby('origin'):
                        summary.loc[origin, f'{method_name} IBNR'] = values.iloc[0, 0]
            except Exception as e:
                print(f"Error adding {method_name} IBNR to summary: {e}")

    # Calculate percentage differences from actual IBNR if we have them
    if 'Actual IBNR' in summary.columns:
        for col in summary.columns:
            if 'IBNR' in col and col != 'Actual IBNR':
                summary[f'{col} % Diff'] = (summary[col] - summary['Actual IBNR']) / summary['Actual IBNR'] * 100

    return summary


def plot_results(comparison):
    """
    Plot the comparison of ultimate loss estimates from different methods.

    Parameters:
    - comparison: DataFrame with comparison of methods
    """
    # Extract methods and accident years
    methods = [col for col in comparison.columns if '% Diff' not in col and col != 'Actual Ultimate']

    if not methods:
        print("Warning: No methods available for plotting")
        return

    accident_years = comparison.index

    try:
        # Create a plot for ultimates
        plt.figure(figsize=(12, 8))

        # Plot actual ultimates if available
        if 'Actual Ultimate' in comparison.columns:
            plt.plot(accident_years, comparison['Actual Ultimate'], 'ko-', linewidth=2, label='Actual Ultimate')

        # Plot method estimates
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for i, method in enumerate(methods):
            color = colors[i % len(colors)]
            plt.plot(accident_years, comparison[method], f'{color}o-', label=method)

        plt.title('Comparison of Ultimate Loss Estimates')
        plt.xlabel('Accident Year')
        plt.ylabel('Ultimate Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(accident_years)
        plt.tight_layout()
        plt.show()

        # Create a second plot for percentage differences if available
        if 'Actual Ultimate' in comparison.columns:
            plt.figure(figsize=(12, 8))

            # Plot percentage differences
            for i, method in enumerate(methods):
                color = colors[i % len(colors)]
                plt.plot(accident_years, comparison[f'{method} % Diff'], f'{color}o-', label=f'{method} % Diff')

            plt.title('Percentage Difference from Actual Ultimate')
            plt.xlabel('Accident Year')
            plt.ylabel('Percentage Difference (%)')
            plt.legend()
            plt.grid(True)
            plt.xticks(accident_years)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error plotting results: {e}")


def plot_method_comparison(comparison, metric='% Diff'):
    """
    Create a box plot comparison of estimation errors across methods.

    Parameters:
    - comparison: DataFrame with comparison data
    - metric: 'Ultimate' or '% Diff' to determine what to plot
    """
    try:
        if metric == '% Diff' and 'Actual Ultimate' not in comparison.columns:
            print("Warning: Cannot plot percentage differences without actual ultimates")
            return

        if metric == '% Diff':
            # Extract only the percentage difference columns
            plot_data = comparison[[col for col in comparison.columns if '% Diff' in col]]
            # Rename columns to remove the '% Diff' suffix
            plot_data.columns = [col.replace(' % Diff', '') for col in plot_data.columns]
        else:
            # Extract only the ultimate estimate columns
            plot_data = comparison[[col for col in comparison.columns if '% Diff' not in col]]

        if plot_data.empty:
            print(f"Warning: No data available for {metric} plot")
            return

        # Create box plot
        plt.figure(figsize=(12, 8))
        plot_data.boxplot()

        if metric == '% Diff':
            plt.title('Distribution of Percentage Differences from Actual Ultimate')
            plt.ylabel('Percentage Difference (%)')
            # Add a horizontal line at 0 (perfect estimation)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        else:
            plt.title('Distribution of Ultimate Loss Estimates')
            plt.ylabel('Ultimate Loss')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in method comparison plot: {e}")

