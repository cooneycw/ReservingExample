"""
Main runner script for insurance reserving analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from src_code.step_01 import generate_claim_data, plot_development_patterns
from src_code.step_03 import (
    run_reserving_analysis,
    compare_results,
    create_reserve_summary,
    plot_results,
    plot_method_comparison
)


def analyze_results(comparison, ibnr_summary):
    """
    Perform additional analysis on results to identify patterns and insights.

    Parameters:
    - comparison: DataFrame with comparison of ultimates
    - ibnr_summary: DataFrame with IBNR summary
    """
    # Skip additional analysis if we don't have actual ultimates for comparison
    if 'Actual Ultimate' not in comparison.columns:
        print("Warning: Skipping additional analysis - no actual ultimates for comparison")
        return {}

    try:
        # Set up the plot style
        sns.set_style('whitegrid')
        plt.rcParams.update({'font.size': 12})

        # 1. Create a heatmap of percentage differences
        diff_cols = [col for col in comparison.columns if '% Diff' in col]

        if diff_cols:
            plt.figure(figsize=(14, 8))

            # Prepare data for heatmap
            heatmap_data = comparison[diff_cols].copy()
            heatmap_data.columns = [col.replace(' % Diff', '') for col in heatmap_data.columns]

            # Create heatmap
            ax = sns.heatmap(
                heatmap_data,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt=".1f",
                linewidths=0.5,
                cbar_kws={'label': 'Percentage Difference (%)'}
            )

            plt.title('Estimation Error by Method and Accident Year')
            plt.ylabel('Accident Year')
            plt.tight_layout()
            plt.show()

            # 2. Create a method reliability analysis
            # Calculate mean absolute percentage error (MAPE) for each method
            mape_by_method = {}
            for col in diff_cols:
                method_name = col.replace(' % Diff', '')
                mape_by_method[method_name] = abs(comparison[col]).mean()

            # Convert to DataFrame and sort
            mape_df = pd.DataFrame(list(mape_by_method.items()), columns=['Method', 'MAPE'])
            mape_df = mape_df.sort_values('MAPE')

            # Plot MAPE by method
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Method', y='MAPE', data=mape_df)

            # Add value labels on top of bars
            for i, v in enumerate(mape_df['MAPE']):
                ax.text(i, v + 0.5, f'{v:.2f}%', ha='center')

            plt.title('Mean Absolute Percentage Error by Method')
            plt.ylabel('Mean Absolute Percentage Error (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # 3. Create reliability by accident year analysis
            # Calculate mean absolute percentage error by accident year
            mape_by_ay = {}
            for ay in comparison.index:
                abs_pct_diff = abs(comparison.loc[ay, diff_cols])
                mape_by_ay[ay] = abs_pct_diff.mean()

            # Convert to DataFrame
            mape_ay_df = pd.DataFrame(list(mape_by_ay.items()), columns=['Accident Year', 'MAPE'])

            # Plot MAPE by accident year
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Accident Year', y='MAPE', data=mape_ay_df)

            # Add value labels on top of bars
            for i, v in enumerate(mape_ay_df['MAPE']):
                ax.text(i, v + 0.5, f'{v:.2f}%', ha='center')

            plt.title('Mean Absolute Percentage Error by Accident Year')
            plt.ylabel('Mean Absolute Percentage Error (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # 4. Compare IBNR estimates with actual IBNR
        if 'Actual IBNR' in ibnr_summary.columns:
            ibnr_methods = [col for col in ibnr_summary.columns if 'IBNR' in col and col != 'Actual IBNR']

            if ibnr_methods:
                ibnr_plot_data = ibnr_summary[['Actual IBNR'] + ibnr_methods]

                # Melt the DataFrame for easier plotting
                ibnr_plot_long = pd.melt(
                    ibnr_plot_data.reset_index(),
                    id_vars=['accident_year'],
                    value_vars=['Actual IBNR'] + ibnr_methods,
                    var_name='Method',
                    value_name='IBNR'
                )

                # Plot IBNR by method and accident year
                plt.figure(figsize=(14, 8))
                sns.lineplot(
                    data=ibnr_plot_long,
                    x='accident_year',
                    y='IBNR',
                    hue='Method',
                    marker='o'
                )

                plt.title('IBNR Estimates by Method and Accident Year')
                plt.xlabel('Accident Year')
                plt.ylabel('IBNR')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.show()

        # Return analysis results
        return {
            'mape_by_method': mape_df if 'mape_df' in locals() else None,
            'mape_by_accident_year': mape_ay_df if 'mape_ay_df' in locals() else None
        }

    except Exception as e:
        print(f"Error in additional analysis: {e}")
        traceback.print_exc()
        return {}


def run_full_analysis(
        accident_years_range=(2017, 2024),
        dev_years=8,
        claim_inflation=0.04,
        loss_ratio_mean=0.65,
        premium_base=1000000,
        premium_growth=0.03,
        random_seed=42
):
    """
    Run a full reserving analysis workflow including data generation, method comparison, and visualization.

    Parameters:
    - accident_years_range: Tuple with the range of accident years
    - dev_years: Maximum development years
    - claim_inflation: Annual claim inflation rate
    - loss_ratio_mean: Average loss ratio
    - premium_base: Base premium amount
    - premium_growth: Annual premium growth rate
    - random_seed: Random seed for reproducibility

    Returns:
    - Dictionary with all results
    """
    print(f"Generating claim data with accident years {accident_years_range}, development periods {dev_years}...")

    try:
        # Generate claim data with the specified parameters
        claim_df, premium_df = generate_claim_data(
            accident_years_range=accident_years_range,
            dev_years=dev_years,
            claim_inflation=claim_inflation,
            loss_ratio_mean=loss_ratio_mean,
            premium_base=premium_base,
            premium_growth=premium_growth,
            random_seed=random_seed
        )

        # Display basic info about the generated data
        print(f"Generated data for {len(claim_df['accident_year'].unique())} accident years")
        print(f"Total claim records: {len(claim_df)}")
        print(f"Total premium records: {len(premium_df)}")

        # Show the latest diagonal of data
        latest_data = claim_df.sort_values(['accident_year', 'dev_year'])
        latest_data = latest_data.groupby('accident_year').last()
        print("\nLatest diagonal of claim data:")
        print(latest_data[['dev_year', 'paid_losses', 'case_reserves', 'incd_losses', 'aa_ibnr']])

        # Plot development patterns
        print("\nPlotting development patterns...")
        plot_development_patterns(claim_df, metric='paid_losses')
        plot_development_patterns(claim_df, metric='incd_losses')

        # Run reserving analysis with all methods
        print("\nRunning reserving analysis with all methods...")
        results = run_reserving_analysis(claim_df, premium_df, avg_loss_ratio=loss_ratio_mean)

        # Compare results to actual ultimates
        comparison = compare_results(results, claim_df)
        print("\n===== Comparison of Methods =====")
        print(comparison)

        # Create IBNR summary
        ibnr_summary = create_reserve_summary(results, claim_df)
        print("\n===== IBNR Summary =====")
        print(ibnr_summary)

        # Plot comparisons
        print("\nPlotting result comparisons...")
        plot_results(comparison)
        plot_method_comparison(comparison, 'Ultimate')
        if 'Actual Ultimate' in comparison.columns:
            plot_method_comparison(comparison, '% Diff')

        # Perform additional analysis
        print("\nPerforming additional analysis...")
        analysis_results = analyze_results(comparison, ibnr_summary)

        # Print method ranking by MAPE if available
        if analysis_results.get('mape_by_method') is not None:
            print("\n===== Method Ranking by MAPE =====")
            print(analysis_results['mape_by_method'])

        # Return all results
        return {
            'claim_data': claim_df,
            'premium_data': premium_df,
            'reserving_results': results,
            'comparison': comparison,
            'ibnr_summary': ibnr_summary,
            'analysis_results': analysis_results
        }

    except Exception as e:
        print(f"Error in analysis: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the full analysis
    full_results = run_full_analysis(
        accident_years_range=(2017, 2024),
        dev_years=8,
        claim_inflation=0.04,
        loss_ratio_mean=0.65,
        random_seed=42
    )

    if full_results:
        print("\nAnalysis complete!")
    else:
        print("\nAnalysis failed!")