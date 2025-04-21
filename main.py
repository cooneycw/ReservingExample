import pandas as pd

# Format floating point numbers to have a specific number of decimal places
pd.set_option('display.float_format', '{:.2f}'.format)
# Increase the maximum column width (for columns with long text)
pd.set_option('display.max_columns', None)
# Set width to accommodate all columns horizontally
pd.set_option('display.width', 1000)
# Increase column width for text columns if needed
pd.set_option('display.max_colwidth', 100)
from src_code.step_01 import generate_claim_data, plot_development_patterns
from src_code.step_02 import (import_triangle, analyze_triangle, calculate_aa_ultimates,
                              compare_ultimates, plot_development_to_ultimate)


def main():
    # Generate claim data
    claim_df, premium_df = generate_claim_data()

    # Display the first few rows of each DataFrame
    print("Claim Data:")
    print(claim_df.head(10))
    print("\nPremium Data:")
    print(premium_df.head(10))

    # Plot development patterns
    plot_development_patterns(claim_df, metric='paid_losses')
    plot_development_patterns(claim_df, metric='incd_losses')

    # Calculate AA ultimates
    aa_ultimates = calculate_aa_ultimates(claim_df)
    print("\n--- Appointed Actuary Ultimates ---")
    print(aa_ultimates)

    # Create and analyze paid triangle
    paid_triangle = import_triangle(claim_df, premium_df, triangle_type='paid')
    cl_paid_model = analyze_triangle(paid_triangle, triangle_type='paid', aa_ultimates=aa_ultimates)

    # Create and analyze incurred triangle
    incurred_triangle = import_triangle(claim_df, premium_df, triangle_type='incurred')
    cl_incurred_model = analyze_triangle(incurred_triangle, triangle_type='incurred', aa_ultimates=aa_ultimates)

    # Compare ultimates
    comparison = compare_ultimates(cl_paid_model, cl_incurred_model, aa_ultimates)

    # Plot development to ultimate for all methods
    plot_development_to_ultimate(claim_df, cl_paid_model, cl_incurred_model, aa_ultimates)

    # Additional analysis - percentage differences
    print("\n--- Percentage Differences from AA Ultimate ---")
    # Calculate percentage differences excluding the total row
    comparison_for_pct = comparison[comparison['accident_year'] != 'Total'].copy()
    comparison_for_pct['pct_diff_paid'] = (
                comparison_for_pct['cl_paid_vs_aa'] / comparison_for_pct['aa_ultimate'] * 100).round(2)
    comparison_for_pct['pct_diff_incurred'] = (
                comparison_for_pct['cl_incurred_vs_aa'] / comparison_for_pct['aa_ultimate'] * 100).round(2)

    # Add total row for percentage differences
    total_aa = comparison_for_pct['aa_ultimate'].sum()
    total_paid_diff = comparison_for_pct['cl_paid_vs_aa'].sum()
    total_incurred_diff = comparison_for_pct['cl_incurred_vs_aa'].sum()

    total_pct_row = pd.DataFrame({
        'accident_year': ['Total'],
        'pct_diff_paid': [round(total_paid_diff / total_aa * 100, 2)],
        'pct_diff_incurred': [round(total_incurred_diff / total_aa * 100, 2)]
    })

    pct_comparison = pd.concat([comparison_for_pct[['accident_year', 'pct_diff_paid', 'pct_diff_incurred']],
                                total_pct_row], ignore_index=True)
    print(pct_comparison)


if __name__ == "__main__":
    main()