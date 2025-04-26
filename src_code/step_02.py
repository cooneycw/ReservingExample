import chainladder as cl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def import_triangle(claim_df: pd.DataFrame, premium_df: pd.DataFrame, triangle_type='paid'):
    """
    Converts claim data into a chainladder Triangle object.

    Args:
        claim_df: DataFrame with claim data
        premium_df: DataFrame with premium data
        triangle_type: 'paid' or 'incurred'
    """
    df = claim_df.copy()
    print(f"Creating {triangle_type} triangle...")

    # Create accident dates
    df['accident_date'] = pd.to_datetime(df['accident_year'].astype(str) + '-12-31')

    # Create valuation dates based on calendar year
    df['valuation_date'] = pd.to_datetime(df['calendar_year'].astype(str) + '-12-31')

    # Select column based on triangle type
    column = 'paid_losses' if triangle_type == 'paid' else 'incd_losses'

    # Create triangle
    triangle = cl.Triangle(
        data=df,
        origin='accident_date',
        development='valuation_date',
        columns=[column],
        cumulative=True
    )

    return triangle


def calculate_aa_ultimates(claim_df: pd.DataFrame):
    """
    Calculate appointed actuary ultimates for each accident year.
    AA Ultimate = paid_losses + case_reserves + aa_ibnr at the latest evaluation date
    """
    latest_eval = claim_df.groupby('accident_year').agg({
        'paid_losses': 'last',
        'case_reserves': 'last',
        'aa_ibnr': 'last',
        'dev_month': 'max'
    }).reset_index()

    latest_eval['aa_ultimate'] = (latest_eval['paid_losses'] +
                                  latest_eval['case_reserves'] +
                                  latest_eval['aa_ibnr'])

    return latest_eval


def analyze_triangle(triangle: cl.Triangle, triangle_type='paid', aa_ultimates=None):
    """
    Performs basic chainladder analysis.
    """
    print(f"\n--- {triangle_type.capitalize()} Triangle Analysis ---")
    print(f"\nTriangle Data (Original):")
    print(triangle)

    # Chain Ladder
    print(f"\n--- Chain Ladder Estimates ({triangle_type.capitalize()}) ---")
    cl_model = cl.Chainladder().fit(triangle)
    print(cl_model.ultimate_)

    # Development factors
    print("\n--- Development Factors ---")
    print(cl_model.ldf_)

    # Reserves
    print("\n--- Reserves ---")
    print(cl_model.ibnr_)

    # Visualization of development pattern
    plt.figure(figsize=(10, 6))
    triangle.plot(legend=True)
    plt.title(f'Cumulative {triangle_type.capitalize()} Development by Accident Year')
    plt.xlabel('Development Period (months)')
    plt.ylabel(f'{triangle_type.capitalize()} Losses')
    plt.tight_layout()
    plt.show()

    # Plot link ratios
    plt.figure(figsize=(10, 6))
    link_ratios = triangle.link_ratio
    link_ratios.plot(kind='bar', legend=False)
    plt.title(f'Link Ratios by Development Period ({triangle_type.capitalize()})')
    plt.ylabel('Link Ratio')
    plt.tight_layout()
    plt.show()

    return cl_model


def compare_ultimates(cl_paid_model, cl_incurred_model, aa_ultimates):
    """
    Compare AA ultimates with chainladder ultimates for both paid and incurred.
    Create visualization showing the differences.
    """
    # Extract chainladder ultimates
    paid_ultimate_triangle = cl_paid_model.ultimate_
    # Convert chainladder triangle to pandas DataFrame
    paid_ultimate_values = paid_ultimate_triangle.sum('development').to_frame()
    paid_ultimate_values.reset_index(inplace=True)
    paid_ultimate_values.columns = ['accident_year', 'cl_paid_ultimate']
    paid_ultimate_values['accident_year'] = paid_ultimate_values['accident_year'].astype(str).str.slice(0, 4).astype(
        int)

    incurred_ultimate_triangle = cl_incurred_model.ultimate_
    # Convert chainladder triangle to pandas DataFrame
    incurred_ultimate_values = incurred_ultimate_triangle.sum('development').to_frame()
    incurred_ultimate_values.reset_index(inplace=True)
    incurred_ultimate_values.columns = ['accident_year', 'cl_incurred_ultimate']
    incurred_ultimate_values['accident_year'] = incurred_ultimate_values['accident_year'].astype(str).str.slice(0,
                                                                                                                4).astype(
        int)

    # Combine with AA ultimates
    comparison = aa_ultimates.copy()
    comparison = comparison.merge(paid_ultimate_values, on='accident_year', how='left')
    comparison = comparison.merge(incurred_ultimate_values, on='accident_year', how='left')

    # Calculate differences (negative when AA is higher)
    comparison['cl_paid_vs_aa'] = comparison['cl_paid_ultimate'] - comparison['aa_ultimate']
    comparison['cl_incurred_vs_aa'] = comparison['cl_incurred_ultimate'] - comparison['aa_ultimate']

    # Add totals row
    totals = pd.DataFrame({
        'accident_year': ['Total'],
        'aa_ultimate': [comparison['aa_ultimate'].sum()],
        'cl_paid_ultimate': [comparison['cl_paid_ultimate'].sum()],
        'cl_incurred_ultimate': [comparison['cl_incurred_ultimate'].sum()],
        'cl_paid_vs_aa': [comparison['cl_paid_vs_aa'].sum()],
        'cl_incurred_vs_aa': [comparison['cl_incurred_vs_aa'].sum()]
    })

    comparison_with_totals = pd.concat([comparison, totals], ignore_index=True)

    print("\n--- Ultimate Comparison ---")
    print(comparison_with_totals[['accident_year', 'aa_ultimate', 'cl_paid_ultimate', 'cl_incurred_ultimate',
                                  'cl_paid_vs_aa', 'cl_incurred_vs_aa']])
    print("\nNote: Negative differences indicate AA estimates are higher than CL estimates\n")

    # For visualization, use original comparison without totals to avoid string/numeric issues
    # Visualization 1: Ultimates comparison
    plt.figure(figsize=(12, 8))
    width = 0.25
    x = np.arange(len(comparison))

    plt.bar(x - width, comparison['aa_ultimate'], width, label='AA Ultimate', alpha=0.8)
    plt.bar(x, comparison['cl_paid_ultimate'], width, label='CL Paid Ultimate', alpha=0.8)
    plt.bar(x + width, comparison['cl_incurred_ultimate'], width, label='CL Incurred Ultimate', alpha=0.8)

    plt.xlabel('Accident Year')
    plt.ylabel('Ultimate Loss')
    plt.title('Comparison of Ultimate Loss Estimates')
    plt.xticks(x, comparison['accident_year'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Differences with totals
    plt.figure(figsize=(12, 8))
    width = 0.35
    x = np.arange(len(comparison) + 1)  # Include space for totals

    # Create data with totals for this visualization
    labels = list(comparison['accident_year'].astype(str)) + ['Total']
    paid_diff = list(comparison['cl_paid_vs_aa']) + [comparison['cl_paid_vs_aa'].sum()]
    incurred_diff = list(comparison['cl_incurred_vs_aa']) + [comparison['cl_incurred_vs_aa'].sum()]

    plt.bar(x - width / 2, paid_diff, width, label='CL Paid - AA', color='orange', alpha=0.8)
    plt.bar(x + width / 2, incurred_diff, width, label='CL Incurred - AA', color='green', alpha=0.8)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Accident Year')
    plt.ylabel('Difference (CL - AA)')
    plt.title('Chainladder vs AA Ultimate Estimates (Including Total)')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Visualization 3: Migration to ultimate (without totals)
    plt.figure(figsize=(12, 8))
    for idx, row in comparison.iterrows():
        ay = row['accident_year']
        plt.plot(['AA', 'CL Paid', 'CL Incurred'],
                 [row['aa_ultimate'], row['cl_paid_ultimate'], row['cl_incurred_ultimate']],
                 'o-', label=f'AY {ay}')

    plt.xlabel('Method')
    plt.ylabel('Ultimate Loss')
    plt.title('Migration of Ultimate Estimates by Accident Year')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualization 4: Summary table with totals
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')

    # Format the data for the table
    table_data = []
    for idx, row in comparison_with_totals.iterrows():
        if row['accident_year'] == 'Total':
            # Bold formatting for totals
            table_data.append([
                f"**{row['accident_year']}**",
                f"**{row['aa_ultimate']:,.0f}**",
                f"**{row['cl_paid_ultimate']:,.0f}**",
                f"**{row['cl_incurred_ultimate']:,.0f}**",
                f"**{row['cl_paid_vs_aa']:,.0f}**",
                f"**{row['cl_incurred_vs_aa']:,.0f}**"
            ])
        else:
            table_data.append([
                str(row['accident_year']),
                f"{row['aa_ultimate']:,.0f}",
                f"{row['cl_paid_ultimate']:,.0f}",
                f"{row['cl_incurred_ultimate']:,.0f}",
                f"{row['cl_paid_vs_aa']:,.0f}",
                f"{row['cl_incurred_vs_aa']:,.0f}"
            ])

    table = plt.table(cellText=table_data,
                      colLabels=['Accident Year', 'AA Ultimate', 'CL Paid Ultimate',
                                 'CL Incurred Ultimate', 'CL Paid - AA', 'CL Incurred - AA'],
                      cellLoc='center',
                      loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight total row
    for col in range(6):
        table[(len(comparison_with_totals), col)].set_facecolor('#f0f0f0')

    plt.title('Summary Comparison with Totals', pad=20)
    plt.tight_layout()
    plt.show()

    return comparison_with_totals


def plot_development_to_ultimate(claim_df, cl_paid_model, cl_incurred_model, aa_ultimates):
    """
    Create visualizations showing the development to ultimate for each method.
    """
    # Prepare data for visualization
    accident_years = aa_ultimates['accident_year'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Development to Ultimate by Method', fontsize=16)

    # Plot 1: AA development
    ax = axes[0, 0]
    for ay in accident_years:
        df_ay = claim_df[claim_df['accident_year'] == ay].sort_values('dev_month')
        df_ay['aa_emerging'] = df_ay['paid_losses'] + df_ay['case_reserves'] + df_ay['aa_ibnr']
        ax.plot(df_ay['dev_month'], df_ay['aa_emerging'], 'o-', label=f'AY {ay}')
    ax.set_xlabel('Development Month')
    ax.set_ylabel('Emerging Loss')
    ax.set_title('AA Development')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Paid development
    ax = axes[0, 1]
    for ay in accident_years:
        df_ay = claim_df[claim_df['accident_year'] == ay].sort_values('dev_month')
        ax.plot(df_ay['dev_month'], df_ay['paid_losses'], 'o-', label=f'AY {ay}')
    ax.set_xlabel('Development Month')
    ax.set_ylabel('Paid Loss')
    ax.set_title('Paid Development')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Incurred development
    ax = axes[1, 0]
    for ay in accident_years:
        df_ay = claim_df[claim_df['accident_year'] == ay].sort_values('dev_month')
        ax.plot(df_ay['dev_month'], df_ay['incd_losses'], 'o-', label=f'AY {ay}')
    ax.set_xlabel('Development Month')
    ax.set_ylabel('Incurred Loss')
    ax.set_title('Incurred Development')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Ultimate comparison
    ax = axes[1, 1]
    comparison_data = []

    # Extract ultimates properly using chainladder's to_frame method
    paid_ultimate_triangle = cl_paid_model.ultimate_
    paid_ultimate_values = paid_ultimate_triangle.sum('development').to_frame()
    paid_ultimate_values.reset_index(inplace=True)
    paid_ultimate_values.columns = ['accident_year', 'cl_paid_ultimate']
    paid_ultimate_values['accident_year'] = paid_ultimate_values['accident_year'].astype(str).str.slice(0, 4).astype(
        int)

    incurred_ultimate_triangle = cl_incurred_model.ultimate_
    incurred_ultimate_values = incurred_ultimate_triangle.sum('development').to_frame()
    incurred_ultimate_values.reset_index(inplace=True)
    incurred_ultimate_values.columns = ['accident_year', 'cl_incurred_ultimate']
    incurred_ultimate_values['accident_year'] = incurred_ultimate_values['accident_year'].astype(str).str.slice(0,
                                                                                                                4).astype(
        int)

    for ay in accident_years:
        aa_ult = aa_ultimates[aa_ultimates['accident_year'] == ay]['aa_ultimate'].values[0]
        cl_paid_ult = paid_ultimate_values[paid_ultimate_values['accident_year'] == ay]['cl_paid_ultimate'].values[0]
        cl_incurred_ult = \
        incurred_ultimate_values[incurred_ultimate_values['accident_year'] == ay]['cl_incurred_ultimate'].values[0]

        comparison_data.append({'Accident Year': ay, 'AA': aa_ult,
                                'CL Paid': cl_paid_ult, 'CL Incurred': cl_incurred_ult})

    comp_df = pd.DataFrame(comparison_data)
    comp_df.plot(x='Accident Year', y=['AA', 'CL Paid', 'CL Incurred'],
                 kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Accident Year')
    ax.set_ylabel('Ultimate Loss')
    ax.set_title('Ultimate Loss Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
