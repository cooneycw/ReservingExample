import chainladder as cl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def import_triangle(claim_df: pd.DataFrame, premium_df: pd.DataFrame):
    """
    Converts claim data into a chainladder Triangle object.
    First, check if the data is cumulative or incremental.
    """
    # Check if data is cumulative by examining if values increase over development
    is_cumulative = False

    df = claim_df.copy()
    print(f"Data appears to be {'cumulative' if is_cumulative else 'incremental'}")
    # Create accident dates
    df['accident_date'] = pd.to_datetime(df['accident_year'].astype(str) + '-12-31')

    # Create valuation dates based on calendar year
    df['valuation_date'] = pd.to_datetime(df['calendar_year'].astype(str) + '-12-31')

    # Create triangle
    triangle = cl.Triangle(
        data=df,
        origin='accident_date',
        development='valuation_date',
        columns=['paid_losses'],
        cumulative=True
    )

    return triangle


def analyze_triangle(triangle: cl.Triangle):
    """
    Performs various reserving analyses and plots development.
    """
    print("\nTriangle Data (Original):")
    print(triangle)

    # If data is cumulative, we might want to convert to incremental for better visualization
    if triangle.is_cumulative:
        print("\nTriangle Data (Incremental):")
        print(triangle.incr_to_cum())

    # Chain Ladder
    print("\n--- Chain Ladder Estimates ---")
    cl_model = cl.Chainladder().fit(triangle)
    print(cl_model.ultimate_)

    # Development factors
    print("\n--- Development Factors ---")
    print(cl_model.ldf_)

    # Reserves
    print("\n--- Reserves ---")
    print(cl_model.ibnr_)

    # Visualization
    # Plot development pattern
    triangle.plot(legend=True)
    plt.title('Cumulative Paid Losses by Accident Year')
    plt.xlabel('Development Period (months)')
    plt.ylabel('Paid Losses')
    plt.tight_layout()
    plt.show()

    # Plot link ratios
    if triangle.is_cumulative:
        link_ratios = triangle.link_ratio
        link_ratios.plot(kind='bar', legend=False)
        plt.title('Link Ratios by Development Period')
        plt.ylabel('Link Ratio')
        plt.tight_layout()
        plt.show()