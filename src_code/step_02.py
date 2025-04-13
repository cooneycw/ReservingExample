import pandas as pd
import numpy as np
import chainladder as cl
import sys
from datetime import datetime


def import_triangle(claim_df, premium_df):
    """Import the claims triangle into chainladder"""
    # Print version information
    print("Package Versions:")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"chainladder: {cl.__version__}")
    print(f"Python {sys.version} on {sys.platform}")

    # Convert accident_year to datetime
    claim_df['origin'] = pd.to_datetime(claim_df['accident_year'], format='%Y')

    # Create valuation dates (year-end)
    claim_df['valuation'] = pd.to_datetime(claim_df['calendar_year'], format='%Y') + pd.offsets.YearEnd(0)

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

    return triangle

