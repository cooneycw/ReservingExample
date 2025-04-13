import pandas as pd
import numpy as np
import chainladder as cl
from src_code.step_01 import generate_claim_data


def import_triangle(claim_df, premium_df):
    # Print version information
    print("Package Versions:")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"chainladder: {cl.__version__}")
    print("\n")

    # Convert to triangle format
    triangle = cl.Triangle(
        claim_df,
        origin='accident_year',
        development='dev_year',
        columns='paid_losses',
        cumulative=True
    )

    # Display the triangle
    print("Claims Triangle:")
    print(triangle)

    return triangle
