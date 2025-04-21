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
from src_code.step_02 import import_triangle, analyze_triangle


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
    triangle = import_triangle(claim_df, premium_df)
    analyze_triangle(triangle)

if __name__ == "__main__":
    main()