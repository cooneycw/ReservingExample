from src_code.step_01 import generate_claim_data, plot_development_patterns


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


if __name__ == "__main__":
    main()