import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def percentage_supplemental_required_method(claim_df, method_params=None):
    """
    Implement the "Percentage Supplemental Required" reserving method.

    This method evaluates case incurred adequacy based on historical patterns and
    projects ultimate losses based on the historical need for supplemental reserves
    beyond case incurred amounts.

    Parameters:
    - claim_df: DataFrame with claim data containing at minimum:
                accident_year, dev_year, incd_losses, ultimate_loss (for mature years)
    - method_params: Dictionary of parameters including:
                    - mature_years: list of accident years considered fully developed
                    - averaging_method: how to compute factors ('simple', 'weighted', 'medial')
                    - exclude_outliers: whether to exclude outlier factors
                    - outlier_threshold: z-score threshold for outlier detection

    Returns:
    - Dictionary containing ultimate loss and IBNR estimates by accident year
    """
    if method_params is None:
        method_params = {
            'mature_years': None,  # Will be determined based on sufficiently developed years
            'averaging_method': 'medial',  # Options: 'simple', 'weighted', 'medial'
            'exclude_outliers': True,
            'outlier_threshold': 2.0  # Z-score threshold
        }

    # Create a copy of the data to avoid modifying the original
    df = claim_df.copy()

    # If mature years not specified, determine based on development
    # (e.g., years with at least 7 development periods)
    if method_params['mature_years'] is None:
        dev_by_ay = df.groupby('accident_year')['dev_year'].max()
        method_params['mature_years'] = dev_by_ay[dev_by_ay >= 7].index.tolist()

    # Extract ultimate losses for mature years (if in data) or use highest developed incurred as proxy
    ultimates = {}
    for ay in method_params['mature_years']:
        ay_data = df[df['accident_year'] == ay]
        if 'ultimate_loss' in df.columns:
            # If ultimate loss is directly in the data (from simulation)
            ultimates[ay] = ay_data['ultimate_loss'].iloc[0]
        else:
            # Use the highest development year's incurred loss as proxy for ultimate
            max_dev = ay_data['dev_year'].max()
            ultimates[ay] = ay_data[ay_data['dev_year'] == max_dev]['incd_losses'].iloc[0]

    # Create a new DataFrame to store case incurred adequacy ratios
    adequacy_data = []

    # Calculate historical case incurred adequacy by accident year and development year
    for ay in method_params['mature_years']:
        ultimate = ultimates[ay]

        # Get case incurred at each development period
        ay_data = df[df['accident_year'] == ay]
        for _, row in ay_data.iterrows():
            dev_year = row['dev_year']
            incd_loss = row['incd_losses']

            # Calculate the ratio of ultimate to case incurred
            # This represents how much the case incurred needs to be multiplied by
            # to reach the ultimate loss (the "supplemental factor")
            if incd_loss > 0:
                supplemental_factor = ultimate / incd_loss
            else:
                supplemental_factor = np.nan

            adequacy_data.append({
                'accident_year': ay,
                'dev_year': dev_year,
                'incurred_loss': incd_loss,
                'ultimate_loss': ultimate,
                'supplemental_factor': supplemental_factor
            })

    # Convert to DataFrame
    adequacy_df = pd.DataFrame(adequacy_data)

    # Calculate selected supplemental factors by development period
    selected_factors = {}
    for dev in sorted(adequacy_df['dev_year'].unique()):
        dev_data = adequacy_df[adequacy_df['dev_year'] == dev]
        factors = dev_data['supplemental_factor'].dropna().values

        if len(factors) == 0:
            selected_factors[dev] = 1.0
            continue

        # Exclude outliers if specified
        if method_params['exclude_outliers'] and len(factors) > 3:
            z_scores = np.abs((factors - np.mean(factors)) / np.std(factors))
            factors = factors[z_scores < method_params['outlier_threshold']]

        # Apply the selected averaging method
        if method_params['averaging_method'] == 'simple':
            # Simple average
            selected_factors[dev] = np.mean(factors)
        elif method_params['averaging_method'] == 'weighted':
            # Weight by accident year (more recent years get higher weight)
            weights = dev_data['accident_year'].values - min(dev_data['accident_year'].values) + 1
            selected_factors[dev] = np.average(factors, weights=weights)
        elif method_params['averaging_method'] == 'medial':
            # Average of the middle 50% (exclude high/low 25%)
            q1, q3 = np.percentile(factors, [25, 75])
            medial_factors = factors[(factors >= q1) & (factors <= q3)]
            if len(medial_factors) > 0:
                selected_factors[dev] = np.mean(medial_factors)
            else:
                selected_factors[dev] = np.mean(factors)
        else:
            # Default to simple average
            selected_factors[dev] = np.mean(factors)

    # Now apply these factors to estimate ultimates for all accident years
    results = {}
    for ay in sorted(df['accident_year'].unique()):
        ay_data = df[df['accident_year'] == ay]
        latest_dev = ay_data['dev_year'].max()
        latest_row = ay_data[ay_data['dev_year'] == latest_dev].iloc[0]

        # Get case incurred and paid losses
        case_incurred = latest_row['incd_losses']
        paid_loss = latest_row['paid_losses']
        case_reserve = latest_row['case_reserves']

        # Apply supplemental factor based on development age
        if latest_dev in selected_factors:
            supplemental_factor = selected_factors[latest_dev]
        else:
            # If dev period not in factors, use the highest available
            max_dev_in_factors = max(selected_factors.keys())
            supplemental_factor = selected_factors[max_dev_in_factors]

        # Calculate estimated ultimate
        ultimate_estimate = case_incurred * supplemental_factor

        # Calculate IBNR (ultimate - case incurred)
        ibnr_estimate = ultimate_estimate - case_incurred

        results[ay] = {
            'accident_year': ay,
            'development_age': latest_dev,
            'paid_losses': paid_loss,
            'case_reserves': case_reserve,
            'case_incurred': case_incurred,
            'supplemental_factor': supplemental_factor,
            'ultimate_estimate': ultimate_estimate,
            'ibnr_estimate': ibnr_estimate,
            'percent_supplemental': (supplemental_factor - 1) * 100  # Percentage supplemental required
        }

    # Convert to DataFrame
    results_df = pd.DataFrame(results).T.reset_index(drop=True)

    return {
        'supplemental_factors': pd.Series(selected_factors),
        'results': results_df,
        'ultimate': results_df.set_index('accident_year')['ultimate_estimate'],
        'ibnr': results_df.set_index('accident_year')['ibnr_estimate']
    }


def plot_supplemental_factors(factors):
    """
    Plot the supplemental factors by development period.

    Parameters:
    - factors: Series of supplemental factors indexed by development period
    """
    plt.figure(figsize=(10, 6))
    plt.plot(factors.index, factors.values, 'o-', linewidth=2)
    plt.title('Supplemental Factors by Development Age')
    plt.xlabel('Development Age')
    plt.ylabel('Supplemental Factor')
    plt.grid(True)
    plt.xticks(factors.index)

    # Add a horizontal line at 1.0 (no supplemental required)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)

    # Add data labels
    for x, y in zip(factors.index, factors.values):
        plt.annotate(f'{y:.2f}',
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.show()


# Example usage
def test_supplemental_required():
    from src_code.step_01 import generate_claim_data

    # Generate some test data
    claim_df, premium_df = generate_claim_data(
        accident_years_range=(2017, 2024),
        dev_years=8
    )

    # Run the percentage supplemental required method
    psr_results = percentage_supplemental_required_method(
        claim_df,
        method_params={
            'mature_years': [2017, 2018],  # Years considered fully developed
            'averaging_method': 'medial',
            'exclude_outliers': True
        }
    )

    # Display results
    print("Supplemental Factors by Development Age:")
    print(psr_results['supplemental_factors'])

    print("\nDetailed Results:")
    print(psr_results['results'])

    # Plot the supplemental factors
    plot_supplemental_factors(psr_results['supplemental_factors'])

    return psr_results


if __name__ == "__main__":
    test_supplemental_required()