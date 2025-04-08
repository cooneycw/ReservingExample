import pandas as pd
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt


def generate_claim_data(
        accident_years_range=(2017, 2024),  # Range of accident years
        dev_years=9,  # Maximum development years
        claim_inflation=0.05,  # Annual claim inflation rate
        payment_patterns=None,  # Custom payment patterns by development year
        case_reserve_patterns=None,  # Custom case reserve patterns
        ibnr_patterns=None,  # Custom IBNR patterns
        loss_ratio_mean=0.65,  # Average loss ratio
        loss_ratio_std=0.08,  # Standard deviation of loss ratio
        premium_base=1000000,  # Base premium amount
        premium_growth=0.03,  # Annual premium growth rate
        random_seed=42  # Random seed for reproducibility
):
    """
    Generate synthetic insurance claim data for reserving analysis.

    Returns:
        Tuple containing:
        - claim_data: DataFrame with accident_year, calendar_year, paid_losses, case_reserves, incurred_losses, and aa_ibnr
        - premium_data: DataFrame with accident_year and earned_premiums
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Extract accident years range
    min_accident_year, max_accident_year = accident_years_range
    accident_years = list(range(min_accident_year, max_accident_year + 1))
    n_accident_years = len(accident_years)

    # Set up default payment patterns if not provided
    # This represents cumulative percentage paid by development year
    if payment_patterns is None:
        # Creating a realistic payment pattern (cumulative %)
        # For long-tail business, claims develop over several years
        payment_patterns = {
            0: 0.30,  # 30% paid in the first year
            1: 0.60,  # 60% paid by end of second year
            2: 0.75,
            3: 0.85,
            4: 0.90,
            5: 0.93,
            6: 0.96,
            7: 0.98,
            8: 0.99,
            9: 1.00  # Fully paid by year 10
        }

    # Set up default case reserve patterns if not provided
    if case_reserve_patterns is None:
        # Case reserves as % of ultimate by development year
        # Initially high, then decreases as claims are paid
        case_reserve_patterns = {
            0: 0.50,  # Initially, 50% of ultimate is set as case reserves
            1: 0.35,  # After 1 year, 35% of ultimate remains as case reserves
            2: 0.23,
            3: 0.14,
            4: 0.09,
            5: 0.06,
            6: 0.03,
            7: 0.02,
            8: 0.01,
            9: 0.00  # No case reserves by year 10
        }

    # Set up default IBNR patterns if not provided
    if ibnr_patterns is None:
        # IBNR as % of ultimate by development year
        # Initially high, then decreases as claims are reported and developed
        ibnr_patterns = {
            0: 0.20,  # 20% of ultimate is IBNR in first year
            1: 0.05,  # 5% of ultimate is IBNR after 1 year
            2: 0.02,
            3: 0.01,
            4: 0.01,
            5: 0.01,
            6: 0.01,
            7: 0.00,
            8: 0.00,
            9: 0.00  # No IBNR by year 10
        }

    # Generate earned premiums with growth and random fluctuation
    premiums = []
    for i, ay in enumerate(accident_years):
        # Base premium with annual growth
        base_premium = premium_base * (1 + premium_growth) ** i

        # Add some random fluctuation to premium (±5%)
        fluctuation = 1 + (np.random.random() - 0.5) * 0.1
        earned_premium = base_premium * fluctuation

        premiums.append({
            'accident_year': ay,
            'earned_premium': round(earned_premium, 2)
        })

    premium_df = pd.DataFrame(premiums)

    # Generate ultimate losses based on loss ratios
    ultimates = {}
    for ay in accident_years:
        # Sample a loss ratio with some randomness
        loss_ratio = np.random.normal(loss_ratio_mean, loss_ratio_std)
        loss_ratio = max(0.4, min(loss_ratio, 0.9))  # Bound between 40% and 90%

        # Get premium for this accident year
        premium = premium_df[premium_df['accident_year'] == ay]['earned_premium'].values[0]

        # Calculate ultimate loss amount
        ultimate = premium * loss_ratio

        # Add inflation impact based on accident year
        # Later years have higher inflation impact
        inflation_factor = (1 + claim_inflation) ** (ay - min_accident_year)
        ultimate *= inflation_factor

        ultimates[ay] = ultimate

    # Now we'll create the triangle/rectangle dataset
    rows = []

    # For each accident year
    for ay in accident_years:
        ultimate_loss = ultimates[ay]

        # Loop through all possible development years up to the maximum
        for dev_year in range(min(dev_years, datetime.now().year - ay + 1)):
            # Calendar year is accident year + development year
            cy = ay + dev_year

            # Calculate paid losses based on payment pattern and ultimate loss
            paid_percent = payment_patterns.get(dev_year, 1.0)  # Default to 100% if beyond pattern
            paid_loss = ultimate_loss * paid_percent

            # Add some random variation to paid amounts (±3%)
            paid_random_factor = 1 + (np.random.random() - 0.5) * 0.06
            paid_loss *= paid_random_factor

            # Calculate case reserves based on pattern
            case_percent = case_reserve_patterns.get(dev_year, 0.0)
            case_reserve = ultimate_loss * case_percent

            # Add some random variation to case reserves (±10%)
            case_random_factor = 1 + (np.random.random() - 0.5) * 0.2
            case_reserve *= case_random_factor

            # Calculate IBNR (actuary's estimate of IBNR, not actual)
            ibnr_percent = ibnr_patterns.get(dev_year, 0.0)
            aa_ibnr = ultimate_loss * ibnr_percent

            # Add some random variation to IBNR (±15%)
            ibnr_random_factor = 1 + (np.random.random() - 0.5) * 0.3
            aa_ibnr *= ibnr_random_factor

            # Incurred losses = paid + case reserves
            incd_loss = paid_loss + case_reserve

            # Create row entry
            row = {
                'accident_year': ay,
                'calendar_year': cy,
                'dev_year': dev_year,
                'paid_losses': round(paid_loss, 2),
                'case_reserves': round(case_reserve, 2),
                'incd_losses': round(incd_loss, 2),
                'aa_ibnr': round(aa_ibnr, 2),
                'ultimate_loss': round(ultimate_loss, 2)  # For reference
            }
            rows.append(row)

    # Create DataFrame
    claim_df = pd.DataFrame(rows)

    # Filter to ensure we only have calendar years up to current
    max_cy = datetime.now().year
    claim_df = claim_df[claim_df['calendar_year'] <= max_cy]

    return claim_df, premium_df


def plot_development_patterns(claim_df, metric='paid_losses', by='accident_year'):
    """Plot development patterns for a specified metric"""

    # Pivot the data to get development by specified grouping
    pivot_df = claim_df.pivot_table(
        index=by,
        columns='dev_year',
        values=metric,
        aggfunc='sum'
    )

    # Plot the development
    plt.figure(figsize=(12, 8))
    for year in pivot_df.index:
        plt.plot(pivot_df.columns, pivot_df.loc[year], 'o-', label=f'{by}={year}')

    plt.title(f'Development of {metric} by {by}')
    plt.xlabel('Development Year')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.show()

