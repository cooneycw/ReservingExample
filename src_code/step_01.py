import pandas as pd
#Format floating point numbers to have a specific number of decimal places
pd.set_option('display.float_format', '{:.2f}'.format)
# Increase the maximum column width (for columns with long text)
pd.set_option('display.max_columns', None)
# Set width to accommodate all columns horizontally
pd.set_option('display.width', 1000)
# Increase column width for text columns if needed
pd.set_option('display.max_colwidth', 100)
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt


def generate_claim_data(
        accident_years_range=(2017, 2024),  # Range of accident years
        dev_years=9,                        # Max development periods (in years)
        claim_inflation=0.05,               # Annual claim inflation
        payment_patterns=None,              # % paid by development month
        case_reserve_patterns=None,         # % case reserves by dev month
        ibnr_patterns=None,                 # % IBNR by dev month
        loss_ratio_mean=0.65,
        loss_ratio_std=0.08,
        premium_base=1_000_000,
        premium_growth=0.03,
        random_seed=42,
        evaluation_year=2024                # Add this parameter
):
    """
    Generate synthetic insurance claim data for reserving analysis.

    Returns:
        claim_df: DataFrame with accident_year, calendar_year, dev_month, paid_losses, case_reserves, incd_losses, aa_ibnr, ultimate_loss
        premium_df: DataFrame with accident_year, earned_premium
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    min_ay, max_ay = accident_years_range
    accident_years = list(range(min_ay, max_ay + 1))

    # Default payment pattern (cumulative % paid by dev month)
    if payment_patterns is None:
        payment_patterns = {
            12: 0.30,
            24: 0.60,
            36: 0.75,
            48: 0.85,
            60: 0.90,
            72: 0.93,
            84: 0.96,
            96: 0.98,
            108: 0.99,
            120: 1.00
        }

    if case_reserve_patterns is None:
        case_reserve_patterns = {
            12: 0.50,
            24: 0.35,
            36: 0.23,
            48: 0.14,
            60: 0.09,
            72: 0.06,
            84: 0.03,
            96: 0.02,
            108: 0.01,
            120: 0.00
        }

    if ibnr_patterns is None:
        ibnr_patterns = {
            12: 0.25,
            24: 0.07,
            36: 0.04,
            48: 0.03,
            60: 0.02,
            72: 0.02,
            84: 0.01,
            96: 0.01,
            108: 0.00,
            120: 0.00
        }

    # Generate premium data
    premiums = []
    for i, ay in enumerate(accident_years):
        base_premium = premium_base * (1 + premium_growth) ** i
        fluctuation = 1 + (np.random.random() - 0.5) * 0.1
        earned_premium = base_premium * fluctuation
        premiums.append({
            'accident_year': ay,
            'earned_premium': round(earned_premium, 2)
        })
    premium_df = pd.DataFrame(premiums)

    # Generate ultimate losses
    ultimates = {}
    for ay in accident_years:
        loss_ratio = np.random.normal(loss_ratio_mean, loss_ratio_std)
        loss_ratio = np.clip(loss_ratio, 0.4, 0.9)
        premium = premium_df.loc[premium_df.accident_year == ay, 'earned_premium'].values[0]
        inflation_factor = (1 + claim_inflation) ** (ay - min_ay)
        ultimate = premium * loss_ratio * inflation_factor
        ultimates[ay] = ultimate

    # Create triangle-like structure
    rows = []
    current_year = evaluation_year  # Changed from datetime.now().year
    for ay in accident_years:
        ultimate_loss = ultimates[ay]

        for dev_year in range(min(dev_years + 1, current_year - ay + 1)):
            dev_month = (dev_year + 1) * 12
            cy = ay + dev_year

            if dev_month > 120:
                continue

            paid = ultimate_loss * payment_patterns.get(dev_month, 1.0)
            paid *= 1 + (np.random.random() - 0.5) * 0.06  # ±3%

            case = ultimate_loss * case_reserve_patterns.get(dev_month, 0.0)
            case *= 1 + (np.random.random() - 0.5) * 0.2  # ±10%

            ibnr = ultimate_loss * ibnr_patterns.get(dev_month, 0.0)
            ibnr *= 1 + (np.random.random() - 0.5) * 0.3  # ±15%

            incurred = paid + case

            rows.append({
                'accident_year': ay,
                'calendar_year': cy,
                'dev_month': dev_month,
                'paid_losses': round(paid, 2),
                'case_reserves': round(case, 2),
                'incd_losses': round(incurred, 2),
                'aa_ibnr': round(ibnr, 2),
                'ultimate_loss': round(ultimate_loss, 2)
            })

    claim_df = pd.DataFrame(rows)
    return claim_df, premium_df


def plot_development_patterns(claim_df, metric='paid_losses', by='accident_year'):
    """Plot development patterns for a specified metric"""

    pivot_df = claim_df.pivot_table(
        index='accident_year', columns='dev_month', values=metric, aggfunc='sum'
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