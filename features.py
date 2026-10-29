import numpy as np
import pandas as pd


def add_engineered_features(df, avg_monthly_charge):
    """Create the engineered features exactly as in the training notebook.

    Input : DataFrame with the original (cleaned, binary-encoded) columns.
    Output: a copy with 5 extra columns.
    """
    out = df.copy()

    out['tenure_group'] = pd.cut(
        out['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12m', '13-24m', '25-48m', '49-72m']
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        out['charges_ratio'] = np.where(
            out['TotalCharges'] > 0,
            out['MonthlyCharges'] / out['TotalCharges'],
            1.0
        )

    out['num_services'] = out[['PhoneService', 'PaperlessBilling',
                               'Partner', 'Dependents']].sum(axis=1)

    out['is_high_risk'] = (
        (out['Contract'] == 'Month-to-month') & (out['tenure'] <= 12)
    ).astype(int)

    out['monthly_charge_above_avg'] = out['MonthlyCharges'] - avg_monthly_charge

    return out
