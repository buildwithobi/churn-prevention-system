"""
Feature Engineering Module for Churn Prevention Project
This module contains all feature engineering functions used across notebooks and dashboards.
"""

import pandas as pd


def engineer_features(df):
    """
    Apply feature engineering transformations to the input dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw customer data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    df = df.copy()

    # Create tenure_months from tenure_days
    df['tenure_months'] = df['tenure_days'] / 30

    # Power user score
    df['power_user_score'] = df['logins_30d'] * df['features_used']

    # Engagement velocity
    df['engagement_velocity'] = df['features_used'] / (df['tenure_months'] + 1)
    
    # Risk scores
    df['dormancy_risk'] = (df['logins_30d'] < 5).astype(int)
    df['low_engagement_risk'] = (df['features_used'] < 10).astype(int)
    df['support_risk'] = (df['support_tickets_30d'] > 3).astype(int)
    df['payment_risk'] = (df['payment_failures'] > 0).astype(int)
    df['sentiment_risk'] = (df['ticket_sentiment'] < 0.3).astype(int)
    
    # Quality scores
    df['support_quality_score'] = 1 - (df['support_tickets_30d'] / (df['support_tickets_30d'].max() + 1))
    df['satisfaction_index'] = (df['ticket_sentiment'] + df['net_promoter_score'] / 10) / 2
    df['value_realization'] = df['features_used'] / (df['subscription_tier'].map({'Basic': 10, 'Standard': 20, 'Premium': 30}).fillna(20))
    
    # Revenue and engagement
    df['revenue_risk_score'] = df['payment_failures'] * df['monthly_reoccuring_revenue']
    df['estimated_ltv'] = df['tenure_months'] * df['subscription_tier'].map({'Basic': 29, 'Standard': 49, 'Premium': 99}).fillna(49)
    df['engagement_tenure'] = df['logins_30d'] * df['tenure_months']
   
    # Additional useful features
    df['recency_score'] =  1 / (df['days_since_last_login'] + 1)
    df['usage_efficiency'] = df['features_used'] / (df['session_duration_avg'] + 1)
    df['engagement_ratio'] = df['usage_vs_plan'] * df['logins_30d']
   
    return df
