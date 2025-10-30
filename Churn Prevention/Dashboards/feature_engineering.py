"""
Feature Engineering Module for Churn Prevention Project
This module contains all feature engineering functions used across notebooks and dashboards.
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Apply feature engineering transformations matching the trained model.
    """
    df = df.copy()
    
    # Calculate tenure_days if it doesn't exist
    if 'tenure_days' not in df.columns and 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['tenure_days'] = (pd.Timestamp.now() - df['signup_date']).dt.days
    
    # Engagement score
    df['engagement_score'] = (
        df['logins_30d'] * 0.3 +
        df['features_used'] * 0.3 +
        df['session_duration_avg'] * 0.2 +
        df['power_feature_usage'] * 0.2
    )
    
    # Health score
    df['health_score'] = (
        (df['logins_30d'] / df['logins_30d'].max() * 30) +
        (df['session_duration_avg'] / df['session_duration_avg'].max() * 20) +
        (df['features_used'] / df['features_used'].max() * 20) +
        ((df['ticket_sentiment'] + 1) / 2 * 15) +
        (df['net_promoter_score'] / 10 * 15)
    )
    
    # Activity and usage metrics
    df['activity_recency'] = 1 / (df['days_since_last_login'] + 1)
    df['usage_efficiency'] = df['features_used'] / (df['logins_30d'] + 1)
    df['support_intensity'] = df['support_tickets_30d'] / ((df['tenure_days'] / 30) + 1)
    
    # Interaction features
    df['usage_satisfaction'] = df['features_used'] * df['ticket_sentiment']
    df['login_feature_interaction'] = df['logins_30d'] * df['features_used']

    # Power user and engagement metrics
    df['power_user_score'] = df['logins_30d'] * df['features_used']
    df['engagement_velocity'] = df['features_used'] / (df['tenure_days'] / 30 + 1)
    df['engagement_tenure'] = df['logins_30d'] * df['tenure_days']
    
    # Risk flags
    df['dormancy_risk'] = (df['logins_30d'] < 5).astype(int)
    df['low_engagement_risk'] = (df['features_used'] < 10).astype(int)
    df['support_risk'] = (df['support_tickets_30d'] > 3).astype(int)
    df['payment_risk'] = (df['payment_failures'] > 0).astype(int)
    df['sentiment_risk'] = (df['ticket_sentiment'] < 0.3).astype(int)
    
    # Quality and satisfaction scores
    df['support_quality_score'] = 1 - (df['support_tickets_30d'] / (df['support_tickets_30d'].max() + 1))
    df['satisfaction_index'] = (df['ticket_sentiment'] + 1) / 2 * 0.5 + (df['net_promoter_score'] / 10) * 0.5
    
    # Value and revenue metrics
    tier_value_map = {'free': 10, 'basic': 20, 'premium': 30, 'Free': 10, 'Basic': 20, 'Premium': 30}
    df['value_realization'] = df['features_used'] / df['subscription_tier'].map(tier_value_map).fillna(20)
    
    df['revenue_risk_score'] = df['payment_failures'] * df['days_since_last_login']
    
    tier_price_map = {'free': 0, 'basic': 29, 'premium': 99, 'Free': 0, 'Basic': 29, 'Premium': 99}
    df['estimated_ltv'] = (df['tenure_days'] / 30) * df['subscription_tier'].map(tier_price_map).fillna(29)
    
    # Subscription tier metrics
    df['subscription_tier_numeric'] = df['subscription_tier'].map({
        'free': 0, 'basic': 1, 'premium': 2,
        'Free': 0, 'Basic': 1, 'Premium': 2
    })
    df['subscription_tier_engagement'] = df['subscription_tier_numeric'] * df['features_used']
    
    # Time-based features
    df['tenure_velocity'] = df['logins_30d'] / (df['tenure_days'] / 30 + 1)
    df['recently_active'] = (df['logins_30d'] > 20).astype(int)
    df['stale_account'] = (df['logins_30d'] < 3).astype(int)
    
    # High risk flag
    df['high_risk_flag'] = (
        (df['days_since_last_login'] > 14) |
        (df['logins_30d'] < 5) |
        (df['support_tickets_30d'] > 3) |
        (df['payment_failures'] > 0)
    ).astype(int)
    
    # Encode categorical variables
    df['subscription_tier_encoded'] = df['subscription_tier'].str.lower().map({
        'free': 0, 'basic': 1, 'premium': 2
    })
    
    df['size_encoded'] = df['company_size'].map({
        '1-10': 0, '11-50': 1, '51-200': 2, '200+': 3
    })
    
    df['industry_encoded'] = df['industry'].str.lower().map({
        'tech': 0, 'finance': 1, 'healthcare': 2, 'retail': 3, 'other': 4
    })
    
    # Create lifecycle stage if not present
    if 'lifecycle_stage' not in df.columns:
        def assign_lifecycle(row):
            if row['tenure_days'] < 90:
                return 'Onboarding'
            elif row['days_since_last_login'] > 30:
                return 'At Risk'
            elif row['logins_30d'] > 15:
                return 'Active'
            else:
                return 'At Risk'
        df['lifecycle_stage'] = df.apply(assign_lifecycle, axis=1)
    
    df['lifecycle_encoded'] = df['lifecycle_stage'].map({
        'Onboarding': 0, 'Active': 1, 'At Risk': 2, 'Churned': 3
    })
    
    # NPS category
    if 'net_promoter_score_category' not in df.columns:
        df['net_promoter_score_category'] = pd.cut(
            df['net_promoter_score'],
            bins=[-1, 6, 8, 10],
            labels=['Detractor', 'Passive', 'Promoter']
        )
    
    df['net_promoter_score_category_encoded'] = df['net_promoter_score_category'].map({
        'Detractor': 0, 'Passive': 1, 'Promoter': 2
    })
    
    # Usage category
    if 'usage_category' not in df.columns:
        df['usage_category'] = pd.cut(
            df['features_used'],
            bins=[0, 5, 15, 100],
            labels=['Low', 'Medium', 'High']
        )
    
    df['usage_category_encoded'] = df['usage_category'].map({
        'Low': 0, 'Medium': 1, 'High': 2
    })
    
    return df
