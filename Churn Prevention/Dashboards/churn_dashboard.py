### Streamlit Dashboard for Churn Prediction And Prevention System
### Save this file as: dashboards/churn_dashboard.py
### Run with: streamlit run dashboards/churn_dashboard.py

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
from feature_engineering import engineer_features

### Page configuration
st.set_page_config(
    page_title = "Churn Prevention Dashboard",
    page_icon = "🎯",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

### Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ddabce;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #2d2d30;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

### =============================================================================
### LOAD DATA AND MODELS
### =============================================================================

@st.cache_data
def load_data():
    """Load customer data"""
    try:
        # Get path to "Churn Prevention" folder
        script_dir = Path(__file__).parent.parent.resolve()
        data_path = script_dir / 'Datasets' / 'customer_churn_data.csv'
        
        df = pd.read_csv(data_path)
        return df
        
    except FileNotFoundError:
        st.error(f"""
        **Data file not found!**
        
        Looking for: `{data_path}`
        
        Please ensure the file exists at the correct location.
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


@st.cache_resource
def load_models():
    """Load trained model and scaler"""
    try:
        # Get path to "Churn Prevention" folder
        script_dir = Path(__file__).parent.parent.resolve()
        
        model = joblib.load(script_dir / 'ML Models' / 'churn_model.pkl')
        scaler = joblib.load(script_dir / 'ML Models' / 'scaler.pkl')
        model_info = joblib.load(script_dir / 'ML Models' / 'model_info.pkl')
        
        return model, scaler, model_info
        
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.error("Please run Notebooks 1 and 2 first to generate data and train models!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

### Load Data
### Load Data
try:
    df = load_data()
    model, scaler, model_info = load_models()
    feature_cols = model_info['feature_columns']
    
    # DEBUG: Show what features the model expects
    st.write("### 🔍 DEBUG: Model Feature Analysis")
    st.write("**Model expects these features:**")
    st.write(feature_cols)
    st.write(f"Total features expected: {len(feature_cols)}")
    
    ### Apply feature engineering
    df_processed = engineer_features(df)
    
    # DEBUG: Show what features we created
    st.write("\n**Features we created:**")
    st.write(df_processed.columns.tolist())
    st.write(f"Total features created: {len(df_processed.columns)}")
    
    # DEBUG: Show missing features
    missing_features = set(feature_cols) - set(df_processed.columns)
    if missing_features:
        st.error(f"❌ Missing {len(missing_features)} features:")
        st.write(sorted(list(missing_features)))
        st.stop()
    
    ### Make predictions for all customers
    X = df_processed[feature_cols]
    churn_probabilities = model.predict_proba(X)[:, 1]
    df_processed['churn_risk_score'] = churn_probabilities * 100  # Probability as percentage
    df_processed['churn_prediction'] = (churn_probabilities > 0.5).astype(int)  # Binary prediction
    
    # Use churn_risk_score throughout the dashboard (not 'churned')
    df_processed['churned'] = df_processed['churn_risk_score']  # For backward compatibility

except Exception as e:
    st.error(f"❌ Error occurred: {type(e).__name__}: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

### =============================================================================
### HEADER
### =============================================================================

st.markdown('<div class="main-header">🎯 Customer Churn Prevention Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown("### AI-Powered Early Warning System for Customer Retention")
st.markdown("---")

### =============================================================================
### SIDEBAR - FILTERS
### =============================================================================

st.sidebar.header("🔧 Filters & Settings")

### Risk threshold
risk_threshold = st.sidebar.slider(
    "Churn Risk Threshold (%)",
    min_value = 0,
    max_value = 100,
    value = 70,
    help = "Customers above this threshold are flagged as high-risk"
)

### Subscription Tier Filter
subscription_tier_filter = st.sidebar.multiselect(
    "Subscription Tier",
    options = df['subscription_tier'].unique(),
    default = df['subscription_tier'].unique()
)

### Company size filter
company_size_filter = st.sidebar.multiselect(
    "Company Size",
    options = df['company_size'].unique(),
    default = df['company_size'].unique()
)

### Apply filters
filtered_df = df_processed[
    (df_processed['subscription_tier'].isin(subscription_tier_filter)) &
    (df_processed['company_size'].isin(company_size_filter))
]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} customers")

### =============================================================================
### KEY METRICS
### =============================================================================

st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total_customers = len(filtered_df)
at_risk_customers = (filtered_df['churned'] >= risk_threshold).sum()
at_risk_pct = (at_risk_customers / total_customers * 100) if total_customers > 0 else 0
average_health = filtered_df['health_score'].mean()
total_monthly_reoccuring_revenue = filtered_df['monthly_reoccuring_revenue'].sum()
monthly_reoccuring_revenue_at_risk = filtered_df[filtered_df['churned'] >= risk_threshold]['monthly_reoccuring_revenue'].sum()

with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("At-Risk Customers", f"{at_risk_customers:,}", 
              f"{at_risk_pct:.1f}%",
              delta_color="inverse")

with col3:
    st.metric("Average Health Score", f"{average_health:.1f}/100") if average_health > 0 else "N/A"

with col4:
    st.metric("Total Monthly Reoccuring Revenue", f"${total_monthly_reoccuring_revenue:,.0f}")

with col5:
    st.metric("Monthly Reoccuring Revenue at Risk", f"${monthly_reoccuring_revenue_at_risk:,.0f}",
              f"{monthly_reoccuring_revenue_at_risk/total_monthly_reoccuring_revenue*100:.1f}%" if total_monthly_reoccuring_revenue > 0 else "0%",
              delta_color="inverse")

st.markdown("---")

### =============================================================================
### HIGH-RISK CUSTOMERS TABLE
### =============================================================================

st.markdown("## 🚨 High-Risk Customers Requiring Immediate Attention")

high_risk_df = filtered_df[filtered_df['churned'] >= risk_threshold].copy()
high_risk_df = high_risk_df.sort_values('churned', ascending=False)

if len(high_risk_df) > 0:
    ### Prepare display columns
    # display_df = high_risk_df[[
    display_cols = ['customer_id', 'subscription_tier', 'monthly_reoccuring_revenue', 'churned',
                    'health_score', 'days_since_last_login', 'support_tickets_30d',
                    'logins_30d', 'payment_failures']
    display_df = high_risk_df[display_cols].head(20).copy()
    
    ### Format columns
    display_df['churned'] = display_df['churned'].round(1)
    display_df['health_score'] = display_df['health_score'].round(1)
    display_df['monthly_reoccuring_revenue'] = display_df['monthly_reoccuring_revenue'].apply(lambda x: f'${x:.0f}')
    
    ### Rename for display
    display_df.columns = [
        'Customer ID', 'Subscription Tier', 'Monthly Reocurring Revenue', 'Risk Score', 'Health Score',
        'Days Inactive', 'Support Tickets', 'Logins', 'Payment Fails'
    ]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    ### Download button
    csv = high_risk_df.to_csv(index=False)
    st.download_button(
        label = "📥 Download High-Risk Customer List (CSV)",
        data = csv,
        file_name = f"high_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
        mime = "text/csv"
    )
else:
    st.success("✅ No high-risk customers at current threshold!")

st.markdown("---")

### =============================================================================
### RISK DISTRIBUTION CHARTS
### =============================================================================

st.markdown("## 📈 Churn Risk Analysis")

col1, col2 = st.columns(2)

with col1:
    ### Risk score distribution
    fig = px.histogram(
        filtered_df,
        x = 'churned',
        nbins = 50,
        title = "Distribution of Churn Risk Scores",
        labels = {'churned': 'Churn Risk Score (%)'},
        color_discrete_sequence = ['#1f77b4']
    )
    fig.add_vline(x = risk_threshold, line_dash = "dash", line_color = "red",
                 annotation_text = "Risk Threshold",
                 annotation_position = "top right")
    fig.update_layout(showlegend = False)
    st.plotly_chart(fig, use_container_width = True)

with col2:
    ### Risk by subscription tier
    risk_by_subscription_tier = filtered_df.groupby('subscription_tier').agg({
        'churned': 'mean',
        'customer_id': 'count'
    }).reset_index()
    risk_by_subscription_tier.columns = ['Subscription Tier', 'Average Risk Score', 'Customer Count']
    
    fig = px.bar(
        risk_by_subscription_tier,
        x = 'Subscription Tier',
        y = 'Average Risk Score',
        title = "Average Risk Score by Subscription Tier",
        text = 'Customer Count',
        color = 'Average Risk Score',
        color_continuous_scale = 'RdYlGn_r'
    )
    fig.update_traces(textposition = 'outside')
    st.plotly_chart(fig, use_container_width = True)

st.markdown("---")

### =============================================================================
### CUSTOMER SEGMENTATION
### =============================================================================

st.markdown("## 🎯 Customer Segmentation: Health vs Risk")

# Check if health_score exists, otherwise use alternative metric
if 'health_score' in filtered_df.columns:
    x_col = 'health_score'
    x_label = 'Health Score (0-100)'
else:
    # Create simple health proxy from available metrics
    filtered_df['simple_health'] = (
        (filtered_df['logins_30d'] / filtered_df['logins_30d'].max() * 50) +
        (filtered_df['features_used'] / filtered_df['features_used'].max() * 50)
    )
    x_col = 'simple_health'
    x_label = 'Engagement Score (0-100)'

### Create segmentation scatter plot
fig = px.scatter(
    filtered_df,
    x = x_col,
    y = 'churned',
    size = 'monthly_reoccuring_revenue',
    color = 'subscription_tier',
    hover_data = ['customer_id', 'logins_30d', 'support_tickets_30d', 'days_since_last_login'],
    title = "Customer Health Score vs Churn Risk (bubble size = Monthly Reoccuring Revenue)",
    labels = {x_col : x_label, 'churned': 'Churn Risk Score (%)'},
    color_discrete_sequence = px.colors.qualitative.Set2
)

### Add quadrant lines
fig.add_hline(y = risk_threshold, line_dash = "dash", line_color = "red", opacity = 0.5)
fig.add_vline(x = 50, line_dash = "dash", line_color = "blue", opacity = 0.5)

### Add quadrant annotations
fig.add_annotation(x = 75, y = 90, text = "High Risk,<br>High Engagement<br>(Unexpected)", 
                  showarrow = False, bgcolor = "rgba(255,200,200,0.8)", borderpad = 4)
fig.add_annotation(x = 25, y = 90, text = "High Risk,<br>Low Engagement<br>(Critical)", 
                  showarrow = False, bgcolor = "rgba(255,150,150,0.8)", borderpad = 4)
fig.add_annotation(x = 75, y = 20, text = "Low Risk,<br>High Engagement<br>(Healthy)", 
                  showarrow=False, bgcolor = "rgba(200,255,200,0.8)", borderpad = 4)
fig.add_annotation(x = 25, y = 20, text = "Low Risk,<br>Low Engagement<br>(Dormant)", 
                  showarrow = False, bgcolor = "rgba(255,255,200,0.8)", borderpad = 4)

st.plotly_chart(fig, use_container_width = True)

st.markdown("---")

### =============================================================================
### FEATURE IMPORTANCE
### =============================================================================

st.markdown("## 🔍 Key Churn Indicators")

col1, col2 = st.columns([2, 1])

with col1:
    ### Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending = False).head(10)
        
        fig = px.bar(
            feature_importance,
            y = 'Feature',
            x = 'Importance',
            orientation = 'h',
            title = "Top 10 Features Predicting Churn",
            labels = {'Importance': 'Importance Score', 'Feature': ''},
            color = 'Importance',
            color_continuous_scale = 'Viridis'
        )
        fig.update_layout(showlegend = False, yaxis = {'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width = True)

with col2:
    st.markdown("### Key Insights")
    st.markdown("""
    **High Risk Indicators:**
    - 🔴 Days since last login > 14
    - 🔴 Logins < 5 per month
    - 🔴 Support tickets > 3
    - 🔴 Payment failures > 0
    - 🔴 Low engagement score
    - 🔴 Poor sentiment scores
    
    **Protective Factors:**
    - 🟢 Regular feature usage
    - 🟢 High NPS score
    - 🟢 Premium tier
    - 🟢 Long tenure
    """)

st.markdown("---")

### =============================================================================
### INDIVIDUAL CUSTOMER ANALYSIS
### =============================================================================

st.markdown("## 🔎 Individual Customer Deep Dive")

### Customer selector
customer_list = filtered_df.sort_values('churned', ascending=False)['customer_id'].tolist()
selected_customer = st.selectbox(
    "Select Customer for Detailed Analysis",
    options=customer_list,
    format_func=lambda x: f"{x} (Risk: {filtered_df[filtered_df['customer_id']==x]['churned'].values[0]:.1f}%)"
)

if selected_customer:
    customer_data = filtered_df[filtered_df['customer_id'] == selected_customer].iloc[0]
    
    ### Customer overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_color = "🔴" if customer_data['churned'] >= risk_threshold else "🟢"
        st.metric(f"{risk_color} Churn Risk", f"{customer_data['churned']:.1f}%")
        st.metric("Health Score", f"{customer_data['health_score']:.1f}/100")
    
    with col2:
        st.metric("Subscription Tier", customer_data['subscription_tier'].title())
        st.metric("Monthly Reoccuring Revenue", f"${customer_data['monthly_reoccuring_revenue']:.0f}")
    
    with col3:
        st.metric("Tenure Days", f"{customer_data['tenure_days']} days")
        st.metric("Last Login", f"{customer_data['days_since_last_login']} days ago")
    
    with col4:
        st.metric("Logins (30 Days)", f"{customer_data['logins_30d']}")
        st.metric("Support Tickets (30 Days)", f"{customer_data['support_tickets_30d']}")
    
    ### Customer profile radar chart
    st.markdown("### Customer Profile Analysis")
    
    profile_metrics = {
        'Engagement Score': min(customer_data['engagement_score'] / 50, 1),
        'Usage': customer_data['usage_vs_plan'],
        'Ticket Sentiment': (customer_data['ticket_sentiment'] + 1) / 2,
        'Activity': max(0, 1 - customer_data['days_since_last_login'] / 30),
        'Net Promoter Score': customer_data['net_promoter_score'] / 10
    }
    
    fig = go.Figure(data=go.Scatterpolar(
        r=list(profile_metrics.values()),
        theta=list(profile_metrics.keys()),
        fill='toself',
        line_color='rgb(31, 119, 180)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title="Customer Health Radar (0-1 scale)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    ### Intervention recommendations
    st.markdown("### 💡 Recommended Interventions")
    
    interventions = []
    priority_count = {'🔴 CRITICAL': 0, '🟠 HIGH': 0, '🟡 MEDIUM': 0}
    
    ### Critical interventions
    if customer_data['payment_failures'] > 0:
        interventions.append(("🔴 CRITICAL", "Billing Issue Resolution", 
                            "Immediate outreach to resolve payment failure within 24 hours"))
        priority_count['🔴 CRITICAL'] += 1
    
    if customer_data['days_since_last_login'] > 30:
        interventions.append(("🔴 CRITICAL", "Dormancy Re-engagement", 
                            "Multi-touch campaign to reactivate account immediately"))
        priority_count['🔴 CRITICAL'] += 1
    
    ### High priority interventions
    if customer_data['support_tickets_30d'] > 3:
        interventions.append(("🟠 HIGH", "Customer Success Check-in", 
                            "Schedule call within 3 days to address pain points"))
        priority_count['🟠 HIGH'] += 1
    
    if customer_data['logins_30d'] < 5 and customer_data['tenure_days'] < 90:
        interventions.append(("🟠 HIGH", "Onboarding Enhancement", 
                            "Provide personalized onboarding session within 1 week"))
        priority_count['🟠 HIGH'] += 1
    
    if customer_data['churned'] > 75 and customer_data['monthly_reoccuring_revenue'] > 50:
        interventions.append(("🟠 HIGH", "Retention Offer", 
                            "Consider special pricing or feature upgrade within 2 days"))
        priority_count['🟠 HIGH'] += 1
    
    ### Medium priority interventions
    if customer_data['usage_vs_plan'] < 0.4:
        interventions.append(("🟡 MEDIUM", "Feature Adoption Program", 
                            "14-day guided tour of underutilized features"))
        priority_count['🟡 MEDIUM'] += 1
    
    if customer_data['features_used'] < 5:
        interventions.append(("🟡 MEDIUM", "Product Education", 
                            "Share tutorials and best practices via email"))
        priority_count['🟡 MEDIUM'] += 1
    
    if interventions:
        for priority, action, description in interventions:
            st.markdown(f"**{priority} {action}**")
            st.markdown(f"→ {description}")
            st.markdown("")
    else:
        st.success("✅ Customer is healthy - no immediate interventions needed. Continue monitoring.")
    
    ### Summary
    if priority_count['🔴 CRITICAL'] > 0 or priority_count['🟠 HIGH'] > 0:
        st.error(f"⚠️ Action Required: {priority_count['🔴 CRITICAL']} critical + {priority_count['🟠 HIGH']} high priority interventions")
    else:
        st.info(f"ℹ️ {priority_count['🟡 MEDIUM']} medium priority suggestions for optimization")

st.markdown("---")

### =============================================================================
### MODEL PERFORMANCE
### =============================================================================

st.markdown("## 📊 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", model_info['model_type'])
    st.metric("AUC Score", f"{model_info['auc_score']:.4f}")

with col2:
    total_predicted_churn = df_processed['churn_prediction'].sum()
    actual_churn = df_processed['churned'].sum()
    st.metric("Customers Flagged", f"{total_predicted_churn:,}")
    st.metric("Actual Churners", f"{actual_churn:,}")

with col3:
    if actual_churn > 0:
        detection_rate = (df_processed['churn_prediction'] & df_processed['churned']).sum() / actual_churn
        st.metric("Detection Rate", f"{detection_rate*100:.1f}%")
    st.metric("Total Features", len(feature_cols))

### Model update info
st.info("""
📅 **Model Training Date:** September 2024  
🔄 **Recommended Retraining:** Every 30 days  
📈 **Next Update:** Monitor for data drift and performance degradation
""")

st.markdown("---")

### =============================================================================
### EXPORT & ACTIONS
### =============================================================================

st.markdown("## 📤 Export & Actions")

col1, col2, col3 = st.columns(3)

with col1:
    ### Export all predictions
    export_df = filtered_df[['customer_id', 'subscription_tier', 'monthly_reoccuring_revenue', 
                             'churned', 'health_score', 'churned']]
    csv_all = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Export All Customer Scores",
        data=csv_all,
        file_name=f"all_customer_scores_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    ### Export high-risk only
    if len(high_risk_df) > 0:
        csv_high_risk = high_risk_df.to_csv(index=False)
        st.download_button(
            label="⚠️ Export High-Risk Customers",
            data=csv_high_risk,
            file_name=f"high_risk_only_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    ### Export intervention list
    if len(high_risk_df) > 0:
        intervention_df = high_risk_df[['customer_id', 'subscription_tier', 'monthly_reoccuring_revenue', 
                                       'churned', 'days_since_last_login',
                                       'support_tickets_30d']].head(50)
        intervention_df['priority'] = intervention_df['churned'].apply(
            lambda x: 'CRITICAL' if x > 85 else 'HIGH' if x > 70 else 'MEDIUM'
        )
        csv_interventions = intervention_df.to_csv(index=False)
        st.download_button(
            label="🎯 Export Intervention Queue",
            data=csv_interventions,
            file_name=f"intervention_queue_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")

### =============================================================================
### FOOTER
### =============================================================================

st.markdown("""
---
### 📖 About This Dashboard

This AI-powered churn prevention system uses machine learning to predict customer churn risk 
45-60 days in advance, enabling proactive intervention strategies.

**Key Features:**
- Real-time risk scoring for all customers
- Automated intervention recommendations
- Customer segmentation and health monitoring
- Exportable action lists for customer success teams

**Built with:** Python, Scikit-learn, XGBoost, Streamlit, Plotly

💡 **Tip:** Adjust the risk threshold slider to see how it affects the number of flagged customers.
""")

st.markdown("---")
st.markdown("*Dashboard last updated: {}*".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


print(df.columns)
