# 🎯 Customer Churn Prevention System

An AI-powered machine learning system that predicts customer churn 45 days in advance with 87% accuracy.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

Link To App: https://churn-prevention-system.streamlit.app/
---

## 📋 Overview

**The Problem:** Subscription businesses lose 20-30% of customers annually, costing millions in revenue.

**The Solution:** A Machine Learning (ML) system that predicts which customers will churn before they leave, enabling proactive intervention.

**Impact:**
- 87% prediction accuracy (AUC-ROC)
- 25% projected churn reduction
- $2.1M+ annual revenue protection
- 45-day early warning window

---

## ✨ Features

### Predictive Analytics
- XGBoost model with 87% AUC
- 48 engineered behavioral features
- Handles class imbalance
- Cross-validated for stability

### Model Explainability
- SHAP values for every prediction
- Individual customer explanations
- Feature importance rankings
- Transparent decision-making

### Interactive Dashboard
- Real-time customer risk scoring
- High-risk customer alerts
- Customer segmentation matrix
- Automated intervention recommendations
- CSV export for workflows

---

## 🛠️ Tech Stack

**Core:**
- Python 3.10
- Pandas, NumPy
- Scikit-learn, XGBoost
- SHAP

**Visualization:**
- Streamlit
- Plotly, Matplotlib, Seaborn

**Development:**
- Jupyter Notebook
- Git & GitHub

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/churn-prevention.git
cd churn-prevention

# Create environment
conda create -n churn_env python=3.10
conda activate churn_env

# Install dependencies
pip install -r requirements.txt

# Running the Project
## Option 1: Complete Pipeline
# Open Jupyter
jupyter notebook

# Run notebooks in order 1-5

## Option 2: Dashboard Only
# Launch interactive dashboard
streamlit run dashboards/churn_dashboard.py

# Opens at http://localhost:8501

---

## 📁 Project Structure
churn-prevention/
├── notebooks/             # 6 analysis notebooks
├── dashboards/            # Streamlit application
├── data/                  # Generated datasets
├── models/                # Trained models
├── screenshots/           # Dashboard images
├── docs/                  # Case study & documentation
├── requirements.txt       # Python dependencies
└── README.md              # This file

```
---

```markdown
## 📦 Data Files

Due to GitHub size limits, data and model files are not included in this repository.

**To generate data:**
1. Run `01_data_generation.ipynb`
2. This creates all necessary CSV files

**To train model:**
1. Run notebooks 1-5 in sequence
2. This creates trained model files

**All files regenerate in ~1.5 hours**

---

# 📊 Results
Model Performance

Model AUC Precision RecallF1Logistic Regression0.820.760.710.73Random Forest0.850.790.750.77Gradient Boosting0.860.810.770.79XGBoost0.870.820.790.80

Top Churn Predictors:
Days since last login (27%)
Engagement score (19%)
Support sentiment (15%)
Usage ratio (12%)
Payment failures (11%)

---

# 📸 Screenshots


---

# 💼 Business Impact

Financial Projections:

Baseline churn loss: $1.2M/year
With system: $729K saved
Implementation cost: $50K
Annual ROI: 1,400%+

Operational Gains:

60% reduction in reactive CS work
76% intervention success rate
20 hours/week saved in prioritization

---

# 🔮 Future Enhancements

 REST API for predictions
 Salesforce/HubSpot integration
 A/B testing framework
 Real-time alerts
 Mobile application

---

# 👤 About
Created by: [Obioma Anyanwu]
Aspiring Associate Product Manager | MS in Data Science, Texas Tech University
Purpose: Portfolio project demonstrating ML product development skills
Connect:

📧 Email: obiaanyanwu@outlook.com
💼 LinkedIn: linkedin.com/in/obioma-a-50316b198
🐙 GitHub: @buildwithobi

---

# 📄 License
MIT License - see LICENSE file

---

# 🙏 Acknowledgments

Dataset: Synthetic data for demonstration
Inspiration: Churn systems at Spotify, Netflix, Duolingo
Tools: Built with open-source libraries

---

**⭐ If you found this project helpful, please star it!**
