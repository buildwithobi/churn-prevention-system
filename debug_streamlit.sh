#!/bin/bash
echo "🔍 STREAMLIT TROUBLESHOOTING"
echo "================================"
echo ""

# 1. Check current location
echo "📍 Current Directory:"
pwd
echo ""

# 2. Check if in project root
if [[ $(basename "$PWD") == "Churn Prevention" ]]; then
    echo "✅ You are in project root (correct)"
else
    echo "❌ You are NOT in project root"
    echo "   Navigate to: cd ~/Documents/AMP\ Projects/Churn\ Prevention"
fi
echo ""

# 3. Check required data files
echo "📊 Checking Data Files:"
if [ -f "Datasets/customer_churn_data.csv" ]; then
    size=$(ls -lh Datasets/customer_churn_data.csv | awk '{print $5}')
    echo "✅ customer_churn_data.csv exists ($size)"
else
    echo "❌ customer_churn_data.csv MISSING"
    echo "   Solution: Run Notebook 1"
fi

if [ -f "Datasets/customer_churn_engineered.csv" ]; then
    size=$(ls -lh Datasets/customer_churn_engineered.csv | awk '{print $5}')
    echo "✅ customer_churn_engineered.csv exists ($size)"
else
    echo "❌ customer_churn_engineered.csv MISSING"
    echo "   Solution: Run Notebook 3"
fi
echo ""

# 4. Check required model files
echo "🤖 Checking Model Files:"
if [ -f "ML Models/churn_model.pkl" ]; then
    size=$(ls -lh ML Models/churn_model.pkl | awk '{print $5}')
    echo "✅ churn_model.pkl exists ($size)"
else
    echo "❌ churn_model.pkl MISSING"
    echo "   Solution: Run Notebook 4"
fi

if [ -f "ML Models/scaler.pkl" ]; then
    echo "✅ scaler.pkl exists"
else
    echo "❌ scaler.pkl MISSING"
    echo "   Solution: Run Notebook 4"
fi

if [ -f "ML Models/model_info.pkl" ]; then
    echo "✅ model_info.pkl exists"
else
    echo "❌ model_info.pkl MISSING"
    echo "   Solution: Run Notebook 4"
fi
echo ""

# 5. Check dashboard file
echo "📱 Checking Dashboard:"
if [ -f "Dashboards/churn_dashboard.py" ]; then
    echo "✅ churn_dashboard.py exists"
else
    echo "❌ churn_dashboard.py MISSING"
    echo "   Solution: Create dashboard file"
fi
echo ""

# 6. Summary
echo "================================"
echo "📋 SUMMARY:"
echo ""
all_files_exist=true

if [ ! -f "Datasets/customer_churn_data.csv" ] || [ ! -f "Datasets/customer_churn_engineered.csv" ]; then
    echo "❌ Missing data files - Run Notebooks 1 and 3"
    all_files_exist=false
fi

if [ ! -f "ML Models/churn_model.pkl" ] || [ ! -f "ML\ Models/scaler.pkl" ]; then
    echo "❌ Missing model files - Run Notebook 4"
    all_files_exist=false
fi

if [ "$all_files_exist" = true ]; then
    echo "✅ All files present!"
    echo "   Try: streamlit run Dashboards/churn_dashboard.py"
fi

echo ""
echo "================================"