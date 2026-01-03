import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import the Prediction Pipeline
# (This works because we will run the script from the project root)
from src.causal_uplift_service.pipelines.prediction_pipeline import PredictPipeline

def generate_uplift_report():
    # --- 1. CONFIGURATION ---
    # We assume the script is run from the PROJECT ROOT
    input_csv_path = os.path.join('artifacts', 'customer_data.csv')
    
    # Define where to save the image: src/causal_uplift_service/results/
    results_dir = os.path.join('src', 'causal_uplift_service', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Checking for data at: {input_csv_path}")
    
    # --- 2. LOAD DATA ---
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print("❌ Error: Could not find 'artifacts/customer_data.csv'.")
        print("Please run the Data Ingestion step first.")
        return

    # --- 3. CREATE TEST SET (Simulation) ---
    # We use random_state=42 to replicate the exact split used during training
    print("Splitting data (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Test Data Size: {len(test_df)} records")

    # --- 4. PREDICT SCORES ---
    print("Running Prediction Pipeline on Test Data...")
    pipeline = PredictPipeline()
    
    try:
        # The pipeline handles preprocessing internally
        uplift_scores = pipeline.predict(test_df)
    except Exception as e:
        print(f"⚠️ Pipeline Error: {e}")
        print("Attempting to drop target columns and retry...")
        # Fallback: remove target columns if they exist and confuse the pipeline
        cols_to_drop = ['conversion', 'spend', 'visit']
        X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
        uplift_scores = pipeline.predict(X_test)

    # --- 5. CALCULATE METRICS BY DECILE ---
    print("Calculating Uplift Deciles...")
    results_df = test_df.copy()
    results_df['uplift_score'] = uplift_scores
    
    # Sort High to Low
    results_df = results_df.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)
    
    # Create Deciles (10 groups)
    results_df['decile'] = pd.qcut(results_df.index, 10, labels=False)
    
    metrics = []
    
    # Define Column Names (Ensure these match your CSV)
    col_treatment = 'treatment'   # 1 = Treated, 0 = Control
    col_outcome = 'conversion'    # 1 = Bought, 0 = Not Bought

    for i in range(10):
        # Get customers in this decile
        subset = results_df[results_df['decile'] == i]
        
        # Calculate conversion rates
        conv_treated = subset[subset[col_treatment] == 1][col_outcome].mean()
        conv_control = subset[subset[col_treatment] == 0][col_outcome].mean()
        
        # Calculate Lift
        lift = conv_treated - conv_control
        
        metrics.append({
            'Decile': i + 1,
            'Uplift': lift * 100, # Percentage
            'Size': len(subset)
        })

    metric_df = pd.DataFrame(metrics)

    # --- 6. PLOT AND SAVE ---
    print("Generating Chart...")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Color bars: Green for positive lift, Red for negative
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in metric_df['Uplift']]
    
    sns.barplot(x='Decile', y='Uplift', data=metric_df, palette=colors)
    
    plt.title('Uplift by Decile (Validation on Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Decile (1 = Most Persuadable)', fontsize=12)
    plt.ylabel('Incremental Conversion Lift (%)', fontsize=12)
    plt.axhline(0, color='black', linewidth=1)
    
    # Save to the results folder
    output_path = os.path.join(results_dir, 'uplift_decile_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Chart saved successfully at: {output_path}")

if __name__ == "__main__":
    generate_uplift_report()