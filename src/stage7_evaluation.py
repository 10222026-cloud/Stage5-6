import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_all_methods(results_dict):
    """
    Stage 7: Evaluation and Benchmarking
    Takes a dictionary of results from all methods and generates a comparison 
    table and a financial cost-saving bar chart.
    """
    print("\n" + "="*50)
    print("STAGE 7: FINAL EVALUATION AND BENCHMARKING")
    print("="*50)
    
    # 1. Create the Metric Table
    # results_dict should look like: {'SMOTE': {'F1': 0.85, 'Savings': 12000000}, ...}
    df = pd.DataFrame(results_dict).T
    
    print("\n--- Metric Comparison Table ---")
    print(df.to_string())
    
    # Save the table to a text file for the report
    os.makedirs('results', exist_ok=True)
    with open('results/results_summary.txt', 'w') as f:
        f.write("Stage 7 Metric Comparison Table\n")
        f.write("="*35 + "\n")
        f.write(df.to_string())
    print("\n[+] Table saved to: results/results_summary.txt")
    
    # 2. Create the ML Metric Comparison Bar Chart
    # Plot Accuracy, Recall, Precision, and ROC-AUC side-by-side
    metrics_to_plot = ['Accuracy', 'Recall', 'Precision', 'ROC-AUC']
    plot_cols = [col for col in metrics_to_plot if col in df.columns]
    
    if plot_cols:
        # Create a grouped bar chart using pandas
        ax = df[plot_cols].plot(kind='bar', figsize=(12, 6), width=0.8)
        
        plt.title('Machine Learning Metrics Comparison by Method', fontsize=14, fontweight='bold')
        plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
        plt.xlabel('Machine Learning Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.15) # Leave space at the top for the legend
        plt.legend(loc='lower center', ncol=4)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the graph
        save_path = 'results/metric_comparison_chart.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[+] Metric comparison chart saved to: {save_path}")

    return df