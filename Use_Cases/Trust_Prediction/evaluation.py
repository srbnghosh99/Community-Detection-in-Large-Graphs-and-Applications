from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import os
from os.path import dirname, join as pjoin


def evaluation(inputdir):
    csv_files = [file for file in os.listdir(inputdir) if file.endswith('.csv')]
    for col_idx, file in enumerate(csv_files):
        # filepath =  os.listdir('Spectral') + file
        # filepath = os.path.dirname(file)
        # print(filepath)
        df = pd.read_csv(inputdir+file)
        # Define a range of threshold values to test
        thresholds = np.arange(0, 1, 0.15)  # From 0 to 1 with a step of 0.15

        # List to store results
        results = []

        # Iterate over each threshold value
        for threshold in thresholds:
            print('Threshold:', threshold)
            
            # Apply the threshold to get predicted values
            # df['Predicted_TrustValue'] = df['Predicted_score'].apply(lambda avg: 1 if avg > threshold else 0)
            df['Predicted_TrustValue'] = df['score'].apply(lambda avg: 1 if avg > threshold else 0)
            
            # # Convert columns to lists
            ground_truth_common = df['TrustValue'].tolist()
            predicted_values_common = df['Predicted_TrustValue'].tolist()
            # ground_truth_common = df['ground_truth'].tolist()
            report = classification_report(ground_truth_common, predicted_values_common, labels=[0,1])
            # print(cmeasure)
            print(report)
            auc = roc_auc_score(ground_truth_common, predicted_values_common)
            print("AUC:", auc)
            
            # Calculate metrics
            precision = precision_score(ground_truth_common, predicted_values_common)
            recall = recall_score(ground_truth_common, predicted_values_common)
            f1 = f1_score(ground_truth_common, predicted_values_common)
            fpr, tpr, _ = roc_curve(ground_truth_common, predicted_values_common)
            # roc_auc = auc(fpr, tpr)

            # Print metrics
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            # print(f'AUC: {roc_auc:.4f}')
            
            # Append results to the list
            results.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F1 Score': f1,'AUC Score': auc})

        # Create a DataFrame from results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['AUC Score'])

        # Save results to CSV file
        
        file_name = os.path.splitext(file)[0]
        file_name = inputdir +file_name+'_threshold_evaluation_results.csv'
        results_df.to_csv(file_name, index=False)

        print('Results saved to threshold_evaluation_results.csv')


def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--inputdir",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    curr_directory = os.getcwd()
    inputdir = pjoin(curr_directory,inputs.dataset, inputs.inputdir)
    evaluation(inputdir)
  
if __name__ == '__main__':
    main()
