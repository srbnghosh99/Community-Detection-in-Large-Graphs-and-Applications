import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import seaborn as sns
import os



def plotdist(file):

    df = pd.read_csv(file)
    prediction_scores = df['score'].tolist()
    # plt.plot(prediction_scores, bins=10, color='skyblue', edgecolor='black')

    # # Sort the prediction scores
    # sorted_scores = sorted(prediction_scores)

    # # Calculate the cumulative distribution function (CDF)
    # cdf = np.linspace(0, 1, len(sorted_scores))

    # # Plot the curve
    # plt.plot(sorted_scores, cdf, marker='o', linestyle='-')
    # plt.xlabel('Prediction Score')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Prediction Scores')
    # plt.grid(True)
    # plt.show()

    # Extract predicted values and ground truth values from DataFrame
    prediction_values = df['predicted_value'].tolist()
    ground_truth_values = df['ground_truth'].tolist()  # Assuming 'ground_truth' is the column name in your DataFrame

    # Sort the predicted values
    sorted_scores = sorted(prediction_values)

    # Calculate the cumulative distribution function (CDF) for predicted values
    cdf_predicted = np.linspace(0, 1, len(sorted_scores))

    # Sort the ground truth values
    sorted_ground_truth = sorted(ground_truth_values)

    # Calculate the cumulative distribution function (CDF) for ground truth values
    cdf_ground_truth = np.linspace(0, 1, len(sorted_ground_truth))

    # Plot the predicted values and ground truth values
    plt.plot(sorted_scores, cdf_predicted, marker='o', linestyle=' ', label='Predicted')
    plt.plot(sorted_ground_truth, cdf_ground_truth, marker='o', linestyle=' ', label='Ground Truth')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Distribution of Predicted and Ground Truth Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    '''

    # Sort the predicted values
    sorted_scores = sorted(prediction_scores)

    # Calculate the cumulative distribution function (CDF) for predicted values
    cdf_predicted = np.linspace(0, 1, len(sorted_scores))

    # Sort the ground truth values
    sorted_ground_truth = sorted(ground_truth_values)

    # Calculate the cumulative distribution function (CDF) for ground truth values
    cdf_ground_truth = np.linspace(0, 1, len(sorted_ground_truth))

    # Plot the predicted values and ground truth values
    plt.plot(sorted_scores, cdf_predicted, marker='o', linestyle='-', color='blue', label='Predicted')
    plt.plot(sorted_ground_truth, cdf_ground_truth, marker='s', linestyle='-', color='red', label='Ground Truth')

    # Plot markers where predicted values match ground truth values
    for pred_value in prediction_scores:
        if pred_value in ground_truth_values:
            idx = ground_truth_values.index(pred_value)
            plt.plot(pred_value, cdf_ground_truth[idx], marker='o', markersize=8, color='green')

    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Distribution of Predicted and Ground Truth Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

def prediction_scores_plot(main_folder_path):
    # List all subdirectories in the main folder
    # Predefined list of subfolder names in the desired order
    predefined_subfolder_order = ['Louvain', 'Label Propagation', 'Spectral', 'Ego Splitting']

    # Keep only the subfolders that exist in the predefined order
    subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir() and f.name in predefined_subfolder_order]

    # Sort the subfolders according to the predefined list
    subfolders = sorted(subfolders, key=lambda x: predefined_subfolder_order.index(os.path.basename(x)))

    # subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]

    # Number of rows will be equal to the number of subfolders
    n_rows = len(subfolders)

    # Create a figure with n_rows subplots, and 5 plots per row
    fig, axs = plt.subplots(n_rows, 5, figsize=(15, 6 * n_rows))

    # Loop through each subfolder and plot the CSVs in a row
    predefined_file_order = ['Betweenness.csv', 'MaxDegree.csv', 'MaxTrustor.csv', 'MaxTrustee.csv','Random.csv']
    for row_idx, subfolder in enumerate(subfolders):
        # csv_files = [file for file in os.listdir(subfolder) if file.endswith('.csv')]
        csv_files = [file for file in predefined_file_order if file in os.listdir(subfolder) and file.endswith('.csv')]

        # Sort and take the first 5 CSV files
        csv_files = csv_files[:5]

        # Plot CSVs in the current row (row_idx)
        for col_idx, file in enumerate(csv_files):
            # Read the CSV file
            df = pd.read_csv(os.path.join(subfolder, file))

            # Split the data based on TrustValue (1 or 0)
            df_gt_1 = df[df['TrustValue'] == 1]
            df_gt_0 = df[df['TrustValue'] == 0]

            # Plot KDEs for ground truth = 1 and ground truth = 0
            sns.kdeplot(df_gt_1['Predicted_score'], fill=True, color='blue', legend=False,  ax=axs[row_idx, col_idx])
            sns.kdeplot(df_gt_0['Predicted_score'], fill=True, color='red', legend=False, ax=axs[row_idx, col_idx])

            # Use file name as plot title (without the ".csv" extension)
            file_name = os.path.splitext(file)[0]
            axs[row_idx, col_idx].set_title(f'{file_name}', fontsize=7)

            # Set labels for individual plots
            axs[row_idx, col_idx].set_xlabel('', fontsize=10)
            axs[row_idx, col_idx].set_ylabel('Density', fontsize=10)

            # Add a legend to each plot
            axs[row_idx, col_idx].legend()

        handles = [plt.Line2D([0], [0], color='blue', lw=4, label='TrustValue = 1'),
           plt.Line2D([0], [0], color='red', lw=4, label='TrustValue = 0')]
        fig.legend(handles=handles, loc='right', fontsize='large')

        # Add the folder name as the row heading
        axs[row_idx, 0].annotate(f'{os.path.basename(subfolder).capitalize()}', xy=(0, 0.5), xytext=(-axs[row_idx, 0].yaxis.labelpad - 5, 0),
                                 xycoords=axs[row_idx, 0].yaxis.label, textcoords='offset points',
                                 size='large', ha='right', va='center', rotation=90)
        
        fig.suptitle('TrustValue Prediction Score Distributiion Across Different Centrality Measures', fontsize=16)


    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.2, hspace=0.5, wspace=0.3)  # Adjust margins
    plt.show()
    # plt.tight_layout()
    # plt.show()
'''
def prediction_scores_plot(file):

    df = pd.read_csv(file)
    df_gt_1 = df[df['TrustValue'] == 1]
    df_gt_0 = df[df['TrustValue'] == 0]

    # Plot overlapping KDEs
    plt.figure(figsize=(10, 6))

    sns.kdeplot(df_gt_1['Predicted_score'], fill=True, color='blue', label='Ground Truth = 1')
    sns.kdeplot(df_gt_0['Predicted_score'], fill=True, color='red', label='Ground Truth = 0')

    plt.title('Trust Prediction Distribution')
    plt.xlabel('Trust Prediction')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--folderpath",type = str)

    return parser.parse_args()

def main():
    inputs=parse_args()
#    plotdist(inputs.filename)
    prediction_scores_plot(inputs.folderpath)

  

if __name__ == '__main__':
    main()



    '''
    # Plot KDE for ground truth value 1
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(df_gt_1['score'], shade=True, color='blue')
    plt.title('Trust Prediction Distribution for Ground Truth = 1')
    plt.xlabel('Trust Prediction')
    plt.ylabel('Density')

    # Plot KDE for ground truth value 0
    plt.subplot(1, 2, 2)
    sns.kdeplot(df_gt_0['score'], shade=True, color='red')
    plt.title('Trust Prediction Distribution for Ground Truth = 0')
    plt.xlabel('Trust Prediction')
    plt.ylabel('Density')

    # Show the plots
    plt.tight_layout()
    plt.show()
    '''
