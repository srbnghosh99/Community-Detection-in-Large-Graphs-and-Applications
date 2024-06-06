import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import seaborn as sns



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

def prediction_scores_plot(file):

    df = pd.read_csv(file)
    df_gt_1 = df[df['ground_truth'] == 1]
    df_gt_0 = df[df['ground_truth'] == 0]

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
    # Plot overlapping KDEs
    plt.figure(figsize=(10, 6))

    sns.kdeplot(df_gt_1['score'], fill=True, color='blue', label='Ground Truth = 1')
    sns.kdeplot(df_gt_0['score'], fill=True, color='red', label='Ground Truth = 0')

    plt.title('Trust Prediction Distribution')
    plt.xlabel('Trust Prediction')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--filename",type = str)

    return parser.parse_args()

def main():
    inputs=parse_args()
#    plotdist(inputs.filename)
    prediction_scores_plot(inputs.filename)

  

if __name__ == '__main__':
    main()
