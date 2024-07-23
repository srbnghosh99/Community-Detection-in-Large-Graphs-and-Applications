import pandas as pd
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import os
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def create_folder(outdirectory):
    print('create folder')
    try:
        os.mkdir(outdirectory)
        print(f"Directory '{outdirectory}' created successfully")
    except FileExistsError:
        print(f"Directory '{outdirectory}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_metrics(inputdir):
    curr_directory = os.getcwd()
    directory = pjoin(curr_directory,inputdir)
    

    # List to store all CSV files in the directory
    csv_files = [filename for filename in os.listdir(directory) if filename.endswith('.csv')]

    # Determine the number of rows and columns for subplots (e.g., 2 columns)
    n_files = len(csv_files)
    n_cols = 2
    n_rows = (n_files + n_cols - 1) // n_cols  # Ensure enough rows

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Loop through all CSV files and plot each in a subplot
    for i, filename in enumerate(csv_files):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        ax = axes[i]
        for column in df.columns:
            ax.plot(df[column], marker='o', linestyle='-', label=column)
        
        filename_no_underscore = os.path.splitext(filename)[0].replace('_', ' ').title()
        ax.set_title(f"{filename_no_underscore} Comparison")
        ax.set_xlabel('Index')
        ax.set_ylabel(filename_no_underscore)
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot to a file
    output_path = os.path.join(directory, 'all_metrics_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputdir",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()

    plot_metrics(inputs.inputdir)
  

if __name__ == '__main__':
    main()

