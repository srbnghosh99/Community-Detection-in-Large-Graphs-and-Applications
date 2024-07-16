import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import os


def plot_metrices(directory_path):
    # List all CSV files in the given directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    # Extract dataset names from filenames
    dataset_names = [os.path.splitext(file)[0] for file in csv_files]

    # Read each CSV file into a DataFrame
    dataframes = [pd.read_csv(os.path.join(directory_path, file)) for file in csv_files]

    # Extract 'cluster_coeff_lis' column values from each DataFrame
    cluster_coeff_lis_values = [df['cluster_coeff_lis'].dropna().tolist() for df in dataframes]

    density_lis_values = [df['density_lis'].dropna().tolist() for df in dataframes]
    
    conductance_lis_values = [df['conductance_lis'].dropna().tolist() for df in dataframes]

    coverage_lis_values = [df['coverage_lis'].dropna().tolist() for df in dataframes]

    edge_cut_values = [df['edge_cut_lis'].dropna().tolist() for df in dataframes]

    normalized_cut_values = [df['normalized_cut_lis'].dropna().tolist() for df in dataframes]

    num_communities = [df['Number of communities'][0] for df in dataframes]
    modularity = [df['Modularity'][0] for df in dataframes]
    avg_clustering_coefficient = [df['overall_avg_clustering_coefficient'][0] for df in dataframes]


    # Function to calculate statistical summary
    def calculate_stats(values):
        mean_value = np.mean(values)
        median_value = np.median(values)
        percentile_75 = np.percentile(values, 75)
        percentile_90 = np.percentile(values, 90)
        return mean_value, median_value, percentile_75, percentile_90
    
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    
    def iterate_dataset(lis_values):
        # Calculate stats for each dataset
        for i, values in enumerate(lis_values):
            stats = calculate_stats(values)
            print(f"Stats for Dataset {i+1}: Mean={stats[0]}, Median={stats[1]}, 75th Percentile={stats[2]}, 90th Percentile={stats[3]}")
        


        lis_values_norm = [normalize(np.array(values)) for values in lis_values]

        # Apply log transformation to the data
        values_log = [np.log1p(values) for values in lis_values]
        return values_log


    cluster_coeff_values_log = iterate_dataset(cluster_coeff_lis_values)
    density_values_log = iterate_dataset(density_lis_values)
    conductance_values_log = iterate_dataset(conductance_lis_values)
    coverage_values_log = iterate_dataset(coverage_lis_values)
    edge_cut_values_log = iterate_dataset(edge_cut_values)
    normalized_cut_values_log = iterate_dataset(normalized_cut_values)



    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Log-Transformed Box Plots of Various Metrics of Non-Overlapping Community Detection Methods')

    sns.boxplot(data=cluster_coeff_values_log, ax=axs[0, 0])
    axs[0, 0].set_title('Clustering Coefficient')
    axs[0, 0].set_xlabel('Clustering Methods')
    axs[0, 0].set_ylabel('Log-Transformed Values')
    axs[0, 0].set_xticklabels(dataset_names)

    sns.boxplot(data=density_values_log, ax=axs[0, 1])
    axs[0, 1].set_title('Density')
    axs[0, 1].set_xlabel('Clustering Methods')
    axs[0, 1].set_ylabel('Log-Transformed Values')
    axs[0, 1].set_xticklabels(dataset_names)

    sns.boxplot(data=conductance_values_log, ax=axs[0, 2])
    axs[0, 2].set_title('Conductance')
    axs[0, 2].set_xlabel('Clustering Methods')
    axs[0, 2].set_ylabel('Log-Transformed Values')
    axs[0, 2].set_xticklabels(dataset_names)

    sns.boxplot(data=coverage_values_log, ax=axs[1, 0])
    axs[1, 0].set_title('Coverage')
    axs[1, 0].set_xlabel('Clustering Methods')
    axs[1, 0].set_ylabel('Log-Transformed Values')
    axs[1, 0].set_xticklabels(dataset_names)

    sns.boxplot(data=edge_cut_values_log, ax=axs[1, 1])
    axs[1, 1].set_title('Edge Cut')
    axs[1, 1].set_xlabel('Clustering Methods')
    axs[1, 1].set_ylabel('Log-Transformed Values')
    axs[1, 1].set_xticklabels(dataset_names)

    sns.boxplot(data=normalized_cut_values_log, ax=axs[1, 2])
    axs[1, 2].set_title('Normalized Cut')
    axs[1, 2].set_xlabel('Clustering Methods')
    axs[1, 2].set_ylabel('Log-Transformed Values')
    axs[1, 2].set_xticklabels(dataset_names)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    plt.savefig(directory_path+'/log_transformed_box_plots.png')

    # plt.show()

    # Indices for bar positions
    indices = range(len(num_communities))

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Various Metrics of Non-Overlapping Community Detection Methods')

    # Plot num_communities as bar plot
    axs[0].bar(indices, num_communities, color='skyblue')
    axs[0].set_title('Number of Communities')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Number of Communities')
    axs[0].set_xticks(indices)
    axs[0].set_xticklabels(dataset_names)

    # Plot modularity as bar plot
    axs[1].bar(indices, modularity, color='lightgreen')
    axs[1].set_title('Modularity')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Modularity')
    axs[1].set_xticks(indices)
    axs[1].set_xticklabels(dataset_names)

    # Plot clustering_coefficient as bar plot
    axs[2].bar(indices, avg_clustering_coefficient, color='lightcoral')
    axs[2].set_title('Clustering Coefficient')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Clustering Coefficient')
    axs[2].set_xticks(indices)
    axs[2].set_xticklabels(dataset_names)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(directory_path +'/metrics_barplots.png')

    # Show plot
    # plt.show()




def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--directory_path",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    plot_metrices(inputs.directory_path)
  

if __name__ == '__main__':
    main()


# command
#  python3 /Users/shrabanighosh/Downloads/data/recommendation_system/plot_community_metrices.py --directory_path /Users/shrabanighosh/Downloads/data/recommendation_system/ciao/community_metrices


    '''
    # Create a log-transformed box plot for visual inspection
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=normalized_cut_values)
    plt.title('Log-Transformed Box Plot of Clustering Coefficient (Combined)')
    plt.xlabel('Clustering Methods')
    plt.ylabel('Log-Transformed cluster_coeff_lis Values')
    plt.xticks(ticks=range(len(csv_files)), labels=dataset_names)
    plt.show()
    

   
def plot_metrices(directory_path):

    df1 = pd.read_csv("/Users/shrabanighosh/Downloads/data/recommendation_system/ciao/community_metrices/community_metrics_spectral.csv")
    df2 = pd.read_csv("/Users/shrabanighosh/Downloads/data/recommendation_system/ciao/community_metrices/community_metrics_label_prop.csv")
    df3 = pd.read_csv("/Users/shrabanighosh/Downloads/data/recommendation_system/ciao/community_metrices/community_metrics_louvain.csv")
    print(df1)
    # transposed_df = df.transpose()  # or df.T
    # print("\nTransposed dataframe:")
    # print(transposed_df)
    # Data from the datasets
    values1 = df1['cluster_coeff_lis'].tolist()
    values2 = df2['cluster_coeff_lis'].tolist()
    values3 = df3['cluster_coeff_lis'].tolist()
    # values1 = [6623.0, 15548.0, 9009.0, 1313.0, 18035.0, 16783.0, 18199.0, 5086.0, 23172.0, 4144.0]
    # values2 = [5551.0, 5.0, 11499.0, 3.0, 9247.0, 4.0, 10774.0, 4.0, 9735.0, 2.0, 4.0]
    # values3 = [4.0, 42.0, 2.0, 5.0, 7912.0, 3.0, 4.0, 13.0, 4.0, 10.0, 14.0, 15.0, 3.0, 5.0, 7900.0, 1.0, 20.0, 36.0, 2.0, 7.0, 2.0, 4.0, 11.0, 1.0, 6.0, 18.0, 4.0]

    # Function to calculate statistical summary
    def calculate_stats(values):
        mean_value = np.mean(values)
        median_value = np.median(values)
        percentile_75 = np.percentile(values, 75)
        percentile_90 = np.percentile(values, 90)
        return mean_value, median_value, percentile_75, percentile_90

    # Calculate stats for each dataset
    stats1 = calculate_stats(values1)
    stats2 = calculate_stats(values2)
    stats3 = calculate_stats(values3)

    print(f"Stats for Dataset 1: Mean={stats1[0]}, Median={stats1[1]}, 75th Percentile={stats1[2]}, 90th Percentile={stats1[3]}")
    print(f"Stats for Dataset 2: Mean={stats2[0]}, Median={stats2[1]}, 75th Percentile={stats2[2]}, 90th Percentile={stats2[3]}")
    print(f"Stats for Dataset 3: Mean={stats3[0]}, Median={stats3[1]}, 75th Percentile={stats3[2]}, 90th Percentile={stats3[3]}")

    # Normalize the data
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    values1_norm = normalize(np.array(values1))
    values2_norm = normalize(np.array(values2))
    values3_norm = normalize(np.array(values3))

    # Combine normalized datasets into a list
    values_combined_norm = [values1_norm, values2_norm, values3_norm]

    # Apply log transformation to the data
    values1_log = np.log1p(values1)
    values2_log = np.log1p(values2)
    values3_log = np.log1p(values3)

    # Combine log-transformed datasets into a list
    values_combined_log = [values1_log, values2_log, values3_log]


    # Create a log-transformed box plot for visual inspection
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=values_combined_log)
    plt.title('Log-Transformed Box Plot of edge_cut_lis Values (Combined)')
    plt.xlabel('Dataset')
    plt.ylabel('Log-Transformed edge_cut_lis Values')
    plt.xticks([0, 1, 2], ['First Dataset', 'Second Dataset', 'Third Dataset'])
    plt.show()

'''

    # Data
    # num_communities = [10, 12, 27]
    # modularity = [0.15011985079758200, 0.5184676487952370, 0.37151332882420900]
    # clustering_coefficient = [0.22127307382854700, 0.22127307382854700, 0.22127307382854700]

    # # Set the width of the bars
    # bar_width = 0.35

    # # Set position of bar on X axis
    # r1 = np.arange(len(num_communities))
    # r2 = [x + bar_width for x in r1]

    # # Create bar plot
    # plt.figure(figsize=(10, 6))
    # plt.bar(r1, modularity, color='b', width=bar_width, edgecolor='grey', label='Modularity')
    # plt.bar(r2, clustering_coefficient, color='r', width=bar_width, edgecolor='grey', label='Clustering Coefficient')

    # # Add labels
    # plt.xlabel('Number of Communities', fontweight='bold')
    # plt.ylabel('Values', fontweight='bold')
    # plt.title('Number of Communities vs Modularity and Clustering Coefficient')
    # plt.xticks([r + bar_width/2 for r in range(len(num_communities))], num_communities)
    # plt.legend()

    # # Show plot
    # plt.show()