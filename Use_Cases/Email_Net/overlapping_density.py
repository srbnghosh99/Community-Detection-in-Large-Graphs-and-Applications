import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def overlap_density(filename):
    df = pd.read_csv(filename)
    df['Community'] = df['Community'].apply(ast.literal_eval)

    # Flatten the community lists and count occurrences
    all_communities = [community for sublist in df['Community'] for community in sublist]
    community_counter = Counter(all_communities)

    # print(community_counter)
    # Get the top 50 communities
    top_communities = [community for community, _ in community_counter.most_common(10)]

    # print((top_communities))

    # Filter the DataFrame to include only nodes belonging to the top 50 communities
    # df['Filtered_Community'] = df[df['Community'].apply(lambda x: any(item in top_communities for item in x))]
    df['Filtered_Community'] = df['Community'].apply(lambda x: [item for item in x if item in top_communities])

    # print(df)

    # Extract unique communities
    unique_communities = sorted(set(c for sublist in df['Filtered_Community'] for c in sublist))

    # print(unique_communities)

    # Initialize the overlap matrix
    overlap_matrix = np.zeros((len(unique_communities), len(unique_communities)))


    # Create a dictionary mapping communities to the set of nodes in each community

    community_to_nodes = {community: set() for community in unique_communities}


    for index, row in df.iterrows():
        for community in row['Filtered_Community']:
            community_to_nodes[community].add(row['Node'])

    # Fill the overlap matrix
    for i, comm_i in enumerate(unique_communities):
        for j, comm_j in enumerate(unique_communities):
            if i <= j:
                shared_nodes = len(community_to_nodes[comm_i].intersection(community_to_nodes[comm_j]))
                shared_nodes = int(shared_nodes)
                # print('shared_nodes',shared_nodes)
                overlap_matrix[i, j] = shared_nodes
                overlap_matrix[j, i] = shared_nodes  # Symmetric matrix

    overlap_matrix = overlap_matrix.astype(int)
    # Convert the overlap matrix to a DataFrame for better readability
    overlap_df = pd.DataFrame(overlap_matrix, index=unique_communities, columns=unique_communities)
    overlap_df = overlap_df.astype(int)

    # Display the overlap matrix
    # print(overlap_df.dtypes)


    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_df, fmt='d', annot=True, cmap='coolwarm', cbar=True)

    plt.title('Heatmap of Overlapping Densities')
    plt.xlabel('Communities')
    plt.ylabel('Communities')
    plt.savefig('Overlapping Densities.png')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--filename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    overlap_density(inputs.filename)
  

if __name__ == '__main__':
    main()
