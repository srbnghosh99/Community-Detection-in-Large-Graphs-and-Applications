import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Assuming G is your directed graph representing the trust network

def distribution(graphfile):

  # G = nx.read_edgelist('/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/renumbered_graph_epinions.edgelist',delimiter=' ', nodetype=int, create_using=nx.DiGraph)
  # print(G.number_of_nodes(),G.number_of_edges())
  
  G = nx.read_edgelist(graphfile,delimiter=' ', nodetype=int, create_using=nx.DiGraph)
  print(G.number_of_nodes(),G.number_of_edges())


  trustor_counts = Counter(dict(G.out_degree()).values())
  trustee_counts = Counter(dict(G.in_degree()).values())
  print("Number of Trustee", len(trustee_counts))
  print("Number of Trustor", len(trustor_counts))
  
  # Convert trustor and trustee counts to lists for plotting
  trustor_degrees, trustor_counts = zip(*sorted(trustor_counts.items()))
  trustee_degrees, trustee_counts = zip(*sorted(trustee_counts.items()))
  
  # Normalize counts to probabilities
  total_trustors = sum(trustor_counts)
  total_trustees = sum(trustee_counts)
  print(total_trustors, total_trustees)
  trustor_probs = [count / total_trustors for count in trustor_counts]
  trustee_probs = [count / total_trustees for count in trustee_counts]
  
  
  fig, axs = plt.subplots(1, 2, figsize=(14, 7))
  # Plot the trustor distribution
  axs[0].plot(trustor_degrees, trustor_counts, label='Trustors', color='b', marker='o', linestyle='-')
  axs[0].set_xlabel('Number of Users', fontsize=14)
  axs[0].set_ylabel('Number of Trustors', fontsize=14)
  axs[0].set_title('Trustor Distribution', fontsize=16)
  axs[0].legend()
  axs[0].grid(True)
  
  # Plot the trustee distribution
  axs[1].plot(trustee_degrees, trustee_counts, label='Trustees', color='r', marker='o', linestyle='-')
  axs[1].set_xlabel('Number of Users', fontsize=14)
  axs[1].set_ylabel('Number of Trustees', fontsize=14)
  axs[1].set_title('Trustee Distribution', fontsize=16)
  axs[1].legend()
  axs[1].grid(True)
  
  # Adjust layout
  plt.tight_layout()
  plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--graphfile",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.graphfile)
    distribution(inputs.graphfile)
  

if __name__ == '__main__':
    main()
