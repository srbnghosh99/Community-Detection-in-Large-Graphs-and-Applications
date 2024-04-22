import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Assuming G is your directed graph representing the trust network

def distribution(graphfile1, graphfile2):
  
  epi = nx.read_edgelist(graphfile1,delimiter=' ', nodetype=int, create_using=nx.DiGraph)
  ciao = nx.read_edgelist(graphfile1,delimiter=' ', nodetype=int, create_using=nx.DiGraph)

  trustor_counts = Counter(dict(epi.in_degree()).values())
  trustee_counts = Counter(dict(epi.out_degree()).values())
  
  # Convert trustor and trustee counts to lists for plotting
  epi_trustor_degrees, epi_trustor_counts = zip(*sorted(trustor_counts.items()))
  epi_trustee_degrees, epi_trustee_counts = zip(*sorted(trustee_counts.items()))
  
  trustor_counts = Counter(dict(ciao.in_degree()).values())
  trustee_counts = Counter(dict(ciao.out_degree()).values())
  
  # Convert trustor and trustee counts to lists for plotting
  ciao_trustor_degrees, ciao_trustor_counts = zip(*sorted(trustor_counts.items()))
  ciao_trustee_degrees, ciao_trustee_counts = zip(*sorted(trustee_counts.items()))
  fig, axs = plt.subplots(2, 2, figsize=(16, 16))

  # Plot: epinions Trustor distribution
  axs[0, 0].plot(epi_trustor_counts, epi_trustor_degrees, label='Trustors', color='b', marker='o', linestyle='-')
  axs[0, 0].set_xlabel('Count of Trustors', fontsize=14)
  axs[0, 0].set_ylabel('Degree of Trustors', fontsize=14)
  axs[0, 0].set_title('Epinions Trustor Distribution', fontsize=16)
  axs[0, 0].legend()
  axs[0, 0].grid(True)
  
  axs[0, 1].plot(epi_trustee_counts, epi_trustee_degrees, label='Trustees', color='r', marker='o', linestyle='-')
  axs[0, 1].set_xlabel('Count of Trustees', fontsize=14)
  axs[0, 1].set_ylabel('Degree of Trustees', fontsize=14)
  axs[0, 1].set_title('Epinions Trustee Distribution', fontsize=16)
  axs[0, 1].legend()
  axs[0, 1].grid(True)
  
  
  # Plot: ciao Trustor distribution
  
  axs[1, 0].plot(ciao_trustor_counts, ciao_trustor_degrees, label='Trustors', color='b', marker='o', linestyle='-')
  axs[1, 0].set_xlabel('Count of Trustors', fontsize=14)
  axs[1, 0].set_ylabel('Degree of Trustors', fontsize=14)
  axs[1, 0].set_title('Ciao Trustor Distribution', fontsize=16)
  axs[1, 0].legend()
  axs[1, 0].grid(True)
  

  axs[1, 1].plot(ciao_trustee_counts, ciao_trustee_degrees, label='Trustees', color='r', marker='o', linestyle='-')
  axs[1, 1].set_xlabel('Count of Trustees', fontsize=14)
  axs[1, 1].set_ylabel('Degree of Trustees', fontsize=14)
  axs[1, 1].set_title('Ciao Trustee Distribution', fontsize=16)
  axs[1, 1].legend()
  axs[1, 1].grid(True)
  
  # Adjust layout
  plt.tight_layout()
  plt.savefig('data_distribution.png')
  plt.show()


  # trustor_counts = Counter(dict(G.out_degree()).values())
  # trustee_counts = Counter(dict(G.in_degree()).values())
  # print("Number of Trustee", len(trustee_counts))
  # print("Number of Trustor", len(trustor_counts))
  
  # # Convert trustor and trustee counts to lists for plotting
  # trustor_degrees, trustor_counts = zip(*sorted(trustor_counts.items()))
  # trustee_degrees, trustee_counts = zip(*sorted(trustee_counts.items()))
  
  # # Normalize counts to probabilities
  # total_trustors = sum(trustor_counts)
  # total_trustees = sum(trustee_counts)
  # print(total_trustors, total_trustees)
  # trustor_probs = [count / total_trustors for count in trustor_counts]
  # trustee_probs = [count / total_trustees for count in trustee_counts]
  
  
  # fig, axs = plt.subplots(1, 2, figsize=(14, 7))
  # # Plot the trustor distribution
  # axs[0].plot(trustor_degrees, trustor_counts, label='Trustors', color='b', marker='o', linestyle='-')
  # axs[0].set_xlabel('Number of Users', fontsize=14)
  # axs[0].set_ylabel('Number of Trustors', fontsize=14)
  # axs[0].set_title('Trustor Distribution', fontsize=16)
  # axs[0].legend()
  # axs[0].grid(True)
  
  # # Plot the trustee distribution
  # axs[1].plot(trustee_degrees, trustee_counts, label='Trustees', color='r', marker='o', linestyle='-')
  # axs[1].set_xlabel('Number of Users', fontsize=14)
  # axs[1].set_ylabel('Number of Trustees', fontsize=14)
  # axs[1].set_title('Trustee Distribution', fontsize=16)
  # axs[1].legend()
  # axs[1].grid(True)
  
  # # Adjust layout
  # plt.tight_layout()
  # plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--graphfile1",type = str)
    parser.add_argument("--graphfile2",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.graphfile)
    distribution(inputs.graphfile1,inputs.graphfile2 )
  

if __name__ == '__main__':
    main()
