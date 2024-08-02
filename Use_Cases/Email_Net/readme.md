# Personalized Email Community Detection

## Overview

This repository implements a personalized email community detection system based on the paper "Personalized Email Community Detection using Collaborative Similarity Measure". Our system leverages collaborative similarity measures to detect email communities and enhance personalized email interactions by grouping users with similar email behaviors.

## How It Works

1. **Extract Personalized Social Graph**:
   - A personalized social graph is extracted from a set of emails, where each node represents a user and is uniquely defined by their communication behavior.

2. **Collaborative Similarity Measure (CSM)**:
   - The Collaborative Similarity Measure (CSM) is employed to assess the similarity between users based on their interactions and behavior patterns. 

3. **Intra-Graph Clustering**:
   - An intra-graph clustering approach is used to detect personalized communities within the social graph. This approach identifies groups of users with similar behavioral patterns. We used different non-overlapping clustering methods for clustereing. 

## Evaluating Process

The effectiveness of the detected communities is evaluated using metrics such as density, entropy, and f-measure to ensure the communities are meaningful and well-formed.

## How This Code Works

This codebase implements the concepts from the paper "A Community-Based Collaborative Filtering Method for Social Recommender Systems" with some modifications to suit our specific use case.
   - Ensure the output file of communities in csv format:
     ```bash
     ./script.sh
     ```
     
## Reference

This code is based on the methodologies and concepts presented in the paper:
- "Personalized Email Community Detection using Collaborative Similarity Measure"

By following these steps and utilizing the provided scripts, you can implement a robust email community detection system tailored to enhance personalized email interactions. Feel free to modify the code to better fit your specific use case and improve the system's effectiveness.


