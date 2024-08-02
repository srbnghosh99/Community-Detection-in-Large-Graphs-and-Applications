## Run Commands

# Personalized Email Community Detection

## Overview

This repository implements a personalized email community detection system based on the paper "Personalized Email Community Detection using Collaborative Similarity Measure". Our system leverages collaborative similarity measures to detect email communities and enhance personalized email interactions by grouping users with similar email behaviors.

## How It Works

1. **Community Detection**:
   - Communities are detected within the email network. This process identifies groups of users who share similar email communication habits.

3. **Personalized Community Identification**:
   - Personalized communities are identified by analyzing the detected communities and matching users to the communities that best fit their email behavior and communication patterns.

## How It Works

1. **Extract Personalized Social Graph**:
   - A personalized social graph is extracted from a set of emails, where each node represents a user and is uniquely defined by their communication behavior.

2. **Collaborative Similarity Measure (CSM)**:
   - The Collaborative Similarity Measure (CSM) is employed to assess the similarity between users based on their interactions and behavior patterns.

3. **Intra-Graph Clustering**:
   - An intra-graph clustering approach is used to detect personalized communities within the social graph. This approach identifies groups of users with similar behavioral patterns.

## Evaluating Process

The effectiveness of the detected communities is evaluated using metrics such as density, entropy, and f-measure to ensure the communities are meaningful and well-formed.


## Reference

This code is based on the methodologies and concepts presented in the paper:
- "Personalized Email Community Detection using Collaborative Similarity Measure"

By following these steps and utilizing the provided scripts, you can implement a robust email community detection system tailored to enhance personalized email interactions. Feel free to modify the code to better fit your specific use case and improve the system's effectiveness.


run bash file -->   ./script.sh


