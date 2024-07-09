#  <#Title#>

import json
import pandas as pd
import os
import argparse
import sys
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np
import ast
import itertools
from surprise.model_selection import cross_validate, KFold
from sklearn.utils import shuffle
from tqdm import tqdm 



def node_propensity(dataset,trustnetfile,ratingfile,communityfile,inputdir,output_dir,overlap):
    rating = pd.read_csv(ratingfile)
    detected_community_df = pd.read_csv(communityfile)
    print(rating,detected_community_df)
    trustnet = pd.read_csv(trustnetfile, sep = ' ')
    print(trustnet)
    trustnet.rename(columns={trustnet.columns[0]:'Node1',trustnet.columns[1]:'Node2'}, inplace=True)
    if (overlap == 'overlapping'):
        detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)

        user_communities = detected_community_df.to_dict()

        user_communities = detected_community_df.set_index('Node')['Community'].to_dict()

        community_mapping = {}
        for index, row in detected_community_df.iterrows():
            nodes = row['Node']
            community = row['Community']
            # print(community)
            for c in community:
                # print(c)
                if c in community_mapping:
                    community_mapping[c].append(nodes)
                else:
                    community_mapping[c] = [nodes]


        community_mapping_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_mapping_df = community_mapping_df.sort_values(by='Community', ascending=True)
        community_preference_vector = {}
        for index, row in community_mapping_df.iterrows():
            comm = row['Community']
            node = row['Nodes']
            trunc_rating = rating[rating['userid'].isin(node)]
            community_preference_vector[comm]=trunc_rating['rating'].mean()
        community_preference_vector_df = pd.DataFrame(list(community_preference_vector.items()), columns=['Community', 'Rating_mean'])
        print(community_preference_vector_df)

        filename = dataset + "_community_preference_vector_df.csv"
        print(filename)
        output_path = os.path.join(output_dir, filename)

        community_preference_vector_df.to_csv(output_path,index = False)


        # Assuming 'ratings_df' is your ratings DataFrame
        # Replace 'userid' and 'rating' with your actual column names
        user_interests = rating.groupby('userid')['rating'].mean()
        user_interests_df = user_interests.reset_index()

        # Rename the columns for clarity (optional)
        user_interests_df.columns = ['userid', 'inherent_interest']

        # Display the User Interest Vector
        print(user_interests_df)

        filename = dataset+ "_user_interests_df.csv"
        output_path = os.path.join(output_dir, filename)
        user_interests_df.to_csv(output_path,index = False)

        user_interests_df = pd.read_csv(output_dir + dataset +"_user_interests_df.csv")
        community_preference_vector_df = pd.read_csv(output_dir + dataset + "_community_preference_vector_df.csv")
        #column_names = ['userid', 'Closeness', 'SameAsDegreeCentrality','Betweenness']
        node_propensity_df = pd.read_csv(output_dir + dataset + "_node_propensity_dataframe.csv")

        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(rating[['userid', 'productid', 'rating']], reader)
        trainset = data.build_full_trainset()
        num_users = trainset.n_users
        num_items = trainset.n_items
        print('num_users',num_users)
        print('num_items',num_items)
        

        print(f"Number of users in the trainset: {num_users}")
        print(f"Number of items in the trainset: {num_items}")

        ## Use SVD model (you can choose a different model)
        model = SVD()
        model.fit(trainset)

        print("section 1")

        
        col1 = trustnet['Node1'].tolist()
        col2 = trustnet['Node2'].tolist()
        nodelis = col1 + col2

        print(rating)
        list2 = list(rating['userid'].unique())
        list1 = list(rating['productid'].unique())
        print(len(list1),len(list2))
        values_to_exclude = list(set(list2) - set(nodelis))
        print(len(values_to_exclude))
        

        # Exclude sets with matching first values
        _, testset = train_test_split(data, test_size=0.2, random_state=42)
        filtered_testset = {item for item in testset if item[0] not in values_to_exclude}
        print(node_propensity_df)
        actual_ratings = []
        predicted_ratings = []
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        predicted_ratings_set = set()

        propensity_variable = 'Closeness'
        for trainset2, testset2 in tqdm(kf.split(data)):    
            print('kfold')
            count = 0
            shuffled_testset = testset2
            for userid, productid, rating in set((user,item,rating) for user, item, rating in shuffled_testset):
                is_userid_in_dataframe = userid in node_propensity_df['userid'].values
                if is_userid_in_dataframe is False:
                    count = count + 1
                    continue
                actual_ratings.append(rating)
                # print('node_propensity_df',node_propensity_df)
                communities = ast.literal_eval(node_propensity_df[node_propensity_df['userid'] == userid]['Community'].iloc[0])
                # print('communities',communities[0])
                propensity = ast.literal_eval(node_propensity_df[node_propensity_df['userid'] == userid][propensity_variable].iloc[0])
                # print(propensity)
                result = 0
                for index, prop in enumerate(propensity):
                    # print('index',index)
                    comm = communities[index]
                    # print('comm',comm)
                    # print(community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0])
                    result += prop * community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0]
                inner_product_id = trainset.to_inner_iid(productid)
                inner_user_id = trainset.to_inner_uid(userid)
                bu = model.bu[inner_user_id]
                bi = model.bi[inner_product_id]
                qi = model.qi[inner_product_id]
                pu = model.pu[inner_user_id]
                result = pu + result
                qi_transposed = qi.T
                predicted_rating = model.trainset.global_mean + bu + bi + np.dot(qi_transposed, result)
                predicted_ratings.append(predicted_rating)
                predicted_ratings_set.add((userid, productid, predicted_rating))
            
            print('count',count)


        # Create a DataFrame
        df = pd.DataFrame({'Predicted Rating': predicted_ratings, 'Actual Rating': actual_ratings})
        df.to_csv(output_dir + dataset + '_accuracy_epinions.csv',index = False)
        # Display the DataFrame
        print(df)

        df = pd.read_csv(output_dir+dataset+ "_accuracy_epinions.csv")
        # Calculate Root Mean Squared Error (RMSE)
        predicted_ratings = df['Predicted Rating'].tolist()
        actual_ratings = df['Actual Rating'].tolist()
        rmse = np.sqrt(np.mean((np.array(predicted_ratings) - np.array(actual_ratings)) ** 2))

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(actual_ratings)))

        print("RMSE:", rmse)
        print("MAE:", mae)
    
    else: 
        user_communities = detected_community_df.to_dict()
        user_communities = detected_community_df.set_index('Node')['Community'].to_dict()
        community_mapping_df = detected_community_df.groupby('Community')['Node'].apply(list).reset_index()
        community_mapping_df = community_mapping_df.rename(columns={'Node': 'Nodes'})
        print(community_mapping_df.columns)
        
        
        community_preference_vector = {}
        for index, row in community_mapping_df.iterrows():
            comm = row['Community']
            node = row['Nodes']
            trunc_rating = rating[rating['userid'].isin(node)]
            community_preference_vector[comm]=trunc_rating['rating'].mean()

        community_preference_vector_df = pd.DataFrame(list(community_preference_vector.items()), columns=['Community', 'Rating_mean'])

        print(community_preference_vector_df)

        filename = dataset + "_community_preference_vector_df.csv"
        print(filename)
        output_path = os.path.join(output_dir, filename)

        community_preference_vector_df.to_csv(output_path,index = False)
        user_interests = rating.groupby('userid')['rating'].mean()
        user_interests_df = user_interests.reset_index()
        user_interests_df.columns = ['userid', 'inherent_interest']
        filename = dataset+ "_user_interests_df.csv"
        output_path = os.path.join(output_dir, filename)
        print('output_path',output_path)
        user_interests_df.to_csv(output_path,index = False)

        user_interests_df = pd.read_csv(output_dir + dataset +"_user_interests_df.csv")
        community_preference_vector_df = pd.read_csv(output_dir + dataset + "_community_preference_vector_df.csv")
        #column_names = ['userid', 'Closeness', 'SameAsDegreeCentrality','Betweenness']
        node_propensity_df = pd.read_csv(output_dir + dataset + "_node_propensity_dataframe.csv")
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(rating[['userid', 'productid', 'rating']], reader)
        trainset = data.build_full_trainset()
        num_users = trainset.n_users
        num_items = trainset.n_items

        print(f"Number of users in the trainset: {num_users}")
        print(f"Number of items in the trainset: {num_items}")

        ## Use SVD model (you can choose a different model)
        model = SVD()
        model.fit(trainset)

        print("section 1")

        trustnet = pd.read_csv(trustnetfile, sep = ' ')
        print(trustnet)

        trustnet.rename(columns={trustnet.columns[0]:'Node1',trustnet.columns[1]:'Node2'}, inplace=True)
        col1 = trustnet['Node1'].tolist()
        col2 = trustnet['Node2'].tolist()
        nodelis = col1 + col2

        print(rating)
        list2 = list(rating['userid'].unique())
        list1 = list(rating['productid'].unique())
        print(len(list1),len(list2))
        values_to_exclude = list(set(list2) - set(nodelis))
        print(len(values_to_exclude))
        

        # Exclude sets with matching first values
        _, testset = train_test_split(data, test_size=0.2, random_state=42)
        filtered_testset = {item for item in testset if item[0] not in values_to_exclude}
        print(node_propensity_df)
        actual_ratings = []
        predicted_ratings = []
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        predicted_ratings_set = set()

        for trainset2, testset2 in kf.split(data):
            print('kfold')
            count = 0
            shuffled_testset = testset2
            for userid, productid, rating in set((user,item,rating) for user, item, rating in shuffled_testset):
                is_userid_in_dataframe = userid in node_propensity_df['userid'].values
                if is_userid_in_dataframe is False:
                    count = count + 1
                    continue
                actual_ratings.append(rating)
                comm = node_propensity_df[node_propensity_df['userid'] == userid]['Community'].iloc[0]
                # print(communities)
                propensity = node_propensity_df[node_propensity_df['userid'] == userid]['Closeness'].iloc[0]
                # print("propensity",propensity)
                result = 0
                # for index, prop in enumerate(propensity):
                #     comm = communities[index]
                # print('community_preference_vector_df',community_preference_vector_df)
                result = propensity * community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0]
                inner_product_id = trainset.to_inner_iid(productid)
                inner_user_id = trainset.to_inner_uid(userid)
                bu = model.bu[inner_user_id]
                bi = model.bi[inner_product_id]
                qi = model.qi[inner_product_id]
                pu = model.pu[inner_user_id]
                result = pu + result
                qi_transposed = qi.T
                predicted_rating = model.trainset.global_mean + bu + bi + np.dot(qi_transposed, result)
                predicted_ratings.append(predicted_rating)
                predicted_ratings_set.add((userid, productid, predicted_rating))
            print('count',count)


        # Create a DataFrame
        df = pd.DataFrame({'Predicted Rating': predicted_ratings, 'Actual Rating': actual_ratings})
        df.to_csv(output_dir + dataset + '_accuracy_epinions.csv',index = False)
        # Display the DataFrame
        print(df)

        df = pd.read_csv(output_dir+dataset+ "_accuracy_epinions.csv")
        # Calculate Root Mean Squared Error (RMSE)
        predicted_ratings = df['Predicted Rating'].tolist()
        actual_ratings = df['Actual Rating'].tolist()
        rmse = np.sqrt(np.mean((np.array(predicted_ratings) - np.array(actual_ratings)) ** 2))

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(actual_ratings)))

        print("RMSE:", rmse)
        print("MAE", mae)

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--trustnet",type = str)
    parser.add_argument("--ratingfile",type = str)
    parser.add_argument("--communityfile",type = str)
    parser.add_argument("--inputdir",type = str)
    parser.add_argument("--output_dir",type = str)
    parser.add_argument("--overlap",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    node_propensity(inputs.dataset,inputs.trustnet,inputs.ratingfile,inputs.communityfile,inputs.inputdir,inputs.output_dir,inputs.overlap)
  

if __name__ == '__main__':
    main()


 
