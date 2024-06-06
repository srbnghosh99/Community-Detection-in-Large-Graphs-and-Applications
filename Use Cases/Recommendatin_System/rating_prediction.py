
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
from datetime import datetime, timedelta
import time



# Objective function components
# objective_function(bu_grad, bi_grad, pu_grad, p_tilde_c_grad, qi_grad, lambda_)
# def objective_function(bu, bi, pu, p_tilde_c, qi, lambda_):

#     regularization = lambda_ * (np.sum(bu**2) + np.sum(bi**2) + np.sum(np.linalg.norm(pu, axis=1)**2) + 
#                                 np.sum(np.linalg.norm(p_tilde_c, axis=1)**2) + np.sum(np.linalg.norm(qi, axis=1)**2))
#     return regularization

def objective_function(bu, bi, pu, p_tilde_c, qi, lambda_):
    pu = np.atleast_2d(pu)
    qi = np.atleast_2d(qi)
    # print('pu:',{pu}, 'qi:', {qi})
    print(f"bu shape: {bu.shape}, bi shape: {bi.shape}, pu shape: {pu.shape}, p_tilde_c: {p_tilde_c}, qi shape: {qi.shape}, lambda shape: {np.shape(lambda_)}")
    
    regularization = lambda_ * (np.sum(bu**2) + np.sum(bi**2) + 
                                np.sum(np.linalg.norm(pu, axis=1)**2) + 
                                p_tilde_c**2 +  # Assuming p_tilde_c is a scalar value
                                np.sum(np.linalg.norm(qi, axis=1)**2))
    print('regularization',regularization)
    return regularization


def get_Ui(iid,trainset):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
def get_Iu(uid,trainset):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0

def node_propensity(dataset,trustnetfile,ratingfile,communityfile,inputdir,output_dir,overlap):   
    rating = pd.read_csv(ratingfile)
    detected_community_df = pd.read_csv(communityfile)
    trustnet = pd.read_csv(trustnetfile, sep = ' ')
    trustnet.rename(columns={trustnet.columns[0]:'Node1',trustnet.columns[1]:'Node2'}, inplace=True)
    print('trustnet dataframe',trustnet)
    if (overlap == 'overlapping'):
        print(overlap)
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
        #print(community_mapping[0])
        #community_mapping_df = pd.DataFrame.from_dict(community_mapping, orient='index')
        community_mapping_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_mapping_df = community_mapping_df.sort_values(by='Community', ascending=True)
        #print(community_mapping_df)
 

        community_preference_vector = {}
        for index, row in community_mapping_df.iterrows():
            comm = row['Community']
            node = row['Nodes']
            trunc_rating = rating[rating['userid'].isin(node)]
        #    print(trunc_rating['rating'].mean())
            community_preference_vector[comm]=trunc_rating['rating'].mean()
        community_preference_vector_df = pd.DataFrame(list(community_preference_vector.items()), columns=['Community', 'Rating_mean'])
        # print(community_preference_vector_df)

        filename = dataset + "_community_preference_vector_df.csv"
        print(filename)
        output_path = os.path.join(output_dir, filename)

        community_preference_vector_df.to_csv(output_path,index = False)

        user_interests = rating.groupby('userid')['rating'].mean()
        user_interests_df = user_interests.reset_index()

        # Rename the columns for clarity (optional)
        user_interests_df.columns = ['userid', 'inherent_interest']

        # Display the User Interest Vector
        # print(user_interests_df)

        filename = dataset+ "_user_interests_df.csv"
        output_path = os.path.join(output_dir, filename)
        user_interests_df.to_csv(output_path,index = False)

        user_interests_df = pd.read_csv(output_dir + dataset +"_user_interests_df.csv")
        community_preference_vector_df = pd.read_csv(output_dir + dataset + "_community_preference_vector_df.csv")
        #column_names = ['userid', 'Closeness', 'SameAsDegreeCentrality','Betweenness']
        node_propensity_df = pd.read_csv(output_dir + "epinions_node_propensity_dataframe.csv") 
        node_propensity_df.rename(columns={'Node':'userid'}, inplace=True) 
        print(node_propensity_df.columns)
        print('node_propensity_df',node_propensity_df)

        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(rating[['userid', 'productid', 'rating']], reader)
        full_trainset = data.build_full_trainset()
        lambda_ = 0.02
        reg = 0.02 
        learning_rate = 0.005

    
        latent_dim = 100
        total_num_users = full_trainset.n_users
        total_num_items = full_trainset.n_items
        global_mean = full_trainset.global_mean
        bu_grad = np.zeros(total_num_users)
        bi_grad = np.zeros(total_num_items)
        pu_grad = np.zeros((total_num_users, latent_dim))
        qi_grad = np.zeros((total_num_items, latent_dim))

        print(f"Number of users in the trainset: {total_num_users}")
        print(f"Number of items in the trainset: {total_num_items}")

        # ## Use SVD model (you can choose a different model)
        # model = SVD()
        model = SVD(reg_all=0.02)  # Example initial regularization value
        # model.fit(trainset)

        print("section 1")        
        col1 = trustnet['Node1'].tolist()
        col2 = trustnet['Node2'].tolist()
        nodelis = col1 + col2

        # print(rating)
        list2 = list(rating['userid'].unique())
        list1 = list(rating['productid'].unique())
        print('list1 & list2',len(list1),len(list2))
        values_to_exclude = list(set(list2) - set(nodelis))
        print('values_to_exclude',len(values_to_exclude))
        print(data)

        # Exclude sets with matching first values
        # _, testset = train_test_split(data, test_size=0.2, random_state=42) # this line for the one go training and testing.
        filtered_dataset = {item for item in full_trainset.all_ratings() if item[0] not in values_to_exclude}
        # Convert filtered dataset to a DataFrame
        filtered_df = pd.DataFrame(filtered_dataset, columns=['userID', 'itemID', 'rating'])

        # Use Surprise's Reader to parse the DataFrame
        reader = Reader(rating_scale=(1, 5))  # Adjust the rating scale as needed
        filtered_data = Dataset.load_from_df(filtered_df, reader)

        actual_ratings = []
        predicted_ratings = []
        kf = KFold(n_splits=2, random_state=42, shuffle=True)
        predicted_ratings_set = set()
        # 'Closeness',  'SameAsDegreeCentrality', 'Betweenness'
        propensity_variable = 'Betweenness'
        print('propensity_variable', propensity_variable)
        fold = 0
        losses = []
        num_iterations = 5
        overall_losses = []
        overall_rmse = []
        overall_mae = []
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            iteration_losses = []
            fold = 0

            for trainset, testset in tqdm(kf.split(filtered_data)): 

                num_users = trainset.n_users
                num_items = trainset.n_items
                global_mean = trainset.global_mean
                total_loss = 0
                regularization_loss = 0
                bu = bi = 0
                # bu_grad = np.zeros(num_users)
                # bi_grad = np.zeros(num_items)
                qi = np.zeros(model.n_factors)
                pu = np.zeros(model.n_factors)

                bu_grad = np.zeros(total_num_users)
                bi_grad = np.zeros(total_num_items)
                pu_grad = np.zeros((total_num_users, latent_dim))
                qi_grad = np.zeros((total_num_items, latent_dim))

                # Initialize gradients
                # bu_grad = np.zeros(num_users)
                # bi_grad = np.zeros(num_items)
                # pu_grad = np.zeros((num_items, latent_dim))
                # qi_grad = np.zeros((num_items, latent_dim))

                model.fit(trainset)
                fold += 1
                print('kfold',fold)
                count = 0
                squarred_error_sum = 0

                for userid, productid, rating in set((user,item,rating) for user, item, rating in testset):
                    is_userid_in_dataframe = userid in node_propensity_df['userid'].values
                    if is_userid_in_dataframe is False:
                        count = count + 1
                        continue
                    actual_ratings.append(rating)
                    communities = ast.literal_eval(node_propensity_df[node_propensity_df['userid'] == userid]['Community'].iloc[0])
                    propensity = ast.literal_eval(node_propensity_df[node_propensity_df['userid'] == userid][propensity_variable].iloc[0])
                    p_tilde_c_grad = 0
                    for index, prop in enumerate(propensity):
                        comm = communities[index]
                        # print('comm',comm)
                        # print(community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0])
                        p_tilde_c_grad += prop * community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0]
                    
                    
                    # print('p_tilde_c_grad',p_tilde_c_grad)
                    
                    if trainset.knows_user(userid):
                        # print('userid',userid)
                        inner_user_id = get_Iu(userid,trainset)
                        # inner_user_id = trainset.to_inner_uid(userid)
                        bu_grad[userid] = model.bu[inner_user_id]
                        pu_grad[userid] = model.pu[inner_user_id]
                        result = pu_grad[userid] + p_tilde_c_grad
                    else: 
                        bu_grad[userid] = bu
                        pu_grad[userid] = pu
                        result = pu + p_tilde_c_grad

                    if trainset.knows_item(productid):
                        # print('productid',productid)
                        inner_item_id = get_Ui(productid, trainset)
                        bi_grad[productid] = model.bi[inner_item_id]
                        qi_grad[productid] = model.qi[inner_item_id]
                        qi_transposed = qi_grad[productid].T
                    else: 
                        bi_grad[productid] = bi
                        qi_grad[productid] = qi
                        qi_transposed = qi.T
                                
                    predicted_rating = model.trainset.global_mean + bu_grad[userid] + bi_grad[productid] + np.dot(qi_transposed, result)
                    
                    predicted_rating = np.clip(predicted_rating, 1, 5)

                    predicted_ratings.append(predicted_rating)

                    predicted_ratings_set.add((userid, productid, predicted_rating))  
                    
                    squarred_error_sum += (rating - predicted_rating) ** 2
                print('squarred_error_sum',squarred_error_sum)
                    # break
                # Calculate loss for this fold
                print(f"bu shape: {bu_grad.shape}, bi shape: {bi_grad.shape}, pu shape: {pu_grad.shape}, p_tilde_c_grad: {p_tilde_c_grad}, qi shape: {qi_grad.shape}, lambda shape: {lambda_}")

                regularization_loss = objective_function(bu_grad, bi_grad, pu_grad, p_tilde_c_grad, qi_grad, lambda_)
                total_loss = squarred_error_sum + regularization_loss
                print('count',count)
                print('total_loss',total_loss)
                iteration_losses.append(total_loss)
            # Create a DataFrame
            df = pd.DataFrame({'Predicted Rating': predicted_ratings, 'Actual Rating': actual_ratings})
            # df.to_csv(output_dir + dataset + '_accuracy_epinions.csv',index = False)
            # Display the DataFrame
            # print(df)

            # df = pd.read_csv(output_dir+dataset+ "_accuracy_epinions.csv")
            # Calculate Root Mean Squared Error (RMSE)
            predicted_ratings = df['Predicted Rating'].tolist()
            actual_ratings = df['Actual Rating'].tolist()
            rmse = np.sqrt(np.mean((np.array(predicted_ratings) - np.array(actual_ratings)) ** 2))

            # Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(actual_ratings)))

            print("RMSE:", rmse)
            print("MAE:", mae)
            overall_mae.append(mae)
            overall_rmse.append(rmse)

        # final_loss = np.mean(overall_losses)
        # print(f'Final average loss: {final_loss}')
        final_mae = np.mean(overall_mae)
        final_rmse = np.mean(overall_rmse)
        print(f'Final average mae: {final_mae}')
        final_loss = np.mean(overall_losses)
        print("propensity_variable",propensity_variable)
        print(f'Final average rmse: {final_rmse}')
    
    else: 
        user_communities = detected_community_df.to_dict()
        user_communities = detected_community_df.set_index('Node')['Community'].to_dict()
        # print("user_communities",user_communities)
        community_mapping_df = detected_community_df.groupby('Community')['Node'].apply(list).reset_index()
        community_mapping_df = community_mapping_df.rename(columns={'Node': 'Nodes'})
        print(community_mapping_df.columns)
        print('rating',rating)
              
        community_preference_vector = {}
        for index, row in community_mapping_df.iterrows():
            comm = row['Community']
            node = row['Nodes']
            trunc_rating = rating[rating['userid'].isin(node)]
        #    print(trunc_rating['rating'].mean())
            community_preference_vector[comm]=trunc_rating['rating'].mean()

        community_preference_vector_df = pd.DataFrame(list(community_preference_vector.items()), columns=['Community', 'Rating_mean'])

        print('community_preference_vector_df', community_preference_vector_df)

        filename = dataset + "_community_preference_vector_df.csv"
        print(filename)
        output_path = os.path.join(output_dir, filename)

        community_preference_vector_df.to_csv(output_path,index = False)
        user_interests = rating.groupby('userid')['rating'].mean()
        user_interests_df = user_interests.reset_index()

        # Rename the columns for clarity (optional)
        user_interests_df.columns = ['userid', 'inherent_interest']

        # Display the User Interest Vector
        print(user_interests_df)

        filename = dataset+ "_user_interests_df.csv"
        output_path = os.path.join(output_dir, filename)
        print('output_path',output_path)
        user_interests_df.to_csv(output_path,index = False)

        user_interests_df = pd.read_csv(output_dir + dataset +"_user_interests_df.csv")
        community_preference_vector_df = pd.read_csv(output_dir + dataset + "_community_preference_vector_df.csv")
        #column_names = ['userid', 'Closeness', 'SameAsDegreeCentrality','Betweenness']
        node_propensity_df = pd.read_csv(output_dir + dataset + "_node_propensity_dataframe.csv")
        node_propensity_df.rename(columns={'Node':'userid'}, inplace=True)
        
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(rating[['userid', 'productid', 'rating']], reader)
        full_trainset = data.build_full_trainset()

        lambda_ = 0.02
        reg = 0.02 
        learning_rate = 0.005

    
        latent_dim = 100
        total_num_users = full_trainset.n_users
        total_num_items = full_trainset.n_items
        global_mean = full_trainset.global_mean
        bu_grad = np.zeros(total_num_users)
        bi_grad = np.zeros(total_num_items)
        pu_grad = np.zeros((total_num_users, latent_dim))
        qi_grad = np.zeros((total_num_items, latent_dim))

        print(f"Number of users in the trainset: {total_num_users}")
        print(f"Number of items in the trainset: {total_num_items}")

        ## Use SVD model (you can choose a different model)
        model = SVD(reg_all=0.02) 
        # model.fit(trainset)

        print("section 1")

        trustnet = pd.read_csv(trustnetfile, sep = ' ')
        print(trustnet)
        # rating = pd.read_csv(rating)

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
        # _, testset = train_test_split(data, test_size=0.2, random_state=42)
        filtered_dataset = {item for item in full_trainset.all_ratings() if item[0] not in values_to_exclude}
        # Convert filtered dataset to a DataFrame
        filtered_df = pd.DataFrame(filtered_dataset, columns=['userID', 'itemID', 'rating'])
        print(node_propensity_df)

         # Use Surprise's Reader to parse the DataFrame
        reader = Reader(rating_scale=(1, 5))  # Adjust the rating scale as needed
        filtered_data = Dataset.load_from_df(filtered_df, reader)
        
        actual_ratings = []
        predicted_ratings = []
        kf = KFold(n_splits=2, random_state=42, shuffle=True)
        predicted_ratings_set = set()
        losses = []
        num_iterations = 5
        overall_losses = []
        overall_rmse = []
        overall_mae = []
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            iteration_losses = []
            fold = 0

            for trainset, testset in tqdm(kf.split(filtered_data)): 

                num_users = trainset.n_users
                num_items = trainset.n_items
                global_mean = trainset.global_mean
                total_loss = 0
                regularization_loss = 0
                bu = bi = 0
                # bu_grad = np.zeros(num_users)
                # bi_grad = np.zeros(num_items)
                qi = np.zeros(model.n_factors)
                pu = np.zeros(model.n_factors)

                bu_grad = np.zeros(total_num_users)
                bi_grad = np.zeros(total_num_items)
                pu_grad = np.zeros((total_num_users, latent_dim))
                qi_grad = np.zeros((total_num_items, latent_dim))

                # Initialize gradients
                # bu_grad = np.zeros(num_users)
                # bi_grad = np.zeros(num_items)
                # pu_grad = np.zeros((num_items, latent_dim))
                # qi_grad = np.zeros((num_items, latent_dim))

                model.fit(trainset)
                fold += 1
                print('kfold',fold)
                count = 0
                squarred_error_sum = 0
            for userid, productid, rating in set((user,item,rating) for user, item, rating in testset):
                is_userid_in_dataframe = userid in node_propensity_df['userid'].values
                if is_userid_in_dataframe is False:
                    count = count + 1
                    continue
                actual_ratings.append(rating)
                comm = node_propensity_df[node_propensity_df['userid'] == userid]['Community'].iloc[0]
                # 'Closeness',  'SameAsDegreeCentrality', 'Betweenness'
                propensity_variable = 'Betweenness'
                # print('propensity_variable', propensity_variable)
                propensity = node_propensity_df[node_propensity_df['userid'] == userid][propensity_variable].iloc[0]
                # print("propensity",propensity)
                result = 0
                p_tilde_c_grad = propensity * community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0]
                # for index, prop in enumerate(propensity):
                #     comm = communities[index]
                # print('community_preference_vector_df',community_preference_vector_df)
                if trainset.knows_user(userid):
                        # print('userid',userid)
                        inner_user_id = get_Iu(userid,trainset)
                        # inner_user_id = trainset.to_inner_uid(userid)
                        bu_grad[userid] = model.bu[inner_user_id]
                        pu_grad[userid] = model.pu[inner_user_id]
                        result = pu_grad[userid] + p_tilde_c_grad
                else: 
                    bu_grad[userid] = bu
                    pu_grad[userid] = pu
                    result = pu + p_tilde_c_grad

                if trainset.knows_item(productid):
                    # print('productid',productid)
                    inner_item_id = get_Ui(productid, trainset)
                    bi_grad[productid] = model.bi[inner_item_id]
                    qi_grad[productid] = model.qi[inner_item_id]
                    qi_transposed = qi_grad[productid].T
                else: 
                    bi_grad[productid] = bi
                    qi_grad[productid] = qi
                    qi_transposed = qi.T
                

                predicted_rating = model.trainset.global_mean + bu_grad[userid] + bi_grad[productid] + np.dot(qi_transposed, result)
                        
                predicted_rating = np.clip(predicted_rating, 1, 5)

                predicted_ratings.append(predicted_rating)

                predicted_ratings_set.add((userid, productid, predicted_rating))  
                
                squarred_error_sum += (rating - predicted_rating) ** 2
            print('squarred_error_sum',squarred_error_sum)    
            # Calculate loss for this fold
            print(f"bu shape: {bu_grad.shape}, bi shape: {bi_grad.shape}, pu shape: {pu_grad.shape}, p_tilde_c_grad: {p_tilde_c_grad}, qi shape: {qi_grad.shape}, lambda shape: {lambda_}")

            regularization_loss = objective_function(bu_grad, bi_grad, pu_grad, p_tilde_c_grad, qi_grad, lambda_)
            total_loss = squarred_error_sum + regularization_loss
            print('count',count)
            print('total_loss',total_loss)
            iteration_losses.append(total_loss)

        # Create a DataFrame
        df = pd.DataFrame({'Predicted Rating': predicted_ratings, 'Actual Rating': actual_ratings})
        df.to_csv(output_dir + dataset + '_accuracy.csv',index = False)
        # # Display the DataFrame
        # print(df)

        # df = pd.read_csv(output_dir+dataset+ "_accuracy.csv")
        # # Calculate Root Mean Squared Error (RMSE)
        predicted_ratings = df['Predicted Rating'].tolist()
        actual_ratings = df['Actual Rating'].tolist()
        rmse = np.sqrt(np.mean((np.array(predicted_ratings) - np.array(actual_ratings)) ** 2))

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(actual_ratings)))

        print("RMSE:", rmse)
        print("MAE", mae)
        overall_mae.append(mae)
        overall_rmse.append(rmse)

        final_mae = np.mean(overall_mae)
        final_rmse = np.mean(overall_rmse)
        print(f'Final average mae: {final_mae}')
        final_loss = np.mean(overall_losses)
        print("propensity_variable",propensity_variable)
        print(f'Final average rmse: {final_rmse}')

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
#    print(inputs.cdfile)
#    print(inputs.inputdir)
#    print(inputs.outputdir)
#    node_propensity(inputs.dataset,inputs.inputdir,inputs.outputdir)
    start_time = time.time()
    node_propensity(inputs.dataset,inputs.trustnet,inputs.ratingfile,inputs.communityfile,inputs.inputdir,inputs.output_dir,inputs.overlap)
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time

    # Convert elapsed time to hours and minutes
    elapsed_hours = int(elapsed_time_seconds // 3600)
    elapsed_minutes = int((elapsed_time_seconds % 3600) // 60)

    # print("Start Time:", start_time)
    # print("End Time:", end_time)
    print("Elapsed Time:", elapsed_hours, "hours", elapsed_minutes, "minutes")
  

if __name__ == '__main__':
    main()


 
