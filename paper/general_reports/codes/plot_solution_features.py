import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
import os
import argparse

def create_folder(outdirectory):
    print('create folder')
    try:
        os.mkdir(outdirectory)
        print(f"Directory '{outdirectory}' created successfully")
    except FileExistsError:
        print(f"Directory '{outdirectory}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")

def clear_folder(outdirectory):
    print('Clear Folder')
    # Check if the folder exists
    if os.path.exists(outdirectory):
        # Remove all files in the folder
        for filename in os.listdir(outdirectory):
            file_path = os.path.join(outdirectory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {outdirectory} does not exist.")


def features_correlation(inputfile,outputdir,opti_func):
    df = pd.read_csv(inputfile)
    columnslist = df.columns
    print(columnslist)

    lis = ['Fitness_val','Modularity','Avg_density','Communities','Avg_Conductance','Avg_Cut_size','Avg_Clustering_coeff','avg_Centralization']
    print(df)
    feature_list = [col for col in lis if col in df.columns]
    print(feature_list)
    df=df[feature_list]
    create_folder(outputdir)
    clear_folder(outputdir)
    inc = 1
    df = df.rename(columns={'Fitness_val': opti_func})
    
    feature_list = df.columns
    print('feature_list',feature_list)
    for f1, f2 in combinations(feature_list, 2):
        print(f"Now working on pair: ({f1}, {f2})")
        x_vals = df[f1].tolist()
        y_vals = df[f2].tolist()

        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, y_vals, alpha=0.7)
        plt.title(f'Scatter Plot: {f1} vs {f2}')
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        # filename = 'figure/plots/'+ str(inc) +f1+'vs'+f2 + '.png'
        filename = outputdir + '/' + str(inc) + '.png'
        inc +=1
        plt.savefig(filename)
        # plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputfile", type=str)
    parser.add_argument("--outputdir", type=str)
    parser.add_argument("--opti_func", type=str)
    return parser.parse_args()


def main():
    inputs = parse_args()
    print(inputs.inputfile)
    print(inputs.outputdir)
    print(inputs.opti_func)
    features_correlation(inputs.inputfile,inputs.outputdir,inputs.opti_func)



if __name__ == '__main__':
    main()
