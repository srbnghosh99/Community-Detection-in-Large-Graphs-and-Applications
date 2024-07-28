import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import ast
import subprocess
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil
import time
import os
from os.path import dirname, join as pjoin


closeness = []
sameAsDegreeCentrality = []
betweenness = []
inCentrality = []
outCentrality = []

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

def node_propensity(dataset,inDirectory,outdirectory,path_to_ngraph):
    
    directory = os.getcwd()
    inDirectory = pjoin(directory,dataset, inDirectory)
    outdirectory = pjoin(directory,dataset, outdirectory)
    
    files = os.listdir(inDirectory)
    create_folder(outdirectory)
    clear_folder(outdirectory)
    

    # Iterate through the files
    for file_name in files:
        # Check if the item is a file (not a directory)
        if os.path.isfile(os.path.join(inDirectory, file_name)):
            # Full path to the file
            input_file_name = os.path.join(inDirectory, file_name)
            print('input_file_name',input_file_name)
            # with open(json_file, 'r') as file:
            #     data = json.load(file)
            #node_script_path = '/Users/shrabanighosh/Downloads/ngraph.centrality-main/myscript_copy.js'
            node_script_path = path_to_ngraph+'/myscript.js'

            # Get the directory of the Node.js script
            script_directory = os.path.dirname(os.path.abspath(node_script_path))
#            print("Script Directory:", script_directory)
            # Replace 'file_name.json' and 'output_file.json' with your actual parameters
    
            output_file_name = outdirectory +'/' +file_name

            # Construct the command to run the Node.js script
            command = ['node', node_script_path, input_file_name, output_file_name]
            script_directory = os.path.dirname(os.path.realpath(__file__))

            # Ensure the PATH includes the directory where nvm installs node
            nvm_dir = os.path.expanduser('~/.nvm')
            node_version = 'v18.17.1'  # Adjust this to the version you installed
            node_path = os.path.join(nvm_dir, 'versions/node', node_version, 'bin')

            # Add the node path to the PATH environment variable
            os.environ['PATH'] = node_path + os.pathsep + os.environ.get('PATH', '')

            # Run the command
            try:
                subprocess.run(command, check=True, cwd=script_directory)
                print(f"Node.js script executed successfully for {input_file_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error executing Node.js script: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--inDirectory",type = str)
    parser.add_argument("--outDirectory",type = str)
    parser.add_argument("--path_to_ngraph",type = str)
    return parser.parse_args()

def main():
    start_time = time.time()
    inputs=parse_args()
    print(inputs.inDirectory)
    print(inputs.outDirectory)
    node_propensity(inputs.dataset,inputs.inDirectory,inputs.outDirectory,inputs.path_to_ngraph)
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_hours = int(elapsed_time_seconds // 3600)
    elapsed_minutes = int((elapsed_time_seconds % 3600) // 60)

    print("Elapsed Time:", {elapsed_hours}, "hours", {elapsed_minutes}, "minutes")
  

if __name__ == '__main__':
    main()

