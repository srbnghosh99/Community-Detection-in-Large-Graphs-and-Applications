import csv
import json
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import ast
import os

#convert overlapping and non-overlapping community outputs in csv files.
# Specify the path to your JSON file and CSV output file
#json_file_path = 'author_names_communities_1.json'
#csv_file_path = 'author_names_communities_1.csv'


def jsontocsv(dataset,inputfilename):
# Open and read the JSON file

    current_directory = os.getcwd()
    json_file_path = os.path.join(current_directory, dataset, inputfilename)
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # rows = [{key: value[i] if i < len(value) else None for key, value in data.items()} for i in range(max(map(len, data.values())))]
    rows = [{"Node": key, "Community": value} for key, value in data.items()]


    # # Open the CSV file for writing
    # with open(csv_file_path, 'w', newline='') as csv_file:
    #     # Create a CSV writer object
    #     csv_writer = csv.writer(csv_file)

    #     # Write the header row (assuming the keys in the JSON are the header names)
    #     csv_writer.writerow(data.keys())

    #     # Write the values from the JSON dictionary
    #     csv_writer.writerow(data.values())

    # with open(csv_file_path, 'w', newline='') as csvfile:
    #     fieldnames = data.keys()
    #     csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     csvwriter.writeheader()
    #     csvwriter.writerows(rows)
    base_filename = os.path.splitext(inputfilename)[0]  # Remove extension
    csv_file_name = base_filename + ".csv"
    
    csv_file_path = os.path.join(current_directory, dataset, csv_file_name)
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ["Node", "Community"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        csvwriter.writeheader()
        csvwriter.writerows(rows)
    
    
    df = pd.read_csv(csv_file_path)
#    df['Community'] = df['Community'].apply(ast.literal_eval)
    print('Number of Communities',df['Community'].nunique())


def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--inputfilename",type = str)
#    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.dataset)
    print(inputs.inputfilename)
    jsontocsv(inputs.dataset,inputs.inputfilename)
  

if __name__ == '__main__':
    main()


#df['Community'] = df['Community'].apply(ast.literal_eval)
