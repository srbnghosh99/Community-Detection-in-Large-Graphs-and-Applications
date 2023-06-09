#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:58:59 2023

@author: shrabanighosh
"""

import pandas as pd 
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def author_nameid_map(inputfile,outputfile):
    df = pd.read_csv(inputfile,sep = ' ')
    df1 = df['Author1']
    df2 = df['Author2']
    frames = [df1, df2]
    result = pd.concat(frames)
    df3 = pd.DataFrame(result)
    author_namelist = df3[0].drop_duplicates().tolist()
    author_idlist = list(range(1,len(author_namelist)+1)) 
    print("Total Number of authors ", len(author_idlist))
    res = {author_namelist[i]: author_idlist[i] for i in range(len(author_namelist))}
    # print(res)
    df3 = df3[0].drop_duplicates().reset_index()
    df3['Id'] = df3[0].map(res)
    df3 = df3.rename(columns={0: "Author"})
    # print(df3)
    
    df3 = df3[['Author','Id']]
    df3.to_csv("author_id_mapped_file.csv", index = False, sep = ' ')
    print("Author name and id mapped file created")
    df['Author1']=df['Author1'].map(res)
    df['Author2']=df['Author2'].map(res)
    df.columns = ['node1','node2']
    df.to_csv(outputfile,index = False,sep = ' ', header= True)
    print("Coauthor Net generated based on author mapped id")

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--inputfile",type = str)
    parser.add_argument("--outputfile",type = str)
    
    return parser.parse_args()

def main():
    inputs=parse_args()
    print("Input file name:: ", inputs.inputfile)
    print("Output file name:: ",inputs.outputfile)
    author_nameid_map(inputs.inputfile,inputs.outputfile)
  

if __name__ == '__main__':
    main()
