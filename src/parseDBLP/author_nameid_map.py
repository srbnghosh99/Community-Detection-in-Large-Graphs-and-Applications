#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 01:54:23 EDT 2023

@author: shrabanighosh
"""

import pandas as pd 
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

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
    #print(df)
    #print(df3)

    newf = os.path.splitext(inputfile)[0]
    newf = newf+"_author_nameid.csv"
 
    df3 = df3[['Author','Id']]
    df3.to_csv(newf, index = False, sep = ' ')
    print("Number of authors  ",df3.shape)
    print("Author name and id mapped file created")

    df['Author1']=df['Author1'].map(res)
    df['Author2']=df['Author2'].map(res)
    #print(df)
    # df.columns = ['node1','node2']
    df.to_csv(outputfile,index = False,sep = ' ', header= False)
    print("Coauthor Net generated based on author mapped id")

def author_nameid_to_graph(input,output):
    colnames = ['Author1', 'Author2', 'Weight']
    df = pd.read_csv(input,names=colnames,header = None,sep = ' ')
    df1 = pd.read_csv("sorted_dblp_author_nameid.csv", sep = ' ')
    print(df1)
    author_dic = dict(zip(df1.Author, df1.Id))
    df['Author1'] = df['Author1'].map(author_dic)
    df['Author2'] = df['Author2'].map(author_dic)
    df.to_csv(output, header = False, index = False, sep = ' ')
    
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
