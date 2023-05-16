#!/usr/bin/env python
import os
import csv
import xmltodict 
import argparse
import sys
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_dblp(inputpath,outputpath):

#  inputpath = "/content/sample_data/files"
  dir_list = os.listdir(inputpath)
  for filename in dir_list:
    if (filename.endswith(".xml") == False):
      continue
    else:
      print(filename)
      filen = inputpath +'/' + filename
      with open(filen, 'r') as file:
        filedata = file.read()
        data_dict = xmltodict.parse(filedata)
        keysList = list(data_dict['result']['hits']['hit'])
        score = []
        id = []
        pid = []
        name = []
        title = []
        venue = []
        pages = []
        year = []
        typ = []
        access = []
        key = []
        doi = []
        ee = []
        url = []

        # data = []
        for index in range(len(keysList)):
          # print(index)
          if 'authors' in keysList[index]['info']:
            authorlist = keysList[index]['info']['authors']['author']
            # print(authorlist)
            if len(authorlist) > 2:
              authors_pid = []
              authors_name = []   
              for authors in authorlist:
                # print(authors,len(authorlist))
                authors_pid.append(authors['@pid']) if '@pid' in authors else authors_pid.append('')
                authors_name.append(authors['#text']) if '#text' in authors else authors_name.append('')
                # authors_pid.append(authors['@pid']) 
                # authors_name.append(authors['#text'])
            else:
              authors_pid.append('')
              authors_name.append('')
          score.append(keysList[index]['info']['score']) if 'score' in keysList[index]['info'] else score.append('')
          id.append(keysList[index]['@id'])
          pid.append(authors_pid)
          name.append(authors_name)
          title.append(keysList[index]['info']['title']) if 'title' in keysList[index]['info'] else title.append('')
          venue.append(keysList[index]['info']['venue']) if 'venue' in keysList[index]['info'] else venue.append('')
          pages.append(keysList[index]['info']['year']) if 'pages' in keysList[index]['info'] else pages.append('')
          year.append(keysList[index]['info']['year']) if 'year' in keysList[index]['info'] else year.append('')
          typ.append(keysList[index]['info']['type']) if 'type' in keysList[index]['info'] else typ.append('')
          access.append(keysList[index]['info']['access']) if 'access' in keysList[index]['info'] else access.append('')
          key.append(keysList[index]['info']['key']) if 'key' in keysList[index]['info'] else key.append('')
          doi.append(keysList[index]['info']['doi']) if 'doi' in keysList[index]['info'] else doi.append('')
          ee.append(keysList[index]['info']['ee']) if 'ee' in keysList[index]['info'] else ee.append('')
          url.append(keysList[index]['info']['url']) if 'url' in keysList[index]['info'] else doi.append('')


        new_file_name = filename.replace('.xml', '.csv') 
        new_file_name = outputpath + '/' +  new_file_name
        print(new_file_name) 
        df = pd.DataFrame({'score':score, 'id':id, 'authors_pid': pid,'authors_name':name, 'title':title, 'venue':venue, 'pages': pages, 'year':year, 'type':typ, 'access':access, 'key':key, 'doi':doi, 'ee':ee, 'url':url})
        df.to_csv(new_file_name)


def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--inputdirectory",type = str)
    parser.add_argument("--outputdirectory",type = str)
    
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputdirectory)
    print(inputs.outputdirectory)
    parse_dblp(inputs.inputdirectory,inputs.outputdirectory)
  

if __name__ == '__main__':
    main()
