/*
 * Util.h
 *
 *  Created on: Mar 5, 2012
 *      Author: onur
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <limits>
#include <math.h>
#include <cstring>
#include <assert.h>
using namespace std;

// defaults
char emptystring[] = "";
string emptystring2;
char phdthesis[] = "PhD dissertation";
char mastersthesis[] = "Masters thesis";
string arxividstart("oai:arXiv.org:");
char citeseeridstart[] = "oai:CiteSeerX.psu:";
set<int> emptyset;

// output file names
const char* ofile_sets = "oai_sets.txt";
const char* ofile_setdefs = "oai_setdefs.txt";
const char* ofile_ids = "oai_ids.txt";
const char* ofile_titles = "oai_titles.txt";
const char* ofile_abstracts = "oai_abstracts.txt";
const char* ofile_links = "oai_links.txt";
const char* ofile_dois = "oai_dois.txt";
const char* ofile_refinfos = "oai_refinfos.txt";
const char* ofile_sources = "oai_sources.txt";
const char* ofile_years = "oai_years.txt";
const char* ofile_volumes = "oai_volumes.txt";
const char* ofile_issues = "oai_issues.txt";
const char* ofile_fpages = "oai_fpages.txt";
const char* ofile_lpages = "oai_lpages.txt";
const char* ofile_arxivs = "oai_arxivs.txt";
const char* ofile_venues = "oai_venues.txt";
const char* ofile_types = "oai_types.txt";
const char* ofile_pmcids = "oai_pmcids.txt";
const char* ofile_citeseerids = "oai_citeseer.txt";
const char* ofile_authors = "oai_authors.txt";
const char* ofile_recordsets = "oai_recordsets.txt";
const char* ofile_recordauthors = "oai_recordauthors.txt";
const char* ofile_recordsubjects = "oai_recordsubjects.txt";
const char* ofile_recordrelations = "oai_recordrelations.txt";
const char* ofile_wordmap = "wordlist.txt";
const char* ofile_blacklist = "blacklist.txt";

// output filenames for the service
const char* finalfile_ids = "theadvisor_ids.txt";
const char* finalfile_titles = "theadvisor_titles.txt";
const char* finalfile_years = "theadvisor_years.txt";
const char* finalfile_dois = "theadvisor_dois.txt";
const char* finalfile_paperauthors = "theadvisor_paperauthors.txt";
const char* finalfile_authornames = "theadvisor_authornames.txt";
const char* finalfile_authors = "theadvisor_authors.txt";
const char* finalfile_papervenue = "theadvisor_papervenue.txt";
const char* finalfile_venuenames = "theadvisor_venuenames.txt";
const char* finalfile_venues = "theadvisor_venues.txt";
const char* finalfile_volumes = "theadvisor_volumes.txt";
const char* finalfile_issues = "theadvisor_issues.txt";
const char* finalfile_fpages = "theadvisor_fpages.txt";
const char* finalfile_lpages = "theadvisor_lpages.txt";
const char* finalfile_types = "theadvisor_types.txt";
const char* finalfile_citeseerids = "theadvisor_citeseerids.txt";
const char* finalfile_pmcids = "theadvisor_pmcids.txt";
const char* finalfile_refinfos = "theadvisor_refinfos.txt";
const char* finalfile_graph = "theadvisor_citeref.gr";

// check if file exists
bool fexists(const char *filename)
{
  ifstream ifile(filename);
  bool exists = ifile?true:false;
  ifile.close();
  return exists;
}

struct paircomp {
  bool operator() (const pair<int,int>& lhs, const pair<int,int>& rhs) const
  {return (lhs.first==rhs.first) ? lhs.second<rhs.second : lhs.first<rhs.first;}
};

struct stringcomp {
  bool operator() (const string& lhs, const string& rhs) const
  {return lhs.compare(rhs) < 0;}
};

struct StringEq {
   const char* name;
   StringEq(const char* _name) : name(_name) {}
   bool operator()(const char* r1) const { return !strcmp(r1, name); }
};

template <typename T>
void FreeAll( T & t ) {
    T tmp;
    t.swap( tmp );
}

// trim string
static inline string &trim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

// generate a char* from a string
static inline char* stringtochar(string s) {
	char* ca = new char[s.size()+1];
	strcpy(ca, s.c_str());
	ca[s.size()] = '\0';
	return ca;
}

bool filter(char c)
{
	return c==',' || c=='?' || c=='\"' || c=='\'' || c=='(' || c==')' || c=='[' ||
	c==']' || c=='`' || c=='$' || c==':' || c==';' || c=='#' || c=='\\' || c=='{'
	 || c=='}' || c=='_' || c=='^' || c=='`';
}

bool filter2(char c) {
	return c=='/' || c=='-' || c=='.';
}

bool inline isUpperChar(char c) {
	return c>='A' && c<='Z';
}

bool inline isLowerChar(char c) {
	return c>='a' && c<='z';
}

bool inline isNumber(char c) {
	return c>='0' && c<='9';
}

bool inline hasNumber(string s) {
	for (unsigned int i=0; i<s.length(); i++)
		if (isNumber(s[i]))
			return true;
	return false;
}

bool inline isYear(string s) {
	try {
		int year = atoi(s.c_str());
		return year>1900 && year<=2012;
	}
	catch(...) {
		return false;
	}
}

char* extract_year(char* str) {
	string str2(str);
	if (str2[str2.length()-1]=='.')
		str2.erase(str2.length()-1);
	if (isYear(str2.substr(str2.length()-4)))
		return stringtochar(str2.substr(str2.length()-4,4));
	int pos=0;
	for (pos=str2.find('(',pos); pos!=string::npos; pos=str2.find('-',pos+1))
		if (str2[pos+5]==')' && isYear(str2.substr(pos+1,4)))
			return stringtochar(str2.substr(pos+1,4));
//	str2.resize(remove_if(str2.begin(), str2.end(), filter) - str2.begin());
	replace_if(str2.begin(), str2.end(), filter, ' ');
	replace_if(str2.begin(), str2.end(), filter2, ' ');
	stringstream iss(str2);
	string word;
	while (getline(iss, word, ' '))
		if (isYear(word))
			return stringtochar(word.substr(0,4));
	stringstream iss2(str2);
	while (getline(iss2, word, ' '))
		if (word.length()>4 && !hasNumber(word.substr(0,word.length()-4)) && isYear(word.substr(word.size()-4,4)))
			return stringtochar(word.substr(word.size()-4,4));
	return emptystring;
}

void inline clean_string(string& str) {
	if (str[str.length()-1] == '.')
		str.erase(str.length()-1);
	// remove certain characters
	str.resize(remove_if(str.begin(), str.end(), filter) - str.begin());
	// lowercase all characters
	transform(str.begin(), str.end(), str.begin(), (int(*)(int)) std::tolower);
}

// writes an array to a file
template <typename T>
void write_array(const char* fn, T* arr, int size) {
	ofstream file(fn);
	for (int i=0; i<size; i++)
		file << arr[i] << endl;
	file.close();
}

// writes a vector to a file
template <typename T>
void write_vector(const char* fn, vector<T>& vec) {
	ofstream file(fn);
	for (typename vector<T>::iterator it = vec.begin(); it != vec.end(); it++)
		file << *it << endl;
	file.close();
}

// writes a vector of vectors to a file with a delimiter
template <typename T>
void write_vectors(const char* fn, vector<vector<T> >& vec, const char dlm=',') {
	ofstream file(fn);
	typename vector< vector<T> >::iterator it;
	typename vector<T>::iterator it2;
	for (it = vec.begin(); it != vec.end(); it++) {
		if (it->empty())
			file << endl;
		else {
			stringstream ss;
			ss << *(it->begin());
			for (it2 = it->begin()+1; it2 != it->end(); it2++)
				ss << dlm << *it2;
			file << ss.str() << endl;
		}
	}
	file.close();
}

// writes a vector of vectors to a file using corresponding ids
void write_vectors_with_ids(const char* fn, vector<vector<char*> >& vec, vector<char*> ids, bool isCiteseer = false) {
	ofstream file(fn);
	for (unsigned int i=0; i<vec.size(); i++)
		if (!vec[i].empty())
			for (vector<char* >::iterator it=vec[i].begin(); it!=vec[i].end(); it++)
				if (isCiteseer)
					file << ids[i] << " " << citeseeridstart << *it << endl;
				else
					file << ids[i] << " " << *it << endl;
	file.close();
}

// read lines to string vector
void read_file_to_string_vector(const char* path, const char* fn, vector<string>& list, bool clean=false, bool isCiteseer=false) {
	stringstream ss;
	ss << path << fn;
	ifstream input(ss.str().c_str(), ios::in);
	string line;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
		if (clean) clean_string(line);
    if (isCiteseer) line = "cid=" + line;
		list.push_back(line);
	}
	input.close();
}

void read_file_to_string_vector_2(const char* path, const char* fn, vector< vector<string> >& list, const char dlm=';') {
	stringstream ss;
	ss << path << fn;
	ifstream input(ss.str().c_str(), ios::in);
	string line, author;
	int t=0;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
		stringstream iss(line);
		vector<string> vec;
		while (getline(iss, author, dlm))
			vec.push_back(author);
		list.push_back(vec);
		t++;
	}
	input.close();
}

void inline read_file_to_string_array(const char* fn, string* arr, int* rowperm = NULL) {
	ifstream input(fn, ios::in);
	string line;
	int t=0;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
		if (rowperm==NULL)
			arr[t++] = line;
		else
			arr[rowperm[t++]] = line;
	}
	input.close();
}

// read lines to int vector
void read_file_to_int_vector(const char* path, const char* fn, vector<int>& list) {
	stringstream ss;
	ss << path << fn;
	ifstream input(ss.str().c_str(), ios::in);
	string line;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
		list.push_back(atoi(line.c_str()));
	}
	input.close();
}

// read lines and put them in a map according to their line numbers
void read_file_to_map(const char* path, const char* fn, map<string,int>& hmap, bool isCiteseer=false) {
	stringstream ss;
	ss << path << fn;
	ifstream input(ss.str().c_str(), ios::in);
	string line;
	int t=hmap.size();
	while(input.good()) {
		getline(input,line);
		if (!input) break;
    if (isCiteseer) line = "cid=" + line;
		hmap[line] = t;
		t++;
	}
	input.close();
}

void read_file_to_map_2(const char* path, const char* fn, map<string,vector<int> >& hmap, const char dlm=',') {
	stringstream ss;
	ss << path << fn;
	ifstream input(ss.str().c_str(), ios::in);
	string line, word, id;
	int t=0;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
		stringstream iss(line);
		getline(iss, word, dlm);
		while (getline(iss, id, dlm))
			hmap[word].push_back(atoi(id.c_str()));
		t++;
	}
	input.close();
}

// read file, using the mapping put entries into a set
void read_file_map_to_set(const char* path, const char* fn, map<string,int>& hmap, set<int>& tset, bool isCiteseer=false) {
	stringstream ss;
	ss << path << fn;
	ifstream input(ss.str().c_str(), ios::in);
	string line;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
    if (isCiteseer) line = "cid=" + line;
		if (hmap.find(line)!=hmap.end())
			tset.insert(hmap[line]);
	}
	input.close();
}

// read file, map to ids, add as a pair if none of the records is in blacklist
void read_relations_map_to_pairs(const char* fn,map<string,int>& idmap,vector<pair<int,int> >& out,set<int>& blacklist=emptyset, bool isCiteseer1=false, bool isCiteseer2=false) {
	ifstream input(fn, ios::in);
	string line, e1, e2;
	int r1, r2;
	while(input.good()) {
		getline(input,line);
		if (!input) break;
		stringstream iss(line);
		getline(iss, e1, ' ');
    if (isCiteseer1) e1 = "cid=" + e1;
		if (idmap.find(e1)==idmap.end()) continue;
		r1 = idmap[e1];
		if (blacklist.find(r1)!=blacklist.end()) continue;
		getline(iss, e2, ' ');
    if (isCiteseer2) e2 = "cid=" + e2;
		if (idmap.find(e2)==idmap.end()) continue;
		r2 = idmap[e2];
		if (blacklist.find(r2)!=blacklist.end()) continue;
		out.push_back(make_pair(r1,r2));
	}
	input.close();
}

#endif /* UTIL_H_ */
