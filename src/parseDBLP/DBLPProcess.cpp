//XMLInspector from  git@github.com:mlabouardy/xmlinspector.git
#include "XmlInspector.hpp"
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>

using namespace std;

struct DBLP_Paper {
  std::vector<std::string> authors;
};



//replace spaces by underscores in names
std::string convertName(const string& s) {
  std::string ret = s;

  for (auto& c : ret) {
    if (c == ' ')
      c = '_';
  }
  
  return ret;
}

int main(int argc, const char * argv[]) {

  // check if we got the dblp xml file
  if (argc != 3 ){
    cout << "Usage: DBLPProcess <dblp.xml> <outfile>" << endl;
    exit(1);
  }
  
  Xml::Inspector<Xml::Encoding::Utf8Writer> inspector(argv[1]);

  
  
  DBLP_Paper p;

  int papercount=0;
  
  std::string prevtag;
  std::string prevtext;

  std::map<std::pair<std::string, std::string>, int> edges;
  
  auto processDBLP = [&](DBLP_Paper& p) {

    papercount++;

    if (0) {
      std::cout<<p.authors.size()<<"\n";
      for (auto s: p.authors) {
	std::cout<<s<<"\n";
      }
      std::cout<<"\n";
    }
    
    if (p.authors.size() > 1){
      for (int i=0;i<p.authors.size()-1; ++i) {
	std::pair<std::string, std::string> e;
	e.first = p.authors[i];
	e.second = p.authors[i+1];
	
	if (e.first < e.second)
	  std::swap(e.first, e.second);
	
	++edges[e];
      }
    }

    if (papercount % 5000) {
      std::cout<<"processing ... "<<papercount<<" currently"<<"\n";
    }
    
  };
  
  while (inspector.Inspect())
    {
      switch (inspector.GetInspected())
        {
	case Xml::Inspected::StartTag:
	  //std::cout << "StartTag name: " << inspector.GetName() << "\n";
	  prevtag = inspector.GetName();
	  break;
	case Xml::Inspected::EndTag:
	  //std::cout << "EndTag name: " << inspector.GetName() << "\n";
	  prevtag = "";
	  if (inspector.GetName() == "author") {
	    p.authors.push_back(prevtext);
	  }
	  if (inspector.GetName() == "article" || inspector.GetName() == "inproceedings")
	    {
	      processDBLP(p);
	      p= DBLP_Paper();
	    }
	  prevtext="";
	  break;
	case Xml::Inspected::EntityReference:
	  prevtext+= inspector.GetLocalName();
	  break;
	case Xml::Inspected::EmptyElementTag:
	  //std::cout << "EmptyElementTag name: " << inspector.GetName() << "\n";
	  break;
	case Xml::Inspected::Text:
	  //std::cout << "Text value: " << inspector.GetValue() << "\n";
	  prevtext += inspector.GetValue();
	  break;
	case Xml::Inspected::Comment:
	  //std::cout << "Comment value: " << inspector.GetValue() << "\n";
	  break;
	default:
	  // Ignore the rest of elements.
	  break;
        }
    }
  
  if (inspector.GetErrorCode() != Xml::ErrorCode::None)
    {
      std::cout << "Error: " << inspector.GetErrorMessage() <<
	" At row: " << inspector.GetRow() <<
	", column: " << inspector.GetColumn() << ".\n";
    }

  std::cout<<papercount<<" papers parsed"<<"\n";

  {
    ofstream outfile (argv[2]);
    for (auto p: edges) {
      outfile<<convertName(p.first.first)<<" "<<convertName(p.first.second)<<" "<<p.second<<"\n";
    }
  }

  
  
  return 0;
}
