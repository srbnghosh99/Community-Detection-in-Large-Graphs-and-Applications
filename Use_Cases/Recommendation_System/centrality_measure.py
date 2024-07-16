#  <#Title#>
import subprocess
import os


def centrality_measure(inputfolder,outputfolder):
    node_script = 'myscript_copy.js'
    # Specify the folder directory you want to iterate over
    folder_path = inputfolder
#     "/Users/shrabanighosh/Downloads/data/recommendation_system/epinions/community_json"
    #folder_path = "subgraphs_epinions"
    #outpath = '../data/recommendation_system/propensity'
    outpath =outputfolder
#    "/Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity"

    # List all files in the folder
    file_names = os.listdir(folder_path)

    # Iterate over the file names
    for file_name in file_names:
        # Print each file name
        print(file_name)
        root, extension = os.path.splitext(file_name)
        # Specify the parameter to pass to the function
        inputfile = folder_path + '/'+ file_name
        outputfile = outpath + '/' + root + '_measure.json'
        print(inputfile,outputfile)
    #    measure = "closeness"

    #    # Run the Node.js script from the Python script and pass the parameter
        try:
            subprocess.run(['node', node_script, inputfile, outputfile], check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the Node.js script: {e}")
        


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputfolder",type = str)
    parser.add_argument("--outputfolder",type = str)
#    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfolder)
    print(inputs.outputfolder)
    centrality_measure(inputs.inputfolder,inputs.outputfolder)
  

if __name__ == '__main__':
    main()

    
