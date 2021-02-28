"""
1. This script generates 9 x 9 sudoku puzzles that have more than one solution.

2. It dumps a pickle file (args.ambfile) containing all the generated puzzles. 

3. args.ambfile is read by 'data_sampling_sudoku.py', which samples queries such 
that we have a uniform distribution of number of givens between 17 and 34: 

4. Start with the dataset proposed by Palm et al. (args.ifile). Available at "https://data.dgl.ai/dataset/sudoku-hard.zip. (args.ifile). It has 180k queries with only unique
solution and the number of givens are uniformly distributed in the range from 17 to 34.

5. Using the queries with 17-givens from the entire dataset of size 180k,  
randomly remove 1 of the givens, giving us a 16-givens puzzle which necessarily has more than 1 correct
solution (Mc Guire et al 2012). 

6. Randomly add 1 to 18 of the digits back from the solution of the original puzzle,
while ensuring that the query continues to have more than 1 solution. To find all the solutions of a query, use a third party tool jsolve (http://www.enjoysudoku.com/JSolve12.zip) (args.jsolve) 

7. Filter all queries having less than 50 solutions and  dump them at args.ambfile. 

8. This procedure gives us multi-solution queries with givens in the range of 17 to 34, just as the original dataset of puzzles
with only unique solution. 

9. Dump queries with unique solution to args.unqfile.
"""




from copy import deepcopy
import numpy as np
import pickle
from collections import Counter
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import argparse
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--ambfile', required=True, type=str, help='path to the output file')
parser.add_argument('--ifile', required=True, type=str, help='path to the input file containing unique sudoku puzzles and their solutions')
parser.add_argument('--jsolve', required=True, type=str, help='path to Jsolve binary file')
parser.add_argument('--unqfile', default=None, type=str, help='path to the output file')
args = parser.parse_args()


def generate_query(datapoint,num_add=5):
    zero_ind = np.nonzero(datapoint["query"]==0)[0]
    nonzero_ind = np.nonzero(datapoint["query"])[0]
    new_cells_ind = np.random.choice(zero_ind,size=num_add,replace=False)
    query = deepcopy(datapoint["query"])
    query[new_cells_ind] = datapoint["target"][new_cells_ind]
    mask_essential = np.random.choice(nonzero_ind,1)
    query[mask_essential]=0
    return query

def generate_input(queries, inp_file):
    with open(inp_file,"w") as f:
        for query in queries:
            print("".join(map(str,query)).replace("0","."),file=f)

def get_output(output_file):
    with open(output_file,"r") as f:
        l = f.readlines()
    l = l[:-1]
    ret_set = []
    line_count = len(l)
    index = 0
    while(index<line_count):
        n = int(l[index])
        target_set = [np.array(list(x.strip())).astype(np.int8) for x in set(l[index+1:n+index+1])]
        index += n+1
        ret_set.append(target_set)
        if len(ret_set)%2000==1999:
            print("Read solutions for {} queries".format(len(ret_set)))
    return ret_set

# read unique solutions puzzles and generate queries to be solved by Jsolve
rrn_data = pd.read_csv(args.ifile,header=None)
sudoku_queries = np.array([np.array(list(x)).astype(np.int8) for x in rrn_data[0]])
sudoku_sols = np.array([np.array(list(x)).astype(np.int8) for x in rrn_data[1]])

z = [dict(query=sudoku_queries[i],target=sudoku_sols[i]) for i in range(len(rrn_data)) 
     if (len(np.nonzero(sudoku_queries[i])[0])<18)]
# upsample 17 and 18 given queries since many of them tend to get rejected
queries = [generate_query(x,i) for i in ([1]*15 + [2]*5 + list(range(1,19))*2) for x in z]
print("Generated {} queries".format(len(queries)))

# generate input for Jsolve
generate_input(queries,"temp.in")

# call Jsolve
print("Running Jsolve")
subprocess.check_output(args.jsolve+ " ./temp.in > ./temp.out", shell=True)
print("All queries solved")

# read Jsolve output 
target_set = get_output("temp.out")

dataset = [dict(query=queries[i], target_set=target_set[i], count=len(target_set[i])) for i in range(len(target_set)) if (len(target_set[i])<50)]

counter = dict(Counter([len(np.nonzero(x["query"])[0]) for x in dataset]))
for key in sorted(counter.keys()):
    print(key,"givens:",counter[key])
    
print("Dumping multi-solution data to",args.ambfile)
with open(args.ambfile,"wb") as f:
    pickle.dump(dataset,f)

if args.unqfile is not None:
    print("Dumping unique-solution data to",args.unqfile)
    ch = list(range(len(sudoku_queries)))
    np.random.shuffle(ch)
    unique_dataset = [dict(query=sudoku_queries[i], target_set=[sudoku_sols[i]], count=1) for i in ch]
    with open(args.unqfile,"wb") as f:
        pickle.dump(unique_dataset,f)
