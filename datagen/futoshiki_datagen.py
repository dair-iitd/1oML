'''
Data generation script for Futoshiki 

Process:
    Generate a set of possible solutions for a given board size (if yfile not passed)
    loop over solutions:
        Randomly mask out k grid cells from the solution to generate k out of N missing problem
        Do dfs like search to find possible solutions for above generated query
'''

import numpy as np
import itertools
from tqdm.auto import tqdm
import pickle
from collections import Counter
from copy import deepcopy
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ofile', required=True, type=str, help='path to the output file')
parser.add_argument('--yfile', type=str, default=None, help='path to the file containing targets')
parser.add_argument('--num-samples', default=10000, type=int, help='num samples', required=True)
parser.add_argument('--board-size', default=5, type=int, help='size of board', required=True)
parser.add_argument('--num-constraints', default=5, type=int, help='max number of inequality contraints', required=True)
parser.add_argument('--num-missing', default=10, type=int, help='number of missing board positions', required=True)
parser.add_argument('--nthreads', type=int, default=1, help='number of threads to use for computation')
parser.add_argument('--mode', type=str, default="train", help='train, test or val mode')
args = parser.parse_args()

def permute(l):
    '''
    generate all possible permutations of given list l
    '''
    ll=[]
    num=len(l)
    if num==1:
        return [l]
    else:
        for i in range(num):
            tmp=permute(l[:i]+l[i+1:])
            for j in tmp:
                ll.append([l[i]]+j)
    return ll 

def fact(n):
    '''factorial function'''
    return np.product(np.arange(1,n+1))

def shuffle(a):
    '''shuffle given array'''
    np.random.shuffle(a)
    return a

def check_validity(grid, constraints=None):
    '''
    check if given grid satisfies all Futoshiki constraints 
    including given inequality constraints
    '''
    for x in range(len(grid)):
        row = set(grid[x])
        if len(row)!=len(grid): # check each row has unique elements
            return False
        col = set(grid[:,x])
        if len(col)!=len(grid): # check each col has unique elements
            return False
    if constraints is None:
        return True
    gt = zip(*np.nonzero(constraints[0]))
    for ind in gt:
        next_ind = (ind[0],ind[1]+1)
        if grid[next_ind]>grid[ind]: # verify grater than constraints
            return False
    lt = zip(*np.nonzero(constraints[1]))
    for ind in lt:
        next_ind = (ind[0],ind[1]+1)
        if grid[next_ind]<grid[ind]: # verify less than constraints
            return False
    return True

def generate_all_puzzle(grid_size, dumpfile):
    permutations = np.array(permute(list(range(1,grid_size+1))))
    puzzles = []
    offset_size = fact(grid_size-1)
    
    # find combination of rows which lead to solution
    for index in tqdm(itertools.product(*[shuffle(list(range(offset_size))) for x in range(grid_size)])):
        offset_index = [offset_size*i+x for i,x in enumerate(index)]
        grid = permutations[offset_index]
        if check_validity(grid):
            puzzles.append(grid)
            print("Found",len(puzzles))
        if len(puzzles)>300: # cutting off search since enumerating all possible solutions is intractable
            break
    
    # permute the rows of the solutions found 
    permuted_puzzles = []
    for puzzle in puzzles:
        for permut in permutations:
            permuted_puzzles.append(puzzle[permut-1])
    
    with open(dumpfile,"wb") as f:
        pickle.dump(permuted_puzzles,f)
    
    return permuted_puzzles

class FutoshikiDataset:
    """The dataset class for futoshiki tasks."""
    def __init__(self,
                 n=5,
                 num_missing = 1,
                 num_constraints = 5,
                 random_seed = 42,
                 data_file = None):
     
        super().__init__()

        self._n = n
        self.num_missing = num_missing
        self.relations = self.get_relation()
        self.num_constraints = num_constraints
     
        with open(data_file,"rb") as f:
            self.dataset = pickle.load(f)
         
        np.random.seed(random_seed)

    def get_relation(self):
        '''
        get binary relations between different grid cells
        '''
        n = self._n
        n2 = self._n**2
        n3 = self._n**3
        relations = np.zeros((n3, n3,3))
     
        for x in range(n3):
            row = int(x/n2)
            col = int((x%n2)/n)
            num = int(x%n2)%n
         
            for y in range(n):
                # cell constraints
                relations[x][row*n2+col*n+y][0]=1
             
                # row constraints
                relations[x][y*n2+col*n+num][1]=1
             
                # column constraints
                relations[x][row*n2+y*n+num][2]=1
        return relations

    def get_one_hot(self,grid):
        '''
        convert grid to one hot 
        '''
        grid = grid.flatten()
        expand_grid = np.zeros((grid.size, self._n+1))
        expand_grid[np.arange(grid.size),grid] = 1
        expand_grid = expand_grid[:,1:]
        expand_grid = expand_grid.flatten()
        return expand_grid

    def find_solutions(self,query,zero_ind, constraints):
        '''
        find solution to given query
        '''
        size = self._n
        query_tight = query.reshape(size,size)
        full_set = set(range(size+1))

        fill_sets = []
        # for each empty grid cell find candidates to fill in
        # candidates are determined by considering all elements missing from row, column of grid cell
        for ind in zero_ind:
            row_set = set(query_tight[int(ind/size)])
            col_set = set(query_tight[:,ind%size])
            fill_sets.append(list(full_set.difference(row_set.union(col_set))))

        solutions = []
        counter = 0
        # take cross product over candidate sets for each grid cell and check validity
        for sol in itertools.product(*fill_sets):
            solution = query_tight.flatten()
            solution[zero_ind] = sol
            solution = solution.reshape(size,size)
            if check_validity(solution, constraints):
                solutions.append(self.get_one_hot(solution))
                counter+=1
                if counter>1:
                    return solutions
        return solutions

    def pad_set(self,target_set):
        target_set = target_set[:self.max_count]
        pad_counter = self.max_count - len(target_set)
        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def get_constraints(self,grid):
        '''
        sample inequality constraints
        '''
        offset_grid = np.roll(grid,-1,axis=1)
        gt = grid>offset_grid
        gt[:,-1]=False
        lt = grid<offset_grid
        lt[:,-1]=False

        c = list(zip(*gt.nonzero()))
        idx = np.random.choice(range(len(c)),self.num_constraints,replace=True)
        gt_constraints = np.zeros_like(gt)
        for i in idx:
            gt_constraints[c[i]]=1

        c = list(zip(*lt.nonzero()))
        idx = np.random.choice(range(len(c)),self.num_constraints,replace=True)
        lt_constraints = np.zeros_like(lt)
        for i in idx:
            lt_constraints[c[i]]=1

        return np.stack([gt_constraints,lt_constraints])

    def generate_data(self, unique=False, ambiguous=False):
        for _ in range(100):
            ind = np.random.choice(range(len(self.dataset)))
            grid = self.dataset[ind]
            expanded_grid = self.get_one_hot(grid)
            board_dim = self._n
         
            constraints = self.get_constraints(grid)

            query = grid.flatten()
            zero_ind = np.random.choice(range(len(query)),self.num_missing,replace=False)
            mask = np.ones_like(query)
            mask[zero_ind]=0
            query = query*mask
            target_set = self.find_solutions(query,zero_ind,constraints)
            count = len(target_set)
            is_ambiguous = 1 if count>1 else 0
            if is_ambiguous and unique:
                continue
            if is_ambiguous==0 and ambiguous:
                continue
            qid = np.array([ind]+list(zero_ind))
            target = target_set[np.random.randint(len(target_set))]
            query = self.get_one_hot(query)
            gt_constraints = constraints[0].flatten().repeat(self._n)
            lt_constraints = constraints[1].flatten().repeat(self._n)
            query = np.stack([query,gt_constraints,lt_constraints]).transpose()
            return dict(n=board_dim, query=query, target=target, target_set=target_set, count=count, is_ambiguous=int(is_ambiguous), qid=qid, relations=self.relations)

        raise
    
def meta_gen(i,mode):
    futo = FutoshikiDataset(
                 n=args.board_size,
                 num_missing = args.num_missing,
                 num_constraints=args.num_constraints,
                 random_seed = i+20*mode,
                 data_file = args.yfile)
    return [futo.generate_data() for i in range(args.num_samples//args.nthreads)]


if args.yfile is None:
    generate_all_puzzle(args.board_size,"futoshiki_"+str(args.board_size)+".pkl")
    args.yfile = "futoshiki_"+str(args.board_size)+".pkl"

mode=0
if args.mode=="val":
    mode=1
if args.mode=="test":
    mode=2
data = Parallel(n_jobs=args.nthreads,backend="multiprocessing")(delayed(meta_gen)(i,mode) for i in range(args.nthreads))
data = [x for group in data for x in group]

with open(args.ofile,"wb") as f:
    pickle.dump(data,f)
