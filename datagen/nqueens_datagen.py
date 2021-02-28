'''
Data generation script for N-Queens

Process:
    Given an integer N, generate all possible solutions of N-Queens.
    Loop over solutions:
        Randomly mask out k rows from the solution to generate k out of N missing problem
        Match against the previously generated solutions to identify all the possible solutions to the given problem.
'''

import numpy as np
import pickle 
import copy
import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--board-size', type=int, default=10,
                    help='board dimension') 
parser.add_argument('--num-missing', type=int, default = 0, help='number of missing queens')
parser.add_argument('--num-samples', type=int, default = None, help='number of datapoints')
parser.add_argument('--ofile', type=str, default="", help='output file name')
parser.add_argument('--sample',action='store_true',help="sample choice of missing queens or try out all missing queen combinations")

def get_xy(a,n):
    row = a//n
    col = a % n
    return (row,col)

class NQueenSolution(object):
    '''
    Generate all possible solutions for N-Queens
    '''
    def __init__(self):
        self.solutions = []

    def reset(self):
        self.solutions = []

    def solve(self, n):
        grid = np.zeros((n,n))
        solved = self.helper(n, 0, grid)

    def helper(self, n, row, grid):
        if n == row:
            self.solutions.append(copy.deepcopy(grid))
            return
        for col in range(n):
            if self.is_safe(row, col, grid):
                grid[row][col] = 1
                self.helper(n, row + 1, grid)
                if len(self.solutions) >= 4000:
                    return
                grid[row][col] = 0
                
    def is_safe(self, row, col, board):
        for i in range(len(board)):
            if board[row][i] == 1 or board[i][col] == 1:
                return False
        i = 0
        while row - i >= 0 and col - i >= 0:
            if board[row - i][col - i] == 1:
                return False
            i += 1
        i = 0
        while row + i < len(board) and col + i < len(board):
            if board[row + i][col - i] == 1:
                return False
            i += 1
        i = 1
        while row + i < len(board) and col - i >= 0:
            if board[row+ i][col - i] == 1:
                return False
            i += 1
        i = 1
        while row - i >= 0 and col + i < len(board):
            if board[row - i][col + i] == 1:
                return False
            i += 1
        return True

def generate_comb_helper(arr, num):
    '''
    Enumerate all possible subsets of given array having given number of elements

    arr: array of objects whose subsets need to be returned
    num: number of elements in each returned subset

    rtype: list of list 
    '''
    if len(arr)<num:
        return []
    if num==1:
        return [[x] for x in arr] 
    for i,x in enumerate(arr[:-(num-1)]):
        # include first element
        retval = [[arr[0]]+x for x in generate_comb_helper(arr[1:],num-1)]
        # exclude first element
        retval.extend(generate_comb_helper(arr[1:],num))
    return retval

def generate_all_combination(board_dim=10, num_missing=5):
    return np.array(generate_comb_helper(list(range(board_dim)),num_missing))

def generate_data(board_dim=10, num_missing=5, sample=False):
    '''
    Generate dataset for N-Queens problem

    Args:
    board_dim (int): dimension of chess board i.e. N in N-Queens
    num_missing (int): number of queens missing from the board
    sample (bool): whether to subsample number of combinations for each solution 

    rtype: list of datapoints where each datapoint is dictionary containing
            {
                n: board dimension
                query: N-Queens puzzle with N-num_missing queens placed
                target_set: set of all possible solutions of N-Queens matching the above query
                count: len of target_set
                is_ambiguous: whether given query is_ambiguous or not i.e. count>1
                qid: unique identifier for query
            }

    '''
    solver = NQueenSolution()
    # generate solutions for N-Queens problem with given board_dim
    solver.solve(board_dim)
    solutions = [x for x in solver.solutions]
    
    def match_solution(query):
        '''
        utility function to match solutions against given query 
        '''
        match_set = []
        for sol in solutions:
            solution = sol.flatten()
            if np.sum(np.abs(solution-query))==num_missing:
                match_set.append(solution)
        return match_set
    
    # generate all possible enumeration to mask num_missing out of board_dim rows
    choice_list = generate_all_combination(board_dim, num_missing)
    choice_iter = choice_list
    print("Generated solutions and permutations")

    dataset = []
    query_set = set()
    for i in tqdm(range(len(solutions))):
        solution = solutions[i]
        if sample:
            choice_iter = choice_list[np.random.choice(range(len(choice_list)), size=5)]  # sample 5 possible maskings
        for choice in choice_iter:
            query = copy.deepcopy(solution)
            query[choice]=0  # mask out query 
            if tuple(query.flatten()) in query_set: # if query already included in query set then skip
                continue
            query = query.flatten()
            target_set = np.stack([x.flatten() for x in match_solution(query)]) # find solutions for the given query
            count = len(target_set)
            is_ambiguous = 0 if count==1 else 1
            query_set.add(tuple(query))
            dataset.append(dict(n=board_dim, query=query, target_set = target_set, count=count, is_ambiguous=int(is_ambiguous), qid =(i,tuple(choice))))
    return dataset
    

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = generate_data(args.board_size, args.num_missing, args.sample)
    if args.num_samples is not None:
        ch = np.random.choice(len(dataset),size=args.num_samples,replace=False)
        dataset = [dataset[i] for i in ch]
    if args.ofile=="":
        outfile = "nqueens_data_"+str(args.board_size)+"_"+str(args.num_missing)+".pkl"
    else:
        outfile = args.ofile
    with open(outfile,"wb") as f:
        pickle.dump(dataset,f)
