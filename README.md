# One-of-Many Learning

Training and data generation codes for [Learning One-of-Many Solutions for Combinatorial Problems in Structured Output Spaces](https://openreview.net/forum?id=ATp1nW2FuZL).

For replicating the experiments exactly from the paper, check out our working directory [https://github.com/dair-iitd/1oML\_workdir](https://github.com/dair-iitd/1oML\_workdir).

We experiment with 2 base models: [Neural Logic Machines (NLM)](https://arxiv.org/abs/1904.11694) and [Recurrent Relational Networks (RRN)](https://arxiv.org/abs/1711.08028).

Our training code has been adapted from [google/neural-logic-machines](https://github.com/google/neural-logic-machines) and uses [Jacinle python toolbox](https://github.com/vacancy/Jacinle).


<p align="center">
  <img src="assets/framework.jpg" width="600">
  <br>
    <em>SelectR model for handling solution multiplicity.</em>
</p>

## Installation
Before installation you need 
* python 3
* numpy
* tqdm
* yaml
* pandas
* PyTorch >= 1.5.0 (should probably work with >=1.0 versions, tested extensively with >=1.5)


Clone this repository:

```
git clone https://github.com/dair-iitd/1oML --recursive
```

Install [Jacinle](https://github.com/vacancy/Jacinle) included in `third_party/Jacinle`. You need to add the bin path to your global `PATH` environment variable:

```
export PATH=<path_to_1oML>/third_party/Jacinle/bin:$PATH
```

Create a conda environment for 1oML, and install the requirements. This includes the required python packages
from both Jacinle and NLM. Most of the required packages have been included in the built-in `anaconda` package:

```
conda create -n nlm anaconda
conda install pytorch torchvision -c pytorch
```


Install [dgl](https://github.com/dmlc/dgl) for RRN. 


## Data Generation
This repo contains three tasks over which we experiment in 1oML setting - NQueens, Futoshiki and Sudoku. 
To generate data for training and evaluating the models follow the instruction below. Alternatively you can also download datasets from this [drive link](https://drive.google.com/drive/folders/1s9QZXJGeAzCRuVcS7bSo7vQooCDRcTZe?usp=sharing).

* NQueens
```
python datagen/nqueens_datagen.py --ofile [output pkl file] --board-size 10 --num-missing 5 --sample
```

* Futoshiki
```
python datagen/futoshiki_datagen.py --ofile [output pkl file] --num-samples 10000 --nthreads 4 --board-size 5 --num-missing 10 --num-constraints 5 --mode [train/test/val]
```

* Sudoku
```
# Compile Jsolve for solving Sudoku puzzles.
cd third_party/Jsolve
make

# Create multiple-solution and unique-solution pickle files.
python datagen/sudoku_datagen.py --ambfile [multi-solution output path] --unqfile [unique-solution output path] --ifile [input file containing unique-solution sudoku puzzles] --jsolve third_party/Jsolve/jsolve 

# Sample queries from multi-solution and unique-solution pickle files to have uniform distribution of givens.

python datagen/data_sampling_sudoku.py --ifile [multi-solution pkl path] --ufile [unique-solution pkl path] --ofile [output path] --num_samples 5000
```

## Training and evaluating models

* NQueens | Futoshiki

```
# Naive Baseline
jac-run trainer/train.py --task [nqueens|futoshiki] --nlm-residual 1 --use-gpu  --skip-warmup 0 --test-interval 1 --save-interval 20 --epochs 0 --train-number [10|5] --nlm-depth 30 --test-number-begin [11|6] -test-number-end [11|6] --model nlm --seed 42 --arbit-solution 0 --warmup-data-sampling rsxy --wt-decay 0.0001 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] 

# Random Baseline
jac-run trainer/train.py --task [nqueens|futoshiki] --nlm-residual 1 --use-gpu --skip-warmup 0 --epochs 0 --train-number [10|5] --nlm-depth 30 --test-number-begin [11|6] --test-number-end [11|6] --model nlm --seed 42 --arbit-solution 1 --warmup-data-sampling rs --wt-decay [0|0.0001|0.00001] --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file]

# Unique Baseline
jac-run trainer/train.py --task [nqueens|futoshiki] --nlm-residual 1 --use-gpu --skip-warmup 0 --epochs 0 --train-number [10|5] --nlm-depth 30 --test-number-begin [11|6] --test-number-end [11|6] --model nlm --seed 42 --arbit-solution 1 --warmup-data-sampling unique --wt-decay [0|0.0001|0.00001] --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file]

# Min Loss
jac-run trainer/train.py --task [nqueens|futoshiki] --nlm-residual 1 --use-gpu  --epochs 0 --train-number [10|5] --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --model nlm --seed 42 --arbit-solution 0 --warmup-data-sampling [rs|one-one|two-one|three-one|four-one] --wt-decay [0|0.0001|0.00001] --min-loss 1 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file]

#RL model
jac-run trainer/train.py --task [nqueens|futoshiki] --nlm-residual 1 --use-gpu --skip-warmup 1 --seed 42 --no-static [0|1] --copy-back-frequency [0,10] --lr-hot [0.001,0.0005] --latent-wt-decay [0|1e-04|1e-05] --wt-decay [0|1e-04|1e-05] --pretrain-phi 1 --rl-reward count --hot-data-sampling [rs|one-one|two-one|three-one|ambiguous] --epochs 200 --latent-model [nlm|det] --latent-annealing [1|0.5] --train-number [10|5] --nlm-depth 30 --test-number-begin [11|6] --test-number-end [11|6] --model nlm --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] --load-checkpoint [path to the pretrained model] 

```
* Sudoku
```
# Naive Baseline
jac-run trainer/train.py --task sudoku --use-gpu  --model rrn --batch-size 32 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 42 --arbit-solution 0 --warmup-data-sampling rsxy --wt-decay 0.0001 --grad-clip 5 --lr 0.001 --skip-warmup 0 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] 

# Random Baseline
jac-run trainer/train.py --task sudoku --use-gpu  --model rrn --batch-size 32 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 42 --arbit-solution 1 --warmup-data-sampling rs --wt-decay 0.0001 --grad-clip 5 --lr 0.001 --skip-warmup 0 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] 

# Unique Baseline
jac-run trainer/train.py --task sudoku --use-gpu  --model rrn --batch-size 32 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 42 --arbit-solution 0 --warmup-data-sampling unique --wt-decay 0.0001 --grad-clip 5 --lr 0.001 --skip-warmup 0 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] 

# Min Loss
jac-run trainer/train.py --task sudoku --use-gpu  --model rrn --batch-size 32 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 42 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.0001 --grad-clip 5 --lr 0.001 --skip-warmup 0 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] --min-loss 1 

#RL model
jac-run trainer/train.py --task sudoku --use-gpu --skip-warmup 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --seed 42 --no-static 1 --copy-back-frequency 0 --pretrain-phi 1 --rl-reward count --hot-data-sampling rs --epochs 200 --grad-clip 5 --latent-wt-decay 0.0001 --wt-decay 0.0001 --lr-hot 0.0001 --lr-latent 0.0005 --latent-model [conv|det] --latent-annealing [1,0.5] --train-number 9 --test-number-begin 9 --test-number-end 9 --dump-dir [path to output folder] --train-file [path to train file] --test-file [path to validation file] --load-checkpoint [path to the pretrained model] 

```

To evaluate the model on test file give test file path instead of validation file and add flag `--test-only`.

Check out further command line arguments and their description by running

```
jac-run trainer/train.py --help
```
