15th Jan 
Results:
gpt encoder with gpt optim and no sudoku attn (0,0): 0.091727, 0.746000
gpt encoder with 1oml optim and no sudoku attn (1,0): 0.161871, 0.779200
gpt encoder with gpt optim and explicit sudoku attn (0,1): 0.100919, 0.824000 
***gpt encoder with 1oml optim and explicit sudoku attn (1,1): 0.197042, 0.802600  


RRN numbers: 09.13, 87.85 

Working: 
gpt encoder decoder with 1oml optim and explicit sudoku attn (1,1) and no loss on encoder (0): 0.346523, 0.710200 
gpt encoder decoder with 1oml optim and explicit sudoku attn (1,1) and no loss on encoder (1): 0.451239, 0.849800 

#
# Naive:
encoder only: gpt encoder with 1oml optim and explicit sudoku attn (1,1): 0.197042, 0.802600  
encoder-decoder: gpt encoder decoder with 1oml optim and explicit sudoku attn (1,1) and no loss on encoder (1): 0.451239, 0.849800 
                MS,,,,,,,OS

                
# Naive:
Model, MS,  OS
RRN, 0.0913, 0.8785
encoder only, 0.197042, 0.802600  
encoder-decoder, 0.451239, 0.849800 

# Random:
RRN, 0.1365, 0.8753
encoder only, 0.213030, 0.823600
encoder-decoder, 0.451439, 0.825400  

# Unique
RRN, 0.6639, 0.8919
encoder only, 0.734213, 0.770800
encoder-decoder, 0.817946, 0.830000 

# Minloss
encoder only, 0.783973, 0.806200
encoder-decoder, 0.784772, 0.777400


# cc-loss
encoder only, 0.793965, 0.824800

# IExplr
encoder only, 0.818745, 0.942000


#1. Naive 
#encoder only #with lr reduction (window 8 running, logs in logs/encoder_naive_reducelr1 (pid 12088) 
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rsxy --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/naive_baseline_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 1 --sudoku-attention-mask 1 > logs/encoder_naive_reducelr1 2>&1 &



#encoder-decoder (window 3)
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 1 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/random_baseline --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 



#1. Random 
#encoder only (window 2)
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 1 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/randome_baseline --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 0 --sudoku-attention-mask 1 

#with lr reduction (running ) (pid 11761)
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 1 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/random_baseline_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 1 --sudoku-attention-mask 1 


#encoder-decoder (window 3)
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 1 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/random_baseline --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 


#2. Unique only

#encoder only (window 4)
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling unique --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/unique_baseline --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 0 --sudoku-attention-mask 1 

# with lr reduction (running window 6)
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling unique --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/unique_baseline_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 1 --sudoku-attention-mask 1 

#encoder-decoder (window 5)
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling unique --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/unique_baseline --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 

#3. CC-loss (need to make code changes for this)

#encoder only (window 3) 
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/ccloss --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 0 --sudoku-attention-mask 1 --cc-loss 1 

#with reduce lr - 1 (window 5 running) 
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/ccloss_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 1 --sudoku-attention-mask 1 --cc-loss 1 

#encoder-decoder (window 4) 

jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/ccloss --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 --cc-loss 1 

#reducelr 1 (running in logs/decoder_ccloss_reducelr1) (pid 12431) 
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/ccloss_reducelr1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 1 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 --cc-loss 1 > logs/decoder_ccloss_reducelr1 2>&1 & 


#4. Min-loss (need to make code changes for this)

#encoder only (window 1)
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/minloss --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 0 --sudoku-attention-mask 1 --min-loss 1 

#with lr reduction (window 2 running)
jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/minloss_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --sudoku-attention-mask 1 --min-loss 1 


#encoder-decoder (window 2) 

jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/minloss --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 --min-loss 1 

# rerun encoder -decoder with reduction in lr at plateau (running window 4)
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 0 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --skip-warmup 0 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/minloss_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 1 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 --min-loss 1 


#5. I-Explr

# encoder only (window 5)
jac-run trainer/train.py --latent-model det --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.0001 --pretrain-phi 0 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 0 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/iexplr --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 0 --sudoku-attention-mask 1 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/minloss/checkpoints/checkpoint_best_warmup.pth

#with reduce lr (pid 7845)
jac-run trainer/train.py --latent-model det --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.0001 --pretrain-phi 0 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 0 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/iexplr_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 1 --sudoku-attention-mask 1 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/minloss_reducelr-1/checkpoints/checkpoint_best_warmup.pth > logs/encoder_ixplr_reducelr1


# encoder-decoder (on top of unique model) (window 6) 
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --latent-model det --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.0001 --pretrain-phi 0 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 0 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/iexplr_on_unique --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 5 --get-optim-from-model 0 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/unique_baseline/checkpoints/checkpoint_best_warmup.pth

#reduce lr on minloss pid 7727 
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --latent-model det --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.0001 --pretrain-phi 0 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 0 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/iexplr_on_minloss-reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 1 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 5 --get-optim-from-model 0 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/minloss_reducelr-1/checkpoints/checkpoint_best_warmup.pth > logs/decoder_ixplr_on_minloss_reducelr1

#6 SelectR
# encoder only (window 2 - failed)
## @TODO: there is some bug: grad of latent model is 0

(window 4)
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --latent-model conv --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.1 --pretrain-phi 1 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/debug_selectr --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 0 --sudoku-attention-mask 1 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/minloss/checkpoints/checkpoint_best_warmup.pth

#reduce lr (pid 7792)
jac-run trainer/train.py --latent-model conv --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.1 --pretrain-phi 1 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/selectr_reducelr-1 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --get-optim-from-model 0 --reduce-lr 1 --sudoku-attention-mask 1 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_sudoku_attn_1oml_optim/minloss_reducelr-1/checkpoints/checkpoint_best_warmup.pth > logs/encoder_selectr_reducelr1


# encoder-decoder (on top of unique model) (window 3 killed now) 
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --latent-model conv --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.1 --pretrain-phi 1 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/selectr_on_unique --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 5 --get-optim-from-model 0 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/unique_baseline/checkpoints/checkpoint_best_warmup.pth

#reduce lr (pid 8036)
CUDA_VISIBLE_DEVICES=1 jac-run trainer/train.py --latent-model conv --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.0001 --lr-latent 0.0005 --latent-wt-decay 0.1 --pretrain-phi 0 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 200 --epochs 100 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/selectr_on_minloss_reducelr-1_nopretrainphi --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 1 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 5 --get-optim-from-model 0 --load-checkpoint sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/minloss_reducelr-1/checkpoints/checkpoint_best_warmup.pth > decoder_selectr_on_minloss_reducelr1 2>&1 &

#7. # IExplr for encoder-decoder from the beginning (window 2)

CUDA_VISIBLE_DEVICES=0 jac-run trainer/train.py --latent-model det --no-static 1 --copy-back-frequency 0 --skip-warmup 1 --hot-data-sampling rs --lr-hot 0.001 --lr-latent 0.0005 --latent-wt-decay 0.0001 --pretrain-phi 0 --rl-reward count --task sudoku --use-gpu  --model gpt_encoder_decoder_sudoku --sudoku-num-steps 32 --batch-size 16 --test-batch-size 32 --epoch-size 1250 --train-number 9 --test-number-begin 9 --test-number-end 9 --warmup-epochs 0 --epochs 200 --seed 1027 --arbit-solution 0 --warmup-data-sampling rs --wt-decay 0.1 --grad-clip 1 --lr 0.001 --dump-dir sudoku_models/autoregressive/gpt_encoder_decoder_transformer_sudoku_attn_1oml_optim_loe-1/iexplr_from_start --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_train_e.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/sudoku_9_dev_e.pkl --reduce-lr 0 --sudoku-attention-mask 1 --loss-on-encoder 1 --test-begin-epoch 50 --get-optim-from-model 0 


24th Dec 2024

NOTE: have introduced hacks in  thutils_rl.py to make minloss work for encoder-decoder.1. passing inputs to instance accuracy. 2. not computing reward and target set accuracy..1.
--------------------------------------------------------------------------------------------------


