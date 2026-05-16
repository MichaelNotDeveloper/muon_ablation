#!/usr/bin/env python3
python3 training.py \
  --mode sweep \
  --kinds flat_min flat_max \
  --outdir runs \
  --seed 123 \
  --n 128 \
  --d_in 128 \
  --d_out 128 \
  --steps 500 \
  --s_min 1e-4 \
  --s_max 10.0 \
  --alpha 1.0 \
  --num_experiments 10 \
  --lrs 0.001 0.01 0.1 \
  --w_star random \
  --reg_type nuclear \
  --reg_alpha 1e-3 \
  --grad_reg_type zero \
  --batch_size 32 
  #--w_star_s_min 0.1 \
  #--w_star_s_max 1 \
  #--w_star_alpha 1.0\
  # uniform gaussian geometric_0.9 linear_decay_to_smax \

python3 training.py --mode plot --outdir runs --no_show