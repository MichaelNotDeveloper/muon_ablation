#!/usr/bin/env python3
python3 training.py \
  --mode sweep \
  --kinds flat_min flat_max uniform gaussian geometric_0.9 linear_decay_to_smax \
  --outdir runs \
  --seed 123 \
  --n 16 \
  --d_in 16 \
  --d_out 16 \
  --steps 500 \
  --s_min 1e-4 \
  --s_max 10.0 \
  --alpha 1.0 \
  --num_experiments 10 \
  --lrs 0.001 0.01 0.1
python3 training.py --mode plot --outdir runs --no_show