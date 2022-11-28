import os
from utils.params import params

for i in range(3):
  for j in ['Haha', 'HaHackathon', 'joker', 'headlines']:
    os.system(f'\
    python main.py -model encoder -mode predict -wm online -bs 64 -df data/test.csv -output preds/{params.model_mult[i]} \
    -id {j} -desc {j} -wp logs/{params.model_mult[i]}/{params.model_mult[i]}_{j}.pt\
      ')

# for i in range(3):
#   for j in ['Haha', 'HaHackathon', 'joker', 'headlines']:
#     os.system(f'\
#     python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}\
#     -id {j} -desc {j}\
#       ')