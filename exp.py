import os
from utils.params import params


# for i in range(3):
#     os.system(f'python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}')

for i in range(3):
    os.system(f"\
    python main.py -model encoder -mode predict -wm online -l ml  -bs 64 -df data/test.csv -output preds/{params.model_mult[i].split('/')[-1]} -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml.pt")

for i in range(3):
    os.system(f"\
    python main.py -model encoder -mode predict -wm online -l ml  -desc backtrans -bs 64 -df data/test_back.csv -output preds/{params.model_mult[i].split('/')[-1]} -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml.pt")


# for i in range(3):
#   for j in ['Haha', 'HaHackathon', 'joker', 'headlines']:
#     os.system(f"\
#     python main.py -model encoder -mode predict -wm online -l ml  -bs 64 -df data/test.csv -output preds/{params.model_mult[i].split('/')[-1]} \
#     -id {j} -desc {j} -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml_{j}.pt\
#       ")

# for i in range(3):
#   for j in ['Haha', 'HaHackathon', 'joker', 'headlines']:
#     os.system(f'\
#     python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}\
#     -id {j} -desc {j}\
#       ')