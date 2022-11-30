import os
from utils.params import params

interest = ['HaHackathon', 'joker', 'headlines']
for i in range(3):

  os.system(f"python main.py -model encoder -mode predict -wm online -l ml\
      -bs 64 -df data/test.csv -output preds_batch2/{params.model_mult[i].split('/')[-1]} \
      -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml_{'_'.join(interest)}.pt\
      -desc cross_ling_from_engl -id {'Haha'}")

for i in range(3):

  os.system(f"python main.py -model encoder -mode predict -wm online -l ml\
      -bs 64 -df data/test.csv -output preds_batch2/{params.model_mult[i].split('/')[-1]} \
      -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml_HaHa.pt\
      -desc cross_ling_from_spa -id {'_'.join(interest)}")

  # os.system(f"\
  #    python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}\
  #    -id {'_'.join(interest)} -desc {'_'.join(interest)}\
  #     ")

# interest = ['HaHackathon', 'headlines']
# for i in range(3):
#   for j in range(2):

#       os.system(f"python main.py -model encoder -mode predict -wm online -l ml\
#          -bs 64 -df data/test.csv -output preds/{params.model_mult[i].split('/')[-1]} \
#           -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml_{interest[j]}.pt\
#           -desc {interest[j]}__all ")

# interest = ['HaHackathon', 'joker', 'headlines']
# for i in range(3):
#   for j in range(3):
#     for k in range(j+1, 3):

#       l = [m for m in range(3) if m not in[j, k] ][0]
#       os.system(f"python main.py -model encoder -mode predict -wm online -l ml\
#          -bs 64 -df data/test.csv -output preds/{params.model_mult[i].split('/')[-1]} \
#           -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml_{interest[j]}_{interest[k]}.pt\
#           -desc {interest[j]}_{interest[k]}--{interest[l]} -id {interest[l]}")


#interest = ['HaHackathon', 'joker', 'headlines']
#for i in range(3):
#  for j in range(3):
#    for k in range(j+1, 3):


#      os.system(f'\
#      python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}\
#      -id {interest[j]}_{interest[k]} -desc {interest[j]}_{interest[k]}\
#        ')

# for i in range(3):
#     os.system(f'python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}')

#for i in range(3):
#    os.system(f"\
#    python main.py -model encoder -mode predict -wm online -l ml  -bs 64 -df data/test.csv -output preds/{params.model_mult[i].split('/')[-1]} -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml.pt")

#for i in range(3):
#    os.system(f"\
#    python main.py -model encoder -mode predict -wm online -l ml  -desc backtrans -bs 64 -df data/test_back.csv -output preds/{params.model_mult[i].split('/')[-1]} -mi {i} -wp logs/{params.model_mult[i].split('/')[-1]}/{params.model_mult[i].split('/')[-1]}_ml.pt")


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