import os

for i in range(3):
  for j in ['Haha', 'HaHackathon', 'joker', 'headlines']:
    os.system(f'\
    python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv -mi {i}\
    -id {j}\
      ')