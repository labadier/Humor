#!/bin/bash

# python main.py -model makedata -mode merge
python main.py -model makedata -mode translate -tf data/train_inverted.csv -output train
python main.py -model makedata -mode translate -tf data/test_inverted.csv -output test

# #Train Models
# #monolingual - en
# python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l en -bs 64 -epoches 8 -df data/test.csv

# #monolingual - es
# python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l es -bs 64 -epoches 8 -df data/test.csv

# #multilingual
# python main.py -model encoder -mode train -lr 1e-5 -wm online -tf data/train.csv -l ml -bs 64 -epoches 8 -df data/test.csv


# #monolingual
# # Encode test - train
# python main.py -model encoder -mode encode -wm online -l en -bs 64 -df data/test.csv -output logs -wp logs/bertweet-base_1.pt
# python main.py -model encoder -mode encode -wm online -l es -bs 64 -df data/test.csv -output logs -wp logs/bert-base-spanish-wwm-cased_1.pt
# python main.py -model encoder -mode encode -wm online -l en -bs 64 -df data/train.csv -output logs -wp logs/bertweet-base_1.pt
# python main.py -model encoder -mode encode -wm online -l es -bs 64 -df data/train.csv -output logs -wp logs/bert-base-spanish-wwm-cased_1.pt

# # Predict test - train
# python main.py -model encoder -mode predict -wm online -l en -bs 64 -df data/test.csv -output logs -wp logs/bertweet-base_1.pt
# python main.py -model encoder -mode predict -wm online -l es -bs 64 -df data/test.csv -output logs -wp logs/bert-base-spanish-wwm-cased_1.pt
# python main.py -model encoder -mode predict -wm online -l es -bs 64 -df data/train.csv -output logs -wp logs/bert-base-spanish-wwm-cased_1.pt
# python main.py -model encoder -mode predict -wm online -l en -bs 64 -df data/train.csv -output logs -wp logs/bertweet-base_1.pt


# #multilingual
# # Encode test - train
# python main.py -model encoder -mode encode -wm online -l ml -bs 64 -df data/test.csv -output logs -wp logs/bert-base-multilingual-uncased-sentiment_1.pt
# python main.py -model encoder -mode encode -wm online -l ml -bs 64 -df data/train.csv -output logs -wp logs/bert-base-multilingual-uncased-sentiment_1.pt

# # Predict test
# python main.py -model encoder -mode predict -wm online -l ml -bs 64 -df data/test.csv -output logs -wp logs/bert-base-multilingual-uncased-sentiment_1.pt
# python main.py -model encoder -mode predict -wm online -l ml -bs 64 -df data/train.csv -output logs -wp logs/bert-base-multilingual-uncased-sentiment_1.pt
