#!/bin/bash

# python main.py -model makedata -mode merge
# python main.py -model makedata -mode translate -tf data/train.csv -output train
# python main.py -model makedata -mode translate -tf data/test.csv -output test


python main.py -model encoder -mode train -lr 2e-5 -wm online -tf data/train.csv -l en -bs 64 -epoches 8 -df data/test.csv
python main.py -model encoder -mode encode -wm online -l en -bs 64 -df data/test.csv -output logs -wp logs/bertweet-base_1.pt
python main.py -model encoder -mode predict -wm online -l en -bs 64 -df data/test.csv -output logs -wp logs/bertweet-base_1.pt