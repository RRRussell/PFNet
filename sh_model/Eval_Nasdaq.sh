#!/usr/bin/env bash

python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/nasdaq100_padding.csv --save ../model/nasdaq_ho3_hw4.pt --horizon 3 --highway_window 4 
python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/nasdaq100_padding.csv --save ../model/nasdaq_ho6_hw4.pt --horizon 6 --highway_window 4 
python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/nasdaq100_padding.csv --save ../model/nasdaq_ho12_hw4.pt --horizon 12 --highway_window 4 
python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/nasdaq100_padding.csv --save ../model/nasdaq_ho24_hw16.pt --horizon 24 --highway_window 16 






