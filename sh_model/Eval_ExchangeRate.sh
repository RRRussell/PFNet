#!/usr/bin/env bash

python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/exchange_rate.txt --save ../model/exchange_ho3_hw8.pt --horizon 3 --highway_window 8 
python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/exchange_rate.txt --save ../model/exchange_ho6_hw16.pt --horizon 6 --highway_window 16 
python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/exchange_rate.txt --save ../model/exchange_ho12_hw32.pt --horizon 12 --highway_window 32 
python ../train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data ../dataset/exchange_rate.txt --save ../model/exchange_ho24_hw8.pt --horizon 24 --highway_window 8 






