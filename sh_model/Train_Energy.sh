#!/usr/bin/env bash

python ../train.py --Triplet_loss 0 --Test 0 --window 32 --batch_size 128 --data ../dataset/energydata_complete.txt --save ../model/energy.pt --horizon 3 --highway_window 4 
