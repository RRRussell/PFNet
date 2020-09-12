# PFNet

This repository is the official implementation of [Parallel Extraction of Long-term Trends and Short-term Fluctuation Framework for Multivariate Time Series Forecasting](https://arxiv.org/abs/2008.07730). 

## Requirements
- Python 3.8.3
- PyTorch 1.5.0

To install requirements:

```setup
pip install -r requirements.txt
```

In this study, all experiments were carried on a computer equipped with a  GTX 1650 with Max-Q design, Intel Core i7-9750H CPU, @ 2.60GHz, 16GB RAM. 

## Overview

#### Dataset

We conduct experiments on three benchmark datasets for multivariate time series forecasting tasks, this table shows dataset statistics:

| Dataset      | T                  | D       | L          |
| -------------|--------------------| --------| -----------|
| Exchange_rate|7588                |    8    |  1 day     |
| Energy       |19736               |    26   |  10 minutes|
| Nasdaq       |40560               |    82   |  1 minute  |

where `T` is the length of time series, `D` is the number of variables, `L` is the sample rate.

Dataset can be downloaded from [Exchange_rate](https://github.com/laiguokun/multivariate-time-series-data) (546.3KB), [Energy](https://github.com/smallGum/MLCNN-Multivariate-Time-Series/tree/master/data) (3.6MB), [Nasdaq](https://github.com/smallGum/MLCNN-Multivariate-Time-Series/tree/master/data) (21.4MB) and put them under `dataset` folder.
#### Preprocessing
We split the raw data into `train set`, `validation set` and `test set`, in the ratio of 6:2:2. 

In each set, consecutive time series with certain length of `window size` are sampled as a slice, which forms a forecasting unit. The window size is set to 32 for PFNet model. The slice window moves over the entire time series in the pace of 1 step each time. 

## Training

To train the model(s) in the paper, run this code (an example):

```train
python train.py --Triplet_loss 0 --Test 0 --window 32 --batch_size 128 --data dataset/exchange_rate.txt --save model/exchange_rate.pt --horizon 3 --highway_window 4 
```

## Evaluation

To evaluate the model in the paper, run this code (an example):

```eval
python train.py --Triplet_loss 1 --Test 1 --window 32 --batch_size 1 --data dataset/exchange_rate.txt --save model/exchange_ho3_hw8.pt --horizon 3 --highway_window 8 
```

## Pre-trained Models

You can download pre-trained models here: https://drive.google.com/file/d/12ewsFVVzz2GyXUQ6kppoevXmvaeiC6W5/view?usp=sharing and put them into model folder.

## Results

We train PFNet for 1000 epochs for each train option, and use the model that has the best performance on validation set for test. 

We use three conventional evaluation metrics to evaluate the performance of TEGNN model: Mean Absolute Error (**MAE**), Relative Absolute Error (**RAE**) and Empirical Correlation Coefficient (**CORR**), the following table shows the results:


| Dataset            | horizon | RSE    | RAE    | CORR   |
|--------------------| --------| -------| -------| -------|
|||||||
|                    |    3    |  0.0156| 0.0121 | 0.9813 |
|Exchange_Rate       |    6    |  0.0229| 0.0180 | 0.9732 |
|                    |    12   |  0.0332| 0.0268 | 0.9583 |
|                    |    24   |  0.0437| 0.0367 | 0.9386 |
|||||||
|                    |    3    |  0.1074| 0.0278 | 0.9610 |
|Energy 			 |    6    |  0.1159| 0.0381 | 0.9226 |
|                    |    12   |  0.1221| 0.0524 | 0.8562 |
|                    |    24   |  0.1312| 0.0714 | 0.7600 |
|||||||
|                    |    3    |  0.0008| 0.0007 | 0.9987 |
|Nasdaq              |    6    |  0.0009| 0.0009 | 0.9968 |
|                    |    12   |  0.0013| 0.0013 | 0.9931 |
|                    |    24   |  0.0020| 0.0019 | 0.9854 |

Examples with parameters to run different datasets are in `sh_model` folder: `Train/Eval_ExchangeRate.sh`, `Train/Eval_Energy.sh` and `Train/Eval_Nasdaq.sh`, in which specific hyper-parameters for each training/testing options are listed.

Code for baseline models are in `models` folder, the hyper-parameters used are listed in our paper [Parallel Extraction of Long-term Trends and Short-term Fluctuation Framework for Multivariate Time Series Forecasting](https://arxiv.org/abs/2008.07730). 

## Modify and Repeat

Once you've successfully run the baseline system, you'll likely want to improve on it and measure the effect of your improvements.

To help you get started in modifying the baseline system to incorporate your new idea -- or incorporating parts of the baseline system's code into your own system -- we provide an overview of how the code is organized:

1. [Baseline_models/X_CNN.py] - Code that defines how to use CNN to extract long-term trends.

2. [Baseline_models/delta_X_CNN.py] - Code that defines how to use CNN to extract short-term fluctuations.

3. [TripletNet.py] - The core PyTorch model code. If you want to change the overall structure of PFNet, it is recommended to start with this file.

4. [utils.py] - Code containing data preprocessing and other operations. Note that if the model is PFNet, there will be some differences in the data interface from other conventional models.

5. [Optim.py] - Code related to the implementation of the optimizer, such as SGD, Adam, etc.

6. [train.py] - The main driver script that uses all of the above and calls PyTorch to do the main training and testing loops.
