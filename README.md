# RTS
**RTS (Robust Test Selection for Deep Neural Networks)** could be used to select an effective subset from massive unlabelled data, for saving the cost of DNN testing. 

## Installation

```
pip install -r requirements.txt
```

## The structure of the repository 

In the experiment, our method and all baselines are conducted upon `Keras 2.3.1` with `TensorFlow 1.13.1`. All experiments are performed on a `Ubuntu 20.04` with `four NVIDIA GeForce RTX 3090 GPUs`, one `12-core processor`, and `256GB memory`.

main_folder:

```
├── ATS "adaptive test selection method"
├── gen_data/ "load data"
├── gen_model/ "to get the model in experiment" 
├── selection_method/ "test select method"
├── utils/ "some tool functions"
├── mop/  "data mutation operators"
├── statistics_utils/	"statistics raw results"
├── exp_fault.py "RQ1"
├── exp_retrain_*.py "RQ2"
├── exp_utils.py "some experiment utils"
├── statistics_result "a interface to get the pictures and tables in experiment"
├── init.py "init dirs and download svhn data"
├── RQ3.py "RQ3"
├── motivation_2.py "The motivation for noise impact"
├── BestSolution.py "Robust Test Selection for Deep Neural Networks"
```

others:

```
├── result_raw/ "the raw output of experiment"
├── result/ "pictures and tables of experimental results"
├── data/ "svhn data"
├── dau/ "Data Augmentation"
├── model/ "keras model"
├── temp/ "temp files"
├── README.md
└── requirements.txt
```

## Usage

In all results, the "DeepDiv" refers to ATS method, The "DeepDAC" refers to our proposed method RTS.


If you want to reproduce our experiment:

1. initial  models and datasets

   - you candownload by this link

     link： https://pan.baidu.com/s/133DcG8ouJUrSDgAndVYuJg
     Extraction Code: rtse

   - or initial  by python files

     1. initial dirs , svhn data and models

        `python init.py`

     2. data augmentation

        `python -m gen_data.{MnistDau}/{CifarDau}/{FashionDau}/{SvhnDau}`

2. experiment

   - `python exp_retrain_cov.py`

     `python exp_retrain_rank.py`

     Here, we get the priority sequence of all selection methods.

   - `python exp_fault.py`

     Here, we get the information of fault number of all priority sequences

3. get results

   `python statistics_result.py`




