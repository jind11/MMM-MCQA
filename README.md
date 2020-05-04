# MMM-MCQA
Source code for our "MMM" paper at AAAI 2020: [Jin, Di, Shuyang Gao, Jiun-Yu Kao, Tagyoung Chung, and Dilek Hakkani-tur. "MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension." AAAI (2020).](https://arxiv.org/pdf/1910.00458.pdf). If you use the code, please cite the paper:

```
@article{jin2019mmm,
  title={MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension},
  author={Jin, Di and Gao, Shuyang and Kao, Jiun-Yu and Chung, Tagyoung and Hakkani-tur, Dilek},
  journal={arXiv preprint arXiv:1910.00458},
  year={2019}
}
```

## Requirements
### Python packages
- Pytorch

## Usage
1. All five MCQA datasets are put in the folder "data" and to unzip the RACE data, run the following command:
```
tar -xf RACE.tar.gz
```

2. To train the BERT model (including base and large versions), use the following command:

```
python run_classifier_bert_exe.py TASK_NAME MODEL_DIR BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS
```
Here we explain each required argument in details:
- TASK_NAME: It can be a single task or multiple tasks. If a single task, the options are: dream, race, toefl, mcscript, mctest160, mctest500, mnli, snli, etc. Multiple tasks can be any combinations of those above-mentioned single tasks. For example, if you want to train a multi-task model on the dream and race tasks together, then this variable should be set as "dream,race".
- MODEL_DIR: Model would be initialized by the parameters stored in this directory. 
- BATCH_SIZE_PER_GPU: Batch size of data in a single GPU.
- GRADIENT_ACCUMULATION_STEPS: How many steps to accumulate the gradients for one step of back-propagation.

One note: the effective batch size for training is important, which is the product of three variables: BATCH_SIZE_PER_GPU, NUM_OF_GPUs, and GRADIENT_ACCUMULATION_STEPS. In my experience, it should be at least higher than 12 and 24 would be great. 

3. To train the RoBERTa model (including base and large versions), use the following command:

```
python run_classifier_roberta_exe.py TASK_NAME MODEL_DIR BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS
```

4. To facilitate your use of this code, I provide the trained model parameters for some settings:

| Model Type        | Fine-tune steps           | Download Links  |
| ------------- |:-------------:| -----:|
| BERT-Base      | MNLI,SNLI->DREAM,RACE | [Link](https://drive.google.com/open?id=1EECS9na9PpX9CO_cCzYj9FDkiBvOpyxv) |
| BERT-Large      | MNLI,SNLI->DREAM,RACE | [Link](https://drive.google.com/open?id=1_kEU-26HGpn4kdLseBCTI9QE5xzln4FU) |
| RoBERTa-Large      | MNLI,SNLI->DREAM,RACE | [Link](https://drive.google.com/open?id=1Cz5p6RLuc8F15ABSwv65ctriR-Wi15A3) |
| BERT-Base      | MNLI,SNLI | [Link](https://drive.google.com/open?id=19IL9wLz4QiNJ-XPHusJ1OLH54qpq8Hjr) |
| BERT-Large      | MNLI,SNLI | [Link](https://drive.google.com/open?id=1VtNH4jA7L_vZvi_kKgkfFy_1n1ADueaO) |
| RoBERTa-Large      | MNLI,SNLI | [Link](https://drive.google.com/open?id=1D3p8IXfli0m5PRKb99iusjhROOI7hxwN) |
| BERT-Large      | RACE | [Link](https://drive.google.com/open?id=1y9vD5aIrobCXXXaSn46ISehMu3426tM3) |
| RoBERTa-Large      | RACE | [Link](https://drive.google.com/open?id=1qBX0GKEVK7UoaQ9dO4yYuRzAE2qfQcMs) |
