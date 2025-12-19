##  CAMGT: A Triplet Extraction Model for Swine Disease Knowledge Graphs


## Requirements
The dependency packages can be installed with the command:
```
conda env create -f environment.yml
```
## dataset
Place the dataset under the ./ directory.
## Usage
* **Get pre-trained BERT model**
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `./pre_trained_bert`.

* **Train and select the model**
```
python run.py  --cuda_id=1 --dataset=fold_1
python run.py  --cuda_id=1 --dataset=fold_2
python run.py  --cuda_id=1 --dataset=fold_3 
python run.py  --cuda_id=1 --dataset=fold_4
python run.py  --cuda_id=1 --dataset=fold_5
python run.py  --cuda_id=1 --dataset=fold_6
python run.py  --cuda_id=1 --dataset=fold_7
python run.py  --cuda_id=1 --dataset=fold_8
python run.py  --cuda_id=1 --dataset=fold_9 
python run.py  --cuda_id=1 --dataset=fold_10 
```

* **Evaluate on the test set**
```
python run.py  --cuda_id=1  --train=test

```
## Main results
![](./result_task.png)
The average results of our model throughout the three runs with various random seeds are reported here to demonstrate the reliability and generalizability of our model.
，
