# cse493s_hw2

Homework 2

## To reproduce the results, run the following command:

### Environment Setup

Setup the environment using the following command:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Data

Training and validation data need to be downloaded from [Pile](https://the-eye.eu/public/AI/pile/). We used the first 300K samples of the training data for training and first 10K samples for validation. 


### Training

```
python hw2.py --wandb --optimizer=lion --batch_size 64 --lr 0.0005 --gamma 0.95 --step_size 200 --model_dim=512 --weight_decay 0.1 --epochs 5000
```

This will save the model in the `ckpts` directory. 

### Generation

Once, generated, use the following command to generate the output:

```
python use_model.py
```
