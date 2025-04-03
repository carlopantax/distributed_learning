# Distributed Learning

## Run training

### Install requirements

`pip install -r requirements.txt`

### Start training

`python -m torch.distributed.launch --nproc_per_node=2 train.py`

