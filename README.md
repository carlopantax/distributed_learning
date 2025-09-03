# Distributed Learning

## Run training

This readme file shows how to run the codes in this project in the `train_centralized`, `train_distributed` and `train_distributed_dynamic files`.

### Install requirements
In order to run the code, verify that libraries in requirements are installed, otherwise executes from terminal:

`pip install -r requirements.txt`

### Train Centralized

| Argument        | Type / Default                   | Description                                                                          |
| ---------------- | -------------------------------- | ------------------------------------------------------------------------------------ |
| `--optimizer`    | `str`, default: `sgdm`           | Optimizator-choice: `sgdm`, `adamw`, `lars`, `lamb`.                          |
| `--lr`           | `float`, default: `0.001`        | Learning rate (for `lars`/`lamb` it will be automatically scaled √k).       |
| `--weight_decay` | `float`, default: `1e-4`         | Weight decay.                                                                   |
| `--resume`       | flag, default: `False`           |If present, resume training from`--checkpoint` (model/optimizer/scheduler). |
| `--checkpoint`   | `str`, default: `checkpoint.pth` | Path to the checkpoint file to read/write status from.                       |
| `--batch_size`   | `int`, default: `128`            | Size of the training mini-batch.                                               |
| `--epochs`       | `int`, default: `150`            | Total number of training epochs                                                 |

#### Example:

```python train_centralized.py --optimizer lars --lr 0.001 --weight_decay 1e-4 --batch_size 512 --epochs 150 --resume --checkpoint runs/lars_bs512.pth```

### Train Distributed

| Argument                | Type / Default            | Description                                                                                               |
| ----------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `--tau`                 | `int`, default: `5`       | Local epochs per client before synchronization (LocalSGD period).                                         |
| `--batch_size`          | `int`, default: `128`     | Mini-batch size per client.                                                                               |
| `--lr`                  | `float`, default: `0.001` | Learning rate for the local optimizer (SGD).                                                              |
| `--weight_decay`        | `float`, default: `5e-4`  | weight decay used by local optimizers.                                                                 |
| `--world_size`          | `int`, default: `4`       | Number of simulated clients.                                                                              |
| `--resume`              | flag, default: `False`    | Reserved flag to resume training              |
| `--epochs`              | `int`, default: `150`     | Target total epochs (used to derive rounds together with `--tau`).                                        |
| `--use_global_momentum` | flag, default: `False`    | Apply BMUF-style global momentum when aggregating models.                                                 |
| `--use_slowmo`          | flag, default: `False`    | Apply SlowMo-style outer update when aggregating models. (Use at most one of the two outer optimizers.)|

#### Example:

```python train_distributed.py --tau 8 --world_size 4 --batch_size 128 --lr 0.001 --epochs 150 --use_global_momentum```

### Train Distributed with Dynamic Tau

| Argument                  | Type / Default            | Description                                                                                         |
| ------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------- |
| `--tau`                   | `int`, default: `16`      | Initial local epochs per client before sync (τ).                                                |
| `--min_tau`               | `int`, default: `4`       | Minimum allowed τ.                                                                                  |
| `--max_tau`               | `int`, default: `32`      | Maximum allowed τ.                                                                                  |
| `--patience`              | `int`, default: `3`       | Epochs without sufficient improvement before the controller considers syncing or adjusting τ.       |
| `--improvement_threshold` | `float`, default: `0.01`  | Minimum difference in validation accuracy to count as an improvement.                                        |
| `--batch_size`            | `int`, default: `128`     | Mini-batch size per client.                                                                         |
| `--lr`                    | `float`, default: `0.001` | Learning rate for local SGD.                                                                        |
| `--weight_decay`          | `float`, default: `5e-4`  | L2 weight decay for local optimizers.                                                               |
| `--world_size`            | `int`, default: `4`       | Number of simulated clients.                                                                        |
| `--resume`                | flag, default: `False`    |Reserved flag to resume training               |
| `--epochs`                | `int`, default: `150`     | Target total local epochs per client (also bounds the number of sync rounds).                       |

#### Example:

```python train_distributed_dynamic_tau.py --tau 16 --min_tau --max_tau 64 --patience 3 --improvement_threshold 0.005 --world_size 4 --batch_size 128 --lr 0.001 --epochs 150```




