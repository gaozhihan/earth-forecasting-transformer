# Earthformer Training on EarthNet2021x with auxiliary meso scale weather data

## Train Earthformer on EarthNet2021x
Run the following command to train Earthformer on EarthNet2021x dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_earthnet2021x.py --gpus 2 --cfg cfg.yaml --ckpt_name last.ckpt --save tmp_earthnet2021x
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_earthnet2021x.py --gpus 2 --pretrained --save tmp_earthnet2021x
```
Run the tensorboard command to upload experiment records
```bash
tensorboard dev upload --logdir ./experiments/tmp_earthnet2021x/lightning_logs --name 'tmp_earthnet2021x'
```
