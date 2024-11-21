# 2-phase pseudo-labeling algorithm
## Graph experiment
You can replicate our experimental results by following the steps outlined below:
- step 1 : Run baseline and record pseudo labels
```
python train_Modis_MS.py --dataset cora --train_size 3 # baseline is MoDis
```
- step 2 : Run 2-phase algorithm
```
python train_Two_Phase.py --name modis --dataset cora --train_size 3
```