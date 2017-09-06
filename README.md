# Retina-Net
Focal loss for Dense Object Detection

The code is unofficial version for RetinaNet in focal loss for Dense Object Detection. https://arxiv.org/abs/1708.02002

You can use the focal loss  in https://github.com/unsky/focal-loss

# usage
install mxnet v0.9.5

1. download the dataset in data/

2. download the params in https://onedrive.live.com/?authkey=%21AI3oSHAoAIbxAB8&cid=F371D9563727B96F&id=F371D9563727B96F%21102802&parId=F371D9563727B96F%21102787&action=locate

```
./init.sh
```

# train & test
```
python train.py --cfg kitti.yaml

python test.py --cfg kitti.yaml 
```




