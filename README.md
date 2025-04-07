# DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation

The official implement of paper [DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation](https://arxiv.org/abs/2403.13322).

## Requirements

```
python==3.11
numpy==1.25.2
torch==2.0.1
torchattacks==3.4.0
torchvision==0.15.2
scipy==1.11.1
```

## Train networks on distilled datasets from scratch

After acquiring distilled datasets with distillation methods, you can train your own models on them.

```shell
python train.py --dataset CIFAR10 --model ConvNet --model_num 5 --train_batch 256  --train_epoch 1000 --save_path ./result/convnet_cifar10_dc --optimizer sgd --distilled --distill_method DC --pt_path <path_to_distilled_dataset> [--dsa]
```

## Evaluate trained networks on clean test sets

If you want to evaluate test accuracy of a trained network on clean datasets, you can run scripts below.

```shell
python eval.py --dataset CIFAR10 --model ConvNet --data_path ./data/ --pt_path <path_to_weight>
```

## Perturb test images and evaluate robust accuracy

To evaluate robust accuracy of models, you need to prepare two configuration files before running the scripts. 

The `weights.yaml` contains paths to weight files of the trained models. For example:

```yaml
Path:
 - '../trained_models/dc_cifar10_ipc1.pt'
 - '../trained_models/dc_cifar10_ipc5.pt'
 - '../trained_models/dc_cifar10_ipc10.pt'
 - '../trained_models/dc_cifar10_ipc30.pt'
 - '../trained_models/dc_cifar10_ipc50.pt'
```

The `params.yaml` contains parameters for different attacking algorithms. For example:

```yaml
# eps, alpha, steps
VMIFGSM:
 - [0.0078431373,0.0078431373,10]
 - [0.0078431373,0.0078431373,5]
 
# eps, alpha, steps
PGD:
 - [0.0078431373,0.0078431373,10]
 - [0.0078431373,0.0078431373,5]
```

Please note that all parameter formats must strictly match the corresponding attacking algorithm.

To perturb test images and evaluate robust accuracy, you can run scripts below.

```shell
python robust.py --dataset CIFAR10 --model ConvNet --attacker FGSM --log_path <path_to_save_output> --weights <path_to_weights.yaml> --params <path_to_params.yaml> --repeat 5  
```

## Extension

### Introduce distillation methods

Our code provides a unified function for loading distilled datasets with different formats. If you want to extend a new distilled dataset format which is different from the existing ones, you can add the data loader into `load_distilled_dataset` in `datasets.py`. Specifically, you need to:

1. Add the name of your method to the name list:

   ```python
   methods = ['DC','DSA',...,'YOUR_METHOD']
   ```

2. Add a new `elif` branch and create your customized data loader:

   ```python
   elif method=='YOUR_METHOD':
       train_images = ...
       train_labels = ...
   ```

### Introduce attacking algorithms

If you want to add a new attacking algorithm, you can extend it from the predefined class `ATTACK` in `attack_utils.py` and rewrite the function `perturb`.  Specifically, you can:

1. Add the name of your attacking algorithm to the name list:

   ```python
   ATTACKERS = ['FGSM','PGD',...,'YOUR_ATTACK']
   ```

2. Define your attack:

   ```python
   class YOUR_ATTACK(ATTACKS):
       def __init__(self, model, params=[<DEFAULT PARAMETERS>], normalize=True, mean=None, std=None):
           super().__init__(model, params, normalize, mean, std)
           assert len(params)==<NUMBER OF PARAMETERS>, 'Parameters for YOUR_ATTACK invalid!'
       
       def perturb(self,images,labels):
           ...
           <ATTACKING ALGORITHM>
           ...
           adv_images = ...
           return adv_images
   ```

3. Add it to get_attacker`:

   ```python
   elif name=='YOUR_ATTACK':
       return YOUR_ATTACK
   ```


### Introduce network architecture

If you want to define a new network, you can define it in `models.py` and instantiate it in `get_network`. Specifically, you can:

1. Define your model:

   ```python
   class YOUR_MODEL(nn.Module):
       ...
       <Define your model>
       ...
   ```

2. Add it to `get_network`:

   ```python
   elif model=='YOUR_MODEL':
       net = YOUR_MODEL(<ARGS>)
   ```

## Acknowledgements

We referred to the code from the following repository:

- [DC-Bench](https://github.com/justincui03/dc_benchmark)
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [DatasetCondesation](https://github.com/VICO-UoE/DatasetCondensation)
- [SRe2L](https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L)
- [DREAM](https://github.com/lyq312318224/DREAM)

## Citation

```
@article{wu2025dd,
  title={DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation},
  author={Wu, Yifan and Du, Jiawei and Liu, Ping and Lin, Yuewei and Cheng, Wenqing and Xu, Wei},
  journal={IEEE Transactions on Image Processing}, 
  year={2025},
  volume={34},
  pages={2052-2066},
  doi={10.1109/TIP.2025.3553786}
}
```



