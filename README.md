# Contrastive Complementary Labeling

This repository is the official implementation of ``Boosting Semi-Supervised Learning with Contrastive Complementary Labeling``.

The key contributions of this paper are as follows:

*  We propose a novel Contrastive Complementary Labeling (CCL) method that constructs reliable negative pairs based on the complementary labels, i.e., the classes that a sample does not belong to. Indeed, our CCL effectively exploits low-confidence samples to provide additional information for the training process.
*  We develop a complementary labeling strategy to construct reliable negative pairs. Specifically, for lowconfidence data, we first select multiple classes with the lowest probability as complementary labels. Then reliable complementary labels are applied to construct a large number of negative pairs, which greatly benefits the contrastive learning.
*  Extensive experiments on multiple benchmark datasets show that CCL can effectively improve the performance of existing SSL methods based on pseudo labels. Besides, under the label-scarce settings, CCL effectively unleashes the power of low-confidence samples. For example, in CIFAR-10 with 10, 20, and 40 labeled data, compared to FixMatch, CCL can improve the performance of FixMatch by 5.63%, 2.93%, and 2.43%, respectively.

## 1. Requirements

* To install requirements: 

```
pip install -r requirements.txt
```

## 2. Training

* The configuration of CCL methods can be found in the directory ```config```. By default, CIFAR10, SVHN, and STL-10 use a single GPU for training, while CIFAR-100 uses two GPUs for training. For example, to train CCL-FixMatch on CIFAR-10 with 40 labeled data, we can run this command:

```
python ccl_fixmatch.py --c config/ccl_fixmatch/cifar10/ccl_fixmatch_cifar10_40_seed0.yaml
```
To train CCL-FlexMatch on STL-10 with 20 labeled data, we can run this command:
```
python ccl_flexmatch.py --c config/ccl_flexmatch/stl10/ccl_flexmatch_stl10_20_seed0.yaml
```

## 3. Pretrained models

* We  release our CCL pretrained models [here](https://github.com/qinyideng/ccl/releases/tag/v0.1). The schema of our pretrained models includes: {'model', 'optimizer', 'scheduler', 'it', 'ema_model', 'best_eval_acc',  'best_eval_iter'}, where 'it' denotes current iterations. 'best_eval_acc' denotes the best top-1 accuracy. 'best_eval_iter' denotes the iterations when obtaining the best top-1 accuracy.

* CCL is helpful for FixMatch/FlexMatch for all benchmarks and CCL achieves better performance when the task contains more noise (i.e., fewer labels). 

* Under the label-scare setting, compare to FixMatch/FlexMatch, CCL-FixMatch/CCL-FlexMatch significantly improves the accuracy.

<table style="text-align:center">
    <tr>
        <td rowspan="2">Method</td>
        <td colspan="3">CIFAR-10</td>
        <td colspan="2">CIFAR-100</td>
        <td colspan="2">STL-10</td>
    </tr>
    <tr>
        <td>10</td>
        <td>20</td>
        <td>40</td>
        <td>200</td>
        <td>400</td>
        <td>10</td>
        <td>20</td>
    </tr>
    <tr>
        <td>CCL-FixMatch</td>
        <td>74.92 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar10_10_seed0_acc.74.92.pth">download</a></td>
        <td>89.98 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar10_20_seed0_acc.89.98.pth">download</a></td>
        <td>95.07 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar10_40_seed2_acc.95.07.pth">download</a></td>
        <td>43.83 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar100_200_seed0_acc.43.83.pth">download</td>
        <td>54.41 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar100_400_seed0_acc.54.41.pth">download</a></td>
        <td>47.76 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_stl10_10_seed0_acc.47.76.pth">download</a></td>
        <td>53.23 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_stl10_20_seed0_acc.53.23.pth">download</a></td>
    </tr>
    <tr>
        <td>CCL-FlexMatch</td>
        <td>94.83 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar10_10_seed0_acc.94.83.pth">download</a></td>
        <td>94.99 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar10_20_seed0_acc.94.99.pth">download</a></td>
        <td>95.12 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar10_40_seed0_acc.95.12.pth">download</a></td>
        <td>52.51 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar100_200_seed2_acc.52.51.pth">download</a></td>
        <td>62.20 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar100_400_seed2_acc.62.20.pth">download</a></td>
        <td>50.98 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_stl10_10_seed0_acc.50.98.pth">download</a></td>
        <td>58.36 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_stl10_20_seed2_acc.58.36.pth">download</a></td>
    </tr>
</table>

CCL can also achieve a certain accuracy gain effect under the more labeled data settings.
<table style="text-align:center">
    <tr>
        <td rowspan="2">Method</td>
        <td colspan="2">CIFAR-10</td>
        <td colspan="2">CIFAR-100</td>
    </tr>
    <tr>
        <td>250</td>
        <td>4000</td>
        <td>2500</td>
        <td>10000</td>
    </tr>
    <tr>
        <td>CCL-FixMatch</td>
        <td>95.18 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar10_250_seed0_acc.95.18.pth">download</a></td>
        <td>95.87 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar10_4000_seed1_acc.95.87.pth">download</a></td>
        <td>72.19 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar100_2500_seed1_acc.72.19.pth">download</a></td>
        <td>78.11 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_cifar100_10000_seed0_acc.78.11.pth">download</a></td>
    </tr>
    <tr>
        <td>CCL-FlexMatch</td>
        <td>95.33 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar10_250_seed0_acc.95.33.pth">download</a></td>
        <td>95.92 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar10_4000_seed2_acc.95.92.pth">download</a></td>
        <td>73.77 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar100_2500_seed2_acc.73.77.pth">download</a></td>
        <td>78.17 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_cifar100_10000_seed1_acc.78.17.pth">download</a></td>
    </tr>
</table>


<table style="text-align:center">
    <tr>
        <td rowspan="2">Method</td>
        <td colspan="3">STL-10</td>
        <td colspan="3">SVHN</td>
    </tr>
    <tr>
        <td>40</td>
        <td>250</td>
        <td>1000</td>
        <td>40</td>
        <td>250</td>
        <td>1000</td>
    </tr>
    <tr>
        <td>CCL-FixMatch</td>
        <td>71.38 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_stl10_40_seed1_acc.71.38.pth">download</a></td>
        <td>92.25 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_stl10_250_seed1_acc.92.25.pth">download</a></td>
        <td>94.13 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_stl10_1000_seed0_acc.94.13.pth">download</a></td>
        <td>98.04 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_svhn_40_seed0_acc.98.04.pth">download</a></td>
        <td>98.04 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_svhn_250_seed2_acc.98.04.pth">download</a></td>
        <td>98.11 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_fixmatch_svhn_1000_seed0_acc.98.11.pth">download</a></td>
    </tr>
    <tr>
        <td>CCL-FlexMatch</td>
        <td>76.34 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_stl10_40_seed0_acc.76.34.pth">download</a></td>
        <td>91.90 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_stl10_250_seed1_acc.91.90.pth">download</a></td>
        <td>94.39 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_stl10_1000_seed0_acc.94.39.pth">download</a></td>
        <td>96.69 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_svhn_40_seed0_acc.96.69.pth">download</a></td>
        <td>96.91 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_svhn_250_seed0_acc.96.91.pth">download</a></td>
        <td>95.64 <a href ="https://github.com/qinyideng/ccl/releases/download/v0.1/ccl_flexmatch_svhn_1000_seed1_acc.95.64.pth">download</a></td>
    </tr>
</table>



## 4. Citaiton 

* If you find our work inspiring or use our codebase in your research, please cite our work.

```
@article{deng2022boosting, 
  title={Boosting Semi-Supervised Learning with Contrastive Complementary Labeling}, 
  author={Deng, Qinyi and Guo, Yong and Yang, Zhibang and Pan, Haolin and Chen, Jian}, 
  journal={arXiv preprint arXiv:2212.06643}, 
  year={2022} 
}
```

## 5. Acknowledgements

* The project is developed based on [TorchSSL](https://github.com/TorchSSL/TorchSSL).
