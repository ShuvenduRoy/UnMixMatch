# UnMixMatch

Official implementation of our AAAI 2024 paper:

> [**Scaling Up Semi-supervised Learning with Unconstrained Unlabelled Data**](https://arxiv.org/abs/2306.01222)      
> Shuvendu Roy, Ali Etemad       
> *In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-24)*

**Unconstrained Setting**	

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/image-classification-on-cifar-10-40-labels)](https://paperswithcode.com/sota/image-classification-on-cifar-10-40-labels?p=scaling-up-semi-supervised-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-34)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-34?p=scaling-up-semi-supervised-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-35)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-35?p=scaling-up-semi-supervised-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-29)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-29?p=scaling-up-semi-supervised-learning-with)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-30)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-30?p=scaling-up-semi-supervised-learning-with)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-33)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-33?p=scaling-up-semi-supervised-learning-with)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-svhn-8)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-svhn-8?p=scaling-up-semi-supervised-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-svhn-7)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-svhn-7?p=scaling-up-semi-supervised-learning-with)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-svhn-9)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-svhn-9?p=scaling-up-semi-supervised-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-stl-5)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-stl-5?p=scaling-up-semi-supervised-learning-with)

**OpenSet Setting**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-36)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-36?p=scaling-up-semi-supervised-learning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-37)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-37?p=scaling-up-semi-supervised-learning-with)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scaling-up-semi-supervised-learning-with/semi-supervised-image-classification-on-cifar-38)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-38?p=scaling-up-semi-supervised-learning-with)

### Data
- Labelled data: CIFAR-10, CIFAR-100, SVHN, STL-10 will be downloaded automatically.
- Download and organize imagenet100 following the instructions at https://github.com/danielchyeh/ImageNet-100-Pytorch 

### Run the experiments

1. Modify the config file in `config/cifar10_40_0.yaml` as you need. Include your data directory for imagenet100 in the config file.
2. Run `python unmixmatch.py --c config/cifar10_40_0.yaml`

This settings will run UnMixMatch on CIFAR-10 with 40 labels per class and get and accuracy of 47.91&plusmn;1.1.

### Acknowledgement
We thank the authors of the following repositories for releasing their code. The implementation of UnMixMatch is built over the implementation of ReMixMatch from this repository: https://github.com/TorchSSL/TorchSSL


## Citing UnMixMatch

If you think this toolkit or the results are helpful to you and your research, please cite our paper:

```
@inproceedings{UnMixMatch,
  title={Scaling Up Semi-supervised Learning with Unconstrained Unlabelled Data},
  author={Roy, Shuvendu and Etemad, Ali},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
