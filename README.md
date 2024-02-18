# UnMixMatch

Official implementation of our AAAI 2024 paper:

> [**Scaling Up Semi-supervised Learning with Unconstrained Unlabelled Data**](https://arxiv.org/abs/2306.01222)      
> Shuvendu Roy, Ali Etemad       
> *In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-24)*




### Data
- Labelled data: CIFAR-10, CIFAR-100, SVHN, STL-10 will be downloaded automatically.
- Download and organize imagenet100 follwoing in instructions at https://github.com/danielchyeh/ImageNet-100-Pytorch 

### Run the experiments

1. Modify the config file in `config/cifar10_40_0.yaml` as you need. Include your data directory for imagenet100 in the config file.
2. Run `python unmixmatch.py --c config/cifar10_40_0.yaml`

This settings will run UnMixMatch on CIFAR-10 with 40 labels per class and get and accuracy of 47.91&plusmn;1.1.

### Acknowledgement
We would like to thank the authors of the following repositories for releasing their code. The implementation of UnMixMatch is build over the implementation of ReMixMatch from this repository: https://github.com/TorchSSL/TorchSSL


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
