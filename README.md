# [ICLR 2025] Tailoring Mixup to Data for Calibration

[![Paper](https://img.shields.io/badge/paper-arxiv.2311.01434-B31B1B.svg)](https://arxiv.org/abs/2311.01434)

**Authors:** [Quentin Bouniot](https://qbouniot.github.io/), [Pavlo Mozharovskyi](https://perso.telecom-paristech.fr/mozharovskyi/), [Florence d'Alché-Buc](https://perso.telecom-paristech.fr/fdalche/)

This is the official code for *Similarity Kernel Mixup* on classification tasks (see `classification` folder), and regression tasks (see `regression` folder). Each folder contains a separated README for information about setting up and running experiments. Code to reproduce experiments on toy dataset is included in the `toy_datasets` folder.

## Abstract

Among all data augmentation techniques proposed so far, linear interpolation of training samples, also called Mixup, has found to be effective for a large panel of applications. Along with improved predictive performance, Mixup is also a good technique for improving calibration. However, mixing data carelessly can lead to manifold mismatch, i.e., synthetic data lying outside original class manifolds, which can deteriorate calibration. In this work, we show that the likelihood of assigning a wrong label with mixup increases with the distance between data to mix. To this end, we propose to dynamically change the underlying distributions of interpolation coefficients depending on the similarity between samples to mix, and define a flexible framework to do so without losing in diversity. We provide extensive experiments for classification and regression tasks, showing that our proposed method improves predictive performance and calibration of models, while being much more efficient. 

## Taking into account similarity in Mixup

![Taking into account similarity in Mixup](./images/similarity_mixup.png)

Introducing similarity into the interpolation is more efficient and provides more diversity than explicitly selecting the points to mix.

## Similarity Kernel

**Batch-normalized and centered Gaussian kernel**

![Similarity Kernel design](./images/similarity_kernel.png)

- Amplitude $\tau_{max}$ governs the strength of the interpolation
- Standard deviation $\tau_{std}$ governs the extent of mixing
- Stronger interpolation between similar points and reduce interpolation otherwise

## Avoiding Manifold Mismatch

![Illustration on toy datasets](./images/toy_datasets_sk_mixup.png)


## Reference

If you find our work useful, please star this repo and cite:

```bibtex
@inproceedings{bouniot2025tailoring,
  title={Tailoring Mixup to Data for Calibration},
  author={Bouniot, Quentin and Mozharovskyi, Pavlo and d'Alch{\'e}-Buc, Florence},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
