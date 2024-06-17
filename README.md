# Gradient Regularization in Deep Learning

## Table of Contents
1. [Works Related](#works-related)
2. [Upgrade [2024.6.15]](#upgrade-2024615)
3. [Training using this repo](#training-using-this-repo)
4. [Short intro](#short-intro)

## Works Related

1. "[Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning](https://arxiv.org/abs/2202.03599)"[ICML2022], by Yang Zhao, Hao Zhang and Xiuyuan Hu.
2. "[When Will Gradient Regularization Be Harmful?](https://arxiv.org/abs/2406.09723)"[ICML2024], by Yang Zhao, Hao Zhang and Xiuyuan Hu.

## Upgrade [2024.6.15]

1. **JAX Framework Update**: Upgraded the training framework to the latest version (JAX 0.4.28).
2. **New Paper Implementation**: Integrated the implementation of our latest research paper into this repository.
3. **Additional Model Architectures:**: Included Swin and CaiT Transformer architectures in the model list.

## Training using this repo

* **Environment Setup**: This repository is built using the JAX framework. Begin by setting up the Python environment specified in the `requirements.txt` file.

* **Configuration**: The `config` folder contains all the configuration flags and their default values. You can add custom flags if needed by modifying these files.

* **Model Architectures**: The `model` folder includes various model architectures such as VGG, ResNet, WideResNet, PyramidNet, ViT, Swin and CaiT. To add custom models, follow the Flax model template and register your model using the `_register_model` function in this folder.

* **Dataset Pipeline**: The `ds_pipeline` folder provides the dataset pipeline, based primarily on the [SAM](https://github.com/google-research/sam) repository. Unlike SAM, this repo uses local ImageNet data instead of tensorflow_datasets. Specify the path to your local dataset folders, ensuring the folder structure is:
    ```
    ImageNet folder
    └───n01440764
    │   │   *.JPEG
    │
    └───n01443537
    │   │   *.JPEG
    ...
    ```

* **Optimizers**: The `optimizer` folder contains the optimizers, including SGD (Momentum), AdamW and RMSProp. You can add custom optimizers by modifying these files.

* **Training Recipes**: The `recipe` folder contains `.sh` files, each corresponding to a specific model's training script. To run a training script, use the following command:
    ```
    bash wideresnet-cifar.sh
    ```
  
  Alternatively, to deploy configurations directly (ensuring the config flag is in the config file), use:
    ```
    python3 -m gnp.main.main --config=the-train-config-py-file --working_dir=your-output-dir --config.config-anything-else-here
    ```


## Short intro

#### 1.Overview

Basically, gradient regularization (GR) could be understood as gradient norm penalty, where an additional term regarding the gradient norm $||\nabla_{\theta} L(\theta)||_2$ will be added on top of the empirical loss,

$$\begin{aligned}
L(\theta) = L_{\mathcal{S}}(\theta) + \lambda ||\nabla_{\theta} L_{\mathcal{S}}(\theta)||_2
\end{aligned}$$

Gradient norm is considered as a key property that could characterize the flatness of the loss surface. By penalizing the gradient norm, the optimization is encouraged to converge to flatter minima on the loss surface. This results in improved model generalization.


#### 2. Practical Gradient Computation of Gradient Norm

Based on the chain rule, the gradient of the gradient norm is given by:

$$
\nabla_{\theta} L(\theta) = \nabla_{\theta} L_{\mathcal{S}}(\theta) + \lambda \cdot \nabla_{\theta}^2 L_{\mathcal{S}}(\theta) \frac{\nabla_{\theta} L_{\mathcal{S}}(\theta)}{||\nabla_{\theta} L_{\mathcal{S}}(\theta)||}
$$

Computing the gradient of this gradient norm term directly involves the full computation of the Hessian matrix. To address this, we use a Taylor expansion to approximate the multiplication between the Hessian matrix and vectors, resulting in:

$$\begin{split}
    \nabla_{\theta} L(\theta) & = \nabla_{\theta} L_{\mathcal{S}}(\theta) + \lambda \cdot (\frac{\nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||}) - \nabla_{\theta}L_\mathcal{S}(\theta)}{r}) \\
    & = (1 - \frac{\lambda}{r}) \nabla_{\theta} L_{\mathcal{S}}(\theta) + \frac{\lambda}{r} \cdot \nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||})
\end{split}$$
 
where $r$ is a small scalar value. So, we need to set two parameters for gradient norm penalty $\lambda$, one for the penalty coefficient and the other one for $r$. For practical convenience, we will further set,


$$\begin{split}
 \nabla_{\theta} L(\theta) = (1 - \alpha) \nabla_{\theta} L_{\mathcal{S}}(\theta) + \alpha \cdot \nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||}), ~~~\alpha = \frac{\lambda}{r}
\end{split}$$


Notably, the [SAM](https://github.com/google-research/sam) algorithm is a special implementation of this scheme where $\alpha$ is always set to 1.


#### 3. **Be Careful** When using Gradient Regularization with Adaptive Optimizer

GR can lead to serious performance degeneration in the specific scenarios of adaptive optimization. 

##### Error Rate[Cifar10]
| Model | Adam | Adam + GR | Adam + GR + Zero-GR-Warmup
|----------|:----------:|:----------:|:----------:|
| ViT-Ti   | 14.82  | 13.92   | 13.61 |
| ViT-S   | 12.07  | `12.40` | **10.68** |
| ViT-B   | 10.83  | `12.36` | **9.42** |

With both our empirical observations and theoretical analysis, we find that the biased estimation introduced in GR can induce the instability and divergence in gradient statistics of adaptive optimizers at the initial stage of training, especially with a learning rate warmup technique which originally aims to benefit gradient statistics.

To mitigate this issue, we draw inspirations from the idea of warmup techniques, and propose three GR warmup strategies: $\lambda$-warmup, $r$-warmup and zero-warmup GR. ach of the three strategies can relax the GR effect during warmup course in certain ways to ensure the accuracy of gradient statistics. See paper for details.

## End

If you find this helpful, you could cite the papers as
```
@inproceedings{zhao2022penalizing,
  title={Penalizing gradient norm for efficiently improving generalization in deep learning},
  author={Zhao, Yang and Zhang, Hao and Hu, Xiuyuan},
  booktitle={International Conference on Machine Learning},
  pages={26982--26992},
  year={2022},
  organization={PMLR}
}

@inproceedings{zhaowill,
  title={When Will Gradient Regularization Be Harmful?},
  author={Zhao, Yang and Zhang, Hao and Hu, Xiuyuan},
  booktitle={Forty-first International Conference on Machine Learning}
}