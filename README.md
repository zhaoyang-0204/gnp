# Penalizing Gradient Norm in Deep Learning

This is our work at ICML2022, which entitles "[Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning](https://arxiv.org/abs/2202.03599)", by Yang Zhao, Hao Zhang and Xiuyuan Hu.

### 1. Short intro

##### 1.1 Overview

Basically, to penalize gradient norm in training, an additional term regarding the gradient norm $||\nabla_{\theta} L(\theta)||_2$ will be added on top of the empirical loss, which gives that,

$$\begin{aligned}
L(\theta) = L_{\mathcal{S}}(\theta) + \lambda ||\nabla_{\theta} L_{\mathcal{S}}(\theta)||_2
\end{aligned}$$

Gradient norm is considered a property that could characterize the flatness of the surface. Roughly speaking, penalizing the gradient norm of loss motivates to bias the optimization to converge to flatter minima of loss surface, leading to better model generalization. 

##### 1.2 Practical gradient computation of gradient norm

Based on the chain rule, the gradient of gradient norm is,

$$\begin{aligned}
\nabla_{\theta} L(\theta) = \nabla_{\theta} L_{\mathcal{S}}(\theta) + \lambda \cdot \nabla_{\theta}^2 L_{\mathcal{S}}(\theta) \frac{\nabla_{\theta} L_{\mathcal{S}}(\theta)}{||\nabla_{\theta} L_{\mathcal{S}}(\theta)||}
\end{aligned}$$

Computing the gradient of this gradient norm term in a straightforward way will involve the full computation of Hessian matrix. We introduce Taylor expansion to approximate the multiplication between the Hessian matrix and vectors, giving that

$$\begin{split}
    \nabla_{\theta} L(\theta) & = \nabla_{\theta} L_{\mathcal{S}}(\theta) + \lambda \cdot (\frac{\nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||}) - \nabla_{\theta}L_\mathcal{S}(\theta)}{r}) \\
    & = (1 - \frac{\lambda}{r}) \nabla_{\theta} L_{\mathcal{S}}(\theta) + \frac{\lambda}{r} \cdot \nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||})
\end{split}$$
 
where <img src="https://render.githubusercontent.com/render/math?math=r"> is a small scalar value. So, we need to set two parameters for gradient norm penalty $\lambda$, one for the penalty coefficient  and the other one for <img src="https://render.githubusercontent.com/render/math?math=r">. For practical convenience, we will further set,

$$\begin{split}
 \nabla_{\theta} L(\theta) = (1 - \alpha) \nabla_{\theta} L_{\mathcal{S}}(\theta) + \alpha \cdot \nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||}), ~~~\alpha = \frac{\lambda}{r}
\end{split}$$

In particular, [SAM](https://github.com/google-research/sam) is a special implementation of this scheme, where $\alpha$ will always set equal to 1.

### 2. Training using this repo

* This repo is realized via the JAX framework. One should start by building the python environment listed in the requirements.txt file. 

* The **config** folder stores all the config flags and their default values. One could add extra custom flags in the files if needed. 

* The **model** folder stores the model architectures, currently including VGG, ResNet, WideResNet, PyramidNet and ViT. One could add extra model architectures according to the Flax model template. Do not forget to register the model using the function _register_model in the folder after adding your custom models. 
 
* The **ds_pipeline** folder is for providing dataset pipeline, which is mostly based on that in [SAM](https://github.com/google-research/sam) repo. Unlike that in SAM, ImageNet dataset in this repo uses the local data, not the downloaded tensorflow_datasets. One should specify the path to the local dataset folders, where the folder structure must be 
    ```
    ImageNet folder
    |
    └───n01440764
    │   │   *.JPEG
    │   
    └───n01443537
    |    │   *.JPEG
    ...
    ```

* The **optimizer** folder stores the optimizers, currently including SGD (Momentum) and Adam.  One could add extra custom optimizers in the files if needed.

* The **recipe** folder stores the *.sh files. Each .sh file is one launching file for training a specific model. One could run it using the bash command, e.g.

    ```
    bash wideresnet-cifar.sh
    ```
    
  If one want to deploy config directly (the config flag must be in the config file), one could run with
  
    ```
    python3 -m gnp.main.main  --config=the-train-config-py-file --working_dir=your-output-dir --config.config-anything-else-here
    ```



### 3. End

Corresponding results could be found in the paper. Currently, it only contains models trained from scratch on image classification tasks. We appreciate anyone that would share other results, report bugs or contribute in any way. Also, we will respond as soon as possible if anyone encountered any problem.


If you find this helpful, you could cite the paper as
```
@article{zhao2022penalizing,
        author  = {Yang Zhao and Hao Zhang and Xiuyuan Hu},
        title = {Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning},
        year = 2022,
        journal = {arXiv preprint arxiv:2202.03599}
        eprint = {2202.03599},
}

