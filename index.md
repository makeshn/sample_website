---
title: Home
layout: home
usemathjax: true
---



# Gradio Demo

 <iframe src="https://abidlabs-pytorch-image-classifier.hf.space"></iframe>
 
 
---
author:
- |
  Gowtham Ramesh, Kriti Goyal and Makesh Sreedhar\
  {gramesh4, kgoyal6, msreedhar}@wisc.edu
bibliography:
- ref.bib
date: November 2022
title: CS639 Project Proposal
---

# Introduction

Recently, deep learning models have shown remarkable performance in
different vision tasks ranging from classification, segmentation,
text-to-image generation, and more. Instead of modality-specific
architectures (CNN for vision, LSTM for NLP, etc.), most modern works
use transformer[@vaswani] networks that can work with
images[@vit; @robustvit], text[@devlin-etal-2019-bert] and even some
combination of multiple modalities. In this project, we look at
benchmarking inference times(FLOPS/image per sec) across different
vision architectures from the popular CNN-based ResNet, Vision
transformer, and some efficient sparse
[@wang2020linformer; @wu2021fastformer; @liu2021swin; @choromanski2021rethinking]
attention alternatives on medical image classification. We would also
like to analyze how the attention mechanism differs between different
architectures and its effect on downstream performance.

Edge computing in medical fields and radiology has tremendous potential,
and there has been a lot of interest from the medical and computer
vision community in utilizing the latest research developments to build
better devices. However, one trend we notice among all the research work
on efficient transformers is that there has been no comprehensive
comparison between various methods using the same configuration or task
setting. Many comparisons in such methods cannot be considered fair
because they use different training schemes or model configurations.
This makes the task of choosing a particular architecture difficult for
the stakeholder. In this work, we would like to alleviate this issue by
ensuring a fair comparison between different architectures and attention
mechanisms.

# Problem Statement

In this project, we plan to benchmark the classifier performance and
inference time (computational complexity) of different models (CNNs,
ViT, and efficient transformers) on a common medical classification
task. Previous studies on these various architectures, especially in
comparing quadratic self-attention with other efficient variants, have
had different training schemes or configurations, making it challenging
to identify trade-offs. We aim to alleviate this by conducting a
comprehensive study in which we keep the architectures consistent for
transformer-based models and only modify the attention mechanism to
capture the impact on downstream performance accurately.

# Datasets and Evaluation

The RadImageNet[@RadImageNet] dataset includes 1.35 million annotated
ultrasound, radiographs, and CT scans for several classification tasks
based on medical imaging. It is publicly available for research, and the
RadImageNet team approved our request to use this dataset.

After checking the class distributions of each dataset and fitting some
baseline models, we selected the Brain MRI dataset. This dataset
contains MRI scans of patients with brain injuries/anomalies - an
arteriovenous anomaly, white matter changes, etc. and scans of normal
brains. The dataset contains 10 classes - 9 types of brain injuries/
anomalies and normal brain scans. The dataset is also imbalanced, making
it a challenging benchmark to baseline our models. Table
[1](#brain-mri){reference-type="ref" reference="brain-mri"} shows the
class distribution of this dataset.

::: {#brain-mri}
  **Diagnosis**           **Count**
  ----------------------- -----------
  Acute infarct           513
  Arteriovenous anomaly   272
  Chronic infarct         2307
  Edema                   125
  Extra                   1259
  Focal flair hyper       751
  Intra                   1721
  Normal                  27374
  Pituatary lesion        83
  White matter changes    10266

  : Label Distribution of the Brain MRI dataset
:::

![Sample MRI images showing brain injuries. The bright white regions in
the second scan indicate a type of Intracranial
Hemorrhage.](images/brain_bleeds_mri.png){#fig:brain-bleed width="100%"}

For image classification - accuracy, AUC, sensitivity, and specificity /
Cohen Kappa are some standard evaluation metrics. Each has its own
metrics and depends on the class imbalance, importance of output
misclassification, interpretability, etc. For the midterm report, we
have baselined the models with evaluation top1 and top5 accuracy. Top1
accuracy measures the accuracy of the top prediction of the model while
Top5 measures whether the right class is predicted in the top5
predictions of the model. We will also include metrics that take class
imbalances into account for the final project submission.

We will use FLOPS (Floating point operations per second) and inference
latency to measure model efficiency during deployment. We will also
benchmark these models across different platforms - CPU machine with
multiple cores like Intel Xeon, a slightly older Nvidia GPU like
1080Ti/K80 (which doesn't have tensor cores), and a modern Nvidia GPU
like RTX 3090.

We choose the above efficiency metrics as FLOPs are not a proxy for
latency [@wang2020hat], i.e., a model with the same FLOPS has different
latencies on different hardware.

# Models

## CNN Baseline

#### Resnet-50

ResNet-50[@he2015deep] is a deep convolutional neural network trained on
more than a million images from the ImageNet database. The network is 50
layers deep and can classify images into 1000 different categories. It
is one of the first networks to propose using residual connections that
help the training process by allowing information to flow more freely
between layers. Residual connections help the model learn the identity
function and enable multiple layers to be stacked, allowing us to create
much deeper models.

## Vision Transformer

The ViT model uses a Transformer-like architecture to classify images.
An image is split into fixed-size patches, each patch is embedded,
position embeddings are added, and the resulting sequence of vectors is
fed to a standard Transformer encoder. The standard approach of adding
an extra learnable classification token to the sequence is used to
perform classification.

The self-attention mechanism in the transformer is used to calculate a
weighted sum of the input vectors, where the weight for each vector is
based on the dot product of the query vector with the key vector for
that vector.

This can be written mathematically as:

$$x^2$$
    
<span> $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$ </span>

where $Q$ is the query matrix, $K$ is the key matrix, $V$ is the value
matrix, and $d_k$ is the dimension of the keys.

## Efficient Transformers

#### Linformer

[@wang2020linformer]

Linformer is a linear transformer that breaks down the self-attention
mechanism into smaller, linear attentions (converts O($n^2$) in self
attention to O($n$) with linear attention). This allows the Transformer
model to avoid the self-attention bottleneck. The original scaled
dot-product attention is decomposed into multiple smaller attentions
through linear projections. This combination of operations forms a
low-rank factorization of the original attention.

$$\operatorname{LA}(\mathbf{q}, \mathbf{k}, \mathbf{v})=\operatorname{softmax}\left(\frac{\mathbf{q}\left[W_{\text {proj }} \mathbf{k}\right]^T}{\sqrt{d_k}}\right) W_{\text {proj }} \mathbf{v}$$

Here, $q$ is the query vector, $k$ is the key vector and $W_{proj}$ is
the projection matrix for the smaller self-attention spans.

#### XCIT

[@xcit]

This model uses a variant of self-attention known as cross-covariance
attention. It is a transposed version of self-attention that operates
across the feature dimension rather than across the tokens. The authors
of this architecture observed that using this form of attention led to
worse interaction between tokens and degraded the quality of
representations learned. To overcome this, XCiT introduced a local patch
interaction module (LPI) consisting of two convolution layers. The
attention mechanism used in this model is mathematically represented as

$$\operatorname{XCA}(\mathbf{q}, \mathbf{k}, \mathbf{v})=\left[\operatorname{softmax}\left(\frac{\|\mathbf{q}\|_2^T\|\mathbf{k}\|_2}{\tau}\right) \mathbf{v}^T\right]^T$$

# Milestones

1.  Oct 8, 2022 - Choose the medical image classification dataset and
    the right evaluation metric to use for our experiments **Completed**

2.  Oct 8, 2022 - Decide on the baseline CNN model, the full
    self-attention ViT model and the variants of efficient attention
    models we would like to test. **Completed**

3.  Oct 22, 2022 - Code and train the baseline CNN and ViT model.
    **Completed**

4.  Nov 10, 2022 - Code and train at least one variant of the efficient
    attention model. Submit the mid-term project report. **Completed**

5.  Nov 15, 2022 - Code and train any remaining models. **In progress**

6.  Nov 20, 2022 - Benchmark the performance of various architectures
    considered on a CPU and GPU machine and analyze the trade-off
    between performance and efficiency.

7.  Nov 30, 2022 - Visualize attention maps to compare how the attention
    mechanisms differ

8.  Dec 1 - Dec 15, 2022 - Report writing, getting the website ready and
    submission

# Results

![Top-1 Accuracy of the models on the dev
set](images/top_1_accuracy.pdf){#fig:top1}

![Top-5 Accuracy of the models on the dev
set](images/top_5_accuracy.pdf){#fig:top5}

For all of the models we have included as part of this report, we adjust
the model configuration (number of layers, feed-forward units, etc.) to
approximately 25M parameters. This allows us to conduct a fair
comparison of the models by changing the attention mechanisms while
keeping all other factors the same.

Fig [2](#fig:top1){reference-type="ref" reference="fig:top1"} shows the
top-1 accuracy of the various models. We see the performance order from
best to worst as linformer $>$ resnet50 $>$ ViT-tiny $>$ ViT-small $>$
xcit.

Fig [3](#fig:top5){reference-type="ref" reference="fig:top5"} shows the
top-5 accuracy of the various models.We find the performance order as
linformer $>$ ViT-tiny $>=$ ViT-small $>$ resnet50 $>$ xcit-tiny.

Hence we see the efficient attention come close to and sometimes exceed
the performance of the baseline ResNet and VIT models. Thus, future
avenues exist to leverage efficient attention models to reduce
computation.

For the final project, we plan to benchmark these models' flops and
inference times across various hardware to show performance vs.
inference latency. Additionally, we will implement other efficient
attention mechanisms - Performer[@performer],
Fastformer[@wu2021fastformer], and Swin Transformer[@liu2021swin].
