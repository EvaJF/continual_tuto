# Continual Learning Tutorial

Repository for PFIA 2025 - Tutorial on Continual Learning for Image Classification
---

The aim of this tutorial is to introduce the stakes, challenges and main algorithmic approaches of continual learning through an overview and practical exercises. It focuses on the main components of the recent algorithms proposed for learning an image classification model incrementally.

__Contents__

1. [Preliminaries](#part1)
2. [The incremental learning framework](#part2)
3. [Fine-tuning-based incremental learning methods](#part3)
4. [Incremental learning methods with a fixed encoder](#part4)
5. [Further reading](#part5)

___

## 1. Preliminaries <a name="part1"></a>

__Set-up__

To get started with the tutorial, clone this repository and 
create a virtual environment with either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pip](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html).
The name of the environment is "cil". 

```bash
git clone git@github.com:EvaJF/continual_tuto.git
cd continual_tuto
```

Conda version:
```bash
conda env create -f cil.yml
conda activate cil
```

Pip version:
```bash
python -m venv cil
source cil/bin/activate 
pip install -r requirements.txt
```
We will use standard libraries for numerical computations (`numpy`), visualization (`matplotlib`) and deep learning (`torch`).

__Repository structure__

This repository contains scripts for training an image classification model following different methods. 
* Fine-tuning-based incremental learning methods are illustrated using the MNIST dataset. See `joint_expe.py` for classic joint learning, `vanilla_expe.py` for vanilla fine-tuning across incremental learning steps, `replay_expe.py` for replay strategies, `distillation_expe.py` for knowledge distillation.
* Incremental learning methods that are based on a fixed encoder are illustrated on larger scale datasets (Flowers-102, Food-101). See `ncm_expe.py` for the Nearest Class Mean classifier (Rebuffi et al., 2017), `dslda_expe.py` for Deep Streaming LDA (Hayes et al., 2020), `fecam1_expe.py` for FeCAM with a common covariance matrix and `fecamN_expe.py`for FeCAM with one covariance matrix per class (Goswani et al., 2024).
Utility functions, e.g. dataloaders and performance metrics, can be found under `utils_tuto`. 

Additionally, running the scripts will create the following folders. 
* `ckp` : to store model checkpoints
* `data` : to store images and their vector representations computed with an encoder, e.g. when using the Food-101 dataset :
    - `data/food-101` will contain the images, 
    - `data/features/food-101/resnet18_in1k/` will contain the vector representations of the images computed using a ResNet18 network pretrained on the ImageNet-1k dataset.
* `logs` : to store experimental results.

NB : Data will be downloaded in the next steps. 


__Basic training and inference pipeline of an image classification model__

Given $\mathcal{X} \subset \mathbb{R}^d$ an input domain and $\mathcal{Y} \subset \mathbb{N}^{n}$ a set of labels, let us consider a dataset $D$ defined as a set of pairs of the form $(x,y) \in \mathcal{X} \times \mathcal{Y}$, where $x$ is a computerized representation of an image and $y$ is a label associated with the image. Here, a label $y$ represents the class of an image.
In the following, we consider the case of a multi-class classification task with $n$ classes where $n \geq 2$.

<img src="media/pred_pipeline.png" alt="prediction pipeline">

A supervised classification problem consists of building a mapping function $`\mathcal{M}_\theta`$ that tries to relate $x$ to $y$ as closely as possible, i.e. for each $(x,y)\in D$, to have $\mathcal{M}_\theta(x)~\approx y$. This mapping function is referred to as a model. In this tutorial, the models we consider are neural networks. 

An image classification model can typically be decomposed into an encoder $\phi : \mathcal{X} \rightarrow \mathbb{R}^H$, also called *feature extractor*, and a classifier $f : \mathbb{R}^H \rightarrow \mathbb{R}^{n}$, i.e. $\mathcal{M}_\theta = f \circ \phi$.
The encoder produces a compact vector representation of the input image, called an *embedding* (or a *feature vector*, or a *latent representation*) and the classifier assigns a class to this representation. 

> In `joint_expe.py` we train a small convolutional neural network to classify handwritten digits using the MNIST dataset. This will be our starting point for the next experiments.
Joint training represents the ideal scenario in terms of final performance but assumes full availability of all classes from one end to another of model training, a hypothesis that not always holds. 

* Execute the script. Analyse the performance report and identify the hyperparameters of the experiment. 

```bash
python joint_expe.py
```

See [here](comments/comments_joint.md) for more detailed explanations on this first experiment. 

___

## 2. The incremental learning framework <a name="part2"></a>

The term *continual learning* was notably used by Ring in 1997 (Ring, 1997), with the following definition: « Continual learning is the constant development of increasingly complex behaviors, the
process of building more complicated skills on top of those already developed. »
The term *lifelong learning* is also used in the literature (Thrun, 1995)

In recent years, reasearch has focused on a particular form of continual learning, namely *incremental learning*. 

__Types of incremental learning__

The literature often distinguishes three types of incremental learning.
* In Task-Incremental Learning (TIL), the goal is to progressively learn a series of (semantically) distinct tasks. In practice, each data sample has a task identifier. Example : learning to recognize hand-written digits, then learning to classify handmade doodles.
* In Domain-Incremental Learning (DIL): the structure of the problem is the same across the learning steps (e.g. same number of classes), but the input distribution changes. Example : learning to recognize digits written by children, when learning to recognize digits written by adults. 
* In Class-Incremental Learning (CIL), a growing number of classes must be recognized, without task identity. Example: learning to recognize hand-written digits from 0 to 4, then learning to recognize hand-written digits from 5 to 9. This is the setting we will focus on in the rest of this tutorial.

<img src="media/IL_types.png" alt="Types of incremental learning">

Since it does not require the notion of task identifiers, CIL can be seen as a more general framework than TIL and DIL. Its challenging nature may also explain the fact that it has received a lot of attention from the continual learning community. 

__Class-incremental learning__

<img src="media/cil_principle.png" alt="CIL principle">

The objective of CIL is to train a model that integrates all classes of a dataset whose examples arrive in a stream.  
We consider a sequential learning process composed of $T$ non-overlapping steps $s_1, s_2,\dots,s_T$. 
For $t \in [1, T]$, a step $s_t$ consists of learning from the examples contained in a dataset $D_t$. 
Each data set $D_t$ corresponds to a set $P_t$ of classes so that each learning example in $D_t$ uses a class belonging to $P_{t}$. 
Each class is only present in a single dataset.

__Catastrophic forgetting and the stability-plasticity balance__

A major issue faced by continual learning models is their tendency to
forget previously acquired information when confronted with new information. This phenomenon is called *catastrophic forgetting* or *catastrophic interference*, as it is caused by the "interference" of new information
with previous information (French, 1999; McCloskey and Cohen, 1989). 

Some works in continual learning take inspiration from neuroscience. In particular, the terms of *stability* and *plasticity*, originally introduced to describe biological neural networks (Mermillod et al., 2013), are also found in the continual learning literature. 
* Stability refers to the ability to retain past information.
* Plasticity to the ability to take new information into account. 

Stability and plasticity are often presented as two complementary but competing aspects of learning. 

__Vanilla fine-tuning__

> In `vanilla_expe.py`, we implement the basic framework for learning MNIST digit classification incrementally (e.g. learning the 10 classes in five steps of 2 classes each instead of learning all 10 classes together). 

```bash 
python vanilla_expe.py
```

* Run the script and compare the final test accuracy with the one obtained previously with the joint training strategy. 
* In the performance report, look at the matrix which coefficient $(i, j)$ displays $Acc_i^j$, the accuracy of model $\mathcal{M}_i$ on the test samples from $D_j$. What happened ?

__If the parameters of a neural network are naively adjusted to the latest training data, a
procedure we refer to as *vanilla fine-tuning*, the model will overfit to the latest training data and information that was useful for classifying the previous data may be forgotten.__ This corresponds to a high plasticity and low stability of the model, resulting in an abrupt degradation of the model's performance on past data.


* Compare the training loop of `vanilla_expe.py`with the training loop of `joint_expe.py`. In particular, take a look at the following aspects.

<u>Incremental scenario set-up: </u> Classes are introduced progressively (e.g. 2 at a time here).
```python
nb_init_cl = 2
nb_incr_cl = 2
nb_tot_cl = 10
nb_steps = (nb_tot_cl - nb_init_cl) // nb_incr_cl
```

<u>The classes accessible at training time and the classes accessible at test time are different: </u> 
```python
train_cl = range(nb_init_cl + nb_incr_cl * (step - 1), nb_curr_cl)
test_cl = range(nb_curr_cl)
```

<u>Classification layer: </u> In joint training, the classifier has a fixed output dimension (10 classes).
In fine-tuning, the output layer is expanded at each step via `update_fc`, preserving weights of existing classes and appending  weights for the new classes.
At each step, the model is tested on all classes seen so far.

<u>Performance evaluation:</u> We can use the final accuracy to compare with joint training. 
We can also compare different incremental learning algorithms based on two complementary indicators, namely average incremental accuracy and average forgetting.

The average incremental accuracy of a model trained over a $T$-step incremental process is defined as the average test accuracy of the model over the $T$ steps of the incremental process (rebuffi2017_icarl). 
We denote it by $A$ and compute it as 
$A = \frac{1}{T} \sum_{i=1}^T Acc_i^{1:i}$, 
where $Acc_i^{1:i}$ is the accuracy of the model $\mathcal{M}_i$ on test samples from $\bigcup_{j=1}^i D_j$, after performing the learning step $s_i$.

Implementation: in `vanilla_expe.py`, the variable `test_acc_list` contains the values $Acc_1^1, Acc_2^{1:2}, ... Acc_T^{1:T}$. We obtain the average incremental accuracy by computing the mean of this list. 

```python
print("\nAvg incr acc: {:.2f}".format(np.mean(test_acc_list)))
```

The average forgetting of a model trained over a $T$-step process is the average of the accuracy gaps $f_i, i \in \llbracket 1,T-1 \rrbracket$, computed for each data subset $D_i$, as the difference between the best accuracy achieved for $D_i$ at any step $s_k$, with $k \leq i$, by a model $\mathcal{M}_k$, and the accuracy of the final model $\mathcal{M}_T$ on $D_i$. 
We denote it by $F$ and compute it as: 
    $ F = \frac{1}{T-1} \sum_{i=1}^{T-1} f_i $,
where $f_i$ is the individual forgetting, computed as $f_i = \max_{i \leq k \leq T} Acc_k^i - Acc_T^i$

Implementation : In `vanilla_expe.py`, the coefficient $i, j$ of the `acc_mat` matrix contains the accuracy $Acc_i^j$ of model $\mathcal{M}_i$ on the test samples from $D_j$. 
We can also compute average forgetting at the class level. 

```python
max_acc = np.max(acc_mat, axis=0)
last_acc = acc_mat[-1]
f = np.average(max_acc-last_acc)
```

* What could be improved in this vanilla fine-tuning experiment ? (Hint : validation set, learning rate, parameter selection).

NB : In CIL experiments, joint training is usually the high baseline, and 
vanilla fine-tuning the low baseline. 


__Further remarks on the challenges of CIL__

Another illustration of the challenges of CIL

* Representation overlap when encountering new classes

<img src="media/representation_overlap.png" alt="representation overlap">

* Representation drift when updating the encoder 
<img src="media/representation_drift.png" alt="representation drift">

___

## 3. Fine-tuning-based incremental learning methods <a name="part3"></a>

__Replay / Usage of a memory buffer__

Some CIL methods assume the availability of a fixed-size memory buffer, which stores a small subset of past training samples (Rebuffi et al., 2017b).

At any incremental step $s_i$​ with $i>1$, the training dataset is constructed as
$D_i \cup B_{1:i−1}$ 
where $D_i$​ is the current batch of data and $B_{1:i−1} ⊆ D_1 \cup D_2 \cup ... D_{i-1}$​ represents the buffer content, containing a subset of images from previous training datasets.
This buffer is used to mitigate forgetting by enabling the model to rehearse on representative examples from earlier tasks.

> In `replay_expe.py` we improve upon the previous vanilla fine-tuning method by adding a memory buffer. Run the script and compare the confusion matrices across the incremental upadates. What happens ? Compare the accuracy and forgetting values.

```bash
python replay_expe.py
```

<u>Memory buffer: </u> See the `Memory` class implemented under `utils_cil.dataset.py`.

> Vary the number of samples in the replay buffer. How does it impact performance ?

<u>Sampling strategy</u> Comment on the sampling strategy. How could it be improved ? Implement a different sampling strategy. 

More on replay strategies : REFS TO ADD.

<u>Dealing with class imbalance:</u> Replace the loss function with a weighted version. What happens ? What other strategies could you implement to deal with class imbalance ?

```python
CE_weights = [ 1-train_count_dict[k]/len(train_loader.dataset) for k in range(nb_curr_cl)]
print(f"Class weights {CE_weights}")
CE_weights = torch.tensor(CE_weights, device=device)
loss_fn = nn.CrossEntropyLoss(weight = CE_weights)
```

__Knowledge distillation__

Knowledge distillation on logits (LwF style, following Hinton et al.)

Currently using [KL div](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) ?? TODO check LwF paper It was L2

```bash
python distillation_expe.py
```

Going further : 
* KD on output features (LUCIR, BSIL). Implementation hint : see the Cosine Embedding loss function [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
* KD on intermediary representations (PODNet)
* Balanced Cross-Entropy Loss

```bash
python feature_distil_expe.py
```

NB : The `BalancedCrossEntropy` class does not change the loss scaling per sample, but instead modifies the softmax distribution via logit adjustment. This is equivalent to shifting the logits using log class priors, and results in a prior-corrected prediction distribution. Thus, it is not equivalent to `CrossEntropyLoss(weight=...)`, which re-weights the loss per sample after softmax and is typically used to rebalance gradient contributions. In other words, `CrossEntropyLoss(weight=...)` is used to increase the loss for under-represented classes, whereas `BalancedCrossEntropy` is used to adjust predictions for class imbalance by biasing logits using prior frequencies.

__Further reading__

TODO add resources / biblio

___

## 4. Incremental learning methods with a fixed encoder <a name="part4"></a>

aka classifier-incremental learning 

__Get the data__

In this part of the tutorial, we assume the encoder to be fixed. Hence, for a given image, no need to compute the forward pass of the encoder multiple times. We compute it once and store it for reuse. 

*Option 1 : Use pre-computed image features*. Download pre-computed features following these steps. These features are obtained using a ResNet18 network pre-trained on ImageNet-1k.

```
wget LINK TO ADD features.tar.gz
tar -xf features.tar.gz
tree -L 2 features
```

*Option 2 : Choose your own dataset and/or encoder.*

We provide a script for extracting features with a choice of dataset, model architecture and pre-training dataset, e.g. the following command will compute image representations for the images of Food-101 using a ViT-Small network pre-trained on the LVD-142m dataset.

```
python ftextract --dataset food-101 --archi vits --pretrain lvd142m
```

__NCM__

```bash
python ncm_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102 
```

__DSLDA__

```bash
python dslda_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102 
```

__FeCAM__

```bash
python fecam1_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102 
python fecamN_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102
```

__Further reading__

RanPAC, FeTrIL...


## 5. Further reading / useful links <a name="part5"></a>

Varying the scenario 

Prompt-based methods

Surveys

Useful repos

____

Before closing this tutorial, you may wish to uninstall the virtual environment you created.

* Conda 

```bash
conda deactivate
conda env remove --name cil
```
* Pip

```bash
deactivate
rm -r cil/
```
____

Cite this repo 

Illustrations are all taken from  TODO CITE manuscript. 


```
@software{Feillet_Tutorial_on_Continual,
author = {Feillet, Eva and Popescu, Adrian and Hudelot, Céline},
license = {MIT},
title = {{Tutorial on Continual Learning for Image Classification}},
url = {https://github.com/EvaJF/continual_tuto},
version = {0.1}
}
```
