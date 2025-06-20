# Continual Learning Tutorial

Repository for PFIA 2025 - Tutorial on Continual Learning for Image Classification
---

The aim of this tutorial is to introduce the stakes, challenges and main algorithmic approaches of continual learning through an overview and practical exercises. It will focus on recent algorithms for learning an image classification model incrementally.

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
In this tutorial, we use standard libraries for numerical computations and visualization (`numpy`, `matplotlib`) and deep learning (`torch`, `timm`).

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

__Repository structure__

The training scripts : 
* Fine-tuning-based incremental learning methods are illustrated using the MNIST dataset. See `joint_expe.py` for classic joint learning, `vanilla_expe.py` for vanilla fine-tuning across incremental learning steps, `replay_expe.py` for replay strategies, `distillation_expe.py` for knowledge distillation.
* Incremental learning methods that are based on a fixed encoder are illustrated on larger scale datasets (Flowers-102, Food-101). See `ncm_expe.py` for the Nearest Class Mean classifier (Rebuffi2017), `dslda_expe.py` for Deep Streaming LDA (Hayes), `fecam1_expe.py` for FeCAM with a common covariance matrix and `fecamN_expe.py`for FeCAM with one covariance matrix per class.

The folders : 
* `ckp` : where we will store model checkpoints
* `data` : where we will store images and their vector representations computed with an encoder, e.g. when using the Food-101 dataset :
    - the data will be downladed under `data/food-101`, 
    - the image representations computed using a ResNet18 network pretrained on the ImageNet-1k dataset will be stored under `data/features/food-101/resnet18_in1k/`.
NB : Data will be downloaded in the next steps. 
* `logs` : where we will store the results of our experiments there
* `methods` : incremental learning algorithms
* `utils_tuto` : utility functions for the tutorial e.g. dataloaders and performance metrics.


__Basic training and inference pipeline of an image classification model__

<img src="media/pred_pipeline.png" alt="prediction pipeline">

<u>Example:</u> In `joint_expe.py` we train a small convolutional neural network on the MNIST dataset. This will be our starting point for the next experiments.

```bash
python joint_expe.py
```

> `joint_expe.py` represents the ideal scenario in terms of final performance but assumes full availability of all classes from the beginning. 

See [here](comments/comments_joint.md) for more detailed explanations on joint training. 

___

## 2. The incremental learning framework <a name="part2"></a>

The term *continual learning* was notably used by Ring in 1997 [Ring, 1997], with the followng definition:
> « Continual learning is the constant development of increasingly complex behaviors, the
process of building more complicated skills on top of those already developed. »

The term *lifelong learning* is also used in the literature [Thrun, 1995]

In recent years, reasearch has focused on a particular form of continual learning, namely *incremental learning*. 

__Types of incremental learning__

The literature often distinguishes three types of incremental learning.
* Task-Incremental Learning (TIL)
* Domain-Incremental Learning (DIL)
* Class-Incremental Learning (CIL)

<img src="media/IL_types.png" alt="Types of incremental learning">

Since it does not require the notion of task identifiers, CIL can be seen as a more general framework than TIL and DIL. Its challenging nature may also explain the fact that it has received a lot of attention from the continual learning community. 

__Class-incremental learning__

<img src="media/cil_principle.png" alt="CIL principle">

TODO add formalism for the framework


__Catastrophic forgetting and the stability-plasticity balance__

A major issue faced by continual learning models is their tendency to
forget previously acquired information when confronted with new information. The phenomenon is called *catastrophic forgetting* or *catastrophic interference*, as it is caused by the "interference" of new information
with previous information [French, 1999; McCloskey and Cohen, 1989]. 

Some works in continual learning take inspiration from neuroscience. In particular, the terms of *stability* and *plasticity*, originally introduced to describe biological neural networks [Mermillod et al., 2013;
Abraham and Robins, 2005], are also found in the continual learning literature. Stability refers to the ability to retain past information, and plasticity to the ability to adapt an existing representation to take new information into account. Stability and plasticity are often presented as two complementary but competing aspects of learning.

> In `vanilla_expe.py`, we implement the basic framework for learning MNIST classification incrementally (e.g. learning the 10 classes in five steps of 2 classes each instead of learning all 10 classes together). 

* Run the script and compare the final performance with the performance obtained previously with the joint training strategy. What happened ?

```bash 
python vanilla_expe.py
```

If the parameters of a neural network are naively adjusted to the latest training data, a
procedure we refer to as vanilla fine-tuning here, the model will overfit to the latest training data and information that was useful for classifying the previous data may be forgotten. This corresponds to a high plasticity and low stability of the model, resulting in an abrupt degradation of the model's performance on past data.


* Compare the training loop of `vanilla_expe.py`with the training loop of `joint_expe.py`. In particular, take a look at the following aspects.

<u>The incremental scenario set-up: </u> Classes are introduced progressively (e.g. 2 at a time).
```python
nb_init_cl = 2
nb_incr_cl = 2
nb_tot_cl = 10
nb_steps = (nb_tot_cl - nb_init_cl) // nb_incr_cl
```

<u>The classes accessible at training time versus at test time are different: </u> 
```python
    train_cl = range(nb_init_cl + nb_incr_cl * (step - 1), nb_curr_cl)
    test_cl = range(nb_curr_cl)
```

The expanding classification layer: In joint training, the classifier has a fixed output dimension (10 classes).
In fine-tuning, the output layer is expanded at each step via `update_fc`, preserving weights of existing classes and appending  weights for the new classes.
At each step, the model is tested on all classes seen so far, simulating a realistic deployment scenario of the model in a changing environment.

<u>Performance evaluation:</u> We can use the final accuracy to compare with joint training. 
We can also compare different incremental learning algorithms based on two complementary indicators, namely :

Average incremental accuracy : 
```python
print("\nMACRO Avg incr acc: {:.2f}".format(np.mean(test_acc_list)))
print(
    "MICRO Avg incr acc: {:.2f}".format(
        np.average(test_acc_list, weights=nb_test_samples)
    )
)
```
Average forgetting : 
```python
def compute_forgetting(acc_mat, weights=None):
    """
    forgetting = final acc - max acc reached at any step for a given dataset
    """
    max_acc = np.max(acc_mat, axis=0)
    last_acc = acc_mat[-1]
    f = np.average(max_acc-last_acc, weights=weights)
    return f
```

* What could be improved in this vanilla fine-tuning experiment ? (Hint : learning rate, parameter selection).

NB : In CIL experiments, joint training is usually the high baseline, and 
vanilla fine-tuning is usually the low baseline. 


__Further remarks on the challenges of CIL__

* Representation overlap
<img src="media/representation_overlap.png" alt="representation overlap">

* Representation drift
<img src="media/representation_drift.png" alt="representation drift">

* Plasticity loss

___

## 3. Fine-tuning-based incremental learning methods <a name="part3"></a>

__Replay / Usage of a memory buffer__

Some CIL methods assume the availability of a fixed-size memory buffer, which stores a small subset of past training samples (Rebuffi et al., 2017b).

At any incremental step $s_i$​ with $i>1$, the training dataset is constructed as:

$D_i \cup B_{1:i−1}$ 

where $D_i$​ is the current batch of data and $B_{1:i−1} ⊆ D_1 \cup D_2 \cup ... D_{i-1}$​ represents the buffer content, containing a subset of images from previous training datasets.
This buffer is used to mitigate forgetting by enabling the model to rehearse on representative examples from earlier tasks.

> In `replay_expe.py` we improve upon the previous vanilla fine-tuning method by adding a memory buffer. Run the script and compare the confusion matrices across the incremental upadates. What happens ? Compare the accuracy and forgetting values.

```bash
python replay_expe.py
```

<u>Memory buffer: </u> See the `Memory` class implemented under `utils_cil.dataset.py`.

> Vary the number of samples in the replay buffer, look at the number of samples per class in the logs.

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

```bash
python distillation_expe.py
```

Going further : 
* KD on output features (LUCIR, BSIL). Implementation hint : see the Cosine Embedding loss function [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
* KD on intermediary representations (PODNet)


__Balanced cross-entropy loss__

TODO : KD on features

```bash
python bsil_expe.py
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

Prompt-based methods

Other methods

Useful repo

Surveys


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

```
@software{Feillet_Tutorial_on_Continual,
author = {Feillet, Eva and Popescu, Adrian and Hudelot, Céline},
license = {MIT},
title = {{Tutorial on Continual Learning for Image Classification}},
url = {https://github.com/EvaJF/continual_tuto},
version = {0.1}
}
```
