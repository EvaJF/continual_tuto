# Continual Learning Tutorial

Repository for PFIA 2025 - Tutorial on Continual Learning for Image Classification

__Contents__

1. [Preliminaries](#part1)
2. [The incremental learning framework](#part2)
3. [Fine-tuning-based incremental learning methods](#part3)
4. [Incremental learning methods with a fixed encoder](#part4)
5. [Further reading](#part5)


## 1. Preliminaries <a name="part1"></a>

__Set-up__

To get started with the tutorial, clone this repository and 
create a virtual environment with either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pip](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html).
The name of the environment is "cil".

```bash
git clone git@github.com:EvaJF/continual_tuto.git
cd continual_tuto
````
Conda version:
```bash
conda create --name cil --file requirements.txt
conda activate cil
```
Pip version:
```bash
python -m venv cil
source cil/bin/activate 
pip install -r requirements.txt
```

TODO : updated version of requirements files

__Repository structure__

The folders : 
* `ckp` : where we will store model checkpoints
* `data` : where we will store images and their vector representations computed with an encoder, e.g. when using the Food-101 dataset :
    - the data will be downladed under `data/food-101`, 
    - the image representations computed using a ResNet18 network pretrained on the ImageNet-1k dataset will be stored under `data/features/food-101/resnet18_in1k/`.
NB : Data will be downloaded in the next steps. 
* `logs` : where we will store the results of our experiments there
* `methods` : incremental learning algorithms
* `utils_tuto` : utility functions for the tutorial e.g. dataloaders and performance metrics.

The training scripts : 
* to do add list and description

__Basic training and inference pipeline of an image classification model__

<img src="media/pred_pipeline.png" alt="prediction pipeline">

See script `joint_expe.py` for an example : training a classifier on the MNIST dataset. This will be our starting point for the next experiments. 

TODO : detailed comment on joint expe
* dataset an data loader
* model, optimizer, scheduler
* training loop

## 2. The incremental learning framework <a name="part2"></a>

__Types of incremental learning__

<img src="media/IL_types.png" alt="Types of incremental learning">

__Class-incremental learning__

<img src="media/cil_principle.png" alt="CIL principle">

__Measuring performance__

TODO explain : 
- acc matrix 
- avg incr acc 
- avg forgetting

__Baselines__

*Joint training* is usually the high baseline.

```
python joint_expe.py
```

*Vanilla fine-tuning* is usually the low baseline. 

TODO comment example script on MNIST
* new data loader
* updating the fc layer
* evaluation

__Challenges__

* Catastrophic forgetting and the stability-plasticity balance
...

* Representation overlap
<img src="media/representation_overlap.png" alt="representation overlap">

* Representation drift
<img src="media/representation_drift.png" alt="representation drift">


## 3. Fine-tuning-based incremental learning methods <a name="part3"></a>

__Replay__

```bash
python replay_expe.py
```

*Memory buffer*

See the `Memory` class implemented under `utils_cil.dataset.py`.

Vary the number of samples in the replay buffer, look at the number of samples per class in the logs.

Comment on the sampling strategy. How could it be improved ? Implement a different sampling strategy. 

More on replay strategies : REFS TO ADD.

*Weighted cross-entropy loss*

Now replace the loss function with a weighted version. What happens ?

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

```bash
python bsil_expe.py
```

NB : The `BalancedCrossEntropy` class does not change the loss scaling per sample, but instead modifies the softmax distribution via logit adjustment. This is equivalent to shifting the logits using log class priors, and results in a prior-corrected prediction distribution. Thus, it is not equivalent to `CrossEntropyLoss(weight=...)`, which re-weights the loss per sample after softmax and is typically used to rebalance gradient contributions. In other words, `CrossEntropyLoss(weight=...)` is used to increase the loss for under-represented classes, whereas `BalancedCrossEntropy` is used to adjust predictions for class imbalance by biasing logits using prior frequencies.

__Further reading__

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
