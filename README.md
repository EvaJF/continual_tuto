# Continual Learning Tutorial

Repository for PFIA 2025 - Tutorial on Continual Learning for Image Classification

__Contents__
* section links

## 1. Preliminaries 

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

## 2. The incremental learning framework

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


## 3. Fine-tuning-based incremental learning methods

__Replay__

__Knowledge distillation__

__Balanced cross-entropy loss__

__Further reading__

## 4. Incremental learning with a fixed encoder 

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

```
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