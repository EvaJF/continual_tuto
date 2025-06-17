# Continual Learning Tutorial

Repository for PFIA 2025 - Tutorial on Continual Learning for Image Classification

__Contents__
* section links

## 1. Preliminaries 

__Set-up__

To get started with the tutorial, clone this repository.

```bash
git clone git@github.com:EvaJF/continual_tuto.git
cd continual_tuto
````

Create a virtual environment with either [pip](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) or with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```bash
conda create --name cil --file requirements.txt
conda activate cil
```
or 
```bash
python -m venv cil
source cil/bin/activate 
pip install -r requirements.txt
```

We will use popular image datasets such as MNIST, Oxford-Flowers-102 and Food-101.
In the `data` folder, we will store images and their vector representations computed with an encoder, e.g. when choosing Food-101 dataset :
- the data will be downladed under `data/food-101`, 
- the image representations computed using a ResNet18 network pretrained on the ImageNet-1k dataset will be stored under `data/features/food-101/resnet18_in1k/`.
Data/features will be downloaded in the next steps. 

__Basic prediction pipeline__

<img src="media/pred_pipeline.png" alt="prediction pipeline">


## 2. The incremental learning framework

__Types of incremental learning__

<img src="media/IL_types.png" alt="Types of incremental learning">

__Class-incremental learning__

<img src="media/cil_principle.png" alt="CIL principle">

__Measuring performance__

TODO implement utility functions : 
- acc matrix 
- avg incr acc 
- avg forgetting

__Baselines__

*Joint training* is usually the high baseline.

```
python joint_expe.py
```

TODO example script on MNIST

*Vanilla fine-tuning* is usually the low baseline. 

TODO example script on MNIST

__Challenges__

* Catastrophic forgetting and the stability-plasticity balance


* Representation overlap
<img src="media/representation_overlap.png" alt="representation overlap">

* Representation drift
<img src="media/representation_drift.png" alt="representation drift">


## 3. Incremental learning with a fixed encoder 

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

## 4. Fine-tuning-based incremental learning methods

__methods to choose__

TODO expe on MNIST