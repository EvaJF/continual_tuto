# Commented code - step by step

The script `joint_expe.py` provides a minimal pipeline for supervised image classification using the MNIST dataset and a custom convolutional neural network (CNN) built with PyTorch. 
It includes data preparation, model definition, training, validation, evaluation, and checkpointing. 
In the following, we comment the code in more details. 

## 1. Joint training
---

**Device Configuration**

* `get_device()` automatically selects GPU (`cuda`), Metal Performance Shaders (`mps` on Apple), or CPU.

```python
device = get_device()
print(f"Running on device: {device}")
```
* In GPU-accelerated computing, data and model must reside on the same device (host-device memory symmetry).
With `model.to(device)` and `tensor.to(device)`, we move objects to the chosen device (`cuda`, `mps`, or `cpu`).

---

**Preprocessing**

Here we perform a minimalistic transform by converting the input images to tensors and centering inputs (zero mean, unit variance). Normalization stabilizes and speeds up training. 

```python
trf = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])
```
* `transforms.Compose` chains multiple preprocessing steps.
* `.Normalize(mean, std)` standardizes each channel independently. Here, MNIST is grayscale, hence one channel.

---

**Downloading and Splitting the Data**

* `datasets.MNIST()` fetches and preprocesses the dataset (it's a convenient built-in PyTorch class). It will automatically download the data into your current working directory.

```python
train_set = datasets.MNIST(...)
test_set = datasets.MNIST(...)
val_set = datasets.MNIST(...)
```
* Using `train_test_split()` from `sklearn`, we split the original MNIST training set (60,000 images) into training (90%) and validation (10%) for model selection. NB : here we manually modify `.data` and `.targets` to make the train/val split, other option is to use PyTorch's `Subset` or `random_split`.


**Data Loading**

* In PyTorch, a DataLoader allows to iterate over batches. We shuffle the data during training. 
* You may want to modify the `batch_size` to better comply with your machine's capacities.

```python
train_loader = torch.utils.data.DataLoader(...)
```

---

**Model Instantiation**

* `myCNN` is a custom CNN model class with two convolutional layers and a fully connected layer before the final classification layer (see `utils_tuto.encoder`). Output dimension equals number of classes (10 digits).

```python
model = myCNN(nb_tot_cl, size_conv_1, size_conv_2, size_fc)
```
See the output of `print(model)` to inspect the architecture.
With `compute_num_params(model)` we report the number of parameters. Here all parameters will be optimized.

---

**Loss Function and Optimizer**

* Stochastic Gradient Descent (SGD) is used with momentum to accelerate convergence (smoothes updates).
* Cross-Entropy Loss is standard for multi-class classification; combines LogSoftmax + Negative Log Likelihood Loss (NLL).

```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
```

---

**Main Loop : Training and Validation**

* An **epoch** = one full pass through the training set.
```python
for epoch in range(EPOCHS):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        ...
```

* We start with `optimizer.zero_grad()` to clear old gradients.
* Warning : `train()` enables dropout/batch-norm updates; `eval()` disables them.
* Why use `no_grad()` at evaluation time ? it reduces memory use when we don't need backprop.
* `loss.backward()` computes new gradients.
* `optimizer.step()` updates parameters.

---

**Performance**

* Here we compute the classification accuracy i.e. number of correct predictions out of the total number of predictions.
* We print the confusion matrix to reveal misclassification patterns (will be useful in the next steps of the tutorial).

---

**Model Selection and Checkpointing**

* We save the model with the best validation performance.
* Note the deep copy of weights to ensure that we do not update the copied weights.

```python
if acc > best_acc:
    best_model_state = copy.deepcopy(model.state_dict())
```

* `state_dict()` stores parameter tensors.
* `torch.save()` serializes model.
* We first instantiate a model (hence, the architecutre). Then with `torch.load(..., weights_only=True)` we load a set of parameters.

---

**Testing Phase**

* Finally, we evaluate model generalization on an unseen test set.
We print the test accuracy and final confusion matrix.


Next, in `vanilla_expe.py` we present the same learning problem in an incremental learning set-up, e.g. instead of learning all 10 MNIST classes at once, learning them 2 by 2 by iteratively fine-tuning the model on the new subset of classes.

TODO : See explanations (here)[link_to_add]