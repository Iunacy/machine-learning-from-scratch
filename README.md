# Machine Learning from Scratch

A university student's learning repository, documenting my progression from basic machine-learning calculations in Python and NumPy toward using PyTorch.

## Why this repository exists

I started with small numerical examples to understand what training does: make predictions, calculate loss, compute gradients, and update parameters. My usual approach is:

**Concept → manual example → Python implementation → NumPy/vectorization → debugging → framework implementation**

The repeated and sometimes verbose implementations reflect that process. They are educational exercises, not a reusable ML library.

## Current learning stage

As of September 30, 2026, I am learning PyTorch after studying mainly through from-scratch implementations. Understanding the NumPy calculations helps me follow what autograd automates. I am also starting to work through the basic attention calculation used in Transformers, and am still building fluency with both.

## Progress and included implementations

The numbered folders broadly follow my learning progression.

| Stage | Included work |
| --- | --- |
| [01_linear_regression](01_linear_regression/) | Single gradient steps, repeated weight/bias updates, and multi-feature batch training with explicit Python loops |
| [02_logistic_regression](02_logistic_regression/) | Sigmoid, binary cross-entropy, vectorized gradients, scaling, and splitting exercises; [breast-cancer classification](02_logistic_regression/breast_cancer_logistic_regression.py) uses training-set statistics to standardize both splits |
| [03_multiclass_classification](03_multiclass_classification/) | Stable softmax, class-index cross-entropy, and [MNIST softmax regression](03_multiclass_classification/mnist_softmax_regression.py) |
| [04_neural_networks](04_neural_networks/) | Manual forward/backpropagation and ReLU; MNIST `784 → 128 → 10`; full-batch and mini-batch Digits exercises |
| [Digits early stopping](04_neural_networks/digits_nn_early_stopping.py) | `64 → 48 → 10`, shuffled batches, train/validation/test splits, validation-loss checkpointing, loss plots, confusion matrices, and per-class/macro/weighted/micro metrics |
| [05_pytorch](05_pytorch/) | First saved [FashionMNIST MLP](05_pytorch/fashion_mnist_mlp.ipynb), using Adam, dropout, and train/test evaluation |
| [06_transformers](06_transformers/) | A small [single-head scaled dot-product attention](06_transformers/scaled_dot_product_attention.py) exercise with learned query, key, and value projections |
| [experiments](experiments/) | Small metric and confidence exercises, mistake inspection, an [inverse-time learning-rate decay experiment](experiments/optimization/learning_rate_decay.ipynb), and incomplete snippets |

The vectorized logistic-regression exercise performs one parameter update; later files add training loops. Early handwritten datasets demonstrate calculations. High accuracy on those examples says little about generalization. Breast Cancer Wisconsin, Digits, MNIST, and FashionMNIST provide real datasets for further practice, but remain educational benchmarks here.

## Lessons from implementation

- Tracing matrix shapes helped me distinguish a layer's inputs, activations, and gradients when debugging backpropagation.
- Working through actual/predicted labels clarified confusion-matrix orientation and precision versus recall.
- The Digits pipeline separates validation from testing, copies the best weights, and uses validation loss for early stopping. This helped me look beyond a single accuracy score.
- In PyTorch, I learned to accumulate correct predictions across all batches instead of reporting only the final batch's accuracy. Matching Python interpreters and notebook kernels was also part of getting the environment working.

These are concepts I am practising, not claims of mastery.

## Selected results and settings

### NumPy MNIST

Historical results from the previous README; scripts are present, but saved logs are absent and scores have not been reproduced.

| Model | Train accuracy | Test accuracy | Current script settings |
| --- | --- | --- | --- |
| Softmax regression | ~90.46% | ~90.46% | 1,000 epochs, learning rate 0.1, full batch |
| One-hidden-layer network | 94.94% | 94.49% | 100 epochs, learning rate 0.01, batch size 256 |

Both scale pixels by 255 and use a nonstandard 80/20 MNIST split (`random_state=42`). Different training schedules prevent a controlled architecture comparison.

### First PyTorch FashionMNIST MLP

The saved notebook uses `784 → 256 → 128 → 10`, ReLU, dropout `p=0.3` after both hidden layers, cross-entropy, Adam (`lr=0.001`), batch size 256, and 20 epochs.

| Saved evaluation output | Accuracy | Loss |
| --- | --- | --- |
| Training set | 91.64% | 0.2181 |
| Test set | 88.61% | 0.3208 |

Saved notebook metrics may vary between runs: no random seed or separate validation set is configured.

An earlier comparison showed a smaller train/test gap with dropout, but slightly lower test accuracy and loss: a regularization tradeoff. The non-dropout baseline is not saved here, so that comparison cannot be reproduced from this repository.

## Tools and running an exercise

The NumPy exercises use Python, NumPy, scikit-learn for datasets/splits, and Matplotlib for plots:

```bash
python -m pip install -r requirements.txt
python 02_logistic_regression/breast_cancer_logistic_regression.py
python 04_neural_networks/digits_nn_early_stopping.py
python 06_transformers/scaled_dot_product_attention.py
```

The PyTorch exercises additionally require PyTorch. The notebook also requires torchvision with the v2 transforms API and a notebook environment such as Jupyter or VS Code. These are not included in `requirements.txt`. The notebook hardcodes `device="cuda"`, so its current version requires a compatible CUDA-enabled installation and NVIDIA GPU; the attention script runs on CPU.

Select the appropriate notebook kernel and run all cells in order from a fresh kernel. FashionMNIST is downloaded into a relative `data/` directory. MNIST scripts download from OpenML and can take longer to train, particularly the full-batch baseline. Dependencies are not version-pinned.

## Next steps and limitations

Next I plan to learn CNN fundamentals, continue practising PyTorch, and build from the attention calculation toward a fuller Transformer implementation. CNNs and a complete Transformer are not implemented here. Deployment, larger architectures, and advanced optimization remain outside my current experience.

Older exercises may contain mistakes, and some snippets are incomplete. Dataset scores describe individual study runs, not real-world reliability or production readiness. The breast-cancer exercise is for learning classification and evaluation, not medical diagnosis.
