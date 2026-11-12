# Comparing 7 Classifiers on Handwritten Digits (5 vs 0/8/9)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarabaranes-dev/ml-classifier-comparison/blob/main/mnist_rework.ipynb)

A reproducible comparison of seven scikit-learn classifiers on a binary handwritten-digit task: recognising the digit **5** among the digits **0, 8 and 9**. Each model is tuned with cross-validation on the training data only and evaluated on 10 random stratified train/test splits.

![F1 score of the 7 classifiers over 10 splits](f1_boxplot.png)

## Background

This project started in the *Data Mining and Machine Learning* course at the Universitat de València (Erasmus+, 2025), as a team project of two students. The original experiment used a course dataset (`BigDigits.mat`) that is no longer available.

In 2026 I re-ran the experiment on a public dataset so that anyone can reproduce it:

- The original notebook printed the number of images per digit in BigDigits: `980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009`. These are exactly the class counts of the **MNIST test set** (10,000 images, 28×28 pixels), so the rework uses that set, downloaded from OpenML.
- The task is the same as in the original run: digit 5 against digits 9, 8 and 0.
- The evaluation protocol was redesigned to fix some methodological issues of the original (see [What changed from the original](#what-changed-from-the-original)).

The original course notebook is kept unchanged in [`ml_classifier_comparison.ipynb`](ml_classifier_comparison.ipynb) for reference. It cannot be run without the original data file.

## Data

| | |
|---|---|
| Source | MNIST (`mnist_784` on OpenML), last 10,000 rows = official test set |
| Images used | 3,855 (digits 0, 5, 8, 9) |
| Positive class (label 1) | digit 5: 892 images |
| Negative class (label 0) | digits 0, 8, 9: 2,963 images |
| Features | 784 pixel values, scaled to [0, 1] |

The classes are imbalanced (about 1:3), so the main metric is the **F1 score of the positive class (digit 5)**. ROC AUC and accuracy are also reported.

## Method

1. **10 random stratified train/test splits** (70% / 30%, `random_state=42`). All models are evaluated on the same 10 splits.
2. **Hyperparameter tuning on the training set only**: `GridSearchCV` with 5-fold cross-validation, optimising F1. The test set of each split is used once, to score the selected model.
3. **Preprocessing inside the pipeline**: models that need it use a `StandardScaler` (and QDA a `PCA`) fitted on the training folds only, so no information leaks from the test set.
4. **Results** are reported as mean ± standard deviation over the 10 splits.
5. **Grid boundaries**: when the best value was at the edge of a grid, the grid was extended once in that direction and then frozen.

| Model | Pipeline | Grid searched | Most frequent choice (out of 10 splits) |
|---|---|---|---|
| k-NN | – | `n_neighbors` 1, 3, 5, 7, 9, 11; `weights` uniform, distance | k=3 (5), k=5 (4), k=7 (1); always uniform |
| Perceptron | scaler | `alpha` 1e-4, 1e-3, 1e-2; `penalty` none, l2, l1 | alpha=1e-3, l1 (7) |
| SGD | scaler | `loss` hinge, log_loss; `alpha` 0.1, 1, 10 | hinge, alpha=1 (9) |
| Logistic Regression | scaler | `C` 0.001, 0.01, 0.1, 1 | C=0.1 (8) |
| LDA | – (solver `lsqr`) | `shrinkage` auto, 0.1, 0.5, 0.9 | 0.1 (5), 0.5 (5) |
| QDA | PCA | `n_components` 50, 100, 150; `reg_param` 0.01, 0.1, 0.5 | 150 (5), 100 (4); reg_param=0.1 (9) |
| MLP (1 hidden layer, early stopping) | scaler | `hidden_layer_sizes` 100, 200; `alpha` 1, 10 | 200 units (8); alpha=1 (7) |

## Results

| Model | Accuracy | F1 (class 5) | ROC AUC |
|:--|:--|:--|:--|
| **QDA (on PCA features)** | 0.989 ± 0.003 | **0.975 ± 0.006** | 0.999 ± 0.000 |
| **k-NN** | 0.986 ± 0.003 | **0.970 ± 0.006** | 0.995 ± 0.001 |
| MLP | 0.979 ± 0.005 | 0.954 ± 0.011 | 0.993 ± 0.006 |
| Logistic Regression | 0.957 ± 0.004 | 0.907 ± 0.009 | 0.981 ± 0.004 |
| LDA | 0.945 ± 0.004 | 0.884 ± 0.009 | 0.975 ± 0.004 |
| Perceptron | 0.941 ± 0.007 | 0.876 ± 0.014 | 0.964 ± 0.005 |
| SGD | 0.940 ± 0.006 | 0.857 ± 0.016 | 0.973 ± 0.006 |

Confusion matrices of the two best models on one test split (split 1 of 10, 1,157 images):

![Confusion matrices of QDA and k-NN](confusion_matrices.png)

On this split QDA makes 8 errors (7 false alarms, 1 missed 5) and k-NN makes 14 (9 false alarms, 5 missed 5s).

## Conclusion

- **QDA on PCA features and k-NN are the best models**, with equivalent F1 scores: the difference (0.975 vs 0.970) is smaller than the variation between splits.
- The **linear models** (Logistic Regression, LDA, Perceptron, SGD) are clearly behind (F1 0.86–0.91). A single linear boundary is not enough to separate the many ways of writing a 5 from the other digits, while k-NN, QDA and the MLP can model non-linear boundaries.
- **QDA went from the worst to the best model.** In the original run it was used on the raw 784 pixels, where it cannot estimate a full covariance matrix per class (about 620 training images of the digit 5 against 784 features), and it scored close to chance. Reducing the data to 50–150 PCA components first makes the estimate possible.

## What changed from the original

| Original (2025) | Rework (2026) |
|---|---|
| Hyperparameters chosen by looking at test-set scores | Tuning with cross-validation on the training set only |
| MLP and linear models trained on unscaled pixels | `StandardScaler` inside a pipeline |
| QDA on 784 raw pixels (AUC ≈ 0.51) | PCA before QDA, with regularisation |
| F1 computed on the majority class (digits 0, 8, 9) | F1 computed on the digit 5 |
| 3–5 splits, the "other" digits chosen at random without a seed | 10 seeded stratified splits, fixed digits |
| Private dataset, now lost | Public dataset (MNIST via OpenML) |

## Limitations

- The rework uses the MNIST test set, which has the same class counts as BigDigits. Whether the two files are identical (image order, pixel scaling) cannot be checked, because the original file is lost.
- One binary task only (5 vs 0/8/9); the results do not cover the full 10-class problem.
- The 10 splits are drawn from the same 3,855 images and overlap, so the standard deviations show stability rather than formal confidence intervals.
- The confusion matrices refer to a single split and are an example, not a general result.

## How to run

**On Google Colab (tested):** click the *Open in Colab* badge above, then *Runtime → Run all*. The full run takes about 30–40 minutes on a free Colab instance and needs internet access to download MNIST.

**Locally:**

```bash
git clone https://github.com/sarabaranes-dev/ml-classifier-comparison.git
cd ml-classifier-comparison
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook mnist_rework.ipynb
```

All splits and models use fixed random seeds, so a full re-run gives the same numbers. The notebook runs SGD, QDA and MLP twice: the second run (extended grids) is the one reported above.

## Repository structure

```
├── mnist_rework.ipynb             # the reproducible experiment (2026)
├── ml_classifier_comparison.ipynb # original course notebook (2025, needs BigDigits.mat)
├── f1_boxplot.png                 # F1 over the 10 splits
├── confusion_matrices.png         # QDA and k-NN on split 1
└── requirements.txt
```

## Tech stack

Python, scikit-learn, NumPy, pandas, Matplotlib, Google Colab.

## My role

- **Original course project (2025, team of 2):** we divided the seven models between us, each tuned their own models, and then we compared the best configurations together and chose the final model.
- **Rework (2026, individual):** I identified the replacement dataset, redesigned the evaluation protocol (cross-validated tuning, scaling and PCA pipelines, 10 seeded splits), ran all seven models and wrote the analysis.

---

## 🇮🇹 In italiano

Confronto riproducibile di sette classificatori scikit-learn su un problema binario di cifre scritte a mano: riconoscere il **5** tra le cifre **0, 8 e 9**.

Il progetto nasce dal corso *Data Mining and Machine Learning* all'Universitat de València (Erasmus+, 2025), lavoro di gruppo in due. Il dataset originale (`BigDigits.mat`) non è più disponibile. Nel 2026 ho rifatto l'esperimento sul **test set di MNIST**, che ha esattamente lo stesso numero di immagini per cifra, con un protocollo corretto: tuning con validazione incrociata solo sul train, standardizzazione e PCA dentro la pipeline, 10 divisioni train/test stratificate con seed fissi, risultati come media ± deviazione standard.

**Risultato:** QDA (su componenti PCA) e k-NN sono i modelli migliori, con F1 equivalenti (0,975 ± 0,006 e 0,970 ± 0,006). I modelli lineari restano nettamente sotto (F1 0,86–0,91). Il notebook si esegue su Google Colab con il pulsante in alto.
