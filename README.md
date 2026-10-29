#  Atelier 2 – Deep Learning with PyTorch

### CNN, Faster R-CNN, VGG16, AlexNet & Vision Transformer (ViT)

**Université Abdelmalek Essaâdi – Faculté des Sciences et Techniques de Tanger (FSTT)**

- **Module:** Deep Learning
- **Professor:** Pr. Lotfi Elaachak
- **Student:** Ayoub LAHLAIBI
- **Environment:** Google Colab (NVIDIA Tesla T4 GPU)

---

##  1. Project Overview

This project was developed as part of the Deep Learning Lab 2 at FST Tanger.

The main objective is to implement, train, evaluate, and compare different Deep Learning architectures for handwritten digit recognition using the MNIST dataset.

Five neural network architectures were explored:

1. Convolutional Neural Network (CNN)
2. Faster Region-Based Convolutional Neural Network (Faster R-CNN)
3. VGG16 (Transfer Learning)
4. AlexNet (Transfer Learning)
5. Vision Transformer (ViT) – From Scratch

The experiments were implemented using PyTorch and executed with GPU acceleration on Google Colab.

##  2. Objectives

- Understand CNN architectures and convolutional operations.
- Implement image classification using PyTorch.
- Explore object detection using Faster R-CNN.
- Apply Transfer Learning using pretrained VGG16 and AlexNet.
- Build a Vision Transformer from scratch.
- Evaluate models using Accuracy, F1-score, Loss, and Training Time.
- Compare different architectures and analyze their performance.

##  3. Dataset – MNIST

The MNIST dataset is a benchmark dataset for handwritten digit classification.

**Dataset characteristics:**

| Property | Description |
|---|---|
| Dataset | MNIST |
| Total images | 70,000 |
| Training images | 60,000 |
| Testing images | 10,000 |
| Image size | 28 × 28 pixels |
| Classes | 10 (digits 0–9) |
| Image type | Grayscale |

**Dataset source:** https://www.kaggle.com/datasets/hojjatk/mnist-dataset

The dataset was downloaded using `torchvision.datasets.MNIST`.

##  4. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- CUDA / GPU acceleration
- Git & GitHub

##  5. Implemented Models

### 5.1 Convolutional Neural Network (CNN)

A custom CNN was implemented using PyTorch.

**Architecture:**

Input Image (1 × 28 × 28)

→ Convolution + ReLU + MaxPooling

→ Convolution + ReLU + MaxPooling

→ Flatten

→ Fully Connected Layer

→ Dropout

→ Output Layer (10 classes)

**Hyperparameters:**

- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 5
- Loss Function: CrossEntropyLoss

**Results:**

- Accuracy: 98.97%
- F1-score: 98.96%
- Test Loss: 0.02958
- Training Time: 74.61 seconds

### 5.2 Faster R-CNN

Faster R-CNN was implemented using the Torchvision detection framework.

Unlike standard CNN classification, Faster R-CNN is designed for object detection.

For this experiment, MNIST was adapted to a simplified object detection task using image-level bounding boxes.

**Experimental configuration:**

- Training samples: 2,000
- Testing samples: 1,000
- Epochs: 2
- Optimizer: Adam
- Learning Rate: 0.0001

**Results:**

- Accuracy: 75.90%
- F1-score: 74.58%
- Training Time: 56.44 seconds

**Observation:**

Faster R-CNN achieved lower classification accuracy than the other models. However, the experiment used a smaller dataset and a different learning task, so its results should not be considered a direct comparison with the classification models.

### 5.3 VGG16 – Transfer Learning

VGG16 is a deep convolutional neural network pretrained on ImageNet.

The original classification layer was replaced with a new output layer containing 10 classes for MNIST.

**Configuration:**

- Pretrained model: VGG16
- Dataset: MNIST
- Input size: 224 × 224
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 3

**Results:**

- Accuracy: 99.02%
- F1-score: 99.02%
- Test Loss: 0.04068
- Training Time: 1174.58 seconds

**Observation:**

VGG16 achieved the highest classification accuracy but required significantly more training time.

### 5.4 AlexNet – Transfer Learning

AlexNet is a convolutional neural network originally designed for image classification.

A pretrained AlexNet model was adapted to recognize the 10 MNIST digit classes.

**Configuration:**

- Pretrained model: AlexNet
- Dataset: MNIST
- Input size: 224 × 224
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 3

**Results:**

- Accuracy: 98.98%
- F1-score: 98.98%
- Test Loss: 0.02990
- Training Time: 320.33 seconds

### 5.5 Vision Transformer (ViT) – From Scratch

A Vision Transformer was implemented from scratch using PyTorch.

Unlike CNN-based architectures, ViT uses Self-Attention mechanisms to learn relationships between image patches.

**Main components:**

- Patch Embedding
- Positional Embedding
- Class Token
- Multi-Head Self-Attention
- Transformer Encoder
- Layer Normalization
- Classification Head

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Image Size | 28 × 28 |
| Patch Size | 7 × 7 |
| Number of Patches | 16 |
| Embedding Dimension | 64 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 0.001 |

**Results:**

- Accuracy: 98.12%
- F1-score: 98.11%
- Test Loss: 0.06297
- Training Time: 224.39 seconds

**Observation:**

The ViT achieved strong classification performance despite being trained from scratch, without pretrained ImageNet weights.

The training loss decreased progressively over 10 epochs, demonstrating successful optimization of the Transformer architecture.

##  6. Experimental Results and Comparison

The following table summarizes the experimental results.

| Model | Accuracy | F1-score | Loss | Training Time (s) |
|---|---:|---:|---:|---:|
| CNN | 98.97% | 98.96% | 0.02958 | 74.61 |
| AlexNet | 98.98% | 98.98% | 0.02990 | 320.33 |
| VGG16 | **99.02%** | **99.02%** | 0.04068 | 1174.58 |
| Faster R-CNN (subset) | 75.90% | 74.58% | N/A | 56.44 |
| ViT | 98.12% | 98.11% | 0.06297 | 224.39 |

### Key Findings

**Highest Classification Accuracy: VGG16**

VGG16 achieved 99.02% accuracy, making it the most accurate model in these experiments.

**Best Accuracy–Training Time Trade-off: CNN**

The custom CNN achieved 98.97% accuracy with only 74.61 seconds of training.

**Transformer Performance: ViT**

The Vision Transformer reached 98.12% accuracy, showing the effectiveness of attention-based architectures on MNIST.

**Faster R-CNN Limitations**

Faster R-CNN was evaluated on a reduced MNIST subset with simplified bounding-box annotations. Its performance and training time cannot be directly compared with the other models under this experimental setup.

The experiments used different numbers of epochs, model architectures, and training settings. Therefore, the comparison is descriptive rather than a fully controlled benchmark.

##  7. Evaluation Metrics

The following metrics were used:

**Accuracy:** Proportion of correctly classified images.

**F1-score:** Harmonic mean of precision and recall, calculated using macro averaging.

**Loss:** Measures prediction error during evaluation for the classification models.

**Training Time:** Total time required to train each model.

Faster R-CNN uses a multi-component detection loss, which differs from the standard classification loss.

##  8. Project Structure

```text
Atelier2_Deep_Learning/
│
├── Atelier2_Deep_Learning.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── results/
    ├── comparison_final.csv
    ├── accuracy_comparison.png
    ├── f1_comparison.png
    └── training_time_comparison.png
```

##  9. How to Run

**Step 1 – Clone the repository**

```bash
git clone https://github.com/Ayoub-lah/Atelier2_Deep_Learning.git
cd Atelier2_Deep_Learning
```

**Step 2 – Install dependencies**

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

**Step 3 – Run the notebook**

Open `Atelier2_Deep_Learning.ipynb` using Google Colab or Jupyter Notebook.

**Step 4 – Enable GPU acceleration**

In Google Colab:

`Runtime → Change runtime type → T4 GPU`

**Step 5 – Execute all notebook cells**

Run the cells sequentially to train and evaluate the models.

Training time may vary depending on GPU availability and hardware performance.

##  10. What I Learned

Through this lab, I gained practical experience in:

- Building CNN architectures using PyTorch.
- Understanding convolution, pooling, activation functions, and fully connected layers.
- Using GPU acceleration for Deep Learning training.
- Applying pretrained models through Transfer Learning.
- Exploring the Faster R-CNN object detection architecture.
- Implementing a Vision Transformer from scratch.
- Understanding Self-Attention and positional embeddings.
- Evaluating neural networks using classification metrics.
- Analyzing model accuracy, loss, and computational cost.

##  11. Conclusion

This lab provided a practical comparison of several Deep Learning architectures for computer vision.

VGG16 achieved the highest accuracy at 99.02%, while the custom CNN provided the best balance between accuracy and computational efficiency.

AlexNet also demonstrated excellent performance through Transfer Learning.

The Vision Transformer successfully classified handwritten digits using an attention-based architecture implemented from scratch.

Faster R-CNN highlighted the differences between object detection and image classification and demonstrated the importance of selecting an architecture appropriate for the target task.

Overall, this laboratory strengthened my understanding of PyTorch, neural network architectures, Transfer Learning, Vision Transformers, and experimental model evaluation.

---


*Deep Learning – Lab 2 | FST Tanger | 2026*
