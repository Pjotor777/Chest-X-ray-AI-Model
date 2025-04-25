# Chest X-Ray Age and Gender Prediction Model

This project implements a deep convolutional neural network (CNN) to perform **joint age regression** and **gender classification** from grayscale chest X-ray images, designed as part of a deep learning course project.

---

## Dataset

- **Images**: 10,702 grayscale chest X-ray images (1024×1024), resized to **128×128** for training.
- **Labels**:
  - Age (positive scalar) from `train_age.csv`
  - Gender (0 = Female, 1 = Male) from `train_gender.csv`
---
The images can be found here: https://www.kaggle.com/competitions/spr-x-ray-age/data

## Model Architecture

The CNN architecture consists of:
- **3 convolutional layers** with ReLU activation, batch normalization, and max-pooling.
- **1 fully connected layer** with 256 neurons and dropout (0.3).
- **Two output heads**:
  - **Age output**: regression (no activation)
  - **Gender output**: binary classification (sigmoid)

```python
Conv2d(1, 16, kernel_size=3) → ReLU → MaxPool
Conv2d(16, 32, kernel_size=3) → ReLU → MaxPool
Conv2d(32, 64, kernel_size=3) → ReLU → MaxPool
Flatten → Linear(64×16×16 → 256) → Dropout(0.3)
 → Linear(256 → 1) [Age]
 → Linear(256 → 1) [Gender]
