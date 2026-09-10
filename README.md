# 🔍 ConvAutoencoder Anomaly Detection

## ✨ What the Model Does

This project implements a **Convolutional Autoencoder (CAE)** for detecting anomalies in images.

Instead of directly classifying an image as *normal* or *anomalous*, the model learns to **reconstruct the visual patterns present in the training data**.

When an image differs significantly from the patterns learned by the Autoencoder, its reconstruction becomes less accurate, producing a **higher reconstruction error**.

> **Low reconstruction error → Normal**
> **High reconstruction error → Anomalous**

---

## 🧠 How It Works

The complete detection process can be summarized as:

```text
                Input Image
                     │
                     ▼
              ┌─────────────┐
              │   Encoder   │
              └──────┬──────┘
                     │
                     ▼
               Latent Space
                     │
                     ▼
              ┌─────────────┐
              │   Decoder   │
              └──────┬──────┘
                     │
                     ▼
            Reconstructed Image
                     │
                     ▼
          Reconstruction Error
                     │
              ┌──────┴──────┐
              ▼             ▼
           Low Error      High Error
              │             │
              ▼             ▼
           NORMAL        ANOMALY
```

The Autoencoder is trained to minimize the difference between the original image and its reconstruction.

---

## 🏗️ Model Architecture

The model consists of two main components:

### Encoder

The Encoder progressively reduces the spatial representation of the image while extracting increasingly meaningful features.

| Layer     | Configuration                |
| --------- | ---------------------------- |
| Conv2D    | 3 → 128 channels, kernel 4   |
| ReLU      | Activation                   |
| AvgPool2D | Kernel 2, stride 2           |
| Conv2D    | 128 → 256 channels, kernel 4 |
| ReLU      | Activation                   |
| AvgPool2D | Kernel 2, stride 2           |
| Conv2D    | 256 → 256 channels, kernel 3 |
| ReLU      | Activation                   |
| AvgPool2D | Kernel 2, stride 2           |

### Decoder

The Decoder reconstructs the image from the compressed representation.

| Layer           | Configuration                |
| --------------- | ---------------------------- |
| ConvTranspose2D | 256 → 256 channels, kernel 4 |
| ReLU            | Activation                   |
| ConvTranspose2D | 256 → 128 channels, kernel 5 |
| ReLU            | Activation                   |
| ConvTranspose2D | 128 → 3 channels, kernel 5   |
| Sigmoid         | Output activation            |

The architecture contains approximately **3.0 million trainable parameters**.

---

## ⚙️ Training

The Autoencoder is trained using **Mean Squared Error (MSE)** as the reconstruction loss.

### Training configuration

| Parameter     |                            Value |
| ------------- | -------------------------------: |
| Batch size    |                               16 |
| Epochs        |                              100 |
| Optimizer     |                             Adam |
| Learning rate |                            0.001 |
| Loss function |                              MSE |
| Device        | CUDA if available, otherwise CPU |

The objective is to minimize:

$$
L = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
$$

where:

* \(x\) = original image
* \(\hat{x}\) = reconstructed image
* \(N\) = number of image elements

During training, the reconstruction loss decreases progressively, reaching approximately **0.0025 by epoch 100**.

---

## 🔄 Reconstruction

After training, an input image is passed through the complete Autoencoder:

```text
Original Image
      │
      ▼
   Encoder
      │
      ▼
Latent Representation
      │
      ▼
   Decoder
      │
      ▼
Reconstructed Image
```

The reconstruction is then compared with the original image.

Images that follow the patterns learned by the model can generally be reconstructed more accurately.

Anomalous regions, on the other hand, tend to produce larger differences between the original and reconstructed images.

---

## 📊 Reconstruction Error

The reconstruction error provides the numerical basis for anomaly detection.

The squared pixel-wise difference is calculated as:

$$
E(x,\hat{x}) = (x-\hat{x})^2
$$

A larger value indicates a greater difference between the original and reconstructed image.

The project also analyzes the **distribution of reconstruction errors** across the evaluated samples.

Conceptually:

```text
Reconstruction Error

Low  ───────────────────────► High
     │                         │
     ▼                         ▼
  Normal                    Anomaly
```

---

## 🚨 Anomaly Detection

The reconstruction error is used as the anomaly score.

A sample with a relatively low reconstruction error is considered consistent with the patterns learned by the Autoencoder.

A sample with a significantly higher error is considered a potential anomaly.

```text
                 Reconstruction Error
                         │
                         ▼
                 ┌───────────────┐
                 │ Anomaly Score │
                 └───────┬───────┘
                         │
                 ┌───────┴───────┐
                 ▼               ▼
              Low Error       High Error
                 │               │
                 ▼               ▼
              NORMAL          ANOMALY
```

This approach allows anomaly detection without requiring the model to directly learn an explicit classification boundary.

---

## 🗺️ Anomaly Maps

The reconstruction error can also be analyzed spatially to identify **where the reconstruction differs from the original image**.

For each pixel, the squared difference is calculated:

$$
D(x,\hat{x}) = (x-\hat{x})^2
$$

The maximum difference across the color channels is then used to obtain an anomaly map:

$$
A(x,\hat{x}) = \max_{c} D_c(x,\hat{x})
$$

This produces a spatial representation in which regions with larger reconstruction errors become more prominent.

```text
Original Image
       │
       ▼
   Autoencoder
       │
       ▼
Reconstructed Image
       │
       ▼
 Pixel-wise Difference
       │
       ▼
   Anomaly Map
```

The anomaly map provides an additional visual interpretation of the model's prediction by highlighting regions that contribute most to the reconstruction error.
