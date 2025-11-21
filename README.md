# **DA5401 – 2025 Data Challenge**

### Team name: error 

### User Name: pragatilce22b089

### Pragati L (CE22B089)


## **Overview**

This project develops a regression model to predict relevance/fitness scores for *(metric, user prompt, response)* triplets.
Because the score distribution is highly skewed and often bimodal, the solution integrates **multilingual embeddings**, **three engineered signals**, **synthetic data augmentation**, and a **cluster-aware LightGBM regressor**.

---

## **Repository Structure**

```
├── eda.ipynb                    # Exploratory Data Analysis
├── embeddings.ipynb # Embedding generation (Gemma Embedding 300M)
├── da5401.ipynb     # Full modeling pipeline
```

---

## **Pipeline Summary**

* **EDA:** score imbalance, length patterns, metric distribution, script detection.
* **Embeddings:** System, user, response, and metric embeddings using Google Gemma.
* **Three Signals:**

  * F1: Anomaly score (Isolation Forest)
  * F2: Metric–text match probability (MLP classifier)
  * F3: User–response semantic coherence (cosine similarity)
* **Data Augmentation:** Synthetic mismatched samples with low scores (0–3.5).
* **Regression Model:** LightGBM with interaction features (S1×S2, S2×S3, etc.).
* **Output:** Predictions clipped to [0,10] and saved as CSV.

---

## **Important Notes**

* The **MLP classifier (Feature F2)** is trained for **150 epochs**, enabling it to learn nonlinear compatibility between metric embeddings and text embeddings.
* Due to neural network initialization and parallelism, the **MLP may produce slightly different outputs each run**, leading to small variation in final predictions.

---

## **How to Use**

1. Run `eda.ipynb` to explore the dataset.
2. Run `embeddings.ipynb` to generate embeddings (or use precomputed `.npy` files).
3. Execute `da5401.ipynb` to compute features, train LightGBM, and generate predictions.

