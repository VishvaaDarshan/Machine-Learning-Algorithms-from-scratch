# K-Nearest Neighbors (KNN) Algorithm

## Overview

K-Nearest Neighbors (KNN) is a simple, yet powerful algorithm used for both classification and regression tasks in machine learning. It's based on the idea that similar data points are close to each other in the feature space. This makes KNN an intuitive and easy-to-understand method for making predictions.

## Table of Contents
1. [Introduction to KNN](#introduction-to-knn)
2. [How KNN Works](#how-knn-works)
3. [Distance Metrics](#distance-metrics)
4. [Choosing the Value of K](#choosing-the-value-of-k)
5. [Algorithm Implementation](#algorithm-implementation)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)

## Introduction to KNN

K-Nearest Neighbors (KNN) is a **non-parametric** and **lazy learning** algorithm:
- **Non-parametric**: KNN doesn't make any assumptions about the underlying data distribution.
- **Lazy learning**: KNN does not involve any training phase. Instead, it stores the entire training dataset and makes predictions based on it.

### Key Concepts:
- **Instance-based learning**: KNN classifies new data points based on the stored instances.
- **Similarity measure**: KNN relies on a distance metric to measure the similarity between data points.

## How KNN Works

### 1. **Data Representation**:
   Each data point is represented as a vector in a multidimensional space, where each dimension corresponds to a feature.

### 2. **Distance Calculation**:
   For a given test point, KNN calculates the distance between the test point and all the training points.

### 3. **Identify Neighbors**:
   The algorithm selects the top \( k \) nearest neighbors (training points) based on the calculated distances.

### 4. **Vote for Class (Classification)**:
   The class of the test point is determined by a majority vote among the \( k \) neighbors.

### 5. **Average the Values (Regression)**:
   For regression tasks, the algorithm predicts the value by averaging the values of the \( k \) nearest neighbors.

## Distance Metrics

The distance metric is crucial in determining the neighbors. Common metrics include:

### 1. **Euclidean Distance**:
   \[
   \text{Distance}(x, x') = \sqrt{\sum_{i=1}^d (x_i - x_i')^2}
   \]
   - Most commonly used distance metric.
   - Works well in cases where all features are continuous.

### 2. **Manhattan Distance**:
   \[
   \text{Distance}(x, x') = \sum_{i=1}^d |x_i - x_i'|
   \]
   - Useful when features have different units or when using grid-like paths.

### 3. **Minkowski Distance**:
   \[
   \text{Distance}(x, x') = \left(\sum_{i=1}^d |x_i - x_i'|^p\right)^{1/p}
   \]
   - Generalization of both Euclidean (p=2) and Manhattan (p=1) distances.

### 4. **Hamming Distance**:
   - Used when features are categorical.
   - Counts the number of mismatches between two strings or binary vectors.

## Choosing the Value of K

Choosing the right value of \( k \) is essential for the performance of KNN:
- **Small \( k \)**: Can be sensitive to noise, leading to overfitting.
- **Large \( k \)**: May smooth out the decision boundary too much, leading to underfitting.

### Rule of Thumb:
- Start with \( k = \sqrt{n} \), where \( n \) is the number of training samples.
- Use cross-validation to fine-tune the value of \( k \).

## Algorithm Implementation

### Step-by-Step Procedure:
1. **Choose the number of neighbors (k)**.
2. **Calculate the distance** between the query point and all training samples.
3. **Sort the distances** and determine the \( k \) nearest neighbors.
4. **For classification**, perform a majority vote among the neighbors to assign a class.
5. **For regression**, compute the average value of the \( k \) nearest neighbors.

### Advanced Concepts:
- **Weighted KNN**: Assigns weights to the neighbors based on their distance, giving closer neighbors more influence.
- **KNN with Cross-Validation**: Uses cross-validation to select the optimal \( k \).

## Advantages and Disadvantages

### Advantages:
- Simple to understand and implement.
- No training phase, making it quick to apply.
- Naturally handles multiclass classification.

### Disadvantages:
- Computationally expensive at prediction time (slow with large datasets).
- Sensitive to irrelevant features and the curse of dimensionality.
- Requires careful feature scaling.


