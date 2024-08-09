# K-Nearest Neighbors (KNN) Algorithm

## Overview

K-Nearest Neighbors (KNN) is a simple yet powerful algorithm used for both classification and regression tasks in machine learning. It is based on the idea that similar data points are close to each other in the feature space, making KNN an intuitive method for making predictions.

## Table of Contents
1. [Introduction to KNN](#introduction-to-knn)
2. [How KNN Works](#how-knn-works)
3. [Distance Metrics](#distance-metrics)
4. [Choosing the Value of K](#choosing-the-value-of-k)
5. [Algorithm Implementation](#algorithm-implementation)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)

## Introduction to KNN

K-Nearest Neighbors (KNN) is a **non-parametric** and **lazy learning** algorithm:
- **Non-parametric**: KNN does not make any assumptions about the underlying data distribution.
- **Lazy learning**: KNN does not involve a training phase. Instead, it stores the entire training dataset and makes predictions based on it.

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

The distance metric is crucial in determining the neighbors. Here are some common metrics:

### 1. **Euclidean Distance**:
   The Euclidean distance between two points \( x \) and \( x' \) in \( d \)-dimensional space is given by:
   \[
   \text{Distance}(x, x') = \sqrt{\sum_{i=1}^d (x_i - x_i')^2}
   \]
   This is the most commonly used distance metric and works well when all features are continuous.

### 2. **Manhattan Distance**:
   The Manhattan distance (also known as L1 distance or city block distance) between two points \( x \) and \( x' \) is calculated as:
   \[
   \text{Distance}(x, x') = \sum_{i=1}^d |x_i - x_i'|
   \]
   This metric is useful when features have different units or when using grid-like paths.

### 3. **Minkowski Distance**:
   The Minkowski distance is a generalized metric that includes both Euclidean and Manhattan distances as special cases. It is defined as:
   \[
   \text{Distance}(x, x') = \left(\sum_{i=1}^d |x_i - x_i'|^p\right)^{\frac{1}{p}}
   \]
   Where \( p \) is a parameter. When \( p = 2 \), it becomes the Euclidean distance, and when \( p = 1 \), it becomes the Manhattan distance.

### 4. **Hamming Distance**:
   Hamming distance is used when dealing with categorical data. It is the number of positions at which the corresponding elements are different:
   \[
   \text{Distance}(x, x') = \sum_{i=1}^d \mathbf{1}(x_i \neq x_i')
   \]
   Where \( \mathbf{1}(x_i \neq x_i') \) is an indicator function that equals 1 if \( x_i \) is not equal to \( x_i' \), and 0 otherwise.

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

### Conclusion

K-Nearest Neighbors is a powerful and intuitive algorithm for both classification and regression tasks. By understanding the fundamentals, you can easily extend KNN to more complex problems and datasets. Experiment with different distance metrics and values of \( k \) to see how they affect your model's performance.
