
### What is Logistic Regression?

**Logistic Regression** is a statistical method used for **binary classification** problems, where the target variable has two possible outcomes (e.g., 0 or 1, true or false, spam or not spam). Despite its name, it is a classification algorithm, not a regression one. Logistic Regression is used to estimate the probability that an observation belongs to a certain class.

### Key Points:
- **Binary Output**: Logistic Regression predicts a probability between 0 and 1, and a threshold (often 0.5) is applied to classify the result into one of two categories.

where `z = β_0 + β_1X_1 + β_2X_2 + ... + β_nX_n`. This function squashes any input value into a range between 0 and 1, which can be interpreted as a probability.

- **Threshold for Classification**: If the probability is greater than or equal to 0.5, the algorithm classifies the observation as 1 (positive class), otherwise as 0 (negative class).

### Why Use Logistic Regression?
- It's simple and easy to implement.
- Works well for binary classification problems.
- Outputs probabilities, which makes it interpretable for understanding the model's confidence in predictions.

### Example Scenarios:
- Predicting whether an email is spam or not spam.
- Predicting if a customer will buy a product (yes/no).

  
