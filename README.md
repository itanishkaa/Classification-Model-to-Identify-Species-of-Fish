# Fish Species Classification Model

This repository contains a Jupyter notebook for classifying fish species based on various attributes such as weight, height, and length. The model uses machine learning algorithms like Decision Tree, Random Forest, and K-Nearest Neighbors (KNN) to predict the species of fish based on their physical measurements.

## Dataset
The dataset `Fish.csv` contains information about different fish species, with the following columns:

- **Category:** Numerical representation of the fish species.
- **Species:** The name of the fish species.
- **Weight:** Weight of the fish.
- **Height:** Height of the fish.
- **Width:** Width of the fish.
- **Length1, Length2, Length3:** Different length measurements of the fish.

```markdown
| Category | Species | Weight | Height | Width | Length1 | Length2 | Length3 |
|:-----|:--------:|:------:|:-----:|:--------:|:------:|:-----:|--------:|
| 1 | Bream | 242.0 | 11.52 | 4.02 | 23.2 | 25.4 | 30.0 |
| 1 | Bream | 290.0 | 12.48 | 4.31 | 24.0 | 26.3 | 31.2 |
```

## Requirements
The following libraries are required to run the code in the Jupyter notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Model Overview
### 1. Data Loading and Preprocessing
The dataset is loaded using pandas, and basic information about the dataset (such as column types and non-null values) is displayed. The features are then separated from the target variable (Species), and the data is split into training and test sets.

### 2. Scaling
The features are standardized using StandardScaler to ensure that all features have the same scale. This is particularly important for models like KNN that are sensitive to the magnitude of the features.

### 3. Modeling
Three different classification algorithms are used:

- **Decision Tree Classifier:** A simple and interpretable model used to predict fish species based on their attributes.
- **Random Forest Classifier:** An ensemble model that aggregates the predictions of multiple decision trees.
- **K-Nearest Neighbors (KNN):** A non-parametric method that classifies a data point based on how its neighbors are classified.

### 4. Model Evaluation
Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score. The confusion matrix is also displayed to visually inspect the model's performance.

### 5. Results Comparison
The predictions of the models are compared to the actual species, and incorrect predictions are highlighted for further analysis.

## Results
- **Decision Tree:** Achieves perfect accuracy of 100% on the test set, with no misclassifications.
- **Random Forest:** Achieves 90.63% accuracy, with some misclassifications in species like `Roach` and `Whitefish`.
- **KNN:** Also achieves 90.63% accuracy with a similar pattern of misclassifications.

## Conclusion
The **Decision Tree** model performed the best with 100% accuracy on the test data.
Both **Random Forest** and **KNN** showed slightly lower accuracy, but they still performed well, with most species being correctly classified.

## Potential Improvements
- **Hyperparameter Tuning:** Fine-tuning the models, especially for the Random Forest and KNN classifiers, could improve performance.
- **Cross-Validation:** Using cross-validation could provide a better understanding of model performance on different subsets of the data.
- **Feature Engineering:** Adding more features or transformations of existing ones may help the models generalize better.
