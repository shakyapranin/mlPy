# Import a dataset
# Train a classifier
# Predict the label for new flower
# Visualize the decision tree

from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)
print(iris.target_names)