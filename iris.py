# Import a dataset
# Train a classifier
# Predict the label for new flower
# Visualize the decision tree

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# print(iris.feature_names)
# print(iris.target_names)

# for data in iris.data:
#     print(data)

# for target in iris.target:
#     print(target)

# Separate the testing data and training data

test_idx = [0, 50, 100]

#training data

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_data, train_target)

print(test_target)
print(classifier.predict(test_data))