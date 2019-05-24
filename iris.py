# Import a dataset
# Train a classifier
# Predict the label for new flower
# Visualize the decision tree

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus

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

#visualize tree

# Ignore the chunk below
# dot_data = StringIO()
# export_graphviz(classifier, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())


from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(classifier,
    out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    impurity=False
)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

