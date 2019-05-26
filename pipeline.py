from sklearn import datasets
iris = datasets.load_iris()

# We can see classifiers as a simple function
# Features f(x) and Label y
# f(x) = y
# something like
# def classify(features):
#  return label
# In supervised learning we don't write these ourself and the algorithm to write this themselves

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# here x represents data and y represents target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# We can replace the two lines above to use different classifiers
# from sklearn.neighbors  import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()
# they all share the .fit and .predit interface

my_classifier.fit(x_train, y_train)
predications = my_classifier.predict(x_test)

#print(y_test)
#print(predications)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predications))