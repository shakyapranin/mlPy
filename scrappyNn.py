# Create our own classifier ScrappyNN
# K nearest neighbors is a simple algorithm
# We store all cases and classify new cases as per a distance function
# In this case we are going to use distance from scipy.spatial
 
import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        pass
    def predict(self, x_test):
        # Since predictions have to be an array
        predictions = []
        for row in x_test:
            # label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_index = 0
        # Find the best distance between all the points in x_test  
        best_dist = euc(row, self.x_train[best_index])
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# Split the iris data and target as 50% into test and train
# We will use the 50% data to train the classifier
# 50% to test its accuracy and reflect on it
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)


# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

my_classifier = ScrappyKNN()

my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test)

# Now lets do a metrics test

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))  
    
