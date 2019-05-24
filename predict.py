from sklearn import tree

#Supervised learning

#Features and #Labels

#Features are something to feed to a Classifier

#Labels are somthing the classifiers outputs

#Training date and Test Data

features = [[5.2, 0], [7.3, 0], [4,1], [4.5, 1]]

labels = [0,0,1,1]

label_names = {
 0 : 'Tall',
 1: 'Medium'
}

print(labels)
print(features)
print(label_names)

for label_name in label_names.values():
    print(label_name)

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)
print(classifier.predict([[5.9, 1]]))