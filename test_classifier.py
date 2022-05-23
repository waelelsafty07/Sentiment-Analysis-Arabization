from sklearn.metrics import accuracy_score, f1_score


def train_n_test_classifier(clf, train_features, train_labels, test_features, test_labels):
    clf.fit(train_features, train_labels) # please learn patterns from the data

    print("score on training data:")
    print(clf.score(train_features, train_labels))
    print('_'*100)

    print("score on testing data:")
    
    pred_y = clf.predict(test_features)
    print('accuracy_score: ')
    print(accuracy_score(test_labels, pred_y))
    
    print('f1_score: ')
    print(f1_score(test_labels, pred_y, average='macro'))
