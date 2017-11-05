import csv

import numpy
from sklearn import preprocessing
from sklearn import tree, svm
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

numpy.set_printoptions(threshold='nan')


class MarketingData(object):
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def __init__(self, X_train, y_train, X_test, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test



def generate_data():

    with open('train.csv', 'r') as train_file:
        train_csv = csv.reader(train_file, delimiter=',')
        next(train_csv)
        train_data = list(train_csv)
        # train_data = train_data[:-10000]
        # validation_data = train_data[-10000:]

    with open('./test.csv', 'r') as test_file:
        test_csv = csv.reader(test_file, delimiter=',')
        next(test_csv)
        test_data = list(test_csv)

    train_data = numpy.array(train_data)
    # delete id column
    train_data = numpy.delete(train_data, 0, 1)
    # validation_data = numpy.array(validation_data)
    # validation_data = numpy.delete(validation_data, 0, 1)
    test_data = numpy.array(test_data)
    test_data = numpy.delete(test_data, 0, 1)

    # One of K encoding of categorical data
    encoder = preprocessing.LabelEncoder()
    for j in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 20):
        train_data[:, j] = encoder.fit_transform(train_data[:, j])
        # validation_data[:, j] = encoder.fit_transform(validation_data[:, j])
    for j in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14):
        test_data[:, j] = encoder.fit_transform(test_data[:, j])

    # Converting numpy strings to floats
    train_data = train_data.astype(numpy.float)
    # validation_data = validation_data.astype(numpy.float)
    test_data = test_data.astype(numpy.float)

    train_data = train_data[:, :-1]
    train_target = train_data[:, -1]
    market = MarketingData(train_data, train_target, test_data)
    return market

def generate_imputed_data():
    # this version data has label on train data
    train_data = numpy.loadtxt("education_imputated.csv", delimiter=",")
    test_data = numpy.loadtxt("education_test_imputated.csv", delimiter=",")
    market = MarketingData(train_data[:, :-1], train_data[:, -1], test_data[:, 1:])
    return market



def learn_decision_tree(data):
    DT = tree.DecisionTreeClassifier(max_depth=7)
    scorer = make_scorer(matthews_corrcoef)
    for i in range(5):
        scores = cross_val_score(DT, data.X_train, data.y_train, cv=10, scoring=scorer)
        print("iteration",i, "dt mean:", scores.mean())
        scores = list(scores)
        print("Decision Tree train scores:\n", scores)
    return DT
    # DT = DT.fit(train_data[:, :-1], train_data[:, -1])
    # predictionsDT = DT.predict(validation_data[:, :-1])

    # validating predicions
    # dtError = 0
    # for i in range(0, len(validation_data)):
    #         if(validation_data[i][20] != predictionsDT[i]):
    #                 dtError = dtError + 1
    # print("DT Error : ", float(dtError)/len(validation_data)*100.0)


def svm_clf(data):
    clf = svm.LinearSVC(C=1)
    for i in range(5):
        scores = cross_val_score(clf, data.X_train, data.y_train, cv=10)
        print("iteration",i, "svm mean:", scores.mean())
        scores = list(scores)
        print("svm train scores:\n", scores)
    return clf


# use knn for impute missing values
def knn(data, predict=False):
    n_neighbors = 3
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    for i in range(5):
        scores = cross_val_score(clf, data.X_train, data.y_train, cv=10)
        print("svm mean:", scores.mean())
        scores = list(scores)
        print("svm train scores:\n", scores)

    # prediction
    best_n = n_neighbors
    clf = KNeighborsClassifier(n_neighbors=best_n)
    return clf


def PredictTest(clf, data, result_filename="submission.csv"):
    clf = clf.fit(data.X_train, data.y_train)
    y_pred = clf.predict(data.X_test)
    with open(result_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(0, len(data.X_test)):
            f.write(','.join([str(i), str(int(y_pred[i]))]))
            f.write('\n')


if __name__ == '__main__':
    #market_data = generate_data()
    # imputed data
    market_data = generate_imputed_data()
    clf = learn_decision_tree(market_data)
    #clf = svm_clf(market_data)
    # predicting
    PredictTest(clf, market_data, result_filename="dt_education_submission.csv")
