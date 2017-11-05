import csv
import numpy
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

numpy.set_printoptions(threshold='nan')

attr_index = {'age': 0, 'job': 1, 'marital': 2, 'education': 3, 'default': 4,
         'housing': 5, 'loan': 6, 'contact': 7, 'month': 8, 'day_of_week': 9,
         'duration': 10, 'campaign': 11, 'pdays': 12, 'previous': 13,
         'poutcome': 14, 'emp.var.rate': 15, 'cons.price.idx': 16,
         'cons.conf.idx': 17, 'euribor3m': 18, 'nr.employed': 19, 'y': 20}
MissValue = {"education": 'unknown'}


class MarketingData(object):
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def generate_data():
    train_data = numpy.loadtxt("train.csv", delimiter=',', skiprows=1)
    # delete id column
    train_data = numpy.delete(train_data, 0, 1)
    # test_data = numpy.array(test_data)
    # test_data = numpy.delete(test_data, 0, 1)

    missV = 0
    # One of K encoding of categorical data
    encoder = preprocessing.LabelEncoder()
    for j in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 20):
        train_data[:, j] = encoder.fit_transform(train_data[:, j])
        if j == attr_index["education"]:
            classes = list(encoder.classes_)
            missV = classes.index(MissValue['education'])
            print(missV)

    # Converting numpy strings to floats
    train_data = train_data.astype(numpy.float)
    print("train_data shape:", train_data.shape)
    # test_data = test_data.astype(numpy.float)
    Xy_test = train_data[train_data[:, 3] == missV]
    Xy_train = train_data[train_data[:, 3] != missV]
    print("shape Xy_train:", Xy_train.shape, "Xy_test:", Xy_test.shape)

    market_data = {}
    for label in [0, 1]:
        label_train = Xy_train[Xy_train[:, -1] == label]
        label_test = Xy_test[Xy_test[:, -1] == label]
        y_train = label_train[:, 3]
        X_train = numpy.delete(label_train, 3, 1)
        X_test = numpy.delete(label_test, 3, 1)
        print("x train shape:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)
        market_data[label] = MarketingData(X_train, y_train, X_test)
    return market_data


def generate_test_data():
    with open('./test.csv', 'r') as test_file:
        test_csv = csv.reader(test_file, delimiter=',')
        next(test_csv)
        test_data = list(test_csv)
    test_data = numpy.array(test_data)
    # delete id column
    # test_data = numpy.delete(test_data, 0, 1)
    # One of K encoding of categorical data
    encoder = preprocessing.LabelEncoder()
    for j in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14):
        test_data[:, j+1] = encoder.fit_transform(test_data[:, j+1])
    # Converting numpy strings to floats
    test_data = test_data.astype(numpy.float)
    missValueIndex = 7
    Xy_test = test_data[test_data[:, 3+1]==missValueIndex]
    Xy_train = test_data[test_data[:, 3+1]!=missValueIndex]
    X_train = numpy.delete(Xy_train, 3+1 ,1)
    y_train = Xy_train[:, 3+1]
    X_test = numpy.delete(Xy_test, 3+1 ,1)
    market_test_data = MarketingData(X_train, y_train, X_test)
    return market_test_data, test_data


# use knn for impute missing values
def knn(data, predict=False, best_n=None):
    if best_n:
        # prediction
        clf = KNeighborsClassifier(n_neighbors=best_n)
        return clf
    knn_scores = []
    for n_neighbors in range(4, 51):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(clf, data.X_train, data.y_train, cv=5)
        knn_scores.append((n_neighbors, scores.mean()))
    knn_scores = sorted(knn_scores, key=lambda x: x[1], reverse=True)
    print(knn_scores)


def PredictTest(clf, data, result_filename="submission.csv"):
    clf = clf.fit(data.X_train, data.y_train)
    y_pred = clf.predict(data.X_test)
    return y_pred

# def WritebackData(data, result_filename="imputed_data.csv"):


if __name__ == '__main__':
    # handle missing for attr education
    education_data = generate_data()
    best_n = 30
    new_data = {}
    for label, label_data in education_data.items():
        print("Processing label %d:" % label)
        # knn(label_data)
        knn_clf = knn(label_data, best_n=best_n)
        label_data.y_test = PredictTest(knn_clf, label_data)
        new_data[label] = label_data

    #with open('education_train.csv', 'w') as f:
    testd,traind = None, None
    imputated = []
    for label, l_data in new_data.items():
        testd = numpy.insert(l_data.X_test, 3, l_data.y_test, axis=1)
        traind = numpy.insert(l_data.X_train, 3, l_data.y_train, axis=1)
        print("final testd:", testd.shape, "traind:", traind.shape)
        imputated.extend([testd, traind])
        #imputated_data = numpy.concatenate((testd, traind), axis=0)
        #print("imputed data", imputated_data.shape)
    imputated_data = numpy.concatenate(imputated, axis=0)
    print("imputed data", imputated_data.shape)
    # numpy.savetxt("education_imputated.csv", imputated_data, delimiter=",")

    education_test_data, origin_testdata = generate_test_data()
    y_train = education_test_data.y_train
    X_train = education_test_data.X_train[:, 1:]
    # +1 include id column
    #y_test = education_test_data[:, 3+1]
    X_test = education_test_data.X_test[:, 1:]
    print("train X dims:", X_train.shape, "y:", y_train.shape, "test X:", X_test.shape)
    testing_market_data = MarketingData(X_train, y_train, X_test)
    clf = knn(testing_market_data, best_n=best_n)
    y_pred = PredictTest(clf, testing_market_data)
    print("pred y dims", type(y_pred), y_pred.shape)

    imputed_mask = list(education_test_data.X_test[:, 0].astype(numpy.int))
    print("imputed index:", len(imputed_mask), imputed_mask[:10])
    origin_testdata[imputed_mask, 3+1] = y_pred
    print("imputed origin test data:", origin_testdata.shape)
    numpy.savetxt("education_test_imputated.csv", origin_testdata, delimiter=",")
    #score = accuracy_score(testing_market_data.y_test, y_pred)
    #print("impute testing data\n", score)
