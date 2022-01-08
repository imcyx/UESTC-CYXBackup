# coding:utf-8
import numpy as np
import sklearn.svm as svm
import joblib
import pickle
from sklearn.model_selection import train_test_split,cross_val_score

class TSVM(object):
    def __init__(self):
        pass

    def initial(self, kernel='linear'):
        '''
        Initial TSVM
        Parameters
        ----------
        kernel: kernel of svm
        '''
        self.Cl, self.Cu = 5.0, 0.001
        self.kernel = kernel
        self.clf = svm.SVC(C=5.0, kernel=self.kernel)

    def load(self, model_path='./TSVM.model'):
        '''
        Load TSVM from model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn and saved by sklearn.externals.joblib
        '''
        self.clf = joblib.load(model_path)

    def train(self, X1, Y1, X2):
        '''
        Train TSVM by X1, Y1, X2
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        '''
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        # First step: Use labeled data fit model and predict unlabeled data
        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        Y2 = np.expand_dims(Y2, 1)
        Y1 = np.expand_dims(Y1, 1)
        # Second step: label assignment, predict and use results expand labeled array
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])

        step = 0
        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3.ravel(), sample_weight=sample_weight)
            while True:
                # Y2_d means linear: w^Tx + b
                Y2_d = self.clf.decision_function(X2)
                Y2 = Y2.reshape(-1)
                # calculate function margin, epsilon means slack variables: ε = 1 - yi * (w^Tx + b)
                epsilon = 1 - Y2 * Y2_d
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]

                # find max error id
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                # when abort rule, invert label and predict again
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.vstack([Y1, Y2])
                    # use new label update model
                    self.clf.fit(X3, Y3.ravel(), sample_weight=sample_weight)
                else:
                    break
            # update Cu
            self.Cu = min(2*self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu

            step += 1
            print('>', end='')
        print('')

    def simple_train(self, X, Y):
        '''
        Train use normal SVM
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        '''
        self.clf.fit(X, Y)

    def score(self, X, Y):
        '''
        Calculate accuracy of TSVM by X, Y
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        Returns
        -------
        Accuracy of TSVM
                float
        '''
        return self.clf.score(X, Y)

    def predict(self, X):
        '''
        Feed X and predict Y by TSVM
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Returns
        -------
        labels of X
                np.array, shape:[n, ], n: numbers of samples
        '''
        return self.clf.predict(X)

    def save(self, path='./TSVM.model'):
        '''
        Save TSVM to model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn
        '''
        joblib.dump(self.clf, path)


def preprocess(data_name):
    x, y = [], []
    data_y = None
    with open(data_name, "r") as test_datasets:
        for data in test_datasets.readlines():
            data = data.strip("\n")
            if data and data[0]!='%':
                data = data.split(",")
                if data[-1] == 'tested_negative':
                    data_y = -1
                elif data[-1] == 'tested_positive':
                    data_y = 1
                data = [float(data_) for data_ in data[:-1]]
            else:
                continue
            x.append(data)
            y.append(data_y)
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == '__main__':
    X0, Y0 = preprocess('diabetes_test.data')
    X00, Y00 = preprocess('diabetes_train.data')
    X = np.concatenate((X00, X0), axis=0)[:,:-2]
    Y = np.concatenate((Y00, Y0), axis=0)
    slice_num = int(len(X)/10)
    X_test, X_label, X_unlabel = X[:slice_num*3], X[slice_num*3:slice_num*4], X[slice_num*4:]
    Y_test, Y_label, Y_unlabel = Y[:slice_num*3], Y[slice_num*3:slice_num*4], Y[slice_num*4:]

    model_TSVM = TSVM()
    model_TSVM.initial()
    model_TSVM.train(X_label, Y_label, X_unlabel)
    Y_hat = model_TSVM.predict(X_test)
    accuracy = model_TSVM.score(X_test, Y_test)

    model_SVM = TSVM()
    model_SVM.initial()
    model_SVM.simple_train(X_label, Y_label)
    Y_hat_SVM = model_SVM.predict(X_test)
    accuracy_SVM = model_SVM.score(X_test, Y_test)

    print('TSVM_ACC:', accuracy)
    print('SVM_ACC:', accuracy_SVM)
