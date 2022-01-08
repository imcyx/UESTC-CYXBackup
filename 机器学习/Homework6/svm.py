import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import matplotlib.pyplot as plt
# import  tensorflow as tf
# from    tensorflow.keras import Input, layers, optimizers, Sequential, metrics

def preprocess(data_name):
    x, y = [], []
    data_y = None
    with open(data_name, "r") as test_datasets:
        for data in test_datasets.readlines():
            data = data.strip("\n")
            if data and data[0]!='%':
                data = data.split(",")
                if data[-1] == 'tested_negative':
                    data_y = 0
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


def try_svm_kernel(kernel):
    correct_num = 0
    false_num = 0

    x_test, y_test = preprocess('diabetes_test.data')
    y_pred = kernel.predict(x_test)
    y_decision = kernel.decision_function(x_test)

    # print('train_attribute\ttest_attribute\tdecision')
    for i, y in enumerate(zip(y_test, y_pred, y_decision)):
        if y[0] == y[1]:
            correct_num += 1
        else:
            false_num += 1
            # print(x_test[i], y)
        # print(y)
    acc = format(correct_num/len(y_test)*100, "4.2f")
    print(f'true:{correct_num}, false:{false_num}, acc:{acc}%')

x, y = preprocess('diabetes_train.data')

#  Linear Kernel
svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=4, loss='hinge')),
    ])
svm_clf.fit(x, y)
#  Polynomial Kernel
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=8))
    ])
poly_kernel_svm_clf.fit(x, y)
# Gaussian RBF Kernel
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.03, C=300)),
    ])
rbf_kernel_svm_clf.fit(x, y)

print('Linear Kernel:', end='\t')
try_svm_kernel(svm_clf)
print('Polynomial Kernel:', end='\t')
try_svm_kernel(poly_kernel_svm_clf)
print('Gaussian RBF Kernel:', end='\t')
try_svm_kernel(rbf_kernel_svm_clf)
