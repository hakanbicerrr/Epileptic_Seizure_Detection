import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics


data_training_seizure = np.load("train_seizure_features_36f_2sn.npy")
data_training_nonseizure = np.load("train_non_seizure_features_36f_2sn.npy")
data_test_seizure = np.load("test_seizure_features_36f_2sn.npy")
data_test_nonseizure = np.load("test_non_seizure_36f_2sn.npy")

label_training_seizure = [1]*len(data_training_seizure)
label_training_nonseizure = [0]*len(data_training_nonseizure)
label_test_seizure = [1]*len(data_test_seizure)
label_test_nonseizure = [0]*len(data_test_nonseizure)

print(type(data_training_seizure), data_training_seizure.shape)
print(type(data_training_nonseizure), data_training_nonseizure.shape)
print(type(data_test_seizure), data_test_seizure.shape)
print(type(data_test_nonseizure), data_test_nonseizure.shape)

reshaped_seizure = data_training_seizure.reshape(len(data_training_seizure), 36*23)
reshaped_nonseizure = data_training_nonseizure.reshape(len(data_training_nonseizure), 36*23)
reshaped_test_seizure = data_test_seizure.reshape(len(data_test_seizure), 36*23)
reshaped_test_nonseizure = data_test_nonseizure.reshape(len(data_test_nonseizure), 36*23)

data_training = np.concatenate((reshaped_seizure, reshaped_nonseizure), axis=0)
data_test = np.concatenate((reshaped_test_seizure, reshaped_test_nonseizure), axis=0)

label_training = label_training_seizure + label_training_nonseizure
label_test = label_test_seizure + label_test_nonseizure

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(data_training)
data_training = scaling.transform(data_training)
data_test = scaling.transform(data_test)

zero = 0
one = 0
clf = SVC(kernel="linear")
clf.fit(data_training, np.array(label_training))
result = clf.predict(data_test)
print(result)
for i in result:
    if i == 1:
        one += 1
    elif i == 0:
        zero += 1
print(one,zero,one+zero)

print("Accuracy:", metrics.accuracy_score(label_test, result))
print("Precision: ", metrics.precision_score(label_test, result))
print("Recall: ", metrics.recall_score(label_test, result))

