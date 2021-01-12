import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def calculate_performance(result, label_test_seizure, label_test):

    # Calculate performance
    tp = np.count_nonzero(result[0:len(label_test_seizure)] == 1)
    fp = np.count_nonzero(result[0:len(label_test_seizure)] == 0)
    fn = np.count_nonzero(result[len(label_test_seizure):] == 1)
    tn = np.count_nonzero(result[len(label_test_seizure):] == 0)
    print("True Positive:", tp,
          "\nFalse Positive:", fp,
          "\nFalse Negative:", fn,
          "\nTrue Negative:", tn)

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    pos_pred_val = tp / (tp + fp)
    neg_pred_val = tn / (tn + fn)
    false_pos_rate = fp / (fp + tn)
    false_neg_rate = fn / (tp + fn)
    false_disc_rate = fp / (tp + fp)
    false_omis_rate = fn / (tn + fn)
    print("Sensitivity: %", sensitivity*100,
          "\nSpecificity: %", specificity*100,
          "\nPositive Predictive Val: %", pos_pred_val*100,
          "\nNegative Predictive Val: %", neg_pred_val*100,
          "\nFalse Positive Rate: %", false_pos_rate*100,
          "\nFalse Negative Rate: %", false_neg_rate*100,
          "\nFalse Discovery Rate: %", false_disc_rate*100,
          "\nFalse Omission Rate: %", false_omis_rate*100)

    print("Accuracy: %", metrics.accuracy_score(label_test, result)*100)
    # print("Precision: ", metrics.precision_score(label_test, result))
    # print("Recall: ", metrics.recall_score(label_test, result))


def classify_with_svm(data_training, label_training, data_test):

    clf = SVC(kernel="linear")
    clf.fit(data_training, np.array(label_training))
    result = clf.predict(data_test)
    return result


def classify_with_random_forest(data_training, label_training, data_test):

    regressor = RandomForestClassifier(n_estimators=20, random_state=0)
    regressor.fit(data_training, label_training)
    print(regressor)
    result = regressor.predict(data_test)

    return result


def main():

    data_training_seizure = np.load("train_seizure_features_36f_2sn.npy")
    data_training_nonseizure = np.load("train_non_seizure_features_36f_2sn.npy")
    data_test_seizure = np.load("test_seizure_features_36f_2sn.npy")
    data_test_nonseizure = np.load("test_non_seizure_36f_2sn.npy")

    # Create labels.
    label_training_seizure = [1] * len(data_training_seizure)
    label_training_nonseizure = [0] * len(data_training_nonseizure)
    label_test_seizure = [1] * len(data_test_seizure)
    label_test_nonseizure = [0] * len(data_test_nonseizure)

    print("Data Training seizure:", data_training_seizure.shape)
    print("Data Training non-seizure:", data_training_nonseizure.shape)
    print("Data Test seizure:", data_test_seizure.shape)
    print("Data Test non-seizure:", data_test_nonseizure.shape)

    reshaped_seizure = data_training_seizure.reshape(len(data_training_seizure), 36 * 23)
    reshaped_nonseizure = data_training_nonseizure.reshape(len(data_training_nonseizure), 36 * 23)
    reshaped_test_seizure = data_test_seizure.reshape(len(data_test_seizure), 36 * 23)
    reshaped_test_nonseizure = data_test_nonseizure.reshape(len(data_test_nonseizure), 36 * 23)

    data_training = np.concatenate((reshaped_seizure, reshaped_nonseizure), axis=0)
    data_test = np.concatenate((reshaped_test_seizure, reshaped_test_nonseizure), axis=0)

    label_training = label_training_seizure + label_training_nonseizure
    label_test = label_test_seizure + label_test_nonseizure

    # Scale data.
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(data_training)
    data_training = scaling.transform(data_training)
    data_test = scaling.transform(data_test)

    print("**********************SVM*************************")
    # SVM Classification.
    result = classify_with_svm(data_training, label_training, data_test)
    print(result)
    calculate_performance(result, label_test_seizure, label_test)

    # Random Forest Classification.
    print("******************Random Forest****************")
    result = classify_with_random_forest(data_training, label_training, data_test)
    print(result)
    calculate_performance(result, label_test_seizure, label_test)


if __name__ == "__main__":
    main()
