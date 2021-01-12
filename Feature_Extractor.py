import scipy.io
import os
import numpy as np
import wfdb
import pywt
from scipy.stats import skew

# Define the directory paths
training_seizure_headers_dir = "Training/Header-seizure/"
training_seizure_annotations_dir = "Training/Annotations/"
training_seizure_data_dir = "Training/With-seizure/"

test_seizure_headers_dir = "Test/Header-seizure/"
test_seizure_annotations_dir = "Test/Annotations/"
test_seizure_data_dir = "Test/With-seizure/"

training_nonseizure_data_dir = "With_Without_Seizure/Without-seizure/"


def get_seizure_features(training_seizure_headers_dir, training_seizure_annotations_dir, training_seizure_data_dir):

    a = 0
    empty = 0
    file_counter = 0
    total = 0
    k = 0
    ann = []  # For seizure annotation
    b = 0
    data_training_seizure = []  # Data that should be trained.
    data_training_seizure_names = []  # Names of the training data
    for file in os.listdir(training_seizure_headers_dir):
        channels = wfdb.rdheader(training_seizure_headers_dir + file[0:-4])  # Extract channels.
        orj_sig_name = channels.sig_name.copy()  # Extract channel names.
        seizure = wfdb.rdann(training_seizure_annotations_dir + file[0:-9] + ".edf", extension="seizures")  # Read seizure annotation.
        ann.append(seizure.sample)  # Save seizure annotation into a list.
        print(file[0:-9], file_counter)  # Count the record of file.
        for j in range(0, len(ann[file_counter]), 2):
            print("file", "#")
            print(file_counter, j, k)
            print("Annotations:", ann[file_counter][j], ann[file_counter][j + 1])

            # Load data according to seizure annotations.
            data_seizure = scipy.io.loadmat(training_seizure_data_dir + file[0:-4] + ".mat",
                                            variable_names=["val"]).get("val")[:,
                           ann[file_counter][j] - 1: ann[file_counter][j + 1]]
            print("Size of a data:", np.size(data_seizure))
            print("Shape of a data:", data_seizure.shape)
            # Check if the data length and annotations match.
            print("Dimensions match: ",
                  ((ann[file_counter][j + 1]) - (ann[file_counter][j] - 1)) == data_seizure.shape[1])

            if np.size(data_seizure) != 0 and \
                    data_seizure.shape[1] >= 5120:  # 5120 = The first 20 seconds of the seizure(epoch,segment)
                # We want seizures more than 20 seconds.
                # Delete some channels that are not needed.
                if channels.n_sig > 23:
                    print(file)
                    print("**************************************** OLD Dimension:", data_seizure.shape)
                    for i, channel in enumerate(channels.sig_name):
                        if channel == "-":
                            print("-", i + 1)
                            data_seizure = np.delete(data_seizure, i, 0)
                            del channels.sig_name[i]

                        elif channel == ".":
                            print(".", i + 1)
                            data_seizure = np.delete(data_seizure, i, 0)
                            del channels.sig_name[i]

                        elif channel == "ECG":
                            print("ECG", i + 1)
                            data_seizure = np.delete(data_seizure, i, 0)
                            del channels.sig_name[i]

                        elif channel == "VNS":
                            print("VNS", i + 1)
                            data_seizure = np.delete(data_seizure, i, 0)
                            del channels.sig_name[i]

                        elif channel == "LOC-ROC":
                            print("LOC-ROC", i + 1)
                            data_seizure = np.delete(data_seizure, i, 0)
                            del channels.sig_name[i]

                        elif channel == "EKG1-CHIN":
                            print("EKG1-CHIN", i + 1)
                            data_seizure = np.delete(data_seizure, i, 0)
                            del channels.sig_name[i]

                    a += 1
                    print("**************************************** NEW Dimension:", data_seizure.shape)
                channels.sig_name = orj_sig_name.copy()            

                # Feature Extraction
                if len(data_seizure) >= 23:  # Check if there are equal to or more than 23 channels.
                    for i in range(int(data_seizure.shape[1]/512)+1):  # Segment data into 2 seconds epoch.
                        # Calculate DWT of 2 sec. epochs using "coif3" and 7 level.
                        coeffs = pywt.wavedec(data_seizure[0:23, i * 512:(i + 1) * 512], "coif3", level=7)
                        cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

                        band1_en = []
                        band2_en = []
                        band3_en = []
                        band4_en = []
                        band5_en = []
                        band6_en = []

                        band1_max = []
                        band2_max = []
                        band3_max = []
                        band4_max = []
                        band5_max = []
                        band6_max = []

                        band1_min = []
                        band2_min = []
                        band3_min = []
                        band4_min = []
                        band5_min = []
                        band6_min = []

                        band1_mean = []
                        band2_mean = []
                        band3_mean = []
                        band4_mean = []
                        band5_mean = []
                        band6_mean = []

                        band1_std = []
                        band2_std = []
                        band3_std = []
                        band4_std = []
                        band5_std = []
                        band6_std = []

                        band1_skew = []
                        band2_skew = []
                        band3_skew = []
                        band4_skew = []
                        band5_skew = []
                        band6_skew = []
                        # Calculate 6 features of DWT coefficients.
                        for i in range(len(cD1)):
                            band1_en.append(np.sum(cD7[i, :] ** 2))
                            band2_en.append(np.sum(cD6[i, :] ** 2))
                            band3_en.append(np.sum(cD5[i, :] ** 2))
                            band4_en.append(np.sum(cD4[i, :] ** 2))
                            band5_en.append(np.sum(cD3[i, :] ** 2))
                            band6_en.append(np.sum(cD2[i, :] ** 2))

                            band1_max.append(np.max(cD7[i, :]))
                            band2_max.append(np.max(cD6[i, :]))
                            band3_max.append(np.max(cD5[i, :]))
                            band4_max.append(np.max(cD4[i, :]))
                            band5_max.append(np.max(cD3[i, :]))
                            band6_max.append(np.max(cD2[i, :]))

                            band1_min.append(np.min(cD7[i, :]))
                            band2_min.append(np.min(cD6[i, :]))
                            band3_min.append(np.min(cD5[i, :]))
                            band4_min.append(np.min(cD4[i, :]))
                            band5_min.append(np.min(cD3[i, :]))
                            band6_min.append(np.min(cD2[i, :]))

                            band1_mean.append(np.mean(cD7[i, :]))
                            band2_mean.append(np.mean(cD6[i, :]))
                            band3_mean.append(np.mean(cD5[i, :]))
                            band4_mean.append(np.mean(cD4[i, :]))
                            band5_mean.append(np.mean(cD3[i, :]))
                            band6_mean.append(np.mean(cD2[i, :]))

                            band1_std.append(np.std(cD7[i, :]))
                            band2_std.append(np.std(cD6[i, :]))
                            band3_std.append(np.std(cD5[i, :]))
                            band4_std.append(np.std(cD4[i, :]))
                            band5_std.append(np.std(cD3[i, :]))
                            band6_std.append(np.std(cD2[i, :]))

                            band1_skew.append(skew(cD7[i, :]))
                            band2_skew.append(skew(cD6[i, :]))
                            band3_skew.append(skew(cD5[i, :]))
                            band4_skew.append(skew(cD4[i, :]))
                            band5_skew.append(skew(cD3[i, :]))
                            band6_skew.append(skew(cD2[i, :]))

                        band1_en = (np.array(band1_en).reshape(1, -1))
                        band2_en = (np.array(band2_en).reshape(1, -1))
                        band3_en = (np.array(band3_en).reshape(1, -1))
                        band4_en = (np.array(band4_en).reshape(1, -1))
                        band5_en = (np.array(band5_en).reshape(1, -1))
                        band6_en = (np.array(band6_en).reshape(1, -1))

                        band1_max = np.array(band1_max).reshape(1, -1)
                        band2_max = np.array(band2_max).reshape(1, -1)
                        band3_max = np.array(band3_max).reshape(1, -1)
                        band4_max = np.array(band4_max).reshape(1, -1)
                        band5_max = np.array(band5_max).reshape(1, -1)
                        band6_max = np.array(band6_max).reshape(1, -1)

                        band1_min = np.array(band1_min).reshape(1, -1)
                        band2_min = np.array(band2_min).reshape(1, -1)
                        band3_min = np.array(band3_min).reshape(1, -1)
                        band4_min = np.array(band4_min).reshape(1, -1)
                        band5_min = np.array(band5_min).reshape(1, -1)
                        band6_min = np.array(band6_min).reshape(1, -1)

                        band1_mean = np.array(band1_mean).reshape(1, -1)
                        band2_mean = np.array(band2_mean).reshape(1, -1)
                        band3_mean = np.array(band3_mean).reshape(1, -1)
                        band4_mean = np.array(band4_mean).reshape(1, -1)
                        band5_mean = np.array(band5_mean).reshape(1, -1)
                        band6_mean = np.array(band6_mean).reshape(1, -1)

                        band1_std = np.array(band1_std).reshape(1, -1)
                        band2_std = np.array(band2_std).reshape(1, -1)
                        band3_std = np.array(band3_std).reshape(1, -1)
                        band4_std = np.array(band4_std).reshape(1, -1)
                        band5_std = np.array(band5_std).reshape(1, -1)
                        band6_std = np.array(band6_std).reshape(1, -1)

                        band1_skew = np.array(band1_skew).reshape(1, -1)
                        band2_skew = np.array(band2_skew).reshape(1, -1)
                        band3_skew = np.array(band3_skew).reshape(1, -1)
                        band4_skew = np.array(band4_skew).reshape(1, -1)
                        band5_skew = np.array(band5_skew).reshape(1, -1)
                        band6_skew = np.array(band6_skew).reshape(1, -1)
                        # Create feature vector.
                        feature_vector = np.concatenate((band1_en, band1_max, band1_min, band1_mean, band1_std, band1_skew,
                                                         band2_en, band2_max, band2_min, band2_mean, band2_std, band2_skew,
                                                         band3_en, band3_max, band3_min, band3_mean, band3_std, band3_skew,
                                                         band4_en, band4_max, band4_min, band4_mean, band4_std, band4_skew,
                                                         band5_en, band5_max, band5_min, band5_mean, band5_std, band5_skew,
                                                         band6_en, band6_max, band6_min, band6_mean, band6_std, band6_skew
                                                         ), axis=0)
                        # print(np.transpose(feature_vector), np.transpose(feature_vector).shape)
                        print(np.transpose(feature_vector).shape)
                        # Obtain training dataset as a list of array.
                        data_training_seizure.append(np.transpose(feature_vector))

                    # data_training.append(data[0:23, 0:5120])
                    data_training_seizure_names.append(file[0: -9])
                    k += 1

                # Discard data and do nothing if the data size is equal to zero.
                # This happens when the annotations are more than 1 million, because we can not read file which has
                # 921600 samples. If annotation is more than 921000, we basically can not obtain the data.
                elif np.size(data_seizure) == 0:
                    empty += 1
                    print("[]")
                    print("**************", empty)

        k = 0
        total += len(ann[file_counter])
        file_counter += 1
        print()

    # print(data_training_seizure, data_training_seizure)
    print("total annotations", total / 2)
    print("empty ones:", empty)
    print("Training data:", len(data_training_seizure))
    print(data_training_seizure[0].shape)

    return data_training_seizure


def get_non_seizure_features(training_nonseizure_data_dir):   # Obtain data without seizure and features.

    # Same procedures as the data with seizure.
    data_training_nonseizure = []

    for file in os.listdir(training_nonseizure_data_dir):
        channels = wfdb.rdheader("Headers/" + file[0:-4])
        orj_sig_name = channels.sig_name.copy()
        data_without_seizure = scipy.io.loadmat(training_nonseizure_data_dir + file, variable_names=["val"]).get("val")[
                               :, 0:4096]  # 0:4096 for training. 4096:5120 for test.
        print(file)
        if channels.n_sig > 23:

            for i, channel in enumerate(channels.sig_name):
                if channel == "-":
                    print("-", i + 1)
                    data_without_seizure = np.delete(data_without_seizure, i, 0)
                    del channels.sig_name[i]

                elif channel == ".":
                    print(".", i + 1)
                    data_without_seizure = np.delete(data_without_seizure, i, 0)
                    del channels.sig_name[i]

                elif channel == "ECG":
                    print("ECG", i + 1)
                    data_without_seizure = np.delete(data_without_seizure, i, 0)
                    del channels.sig_name[i]

                elif channel == "VNS":
                    print("VNS", i + 1)
                    data_without_seizure = np.delete(data_without_seizure, i, 0)
                    del channels.sig_name[i]

                elif channel == "LOC-ROC":
                    print("LOC-ROC", i + 1)
                    data_without_seizure = np.delete(data_without_seizure, i, 0)
                    del channels.sig_name[i]

                elif channel == "EKG1-CHIN":
                    print("EKG1-CHIN", i + 1)
                    data_without_seizure = np.delete(data_without_seizure, i, 0)
                    del channels.sig_name[i]

        channels.sig_name = orj_sig_name.copy()
        if len(data_without_seizure) >= 23:
            for i in range(8):  # 8 for training data. 2 for test.
                coeffs = pywt.wavedec(data_without_seizure[0:23, i * 512:(i + 1) * 512], "coif3", level=7)
                cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
                band1_en = []
                band2_en = []
                band3_en = []
                band4_en = []
                band5_en = []
                band6_en = []

                band1_max = []
                band2_max = []
                band3_max = []
                band4_max = []
                band5_max = []
                band6_max = []

                band1_min = []
                band2_min = []
                band3_min = []
                band4_min = []
                band5_min = []
                band6_min = []

                band1_mean = []
                band2_mean = []
                band3_mean = []
                band4_mean = []
                band5_mean = []
                band6_mean = []

                band1_std = []
                band2_std = []
                band3_std = []
                band4_std = []
                band5_std = []
                band6_std = []

                band1_skew = []
                band2_skew = []
                band3_skew = []
                band4_skew = []
                band5_skew = []
                band6_skew = []

                for i in range(len(cD1)):
                    band1_en.append(np.sum(cD7[i, :] ** 2))
                    band2_en.append(np.sum(cD6[i, :] ** 2))
                    band3_en.append(np.sum(cD5[i, :] ** 2))
                    band4_en.append(np.sum(cD4[i, :] ** 2))
                    band5_en.append(np.sum(cD3[i, :] ** 2))
                    band6_en.append(np.sum(cD1[i, :] ** 2))

                    band1_max.append(np.max(cD7[i, :]))
                    band2_max.append(np.max(cD6[i, :]))
                    band3_max.append(np.max(cD5[i, :]))
                    band4_max.append(np.max(cD4[i, :]))
                    band5_max.append(np.max(cD3[i, :]))
                    band6_max.append(np.max(cD2[i, :]))

                    band1_min.append(np.min(cD7[i, :]))
                    band2_min.append(np.min(cD6[i, :]))
                    band3_min.append(np.min(cD5[i, :]))
                    band4_min.append(np.min(cD4[i, :]))
                    band5_min.append(np.min(cD3[i, :]))
                    band6_min.append(np.min(cD2[i, :]))

                    band1_mean.append(np.mean(cD7[i, :]))
                    band2_mean.append(np.mean(cD6[i, :]))
                    band3_mean.append(np.mean(cD5[i, :]))
                    band4_mean.append(np.mean(cD4[i, :]))
                    band5_mean.append(np.mean(cD3[i, :]))
                    band6_mean.append(np.mean(cD2[i, :]))

                    band1_std.append(np.std(cD7[i, :]))
                    band2_std.append(np.std(cD6[i, :]))
                    band3_std.append(np.std(cD5[i, :]))
                    band4_std.append(np.std(cD4[i, :]))
                    band5_std.append(np.std(cD3[i, :]))
                    band6_std.append(np.std(cD2[i, :]))

                    band1_skew.append(skew(cD7[i, :]))
                    band2_skew.append(skew(cD6[i, :]))
                    band3_skew.append(skew(cD5[i, :]))
                    band4_skew.append(skew(cD4[i, :]))
                    band5_skew.append(skew(cD3[i, :]))
                    band6_skew.append(skew(cD2[i, :]))

                band1_en = np.array(band1_en).reshape(1, -1)
                band2_en = np.array(band2_en).reshape(1, -1)
                band3_en = np.array(band3_en).reshape(1, -1)
                band4_en = np.array(band4_en).reshape(1, -1)
                band5_en = np.array(band5_en).reshape(1, -1)
                band6_en = np.array(band6_en).reshape(1, -1)

                band1_max = np.array(band1_max).reshape(1, -1)
                band2_max = np.array(band2_max).reshape(1, -1)
                band3_max = np.array(band3_max).reshape(1, -1)
                band4_max = np.array(band4_max).reshape(1, -1)
                band5_max = np.array(band5_max).reshape(1, -1)
                band6_max = np.array(band6_max).reshape(1, -1)

                band1_min = np.array(band1_min).reshape(1, -1)
                band2_min = np.array(band2_min).reshape(1, -1)
                band3_min = np.array(band3_min).reshape(1, -1)
                band4_min = np.array(band4_min).reshape(1, -1)
                band5_min = np.array(band5_min).reshape(1, -1)
                band6_min = np.array(band6_min).reshape(1, -1)

                band1_mean = np.array(band1_mean).reshape(1, -1)
                band2_mean = np.array(band2_mean).reshape(1, -1)
                band3_mean = np.array(band3_mean).reshape(1, -1)
                band4_mean = np.array(band4_mean).reshape(1, -1)
                band5_mean = np.array(band5_mean).reshape(1, -1)
                band6_mean = np.array(band6_mean).reshape(1, -1)

                band1_std = np.array(band1_std).reshape(1, -1)
                band2_std = np.array(band2_std).reshape(1, -1)
                band3_std = np.array(band3_std).reshape(1, -1)
                band4_std = np.array(band4_std).reshape(1, -1)
                band5_std = np.array(band5_std).reshape(1, -1)
                band6_std = np.array(band6_std).reshape(1, -1)

                band1_skew = np.array(band1_skew).reshape(1, -1)
                band2_skew = np.array(band2_skew).reshape(1, -1)
                band3_skew = np.array(band3_skew).reshape(1, -1)
                band4_skew = np.array(band4_skew).reshape(1, -1)
                band5_skew = np.array(band5_skew).reshape(1, -1)
                band6_skew = np.array(band6_skew).reshape(1, -1)

                feature_vector = np.concatenate((band1_en, band1_max, band1_min, band1_mean, band1_std, band1_skew,
                                                 band2_en, band2_max, band2_min, band2_mean, band2_std, band2_skew,
                                                 band3_en, band3_max, band3_min, band3_mean, band3_std, band3_skew,
                                                 band4_en, band4_max, band4_min, band4_mean, band4_std, band4_skew,
                                                 band5_en, band5_max, band5_min, band5_mean, band5_std, band5_skew,
                                                 band6_en, band6_max, band6_min, band6_mean, band6_std, band6_skew
                                                 ), axis=0)

                data_training_nonseizure.append(np.transpose(feature_vector))

    print(len(data_training_nonseizure))
    print(data_training_nonseizure[0].shape)

    return data_training_nonseizure


if __name__ == "__main__":

    train_seizure_features = get_seizure_features(training_seizure_headers_dir,
                                                  training_seizure_annotations_dir,
                                                  training_seizure_data_dir)

    test_seizure_features = get_seizure_features(test_seizure_headers_dir,
                                                 test_seizure_annotations_dir,
                                                 test_seizure_data_dir)

    train_non_seizure_features = get_non_seizure_features(training_nonseizure_data_dir)
    test_non_seizure_features = get_non_seizure_features(training_nonseizure_data_dir)

    # Save Extracted Features.
    # np.save("train_seizure_features_36f_2sn", train_seizure_features)
    # np.save("test_seizure_features_36f_2sn", test_seizure_features)
    # np.save("train_non_seizure_features_36f_2sn", train_non_seizure_features)
    # np. save("test_non_seizure_36f_2sn", test_non_seizure_features)
