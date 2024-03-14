# Epileptic_Seizure_Detection
Epileptic Seizure Detection on EEG Data based on CHB-MIT database using Discrete Wavelet Transform with wavelet family 'coif3', 7 level decomposition. Training is done by SVM and Random Forest.

36 Features are extracted from each subband and 23 channels. After DWT decomposition, I calculated Max, Min, Mean, Energy, Standard deviation and Skewness features for 6 subbands. 6 x 6 = 36 features are extracted for just 1 channel. Since we have 23 channel, every feature vector has 23x36 dimension.

Data source for .mat files: https://archive.physionet.org/cgi-bin/atm/ATM

RESULTS: \
****************************** \
Linear SVM \
\
Sensitivity: % 92.0042643923241 \
Specificity: % 79.98338870431894 \
Positive Predictive Val: % 78.17028985507247 \
Negative Predictive Val: % 92.77456647398844 \
False Positive Rate: % 20.016611295681063 \
False Negative Rate: % 7.995735607675907 \
False Discovery Rate: % 21.829710144927535 \
False Omission Rate: % 7.225433526011561 \
Accuracy: % 85.24743230625583 \
****************************** \
SVM with RBF (gamma=0.1) \
 \
Sensitivity: % 89.69276511397423 \
Specificity: % 82.4360105913504 \
Positive Predictive Val: % 81.97463768115942 \
Negative Predictive Val: % 89.98073217726397 \
False Positive Rate: % 17.5639894086496 \
False Negative Rate: % 10.307234886025768 \
False Discovery Rate: % 18.02536231884058 \
False Omission Rate: % 10.01926782273603 \
Accuracy: % 85.85434173669468 \
****************************** \
Random Forest (n_estimators=20, random_state=0) \
 \
Sensitivity: % 97.73584905660377 \
Specificity: % 93.71534195933457 \
Positive Predictive Val: % 93.84057971014492 \
Negative Predictive Val: % 97.6878612716763 \
False Positive Rate: % 6.284658040665435 \
False Negative Rate: % 2.2641509433962264 \
False Discovery Rate: % 6.159420289855073 \
False Omission Rate: % 2.312138728323699 \
Accuracy: % 95.70494864612512


