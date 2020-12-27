# Epileptic_Seizure_Detection
Epileptic Seizure Detection on EEG Data based on CHB-MIT database using Discrete Wavelet Transform with wavelet family 'coif3', 7 level decomposition.

36 Features are extracted from each subband and 23 channels. After DWT decomposition, I calculated Max, Min, Mean, Energy, Standard deviation and skewness features for 6 subbands. 6 x 6 = 36 features are extracted for just 1 channel. Since we have 23 channel, every feature vector has 23x36 dimension.

RESULTS:

Accuracy: 0.8524743230625583 \
Precision:  0.920042643923241 \
Recall:  0.7817028985507246 
