# Urban Sounds Classification using Neural Networks

## Authorship
**Authors**: Eduardo Passos, Pedro Fernandes, Rafael Pacheco  
**University**: Faculty of Science, University of Porto  
**Course**: Machine Learning II (CC3043)  
**Date**: 05/12/2024

## Description
This project explores the application of deep learning models for urban sound classification using the UrbanSound8K dataset. This dataset contains 8732 labeled audio recordings, each up to four seconds long, categorized into the following classes:

| Label              | Class ID |
|--------------------|----------|
| air conditioner    | 0        |
| car horn           | 1        |
| children playing   | 2        |
| dog bark           | 3        |
| drilling           | 4        |
| engine idling      | 5        |
| gun shot           | 6        |
| jackhammer         | 7        |
| siren              | 8        |
| street music       | 9        |

The project involves designing, training, and evaluating two deep learning classifiers:
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN)**

Additionally, we assess the models' robustness against adversarial attacks using the **DeepFool** algorithm.

---

## Solutions Implemented
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN) with LSTM units**

---

## Project Development Phases
### 1. **Data Preprocessing and Feature Extraction**
- **Data Preprocessing**: Includes zero-padding, resampling (22050Hz and 44100Hz), and MinMax scaling.
- **Feature Extraction**:
  - Mel-scaled spectrograms (2D arrays)
  - Chromagrams (2D arrays)
  - Spectral flatness, bandwidth, roll-off, centroid (1D arrays)
  - Log Mel-Scaled Spectrograms (for RNN)

### 2. **Model Development**
#### **Convolutional Neural Network (CNN)**
- **Architecture**:
  - Parallel processing of 1D and 2D features.
  - Leveraged convolutional and pooling layers to extract hierarchical features.
- **Performance Assessment**:
  - Cross-validation with confusion matrix analysis.
  - Achieved a training accuracy of 90.97% but validation accuracy of 69.27%, indicating potential overfitting.
  - Suggested improvements: Attention mechanisms and advanced data augmentation techniques.

#### **Recurrent Neural Network (RNN)**
- **Architecture**:
  - Utilized LSTM units for sequential pattern recognition.
  - Designed to identify time-dependent characteristics of urban sounds.
- **Performance Assessment**:
  - Cross-validation results showed a significant gap between training (70.04%) and validation accuracy (51.90%), highlighting overfitting challenges.
  - Suggested optimizations: Simpler architectures and increased dropout rates.

### 3. **Robustness Evaluation with DeepFool**
- **DeepFool Algorithm**:
  - Evaluates model vulnerability to adversarial examples by computing the smallest perturbation needed to misclassify inputs.
  - Applied to spectrogram representations of audio data.
- **Insights**:
  - The algorithm highlighted outliers and areas where decision boundaries were fragile.
  - Computational limitations prevented full implementation for the RNN.

---

## Results Summary
### CNN Results:
- **Training Accuracy**: 90.97% ± 1.38%
- **Validation Accuracy**: 69.27% ± 3.10%
- **Test Accuracy**: 66.27% ± 4.11%
- **Strengths**:
  - High accuracy for distinct sound classes like `gun_shot` and `dog_bark`.
- **Weaknesses**:
  - Struggled with acoustically similar classes like `engine_idling` and `air_conditioner`.

### RNN Results:
- **Training Accuracy**: 70.04% ± 2.52%
- **Validation Accuracy**: 51.90% ± 3.22%
- **Test Accuracy**: 49.01% ± 3.63%
- **Strengths**:
  - Captured time-dependent features well for some classes.
- **Weaknesses**:
  - Large gap between training and validation accuracy indicated overfitting.

### DeepFool Insights:
- Revealed vulnerabilities in decision boundaries for both models.
- Identified outliers affecting classification robustness.

---

## Conclusion
This project showcased the challenges and potential of applying deep learning models to urban sound classification. While the CNN model demonstrated better performance than the RNN, both struggled with acoustically similar classes and overfitting issues. The DeepFool algorithm provided valuable insights into model vulnerabilities and robustness. Future improvements could include:
- Advanced regularization techniques.
- Incorporation of attention mechanisms.
- More robust data augmentation strategies.


## References
- [Librosa](https://librosa.org/doc/latest/index.html)
- [MFCC Extraction](https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040)
- [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html)
- [Baeldung CS course](https://www.baeldung.com/cs/neural-networks-epoch-vs-iteration#3-batch)
- [Geeks for Geeks](https://www.geeksforgeeks.org/impact-of-learning-rate-on-a-model/)
- [ChatGPT](chatgpt.com)
- [Medium](https://cyborgcodes.medium.com/what-is-early-stopping-in-deep-learning-eeb1e710a3cf)
- [Machine Learning Mastery](https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/)
- [Data Camp](https://www.datacamp.com/tutorial/complete-guide-data-augmentation)
- [Bharati Vidyapeeth’s College of Engineering Paper](https://dergi.neu.edu.tr/index.php/aiit/article/download/740/327/3147) (automatically downloads pdf)
- [St.Petersburg Polytechnic University](https://annals-csis.org/Volume_18/drp/pdf/185.pdf)
- [Multi-Representation CNNs for Audio Classification](https://link.springer.com/article/10.1007/s11042-021-11610-8)
- [Ensemble CNNs for Audio Classification](https://www.mdpi.com/2076-3417/11/13/5796)
- [Braided CNNs](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/iet-spr.2019.0381)
- [Many-to-Many LSTM for Sequence Prediction with TimeDistributed Layers](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)
- [DeepFool algorithm](https://medium.com/machine-intelligence-and-deep-learning-lab/a-review-of-deepfool-a-simple-and-accurate-method-to-fool-deep-neural-networks-b016fba9e48e)



