# Project: Deep Learning in Drug Discovery

[![Check  Report](https://github.com/cybertraining-dsc/sp21-599-359/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/sp21-599-359/actions)
[![Status](https://github.com/cybertraining-dsc/sp21-599-359/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/sp21-599-359/actions)
Status: final, Type: Project


Anesu Chaora, [sp21-599-359](https://github.com/cybertraining-dsc/sp21-599-359/), [Edit](https://github.com/cybertraining-dsc/sp21-599-359/blob/main/project/index.md)

* Code: [predicting_molecular_activity.ipynb](https://github.com/cybertraining-dsc/sp21-599-359/blob/main/project/code/predicting_molecular_activity.ipynb)

{{% pageinfo %}}

## Abstract

Machine learning has been a mainstay in drug discovery for decades. Artificial neural networks have been used in computational approaches to drug discovery since the 1990s [^1]. Under traditional approaches, emphasis in drug discovery was placed on understanding chemical molecular fingerprints, in order to predict biological activity. More recently however, deep learning approaches have been adopted instead of computational methods. This paper outlines work conducted in predicting drug molecular activity, using deep learning approaches.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** Deep Learning, drug discovery. 

## 1. Introduction

### 1.1. De novo molecular design 

Deep learning (DL) is finding uses in developing novel chemical structures. Methods that employ variational autoencoders (VAE) have been used to generate new chemical structures. Approaches have involved encoding input string molecule structures, then reparametrizing the underlying latent variables, before searching for viable solutions in the latent space by using methods such as Bayesian optimizations. The results are then decoded back into simplified molecular-input line-entry system (SMILES) notation, for recovery of molecular descriptors. Variations to this method involve using generative adversarial networks (GAN)s, as subnetworks in the architecture, to generate the new chemical structures [^8].

Other approaches for developing new chemical structures involve recurrent neural networks (RNN), to generate new valid SMILES strings, after training the RNNs on copious quantities of known SMILES datasets. The RNNs use probability distributions learned from training sets, to generate new strings that correspond to molecular structures [^13]. Variations to this approach incorporate reinforcement learning to reward models for new chemical structures, while punishing them for undesirable results [^16].

### 1.2. Bioactivity prediction 

Computational methods have been used in drug development for decades [^7]. The emergence of high-throughput screening (HTS), in which automated equipment is used to conduct large assays of scientific experiments on molecular compounds in parallel, has resulted in generation of enormous amounts of data that require processing. Quantitative structure activity relationship (QSAR) models for predicting the biological activity responses to physiochemical properties of predictor chemicals, extensively use machine learning models like support vector machines (SVM) and random decision forests (RF) for processing [^8], [^5].

While deep learning (DL) approaches have an advantage over single-layer machine learning methods, when predicting biological activity responses to properties of predictor chemicals, they have only recently been used for this [^8]. The need to interpret how predictions are made through computationally oriented drug discovery, is seen - in part - as a factor to why DL approaches have not been adopted as quickly in this area [^6]. However, because DL models can learn complex non-linear data patterns, using their multiple hidden layers to capture patterns in data, they are better suited for processing complex life sciences data, than other machine learning approaches [^6].

Their applications have included profiling tumors at molecular level and predicting drug responses, based on pharmacological and biological molecular structures, functions, and dynamics. This is attributed to their ability to handle high dimensionality in data features, making them appealing for use in predicting drug response [^5].

For example, deep neural networks were used in models that won NIH’s Toxi21 Challenge [^2] on using chemical structure data only to predict compounds of concern to human health [^3]. DL models were also found to perform better than standard RF models [^1] in predicting the biological activities of molecular compounds in the Merck Molecular Activity Challenge on Kaggle [^11]. Details of the challenge follow.


## 2. Related Work

### 2.1. Merck Molecular Activity Challenge on Kaggle

A challenge to identify the best statistical techniques for predicting molecular activity was issued by Merck & Co Pharmaceutical, through Kaggle in October of 2012. The stated goal of the challenge was to ‘help develop safe and effective medicines by predicting molecular activity’ for effects that were both on and off target [^11].

### 2.2. The Dataset

A [dataset](https://www.kaggle.com/c/MerckActivity/data) was provided for the challenge [^11]. It consisted of 15 molecular activity datasets. Each dataset contained rows corresponding to assays of biological activity for chemical compounds. The datasets were subdivided into training and test set files. The training and test dataset split was done by dates of testing [^11], with test set dates consisting of assays conducted after the training set assays.

The training set files each had a column with molecular descriptors that were formulated from chemical molecular structures. A second column in the files contained numeric values, corresponding to raw activity measures. These were not normalized, and indicated measures in different units.

The remainder of the columns in each training dataset file indicated disguised substructures of molecules. Values in each row, under the substructure (atom pair and donor-acceptor pair) codes, corresponded to the frequencies at which each of the substructures appeared in each compound. Figure 1 shows part of the head row for one of the training dataset files, and the first records in the file.


![Figure 1](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/project/images/training_set.jpg) 
**Figure 1**: Head Row of 1 of 15 Training Dataset files


The test dataset files were similar (Figure 2) to the training files, except they did not include the column for activity measures. The challenge presented was to predict the activity measures for the test dataset.


![Figure 2](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/project/images/test_set.jpg) 
**Figure 2**: Head Row of 1 of 15 Test Dataset files


### 2.3. A Deep Learning Algorithm

The entry that won the Merck Molecular Activity Challenge on Kaggle used an ensemble of methods that included a fully connected neural network as the main contributor to the high accuracy in predicting molecular activity [^1]. Evaluations of predictions for molecular activity for the test set assays were then determined using the mean of the correlation coefficient (R2) of the 15 data sets. Sample code in R was provided for evaluating the correlation coefficient. The code, and formula for R2 are appended in Appendix 1.

An approach of employing convolutional networks on substructures of molecules, to concentrate learning on localized features, while reducing the number of parameters in the overall network, was also proposed in literature on improving molecular activity predictions. This methodology of identifying molecular substructures as graph convolutions, prior to further processing, was discussed by authors [^12], [^15].

In line with the above research, an ensemble of networks for predicting molecular activity was planned for this project, using the Merck dataset, and hyperparameter configurations found optimal by the cited authors. Recognized optimal activation functions, for different neural network types and prediction types [^4], were also earmarked for use on the project.


## 4. Project Implementation

Implementation details for the project were as follows:

### 4.1. Tools and Environment

The Python programming language (version 3.7.10) was used on Google Colab (https://colab.research.google.com).

A subscription account to the service was employed, for access to more RAM (High-RAM runtime shape) during development, although the free standard subscription will suffice for the version of code included in this repository.

Google Colab GPU hardware accelerators were used in the runtime configuration.

Prerequisites for the code included packages from [Cloudmesh](http://cloudmesh.github.io/ ), for benchmarking performance, and from [Kaggle](https://www.kaggle.com/ ), for API access to related data.

Keras libraries were used for implementing the molecular activity prediction model.

### 4.2. Implementation Overview

This project's implementation of a molecular activity prediction model consisted of a fully connected neural network.  The network used the Adam [^18] optimization algorithm, at a learning rate of 0.001 and beta_1 calibration of 0.5. Mean Squared Error (MSE) was used for the loss function, and R-Squared [^19] for the metric. Batch sizes were set at 128. These parameter choices were selected by referencing the choices of other prior investigators [^17].

The network was trained on the 15 datasets separately, by iterating through the storage location containing preprocessed data, and sampling the data into training, evaluation and prediction datasets - before running the training. The evaluation and prediction steps, for each dataset, where also executed during the iteration of each molecular activity dataset. Running the processing in this way was necessitated by the fact that the 15 datasets each had different feature set columns, corresponding to different molecular substructures. As such, they could not be readily processed through a single dataframe.

An additional compounding factor was that the data was missing the molecular activity results (actual readings) associated with the dataset provided for testing. These were not available through Kaggle as the original competition withheld these from contestants, reserving them as a means for evaluating the accuracy of the models submitted. In the absence of this data, for validating the results of this project, the available training data was split into samples that were when used for the exercise. The training of the fully connected network was allocated 80% of the data, while the testing/evaluation of the model was allocated 10% of the data. The remaining data (10%) was used for evaluating predictions.

### 4.3. Benchmarks

Benchmarks captured during code execution, using cloudmesh-common [^2], were as follows:

* The data download process from Kaggle, through the Kaggle data API, took 29 seconds.

* Data preprocessing scripts took 8 minutes and 56 seconds to render the data ready for training and evaluation. Preprocessing of data included iterating through the issued datasets separately, since each file contained different combinations of feature columns (molecular substructures).

* The model training, evaluation and prediction step took 7 minutes and 45 seconds.

### 4.4. Findings

The square of the correlation coefficient (R^2) values obtained (coefficient of determination) [^20] during training and evaluation were considerably low (< 0.1). A value of one (1) would indicate a goodness of fit for the model that implies that the model is completely on target with predicting accurate outcomes (molecular activity) from the independent variables (substructures/feature sets). Such a model would thus fully account for the predictions, given a set of substructures as inputs. A value of zero (0) would indicate a total lack of correlation between the input feature values and the predicted outputs. As such, it would imply that there is a lot of unexplained variance in the outputs of the model. The square of the correlation coefficient values obtained for this model (<0.1) therefore imply that it either did not learn enough, or other unexplained (by the model) variance caused unreliable predictions. 


## 5. Discussion

An overwhelming proportion of the data elements provided through the datasets were zeros (0)s, indicating that no frequencies of the molecular substructures/features were present in the molecules represented by particular rows of data elements. This disproportionate representation of absent molecular substructure frequencies, versus the significantly lower instances where there were frequencies appears to have had an effect of dampening the learning of the fully connected neural network.

This supports approaches that advocated for the use of convolutional neural networks [^12], [^15] as auxiliary components to help focus learning on pertinent substructures. While the planning phase of this project had incorporated inclusion of such, the investigator ran out of time to implement an ensemble network that would include the suggestions.

Apart from employing convolutions, other preprocessing approaches for rescaling, and normalizing, the data features and activations [^17] could have helped the learning, and subsequently the predictions made. This reinforces the fact that deep learning models, as is true with other machine learning approaches, rely deeply on the quality of data fed into them.


## 6. Conclusion

Deep learning is a very powerful new approach to solving many machine learning problems, including some that have eluded solutions till now. While deep learning models offer robust and sophisticated ways of learning patterns in data, they are still only half the story. The quality and appropriate preparation of the data fed into models is equally important when seeking to have meaningful results.


## 7. Acknowledgments

Acknowledgements go to Dr. Geoffrey Fox for his excellent guidance on ways to think about deep learning approaches, and for his instructorship of the course 'ENG-E599: AI-First Engineering', for which this project is a deliverable. Acknowledgements also go to Dr. Gregor von Laszewski for his astute tips and recommendations on technical matters, and on coding and documention etiquette.


## 8. Appendix

Square of the Correlation Coefficient (R2) Formula:

![Figure 3](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/project/images/correlation_coefficient.jpg)
[^11]

Sample R2 Code in the R Programming Language:
```
Rsquared <- function(x,y) {
  # Returns R-squared.
  # R2 = \frac{[\sum_i(x_i-\bar x)(y_i-\bar y)]^2}{\sum_i(x_i-\bar x)^2 \sum_j(y_j-\bar y)^2}
  # Arugments: x = solution activities
  #            y = predicted activities
  if ( length(x) != length(y) ) {
    warning("Input vectors must be same length!")
  }
  else {
    avx <- mean(x)
    avy <- mean(y)
    num <- sum( (x-avx)*(y-avy) )
    num <- num*num
    denom <- sum( (x-avx)*(x-avx) ) * sum( (y-avy)*(y-avy) )
    return(num/denom)
  }
}
```
[^11]


## References

[^1]: Junshui Ma, R. P. (2015). Deep Neural Nets as a Method for Quantitative Structure-Activity Relationships. Journal of Chemical Information and Modeling, 263-274.


[^2]:  National Institute of Health. (2014, November 14). Tox21 Data Challenge 2014. Retrieved from [tripod.nih.gov:] <https://tripod.nih.gov/tox21/challenge/>


[^3]: Andreas Mayr, G. K. (2016). Deeptox: Toxicity Prediction using Deep Learning. Frontiers in Environmental Science.


[^4]: Bronlee, J. (2021, January 22). How to Choose an Activation Function for Deep Learning. Retrieved from [https://machinelearningmastery.com:] <https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/>


[^5]: Delora Baptista, P. G. (2020). Deep learning for drug response prediction in cancer. Briefings in Bioinformatics, 22, 2021, 360–379.


[^6]: Erik Gawehn, J. A. (2016). Deep Learning in Drug Discovery. Molecular Informatics, 3 - 14.


[^7]: Gregory Sliwoski, S. K. (2014). Computational Methods in Drug Discovery. Pharmacol Rev, 334 - 395.


[^8]: Hongming Chen, O. E. (2018). The rise of deep learning in drug discovery. Elsevier.


[^9]: Jacobs, V. S. (2019). Deep learning and radiomics in precision medicine, Expert Review of Precision Medicine and Drug Development. In Expert Review of Precision Medicine and Drug Development: Personalized medicine in drug development and clinical practice (pp. 59 - 72). Informa UK Limited, trading as Taylor & Francis Group.


[^10]: Vishwa S. Parekh, M. A. (2018). MPRAD: A Multiparametric Radiomics Framework. ResearchGate.


[^11]: Kaggle. (n.d.). Merck Molecular Activity Challenge. Retrieved from [Kaggle.com:] <https://www.kaggle.com/c/MerckActivity>


[^12]: Kearnes, S., McCloskey, K., Berndl, M., Pande, V., & Riley, P. (2016). Molecular graph convolutions: moving beyond fingerprints. Switzerland: Springer International Publishing .


[^13]: Marwin H. S. Segler, T. K. (2018). Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks. America Chemical Society.


[^14]: MedlinePlus. (2020, September 22). What is precision medicine? Retrieved from [https://medlineplus.gov/:] <https://medlineplus.gov/genetics/understanding/precisionmedicine/definition/>


[^15]: Mikael Henaff, J. B. (2015). Deep Convolutional Networks on Graph-Structured Data.


[^16]: N Jaques, S. G. (2017). Sequence Tutor: Conservative Fine-Tuning of Sequence Generation Models with KL-control. Proceedings of the 34th International Conference on Machine Learning, PMLR (pp. 1645-1654). MLResearchPress.


[^17]: RuwanT (2017, May 16). Merk. Retrieved from [https://github.com:] <https://github.com/RuwanT/merck/blob/master/README.md>


[^18]: Keras. (2021). Adam. Retrieved from [https://keras.io:] <https://keras.io/api/optimizers/adam/>


[^19]: Keras. (2021). Regression Metrics. Retrieved from [https://keras.io:] <https://keras.io/api/metrics/regression_metrics/>


[^20]: Wikipedia (2021). Coefficient of Determination. Retrieved from [https://wikipedia.org:] <https://en.wikipedia.org/wiki/Coefficient_of_determination>
