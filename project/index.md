# Project: Deep Learning in Drug Discovery

[![Check  Report](https://github.com/cybertraining-dsc/sp21-599-359/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/sp21-599-359/actions)
[![Status](https://github.com/cybertraining-dsc/sp21-599-359/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/sp21-599-359/actions)
Status: final, Type: Project

:o2: figure caltions come underneath the figure

:oe: due to markdown limitations refernce section must be last

Anesu Chaora, [sp21-599-359](https://github.com/cybertraining-dsc/sp21-599-359/), [Edit](https://github.com/cybertraining-dsc/sp21-599-359/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract

Machine learning has been a mainstay in drug discovery for decades. Artificial neural networks have been used in computational approaches to drug discovery since the 1990s [^1]. Under the traditional approaches, emphasis in drug discovery was placed on understanding chemical molecular fingerprints in order to predict biological activity. More recently, however, deep learning approaches have been adopted instead of computational methods. This paper outlines work conducted on predicting drug molecular activity using deep learning approaches.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** Deep Learning, drug discovery. 

## 1. Introduction

Please note that an up to date version of these instructions is available at

* <https://github.com/cybertraining-dsc/sp21-599-359/blob/main/project/index.md>

*De novo molecular design* 

Deep learning (DL) is finding new uses in developing novel chemical structures. Methods that employ variational autoencoders (VAE) have been used to generate new chemical structures by 1) encoding input string molecule structures, 2) reparametrizing the underlying latent variables and then 3) searching for viable solutions in the latent space, by using methods such as Bayesian optimizations. A final step involves decoding the results back into simplified molecular-input line-entry system (SMILES) notation, for recovery of molecular descriptors. Variations to this involve using generative adversarial networks (GAN), as subnetworks in the architecture, to generate the new chemical structures [^8].

Other methods for developing new chemical structures include use of recurrent neural networks (RNN) to generate new valid SMILES strings, after training the RNNs on large quantities of known SMILES datasets. The RNNs use probability distributions learned from training sets, to generate new strings that correspond to new molecular structures [^13]. Variations to this approach incorporate reinforcement learning to reward models for new chemical structures, while punishing them for undesirable results [^16].

*Bioactivity prediction* 

Computational methods have been used in drug development for decades [^7]. The emergence of high-throughput screening (HTS), in which automated equipment is used to conduct large assays of scientific experiments on molecular compounds in parallel, has resulted in generation of enormous amounts of data that require processing.  Quantitative structure activity relationship (QSAR) models for predicting the biological activity responses to physiochemical properties of predictor chemicals, regularly use machine learning models like support vector machines (SVM) and random decision forests (RF) for this processing [^8], [^5].

While deep learning (DL) approaches have an advantage over single-layer machine learning methods, when predicting biological activity responses to properties of predictor chemicals, they have only recently been used for this [^8]. The need to interpret how predictions are made through computationally-oriented drug discovery, is seen - in part - as a factor to why DL approaches have not been adopted as quickly in this area [^6]. However, because DL models can learn complex non-linear data patterns, using their multiple hidden layers to capture patterns in data, they are better suited for processing complex life sciences data than other machine learning approaches [^6].

Their applications have included profiling tumors at molecular level, and predicting drug response based on pharmacological and biological molecular structures, functions and dynamics. This is attributed to their ability to handle high dimensionality in data features, making them appealing for use in predicting drug response [^5]. 

For example, DL models were found to perform better than standard RF models [^1] in predicting the biological activities of molecular compounds, using datasets from the Merck Molecular Activity Challenge on Kaggle [^11]. Deep neural networks were also used in models that won NIH’s Toxi21 Challenge [^2] on using chemical structure data only  to predict compounds of concern to human health [^3].
 

## 2. Related Work

### 2.1. Merck Molecular Activity Challenge on Kaggle

A challenge to identify the best statistical techniques for predicting molecular activity was issued by Merck & Co Pharmaceutical, through Kaggle in October of 2012.   The stated goal of the challenge was to ‘help develop safe and effective medicines by predicting molecular activity’ for effects that were both on and off target[^11].

### 2.2. The Dataset

A [dataset](https://www.kaggle.com/c/MerckActivity/data) was provided for the challenge [^11]. It consisted of 15 molecular activity datasets. Each dataset contained rows corresponding to assays of biological activity for chemical compounds. The datasets were subdivided into training and test set files. The training and test dataset split was done by dates of testing[^11], with test set dates consisting of assays conducted after the training set assays.

The training set files each had a column with molecular descriptors that were formulated from chemical molecular structures. A second column in the files contained numeric values, corresponding to raw activity measures. These were not normalized, and indicated measures in different units.

The remainder of the columns in each training dataset file indicated disguised substructures of molecules. Values in each row, under the substructure (atom pair and donor-acceptor pair) codes, corresponded to the frequencies at which each of the substructures appeared in each compound. Figure 1 shows part of the head row for one of the training dataset files, and the first 5 records in the file.

**Figure 1**: Head Row of 1 of 15 Training Dataset files
![Figure 1](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/project/images/training_set.jpg) 

The test dataset files were similar (Figure 2) to the training files, except they did not include the column for activity measures. The challenge presented was to predict the activity measures for the test dataset.

**Figure 2**: Head Row of 1 of 15 Test Dataset files
![Figure 2](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/project/images/test_set.jpg) 

### 2.2. A Deep Learning Algorithm

The entry that won the Merck Molecular Activity Challenge on Kaggle used an ensemble of methods that included a fully connected neural network as the main contributor to the high accuracy in predicting molecular activity[^1]. Evaluations of predictions for molecular activity for the test set assays were then determined using the mean of the correlation coefficient (R2) of the 15 data sets. Sample code in R was provided for evaluating the correlation coefficient. The code, and formula for R2 are appended in Appendix 1.

The approach of employing convolutional networks on substructures of molecules, as a way to concentrate learning on localized features, while reducing the number of parameters in the overall network, holds promise in improving the molecular activity predictions. This methodology of identifying molecular substructures as graph convolutions, prior to further processing, is proposed by a number of authors [^12], [^15].
An ensemble of networks for predicting molecular activity will be built, using the Merck dataset and the approaches listed above. Hyperparameter choices that were found optimal by the cited authors, and recognized optimal activation functions, for different neural network types and prediction types [^4], will also be used.

## 3. Project Timeline

The timeline for execution of the above is as follows: A working fully connected network solution will be developed by April 9th, 2021. This will be followed by an ensemble version that employs a convolutional neural network with the fully connected network. This is scheduled to be complete by April 23rd. 
A full report of the work done will be completed by May 2nd , 2021.


## 4. Benchmark

TODO: Your project must include a benchmark. The easiest is to use cloudmesh-common [^2]
 
## 5. Conclusion

TODO: A convincing but not fake conclusion should summarize what the conclusion of the project is.

## 6. Acknowledgments

TODO: Please add acknowledgments to all that contributed or helped on this project.

## 7. References

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



## Appendix 1

Correlation Coefficient (R2) Formula:

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

