# Execution-Time Performance of Deep Learning Networks on CPU, GPU and TPU Runtime Environments 

:o2: avoid using images, instaed use tables from markdown. Performabnce tables can be copied as is.


## Summary 

A performance review of execution times on Google Colab, for five deep learning network examples, was conducted on CPU, GPU and TPU runtime environments using the MNIST dataset.  The networks were 1) a multi-layer perceptron (MLP) network, 2) a convolutional neural network (CNN), 3) a recurrent neural network (RNN), 4) a long short-term memory network (LSTM), and 5) an autoencoder.   

## General findings 

Training times (Table 1) for all five network exemplars were significantly better on the GPU runtime environment than on Google Colab’s CPU environment. Of the networks, the CNN had the greatest performance improvement on GPUs than CPUs only, with a speedup of over 33 times (3332%). This was followed by the LSTM, which had a speedup of over 22 times (2257%), while speed ups for the autoencoder, MLP and RNN were 1464%, 697% and 229% respectively. 

Execution time performance for model testing was also significantly better on GPUs than CPUs, for the exemplars. Speedups for the LSTM, CNN, RNN, autoencoder and MLP where 1113%, 915%, 601%, 326%, and 177% respectively. 

The TPU runtime environment performed worse than the CPU environment, on training times for the autoencoder, RNN and CNN. Performance time declines were most significant for the autoencoder (-10%). TPU training times were nevertheless significantly better for the LSTM (+9%), and marginally better for the MLP (+1%), than on CPU runtime. All model exemplars performed worse on model evaluation times, on TPUs than on CPUs. 

## Discussion 

To leverage advantages of using TPUs, optimizations could have been applied to the code used for the performance evaluations [^1]. Nevertheless, no customizations were made to the code used, for a head-to-head comparison in the environments. The network code examples were simply run under the three runtime environment options by changing the relevant Colab notebook settings.

## Appendix:

![Table 1](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/runtime_performance.jpg)
**Table 1:** Summary of CPU, GPU, TPU Performance


### Multi-Layer Perceptron (MLP) Example using MNIST Dataset

**Figure 1:** MLP using CPUs only
![Figure 1](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/mlp_cpu.jpg)


**Figure 2:** MLP using GPUs
![Figure 2](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/mlp_gpu.jpg)


**Figure 3:** MLP using TPUs
![Figure 3](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/mlp_tpu.jpg)


### Convolutional Neural Networks (CNN) Example using MNIST Dataset

**Figure 4:** CNN using CPUs only
![Figure 4](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/cnn_cpu.jpg)


**Figure 5:** CNN using GPUs
![Figure 5](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/cnn_gpu.jpg)


**Figure 6:** CNN using TPUs
![Figure 6](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/cnn_tpu.jpg)


### Recurrent Neural Networks (RNN) Example using MNIST Dataset

**Figure 7:** RNN using CPUs only
![Figure 7](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/rnn_cpu.jpg)


**Figure 8:** RNN using GPUs
![Figure 8](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/rnn_gpu.jpg)


**Figure 9:** RNN using TPUs
![Figure 9](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/rnn_tpu.jpg)


### Long Short-Term Memory (LSTM) Example using MNIST Dataset

**Figure 10:** LSTM using CPUs only
![Figure 10](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/lstm_cpu.jpg)


**Figure 11:** LSTM using GPUs
![Figure 11](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/lstm_gpu.jpg)


**Figure 12:** LSTM using TPUs
![Figure 12](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/lstm_tpu.jpg)


### Autoencoder Example using MNIST Dataset

**Figure 13:** Autoencoder using CPUs only
![Figure 13](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/autoencoder_cpu.jpg)


**Figure 14: Autoencoder using GPUs**
![Figure 14](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/autoencoder_gpu.jpg)


**Figure 15:** Autoencoder using TPUs
![Figure 15](https://github.com/cybertraining-dsc/sp21-599-359/raw/develop/Assignments/images/autoencoder_tpu.jpg)


## References

[^1]. Google. (2021, 03 26). TPUs in Colab. Retrieved from [https://colab.research.google.com/:] (https://colab.research.google.com/notebooks/tpu.ipynb#scrollTo=kvPXiovhi3ZZ)

