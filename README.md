# Abstract
With recent advances in the ﬁeld of machine learning, the use of deep neural networks for time series forecasting has become more prevalent. The periodic nature of the solar cycle makes it a good candidate for applying time series forecasting methods. We employ a combination of WaveNet and LSTM to forecast the sunspot number for the years 1749 to 2019 and total sunspot area for the years 1874 to 2019 time series data for the upcoming solar cycle SC25. Three other models involving the use of LSTMs and 1D ConvNets are also compared with our best model. Our analysis shows that the WaveNet and LSTM model is able to better capture the overall trend and learn the inherent long and short term dependencies in time series data. Using this method we forecast 11years of monthly averaged data for SC-25. Our forecasts show that the upcoming solar cycle SC-25 will have a maximum sunspot number around 106 ± 19.75 and maximum total sunspot area around 1771 ± 381.17. This indicates that the cycle would be slightly weaker than solar cycle SC-24.

### References
1. https://www.tensorflow.org/tutorials/structured_data/time_series
2. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
by Aurélien Géron
