# Speech-To-Text-Convertor

Given Project is a Residual Neural network trained on Basic speeches such as verbs and one line words in order to convert the given voice commands into text

We extensively use the Fourier Transform algorithm to convert the audio signal consisting of Frequency,amplitude and time to a Floating point Tensor consisting of only the frequencies
capped between a window size of 16Khz. This function is utilized and provided by the Tensorflow package and using this floating Tesnor we analyze it for pattersn using a 1D convolutions layers
And finally a dense layer to output the classification result

We achieve an accuracy of 99.8% on the validation dataset, however since the dataset is synthetically generated based on the persons voice however its still hard for the model to achieve better results
on real time dataset however it does come close to predict the real command from a voice recording in an uncontrolled environment
