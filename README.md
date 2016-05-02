# POS-tagger

In this project, I implemented an automatic pos-tagger based on structured average perceptron. About 14000 features were mined from given corpus and their weight parameters were trained iteratively. Using this perceptron model, the pos-tag for each word of the development and test set were automatically predicted in a reasonable way.

My code used NumPy to do vector calculation. The model is built in a python module package called “postagger”. Codes to fetch rules are presented in `\getrules` file. All the data is in `\dataset` file including training, developing and testing words and pos-tags.

Because I used 2-dimension Viterbi algorithm to get optimized solution rather than 1-dimension, the training process is __VERY SLOW__ and it can only predict 4 tags, which is annoying. Maybe later I will modify it to 1-dimension.

The project report is presented in root file, which provides more details.
