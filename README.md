# Arabic_DialectClassifier
This repository is a Python implementation for Arabic Dialect Classificaiton based on the dataset in [link to paper](https://arxiv.org/abs/2005.06557)

## Dataset
The dataset contains a wide range of country-level Arabic dialects retrieved from Twitter —covering 18 different countries in the Middle East and North Africa region.The
resultant dataset contains 540k tweets from 2,525 users who are evenly distributed across 18 Arab countries.

## Data Pre-processing
Removing the username tag,emojis,URL,punctuation using Regular Expression, and emoji library, Removing stop words from the text to eliminate the low-level information and give the model the ability to focus on the important information using NLTK library.

## Machine Learning Model [Link to notebook](https://github.com/AbdallahNasr5/Arabic_DialectClassifier/blob/main/Arabic_DialectClassifier/MachineLearning_Model.ipynb)
After training multiple models to choose the final model to train the whole data on, LinearSVC shined the most out of all other models.
-Using TF-IDF vectorization to transform the data with respect to word count and importance.

## DeepLearning models using keras embeddings [link to notebook](https://github.com/AbdallahNasr5/Arabic_DialectClassifier/blob/main/Arabic_DialectClassifier/DeepLearning_Model.ipynb)
In this section we used 2 architictures, CNN and CNN&LSTM pipeline.
The CNN model outperformed the CNN&LSTM model for this task with the same number of parameters.

## DeepLearning model with word2vector embedding [link to notebook](https://github.com/AbdallahNasr5/Arabic_DialectClassifier/blob/main/Arabic_DialectClassifier/Word_To_Vector_DLModel.ipynb)
In this notebook we trained a word2vector model and used it in the embedding layer after converting it to a matrix with respect to the training set corpus.

# Conclusion
The model that outperformed is the LinearSVC model with the highest f1 scores and accuracy score of 53%

## FastApi [link to script](https://github.com/AbdallahNasr5/Arabic_DialectClassifier/blob/main/Arabic_DialectClassifier/FastAPI/FastAPI.py)
To use this api all you need to do is download the LinearSVC pretrained model [pre-trained model](https://drive.google.com/file/d/14fPruMC10ISRoGM11tmABHEouKA8YRG5/view?usp=sharing)
and the tf-idf trained on the dataset corpus [tfidf_vectorizer](https://drive.google.com/file/d/1Nr3ia7IavWhNN8V-2VbjX1CzlR96n24Y/view?usp=sharing)
