# Classiflight

An airflight-question classification system.

## About Classiflight
A classification system with an user interface for the question being asked by customer to an airflight company.

## Getting Started

### Prerequisites
Library:
1. `streamlit`
2. `fasttext`
3. `Python 3.8` or above
4. `gensim`
5. `nltk`
6. `scikit-learn`
7. `WordCloud` (not really needed, unless you need to run the notebooks) 

## Usage
### Notebooks
The Jupyter Notebooks are the code where we use it for training and preprocess our data. 

The model that we had trained:
- [fasttext](https://fasttext.cc/) text classification
- Word2Vec with [Support Vector Classificator (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

The text preprocessing that we had used:
- Word2Vec
    - Lowercase
    - Lemmatization
    - remove stop words
    - vectorise the words with pretrained model (`gensim glove-twitter-50`)
- fasttext
    - lowercase
    - remove stop words
    - preprocess with the required format (`__label__{label} {statement}`)

### User Interface
`Home.py` is the user interface for this project. In order to access it locally, you should use the command below when you are in this directory:
```sh
streamlit run ./Home.py
```

Then you are able to run the webpage. If you error is being thrown, the biggest possibility is that you miss some library that you haven't download. 

If you're interested in the online webpage, then you can click [here](https://lhz0616-classiflight-home-6hzka5.streamlit.app/).


## Acknowledgement
This is a project where it is used to fulfil the assignment that is assigned in WID3002 Natural Language Processing at [University of Malaya](https://www.um.edu.my/).

Special thanks to the author of the dataset that we had used in this project, [ATIS Airline Travel Information System](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem) from Kaggle. 

Although the dataset is not robust enough to be applied in real-world scenario, it can satisfy the requirement of the assignment and able to illustrate the usefulness of this application.

Finally I would like to appreciate my teammates hard work in this project, they are: 
1. [Darren Sow Zhu Jian](https://github.com/Darrensow)
2. [Cheong Yi Fong](https://github.com/CHEONG-YI-FONG)
3. [Khor Zhi Qian](https://github.com/Keyu0625)
4. [Lim Wei Sze](https://github.com/weisze-yo)
5. [Chew Yao Dong](https://github.com/yaodongchew)

