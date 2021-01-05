# Description of sentiment_embedding repo
Code base for experimentation with FastText pre-trained model embedding (vs. baseline of TFIDF); compares older NLP approaches of text classification (TFIDF) to an approach using a pre-trained FastText embedding model

## Data
* Labelled (pos/neg) product reviews from three different sources (Amazon, IMDB, Yelp)
* Data downloaded from UCI ML Repo: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences


## Requirements
- Packages can be installed using conda or pip
- `Python=3.6` or higher
    * `pandas`
    * `numpy`
    * `math`
    * `matplotlib`
    * `string`
    * `uuid`
    * `csv`
    * `scikit-learn`
    * `nltk`
    * `xgboost`
    * `fasttext`


## Project structure
    
- `/src`
    * `sentiment_utilities.py`: Data operations (e.g., text cleaning), data processing (e.g., embedding, TFIDF), data splitting (e.g., into training and validation or folds)
    * `Sentiment_Experiments.ipynb`: Unsupervised analysis and visualizations; model training and evaluations
 
   

## Outstanding tasks/experiments
* Additional model tuning for both TFIDF models and pre-trained embedding models
* Additional model training using FastText library (i.e., training a model using FastText vs. merely using embeddings from pre-trained model)



