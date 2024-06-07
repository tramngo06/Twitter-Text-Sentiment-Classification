# Twitter-Text-Sentiment-Classification
![image](https://github.com/tramngo06/Twitter-Text-Sentiment-Classification/assets/139295222/471ce4b0-2521-4f2f-8a3d-29a573c03546)

## Overview
#### This project purpose is using the trained model on Hugging Face to train a new dataset of texts on Twitter, and classifying them into positive, negative, or neural text. The **dataset** and **model** are used and downloaded from HuggingFace. HuggingFace is a website where it contained the pretrained model to train the new dataset. 
#### For example: 
#### * "Love the trip in German. Together with my imported honey it feels like home" --> It's classifed as Positive Text.
#### * "Finished fixing my twitter.... I had to unfollow everyone" --> It's classfied as Negative Text.
#### **Data:** https://huggingface.co/datasets/tweet_eval
#### **Model:** https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

## Software Requirement
#### Python3: Recommended installation via Anaconda for additional libraries.
#### Libraries: NumPy, NLTK (with all sub-packages), SciPy, and scikit-learn.

## Methodology
#### This project consists RoBERTa-base model (Bidirectional Encoder Representations from Transformers) that has been optimized for Twitter data. 
#### We apply the following pre-processing steps before feeding the data into the classifier:
##### * Using AutoTokenizer from transformer model library
##### * Tokenize the text: Tokenizer will split the text into tokens, convert them into numerical IDs, and create the necessary attention masks.
##### * Inspect the output: The tokenizer returns a dictionary with token IDs, attention masks, and other model-specific inputs
