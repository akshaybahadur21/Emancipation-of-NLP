# Course1 | Natural Language Processing with Classification and Vector Spaces

# Week1 : Logistic Regression

## Sentiment Analysis

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled.png)

You have a text → Extract features from text → Train Logistic Regression → Classify (1 : Positive and 0: Negative)

### Building a vocabulary

You could do it in a couple of ways. 

- **Sparse representation**

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%201.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%201.png)

    Problems with sparse representation - A lot of features will have zeros since each vector is the entire dictionary 

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%202.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%202.png)

- **Positive and negative frequencies** - For each sentiment, you calculate the most frequently occurring words.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%203.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%203.png)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%204.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%204.png)

    So once we calculate, we will get a table as displayed below

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%205.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%205.png)

    So we can actually improve our learning since instead of learning 'V' features now, we can learning only 3 features.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%206.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%206.png)

    Let's use the definition to generate our vector representation

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%207.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%207.png)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%208.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%208.png)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%209.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%209.png)

### Preprocessing

- Stemming - Transforming any word to its base stem(used to construct the word and it's derivatives)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2010.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2010.png)

- Stop words - Remove nonsensical words and punctuation marks. Create a list of stopwords and punctuation → Remove all the elements which are in those 2 lists. (The meaning will remain the same. )

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2011.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2011.png)

    We can even remove all the handle and URLs without loosing the meaning of the text.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2012.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2012.png)

    [utf-8''NLP_C1_W1_lecture_nb_01.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W1_lecture_nb_01.ipynb)

### Overview until now

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2013.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2013.png)

You have the perform the above mentioned task on a set of 'm' tweets.

So finally you will get something like this which is mentioned below. Tweets → preprocessed tweets → frequency mappings.

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2014.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2014.png)

In mathematical term, you will get a matrix 

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2015.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2015.png)

[utf-8''NLP_C1_W1_lecture_nb_02.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W1_lecture_nb_02.ipynb)

## Logistic Regression Overview

- Sigmoid Function

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2016.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2016.png)

So now when you feed in your tweet, you will get something like this → Positive sentiment.

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2017.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2017.png)

## Training Logistic Regression Model

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2018.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2018.png)

[utf-8''NLP_C1_W1_lecture_nb_03.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W1_lecture_nb_03.ipynb)

## Testing Logistic Regression Model

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2019.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2019.png)

- Accuracy

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2020.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2020.png)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2021.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2021.png)

## Logistic Regression Cost function

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2022.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2022.png)

[utf-8''C1_W1_Assignment.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8C1_W1_Assignment.ipynb)

---

---

# Week 2 : Probability and Naive Bayes

Let's imagine this corpus of positive and negative words. The word 'Happy' can be used in both positive and negative sentiments.

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2023.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2023.png)

So what we see below, is calculating probability for a positive tweet. In this case, if we have 20 tweets, out of that 13 express a positive sentiment, there Probability(Positive) tweet = 65%

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2024.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2024.png)

Let's say, if we have to find probability of the word 'Happy' which is also part of positive tweet corpus is 3 / 20 = 0.15

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2025.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2025.png)

## Bayes' Rule

- Conditional Probablities - Probability of a given event 'A' to occur when 'B' happens P(A|B). In the following example, probability of a tweet to have a positive sentiment if the word 'happy' is part of the tweet

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2026.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2026.png)

    We can also think about the probability that a positive tweet will have the word 'happy' in it.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2027.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2027.png)

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2028.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2028.png)

So now with both this information, we can form the Bayes' rule

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2029.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2029.png)

**the numerator in both the terms are same.** Now, we can substitute and derive the Bayes' rule.

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2030.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2030.png)

Probability of (X given Y) = Probabilty(Y Given X) $X$ Probability(X) $/$ Probability(Y)

## Naive Bayes

- Similar to logistic regression
- We assume that the **features are independent** to each other.
- Get the corpus → extract words → count them into positive and negative

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2031.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2031.png)

- Now, let's get conditional probability of each word in positive and negative class

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2032.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2032.png)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2033.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2033.png)

- One interesting thing to note here is that words like 'I', 'am', 'learning' and 'NLP' have identical probabilities and does not really do much while doing sentiment analysis.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2034.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2034.png)

- Words like ''happy", 'sad' and 'not' are power words since they carry a lot of sentiment. let's look at the word 'because' since it doesn't have any occurrence in the negative tweets, the negative conditional probability is zero. To avoid this, we need to smooth the probability function

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2035.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2035.png)

- **Laplacian Smoothing** - We can use this method to avoid our probabilities to be zero.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2036.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2036.png)

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2037.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2037.png)

The '1' added to the numerator will not let the probability to be zero and the 'V' added in the denominator will force the sum of all frequencies to be 1

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2038.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2038.png)

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2039.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2039.png)

## Log Likelihoods

Below, you can see how a word with ratio between positive and negative values. Words with ratio > 1 = Positive and ratio < 1 the word has negative sentiment. Neutral words have ratio almost around 1

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2040.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2040.png)

Naive Bayes'

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2041.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2041.png)

so basically, you calculate the product of ratio for each word in the tweet and if it is greater than 1, the tweet is positive, else negative. We also have the prior ratio which is important when the dataset is not balanced.

the term after the prior ratio is the likelihood

The prior ratio is 1 in the above example since the number of positive tweet and negative tweet are same(balanced dataset)

- **There might be some underflow i.e., the number becomes so small that it can't be stored on the computer. so in that case, we use log**

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2042.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2042.png)

When we use log, we mitigate underflow. Let's see how that happens for the given dataset

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2043.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2043.png)

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2044.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2044.png)

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2045.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2045.png)

So instead of getting product of the ratios, you can calculate sum of the log of the ratio (lambda) in this case. 

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2046.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2046.png)

 

### Training a Navie Bayes

For this example, we will go with sentiment analysis of positive and negative tweets. Let's do it step by step

- Get the data
- Preprocess
    - Lowercase
    - Remove punctuiation, URL, names
    - Remove stop words
    - Stemming, Lemmatization
    - Tokenize the data
- Perform the word count

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2047.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2047.png)

- Calculate the Probability for each word using Laplacian smoothing

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2048.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2048.png)

- Calculate lamba or log of the probabilities

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2049.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2049.png)

- Calculate the log prior

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2050.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2050.png)

### Testing Naive Bayes

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2051.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2051.png)

### Applications of Naive Bayes

There are many applications of naive Bayes including:

- Author identification
- Spam filtering
- Information retrieval
- Word disambiguation

Assumptions
- Features are independent of each other i.e., all words in a sentence are independent to each other
- Frequency is balanced

[utf-8''C1_W2_Assignment(1).ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8C1_W2_Assignment(1).ipynb)

---

---

# Week 3 : Vector Space Models

Represent words and documents as **vectors**

Need for learning Vector space models

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2052.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2052.png)

Applications

- Capture relationships and dependencies between words in a sentence
- Information Extraction
- Machine Translation
- Chatbots

## Word By Word Design

- Design a co-occurrence matrix where 2 words appear together upto a distance of k.

    K could be upto the length of vocabulary.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2053.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2053.png)

## Word By Document Design

- Number of times a word occurs within a specific category in your corpus.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2054.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2054.png)

- Once you have this tabular notation, you can generate a vector space for the corpus.
- Let's plot the different categories keeping data and film on the X and Y axis

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2055.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2055.png)

- We can see that Economy and ML are closely related than to Entertainment category.
- We can use cosine similarity and Euclidean distance to get the Angle and distance between them.

    [utf-8''NLP_C1_W3_lecture_nb_01.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W3_lecture_nb_01.ipynb)

## Euclidean Distance

- Length of the line segment connecting 2 points in vector space.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2056.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2056.png)

- Euclidean distance for n-dimensional vector (norm of the 2 vectors)

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2057.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2057.png)

    Use np.linalg.norm(a - b)

## Cosine Similarity

- Why is cosine similarity better than Euclidean distance - Suppose you have 3 corpora - Agriculture, food and History. Let's assume that food corpus has less words. Now we will get a representation like this.

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2058.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2058.png)

Cosine similarity is not biased by the size of the corpus

- Norm of a vector = magnitude
- 2 definitions that are important

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2059.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2059.png)

- Cosine Similarity

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2060.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2060.png)

    - When $Beta$ is 90 degree, cos(90) = 0
    - When $Beta$ is 0, cos(0) = 1

## Manipulate Word Vectors

- For a given question to find out capitals from country names, you know Washington DC  is the capital of USA. You can use relationship between countries and their capitals
- You have to find the capital of Russia given the relationship USA → Washington DC

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2061.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2061.png)

- First step is to find the difference (Washington - USA)
- Add the same difference to Russia
- Since (10, 4) doesn't have any capitals,  calculate the euclidean distance or cosine similarity between all the capitals and find the closest one.

    [utf-8''NLP_C1_W3_lecture_nb_02.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W3_lecture_nb_02.ipynb)

## Principle Component Analysis (PCA)

- Dimensionality reduction so that vectors can be plotted on an XY plane.

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2062.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2062.png)

- For this example, let's assume that you have a 2D Vector and you want to reduce it to 1D Vector

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2063.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2063.png)

- Eigenvector : Uncorrelated features for your data
- Eigenvalue : The amount of information retained by each feature
- PCA Algo

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2064.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2064.png)

    1. Mean normalize your data
    2. Get Covariance matrix
    3. Perform Single value decomposition (SVD)
    4. You will get Eigen vectors and Eigen values

    Next, we will project the data to a new set of features.

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2065.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2065.png)

    1. Perform dot product between the word embeddings and first 'n' columns of Eigenvectors . (n is the number of diemension you want at the end ~ 2)
    2. Get percentage of variance retained 

        [utf-8''NLP_C1_W3_lecture_nb_03.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W3_lecture_nb_03.ipynb)

[utf-8''C1_W3_Assignment.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8C1_W3_Assignment.ipynb)

---

---

# Week 4 : Machine Translation

How would you do Machine translation intuitively?

- Generate an English vocab and the equivalent French Vocab
- Calculate word embeddings for each of those
- Now, for example, the word "cat" has an embedding [1, 0, 1] int English, and cat has an embedding [2, 3,5] in French. Now, learn a transformation which transforms the embeddings.
- Therefore, you will have learnt conversion from English word vector space to French word vector space.

### Transforming vectors using Matrices

Basically, you need to find a transformation matrix 'R' which when dot multiplied with English word vector, will give me the French word Vector.

$XR = Y$

Now this learning of R becomes a simple learning task.

- Initialize R with certain values
- in a loop :

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2066.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2066.png)

- Frobenius Norm

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2067.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2067.png)

Numpy
A = np.array
Frobenius = np.sqrt(np.sum(np.square(A)))

- We can even use Frobenius norm squared

    ![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2068.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2068.png)

[utf-8''NLP_C1_W4_lecture_nb_01.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W4_lecture_nb_01.ipynb)

### K-Nearest Neighbours

You might not find exact transformed words from English to French.
So we can use the K-Nearest neighbour technique for the same.

### Locality sensitive Hashing

Hashing method that buckets similar locations together. This will be useful for us for K-Nearest neighbours

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2069.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2069.png)

- How to calculate which side is a vector is on the plane?
- Use dot product

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2070.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2070.png)

- When dot product is positive - vector is on the same side
- When dot product is negative - vector is on the opposite side
- When dot product = 0 - vector is on the plane

### Multiple Hash planes

![Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2071.png](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/Untitled%2071.png)

[utf-8''NLP_C1_W4_lecture_nb_02.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8NLP_C1_W4_lecture_nb_02.ipynb)

[utf-8''C1_W4_Assignment.ipynb](Course1%20Natural%20Language%20Processing%20with%20Classific%20575716323c3245bb85460f71fe20002e/utf-8C1_W4_Assignment.ipynb)