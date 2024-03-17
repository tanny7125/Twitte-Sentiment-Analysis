# Twitter-Sentiment-Analysis
 This repository contains code for sentiment analysis on Twitter data using machine learning techniques. It includes data preprocessing, model training, and evaluation scripts, along with a Jupyter notebook demonstrating the analysis pipeline. The project aims to classify tweets into positive, negative, or neutral sentiments, providing insights into public opinion trends on Twitter

 ## Cloning the repo
You can start by cloning this repo in your wordspace and then start playing with the function to make your project done.
```
git clone https://github.com/vai-20-dehi/Twitter-Sentinment-Analysis.git
```

## Packages that need to be installed:

Jupyter notebook or any other software to

## What This project does?
This project helps us to classify the no of positive and negative sentiments from the tweets present in the dataset

# Code for printing the negative and positive words
```
train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white')
print("Negative words")
wordcloud_draw(train_neg)

```

# Code to extract Features
```
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
```
# Code for final output

```
neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt)) 


```

# Output Images of the project
## Positive Words
![Output image](https://github.com/tanny7125/Twitter-Sentiment-Analysis/blob/main/Output%20Images/Screenshot%20(810).png?raw=true)

## Negative Words
![Output image](https://github.com/tanny7125/Twitter-Sentiment-Analysis/blob/main/Output%20Images/Screenshot%20(811).png?raw=true)

## Output
![Output image](https://github.com/tanny7125/Twitter-Sentiment-Analysis/blob/main/Output%20Images/Screenshot%20(812).png?raw=true)
