import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from pandas import Series
import calendar
import datetime
import re
import sklearn
import seaborn as sns


from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 

#Load Data
tweets = pd.read_csv('US_elections.csv', encoding="utf-8")
tweets = tweets[['handle','text','is_retweet']]

#Count Tweets
print (len(tweets))

#Useful Stats
tweets_stat = tweets.groupby('handle').describe()
print(tweets_stat)


df = tweets.loc[tweets['is_retweet'] == False]
df = df.copy().reset_index(drop=True)

df['length_no_url'] = df['text']
df['length_no_url'] = df['length_no_url'].apply(lambda x: len(x.lower().split('http')[0]))
df['message'] = df['text'].apply(lambda x: x.lower().split('http')[0])

def candidate_code(x):
    if x == 'HillaryClinton':
        return 'Hillary'
    elif x == 'realDonaldTrump':
        return 'Trump'
    else:
        return ''

df['label'] = df['handle'].apply(lambda x: candidate_code(x))

messages = df[['label','message']]
#messages_5 = messages[:5]
#print(messages[:5])


#bag of words
def split_into_tokens(message):
    message = message  # convert bytes into proper unicode
    return TextBlob(message).words
messages.message.head()

tweets_split = messages.message.head().apply(split_into_tokens)
#print(tweets_split[:5])


#Tweets_POS = TextBlob("Hello world, how is it going?").tags  # list of (word, POS) pairs
#print(Tweets_POS)

#Lemmatization
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

tweets_lemma = messages.message.head().apply(split_into_lemmas)
#print(tweets_lemma)

#Convert Data to Vector
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
#print(bow_transformer.get_feature_names()[:5])

messages_bow = bow_transformer.transform(messages['message'])
print('sparse matrix shape:', messages_bow.shape)
print('number of non-zeros:', messages_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

#Tf-idf
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
#print(messages_tfidf)
#print(messages_tfidf.shape)

cand_detector = LinearSVC().fit(messages_tfidf, messages['label'])

all_predictions = cand_detector.predict(messages_tfidf)

tr_acc = accuracy_score(messages['label'], all_predictions)
print("Accuracy on training set:  %.2f%%" % (100 * tr_acc))

#print ('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
#print ('(row=expected, col=predicted)')
fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(messages['label'], all_predictions), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Training Set Confusion Matrix')
sns.plt.show()


print (classification_report(messages['label'], all_predictions))

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

print (len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC(probability=True)),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)


svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print (svm_detector.grid_scores_)


#Popular tweets for Trump
print (svm_detector.predict(["How long did it take your staff of 823 people to think that up--and where are your 33,000 emails that you deleted? "])[0])
print (svm_detector.predict(["TODAY WE MAKE AMERICA GREAT AGAIN!"])[0])
print (svm_detector.predict(["The media is spending more time doing a forensic analysis of Melania's speech than the FBI spent on Hillary's emails."])[0])

#Popular Tweets for Hillary
print (svm_detector.predict(["Donald Trump called her Miss Piggy and Miss Housekeeping. Her name is Alicia Machado."])[0])
print (svm_detector.predict(["Women have the power to stop."])[0])
print (svm_detector.predict(["Delete your account."])[0])



predictions = svm_detector.predict(msg_test)
print (confusion_matrix(label_test, svm_detector.predict(msg_test)))
print (classification_report(label_test, svm_detector.predict(msg_test)))

print("Accuracy on test set:  %.2f%%" % (100 * (svm_detector.grid_scores_[0][1])))

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Test Set Confusion Matrix')
sns.plt.show()

#Popular tweets for Trump
print (svm_detector.predict_proba(["How long did it take your staff of 823 people to think that up--and where are your 33,000 emails that you deleted? "])[0])
print (svm_detector.predict_proba(["TODAY WE MAKE AMERICA GREAT AGAIN!"])[0])
print (svm_detector.predict_proba(["The media is spending more time doing a forensic analysis of Melania's speech than the FBI spent on Hillary's emails."])[0])

#Popular Tweets for Hillary
print (svm_detector.predict_proba(["Donald Trump called her Miss Piggy and Miss Housekeeping. Her name is Alicia Machado."])[0])
print (svm_detector.predict_proba(["Women have the power to stop."])[0])
print (svm_detector.predict_proba(["Delete your account."])[0])


top_h = {}
top_t = {}

for w in (bow_transformer.get_feature_names()[:len(bow_transformer.get_feature_names())]):
    p = svm_detector.predict_proba([w])[0][0]
    if len(w) > 3:
        if p > 0.5:
            top_h[w] = p
        elif p < 0.5:
            top_t[w] = p
    else:
        pass
    
top_t_10 = sorted(top_t, key=top_t.get, reverse=False)[:10]
top_h_10 = sorted(top_h, key=top_h.get, reverse=True)[:10]

dic = {}
for l in [top_t_10, top_h_10]:
    for key, values in (top_t.items() | top_h.items()):
        if key in l:
            dic[key] = values
            
top_df = pd.DataFrame(list(dic.items()), columns=['word', 'hillary_prob'])
top_df['trump_prob'] = (1 - top_df['hillary_prob'])
top_df_t = top_df[:int((len(dic)/2))]
top_df_t = top_df_t[['word','trump_prob','hillary_prob']].sort_values(['trump_prob'], ascending=[True])
top_df_h = top_df[int((len(dic)/2)):].sort_values(['hillary_prob'], ascending=[True])

sns.set_context({"figure.figsize": (10, 7)})
top_df_t.plot(kind='barh', stacked=True, color=["#E91D0E","#08306B"]).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.yticks(range(len(top_df_t['word'])), list(top_df_t['word']))
plt.title('Top 10 terms with highest probability of indicating a Trump tweet')
plt.xlabel('Probability')
sns.plt.show()

sns.set_context({"figure.figsize": (11, 7)})
top_df_h.plot(kind='barh', stacked=True, color=["#08306B","#E91D0E"]).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.yticks(range(len(top_df_h['word'])), list(top_df_h['word']))
plt.title('Top 10 terms with highest probability of indicating a Hillary tweet')
plt.xlabel('Probability')
sns.plt.show()







