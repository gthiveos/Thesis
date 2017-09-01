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
tweets = pd.read_csv('tweets.csv', encoding="utf-8")
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



def sentiment_code(y):
    test = re.findall("abuse|accident|addict|aggressi|alarm|allergi|anarchis|angry|annoy|anti|anxious|apologi|aweful|barbar|beg|bitch|blasphe|blind|blood|blur|bomb|bore|bother|boycot|break|broke|brutal|bullsh|bully|bump|burn|cackle|cancer|cannibal|careless|cataclysm|catastroph|cheat|conserv|crash|crazy|crim|crisis|criti|crook|crush|cry|d*mn|damag|damn|danger|dark|dead|debt|decl|decre|defect|defens|delay|demo|depress|despera|destr|diabol|dicta|die|dirt|disa|disc|disg|dish|disl|dism|disp|disr|diss|dist|disv|dizz|doubt|down|drag|drain|drip|egocen|egoti|embar|enem|error|evil|exhaust|expens|explo|extrava|fail|fallen|fanati|fat|fate|fault|fear|forbid|foul|fract|freak|fright|funk|fuzz|gimm|harm|hass|hate|horr|hum|hurt|idiot|idle|ille|immor|immod|immop|impa|impi|impe|impo|inac|inad|inco|inca|inde|indis|inef|inel|inev|inex|infi|infl|inhi|inhu|inho|inju|inop|inor|insi|inso|inst|irre|irri|inva|junk|kill|lie|lone|los|mad|mal|mania|manipulat|mess|milit|mist|mis|mock|mort|murder|naive|negat|nervous|nois|non|ob|offe|oppo|over|overtaxed|pain|panic|passive|pervert|phob|poison|pollu|polution|poor|precari|pretend|prison|problem|propag|protest|provoke|punish|racis|rape|refus|regret|rejec|restrict|rip|risk|rot|ruin|sad|scandal|scar|shame|sick|smash|smu|snob|sorrow|spite|steal|stiff|stole|straggle|strange|strict|struggle|stubborn|stumble|stupid|sub|suffer|suicide|surrender|suspect|suspic|swollen|symptom|terrible|terror|threat|torture|toxic|trag|trap|trash|trauma|trouble|twist|two-fac|ugly|unable|unachievabl|unavailable|uncomfortabl|uncreative|unemploy|unexpect|unfair|unfaith|unfavora|unhapp|unimportant|uninform|unjust|unknown|unlike|unpopular|unprepared|unprove|unreachable|unreliable|unsafe|unskill|unsuccess|untrue|unwork|unworthy|uproar|upset|urgent|venom|vice|victim|viol|virus|vomit|war|warn|waste|weak|weird|whore|wild|woe|worr|worse|wrinkle|wrong",y)
    test1 = re.findall("abound|accessible|acclaim|accolades|accommodative|accomplish|accurate|achiev|acumen|adapt|adequate|adjust|admir|ador|adroit|adulate|advan|agile|agree|all-around|alluring|altruistic|amaze|amenity|ami|amuse|angel|apotheosis|appeal|apprec|appropr|approve|assure|astonish|attract|authent|award|awe|awsome|beaut|beloved|benefit|best|brainy|brand-new|brave|bright|brillian|capab|celebrate|charm|cheer|clarity|clean|clear|clever|comfort|compatible|complement|complim|confiden|congratul|conven|convient|cool|cooperative|correct|courage|cozy|creat|delicious|delight|dominate|dynamic|earn|easy|effect|elite|encourage|endorse|enhance|enjoy|enrich|entertain|enthusias|ex|fabul|faith|fan|fantast|fascin|fast|fav|fine|fortune|free|freedom|friend|ftw|fulfill|fun|gain|gem|genius|glad|glory|good|gorgeous|grace|grate|great|happ|harmless|heal|hearten|help|hero|honest|honor|hope|hospit|humor|ideal|idol|illum|imagin|important|impress|improve|incredible|joy|jubilant|kind|lead|liber|like|logical|love|loyal|luck|lux|magnif|marvel|master|mirac|nice|passion|patient|patriot|patriotic|peace|peps|perfect|please|pleasur|polite|positive|profuse|progress|promise|prompt|pros|prosper|protect|proud|prove|quaint|qualif|quick|rapid|rapt|ready|realistic|reclaim|recomend|refine|reform|refresh|refund|rejoice|reliable|relief|renaissance|renewed|reputable|respect|respectful|responsibly|responsive|restful|restored|restructure|revere|revival|revol|reward|right|risk-free|romantic|rosy|safe|satis|saver|secure|sens||sensitive|sexy|shine|shiny|significant|simplify|sincere|skill|smart|smile|smooth|sociable|soft|solid|soothe|soothingly|sophisticated|soulful|sparkle|spectacular|stable|state-of-the-art|steady|stimul|straightforward|strong|succeed|succes|success|suffice|sufficient|suitable|super|superior|support|supreme|sweet|swift|talent|tempt|tenacious|tender|terrific|thank|thoughtful|thrill|thumb-up|tickle|tingle|together|tolerable|top|treasure|trendy|triumph|trust|truth|unabashed|unabashedly|unaffected|unassailable|unbeatable|unbiased|unbound|uncomplicated|unconditional|undamaged|undaunted|understandable|undisputable|undisputably|undisputed|unencumbered|unequivocal|unequivocally|unfazed|unfettered|unforgettable|unity|unlimited|unmatched|unparalleled|unquestionable|unquestionably|unreal|unrestricted|unrivaled|unselfish|unwavering|up|usable|useable|useful|valiant|valuable|variety|vibrant|vibrantly|victorious|victory|viewable|vigilance|vigilant|virtue|virtuous|virtuously|visionary|vivacious|vivid|vouch|vouchsafe|warm|warmer|warmhearted|warmly|warmth|wealthy|welc|well|whoa|wholeheartedly|wholesome|whooa|whoooa|wieldy|willing|GREAT",y)
    if test:
        return 'Negative'
    elif test1:
        return 'Positive'
    else:
        return 'Neutral'



df['label'] = df['handle'].apply(lambda x: candidate_code(x))
df['sentiment'] = df['text'].apply(lambda y: sentiment_code(y))



messages = df[['label','sentiment','message']]
print(messages[:100])


#bag of words
def split_into_tokens(message):
    message = message  # convert bytes into proper unicode
    return TextBlob(message).words
messages.message.head()

tweets_split = messages.message.head().apply(split_into_tokens)
print(tweets_split[:5])


#Lemmatization
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

tweets_lemma = messages.message.head().apply(split_into_lemmas)
print(tweets_lemma[:20])

#Convert Data to Vector
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
print(bow_transformer.get_feature_names()[:5])

messages_bow = bow_transformer.transform(messages['message'])
print('sparse matrix shape:', messages_bow.shape)
print('number of non-zeros:', messages_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

#Tf-idf
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf)
#print(messages_tfidf.shape)

#-------------------------------------------------------------------------------
#nb classifier for Candidates
cand_detector = MultinomialNB().fit(messages_tfidf, messages['label']) #1
all_predictions = cand_detector.predict(messages_tfidf)
tr_acc = accuracy_score(messages['label'], all_predictions)
print("Accuracy on training set:  %.2f%%" % (100 * tr_acc))


#nb classifier for Candidates polarity
sent_detector = MultinomialNB().fit(messages_tfidf, messages['sentiment'])
all_predictions_1 = sent_detector.predict(messages_tfidf)
tr_acc_1 = accuracy_score(messages['sentiment'], all_predictions_1)
print("Accuracy on training set:  %.2f%%" % (100 * tr_acc_1))
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------
#print ('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
#print ('(row=expected, col=predicted)')
fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(messages['label'], all_predictions), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Training Set Confusion Matrix')
sns.plt.show()


#print ('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
#print ('(row=expected, col=predicted)')
fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(messages['sentiment'], all_predictions_1), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Training Set Confusion Matrix')
sns.plt.show()
#--------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------
print (classification_report(messages['sentiment'], all_predictions_1))

print (classification_report(messages['label'], all_predictions))

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['sentiment'], test_size=0.2)
	
msg_train_1, msg_test_1, label_train_1, label_test_1 = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

print (len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

#--------------------------------------------------------------------------------


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


#----------------------------------------------------------------------------------
scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=1,)  # 1 = use one core                         

scores_1 = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train_1,  # training data
                         label_train_1,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=1,)  # 1 = use one core  						 
						 
print(scores)
print(scores_1)

print(scores.mean(), scores.std())
print(scores_1.mean(), scores_1.std())
#------------------------------------------------------------------------------------------------------


params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

#-----------------------------------------------------------------------------------------
grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=1,  # number of cores to use for parallelization;
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

grid_1 = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=1,  # number of cores to use for parallelization;
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train_1, n_folds=5),  # what type of cross validation to use
)


nb_detector = grid.fit(msg_train, label_train)
print (nb_detector.grid_scores_)

nb_detector_1 = grid_1.fit(msg_train_1, label_train_1)
print (nb_detector_1.grid_scores_)

#Popular tweets for Trump
my_1st_tweet = "With this election we're simultaneously breaking through the glass ceiling and the rock bottom. We got a really big room now"
my_2nd_tweet = "Retweet if you are: -A woman -An immigrant -LGBT+ -Muslim -African American -Latino/Latina-In anywway completely terrified right now"
my_3rd_tweet = "Such a beautiful and important evening! The forgotten man and woman will never be forgotten again. We will all come together as never before"
my_4th_tweet = "How long did it take your staff of 823 people to think that up--and where are your 33,000 emails that you deleted?"
my_5th_tweet = "Women have the power to stop."
my_6th_tweet = "America needs a leader who treats women with respect"
my_7th_tweet = "No one remembers who came in second"
my_8th_tweet = "Sorry losers and haters, but my I.Q. is one of the highest -and you all know it! Please don't feel so stupid or insecure,it's not your fault"
my_9th_tweet = "OH MY GOD VERIZON ELECTION NIGHT IS THE WORST TIME FOR THIS ADVERTISEMENT."
my_10th_tweet = "I hear you, Sanders supporters who plan to vote Trump. One time I asked for Coke but they only had Pepsi, so I set fire to my head."


print("Tweet #1:", "'",my_1st_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_1st_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_1st_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_1st_tweet])[0])), nb_detector.predict([my_1st_tweet])[0])
print("Tweet #2:", "'",my_2nd_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_2nd_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_2nd_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_2nd_tweet])[0])), nb_detector.predict([my_2nd_tweet])[0])
print("Tweet #3:", "'",my_3rd_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_3rd_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_3rd_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_3rd_tweet])[0])), nb_detector.predict([my_3rd_tweet])[0])
print("Tweet #4:", "'",my_4th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_4th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_4th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_4th_tweet])[0])), nb_detector.predict([my_4th_tweet])[0])
print("Tweet #5:", "'",my_5th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_5th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_5th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_5th_tweet])[0])), nb_detector.predict([my_5th_tweet])[0])
print("Tweet #6:", "'",my_6th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_6th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_6th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_6th_tweet])[0])), nb_detector.predict([my_6th_tweet])[0])
print("Tweet #7:", "'",my_7th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_7th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_7th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_7th_tweet])[0])), nb_detector.predict([my_7th_tweet])[0])
print("Tweet #8:", "'",my_8th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_8th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_8th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_8th_tweet])[0])), nb_detector.predict([my_8th_tweet])[0])
print("Tweet #9:", "'",my_9th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_9th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_9th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_9th_tweet])[0])), nb_detector.predict([my_9th_tweet])[0])
print("Tweet #10:", "'",my_10th_tweet, "'", ' \n \n', "I'm about %.0f%%" % (100 * max(nb_detector_1.predict_proba([my_10th_tweet])[0])), "sure this was tweeted by", nb_detector_1.predict([my_10th_tweet])[0], "and the polarity is about %.0f%%" % (100 * max(nb_detector.predict_proba([my_10th_tweet])[0])), nb_detector.predict([my_10th_tweet])[0])


predictions = nb_detector.predict(msg_test)
print (confusion_matrix(label_test, predictions))
print (classification_report(label_test, predictions))

predictions_1 = nb_detector_1.predict(msg_test_1)
print (confusion_matrix(label_test_1, predictions_1))
print (classification_report(label_test_1, predictions_1))

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Test Set Confusion Matrix')
sns.plt.show()

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test_1, predictions_1), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Test Set Confusion Matrix')
sns.plt.show()

print("Accuracy on test set:  %.2f%%" % (100 * (nb_detector.grid_scores_[0][1])))
print("Accuracy on test set:  %.2f%%" % (100 * (nb_detector_1.grid_scores_[0][1])))
#-------------------------------------------------------------------------------------------------

top_h = {}
top_t = {}

for w in (bow_transformer.get_feature_names()[:len(bow_transformer.get_feature_names())]):
    p = nb_detector_1.predict_proba([w])[0][0]
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
plt.title('Top 10 terms with highest probability of indicating a Negative tweet')
plt.xlabel('Probability')
sns.plt.show()

sns.set_context({"figure.figsize": (11, 7)})
top_df_h.plot(kind='barh', stacked=True, color=["#08306B","#E91D0E"]).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.yticks(range(len(top_df_h['word'])), list(top_df_h['word']))
plt.title('Top 10 terms with highest probability of indicating a Positive tweet')
plt.xlabel('Probability')
sns.plt.show()





