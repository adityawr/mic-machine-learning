
from IPython import get_ipython


import numpy as np
import pandas as pd
from time import time
from IPython.display import display 
import visuals as vs


get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv("census.csv")
display(data.head(n=1))

n_records = len(data)   

n_greater_50k = len(data[data['income'] == '>50K'])

n_at_most_50k = len(data[data['income'] == '<=50K']) 

greater_percent = (n_greater_50k / float(n_records))*100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

vs.distribution(data)

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

vs.distribution(features_raw, transformed = True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

display(features_raw.head(n = 1))

features = pd.get_dummies(features_raw)

income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)

encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

print(encoded)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

accuracy = (n_greater_50k/float(n_records)) 

true_positives = n_greater_50k
true_negatives = 0
false_positives = n_at_most_50k
false_negatives = 0
precision = true_positives/float(true_positives+false_positives)
recall = true_positives/float(true_positives + false_negatives)

fscore = (1 + 0.5**2) * ((precision * recall)/float(0.5**2 * (precision * recall) + 1))

print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):  
    results = {}
    
    start = time() 
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() 
    
    results['train_time'] = end-start
        
    start = time() 
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() 
    
    results['pred_time'] = end-start
            
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

clf_A = RandomForestClassifier(random_state=543)
clf_B = GaussianNB()
clf_C = AdaBoostClassifier(random_state=543)

samples_1 = len(X_train)/100
samples_10 = len(X_train)/10
samples_100 = len(X_train)

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

vs.evaluate(results, accuracy, fscore)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

clf = AdaBoostClassifier(random_state=0)

parameters = {'n_estimators':[75,200,500],'learning_rate':[1.0,1.5,2.0]}

scorer = make_scorer(fbeta_score, beta=0.5)

grid_obj = GridSearchCV(clf, parameters,scoring=scorer)

grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

model = AdaBoostClassifier(random_state=0,n_estimators=500).fit(X_train, y_train)

importances =  model.feature_importances_

vs.feature_plot(importances, X_train, y_train)

from sklearn.base import clone

X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

clf = (clone(best_clf)).fit(X_train_reduced, y_train)

reduced_predictions = clf.predict(X_test_reduced)

print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))