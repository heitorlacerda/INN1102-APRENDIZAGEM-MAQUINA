from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sknn.mlp import Classifier as MLPClassifier
from sknn.mlp import Layer
from datasets.read_from_file import DataMgr
from scipy.stats import friedmanchisquare
from scipy.stats.stats import rankdata
from nemenyi import critical_difference
import numpy as np

# Parameters
alpha = 0.01    # Nemenye alpha
n_folds = 40    # Number of folds for Cross Validation

# Classifiers
layers=[Layer("Maxout", units=100, pieces=2),Layer("Sigmoid")]
clf1 = MLPClassifier(layers=layers,learning_rate=0.001,n_iter=25)
clf2 = OneVsRestClassifier(LinearSVC(random_state=0))
clf3 = OneVsRestClassifier(GaussianNB())
clfs = [clf1, clf2, clf3]

# Import Data
dataMgr = DataMgr()
y, X = dataMgr.rawData()

# Create folds for Stratified cross validation
skf = StratifiedKFold(y, n_folds=n_folds, shuffle=False, random_state=None)

# Encode label on binary form for multilabel classification
le = LabelEncoder()
le.fit(y)

# Create list of lists of scores for each classfier
scores = [[] for _ in clfs]

# Create list for score of the combination of individual classifiers
globalScores = []

# Fold counter just for printing
fold = 1

# For each fold run the classification on all data views using each classfier
# Then combine the results on each data view using majority voting
# Finally, after all classifiers have ran, combine their results again using majority voting
for train_index, test_index in skf:
    globalPred = []
    print("Fold = " + str(fold))
    fold = fold + 1
    # Obtain Labels of training and testing
    y_train, y_test = y[train_index], y[test_index]
    
    # Run each classifier
    for clf, score in zip(clfs, scores):
        predictions = []
        
        # For each data view
        for p in range(len(X)):
            
            # Obtain Features for training and for testing
            X_train, X_test = X[p][train_index], X[p][test_index]
            
            # Train Classifier
            clf.fit(X_train, y_train)
            
            # Add classifier prediction on list
            predictions.append(clf.predict(X_test))
        
        # Combine prediction on all data views using majority voting
        predictions = np.asarray(predictions).T
        maj = np.apply_along_axis(lambda x:
                                  np.argmax(np.bincount(x)),
                                  axis=1,
                                  arr=predictions)
        maj = le.inverse_transform(maj)
        
        # Add final classifier prediction on global list
        globalPred.append(maj)
        
        # Add classification score on list
        score.append(accuracy_score(y_test, maj))
    
    # Combine prediction on all classifiers using majority voting
    globalPred = np.asarray(globalPred).T
    globalMaj = np.apply_along_axis(lambda x:
                                  np.argmax(np.bincount(x)),
                                  axis=1,
                                  arr=globalPred)
    globalMaj = le.inverse_transform(globalMaj)
    
    # Add combined classification score on list
    globalScores.append(accuracy_score(y_test, globalMaj))

# After all classifications. Add global scores on score list as a new classifier
scores.append(globalScores)

# Convert scores to np.array for Friedman Test Calculation
scores = np.array(scores)

# Calculate errors from test scores.
errors = np.array([np.subtract(1, score) for score in scores])

# Print accuracy mean and deviation based on all scores evaluation
for score, label in zip(scores, ['MLP', 'Linear SVM', 'naive Bayes', 'Ensemble']):
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (np.mean(score), np.std(score), label))
  
# Print Scores of all classifiers on each fold
for scoreLn in scores.T:
    print(scoreLn)
    
argument = [scores[i, :] for i in np.arange((scores).shape[0])]

# Rank data
data1 = np.vstack(argument).T
data1 = data1.astype(float)
for i in range(len(data1)):
    data1[i] = rankdata(data1[i])
print(data1)

# Calculate Friedman Test mean Rank and pValue
chi, pValue = friedmanchisquare(*argument)

print("Friedman Test: Chi = " + str(chi) + ", pValue = " + str(pValue))

# If pre test is valid, use Nemenyi port-test for final decition
if(pValue <= alpha):
    print("Para alpha = " + str(alpha) + " Hipotese nula nao foi confirmada")
    cd = critical_difference(pvalue=alpha, models=len(scores), datasets=n_folds)
    print("Nemenyi: CD = " + str(cd))
    if(chi > cd):
        print("Pelo Nemenyi Test, hipotese nula foi descartada.")
