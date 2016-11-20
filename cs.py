import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from scipy import stats
from numpy import sqrt

from read_data import *
from normalize import *

X,y = read_data ("../abalone.data")

X = normalize (X,7)
X = normalize (X,3)

net = 	MLPClassifier(verbose=False, activation='logistic', validation_fraction=0.33, hidden_layer_sizes=(5, ), early_stopping=True, learning_rate='constant', learning_rate_init=0.2, max_iter=500, momentum=0.9)

SVM = svm.SVC ()
GNB = GaussianNB()

cv = StratifiedKFold(n_splits=10)

scores_net = cross_val_score(net, X, y, cv=cv, scoring = 'accuracy')
scores_SVM = cross_val_score(SVM, X, y, cv=cv, scoring = 'accuracy')
scores_GNB = cross_val_score(GNB, X, y, cv=cv, scoring = 'accuracy')

ic_net = stats.norm.interval(0.95, loc=1-scores_net.mean(), scale=scores_net.std()/sqrt(len(X)))
ic_SVM = stats.norm.interval(0.95, loc=1-scores_SVM.mean(), scale=scores_SVM.std()/sqrt(len(X)))
ic_GNB = stats.norm.interval(0.95, loc=1-scores_GNB.mean(), scale=scores_GNB.std()/sqrt(len(X)))

print("Error MLP: %0.2f (+/- %0.2f)" % (1-scores_net.mean(), scores_net.mean() - ic_net[0]))
print("Error SVM: %0.2f (+/- %0.2f)" % (1-scores_SVM.mean(), scores_SVM.mean() - ic_SVM[0]))
print("Error GNB: %0.2f (+/- %0.2f)" % (1-scores_GNB.mean(), scores_GNB.mean() - ic_GNB[0]))

print stats.friedmanchisquare(scores_net, scores_SVM, scores_GNB)
