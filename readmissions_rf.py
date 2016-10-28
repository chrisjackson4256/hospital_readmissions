import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, auc, roc_curve,
                             precision_recall_curve)
import matplotlib.pyplot as plt

# read in the training and test data
path_to_data = '/Users/chris.jackson/models/acrr/data/'
train = pd.read_csv(path_to_data + 'train_readmissions.csv')
test = pd.read_csv(path_to_data + 'test_readmissions.csv')

# split into features and labels
Xtrain = np.array(train.drop('re30_0hr', axis=1))
ytrain = np.array(train['re30_0hr'])

Xtest = np.array(test.drop('re30_0hr', axis=1))
ytest = np.array(test['re30_0hr'])

# train the random forest
crf = RandomForestClassifier(n_estimators=1000)
crf = crf.fit(Xtrain, ytrain)

# run on test data
ypred = crf.predict(Xtest)
print ypred
print ypred.size
print ytest
print ytest.size

# assess the performance
print "accuracy score:", accuracy_score(ytest, ypred)
print "AUC:\t\t", roc_auc_score(ytest, ypred)
print "precision:\t", precision_score(ytest, ypred)
print "recall:\t\t", recall_score(ytest, ypred)
print "f1 score:\t", f1_score(ytest, ypred)
print ""

'''
# random forest w/ larger min_samples_split
crf2 = RandomForestClassifier(
            n_estimators=1000,
            min_samples_split=20
)

crf2 = crf2.fit(Xtrain, ytrain)

# run on test data
ypred2 = crf2.predict(Xtest)

# assess the performance
print "accuracy score:", accuracy_score(ytest, ypred2)
print "AUC:\t\t", roc_auc_score(ytest, ypred2)
print "precision:\t", precision_score(ytest, ypred2)
print "recall:\t\t", recall_score(ytest, ypred2)
print "f1 score:\t", f1_score(ytest, ypred2)
print ""

# ROC curves
preds = crf.predict_proba(Xtest)[:,1]
fpr, tpr, _ = roc_curve(ytest, preds)

preds2 = crf2.predict_proba(Xtest)[:,1]
fpr2, tpr2, _ = roc_curve(ytest, preds2)


# PR curves
prec, recall, _ = precision_recall_curve(ytest, preds)
prec_list = prec.tolist()
recall_list = recall.tolist()

prec2, recall2, _ = precision_recall_curve(ytest, preds2)
prec_list2 = prec2.tolist()
recall_list2 = recall2.tolist()

print "AUC (from 'auc'):", auc(fpr, tpr)
print "AUC (from 'auc'):", auc(fpr2, tpr2)

# ROC curve:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([0,1], [0,1], c='k', linestyle='--')
ax1.plot(fpr, tpr, c='b', label='RF: 1000 trees')
ax1.plot(fpr2, tpr2, c='r', label='RF: 1000 trees, min_split=20')
plt.legend(loc='lower right')
plt.title("ROC Curve")
ax1.set_xlabel('fpr')
ax1.set_ylabel('tpr')
ax1.set_ylim([0.0, 1.05])
plt.show()

# Precision-recall curve:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(prec_list, recall_list, c='b', label='RF: 1000 trees')
ax1.plot(prec_list2, recall_list2, c='r', label='RF: 1000 trees, min_split=20')
plt.legend(loc='upper right')
plt.title("PPV-Sensitivity Curve")
ax1.set_xlabel('PPV')
ax1.set_ylabel('sensitivity')
plt.show()
'''
