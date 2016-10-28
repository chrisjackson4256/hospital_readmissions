import numpy as np
import pandas as pd
import tensorflow as tf
from tflearn.data_utils import load_csv
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, auc, roc_curve,
                             precision_recall_curve)
import matplotlib.pyplot as plt

# read in the training and test data
path_to_data = '/Users/chris.jackson/models/acrr/data/'
train = pd.read_csv(path_to_data + 'train_readmissions.csv')
test = pd.read_csv(path_to_data + 'test_readmissions.csv')


# split into features and labels
Xtrain = np.array(train.drop('re30_0hr', axis=1)).astype('float32')
ytrain = np.array(train['re30_0hr']).astype('int64')

Xtest = np.array(test.drop('re30_0hr', axis=1)).astype('float32')
ytest = np.array(test['re30_0hr']).astype('int64')

feat_cols = tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain)
# train the deep neural net
dnn = tf.contrib.learn.DNNClassifier(
            feature_columns = feat_cols,
            hidden_units=[10, 20, 10],
            n_classes=2
)
dnn = dnn.fit(x=Xtrain, y=ytrain, steps=5000)

acc_score = dnn.evaluate(x=Xtest, y=ytest)["accuracy"]
print "Accuracy score: ", acc_score

# run on test data
ypred = dnn.predict(Xtest).astype('int64')

# assess the performance
print "accuracy score:", accuracy_score(ytest, ypred)
#print "AUC:\t\t", roc_auc_score(ytest, ypred)
#print "precision:\t", precision_score(ytest, ypred)
#print "recall:\t\t", recall_score(ytest, ypred)
#print "f1 score:\t", f1_score(ytest, ypred)
#print ""


# ROC curves
preds = dnn.predict_proba(Xtest)[:,1]
fpr, tpr, _ = roc_curve(ytest, preds)

# PR curves
prec, recall, _ = precision_recall_curve(ytest, preds)
prec_list = prec.tolist()
recall_list = recall.tolist()

print "AUC (from 'auc'):", auc(fpr, tpr)


# ROC curve:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([0,1], [0,1], c='k', linestyle='--')
ax1.plot(fpr, tpr, c='b', label='DNN: [10, 20, 10]')
plt.legend(loc='lower right')
plt.title("ROC Curve")
ax1.set_xlabel('fpr')
ax1.set_ylabel('tpr')
ax1.set_ylim([0.0, 1.05])
plt.show()

# Precision-recall curve:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(prec_list, recall_list, c='b', label='DNN: [10, 20, 10]')
plt.legend(loc='upper right')
plt.title("PPV-Sensitivity Curve")
ax1.set_xlabel('PPV')
ax1.set_ylabel('sensitivity')
plt.show()
