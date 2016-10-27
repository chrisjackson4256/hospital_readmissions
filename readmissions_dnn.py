import numpy as np
import tflearn
from tflearn.data_utils import load_csv
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score)

# read in train data (indicate that the first column represents labels)
path_to_data = '/Users/chris.jackson/models/acrr/data/'
data, labels = load_csv(path_to_data+'train_readmissions.csv',
                        target_column=0,
                        categorical_labels=True,
                        n_classes=2)
# Read in the test data
test_data, test_labels = load_csv(path_to_data+'test_readmissions.csv',
                        target_column=0,
                        categorical_labels=True,
                        n_classes=2)
test_array = np.array([test_labels[i][1] for i in range(len(test_labels))])

# Build neural network
net = tflearn.input_data(shape=[None, 6])
#net = tflearn.fully_connected(net, 10, activation='sigmoid')
net = tflearn.fully_connected(net, 10, activation='sigmoid')
#net = tflearn.fully_connected(net, 10, activation='sigmoid')
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=50, batch_size=16, show_metric=True)

# Run the model on the test data
pred_list = model.predict(test_data)
pred_array = np.array([np.round(pred[1]) for pred in pred_list])

# Assess performance
print "accuracy score:", accuracy_score(test_array, pred_array)
print "AUC:\t\t", roc_auc_score(test_array, pred_array)
print "precision:\t", precision_score(test_array, pred_array)
print "recall:\t\t", recall_score(test_array, pred_array)
print "f1 score:\t", f1_score(test_array, pred_array)
