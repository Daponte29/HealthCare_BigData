import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
import utils


RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    
    # Initialize KFold with the specified number of splits
    kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
    
    # Lists to store accuracy and auc for each fold
    accuracies = []
    aucs = []

    # Iterate over the folds
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Train the logistic regression classifier
        logistic_regression_model = LogisticRegression(random_state=RANDOM_STATE)
        logistic_regression_model.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = logistic_regression_model.predict(X_test)

        # Calculate accuracy and auc using models_partc.classification_metrics
        acc = accuracy_score(Y_test, Y_pred)
        auc_ = roc_auc_score(Y_test, Y_pred)

        # Append accuracy and auc to the lists
        accuracies.append(acc)
        aucs.append(auc_)

    # Calculate mean accuracy and mean auc
    accuracy = mean(accuracies)
    auc = mean(aucs)
    return accuracy, auc


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
    #TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    # Initialize ShuffleSplit with the specified number of iterations and test percentage
    rs = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)

    # Lists to store accuracy and auc for each iteration
    accuracies = []
    aucs = []

    # Iterate over the randomized splits
    for train_index, test_index in rs.split(X):
        # Split the data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Train the logistic regression classifier
        logistic_regression_model = LogisticRegression(random_state=RANDOM_STATE)
        logistic_regression_model.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = logistic_regression_model.predict(X_test)

        # Calculate accuracy and auc using models_partc.classification_metrics
        acc = accuracy_score(Y_test, Y_pred)
        auc_ = roc_auc_score(Y_test, Y_pred)

        # Append accuracy and auc to the lists
        accuracies.append(acc)
        aucs.append(auc_)

    # Calculate mean accuracy and mean auc
    accuracy = mean(accuracies)
    auc = mean(aucs)
    return accuracy, auc


def main():
	X,Y = utils.get_data_from_svmlight("../output/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

