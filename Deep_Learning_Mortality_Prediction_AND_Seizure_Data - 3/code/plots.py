import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
from mymodels import MyMLP
from mymodels import MyCNN
from mymodels import MyRNN
from mymodels import MyVariableRNN
from mydatasets import load_seizure_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
#---------------------------------------------------------------------------------------

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    	# TODO: Make plots for loss curves and accuracy curves.
    	# TODO: You do not have to return the plots.
    	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    # Plot Loss Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
     
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
     
    # Show or save the plot
    plt.show()



def plot_confusion_matrix(test_results, class_names):
    """
    Plot the confusion matrix.
    
    Args:
    - test_results (list of tuples): List of tuples containing predicted and true labels.
    - class_names (list of str): List of class names.
    """
    # Extract predicted and true labels from the list of tuples
    predicted_labels, true_labels = zip(*test_results)
    
    # Convert labels to numpy arrays
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    title = 'Normalized Confusion Matrix'
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()















#1.2 b.)----------------------------------
#raw data set INPUTS
input_size = 178
hidden_size = 16
output_size = 5

# Calculate parameters in fc1
params_fc1 = (input_size + 1) * hidden_size

# Calculate parameters in fc2
params_fc2 = (hidden_size + 1) * output_size

# Total trainable parameters
total_params = params_fc1 + params_fc2

print(f"Total Trainable Parameters: {total_params}")










#1.2 c.)------------------
def calculate_metrics(model, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)

            # Append predictions and true labels for confusion matrix
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, normalize='true')
    # Convert to NumPy arrays and return as a dictionary
    result_dict = {
        'average_loss': average_loss,
        'accuracy': accuracy,
        'confusion_mat': (np.array(all_targets), np.array(all_predictions)),
    }

    return result_dict


#GET RAW TENSOR DATA------------------------------------
path_train = "../data/seizure/seizure_train.csv"
path_test = "../data/seizure/seizure_test.csv"
path_validate =  "../data/seizure/seizure_validation.csv"
#get TensorData from Raw Data
training_tensor_data = load_seizure_dataset(path_train, 'MLP')
testing_tensor_data = load_seizure_dataset(path_test, 'MLP')
validating_tensor_data = load_seizure_dataset(path_validate, 'MLP')
print("Training Data Shape:", training_tensor_data.tensors[0].shape)
print("Testing Data Shape:", testing_tensor_data.tensors[0].shape)
print("Validation Data Shape:", validating_tensor_data.tensors[0].shape)

# Initialize your model---------------------------------------
model = MyMLP()  #CHANGE THIS LINE TO EVALUATE DIFFERENT MODELS IN mymodels.py
learning_rate = 0.001
num_epochs = 12 # You can choose any number of epochs based on your training requirements

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader objects (replace placeholders with actual batch_size)
batch_size = 45  # Adjust as needed
train_loader = DataLoader(training_tensor_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(validating_tensor_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testing_tensor_data, batch_size=batch_size, shuffle=False)
#for data, target in train_loader:
#    print(f"Data shape: {data.shape}, Target shape: {target.shape}")
# Lists to store training and validation loss and accuracy values
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

# Inside the training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for data, target in train_loader:
        optimizer.zero_grad()
        # Convert target to torch.long
        target = target.long()
        # Ensure that the model output and target have the correct dimensions
        output = model(data)

        # Print shapes for debugging
        print("Before softmax - Output shape:", output.shape)
        print("Before softmax - Target shape:", target.shape)

        # Apply softmax along the correct dimension (dim=1 for a 2D tensor)
        #output = F.softmax(output, dim=1)

        # Print shapes after applying softmax
        print("After softmax - Output shape:", output.shape)
        print("After softmax - Target shape:", target.shape)

        # Check if the dimensions match
        if output.shape[0] != target.shape[0]:
            raise ValueError(f"Dimensions of input ({output.shape[0]}) do not match target ({target.shape[0]})")

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


    # Record training metrics after each epoch
    train_metrics = calculate_metrics(model, train_loader, criterion)
    train_loss, train_accuracy, _ = train_metrics['average_loss'], train_metrics['accuracy'], train_metrics['confusion_mat']

    # Record validation metrics after each epoch
    valid_metrics = calculate_metrics(model, valid_loader, criterion)
    valid_loss, valid_accuracy, _ = valid_metrics['average_loss'], valid_metrics['accuracy'], valid_metrics['confusion_mat']

    # Append to lists
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)


# Call the plotting function
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

# d.)--------confusion matrix-----------
# Evaluate the model on the test set and plot confusion matrix
model.eval()
test_metrics = calculate_metrics(model, test_loader, criterion)
test_loss, test_accuracy, (test_targets, test_predictions) = (
    test_metrics['average_loss'],
    test_metrics['accuracy'],
    test_metrics['confusion_mat']
)

# Define class names
class_names = [0, 1, 2, 3, 4]

# Call the plotting function for confusion matrix
#plot_confusion_matrix(test_metrics, class_names)










