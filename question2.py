# Import necessary modules
from perceptron import Perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample


def loadsets(digit):
    data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt').reshape((5000, 28, 28))

    # ranges for each digit in teh txt
    digit_ranges = {
        0: (0, 400, 400, 500),
        1: (500, 900, 900, 1000),
        2: (1000, 1400, 1400, 1500),
        3: (1500, 1900, 1900, 2000),
        4: (2000, 2400, 2400, 2500),
        5: (2500, 2900, 2900, 3000),
        6: (3000, 3400, 3400, 3500),
        7: (3500, 3900, 3900, 4000),
        8: (4000, 4400, 4400, 4500),
        9: (4500, 4900, 4900, 5000)
    }
    
    # setting up some lists
    train_data, train_labels = [], []
    test_data, test_labels = [], []

    # split the data up
    for num in range(10):
        start_train, end_train, start_test, end_test = digit_ranges[num]
        train_data.append(data[start_train:end_train])
        test_data.append(data[start_test:end_test])

        # Label target digit as 1, all others as 0
        label = 1 if num == digit else 0
        train_labels.append(np.full(end_train - start_train, label))
        test_labels.append(np.full(end_test - start_test, label))

    # combination
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)
    print(len(train_data), len(train_labels), len(test_data), len(test_labels))

    '''
    plt.imshow(train_data[1234], cmap='gray')
    plt.title(f"Digit: {digit }Label: {train_labels[1234]}")
    plt.colorbar()
    plt.show()
    '''

    # make dataframes for organization
    train_df = pd.DataFrame({'data': list(train_data), 'label': train_labels})
    test_df = pd.DataFrame({'data': list(test_data), 'label': test_labels})
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    return train_df, test_df

#this will train the perceptron for a digit according to how problem 2 wants it
def train_perceptron(perceptron, train_df, learning_rate=0.001, max_epochs=1000, target_error=0.05):
    errors_per_epoch, epochs_completed = [], 0
    for epoch in range(max_epochs):

        #make classes
        class_1 = train_df[train_df['label'] == 1]
        class_0 = train_df[train_df['label'] == 0]
        
        class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=epoch)
        balanced_train_df = pd.concat([class_0, class_1_upsampled]).sample(frac=1).reset_index(drop=True)
        
        # Train perceptron for one epoch
        errors, epoch_completed = perceptron.train(balanced_train_df, epochs=1, learning_rate=learning_rate, target_error=target_error)
        errors_per_epoch.extend(errors)
        epochs_completed += epoch_completed
        
        # stop ahead
        if errors[-1] <= target_error:
            print(f"Training for digit completed with below target error of {target_error} at epoch {epochs_completed}")
            return errors_per_epoch, epochs_completed
    
    print(f"Reached max epochs for digit with final error {errors_per_epoch[-1]:.4f}")
    return errors_per_epoch, max_epochs

# Train perceptron for digit 9 to determine the number of epochs needed
train_df_9, test_df_9 = loadsets(9)
perceptron_9 = Perceptron()
perceptrons = [Perceptron() for _ in range(9)] + [perceptron_9]
datasets = [loadsets(digit) for digit in range(9)]

initial_metrics = [0]*10
after_training_metrics = [0]*10
initial_metrics[9] = perceptron_9.evaluate_metrics(test_df_9)

print(initial_metrics[9]["error_fraction"])
errors_9, epochs_9 = train_perceptron(perceptron_9, train_df_9)
after_training_metrics[9]= perceptron_9.evaluate_metrics(test_df_9)
print(after_training_metrics[9]["error_fraction"])

plt.figure(figsize=(10, 6))  

for digit in range(9):
    train_df, test_df = datasets[digit]
    initial_metrics[digit] = perceptrons[digit].evaluate_metrics(test_df)
    errors, epochs = train_perceptron(perceptrons[digit], train_df, max_epochs=epochs_9)
    after_training_metrics[digit] = perceptrons[digit].evaluate_metrics(test_df)
    plt.plot(errors, label=f'Digit {digit}')


plt.plot(errors_9, label='Digit 9')  # Plot errors for digit 9
plt.axhline(y=0.05, color='red', linestyle='--', label='Target Error (0.05)')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.title('Training Error per Epoch for Perceptrons Targeting Each Digit (0-9)', fontsize=14)
plt.legend(title="Digits", title_fontsize='13', fontsize='11')
plt.grid(True, linestyle='--', alpha=0.7)  

plt.show()

metrics_labels = ['Error Fraction', 'Precision', 'Recall', 'F1 Score']
digits = range(10)
x = np.arange(len(digits))  
width = 0.35  

#setting up the pairs
error_before = [initial_metrics[digit]['error_fraction'] for digit in digits]
error_after = [after_training_metrics[digit]['error_fraction'] for digit in digits]
precision_before = [initial_metrics[digit]['precision'] for digit in digits]
precision_after = [after_training_metrics[digit]['precision'] for digit in digits]
recall_before = [initial_metrics[digit]['recall'] for digit in digits]
recall_after = [after_training_metrics[digit]['recall'] for digit in digits]
f1_before = [initial_metrics[digit]['f1_score'] for digit in digits]
f1_after = [after_training_metrics[digit]['f1_score'] for digit in digits]

#error fraction
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, error_before, width, label='Before Training')
ax.bar(x + width/2, error_after, width, label='After Training')
ax.set_xlabel('Digits')
ax.set_ylabel('Error Fraction')
ax.set_title('Error Fraction Before and After Training by Digit')
ax.set_xticks(x)
ax.set_xticklabels(digits)
ax.legend()
plt.show()

#precision
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, precision_before, width, label='Before Training')
ax.bar(x + width/2, precision_after, width, label='After Training')
ax.set_xlabel('Digits')
ax.set_ylabel('Precision')
ax.set_title('Precision Before and After Training by Digit')
ax.set_xticks(x)
ax.set_xticklabels(digits)
ax.legend()
plt.show()

#recall
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, recall_before, width, label='Before Training')
ax.bar(x + width/2, recall_after, width, label='After Training')
ax.set_xlabel('Digits')
ax.set_ylabel('Recall')
ax.set_title('Recall Before and After Training by Digit')
ax.set_xticks(x)
ax.set_xticklabels(digits)
ax.legend()
plt.show()

# f1score
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, f1_before, width, label='Before Training')
ax.bar(x + width/2, f1_after, width, label='After Training')
ax.set_xlabel('Digits')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Before and After Training by Digit')
ax.set_xticks(x)
ax.set_xticklabels(digits)
ax.legend()
plt.show()
