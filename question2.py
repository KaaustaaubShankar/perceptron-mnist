# Import necessary modules
from perceptron import Perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Function to load the dataset with binary labels for each target digit
def loadsets(digit):
    data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt').reshape((5000, 28, 28))

    # Define ranges for each digit
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
    
    # Initialize lists to hold training and test data and labels
    train_data, train_labels = [], []
    test_data, test_labels = [], []

    # Loop through each digit, assign labels based on the target digit
    for num in range(10):
        start_train, end_train, start_test, end_test = digit_ranges[num]
        train_data.append(data[start_train:end_train])
        test_data.append(data[start_test:end_test])

        # Label target digit as 1, all others as 0
        label = 1 if num == digit else 0
        train_labels.append(np.full(end_train - start_train, label))
        test_labels.append(np.full(end_test - start_test, label))

    # Concatenate all training and test data and labels
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)

    # Convert to DataFrames and shuffle training set
    train_df = pd.DataFrame({'data': list(train_data), 'label': train_labels})
    test_df = pd.DataFrame({'data': list(test_data), 'label': test_labels})
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    return train_df, test_df

# Function to train a perceptron with a balanced dataset by upsampling label 1 since its smaller
def train_perceptron(perceptron, train_df, learning_rate=0.01, max_epochs=1000, target_error=0.05):
    errors_per_epoch, epochs_completed = [], 0
    for epoch in range(max_epochs):
        # Balance classes by upsampling minority class (label 1)
        class_1 = train_df[train_df['label'] == 1]
        class_0 = train_df[train_df['label'] == 0]
        
        class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=epoch)
        balanced_train_df = pd.concat([class_0, class_1_upsampled]).sample(frac=1).reset_index(drop=True)
        
        # Train perceptron for one epoch
        errors, epoch_completed = perceptron.train(balanced_train_df, epochs=1, learning_rate=learning_rate, target_error=target_error)
        errors_per_epoch.extend(errors)
        epochs_completed += epoch_completed
        
        # Early stopping if the error rate meets the target error
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

# Evaluate initial metrics for digit 9
initial_metrics_9 = perceptron_9.evaluate_metrics(test_df_9)

# Train perceptron for digit 9
print("\nTraining perceptron for digit 9")
errors_9, epochs_9 = train_perceptron(perceptron_9, train_df_9)

# Evaluate after-training metrics for digit 9
after_training_metrics_9 = perceptron_9.evaluate_metrics(test_df_9)

# Write metrics for digit 9 to the file
with open('problem2_metrics.txt', 'w') as f:
    f.write("Initial Metrics for Digit 9:\n")
    f.write(f"Error Fraction: {initial_metrics_9['error_fraction']}\n")
    f.write(f"Precision: {initial_metrics_9['precision']}\n")
    f.write(f"Recall: {initial_metrics_9['recall']}\n")
    f.write(f"F1 Score: {initial_metrics_9['f1_score']}\n\n")
    
    f.write("After Training Metrics for Digit 9:\n")
    f.write(f"Error Fraction: {after_training_metrics_9['error_fraction']}\n")
    f.write(f"Precision: {after_training_metrics_9['precision']}\n")
    f.write(f"Recall: {after_training_metrics_9['recall']}\n")
    f.write(f"F1 Score: {after_training_metrics_9['f1_score']}\n\n")



# Train and plot perceptrons for digits 0–8
plt.figure(figsize=(10, 6))  # Set figure size for better visualization

with open('problem2_metrics.txt', 'a') as f:  # Append to the file for digits 0–8
    for digit in range(9):
        train_df, test_df = datasets[digit]
        print(f"\nTraining perceptron for digit {digit}")
        
        # Evaluate initial metrics on the test set for each digit
        initial_metrics = perceptrons[digit].evaluate_metrics(test_df)
        
        # Write initial metrics for the digit to the file
        f.write(f"Initial Metrics for Digit {digit}:\n")
        f.write(f"Error Fraction: {initial_metrics['error_fraction']}\n")
        f.write(f"Precision: {initial_metrics['precision']}\n")
        f.write(f"Recall: {initial_metrics['recall']}\n")
        f.write(f"F1 Score: {initial_metrics['f1_score']}\n\n")
        
        # Train the perceptron for the current digit and retrieve errors per epoch
        errors, epochs = train_perceptron(perceptrons[digit], train_df, max_epochs=epochs_9)
        
        # Evaluate after-training metrics on the test set
        after_training_metrics = perceptrons[digit].evaluate_metrics(test_df)
        
        # Write after-training metrics to the file
        f.write(f"After Training Metrics for Digit {digit}:\n")
        f.write(f"Error Fraction: {after_training_metrics['error_fraction']}\n")
        f.write(f"Precision: {after_training_metrics['precision']}\n")
        f.write(f"Recall: {after_training_metrics['recall']}\n")
        f.write(f"F1 Score: {after_training_metrics['f1_score']}\n\n")
        
        # Plot errors for each digit
        plt.plot(errors, label=f'Digit {digit}')

# Finalize the plot with the target error line and labels
plt.plot(errors_9, label='Digit 9')  # Plot errors for digit 9
plt.axhline(y=0.05, color='red', linestyle='--', label='Target Error (0.05)')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.title('Training Error per Epoch for Perceptrons Targeting Each Digit (0-9)', fontsize=14)
plt.legend(title="Digits", title_fontsize='13', fontsize='11')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability

plt.show()
