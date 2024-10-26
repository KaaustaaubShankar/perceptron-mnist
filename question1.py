import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

np.random.seed(42)
def loadMNIST():

    """
    takes no arguments, returns a tuple of three dataframes: train_df, test_df, challenge_df
    """
    #loading the data and reshaping it
    data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt')
    data = data.reshape((5000, 28, 28))

    #making train and test set
    zeroes_train, zeroes_test = data[:400], data[400:500]
    ones_train, ones_test = data[500:900], data[900:1000]

    #we make the labels
    labels_zeroes_train = np.zeros(len(zeroes_train))
    labels_zeroes_test = np.zeros(len(zeroes_test))
    labels_ones_train = np.ones(len(ones_train))
    labels_ones_test = np.ones(len(ones_test))

    #we concatenate the data and the labels wihtin a np array
    train_data = np.concatenate((zeroes_train, ones_train), axis=0)
    train_labels = np.concatenate((labels_zeroes_train, labels_ones_train), axis=0)
    test_data = np.concatenate((zeroes_test, ones_test), axis=0)
    test_labels = np.concatenate((labels_zeroes_test, labels_ones_test), axis=0)

    #make a dataframe
    train_df = pd.DataFrame({'data': list(train_data), 'label': train_labels})
    test_df = pd.DataFrame({'data': list(test_data), 'label': test_labels})

    #shuffle them
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    #making the challenge dataset
    two_challenge,three_challenge,four_challenge, five_challenge, six_challenge, seven_challenge, eight_challenge,nine_challenge = data[1000:1100], data[1500:1600], data[2000:2100], data[2500:2600], data[3000:3100], data[3500:3600], data[4000:4100], data[4500:4600]

    #grab the labels
    labels_two_challenge = np.full(100, 2)
    labels_three_challenge = np.full(100, 3)
    labels_four_challenge = np.full(100, 4)
    labels_five_challenge = np.full(100, 5)
    labels_six_challenge = np.full(100, 6)
    labels_seven_challenge = np.full(100, 7)
    labels_eight_challenge = np.full(100, 8)
    labels_nine_challenge = np.full(100, 9)

    #concatenate the data and the labels
    challenge_data = np.concatenate((two_challenge, three_challenge, four_challenge, five_challenge, six_challenge, seven_challenge, eight_challenge, nine_challenge), axis=0)
    challenge_labels = np.concatenate((labels_two_challenge, labels_three_challenge, labels_four_challenge, labels_five_challenge, labels_six_challenge, labels_seven_challenge, labels_eight_challenge, labels_nine_challenge), axis=0)

    #make a dataframe
    challenge_df = pd.DataFrame({'data': list(challenge_data), 'label': challenge_labels})

    return train_df, test_df, challenge_df


def evaluate_perceptron_with_bias_range(perceptron, df, bias_weight, range_size=10):
    # Generate the range of bias values around the trained bias weight
    bias_values = np.linspace(bias_weight - range_size, bias_weight + range_size, 21)
    
    # Store metrics for each bias value
    error_fractions = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_points = []

    original_weights = perceptron.weights.copy()

    # Evaluate metrics for each bias value
    for bias in bias_values:
        # Set the bias weight
        perceptron.weights[0] = bias
        
        # Evaluate metrics on the test set
        metrics = perceptron.evaluate_metrics(df)
        error_fractions.append(metrics['error_fraction'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        
        # For ROC curve, store TPR (recall) and FPR
        tpr = metrics['recall']
        fpr = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        roc_points.append((fpr, tpr))
    
    # Restore original weights
    perceptron.weights = original_weights

    # Plot metrics against bias values
    plt.figure(figsize=(12, 8))
    plt.plot(bias_values, error_fractions, label='Error Fraction', marker='o')
    plt.plot(bias_values, precisions, label='Precision', marker='o')
    plt.plot(bias_values, recalls, label='Recall', marker='o')
    plt.plot(bias_values, f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Bias Weight (w0)')
    plt.ylabel('Metrics')
    plt.title('Model Metrics vs. Bias Weight')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot ROC curve
    roc_points = np.array(roc_points)
    plt.figure(figsize=(8, 6))
    plt.plot(roc_points[:, 0], roc_points[:, 1], marker='o')
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.show()

    # Estimate the best bias value (theta*)
    # Find the bias value with the highest TPR for a given FPR (could use other criteria as needed)
    best_index = np.argmax(roc_points[:, 1] - roc_points[:, 0])  # Maximize TPR - FPR
    theta_star = bias_values[best_index]

    # Report if the trained value is the best
    is_best = (bias_weight == theta_star)
    if is_best:
        print("The trained bias weight is the best value.")
    else:
        print(f"Estimated best bias value (θ*): {theta_star}")
    
    return theta_star

#question 1
train_df, test_df, challenge_df = loadMNIST()

#question 2
model = Perceptron()
model.save_initial_weights()

#question 3
error = model.evaluate_metrics(train_df) ["error_fraction"]
print(f"Error Fraction based off of training set: {error:.2f}")
before_training_stats = model.evaluate_metrics(test_df)
print(before_training_stats)

#question 4
errors,_= model.train(train_df, epochs=1000, learning_rate=0.01)
print(errors)

#question 5
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Training Error')
plt.show()

#question 6
after_training_stats = model.evaluate_metrics(test_df)
metrics = ["Error Fraction", "Precision", "Recall", "F1 Score"]
before_values = list(before_training_stats.values())[-4:]
after_values = list(after_training_stats.values())[-4:]

bar_width = 0.35
index = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(index, before_values, bar_width, label='Before Training', color='blue')
bar2 = ax.bar(index + bar_width, after_values, bar_width, label='After Training', color='green')

# Adding labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of Model Metrics Before and After Training')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics, rotation=45)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

#question 7
evaluate_perceptron_with_bias_range(model, test_df, model.weights[0], range_size=10)

#question 8
initial_weights = np.loadtxt('initial_weights.txt')[1:].reshape(28, 28)
trained_weights = model.weights[1:].reshape(28, 28)  # Exclude bias weight

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Initial weights heat map
axes[0].imshow(initial_weights, cmap='gray', aspect='auto')
axes[0].set_title('Initial Weights Heat Map')
axes[0].set_xlabel('Weight Index')
axes[0].set_ylabel('Weight Index')

# Trained weights heat map
axes[1].imshow(trained_weights, cmap='gray', aspect='auto')
axes[1].set_title('Trained Weights Heat Map')
axes[1].set_xlabel('Weight Index')
axes[1].set_ylabel('Weight Index')

plt.tight_layout()
plt.show()

classification_counts = np.zeros((2, 8), dtype=int)

for _, row in challenge_df.iterrows():
    image = row['data'].flatten()
    label = int(row['label'])  
    
    prediction = model.forward(image) 
    
    column_index = label - 2
    classification_counts[prediction, column_index] += 1

# Display results as a table
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(classification_counts, cmap='Blues', aspect='auto')

# Set axis labels
ax.set_xticks(np.arange(8))
ax.set_yticks(np.arange(2))
ax.set_xticklabels([str(digit) for digit in range(2, 10)])
ax.set_yticklabels(['Classified as 0', 'Classified as 1'])

# Add text annotations to each cell
for i in range(2):
    for j in range(8):
        ax.text(j, i, classification_counts[i, j], ha='center', va='center', color='black')

# Add color bar and titles
ax.set_xlabel('Digits (2 to 9)')
ax.set_ylabel('Classification')
ax.set_title('Classification Counts of Challenge Set by Perceptron Model')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

plt.tight_layout()
plt.show()



