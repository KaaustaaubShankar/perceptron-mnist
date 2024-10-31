import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


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

def evaluate_perceptron_with_bias_range(perceptron, df, range_size=10):
    bias_weight = perceptron.bias
    bias_values = np.linspace(bias_weight - range_size, bias_weight + range_size, 21)
    error_fractions, precisions, recalls, f1_scores, roc_points = [], [], [], [], []
    original_bias = perceptron.bias

    for bias in bias_values:
        perceptron.bias = bias
        metrics = perceptron.evaluate_metrics(df)
        error_fractions.append(metrics['error_fraction'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        tpr = metrics['recall']
        fpr = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        roc_points.append((fpr, tpr))
    
    perceptron.bias = original_bias

    plt.figure(figsize=(12, 8))
    plt.plot(bias_values, error_fractions, label='Error Fraction', marker='o')
    plt.plot(bias_values, precisions, label='Precision', marker='o')
    plt.plot(bias_values, recalls, label='Recall', marker='o')
    plt.plot(bias_values, f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Bias')
    plt.ylabel('Metrics')
    plt.title('Model Metrics vs. Bias')
    plt.legend()
    plt.grid()
    plt.show()

    roc_points = np.array(roc_points)
    plt.figure(figsize=(8, 6))
    plt.plot(roc_points[:, 0], roc_points[:, 1], marker='o')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.show()

    best_index = np.argmax(roc_points[:, 1] - roc_points[:, 0])
    theta_star = bias_values[best_index]
    is_best = (bias_weight == theta_star)
    if is_best:
        print(f"The trained bias weight is the best value. {theta_star}")
    else:
        print(f"Estimated best bias value (θ*): {theta_star}")
    
    return theta_star

#question 1
train_df, test_df, challenge_df = loadMNIST()

#question 2
model = Perceptron()
model.save_initial_weights() #this is being saved to a file

#question 3
error = model.evaluate_metrics(train_df) ["error_fraction"]
print(f"Error Fraction based off of training set: {error:.2f}")

#stats based off test dataset
before_training_stats = model.evaluate_metrics(test_df)
print(before_training_stats)

#question 4
error_rate,_= model.train(train_df, epochs=25, learning_rate=0.001,target_error=0.01,stop_early=False)
print(error_rate)

#question 5
plt.plot(error_rate, marker='o')
for i, rate in enumerate(error_rate):
    if (i+1) % 5 == 0:
        plt.text(i, rate, f'{rate:.4f}', ha='center', va='bottom')
plt.xlabel('Epochs (k=50)')
plt.ylabel('Error Rate')
plt.title('Perceptron Training Error vs Epochs')
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

for i in range(len(metrics)):
    ax.text(index[i], before_values[i] + 0.01, f'{before_values[i]:.2f}', ha='center', va='bottom', color='blue')
    ax.text(index[i] + bar_width, after_values[i] + 0.01, f'{after_values[i]:.2f}', ha='center', va='bottom', color='green')

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
evaluate_perceptron_with_bias_range(model, test_df, range_size=10)

#question 8
initial_weights = np.loadtxt('initial_weights.txt')[:784].reshape(28, 28)
trained_weights = model.weights.reshape(28, 28)  # Exclude bias weight

print(f"InitialWeights: {np.sum(initial_weights)}")
print(f"Final Weights: {np.sum(trained_weights)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Initial weights heat map
axes[0].imshow(initial_weights, cmap='coolwarm', aspect='auto')
axes[0].set_title('Initial Weights Heat Map')
axes[0].set_xlabel('Weight Index')
axes[0].set_ylabel('Weight Index')

# Trained weights heat map
axes[1].imshow(trained_weights, cmap='coolwarm', aspect='auto')
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

# text annotations
for i in range(2):
    for j in range(8):
        ax.text(j, i, classification_counts[i, j], ha='center', va='center', color='black')

ax.set_xlabel('Digits (2 to 9)')
ax.set_ylabel('Classification')
ax.set_title('Classification Counts of Challenge Set by Perceptron Model')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

plt.tight_layout()
plt.show()



