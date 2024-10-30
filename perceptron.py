import numpy as np

class Perceptron:
    def __init__(self, input_size=784):
        self.input_size = input_size
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = np.random.uniform(-0.5, 0.5)

    def forward(self, image):
        net_input = np.dot(self.weights, image) + self.bias
        return 1 if net_input > 0 else 0

    def save_initial_weights(self, filename="initial_weights.txt"):
        combined_weights = np.append(self.weights, self.bias)
        np.savetxt(filename, combined_weights, fmt='%f')
        print(f"Weights saved to {filename}")

    def accuracy(self, df):
        correct = sum(self.forward(row['data'].flatten()) == row['label'] for _, row in df.iterrows())
        return correct / len(df)

    def evaluate_metrics(self, df):
        tp = tn = fp = fn = 0
        for _, row in df.iterrows():
            image = row['data'].flatten()
            label = row['label']
            prediction = self.forward(image)
            if prediction == 1 and label == 1:
                tp += 1
            elif prediction == 0 and label == 0:
                tn += 1
            elif prediction == 1 and label == 0:
                fp += 1
            elif prediction == 0 and label == 1:
                fn += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (tpr + tnr) / 2
        error_fraction = 1 - balanced_accuracy
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'balanced_accuracy': balanced_accuracy,
            'error_fraction': error_fraction,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score

        }

    def train(self, df, epochs=1, learning_rate=0.1, target_error=0,stop_early=False):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            df = df.sample(frac=1).reset_index(drop=True)
            for _, row in df.iterrows():
                image = row['data'].flatten()
                label = row['label']
                prediction = self.forward(image)
                error = label - prediction
                if error != 0:
                    total_error += 1
                self.weights += learning_rate * error * image
                self.bias += learning_rate * error
            error_rate = total_error / len(df)
            errors.append(error_rate)
            print(f"Epoch {epoch + 1}: Error rate = {error_rate:.4f}")
            if error_rate <= target_error and stop_early:
                print("Training complete with below target error")
                break
        return errors, epoch + 1
