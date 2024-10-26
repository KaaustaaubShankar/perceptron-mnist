import numpy as np

class Perceptron:
    def __init__(self, input_size=784):
        self.input_size = input_size
        # Initialize weights randomly between -0.5 and 0.5 for better symmetry
        self.weights = np.random.uniform(-0.5, 0.5, input_size + 1)  # +1 for bias weight

    def forward(self, image):
        # Add the bias input x0 = 1 to the image input array
        bias_input = 1
        image_with_bias = np.insert(image, 0, bias_input)
        
        # Compute the net input s(t)
        net_input = np.dot(self.weights, image_with_bias)
        
        # Apply the activation function (step function)
        output = 1 if net_input > 0 else 0
        
        return output

    def save_initial_weights(self, filename="initial_weights.txt"):
        np.savetxt(filename, self.weights, fmt='%f')
        print(f"Weights saved to {filename}")

    def accuracy(self, df):
        correct = 0
        for _, row in df.iterrows():
            image = row['data'].flatten()
            label = row['label']
            prediction = self.forward(image)
            if prediction == label:
                correct += 1
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

        #tpr and tnr for balanced accuracy
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate for the positive class
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate for the negative class
        balanced_accuracy = (tpr + tnr) / 2
        error_fraction = 1 - balanced_accuracy  # Error is defined as 1 - Balanced Accuracy

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'error_fraction': error_fraction,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'balanced_accuracy': balanced_accuracy
        }


    def train(self, df, epochs=1, learning_rate=0.1,target_error=0):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            # Shuffle the training data at the start of each epoch
            df = df.sample(frac=1).reset_index(drop=True)
            
            for _, row in df.iterrows():
                image = row['data'].flatten()
                label = row['label']
                prediction = self.forward(image)
                
                error = label - prediction
                if error != 0:
                    total_error += 1
                
                #doing the entire list in one go
                self.weights += learning_rate * error * np.insert(image, 0, 1)
            
            # Calculate and record the error rate
            error_rate = total_error / len(df)
            errors.append(error_rate)
            print(f"Epoch {epoch + 1}: Error rate = {error_rate:.4f}")
            
            # Early stopping if error rate is zero
            if error_rate <= target_error:
                print("Training complete with below target error")
                break
        
        return errors, epoch + 1
