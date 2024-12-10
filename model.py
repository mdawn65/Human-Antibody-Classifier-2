import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import logging

class StepGraphCallback(Callback):
    def __init__(self):
        # Initialize empty lists to store step-level metrics
        self.steps = []
        self.losses = []
        self.accuracies = []

    def on_train_batch_end(self, batch, logs=None):
        # Append the current step and its metrics
        self.steps.append(batch + 1)  # Step index (starting from 1)
        self.losses.append(logs.get('loss'))  # Loss for the current step
        self.accuracies.append(logs.get('accuracy'))  # Accuracy for the current step

    def on_train_end(self, logs=None):
        # Automatically plot the graphs after training completes
        self.plot_metrics()

    def plot_metrics(self):
        # Plot Loss and Accuracy Per Step
        plt.figure(figsize=(12, 6))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.steps, self.losses, label="Loss", color='red')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss per Step")
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.steps, self.accuracies, label="Accuracy", color='blue')
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Step")
        plt.legend()

        plt.tight_layout()
        plt.show()

# Set logging level to avoid info messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Script started")
# Add logging statements at key points in your script

# Load data
data = pd.read_csv('nmep_antibody_data.csv')
sequences = data['sequence']
labels = data['species']

logging.info("Data loaded")

# Encode labels
amino_acids = "ACDEFGHIKLMNPQRSTVWY" # 20 amino acids
char_to_int = {char: i + 1 for i, char in enumerate(amino_acids)} # Creates a dictionary ('A': 1, 'C': 2, ...) 

# Convert sequences to integers
encoded_sequences = [[char_to_int[c] for c in seq] for seq in sequences] # Assign each sequence's character to corresponding integer

# Pad sequences
max_seq_length = max(len(seq) for seq in encoded_sequences) # Find the longest sequence; this will be what we pad to 
padded_sequences = pad_sequences(encoded_sequences, maxlen=max_seq_length, padding='post') # Pad the sequences so they all have the same length ([1, 2, 3] --> [1, 2, 3, 0, 0])

# Encode labels
label_encoder = LabelEncoder() # One hot encoding
encoded_labels = label_encoder.fit_transform(labels) # Encode the species to integers ('mouse' -> 0, 'human' -> 1, ...)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, encoded_labels, test_size=0.2, random_state=42
) 
# First input: Inputs for the model. Padded_sequences - feature set (X), preprocessed sequences.
# Second input: Target variable. Integer-encoded species labels, predictions the model will learr.
# Testing size: 20% used for testing
# Random state: Seed for random number generator

# X_train: Training features (80% of padded sequences)
# X_test: Testing features (20% of padded sequences)
# y_train: Training labels (80% of encoded labels)
# y_test: Testing labels (20% of encoded labels)

# Build model
model = tf.keras.Sequential([ # Initializes the model as a sequential stack of layers
    # Why? It's simple and works well for models where layers follow a straightforward, linear structure
    Embedding(input_dim=len(amino_acids) + 1, output_dim=32, input_length=max_seq_length), # Embedding layer
        # Why? Converts the integer-encoded amino acids into dense vectors of fixed size (embeddings)
        # Input_dim: Number of unique amino acids + 1 (for padding)
        # Output_dim: Dimension of the dense embedding
        # Input_length: Fixed length of the input sequences
        # Output: Sequence of embeddings (max_seq_length, 32
    LSTM(64, return_sequences=True), # First LSTM layer
        # Process the sequence data with LSTM layer
        # Why? LSTM layers are good for learning patterns in sequences
        # 64: Number of units (neurons) in the LSTM layer. Layer will output a 64-dimension vector at each time step
        # Return_sequences=True: Return the full sequence of outputs (needed bc next LSTM layer requires a sequence input)
        # Output: Sequence of hidden states (max_seq_length, 64)
    LSTM(64), # Second LSTM layer
        # Adds another LSTM layer to capture more complex sequential patterns. Only returns the final hiddne state (not the whole sequence)
        # 64: Number of units in this LSTM layer
        # Output: Final hidden state vector (64)
    Dense(32, activation='relu'), # Hidden layer (Dense layer)
        # Purpose: Adds a fully connected (feedforward) layer to further process the LSTM output and learn non-linear patterns
        # 32: The number of neurons in this dense layer
        # activation='relu': Rectified linear unit (ReLU) activation function, applied to introduce non-linearity and prevent vanishing gradients
        # Output: Vector of size 32
    Dense(len(label_encoder.classes_), activation='softmax') # Output layer (Dense layer)
        # Purpose: Final classification layer, outputting probabilities for each class
        # len(label_encoder.classes_): The number of unique species/classes (e.g. "human", "mouse", etc.)
        # activation='softmax': Converts the outputs into probabilities that sum to 1 across all classes
        # Output: Vector of size len(label_encoder.classes_) (probabilities for each species)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

step_graph_callback = StepGraphCallback()

logging.info("Model training started")

model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=7, 
    batch_size=32,
    callbacks=[step_graph_callback]
    )

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# Save the model

model.save("antibody_model.keras")

model.summary()

model = load_model("antibody_model.keras")
