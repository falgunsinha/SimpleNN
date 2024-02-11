import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix


# Load the dataset
df = pd.read_csv('./iris-enc.csv')

# Extract features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Z-score normalization
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Maximum input values
max_input_values = np.max(X, axis=0)
print("Maximum input values:")
print(max_input_values)

# Normalized inputs
print("\nNormalized inputs:")
print(X_train_normalized[:5])

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=X.shape[1], activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(X_train_normalized, y_train, epochs=100, batch_size=10)

# Predictions
y_pred = model.predict(X_test_normalized)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Number of unclassified samples
num_unclassified = np.sum(y_pred_classes != y_test)
print(f"Number of unclassified samples: {num_unclassified}")