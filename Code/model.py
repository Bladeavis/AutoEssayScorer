import pandas as pd
from huggingface_hub.keras_mixin import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 1: Data Preprocessing

# Load the CSV file
data = pd.read_csv('All_Data_with_features_new.csv')

# Handle missing values (if any)
data.fillna('', inplace=True)

# Extract text and numerical features
text_features = data.iloc[:, 2]  # Assuming text is in the second column
numerical_features = data.iloc[:, 3:]  # Extract all numerical features

# numerical_features = data.iloc[:, 2:-1]  # Extract all numerical features without Wiener Sachtextformel
# numerical_features = data.iloc[:, 2:].drop(columns=['Flesch_Reading_Ease'])

# Normalize numerical features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# ------------------------------------------------------------------------------------

# Step 2: Feature Extraction

# Extract text features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
text_features_tfidf = tfidf_vectorizer.fit_transform(text_features).toarray()

# Combine text and numerical features
combined_features = np.hstack((text_features_tfidf, numerical_features_scaled))

# ------------------------------------------------------------------------------------

# Step 3: Split the data into training, validation, and test sets

# Assuming the last column is the target column
target_column = data.columns[0]  # Cefr = 1 , NCefr = 0

# Create a dictionary to map CEFR levels to numerical values
"""
cefr_mapping = {  # Cefr = 1
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6
}
"""

cefr_mapping = {   # NCefr = 0
    'A': 1,
    'B': 2,
    'C': 3,
}

# Apply this mapping to the target column
data[target_column] = data[target_column].map(cefr_mapping)

# Now the target column contains numerical values for CEFR levels
target = data[target_column]
target = pd.get_dummies(target)  # Convert target values to categorical x6 columns

# Split into training and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(combined_features, target, test_size=0.4, random_state=42)

# Split temp set into validation and test sets
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert target values to categorical
y_train = pd.get_dummies(y_train)
y_valid = pd.get_dummies(y_valid)
y_test = pd.get_dummies(y_test)

# ------------------------------------------------------------------------------------

# Step 4: Neural Network Model

# Define the neural network architecture
model = tf.keras.Sequential([
    Input(shape=(combined_features.shape[1],)),
    Dense(8, activation='relu'),  # 64 (with 6 levels) -   8 (with 3 levels)
    Dropout(0.25),                       # 0.25
    Dense(4, activation='relu'),  # 16 (with 6 levels) -   4 (with 3 levels)
    Dropout(0.25),
    Dense(3, activation='softmax')  # 3 levels or 6 levels !
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Adjust loss for regression

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid),
          callbacks=[callback], verbose=1)


#  epoch = 50 first try
# 1) 100 epoch, batch_size=32 (first classification report) - accuracy 0.89 - with 6 levels
# 2) 100 epoch, batch_size=64 (second classification report) - accuracy 0.84 - with 6 levels
# 3) 100 epoch, batch_size=16 (third classification report) - accuracy 0.89 - with 6 levels

# ------------------------------------------------------------------------------------

# Evaluate the model


# Save the model
# model.save('cefr_classifier_model.h5')


# ------------------------------------------------------------------------------------

# Predict the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test.values, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cefr_mapping.keys(), yticklabels=cefr_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Test')
plt.show()

# ------------------------------------------------------------------------------------

# Compute accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Print the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print('Confusion Matrix:')
print(cm)

# Print detailed classification report
print('Classification Report:')
print(classification_report(y_true_classes, y_pred_classes, target_names=cefr_mapping.keys()))

# ------------------------------------------------------------------------------------

# import pandas as pd

# Assuming your data is in a pandas DataFrame `df`
# and your target variable is in the column 'target'
# correlations = pd.DataFrame(combined_features).corr()

# Correlation of each feature with the target variable
# target_correlations = correlations['target'].sort_values(ascending=False)
# print(target_correlations)