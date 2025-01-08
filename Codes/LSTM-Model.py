import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 1: Data Preprocessing

# Loading the CSV file
data = pd.read_csv('DataCleanUP.csv')

# Handle missing values (if any)
data.fillna('', inplace=True)

# Extracting text and numerical features
text_features = data.iloc[:, 2]  # Text is in the third column
numerical_features = data.iloc[:, 3:]  # Extract all numerical features

# Normalizing numerical features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# ------------------------------------------------------------------------------------

# Step 2: Feature Extraction

# Extracting text features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
text_features_tfidf = tfidf_vectorizer.fit_transform(text_features).toarray()

# Combining text and numerical features
combined_features = np.hstack((text_features_tfidf, numerical_features_scaled))

# ------------------------------------------------------------------------------------

# Step 3: Splitting the data into training, validation, and test sets
# Comment or Uncomment depending on the target column !

# Target column
target_column = data.columns[0]  # NCefr = 0 (3 Niveaus A,B,C)
#target_column = data.columns[1] # Cefr = 1 (6 Niveaus A1-C2)

# Mapping CEFR levels
cefr_mapping_6 = {  # Cefr = 1
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6
}


cefr_mapping_3 = {  # NCefr = 0
    'A': 1,
    'B': 2,
    'C': 3,
}

# Applying mapping to the target column
data[target_column] = data[target_column].map(cefr_mapping_3)
#data[target_column] = data[target_column].map(cefr_mapping_6)

# Target column contains numerical values for CEFR levels (1-6) or (1-3)
target = pd.get_dummies(data[target_column])

# Splitting into training and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(combined_features, target, test_size=0.4, random_state=42)

# Splitting temp set into validation and test sets
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Converting to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Creating DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# ------------------------------------------------------------------------------------

# Step 4: LSTM Model Definition

class CEFRLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CEFRLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

# Instantiating the model
input_size = combined_features.shape[1]
hidden_size = 8
num_layers = 2
output_size = 3 # 3 for CEFR 3
#output_size = 6 # 6 for CEFR 6
model = CEFRLSTMClassifier(input_size, hidden_size, num_layers, output_size)

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
early_stopping_patience = 3
min_valid_loss = np.inf
patience_counter = 0
best_model = None

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            valid_loss += loss.item()

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)

    print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        patience_counter = 0
        best_model = model.state_dict()
    else:
        patience_counter += 1

    if patience_counter > early_stopping_patience:
        print("Early stopping triggered!")
        if best_model is not None:
            model.load_state_dict(best_model)
        break

# Step 5: Evaluation

model.eval()
y_true, y_pred_classes = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        y_pred_classes_batch = torch.argmax(y_pred, dim=1)
        y_true_classes_batch = torch.argmax(y_batch, dim=1)
        y_true.extend(y_true_classes_batch.numpy())
        y_pred_classes.extend(y_pred_classes_batch.numpy())

# Computing confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# ------------------------------------------------------------------------------------

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))

cefr_labels_3 = ['A', 'B', 'C']
#cefr_labels_6 = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cefr_labels_3, yticklabels=cefr_labels_3)
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cefr_labels_6, yticklabels=cefr_labels_6)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
#plt.savefig('OLD-CM_cefr3_LSTM.png')
#plt.savefig('OLD-CM_cefr6_LSTM.png')
#plt.savefig('CM_cefr3_LSTM_NEW.png')
#plt.savefig('CM_cefr6_LSTM_NEW.png')
plt.show()


# ------------------------------------------------------------------------------------

# Computing accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)

# Printing the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Printing detailed classification report
print('Classification Report:')
print(classification_report(y_true, y_pred_classes, target_names=cefr_mapping_3.keys()))
#print(classification_report(y_true, y_pred_classes, target_names=cefr_mapping_6.keys()))

# ------------------------------------------------------------------------------------
''' 
# Results with 3 levels
Accuracy: 0.8097560975609757
Precision: 0.8095470383275262
Recall: 0.8097560975609757
F1-score: 0.8088147981170415
Classification Report:
              precision    recall  f1-score   support

           A       0.80      0.88      0.84        96
           B       0.80      0.75      0.77       177
           C       0.83      0.85      0.84       137

    accuracy                           0.81       410
   macro avg       0.81      0.82      0.82       410
weighted avg       0.81      0.81      0.81       410


# Results with 6 levels
Accuracy: 0.6292682926829268
Precision: 0.6391312611844738
Recall: 0.6292682926829268
F1-score: 0.6251616732147671
Classification Report:
              precision    recall  f1-score   support

          A1       0.96      0.72      0.82        32
          A2       0.60      0.81      0.69        64
          B1       0.64      0.42      0.50        77
          B2       0.62      0.66      0.64       100
          C1       0.58      0.65      0.62        92
          C2       0.62      0.56      0.59        45

    accuracy                           0.63       410
   macro avg       0.67      0.64      0.64       410
weighted avg       0.64      0.63      0.63       410

# -------------------------------------------------------------------
# New Results with new data (new features)

#Results with 3 levels
Accuracy: 0.8341463414634146
Precision: 0.8355251655384257
Recall: 0.8341463414634146
F1-score: 0.8326081687218646
Classification Report:
              precision    recall  f1-score   support

           A       0.81      0.92      0.86        96
           B       0.85      0.75      0.80       177
           C       0.84      0.88      0.86       137

    accuracy                           0.83       410
   macro avg       0.83      0.85      0.84       410
weighted avg       0.84      0.83      0.83       410

#Results with 6 levels

Accuracy: 0.697560975609756
Precision: 0.7043505550652954
Recall: 0.697560975609756
F1-score: 0.6959039951704207
Classification Report:
              precision    recall  f1-score   support

          A1       0.92      0.69      0.79        32
          A2       0.66      0.83      0.74        64
          B1       0.68      0.61      0.64        77
          B2       0.71      0.74      0.73       100
          C1       0.66      0.72      0.69        92
          C2       0.73      0.53      0.62        45


'''