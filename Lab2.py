#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Function to create random data
def create_data():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    return X, y

# Function to create a simple model
def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(10,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Function to train the model with early stopping and return loss history
def train_model_with_early_stopping(model, optimizer, X, y, batch_size, epochs, optimizer_name, patience=5):
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = []
    
    # Early stopping logic
    best_loss = np.inf
    wait = 0  # Counter for patience
    for epoch in range(epochs):
        hist = model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0)
        loss = hist.history['loss'][0]
        history.append(loss)
        
        print(f'Epoch {epoch + 1}/{epochs} - {optimizer_name} loss: {loss:.4f}')
        
        # Check for early stopping condition
        if loss < best_loss:
            best_loss = loss
            wait = 0
        else:
            wait += 1
        
        if wait >= patience:
            print(f"Early stopping after epoch {epoch + 1}")
            break
    
    return history

# Create data
X, y = create_data()

# Create two models with different optimizers
model_sgd = create_model()
model_adam = create_model()

# Define the SGD and Adam optimizers
optimizer_sgd = optimizers.SGD(learning_rate=0.01)
optimizer_adam = optimizers.Adam(learning_rate=0.001)

epochs = 5000
batch_size = 32
patience = 3  # Stop if no improvement after 3 epochs

# Train with SGD and collect the loss history
print('\nTraining with SGD optimizer: ')
sgd_loss_history = train_model_with_early_stopping(model_sgd, optimizer_sgd, X, y, batch_size, epochs, 'SGD', patience)

# Train with Adam and collect the loss history
print('\nTraining with Adam optimizer: ')
adam_loss_history = train_model_with_early_stopping(model_adam, optimizer_adam, X, y, batch_size, epochs, 'Adam', patience)

# After training is complete, plot the loss histories
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(sgd_loss_history) + 1), sgd_loss_history, label='SGD Loss', color='blue')
plt.plot(range(1, len(adam_loss_history) + 1), adam_loss_history, label='Adam Loss', color='red')
plt.title('Loss Over Epochs for SGD and Adam Optimizers with Early Stopping')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

