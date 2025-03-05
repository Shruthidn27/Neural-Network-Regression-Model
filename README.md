# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
This project aims to develop a Neural Network Regression Model capable of accurately predicting a target variable based on given input features. By leveraging deep learning techniques, the model will learn complex patterns within the dataset to deliver reliable and precise predictions.

## Neural Network Model
![image](https://github.com/user-attachments/assets/577f7be1-0f5c-4483-8455-c33f86e0dedd)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Shruthi D.N
### Register Number: 212223240155
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,16)
        self.fc3 = nn.Linear(16,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, criterion, optimizer, X_train_tensor, y_train_tensor, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Using X_train_tensor and y_train_tensor here
        loss = criterion(ai_brain(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')



```
## Dataset Information

![image](https://github.com/user-attachments/assets/a278ba5d-f013-4cc2-b19d-13e9c726f883)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/a4f1e60a-1443-4b7f-9fce-6226b98e893a)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/2865f1c2-fa61-4b16-8e0c-55762d50b7c4)


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
