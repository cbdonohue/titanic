import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.optim as optim
from mlxtend.preprocessing import minmax_scaling
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from torchviz import make_dot
from sklearn.impute import SimpleImputer

df = pd.read_csv('train.csv')
df.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

df.isna().sum()

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df['Age'].values.reshape(-1, 1))
df['Age'] = imp.transform(df['Age'].values.reshape(-1, 1))

df.isna().sum()

df.dropna(inplace=True)

df = pd.get_dummies(df, columns={'Sex'})
df = pd.get_dummies(df, columns={'Pclass'})
df = pd.get_dummies(df, columns={'Embarked'})
df.head()

features = ['Age', 'Fare','Sex_female', 'Sex_male', 'Pclass_1',	'Pclass_2',	'Pclass_3', 'SibSp', 'SibSp', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2, random_state=1)

age_scaler = RobustScaler()
age_scaler.fit(X_train['Age'].values.reshape(-1, 1))
X_train['Age'] = pd.DataFrame(age_scaler.transform(X_train['Age'].values.reshape(-1, 1)), index=X_train.index)

sibsp_scaler = RobustScaler()
sibsp_scaler.fit(X_train['SibSp'].values.reshape(-1, 1))
X_train['SibSp'] = pd.DataFrame(sibsp_scaler.transform(X_train['SibSp'].values.reshape(-1, 1)), index=X_train.index)

parch_scaler = RobustScaler()
parch_scaler.fit(X_train['Parch'].values.reshape(-1, 1))
X_train['Parch'] = pd.DataFrame(parch_scaler.transform(X_train['Parch'].values.reshape(-1, 1)), index=X_train.index)

fare_scaler = RobustScaler()
fare_scaler.fit(X_train['Fare'].values.reshape(-1, 1))
X_train['Fare'] = pd.DataFrame(fare_scaler.transform(X_train['Fare'].values.reshape(-1, 1)), index=X_train.index)

X_test['Age'] = pd.DataFrame(age_scaler.transform(X_test['Age'].values.reshape(-1, 1)), index=X_test.index)

X_test['SibSp'] = pd.DataFrame(sibsp_scaler.transform(X_test['SibSp'].values.reshape(-1, 1)), index=X_test.index)

X_test['Parch'] = pd.DataFrame(parch_scaler.transform(X_test['Parch'].values.reshape(-1, 1)), index=X_test.index)

X_test['Fare'] = pd.DataFrame(fare_scaler.transform(X_test['Fare'].values.reshape(-1, 1)), index=X_test.index)

class TitanicDataset(Dataset):
    def __init__(self, x, y=None):
      self.x = x
      self.y = y
        
    def __len__(self):
      return len(self.x)

    def __getitem__(self, idx):
      if self.y is None:
        return \
              torch.tensor(self.x.iloc[idx][features]).float()
      else:
        return \
              torch.tensor(self.x.iloc[idx][features]).float(), \
              torch.tensor([self.y.iloc[idx]]).float()

BATCH_SIZE = 32

train_dataset = TitanicDataset(X_train, y_train)

titanic_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TitanicDataset(X_test, y_test)

titanic_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.double()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(features), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model

loss_fn = nn.MSELoss()
learning_rate = .001
batch_size = 32
epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
min_valid_loss = np.inf

X_train.tail()

criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

valid_losses = list()
train_losses = list()

for epoch in range(epochs):
    train_loss = 0.0
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(titanic_train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in titanic_test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        target = model(data)
        loss = criterion(target,labels)
        valid_loss = loss.item() * data.size(0)

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(titanic_test_loader)} \t\t Validation Loss: {valid_loss / len(titanic_test_loader)}')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

plt.plot(valid_losses, label='validation loss')
plt.plot(train_losses, label='training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

df = pd.read_csv('test.csv')
df.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

df = pd.get_dummies(df, columns={'Sex'})
df = pd.get_dummies(df, columns={'Pclass'})
df = pd.get_dummies(df, columns={'Embarked'})

df['Age'] = pd.DataFrame(age_scaler.transform(df['Age'].values.reshape(-1, 1)), index=df.index)
df['SibSp'] = pd.DataFrame(sibsp_scaler.transform(df['SibSp'].values.reshape(-1, 1)), index=df.index)
df['Parch'] = pd.DataFrame(parch_scaler.transform(df['Parch'].values.reshape(-1, 1)), index=df.index)
df['Fare'] = pd.DataFrame(fare_scaler.transform(df['Fare'].values.reshape(-1, 1)), index=df.index)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df['Age'].values.reshape(-1, 1))
df['Age'] = imp.transform(df['Age'].values.reshape(-1, 1))

df.tail()

df.tail()

test_dataset = TitanicDataset(df)
titanic_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NeuralNetwork().cuda()
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()

preds = list()
for data in titanic_test_loader:
  if torch.cuda.is_available():
      data, labels = data.cuda(), labels.cuda()
  target = model(data)
  preds = preds + target.tolist()
preds[:10]

def func(i):
  return i[0]
preds = list(map(lambda i:func(i), preds))
result = pd.DataFrame(index = df['PassengerId'])
result['Survived'] = pd.DataFrame(preds, index=result.index)
result['Survived'] = np.where(result['Survived'] < .5, 0, 1)
result.to_csv('submission.csv')

make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
