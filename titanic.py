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


class TitanicNN(nn.Module):
  FEATURES = ['Age', 'Fare','Sex_female', 'Sex_male', 'Pclass_1',	'Pclass_2',	'Pclass_3', 'SibSp', 'SibSp', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
  BATCH_SIZE = 32

  def __init__(self):
    super(TitanicNN, self).__init__()

    self.double()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(len(self.FEATURES), 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    self.load_data()
    self.impute()
    self.clean()
    self.one_hot()
    self.split()
    self.scale()
    self.transform()
    self.cuda()
    self.dataload()
    self.train_model()
    self.test()
    self.pred()
    self.submit()

  def dataload(self):
    self.train_dataset = TitanicDataset(self.X_train, self.y_train)
    self.titanic_train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
    self.valid_dataset = TitanicDataset(self.X_valid, self.y_valid)
    self.titanic_valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
    self.test_dataset = TitanicDataset(self.X_test)
    self.titanic_test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

  def cuda(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)

  def load_data(self):
    self.train_data = pd.read_csv('train.csv')
    self.train_data.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

    self.test_data = pd.read_csv('test.csv')
    self.test_data.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
    self.test_data = self.test_data

  def impute(self):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(self.train_data['Age'].values.reshape(-1, 1))
    self.train_data['Age'] = imp.transform(self.train_data['Age'].values.reshape(-1, 1))

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(self.test_data['Age'].values.reshape(-1, 1))
    self.test_data['Age'] = imp.transform(self.test_data['Age'].values.reshape(-1, 1))

  def clean(self):
    self.train_data.dropna(inplace=True)

  def one_hot(self):
    self.train_data = pd.get_dummies(self.train_data, columns={'Sex'})
    self.train_data = pd.get_dummies(self.train_data, columns={'Pclass'})
    self.train_data = pd.get_dummies(self.train_data, columns={'Embarked'})

    self.test_data = pd.get_dummies(self.test_data, columns={'Sex'})
    self.test_data = pd.get_dummies(self.test_data, columns={'Pclass'})
    self.test_data = pd.get_dummies(self.test_data, columns={'Embarked'})

  def split(self):
      self.X_train, self.X_valid, self.y_train, self.y_valid = \
        train_test_split(self.train_data.drop(columns=['Survived']), self.train_data['Survived'], test_size=0.2, random_state=1)
      self.X_test = self.test_data

  def scale(self):
    self.age_scaler = RobustScaler()
    self.age_scaler.fit(self.X_train['Age'].values.reshape(-1, 1))
    self.X_train['Age'] = pd.DataFrame(self.age_scaler.transform(self.X_train['Age'].values.reshape(-1, 1)), index=self.X_train.index)

    self.sibsp_scaler = RobustScaler()
    self.sibsp_scaler.fit(self.X_train['SibSp'].values.reshape(-1, 1))
    self.X_train['SibSp'] = pd.DataFrame(self.sibsp_scaler.transform(self.X_train['SibSp'].values.reshape(-1, 1)), index=self.X_train.index)

    self.parch_scaler = RobustScaler()
    self.parch_scaler.fit(self.X_train['Parch'].values.reshape(-1, 1))
    self.X_train['Parch'] = pd.DataFrame(self.parch_scaler.transform(self.X_train['Parch'].values.reshape(-1, 1)), index=self.X_train.index)

    self.fare_scaler = RobustScaler()
    self.fare_scaler.fit(self.X_train['Fare'].values.reshape(-1, 1))
    self.X_train['Fare'] = pd.DataFrame(self.fare_scaler.transform(self.X_train['Fare'].values.reshape(-1, 1)), index=self.X_train.index)

  def transform(self):
    self.X_valid['Fare'] = pd.DataFrame(self.fare_scaler.transform(self.X_valid['Fare'].values.reshape(-1, 1)), index=self.X_valid.index)
    self.X_valid['Age'] = pd.DataFrame(self.age_scaler.transform(self.X_valid['Age'].values.reshape(-1, 1)), index=self.X_valid.index)
    self.X_valid['SibSp'] = pd.DataFrame(self.sibsp_scaler.transform(self.X_valid['SibSp'].values.reshape(-1, 1)), index=self.X_valid.index)
    self.X_valid['Parch'] = pd.DataFrame(self.parch_scaler.transform(self.X_valid['Parch'].values.reshape(-1, 1)), index=self.X_valid.index)

    self.X_test['Age'] = pd.DataFrame(self.age_scaler.transform(self.X_test['Age'].values.reshape(-1, 1)), index=self.X_test.index)
    self.X_test['SibSp'] = pd.DataFrame(self.sibsp_scaler.transform(self.X_test['SibSp'].values.reshape(-1, 1)), index=self.X_test.index)
    self.X_test['Parch'] = pd.DataFrame(self.parch_scaler.transform(self.X_test['Parch'].values.reshape(-1, 1)), index=self.X_test.index)
    self.X_test['Fare'] = pd.DataFrame(self.fare_scaler.transform(self.X_test['Fare'].values.reshape(-1, 1)), index=self.X_test.index)

  def train_model(self):
    learning_rate = .001
    optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    epochs = 30
    criterion = nn.MSELoss()
    min_valid_loss = np.inf

    valid_losses = list()
    train_losses = list()

    for epoch in range(epochs):
      train_loss = 0.0
      # enumerate mini batches
      for i, (inputs, targets) in enumerate(self.titanic_train_loader):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = self(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()

      valid_loss = 0.0
      self.eval()     # Optional when not using Model Specific layer
      for data, labels in self.titanic_valid_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        target = self(data)
        loss = criterion(target,labels)
        valid_loss = loss.item() * data.size(0)

      print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(self.titanic_test_loader)} \t\t Validation Loss: {valid_loss / len(self.titanic_test_loader)}')
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)

      if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(self.state_dict(), 'saved_model.pth')

  def test(self):
    self.load_state_dict(torch.load('saved_model.pth'))
    self.eval()

  def pred(self):
    self.preds = list()
    for data in self.titanic_test_loader:
      if torch.cuda.is_available():
          data = data.cuda()
      target = self(data)
      self.preds =self.preds + target.tolist()
  
  def submit(self):
    def func(i):
      return i[0]
    self.preds = list(map(lambda i:func(i), self.preds))
    result = pd.DataFrame(index = self.test_data['PassengerId'])
    result['Survived'] = pd.DataFrame(self.preds, index=result.index)
    result['Survived'] = np.where(result['Survived'] < .5, 0, 1)
    result.to_csv('submission.csv')

  def forward(self, x):
      #x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

class TitanicDataset(Dataset):
    def __init__(self, x, y=None):
      self.x = x
      self.y = y
        
    def __len__(self):
      return len(self.x)

    def __getitem__(self, idx):
      if self.y is None:
        return \
              torch.tensor(self.x.iloc[idx][TitanicNN.FEATURES]).float()
      else:
        return \
              torch.tensor(self.x.iloc[idx][TitanicNN.FEATURES]).float(), \
              torch.tensor([self.y.iloc[idx]]).float()

if __name__ == '__main__':
    nn = TitanicNN()
