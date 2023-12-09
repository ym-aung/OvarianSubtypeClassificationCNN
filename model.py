from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import csv
import itertools

import numpy as np
import os.path

label_to_num_dict = {
    "HGSC": 0,
    "LGSC": 1,
    "EC": 2,
    "CC": 3,
    "MC": 4
}

image_id_list = []
label_list = []

with open("train.csv", newline="") as csvfile:
  fileReader = csv.DictReader(csvfile, delimiter=",")
  for row in fileReader:
    if row["is_tma"] == "False":
      image_id_list.append(row["image_id"])
      label_list.append(label_to_num_dict[row["label"]])

class_weights = []
for i in range(5):
  class_weights.append(len(label_list)/(label_list.count(i)*5))
class_weights = torch.FloatTensor(class_weights)

class ImageDataset(Dataset):
  def __init__(self, image_id_list, label_list, train_index):
    self.x_data = []
    self.y_data = []

    transform = A.Compose([A.ToFloat(), ToTensorV2()])

    for i in train_index:
      img = np.array(Image.open(os.path.join("processed_train_thumbnails", f"{image_id_list[i]}_thumbnail.png")))
      self.x_data.append(transform(image=img))
      self.y_data.append(label_list[i])


    self.len = len(self.x_data)

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.convlayer1 = nn.Conv2d(3, 32, kernel_size=(5,5), stride=1, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.convlayer2 = nn.Conv2d(32, 32, kernel_size=(5,5), stride=1, padding=1)
        self.layer1 = nn.Linear(28800, 512)
        self.fc = nn.Linear(512, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.convlayer1(X)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.convlayer2(out)
        out = nn.functional.relu(out)
        out = self.maxpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.layer1(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out

def reset_weights(m):
  for i in m.children():
    if hasattr(i, 'reset_parameters'):
      i.reset_parameters()

def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['0', '1', '2', '3', '4'])
    return conf_matrix, classification_report
                                                                       
def print_results(conf_matrix, class_report, fold, accuracy):
    print(f"Results for Fold {fold}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print(f"Accuracy: {accuracy}%")

model = CNNModel()

criterion = nn.CrossEntropyLoss(weight=class_weights)

num_folds = 10
kFold = StratifiedKFold(n_splits=num_folds, shuffle=True)
splits = kFold.split(image_id_list, label_list)
num_epochs = 10

results = {}

for n,(train_index,test_index) in enumerate(splits):
  dataset_train = ImageDataset(image_id_list, label_list, train_index)
  dataset_test = ImageDataset(image_id_list, label_list, test_index)

  train_loader = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)
  test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True)

  model = CNNModel()
  model.apply(reset_weights)

  learning_rate = 0.0001
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  total_step = len(train_loader)

  for epoch in range(num_epochs):
    current_loss = 0.0

    for i, data in enumerate(train_loader):
      x_imgs, labels = data
      optimizer.zero_grad()

      # Forward pass
      output = model(x_imgs["image"])
      loss = criterion(output, labels)

      # Backward and optimize
      loss.backward()
      optimizer.step()

      current_loss += loss.item()
      if (i+1) % 10 == 0:
        print ('Fold # {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(n, epoch+1, num_epochs, i+1, total_step, loss.item()))

  y_true = []
  y_pred = []
  correct, all = 0,0
  with torch.no_grad():
    for i, data in enumerate(test_loader):
      x_imgs, labels = data

      output = model(x_imgs["image"])

      _, predicted  = torch.max(output.data, 1)
      print("LABELS: ", labels)
      print('PREDICTED :', predicted)
      all += labels.size(0)
      correct += (predicted == labels).sum().item()

      y_true.extend(labels.cpu().numpy())
      y_pred.extend(predicted.cpu().numpy())

    # Print accuracy
    accuracy = 100.0 * (correct / all)
    conf_matrix, class_report = calculate_metrics(y_true, y_pred)
    print_results(conf_matrix, class_report, n, accuracy)
    print('Accuracy for fold %d: %d %%' % (n, 100.0 * correct / all))
    print('--------------------------------')
    results[n] = accuracy

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
  print(f'Fold {key}: {value} %')
  sum += value
print(f'Average: {sum/len(results.items())} %')

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

conf_matrix = confusion_matrix(y_true, y_pred)
class_names = ['0', '1', '2', '3', '4']

plt.figure(figsize=(10, 10))
plot_confusion_matrix(conf_matrix, classes=class_names)

print(classification_report(y_true, y_pred, target_names=class_names))
