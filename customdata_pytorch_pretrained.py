import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import os

#device=torch.device("cuda" if torch.cuda.is_available else "CPU")
#print(f"using device :{device}")
                    

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128)

])



data_dir=data_dir = r"C:\Users\Keerthana A R\Downloads\archive\hymenoptera_data"

image_datasets={}
image_datasets['train']=datasets.ImageFolder(os.path.join
                                             (data_dir,'train'),
                                             transform=transform)


image_datasets['val']=datasets.ImageFolder(os.path.join
                                             (data_dir,'val'),
                                             transform=transform)


train_loader=DataLoader(image_datasets['train'],batch_size=128,shuffle=True)
test_loader=DataLoader(image_datasets['val'],batch_size=128,shuffle=False)

#load pre train model
model=models.resnet18(pretrained=True)  #weights='IMAGENET1K_V1'
print(model)

#freeze all the layers initially
for param in model.parameters():
    param.requires_grad=False

#replace final fully connected layer
num_features=model.fc.in_features
model.fc=nn.Linear(num_features,2)

criterian=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=0.001)

for epoch in range(20):
    for image,label in train_loader:
        image,label=image,label
        output=model(image)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
correct=0.0
total=0.0
model.eval()
with torch.no_grad():
    for image,label in test_loader:
        image,label=image,label
        output=model(image)
        max,predicted=torch.max(output,1)
        correct+= (predicted==label).sum().item()
        total+=label.size(0)
    print(f"test accuracy:{(correct/total)*100}")

#(fc): Linear(in_features=512, out_features=1000, bias=True)




