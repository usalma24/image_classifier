import torch
import torchvision
import argparse
import json
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import PIL
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from os.path import isdir

def arg_parser():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset.")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--save_dir", dest="save_dir", action="store",        default="./checkpoint.pth")
    parser.add_argument("--arch", dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument("--learning_rate", dest="learning_rate", action="store", type=float, default=0.001, help="Set learning rate")
    parser.add_argument("--hidden_units", type=int, dest="hidden_units", action="store", default=120, help="Set hidden units")
    parser.add_argument("--epochs", type=int, dest="epochs", action="store", default=1, help="Set epochs")
    parser.add_argument("--gpu", action="store", dest="gpu", help="Use GPU for training")
    
    args = parser.parse_args()
    
    return args

def train__transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229, 0.224,0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        
    return train_data                                      
                                       
            
                                
def test_transformer(test_dir):                                                        

    test_transforms =  transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229, 0.224, 0.225])
                                                            ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data


def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
        
    return loader


def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if device == "cpu":
        print("CUDA was not found on device, use CPU instead.")
    return device


def primaryloader_model(architecture="vgg16"):
    
    
       model = models.vgg16(pretrained=True)
       model.name = "vgg16"
       for param in model.parameters():
           param.requires_grad = False
       return model

def initial_classifier(model, hidden_units):
    
    from collections import OrderedDict
    
    
    classifier = nn.Sequential(OrderedDict([
                 ('inputs', nn.Linear(25088, 120)),
                 ('relu1', nn.ReLU()),
                 ('Dropout', nn.Dropout(0.5)),
                 ('hidden_layer1', nn.Linear(120,90)),
                 ('relu2', nn.ReLU()),
                 ('hidden_layer2', nn.Linear(90,70)),
                 ('relu3', nn.ReLU()),
                 ('hidden_layer3', nn.Linear(70,102)),
                 ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    return classifier

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps =torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def network_trainer(Model, Trainloader, Testloader, Device, Criterion, Optimizer, Epochs, Print_every, Steps):
    if type(Epochs) == type(None):
        Epochs = 12
        print("Number of epochs specified as 12")
        
        print("Training process initializing...")
        
        
        for e in range(Epochs):
            running_loss = 0
            Model.train()
            
            
            for ii, (inputs, labels) in enumerate(TrainLoader):
                Steps += 1
                inputs, labels = inputs.to(Device), labels.to(Device)
                
                Optimizer.zero_grad()
                
                
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    model.eval()
                    
                    
                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, validloader, criterion)
                        
                    print("Epoch: {}/{} | ".format(e+1, epochs),
                          "Training Loss: {:.4f} | ".format(running_loss/print_every),
                          "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                          "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
                          
                    running_loss = 0
                    model.train()
            return Model
def validate_model(Model, Testloader, Device):
    correct, total= 0, 0
    with torch.no_grad():
        model.eval()
        for data in train_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test images is: %d%%" % (100 * correct/total))
    
    
def initial_checkpoint(Model, Save_Dir, Train_data):
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified , model will not be savedd")
    else:
        if isdir(Save_Dir):
            model.class_to_idx = image_datasets['train'].class_to_idx
            torch.save({'structure': 'alexnet',
            'hidden_layer1': 120,
            'dropout': 0.5,
            'epochs': 12, 
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optimizer_dict': optimizer.state_dict()},
            'checkpont.pth')
            Model.class_to_idx = Train_data.class_to_idx
            
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            torch.save(checkpoint, 'my_checkpoint.pth')
        else:
            print("directory not found, model will not be saved")
def main():
    
    args = arg_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model = primaryloader_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process is completed..")
    
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data)
    
    
if __name__ == '__main__': main()    
            
            
            
        
          
    
                          
                          
    
        

    
    
                          
