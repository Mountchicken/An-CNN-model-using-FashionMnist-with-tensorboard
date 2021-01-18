import torch
import torchvision
import torchvision.transforms as transforms
from cnnModel import CNN

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def main():
    test_set=torchvision.datasets.FashionMNIST(
        root='E:\LearningStuff\DLcode\data',
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    cnn=CNN()
    ''' 防止权重改变 '''
    cnn.eval() 
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=100,shuffle=True,num_workers=0)
    total_correct=0
    for batchs in test_loader:
        images,labels=batchs
        preds=cnn(images)
        total_correct+=get_num_correct(preds,labels)
    print('The Accuracy of the model on test-set is:',total_correct/len(test_set))
    print(len(test_set))
if __name__=='__main__':
    main()