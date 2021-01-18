import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from RunBuilder import RunBuilder
from cnnModel import CNN
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

#hyperParameters
num_epochs=10
#TestParameters
params=OrderedDict(
learning_rate=[0.01]
,batch_size=[100]
,shuffle=[True]
,device=['cuda']
)


def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def main():
   
    train_set=torchvision.datasets.FashionMNIST(
        root='E:\LearningStuff\DLcode\data',
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    print("Initalizing Network")
    for run in RunBuilder.get_runs(params):
        comment=f'-{run}'
        device=torch.device(run.device)
        cnn=CNN().to(device)
        train_loader=torch.utils.data.DataLoader(train_set,batch_size=run.batch_size,shuffle=True,num_workers=0)
        optimizer=optim.Adam(cnn.parameters(),lr=run.learning_rate)

        ''' Initializing tensorboard '''
        tb=SummaryWriter(comment=comment,flush_secs=1)
        images,labels=next(iter(train_loader))
        grid=torchvision.utils.make_grid(images)
        tb.add_image('images',grid)
        tb.add_graph(cnn,images.to(getattr(run,'device','cpu')))

        '''begin to train'''
        for epoch in range(num_epochs):
            total_loss=0
            total_correct=0
            for batch in train_loader:
                images=batch[0].to(device)
                labels=batch[1].to(device)
                preds=cnn(images)
                loss=F.cross_entropy(preds,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss+=loss.item()
                total_correct+=get_num_correct(preds,labels)
            tb.add_scalar('Loss',total_loss,epoch)
            tb.add_scalar('Number Correct',total_correct,epoch)
            tb.add_scalar('Accuracy',total_correct/len(train_set),epoch)
            print("epoch",epoch,"loss",total_loss,"Accuracy",total_correct/len(train_set))
    tb.close()
    torch.save(cnn.state_dict(),".\model.pkl")
    print("model saved")

if __name__=='__main__':
    main()