# An-CNN-model-using-FashionMnist-with-tensorboard
欢迎！
一.如何使用
1.下载代码，以及预训练的模型参数文件model.pkl
2.运行train.py时，修改main函数中代码
train_set=torchvision.datasets.FashionMNIST(
        root='存储fashionMnist数据集的位置', ---①
        train=True,
        download=False, （若您已经下载了fashionMnist，设为False，否则设为True，建议网盘下载）--②
        transform=transforms.Compose([transforms.ToTensor()])
    )
3.运行test.py时，同样需要修改main函数中代码，修改方式同2

二.如何使用tensorboard
1.pip install tensorboard
2.运行train后，今日anaconda prompt，进入当前环境，键入 tensorboard --logdir=runs
3.得到一个网址，复制进入浏览器即可

