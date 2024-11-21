# 图像分类作业

## 环境

- [x] Pytorch
- [x] food11数据

> 尽量在Ubuntu下运行

> 程序在Windows下运行, 也可以在Ubuntu下，注意修改代码中的路径分隔符。

> 数据在 https://www.kaggle.com/competitions/ml2022spring-hw3b/data 上下载

## 代码

```py
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

class FoodDataset(Dataset):
    ...
```

处理图片数据的代码块

```py
class Classifier(nn.Module):
    ...
```

分类器定义，可以设置网络大小和结构

```py
def Training_Demo():
    ...
```

训练函数，定义各种参数并且训练模型。

```py
def Testing_Demo():
    ...
```

测试函数，测试模型的准确率

```py
def Predict_Demo():
    ...
```

使用模型进行预测分类

## 效果

| | 准确率 |
| --- | --- |
| 训练集 | 0.63411 |
| 测试集 | 0.56291 |
