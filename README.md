# isic2018
神经网络课程设计实验代码，基于isic2018数据集

### File List
```
+---data
|   +---ISIC2018
|   |   +---ISIC2018_Task3_Test_Input
|   |   +---ISIC2018_Task3_Training_Input
|   |   +---ISIC2018_Task3_Validation_Input
|   |   +---ISIC2018_Task3_Training_GroundTruth.csv
|   |   +---ISIC2018_Task3_Validation_GroundTruth.csv
|   |   +---Test_GroundTruth.csv           # 从原训练数据拆分出来的测试数据集
|   |   \---Train_GroundTruth.csv          # 从原训练数据拆分出来的训练数据集
|   \---split_data.py           # 拆分测试集和训练集的脚本
+---net
|   +---__init__.py
|   +---arl.py              # arlnet模型
|   +---resnet.py           # resnet模型
|   \---focal_loss.py       # focal loss 损失函数
+---utils
|   +---__init__.py
|   +---data.py             # 数据集
|   +---logger.py           # 日志
|   \---model.py            # 模型和参数保存及加载
\---demo.py
```

### Dataset
[ISIC2018 Data](https://challenge.isic-archive.com/data/)
> 下载 ISIC2018 Task3 对应的各个数据集，解压并把对应的文件放入文件夹`/data/ISIC2018`下。<br/>
> 由于该数据集的测试集没有 Ground Truth，因此从训练集分出1/4作为测试集，请运行 `/data/split_data.py` 脚本拆分数据，
>运行后会生成 `Test_GroundTruth.csv` 和 `Train_GroundTruth.csv` 两个文件，分别代表拆分后测试机和训练集的 Ground Truth。
>(可以通过修改脚本改变拆分的比例)
```shell script
# download training images
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
# download training ground truth
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
# download validation images
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip
# download validation ground truth
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip
# download test images
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip
```

### 实验设置
> 基于 ISIC2018 数据集，主要在**过拟合**、**类别倾斜**、**脏标签**这三个主题下设计实验。<br/>
> 每个组员独立设计实验并进行验证，所有的实验代码须整理后放入对应的文件夹下。
1. 过拟合
    - ex1
2. 类别倾斜
    - ex2
    - ex3
3. 脏标签
    - ex4
    - ex5

### Tips
```shell script
# check GPU info
nvidia-smi
# clone project from github
git clone https://github.com/chenjiaqiang-a/isic2018.git
# convert .ipynb to .py
jupyter nbconvert --to script <.ipynb file name>
# run in backend
nphup python <.py file name> &
```