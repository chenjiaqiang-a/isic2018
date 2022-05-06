import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("./ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv")
classes = list(df.columns.values)[1:]

arr = np.array(df[classes])
arr = arr.argmax(axis=1)
df["CLASS"] = arr

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.to_csv("./ISIC2018/Train_GroundTruth.csv", index=False)
x_test.to_csv("./ISIC2018/Test_GroundTruth.csv", index=False)
