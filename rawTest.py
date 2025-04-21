import numpy as np
import pandas as pd

def softmax(z):
    z = np.clip(z, -100, 100)
    z = z - np.max(z)
    exps = np.exp(z)
    return exps / np.sum(exps)

def leaky_relu(x):
    for  i, n in enumerate(x):
        for k,_ in enumerate(n):
            x[i,k] = np.maximum(0.01 * x[i,k], x[i,k])
            
    return x

def row(m):
    m = m.reshape(-1,1)
    return m

W1 = np.array(pd.read_csv("CSVs\\W1.csv", header=None))
W2 = np.array(pd.read_csv("CSVs\\W2.csv", header=None))
W3 = np.array(pd.read_csv("CSVs\\W3.csv", header=None))
b1 = np.array(pd.read_csv("CSVs\\b1.csv", header=None))
b2 = np.array(pd.read_csv("CSVs\\b2.csv", header=None))
b3 = np.array(pd.read_csv("CSVs\\b3.csv", header=None))

def ForwardPass(neuralInput):
    neuralInput = row(neuralInput)
    z1 = W1 @ neuralInput + b1
    a1 = leaky_relu(z1)
    z2 = W2 @ a1 + b2
    a2 = leaky_relu(z2)
    neuralOutput = softmax(W3 @ a2 + b3)
    return [neuralOutput, z2, z1]

train_df = pd.read_csv("CSVs\\mnist_train.csv").sample(frac=1).reset_index(drop=True)
test_df = pd.read_csv("CSVs\\mnist_test.csv").sample(frac=1).reset_index(drop=True)
train_img = np.array(train_df.iloc[:,1:]) / 255
train_label = np.array(train_df.iloc[:,0])
test_img = np.array(test_df.iloc[:,1:]) / 255
test_label = np.array(test_df.iloc[:,0])

accurate = 0
for img,label in zip(train_img,train_label):
    prediction = np.argmax(ForwardPass(row(img))[0])
    if prediction == label:
        accurate+=1
for img,label in zip(test_img,test_label):
    prediction = np.argmax(ForwardPass(row(img))[0])
    if prediction == label:
        accurate+=1
        
print((accurate * 100 / 70000))