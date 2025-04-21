import numpy as np
import pandas as pd
import cv2 as cv

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

imagePath = "testImgs\\PP.png" # Enter image path before running

inputImg = cv.imread(imagePath)
inputImg = row(np.ravel(cv.cvtColor(inputImg, cv.COLOR_BGR2GRAY))) / 255
prediction = ForwardPass(inputImg)[0]
for i, n in enumerate(prediction):
    print(f"{i} has a probablity : {round(n[0] * 100, 3)}%")
print("Final Prediction : ",np.argmax(prediction))