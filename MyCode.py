#pip install scikit-learn
#pip install pandas
#pip install numpy

import numpy as np
import pandas as pd

def loadCSV():
    #Load the file
    return pd.read_csv("fish.csv")

def getFishTypes(df):
    return df["Species"].unique()
    
def task1():
    data = loadCSV()

    for x in getFishTypes(data):
        print(x)

    
task1()