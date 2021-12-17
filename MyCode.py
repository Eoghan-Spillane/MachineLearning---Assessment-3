#pip install scikit-learn
#pip install pandas
#pip install numpy

#Author: Eoghan Spillane

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

def loadCSV():
    #Loads the file
    return pd.read_csv("fish.csv")

def getFishTypes(df):
    return df["Species"].unique()
    
def task1():
    data = loadCSV()


    #Extract Differnt Types of Fish
    print("Types of Fish: ")
    # for x in getFishTypes(data):
    #     print("\t*", x)

    #Extract Corrosponding Features and Targets
    Bream = data[data["Species"] == 'Bream']
    Roach = data[data["Species"] == 'Roach']
    Whitefish = data[data["Species"] == 'Whitefish']
    Parkki = data[data["Species"] == 'Parkki']
    Perch = data[data["Species"] == 'Perch']
    Pike = data[data["Species"] == 'Pike']
    Smelt = data[data["Species"] == 'Smelt']

    BreamLabel = Bream["Species"]
    RoachLabel = Roach["Species"]
    WhitefishLabel = Whitefish["Species"]
    ParkkiLabel = Parkki["Species"]
    PerchLabel = Perch["Species"]
    PikeLabel = Pike["Species"]
    SmeltLabel = Smelt["Species"]


    Bream = Bream.loc[:, Bream.columns!='Species']
    Roach = Roach.loc[:, Roach.columns!='Species']
    Whitefish = Whitefish.loc[:, Whitefish.columns!='Species']
    Parkki = Parkki.loc[:, Pike.columns!='Species']
    Perch = Perch.loc[:, Perch.columns!='Species']
    Pike = Pike.loc[:, Pike.columns!='Species']
    Smelt = Smelt.loc[:, Smelt.columns!='Species']


    BreamFeatures = Bream.loc[:, Bream.columns!='Weight']
    RoachFeatures = Roach.loc[:, Roach.columns!='Weight']
    WhitefishFeatures = Whitefish.loc[:, Roach.columns!='Weight']
    ParkkiFeatures = Parkki.loc[:, Parkki.columns!='Weight']
    PerchFeatures = Perch.loc[:, Perch.columns!='Weight']
    PikeFeatures = Pike.loc[:, Pike.columns!='Weight']
    SmeltFeatures = Smelt.loc[:, Smelt.columns!='Weight']

    BreamTarget = Bream["Weight"]
    RoachTarget = Roach["Weight"]
    WhitefishTarget = Whitefish["Weight"]
    ParkkiTarget= Parkki["Weight"]
    PerchTarget= Perch["Weight"]
    PikeTarget = Pike["Weight"]
    SmeltTarget = Smelt["Weight"]

    
    FishData = {"Bream" : Bream, "Roach" : Roach, "Whitefish" : Whitefish, "Parkki" : Parkki, "Perch" : Perch, "Pike" : Pike, "Smelt" : Smelt}
    FishDataFeatures = {"Bream" : BreamFeatures, "Roach" : RoachFeatures, "Whitefish" : WhitefishFeatures, "Parkki" : ParkkiFeatures, "Perch" : PerchFeatures, "Pike" : PikeFeatures, "Smelt" : SmeltFeatures}
    FishDataTarget = {"Bream" : BreamTarget, "Roach" : RoachTarget, "Whitefish" : WhitefishTarget, "Parkki" : ParkkiTarget, "Perch" : PerchTarget, "Pike" : PikeTarget, "Smelt" : SmeltTarget}
    FishLabel = {"Bream" : BreamLabel, "Roach" : RoachLabel, "Whitefish" : WhitefishLabel, "Parkki" : ParkkiLabel, "Perch" : PerchLabel, "Pike" : PikeLabel, "Smelt" : SmeltLabel}
    
    #Select Those for Further Processing with more than 20
    for x in FishData:
        print("\t", x + ":", len(FishData[x]), "Samples")
    
    return BreamFeatures, BreamTarget, BreamLabel, PerchFeatures, PerchTarget, PerchLabel
    
    
def task2():
    BreamFeatures, BreamTarget, BreamLabel, PerchFeatures, PerchTarget, PerchLabel = task1()
    
    deg = 2
    x = np.zeros(num_coefficients(deg))
    print(poly(2, x, BreamFeatures))

def task3():
    BreamFeatures, BreamTarget, BreamLabel, PerchFeatures, PerchTarget, PerchLabel = task1()
    
    deg = 2
    x = np.zeros(num_coefficients(deg))
    f0,J = linearize(deg, x, BreamFeatures)

    print("\nModel Function: \n", f0)
    print("Jacobian: \n ", J)

def task4():
    BreamFeatures, BreamTarget, BreamLabel, PerchFeatures, PerchTarget, PerchLabel = task1()
    
    deg = 2
    x = np.zeros(num_coefficients(deg))
    f0, J = linearize(deg, x, BreamFeatures)

    dp = calculate_update(BreamTarget, f0, J)
    print(dp)


def linearize(deg,p0, data):
    f0 = poly(deg,p0,data)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = poly(deg,p0,data)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
    return f0,J

def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp

def num_coefficients(d):
    t = 0

    for loop1 in range(d + 1):
        for loop2 in range(loop1 + 1):
            for loop3 in range(loop1 + 1):
                for loop4 in range(loop1 + 1):
                    for loop5 in range(loop1 + 1):
                        for loop6 in range(loop1 + 1):
                            if loop2 + loop3 + loop4 + loop5 + loop6 == loop1:
                                t = t+1

    return t

def poly(Degree, ParameterVector, array):
    Polynomials = np.zeros(array.shape[0])    
    t = 0

    for loop1 in range(Degree + 1):
        for loop2 in range(loop1 + 1):  #Length1
            for loop3 in range(loop1 + 1):  #Length2
                for loop4 in range(loop1 + 1):  #Length3
                    for loop5 in range(loop1 + 1):  #Height
                        for loop6 in range(loop1 + 1):  #Weight

                            if loop2 + loop3 + loop4 + loop5 + loop6 == loop1:
                                Polynomials += ParameterVector[t]*(array["Length1"].values**loop2)*(array["Length2"].values**loop3)*(array["Length3"].values**loop4)*(array["Height"].values**loop5)*(array["Width"].values**loop6)
                                t = t+1

    return Polynomials
    
task4()