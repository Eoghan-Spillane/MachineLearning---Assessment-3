#pip install scikit-learn
#pip install pandas
#pip install numpy

#Author: Eoghan Spillane

import numpy as np
import pandas as pd

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
    

    

task2()