import pandas as pd
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler




def marevellousclassifier(Datapath):
    border = "-"*40

    #step1 : load data from csv file

    print(border)
    print("Load Data from csv file")
    print(border)

    df = pd.read_csv(Datapath)

    print("Some entries of file ")
    print(border)
    print(df.head())
    print(border)

    #steps 2: Clean The Dataset

    print(border)
    print("Step2: Clean The Dataset")
    print(border)
    
    df.dropna(inplace=True)
    
    print("Shape of Data Set:", df.shape)
    print("Total Records : ", df.shape[0])
    print("Total Columns : ", df.shape[1])

#step 3: separate independent varibales and train the data

    print(border)
    print("separate independent varibales and train the data")
    print(border)

    X = df.drop(columns=['Class'])
    Y = df["Class"]

    print("Shape Of X is :",X.shape)
    print("Shape of Y is :", Y.shape)

    print(border)
    print(" input Columns are: ", X.columns.to_list())
    print("Output Colums are : Class ")
    print(border)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,test_size=0.2,stratify=Y)

    print(border)
    print("Details of training and testing data")
    print("shape of X_train: ", X_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of Y_train: ", Y_train.shape)
    print("shape of Y_test: ", Y_test.shape)
    print(border)

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train,Y_train)
#step 4: test the data
    Y_pred = model.predict(X_test)

#step 5: accuracy calculation 
    Accuracy = accuracy_score(Y_pred, Y_test)
    print("Accuracy is :", Accuracy*100)

def main():
    marevellousclassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()
