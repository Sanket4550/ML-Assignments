import numpy as np
import math
def EucDistanace(P1, P2):

   ans = math.sqrt(((P1['X']-P2['X'])**2) + ((P1['Y']-P2['Y'])**2))
   return ans


def predict(Data, new_point, K):
    for d in Data:
        d['Distance'] = EucDistanace(d,new_point)


    sorted_data = sorted(Data, key = lambda item : item['Distance'])

    nearest = sorted_data[:K]
    votes = {}
    
    for neighbours in nearest:
        Label = neighbours['Label']
        votes[Label] = votes.get(Label,0) + 1

    iMax = 0
    Name= ""
    for d in votes:
        if (votes[d] > iMax ):
            iMax = votes[d]
            Name = d
        return Name


        
def KNNclassifier():
    border = "-"*40 
    Data = [
        {'point': 'A','X':1, 'Y':2, 'Label': 'Red' },
        {'point': 'B','X':2, 'Y':3, 'Label': 'Red' },
        {'point': 'C','X':3, 'Y':1, 'Label': 'Blue' },
        {'point': 'D','X':6, 'Y':5, 'Label': 'Blue' },
    ]

    print(border)
    print("KNN Classsifier:")
    print(border)

    for i in Data:
        print(i)

    X1 = int(input("Enter X coordinate:"))
    Y1 = int(input("Enter Y Coordinate: "))

    new_point = {'X' : X1, 'Y' : Y1}

    print("Prediction Results\n")
    for K in [1, 3, 5]:
        result = predict(Data, new_point, K)
        print(f"final prediction result is for K = {K}: {result}")

def main():
    KNNclassifier()

if __name__ == "__main__":
    main()