import numpy as np
import math
def EucDistanace(P1, P2):

   ans = math.sqrt(((P1['X']-P2['X'])**2) + ((P1['Y']-P2['Y'])**2))
   return ans

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
    print(border)
    print("Euclidean Distance from all points")
    print(border)
    #calculating distance from all point 
    for d in Data:
        d['Distance'] = EucDistanace(d,new_point)
    #display the calculted dsitance
    for d in Data:
        print(d)


    #sortiing the array by assending order of distance
    sorted_data = sorted(Data, key = lambda item : item['Distance'])

    print(border)
    print("Sorted_data : ")
    print(border)


    #printing sorted data
    for d in sorted_data:
        print(d)

    print(border)

    K = 3
    # as per k value the nearest keeps variables here we are k = 3
    nearest = sorted_data[:K]

    print(border)
    print("Nearest 3 members are: ")
    print(border)

    #printing nearest values 
    for d in nearest:
        print(d)


    #voting
    votes = {}
    for neighbours in nearest:
        Label = neighbours['Label']
        votes[Label] = votes.get(Label,0) + 1

    print(border)


    iMax = 0
    Name= ""
    for d in votes:
        if (votes[d] > iMax ):
            iMax = votes[d]
            Name = d

    print("Final Prediction is : ", Name)
    print(border)

def main():
    KNNclassifier()

if __name__ == "__main__":
    main()