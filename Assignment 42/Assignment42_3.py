import numpy as np
import math


def EucDistance( P1, P2):
    ans = math.sqrt(((P1['Study Hours']-P2['Study Hours'])**2) + ((P1['Attendance']-P2['Attendance'])**2))
    return ans
    

    
def KNNclassifier():

    border = "-"*40

    Data = [
        {'Study Hours': 2, 'Attendance':60, 'Result': 'Fail'},
        {'Study Hours': 5, 'Attendance':80, 'Result': 'Pass'},
        {'Study Hours': 6, 'Attendance':85, 'Result': 'Pass'},
        {'Study Hours': 1, 'Attendance':50, 'Result': 'Fail'}
    ]
    print(border)
    print("KNN Classifier: ")
    for i in Data:
        print(i)


    StudyHours_i = int(input("Enter the Study Hours: "))
    Attendance_i = int(input("Enter the Attendance: "))

    New_Entry = {'Attendance' : Attendance_i, 'Study Hours' : Attendance_i}

    for d in Data:
            d['Distance'] = EucDistance(d,New_Entry)

    sorted_data = sorted(Data, key = lambda item : item['Distance'])

    K = 3
    nearest = sorted_data[:K]
    
    print(border)
    print(" Nearest Sorted Data: ")
    print(border)
    for d in sorted_data:
        print(d)

    #voting
    votes = {}
    for neighbours in nearest:
        Result = neighbours['Result']
        votes[Result] = votes.get(Result,0) + 1

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