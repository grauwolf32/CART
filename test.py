import numpy as np
import random

def add_max_n(array,values,i,value,n):

    if values[n-1] >= value:
        return

    values.pop()
    array.pop()

    values.append(value)
    array.append(i)

    j = n-1
    temp = 0
    while j > 0:
        if values[j-1] <= values[j]:
            temp = values[j]
            values[j] = values[j-1]
            values[j-1] = temp

            temp = array[j]
            array[j] = array[j-1]
            array[j-1] = temp
            j = j-1
        else :
            break
    

def main():
    n = 10
    A = list(np.zeros(n))
    B = list(np.zeros(n))
    C = [random.randint(0,100) for x in xrange(0,3*n)]
    for i in xrange(0,len(C)-1) :
        add_max_n(A,B,i,C[i],n)
        print B
        for i in xrange(0,len(A)):
            print C[int(A[i])]


if __name__ == "__main__":
    main()  