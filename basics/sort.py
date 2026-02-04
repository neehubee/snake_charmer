a=[9,1,3,5,6]
a.sort()
print(a)
a.sort(reverse=True)
print(a)
#sort changes the ds

arr = [(2,3),(1,10),(5,1)]
arr.sort(key=lambda x:x[0],reverse=False)
print(arr)

arr2 = [(2,3),(1,10),(5,1)]
arr2.sort(key=lambda x:x[1],reverse=False)
print(arr2)