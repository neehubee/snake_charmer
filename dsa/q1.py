
arr = list(map(int, input().split()))
#arr = [1,2,2,3,1,4,3]
#Remove duplicates but keep order
seen =set()
result=[]
for x in arr:
    if x not in seen:
        seen.add(x)
        result.append(x)
print(result)    