#input pairs in map if complement is target then they are

n = int(input())
arr = list(map(int, input().split()))
target = int(input())

s=set()
for x in arr:
    if target-x in s:
      print("YES")
      break
    s.add(x)
else:
   print("no")