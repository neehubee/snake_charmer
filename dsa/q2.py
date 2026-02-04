arr = list(map(int, input().split()))
mp={}
for x in arr:
    mp[x]=mp.get(x,0)+1
m=0
k=float('inf')
for key, value in mp.items():
    if value>m or(value==m and key<k):
      m=value
      k=key
print(k)    
