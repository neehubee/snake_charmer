mp={}
mappy={}
a=list(range(5))
for x in a:
      mp[x]=mp.get(x,0)+1
print(mp)
arr=[1,2,3,4,4]
for x in arr:
      mappy[x]=mappy.get(x,0)+1
print(mappy)
for key,freq in mappy.items():
      print(key,freq)
print('*')
for value in mappy.values():
      print(value)
print('*')
for key in mappy.keys():
      print(key)   
