import random

a = [1,2,3,4]
b = [1,2,3]
def aa(a,x):
    a.remove(x)
map(lambda x:aa(a,x),b)
aa(a,1)
print(a)
