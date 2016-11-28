import math
class num:
    def __init__(self,x):
        self.x = x

tset = [num(1),num(3),num(4),num(5),num(6),num(7)]
no1 = num(0)
for t in tset:
    if t.x == 5:
        no1 = t

print(no1.x)