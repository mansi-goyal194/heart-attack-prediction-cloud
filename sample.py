n = int(input())
d = dict(input().split("-") for _ in range(n))
print(d)
k=0
for i in range(n):
    k=k+1
    y=i+1
    x=int(len(d[i])) +int(len(d[y]))
    if x>30 :
        print("Block"+k+": "+d[i])
    elif(x==30):
        print("Block"+k+": "+d[i]+", "+d[y])
