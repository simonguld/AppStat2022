




def f():
    print("hej")

def g():
    print("hello world")

F, G = True, False
names = [F,G]
f_list = [f, g]

for i, func in enumerate (f_list):
    if names[i]:
        func()