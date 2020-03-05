
def func1(*args):
    print(args)


def func2(**args):
    print(args)


func1(1, 2, 3, 4)
func2(name="hank", age=30)
