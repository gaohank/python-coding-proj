
def func1(*args):
    print(args)
    print(type(args))


def func2(**args):
    print(args)


func1(1, 2, 3, 4)
func2(name="hank", age=30)

nums = [12, 23, 34, 56]
func1(*nums)
func1(nums)
