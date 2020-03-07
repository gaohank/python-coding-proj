class Person:
    age = 10
    name = "hank"


setattr(Person, 'sex', 'man')

print(hasattr(Person, 'sex'))


for k, v in Person.__dict__.items():
    if not k.startswith('__'):
        print(k, v)
