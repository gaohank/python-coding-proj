import os

f = open('test.txt', 'r')
lines = f.readlines()
prex = 'path/'
imgs = []
# print(lines)
for line in lines:
    item = line.strip('\n')
    imgs.append(prex + item)

print(imgs)

path = '/home/amax/hank/1.txt'
print(path[path.rfind('/'):].strip('/'))

# if not os.path.isdir('hello'):
#     os.mkdir('hello')
