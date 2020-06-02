# 先定一个node的类
class Node:  # value + next
    def __init__(self, value=None, next_node=None):
        self._value = value
        self._next = next_node

    def getValue(self):
        return self._value

    def getNext(self):
        return self._next

    def setValue(self, new_value):
        self._value = new_value

    def setNext(self, new_next):
        self._next = new_next


# 实现Linked List及其各类操作方法
class LinkedList:
    def __init__(self):  # 初始化链表为空表
        self._head = None
        self._tail = None
        self._length = 0

    # 检测是否为空
    def isEmpty(self):
        return self._head is None

    # add在链表前端添加元素:O(1)
    def add(self, value):
        new_node = Node(value)  # create一个node（为了插进一个链表）
        new_node.setNext(self._head)
        self._head = new_node

    # append在链表尾部添加元素:O(n)
    def append(self, value):
        new_node = Node(value)
        if self.isEmpty():
            self._head = new_node  # 若为空表，将添加的元素设为第一个元素
        else:
            current = self._head
            while current.getNext() is not None:
                current = current.getNext()  # 遍历链表
            current.setNext(new_node)  # 此时current为链表最后的元素

    # search检索元素是否在链表中
    def search(self, value):
        current = self._head
        found_value = False
        while current is not None and not found_value:
            if current.getValue() == value:
                found_value = True
            else:
                current = current.getNext()
        return found_value

    # index索引元素在链表中的位置
    def index(self, value):
        current = self._head
        count = 0
        found = None
        while current is not None and not found:
            count += 1
            if current.getValue() == value:
                found = True
            else:
                current = current.getNext()
        if found:
            return count
        else:
            raise ValueError('%s is not in linkedlist' % value)

    # remove删除链表中的某项元素
    def remove(self, value):
        current = self._head
        pre = None
        while current is not None:
            if current.getValue() == value:
                if not pre:
                    self._head = current.getNext()
                else:
                    pre.setNext(current.getNext())
                break
            else:
                pre = current
                current = current.getNext()

    # insert链表中插入元素
    def insert(self, pos, value):
        if pos <= 1:
            self.add(value)
        elif pos > self.size():
            self.append(value)
        else:
            temp = Node(value)
            count = 1
            pre = None
            current = self._head
            while count < pos:
                count += 1
                pre = current
                current = current.getNext()
            pre.setNext(temp)
            temp.setNext(current)

    def size(self):
        return self._length
