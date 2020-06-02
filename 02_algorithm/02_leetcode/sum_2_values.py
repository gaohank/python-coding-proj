"""
给出两个非空的链表用来表示两个非负的整数。
其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/add-two-numbers
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        n = l1
        i = 1
        num_l1 = 0
        # get num of l1
        while n:
            num_l1 = num_l1 + n.val * i
            i = i * 10
            n = n.next

        m = l2
        j = 1
        num_l2 = 0
        # get num of l2
        while m:
            num_l2 = num_l2 + m.val * j
            j = j * 10
            m = m.next

        str_num = str(num_l1 + num_l2)
        str_num = str_num[::-1]

        res = list_result = ListNode(0)
        for s in str_num:
            list_result.next = ListNode(int(s))
            list_result = list_result.next
        return res.next


if __name__ == '__main__':
    s = Solution()
    l1 = ListNode(3)
    l1_2 = ListNode(4)
    l1_3 = ListNode(2)
    l1_2.next = l1_3
    l1.next = l1_2

    l2 = ListNode(4)
    l2_1 = ListNode(6)
    l2_2 = ListNode(4)
    l2_1.next = l2_2
    l2.next = l2_1
    l3 = s.addTwoNumbers(l1, l2)
    print(l3.val)
    print(l3.next.val)
    print(l3.next.next.val)
