import sys


class Solution:
    def sumCal(self, num, times):
        self.result = 0
        self.number = num
        self.count = 10
        self.i = 0
        self.flag = 0
        if num == 0:
            print(self.result)
            return 1
        if num < 0:
            num = -1 * num
            self.flag = 1
            self.number = num
        while self.i < times:
            self.result += num
            num = num * self.count + self.number
            self.i += 1
        if self.flag == 1:
            print(-1 * self.result)
        else:
            print(self.result)

s = Solution()
s.sumCal(-2,3)