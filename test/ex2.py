import sys
class Solution:
    def printIDCard(self , str):
        self.area = [11,12,13,14,15]
        self.Wi = [7,6,10,5,8]
        self.Y = [1,0,"X",9,8,7,6,5,4,3,2]
        self.Y_P = 0
        self.cleck = 0
        self.sum = 0
        self.i =0
        if str[0:1] in self.area:
            for num in str[:16]:
                sum += str[self.i]*self.Wi[self.i]
            self.Y_P == divmod(sum, 11)
            if str[17] == self.Y[self.Y_P]:
                return True, str[0:17]
        else:
            if str.len() > 18:
                self.printIDCard(str[1:18])
            else:
                return False

