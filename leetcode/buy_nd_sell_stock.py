# leetcode 122: Best Time to Buy and Sell Stock II [easy]
# author: @pennsy


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        for i in range(len(prices)-1):
            profit += max(prices[i+1]-prices[i], 0)
        return profit