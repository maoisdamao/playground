class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cursum = nums[0]
        maxsum = nums[0]
        for i in nums[1:]:
            cursum += i
            if i > cursum:
                cursum = i
            if maxsum < cursum:
                maxsum = cursum
        return maxsum