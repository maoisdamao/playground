# leetcode 217 - Contains Duplicate
# author: @pennsy


class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return len(set(nums)) != len(nums)


solver = Solution()
nums = [1, 2, 3, 1]
print(solver.containsDuplicate(nums))     