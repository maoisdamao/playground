# leetcode 189: rotate array
# author: @pennsy


class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for i in range(k):
            nums.insert(0, nums[-1])
            nums.pop()
        return nums
    
    def rotate2(self, nums, k):
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[:-k]
        return nums



s = Solution()
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
expected = [5,6,7,1,2,3,4]
print (s.rotate(nums, k) == expected)
nums = [1, 2]
k = 3
expected = [2, 1]
print (s.rotate2(nums, k) == expected) 