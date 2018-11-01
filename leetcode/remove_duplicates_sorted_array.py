# leetcode 26: Remove Duplicates from Sorted Array [easy]
# author: @pennsy


def removeDuplicates(nums):
    if len(nums) <= 1:
        return len(nums)
    i = 0
    for j in range(1, len(nums)):
        if nums[j] > nums[i]:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    return (i + 1)

def removeDuplicates2(nums):
    if len(nums) <= 1:
        return len(nums)
    i = 1
    while i < len(nums):
        if nums[i-1] == nums[i]:
            del nums[i]
        else:
            i += 1
    return i

nums = [1 ,1 ,2]
answer = removeDuplicates2(nums)
print(answer)