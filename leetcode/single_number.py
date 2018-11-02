class Solution(object):
    def singleNumber(self, nums):
        unique = set()
        for num in nums:
            if num in unique:
                unique.discard(num)
            else:
                unique.add(num)
        return unique.pop()

    def singleNumber2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 2*(a+b+c)-(a+a+b+b+c)=c
        return 2*sum(set(nums))-sum(nums)

    def singleNumber3(self, nums):
        # hash table need extra memory
        dic = {}
        for a in nums:
            if a in dic:
                dic[a] += 1
            else:
                dic[a] = 1
        for k, v in dic.items():
            if v == 1:
                return k
    
 