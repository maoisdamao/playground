# 350. intersection-of-two-arrays-ii [easy]


class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        mappings = {}
        for i in nums1:
            if i in mappings:
                mappings[i] += 1
            else:
                mappings[i] = 1
        LCS = []
        for j in nums2:
            if j in mappings and mappings[j] > 0:
                mappings[j] -= 1
                LCS.append(j)
        return LCS