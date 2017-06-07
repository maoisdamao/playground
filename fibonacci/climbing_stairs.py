# This code is for leetcode #70 climbling stairs. According to the problem's
# discussion, it is a fibonacci problem, since,
# climb n steps = climb(n-1) + 1 more step, or
# climb n steps = climb(n-2) + 2 more step
# So, ways of climb n steps = ways(n-1) + ways(n-2).
# ref https://discuss.leetcode.com/topic/5371/basically-it-s-a-fibonacci


def fibonacci(n):
    arr = [0, 1, 2]
    for i in range(3, n+1):
        arr.append(arr[i-1]+arr[i-2])
    return arr[n]


if __name__ == "__main__":
    print(fibonacci(100))
