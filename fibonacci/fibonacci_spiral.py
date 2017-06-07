
# time complexity:O(n) space complexity:O(n)
def fibonacci(n):
    arr = [0, 1]
    for i in range(2, n+1):
        arr.append(arr[i-1]+arr[i-2])
    return arr[n]


if __name__ == "__main__":
    print(fibonacci(100))  # 354224848179261915075
