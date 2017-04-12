'''
Function cov_martix calculate the covarience matrix of the input.
Input d is a n*k matrix which contains n dimentional set.
for example,
x = [1, -1, 4]
y = [2, 1, 3]
z = [1, 3, -1]
d = [x, y , z]

Function cov_martix is expected to return a n*n matrix which is the coverience
matrix for input d.
'''


def cov_matrix(d):
    n = len(d[0])
    var_mat = []
    for x in d:
        x_bar = sum(x)/len(x)
        x_var = [i - x_bar for i in x]
        var_mat.append(x_var)

    cov_mat = []
    for i, x in enumerate(d):
        cov_x = []
        for j, y in enumerate(d[i:]):
            cov_xy = sum(var_mat[i][t]*var_mat[i+j][t] for t in range(n))/(n-1)
            cov_x.append(cov_xy)
        cov_mat.append(cov_x)
    print(cov_mat)


if __name__ == "__main__":
    d = [[1, -1, 4], [2, 1, 3], [1, 3, -1]]
    cov_matrix(d)
