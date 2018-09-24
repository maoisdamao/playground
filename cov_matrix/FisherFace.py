import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import linalg


def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1, col))

    for i in range(col):
        r[0, i] = linalg.norm(x[:, i])
    return r


def myLDA(A, Labels):
    # function [W,C,L]=myLDA(A,Labels)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a vector
    # Labels: vector of class labels corresponding to each column in A
    # W: D by K LDA projection matrix
    # C: centers of each class (ie, the templates)
    # L: class labels

    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim, datanum = A.shape
    totalMean = np.mean(A, 1)
    partition = [np.where(Labels == label)[0] for label in classLabels]
    classMean = [(np.mean(A[:, idx], 1), len(idx)) for idx in partition]

    # compute the within-class scatter matrix
    W = np.zeros((dim, dim))
    for idx in partition:
        W += np.cov(A[:, idx], rowvar=1)*len(idx)

    # compute the between-class scatter matrix
    B = np.zeros((dim, dim))
    for mu, class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset, offset)*class_size

    # solve the generalized eigenvalue problem for discriminant directions
    import scipy.linalg as linalg
    import operator
    ew, ev = linalg.eig(B, W+B)
    sorted_pairs = sorted(
        enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind, val in sorted_pairs[:classNum-1]]
    LDAW = ev[:, selected_ind]
    Centers = [np.dot(mu, LDAW) for mu, class_size in classMean]
    Centers = np.transpose(np.array(Centers))
    return LDAW, Centers, classLabels


def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing
    # order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word

    # compute mean, and subtract mean from every column
    [r, c] = A.shape
    m = np.mean(A, 1)
    A = A - np.transpose(np.tile(m, (c, 1)))
    B = np.dot(np.transpose(A), A)
    [d, v] = linalg.eig(B)
    # v is in descending sorted order

    # compute eigenvectors of scatter matrix
    W = np.dot(A, v)
    Wnorm = ComputeNorm(W)
    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1

    LL = d[0:-1]

    W = W2[:, 0:-1]      # omit last column, which is the nullspace

    return W, LL, m


def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face
    # image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = []  # Label will store list of identity label

    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] == 'bmp':
            continue
        infile = os.path.join(directory, f)
        im = Image.open(infile)
        im_arr = np.asarray(im)
        im_arr = im_arr.astype(np.float32)

        # turn an array into vector
        im_vec = np.reshape(im_arr, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A)
    faces = np.transpose(faces)
    idLabel = np.array(Label)

    return faces, idLabel


def PCA_feature_extract(W, f, m):
    y = np.dot(W.T, f-m)
    return y


def lda_feature_extract(W, W1, f, m):
    y = np.dot(W.T, W1.T, f-m)
    return y


def fuse_feature(a, y_e, y_f):
    y = np.concatenate(a*y_e, (1-a)*y_f)
    return y


def confusion_matrix(n):
    m = np.zeros((n, n))
    return m


if __name__ == "__main__":
    train_path = "./train/"
    test_path = "./test/"
    F, labels = read_faces(train_path)  # F is 22400*120
    N = len(labels)  # number of image objects 120
    label_classes = np.unique(labels)
    class_num = len(label_classes)  # 10

    #  PCA train
    W, LL, m = myPCA(F)   # W-22400*119  LL:119-d eigenvalues, m-22400
    K = 30
    W_e = W[:, :K]   # W_e 22400*30
    print("shape of W_e_T:", W_e.T.shape, m.shape)
    z_e = np.zeros((N, K))    # store PCA projection of all images
    F_col = F.shape[1]
    Y_e = PCA_feature_extract(W_e, F, np.tile(m, (F_col, 1)).T)
    print("shape of y_e", Y_e.shape)
    partition = [np.where(labels == l)[0] for l in label_classes]
    classMean = [np.mean(Y_e[:, idx], 1) for idx in partition]
    classMean = np.array(classMean)  # 10*30

    # LDA train
    R = 90
    W1 = W[:, :R]  # W1 is 22400*90
    X = []
    X = np.dot(W1.T, (F-np.tile(m, (F_col, 1)).T))
    print("lda X", X.shape)
    W_lda, C_lda, L_lda = myLDA(X, labels)  # W_lda is 90*9, C_lda is 9*10

    # test begins
    F_test, label_test = read_faces(test_path)
    # PCA test
    M_pca = confusion_matrix(class_num)
    label_result = []
    Y_test_e = PCA_feature_extract(
        W_e, F_test, np.tile(m, (F_test.shape[1], 1)).T)
    print("Y_test_e", Y_test_e.shape)

    for i, y in enumerate(Y_test_e.T):
        distances = [(np.linalg.norm(y - v), i) for i, v in enumerate(classMean)]
        ordered_dis = sorted(distances, key=lambda d: d[0])
        y_label = ordered_dis[0][1]
        label_result.append(y_label)
    pca_score = 0
    for i, r in enumerate(label_result):
        if r == label_test[i]:
            pca_score += 1
        M_pca[label_test[i]][r] += 1
    pca_score = pca_score/120
    print("pca score:", pca_score)
    print("pca M:\n", M_pca)
    print("pca M score", np.trace(M_pca)/np.sum(M_pca))

    # LDA test
    lda_result = []
    Y_test_lda = np.dot(np.dot(W_lda.T, W1.T), (F_test-np.tile(m, (F_test.shape[1], 1)).T))
    print("Y test lda", Y_test_lda.shape)
    for i, y in enumerate(Y_test_lda.T):
        distances = [(np.linalg.norm(y-v), i) for i, v in enumerate(C_lda.T)]
        ordered_dis = sorted(distances, key=lambda d: d[0])
        x_label = ordered_dis[0][1]
        lda_result.append(x_label)
    lda_score = 0
    M_lda = confusion_matrix(class_num)
    for i, r in enumerate(lda_result):
        if r == label_test[i]:
            lda_score += 1
        M_lda[label_test[i]][r] += 1
    lda_score = lda_score/120
    print("lda score:", lda_score)
    print("lda M:\n", M_lda)
    print("lda M score", np.trace(M_lda)/np.sum(M_lda))

    # fusion test
    a_list = []
    score_list = []
    for aa in range(1, 10):
        a = aa/10
        C_fuse = np.concatenate((a*classMean.T, (1-a)*C_lda), axis=0)

        y_e = a*Y_test_e
        y_l = (1-a)*Y_test_lda
        Y_fuse_test = np.concatenate((y_e, y_l), axis=0)
        print("Y test fuse", Y_fuse_test.shape)
        fuse_result = []
        for i, y in enumerate(Y_fuse_test.T):
            distances = [(np.linalg.norm(y-v), i) for i, v in enumerate(C_fuse.T)]
            ordered_dis = sorted(distances, key=lambda d: d[0])
            x_label = ordered_dis[0][1]
            fuse_result.append(x_label)
        fuse_score = 0
        M_fuse = confusion_matrix(class_num)
        for i, r in enumerate(fuse_result):
            if r == label_test[i]:
                fuse_score += 1
            M_fuse[label_test[i]][r] += 1
        # print("fuse M:\n", M_fuse)
        print("a:",a)
        #a_list.append(a)
        #score_list.append(np.trace(M_fuse)/np.sum(M_fuse))
        print("fuse M score", np.trace(M_fuse)/np.sum(M_fuse))

    '''
    plt.plot(a_list, score_list)
    plt.axis([0, 1, 0.5, 1])
    plt.ylabel("accuracy")
    plt.title("accuracy - fuse weight")
    plt.show()
    '''
