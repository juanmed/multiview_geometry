import numpy as np

def get_trifocal_tensor(P1, P2):
    """
    P0 = [I|0]
    P1 = [A|a4]
    P2 = [B|b4]

    A = [a1 a2 a3]
    B = [b1 b2 b3]

    Ti = a_ib4^T - a_4bi^t
    """
    T = list()
    for i in range(3):
        Ti = np.dot(P1[:,i].reshape(3,1),P2[:,3].reshape(3,1).T) - np.dot(P1[:,3].reshape(3,1),P2[:,i].reshape(3,1).T)
        T.append(Ti)
    return np.stack(T, axis=2)

def get_line_equation(x1, x2):
    """
    Return coefficientes a,b,c of line equation ax +by +c = 0
    given to points in the plane through which the line
    passes x1,x2
    """

    #slope
    m = (x2[1]*1.0 - x1[1])/(x2[0]*1.0 - x1[0])
    # normalization
    m0 = -m*x2[0] + x2[1]
    #c = -m*x1[0] + x1[1]

    #coefficients
    a = 1.0*m/m0
    b = -1./m0
    c = 1.0
    return np.array([a,b,c])

def transport_line(l_p, l_pp, tf):
    """
    l_p coefficients of line in 2nd camera
    l_pp coefficients of line in 3rd camera
    tf trifocal tensor of 3x3x3
    """
    l = np.zeros((3))
    for i in range(3):
        l[i] = np.dot(l_p.T, np.dot(tf[:,:,i], l_pp))
    l[0] = l[0]/l[2]
    l[1] = l[1]/l[2]
    l[2] = 1.0
    return l

def get_epipoles(tf):
    """
    Compute epipoles of 2nd a 3rd camera as seen in the first camera
    tf trifocal tensor
    """
    # left null vector
    u = list()
    for i in range(3):
        ns = nullspace(tf[:,:,i].T)
        left_vec = ns[:,0]
        u.append(left_vec.reshape(3,1))
    u = np.hstack(u)

    # right null vector
    v = list()
    for i in range(3):
        ns = nullspace(tf[:,:,i])
        right_vec = ns[:,0]
        v.append(right_vec.reshape(3,1))
    v = np.hstack(v)

    # epipoles
    ep = nullspace(u.T)
    epp = nullspace(v.T)
    return ep, epp    

    #print u

def nullspace(A, atol=1e-13, rtol=0):
    """
    From https://stackoverflow.com/questions/49852455/how-to-find-the-null-space-of-a-matrix-in-python-using-numpy
    """
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def get_fundamental_matrix21(ep, tf, epp):
    """
    Compute fundamental matrix 21
    ep 2n camera epipole
    epp 3rd camera epipole
    tf trifocal tensor
    """
    F = np.dot(hat(ep), np.dot(tf[:,:,0], np.dot(tf[:,:,1], np.dot(tf[:,:,2],epp))))
    return F


def get_fundamental_matrix31(epp, tf, ep):
    """
    Compute fundamental matrix 31
    ep 2n camera epipole
    epp 3rd camera epipole
    tf trifocal tensor
    """
    F = np.dot(hat(epp), np.dot(tf[:,:,0].T, np.dot(tf[:,:,1].T, np.dot(tf[:,:,2].T,ep))))
    return F

# take vector and transform to 3x3 skew-symmetric matrix
def hat(v):
    mx = np.zeros((3,3))
    mx[0][1] = -v[2][0]
    mx[0][2] =  v[1][0]
    mx[1][0] =  v[2][0]
    mx[1][2] = -v[0][0] 
    mx[2][0] = -v[1][0]
    mx[2][1] =  v[0][0]
    return mx

