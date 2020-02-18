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



