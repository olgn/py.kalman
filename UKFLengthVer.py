import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt
# mat_contents= scipy.io.loadmat('smallUnknown.mat') #l = .45
mat_contents= scipy.io.loadmat('smallKnown.mat') #pretty sure l = .38
accel = mat_contents['a_meas'][:,0]
gyro = mat_contents['d_meas'][:,0]
u = mat_contents['u'][:,0]
t = np.arange(0,20.001,0.01)
thetaTruth = mat_contents['th'][:,0]




n = 5
E =  np.concatenate((math.sqrt(n)*np.eye(n),-math.sqrt(n)*np.eye(n)),axis = 1)
xHat = np.zeros((n,len(u)))
yBar = np.zeros((2,len(u)))
y = np.zeros((2,n*2))
P = .01 * np.eye(n)
# P[4,4] = .1
Q = np.zeros((n,n))
Q[0,0] = .0000001
Q[1,1] = .0000001
Q[2,2] = .000001
Q[3,3] = .000001
# Q[4,4] = .00000001

R = .0001 * np.eye(2)

m1 = .6
m2 = 1.9
# l = .38
# l = .4
la = .2
Ts = .01

xHat[4,0] = .4
xHat[0,0] = 3.14159
def f(state,u,steps = 10):
	# print state
	th = state[0]
	# print th
	thDot = state[1]
	l = state[4]
	# for i in range(steps):
	# 	th = th + thDot*Ts/steps
	# 	thDotDot = ((9.81*(m1+m2)-m1*l*math.cos(th)*thDot**2)*math.sin(th)+math.cos(th)*u)/(l*(m1+m2-m1*math.cos(th)**2))
	# 	thDot = thDot + thDotDot*Ts/steps
	# return np.array([th,thDot,state[2],state[3],state[4]])

	thDotDot = ((9.81*(m1+m2)-m1*l*math.cos(th)*thDot**2)*math.sin(th)+math.cos(th)*u)/(l*(m1+m2-m1*math.cos(th)**2))

	return np.array([th+thDot*Ts,thDot+thDotDot*Ts,state[2],state[3],state[4]])
def g(state,u):
	th = state[0]
	thDot = state[1]
	l = state[4]
	a = (l-la)/l*((math.cos(th)-m1*l*math.cos(th)*thDot**2*math.sin(th))*u/(m1+m2-m1*math.cos(th)**2)+(m1+m2)*9.81*math.sin(th)/(m1+m2-m1*math.cos(th)**2))
	return np.array([thDot+state[2],a+state[3]])
for k in xrange(len(u)-1):
	#measurement update
	U, s, V = np.linalg.svd(P, full_matrices=True)
	M = np.dot(U,np.diag(np.sqrt(s)))

	xTilde = np.dot(M,E)
	xTilde += xHat[:,k:k+1]

	for i in xrange(2*n):
		y[:,i] = g(xTilde[:,i],u[k])

	yBar[:,k] = np.mean(y,1)

	PXY = 1.0/(2.0*n)*np.dot(xTilde - xHat[:,k:k+1],np.transpose(y - yBar[:,k:k+1]))
	PY = 1.0/(2.0*n)*np.dot(y - yBar[:,k:k+1],np.transpose(y-yBar[:,k:k+1]))+R
	K = np.dot(PXY,np.linalg.inv(PY))

	yMeas = np.reshape(np.array([gyro[k],accel[k]]),(2,1))

	xHat[:,k:k+1] = xHat[:,k:k+1] + np.dot(K,yMeas-yBar[:,k:k+1])
	P = P - np.dot(K,np.transpose(PXY))

	#time update
	U, s, V = np.linalg.svd(P, full_matrices=True)
	M = np.dot(U,np.diag(np.sqrt(s)))

	xTilde = np.dot(M,E)
	xTilde += xHat[:,k:k+1]

	for i in xrange(2*n):
		xTilde[:,i] = f(xTilde[:,i],u[k])

	xHat[:,k+1] = np.mean(xTilde,1)
	P = 1.0/(2.0*n)*np.dot(xTilde - xHat[:,k+1:k+2],np.transpose(xTilde - xHat[:,k+1:k+2])) + Q

# f, axarr = plt.subplots(5, sharex=True)
# axarr[0].plot(t,u)
# axarr[0].plot(t,xHat[0,:])
# axarr[1].plot(t,xHat[1,:])

# axarr[2].plot(t,xHat[1,:]+xHat[2,:])
# axarr[2].plot(t,gyro)
# axarr[3].plot(t,xHat[2,:])
# axarr[3].plot(t,xHat[3,:])
# axarr[4].plot(t,xHat[4,:])

# plt.show()


f, axarr = plt.subplots(4, sharex=True)
# f.set_title('woo')
axarr[0].set_title('Small Input, Known Length')
# axarr[0].plot(t,u,label = 'Input [N*m]')
axarr[0].plot(t,xHat[0,:], label = 'Theta Est.[rad]')
axarr[0].plot(t,thetaTruth, label = 'Theta [rad]')

# axarr[0].set_title('In')
axarr[0].legend(fontsize = 10)

axarr[1].plot(t,xHat[1,:], label = 'Gyro Est. [rad/s]' )
axarr[1].plot(t,gyro, label = 'Gyro Meas. [rad/s]')
axarr[1].legend(fontsize = 10)

# axarr[2].plot(t,gyro)
axarr[2].plot(t,xHat[2,:],label = 'Gyro Bias [rad/s]')
axarr[2].plot(t,xHat[3,:], label = 'Accel. Bias [m/s^2]')
axarr[2].legend(fontsize = 10)

axarr[3].plot(t,xHat[4,:],label = 'Length Estimate')
axarr[3].legend(fontsize = 10,loc = 4)


plt.xlabel('time [s]')
axarr[0].grid(b=True, which='major', color='black', linestyle='-')
axarr[1].grid(b=True, which='major', color='black', linestyle='-')
axarr[2].grid(b=True, which='major', color='black', linestyle='-')

axarr[3].grid(b=True, which='major', color='black', linestyle='-')
# plt.show()
plt.subplots_adjust(left=0.08, right=.95, top=0.95, bottom=0.05)
# f.figsize=(20,10)
plt.show()


