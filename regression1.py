import numpy as np
import matplotlib.pyplot as plt
data=np.array([[338. ,333. ,328. ,207.,226. ,25. ,179., 60.,208.,606],
				[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]])

mode='normal'
#x-b
#y-w
x=np.arange(-200,-100,1)
y=np.arange(-5,5,0.1)
z=np.zeros((len(x),len(y)))
X,Y=np.meshgrid(x,y)
for i in range(len(x)):
	for  j in range(len(y)):
		b=x[i]
		w=y[j]
		for  n in range(len(data[0])):
			z[j][i]=z[j][i]+(data[1][n]-b-w*data[0][n])**2
		z[j][i]=z[j][i]/len(data[0])	


#y=W*X=w*x+b
#initialize the W and b
W=np.array([-4,-120])
X=np.vstack([data[0] , np.array([[1]*data[0].shape[0]])])
y_hat=data[1]
#caculate the loss
W_lst=[]
W_lst.append([W[1],W[0]])
loss_lst=[]
if mode=='normal':
	#w,x using different learning rate
	# lr_w=0.000001
	# lr_b=0.00001
	lr_w=0.000001
	lr_b=0.000001
	for i in range(100):
		w_grad=((2*(y_hat-np.dot(W,X)))*(-X)).sum()
		b_grad=-2*((y_hat-np.dot(W,X))).sum()
		print('\nmatrix: ',w_grad,b_grad)
		w_grad=0
		b_grad=0
		for n in range(len(data[0])):
			b_grad=b_grad-2.0*(y_hat[n]-W[1]-W[0]*data[0][n])*1.0
			w_grad=w_grad-2.0*(y_hat[n]-W[1]-W[0]*data[0][n])*data[0][n]
		print('\nfor: ',w_grad,b_grad)
		W=np.array([W[0]-w_grad*lr_w,W[1]-b_grad*lr_b])
		# loss=((y_hat-np.dot(W,X))**2).sum()
		# loss_lst.append(loss)
		W_lst.append([W[1],W[0]])
elif mode=='adagrad':
	print('adagrad\n')
	lr=1
	lr_b=0
	lr_w=0
	for i in range(100000):
		w_grad=((2*(y_hat-np.dot(W,X)))*(-X)).sum()
		b_grad=-2*((y_hat-np.dot(W,X))).sum()

		lr_b=lr_b+b_grad**2
		lr_w=lr_w+w_grad**2
		W=np.array([W[0]-w_grad*lr/np.sqrt(lr_w),W[1]-b_grad*lr/np.sqrt(lr_b)])
		loss=((y_hat-np.dot(W,X))**2).sum()
		loss_lst.append(loss)
		W_lst.append([W[1],W[0]])	
print('\nThe solution is :w=',W[0],', b = ',W[1])
fig, ax_lst = plt.subplots()
plt.plot(data[0],data[1],'.')
m = np.linspace(0, 600)
plt.plot(m,W[0]*m+W[1],'r')
fig, ax_lst = plt.subplots()
plt.plot(np.array(loss_lst))

line=np.array(W_lst).T
fig, ax_lst = plt.subplots()
plt.contourf(x,y,z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
print(line)
plt.plot(line[0],line[1],'o-')
plt.plot([-188.4],[2.67],'x',color='red')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.show()