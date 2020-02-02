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

X=np.vstack([data[0] , np.array([[1]*data[0].shape[0]])])
y_hat=data[1]
#caculate the loss
W=np.array([-4,-120])
W_lst=[]
W_lst.append([W[1],W[0]])
loss_lst=[]

#w,x using different learning rate
lr_w=0.000001
lr_b=0.00001
# lr_w=0.000001
# lr_b=0.000001
print('Gradient Descent\n')
for i in range(300000):
	w_grad=((2*(y_hat-np.dot(W,X)))*(-X)).sum()
	b_grad=-2*((y_hat-np.dot(W,X))).sum()
	# print('\nmatrix: ',w_grad,b_grad)
	w_grad=0
	b_grad=0
	for n in range(len(data[0])):
		b_grad=b_grad-2.0*(y_hat[n]-W[1]-W[0]*data[0][n])*1.0
		w_grad=w_grad-2.0*(y_hat[n]-W[1]-W[0]*data[0][n])*data[0][n]
	# print('\nfor: ',w_grad,b_grad)
	W=np.array([W[0]-w_grad*lr_w,W[1]-b_grad*lr_b])
	loss=((y_hat-np.dot(W,X))**2).sum()
	loss_lst.append(loss)
	W_lst.append([W[1],W[0]])
W_normal=W
W_lst_normal=W_lst
loss_lst_normal=loss_lst

W=np.array([-4,-120])
W_lst=[]
W_lst.append([W[1],W[0]])
loss_lst=[]
print('Adagrad\n')
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
W_adagrad=W
W_lst_adagrad=W_lst
loss_lst_adagrad=loss_lst

# print('\nThe solution is :w=',W[0],', b = ',W[1])
fig, ax_lst = plt.subplots()
plt.plot(data[0],data[1],'.')
m = np.linspace(0, 600)
plt.plot(m,W_normal[0]*m+W_normal[1],color="green",label="Gradient Descent",linewidth=1.5)
plt.plot(m,W_adagrad[0]*m+W_adagrad[1],color="red",label="Adagrad",linewidth=1.5)

plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.legend(loc='lower right')
fig, ax_lst = plt.subplots()


plt.plot(np.array(loss_lst_normal),label="Gradient Descent")
plt.plot(np.array(loss_lst_adagrad),label="Adagrad")
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y$',fontsize=16)
plt.legend(loc='lower right')
fig, ax_lst = plt.subplots()
plt.contourf(x,y,z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
line=np.array(W_lst_adagrad).T
plt.plot(line[0],line[1],'.-',label="Adagrad",alpha=0.5,linewidth=1.5)
line=np.array(W_lst_normal).T
plt.plot(line[0],line[1],'.-',label="Gradient Descent",alpha=0.5,linewidth=1.5)
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y$',fontsize=16)
plt.legend(loc='lower right')
plt.plot([-188.4],[2.67],'x',color='red')
plt.xlim(-200,-100)
plt.ylim(-5,5)



plt.show()