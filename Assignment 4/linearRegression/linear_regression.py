import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import jit,vmap,jacfwd, jacrev
from matplotlib import cm
from sklearn.linear_model import LinearRegression as sklearnLR
np.random.seed(45)
from mpl_toolkits.mplot3d import Axes3D
class LinearRegression():
  def __init__(self,fit_intercept=True):
    # Initialize relevant variables
    '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    '''
    self.fit_intercept = fit_intercept 
    self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
    self.all_coef=pd.DataFrame([]) # Stores the thetas for every iteration (theta vectors appended) (for the iterative methods)
    pass
  

  def fit_sklearn_LR(self,X,y):
    # Solve the linear regression problem by calling Linear Regression
    # from sklearn, with the relevant parameters
    tmp=sklearnLR(fit_intercept=self.fit_intercept)
    tmp.fit(X,y)
    theta0=tmp.intercept_
    self.coef=tmp.coef_
    self.coef=np.insert(self.coef,0,theta0)
    pass
  
  def fit_normal_equations(self,X,y):
    # Solve the linear regression problem using the closed form solution
    # to the normal equation for minimizing ||Wx - y||_2^2
    X_nmp=np.array(X)
    if(self.fit_intercept):
      X_nmp=np.insert(X_nmp,0,np.ones((X_nmp.shape[0],)),axis=1)
    y_nmp=np.array(y)

    X_trn=np.transpose(X_nmp)
    tmp=np.dot(X_trn,X_nmp)
    tmp2=np.dot(X_trn,y_nmp)
    tmp3=np.linalg.pinv(tmp)
    self.coef=np.dot(tmp3,tmp2)
    pass

  def fit_SVD(self,X,y):
    # Solve the linear regression problem using the SVD of the 
    # coefficient matrix
    X_nmp=np.array(X)
    if(self.fit_intercept):
      X_nmp=np.insert(X_nmp,0,np.ones((X_nmp.shape[0],)),axis=1)
    y_nmp=np.array(y)
    u,s,vh=np.linalg.svd(X_nmp,full_matrices=False)
    S=np.diag(s)
    S_inv=np.dot(np.linalg.inv(np.dot(S.T,S)),S.T)
    self.coef=np.dot(np.dot(vh.T,S_inv),np.dot(u.T,y))
    # sigma=np.zeros(X_nmp.shape,dtype=complex)
    # sigma[:X_nmp.shape[1]+1,:X_nmp.shape[1]+1]=np.diag(s)
    # tmp1=np.dot(np.transpose(u),y_nmp)
    # tmp2=np.dot(np.transpose(sigma),tmp1)
    # tmp3=np.linalg.inv(np.dot(sigma,np.transpose(sigma)))
    # tmp4=np.dot(np.transpose(vh),tmp3)
    # self.coef=np.dot(tmp4,tmp2)
    pass

  def mse_loss(self,X,y,coef):                
    # Compute the MSE loss with the learned model
    X_nmp=np.array(X)
    y_nmp=np.array(y)
    if(self.fit_intercept):
      X_nmp=np.insert(X_nmp,0,np.ones((X_nmp.shape[0],)),axis=1)
    # if(len(coef)!=0):
    # print(X_nmp.shape,coef.shape)

    y_hat=np.dot(X_nmp,coef)
    # else:
    #   y_hat=np.dot(X_nmp,self.coef)
    error=y_hat-y_nmp
    error_square=error**2
    return np.mean(error_square)
    
    pass

  def compute_gradient(self,X,y,penalty,mu=0.5):
    # Compute the analytical gradient (in vectorized form) of the 
    # 1. unregularized mse_loss,  and 
    # 2. mse_loss with ridge regularization
    # penalty :  specifies the regularization used  , 'l2' or unregularized
    X_nmp=np.array(X)
    y_nmp=np.array(y)

    if(self.fit_intercept):
      X_nmp=np.insert(X_nmp,0,np.ones(X_nmp.shape[0],),axis=1)
    # print(X_nmp.shape)
    if(penalty!='l2'):
      return 2*np.dot(np.transpose(X_nmp),(y_nmp-np.dot(X_nmp,self.coef)))*-1
    else:
      tmp=np.dot((mu*np.eye(X_nmp.shape[1])),self.coef)
      # print(tmp.shape)
      return 2*(np.dot(np.transpose(X_nmp),(y_nmp-np.dot(X_nmp,self.coef)))*-1)+tmp
    pass
  # @jit
  def cost_func_ridge(self,X,y,coef,mu):
    X_jnp=jnp.array(X)
    if(self.fit_intercept):
      X_jnp=jnp.insert(X_jnp,0,jnp.ones(X_jnp.shape[0],),axis=1)
    y_jnp=jnp.array(y)
    coeff=jnp.array(coef)
    tmp1=jnp.dot(X_jnp,coeff)
    tmp2=jnp.dot(jnp.transpose(y_jnp-tmp1),(y_jnp-tmp1))+mu*jnp.dot(jnp.transpose(coeff),coeff)
    return tmp2
  # @jit
  def cost_func_unreg(self,X,y,coef):
    X_jnp=jnp.array(X)
    if(self.fit_intercept):
      X_jnp=jnp.insert(X_jnp,0,jnp.ones(X_jnp.shape[0],),axis=1)
    y_jnp=jnp.array(y)
    coeff=jnp.array(coef)
    tmp1=jnp.dot(X_jnp,coeff)
    tmp2=jnp.dot(jnp.transpose(y_jnp-tmp1),(y_jnp-tmp1))
    return tmp2
  # @jit
  def cost_func_lasso(self,X,y,coef,delta):
    X_jnp=jnp.array(X)
    if(self.fit_intercept):
      X_jnp=jnp.insert(X_jnp,0,jnp.ones(X_jnp.shape[0],),axis=1)
    y_jnp=jnp.array(y)
    coeff=jnp.array(coef)
    tmp1=jnp.dot(X_jnp,coeff)
    tmp2=jnp.dot(jnp.transpose(y_jnp-tmp1),(y_jnp-tmp1))+(delta**2)*jnp.sum(jnp.abs(coeff))
    return tmp2
  
  def compute_jax_gradient(self,penalty,X,y,coef,mu=0.5,delta=0.5):
    # Compute the gradient of the 
    # 1. unregularized mse_loss, 
    # 2. mse_loss with LASSO regularization and 
    # 3. mse_loss with ridge regularization, using JAX 
    # penalty :  specifies the regularization used , 'l1' , 'l2' or unregularized
    if(penalty=="l1"):
      return np.array(jacrev(self.cost_func_lasso,argnums=(2))(X,y,coef,delta))
    elif(penalty=="l2"):
      return np.array(jacrev(self.cost_func_ridge,argnums=(2))(X,y,coef,mu))
    else:
      return np.array(jacrev(self.cost_func_unreg,argnums=(2))(X,y,coef))



    pass

  def fit_gradient_descent(self,X,y, batch_size=10, gradient_type='manual', penalty_type="none",num_iters=20, lr=0.01):
    # Implement batch gradient descent for linear regression (should unregularized as well as 'l1' and 'l2' regularized objective)
    # batch_size : Number of training points in each batch
    # num_iters : Number of iterations of gradient descent
    # lr : Default learning rate
    # gradient_type : manual or JAX gradients
    # penalty_type : 'l1', 'l2' or unregularized
    # X=np.insert(X,0,np.ones(X.shape[0],),axis=1)
    self.coef=100*np.random.random((np.array(X).shape[1]+1,))
    
    # self.coef=np.array([-40,80]).reshape(np.array(X).shape[1]+1,)
    # print(self.coef)
    for epoch in range(num_iters):

      p=np.random.permutation(len(X))
      X_nmp=(np.array(X))[p]
      y_nmp=(np.array(y))[p]
      batches=np.arange(0,len(X),batch_size)
      # batches=np.concatenate([batches,[-1]])
      # print(batches)
      for i in range(len(batches)-1):
        X_b=X_nmp[batches[i]:batches[i+1]]
        y_b=y_nmp[batches[i]:batches[i+1]]
        # self.compute_gradient(X,y,'none')
        # print(self.coef)
        if(gradient_type=='manual'):
          grad=self.compute_gradient(X_b,y_b,penalty_type)
        else:
          grad=self.compute_jax_gradient(penalty_type,X_b,y_b,self.coef)
        # tmp=self.coef
        self.coef=self.coef-(lr*grad)
        # self.co
      self.all_coef[f"{epoch}"]=pd.Series(self.coef,name=f"coef{epoch}")
      

    pass

    

  def fit_SGD_with_momentum(self, X,y,num_iters=20,penalty='l2', lr=0.01,beta=0.9):
    # Solve the linear regression problem using own implementation of SGD
    # penalty: refers to the type of regularization used (ridge)
    self.coef=100*np.random.random((np.array(X).shape[1]+1,))
    batch_size=1
    # self.coef=np.array([-40,80]).reshape(np.array(X).shape[1]+1,)
    # print(self.coef)
    for epoch in range(num_iters):

      p=np.random.permutation(len(X))
      X_nmp=(np.array(X))[p]
      y_nmp=(np.array(y))[p]
      batches=np.arange(0,len(X),batch_size)
      # batches=np.concatenate([batches,[-1]])
      # print(batches)
      vito=0
      for i in range(len(batches)-1):
        X_b=X_nmp[batches[i]:batches[i+1]]
        y_b=y_nmp[batches[i]:batches[i+1]]
        # self.compute_gradient(X,y,'none')
        
        grad=self.compute_gradient(X_b,y_b,'none')
        newvito=beta*vito + (1-beta)*grad
        self.coef=self.coef-(lr*newvito)
        vito=newvito
        # tmp=self.coef
        
        # self.coef=self.coef-(lr*grad)
        # self.co
      self.all_coef[f"{epoch}"]=pd.Series(self.coef,name=f"coef{i}")
      

  def predict(self, X):
    # Funtion to run the LinearRegression on a test data point
    # X_nmp=np.array(X)
    xrr=np.array(X)
    if(self.fit_intercept):
      xrr=xrr.reshape((-1,self.coef.shape[0]-1))
      xrr=np.insert(xrr,0,np.ones((xrr.shape[0],)),axis=1)
    else:
      xrr=xrr.reshape((-1,self.coef-1))

    return np.dot(xrr,self.coef)
    pass


  def plot_surface(self, X, y, theta_0, theta_1,azim=30,elev=30):
    '''
    Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
    theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1 by a
    red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.
      :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
      :param theta_0: Value of theta_0 for which to indicate RSS #pd Series of all theta_0
      :param theta_1: Value of theta_1 for which to indicate RSS #pd Series of all theta_1
      :return matplotlib figure plotting RSS
    '''
    fig,ax=plt.subplots(figsize=(10,10),subplot_kw={"projection": "3d"})
    xlist=np.linspace(-100,200,100)
    ylist=np.linspace(-100,200,100)
    theta0,theta1=np.meshgrid(xlist,ylist)
    xx,yy=theta0.flatten(),theta1.flatten()
    plt.xlabel("theta0")
    plt.ylabel("theta1")

    # grid=np.vstack((yy,xx))
    l=[]
    for i in range(len(xx)):
      tmp=np.array([xx[i],yy[i]])
      # tmp=tmp.reshape(2,1)
      # print(tmp)
      l.append(self.mse_loss(X,y,tmp))
    Z=np.array(l)
    # print(Z)
    Z=Z.reshape(theta0.shape)
    cp=ax.plot_surface(theta0,theta1,Z,cmap=cm.coolwarm)
    for i in range(len(theta_0)):
      ax.scatter(theta_0[i],theta_1[i],self.mse_loss(X,y,np.array([theta_0[i],theta_1[i]])),marker="x",c="red",s=30)
    # ax.scatter(theta_0,theta_1,self.mse_loss(X,y,np.array([theta_0,theta_1])),marker='x',markersize=5,markeredgecolor='red')
    fig.colorbar(cp, shrink=0.5, aspect=5)
    ax.view_init(azim, elev)
    return fig
    pass

  def plot_line_fit(self, X, y, theta_0, theta_1,iter):
    """
    Function to plot fit of the line (y vs. X plot) based on chosen value of theta_0, theta_1. Plot must
    indicate theta_0 and theta_1 as the title.
      :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
      :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
      :param theta_0: Value of theta_0 for which to plot the fit
      :param theta_1: Value of theta_1 for which to plot the fit
      :return matplotlib figure plotting line fit
      
    """
    fig,ax=plt.subplots()
    
    X2=np.linspace(np.min(X),np.max(X),1000).reshape(1000,1)
    X_nmp=np.insert(X2,0,np.ones(X2.shape[0],),axis=1)
    theta=[theta_0,theta_1]
    thetn=np.array(theta)
    y2=np.dot(np.array(X_nmp),thetn)
    plt.plot(X2,y2,'r')
    plt.scatter(X,y)
    plt.xlim(np.min(X),np.max(X))
    plt.ylim(np.min(y),np.max(y))
    plt.xlabel("values of X")
    plt.ylabel("values of Y")
    plt.title(f"iteration number {iter}")
    plt.plot()
    return fig
    pass


  def plot_contour(self, X, y, theta_0, theta_1):
    """
    Plots the RSS as a contour plot. A contour plot is obtained by varying
    theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1, and the
    direction of gradient steps. Uses self.coef_ to calculate RSS.
      :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
      :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
      :param theta_0: Value of theta_0 for which to plot the fit
      :param theta_1: Value of theta_1 for which to plot the fit
      :return matplotlib figure plotting the contour
    """
    fig,ax=plt.subplots()
    xlist=np.linspace(-50,100,500)
    ylist=np.linspace(-50,100,500)
    theta0,theta1=np.meshgrid(xlist,ylist)
    xx,yy=theta0.flatten(),theta1.flatten()
    plt.xlabel("theta0")
    plt.ylabel("theta1")

    # grid=np.vstack((yy,xx))
    l=[]
    for i in range(len(xx)):
      tmp=np.array([xx[i],yy[i]])
      # tmp=tmp.reshape(2,1)
      # print(tmp)
      l.append(self.mse_loss(X,y,tmp))
    Z=np.array(l)
    print(Z)
    Z=Z.reshape(theta0.shape)
    cp=ax.contourf(theta0,theta1,Z)
    fig.colorbar(cp)
    # for i in range(len(theta_0)):
    plt.plot(theta_0,theta_1,marker='x',markersize=5,markeredgecolor='red')
    return fig
    pass