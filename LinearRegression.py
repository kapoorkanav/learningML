import numpy as np
import matplotlib.pyplot as plt

# initializing the training data to roughly map y=2x+3
np.random.seed(42)
X=np.random.rand(100, 1)
y=2*X+3+(0.1*np.random.randn(100,1))
# print(X)
# print(y)

# adding bias (x0=1)
original=X
X_b=np.c_[np.ones((100,1)), X]
X=X_b

# initializing the coefficients
theta=np.random.randn(2,1)

# h function
def predict(input, theta):
    return (np.dot(input, theta))

# print(X.shape)

def mse(input, target, theta):
    # rel=(input@theta)-target
    # # rel.reshape(())
    # rel=rel.T.dot(rel)
    # return np.sum(rel)
    error = X @ theta - y
    return (1 / X.shape[0]) * np.sum(error ** 2)

# function for batch gradient descent
def bgd(train_X, train_y, alpha, iterations, theta):
    for i in range(iterations):
        m=train_X.shape[0]
        gradient=(1/m)*(train_X.T.dot(train_X.dot(theta)-train_y))
        theta-=(alpha*gradient)
        # if((i+1)%100==0):
        #     print('at iteration', ((i+1)), 'theta is', theta)
    return theta

# function for stochastic gradient descent
def sgd(train_X, train_y, alpha, iterations, theta):
    m=train_X.shape[0]
    for i in range(iterations):
        current=i%m
        # print('the shape is ', (predict(train_X[current], theta)-train_y[current]).shape)
        # print('the shape is ', ((train_X[current]@theta)-train_y[current]).shape)
        gradient=(train_X[current].reshape((2,1)).dot(((train_X[current]@theta)-train_y[current].reshape(1,1))))
        # gradient=(train_X[current].reshape((2,1)).dot((predict(train_X[current], theta)-train_y[current]).reshape(1,1)))
        theta-=(gradient*alpha)
    return theta

def mbgd(train_X, train_y, alpha, iterations, theta, batch_size):
    m=train_X.shape[0]
    for i in range(iterations):
        indices=np.random.permutation(m)
        X_shuffled=train_X[indices]
        y_shuffled=train_y[indices]
        for start in range(0, m, batch_size):
            end=start+batch_size
            X_batch=X_shuffled[start:end]
            y_batch=y_shuffled[start:end]
            error=X_batch@theta-y_batch
            gradient=(1/batch_size)*(X_batch.T@error)
            theta-=alpha*gradient
    return theta




plt.scatter(original,y)
# running the code

theta=np.random.randn(2,1)
theta=bgd(X, y, 0.01, 100000, theta)
plt.plot(original, predict(X, theta), color='red', label='BGD')
print('error with bgd', mse(X, y, theta))

theta=np.random.randn(2,1)
theta=sgd(X, y, 0.01, 100000, theta)
plt.plot(original, predict(X, theta), color='purple', label='SGD')
print('error with sgd', mse(X, y, theta))

theta=np.random.randn(2,1)
theta=mbgd(X, y, 0.01, 100000, theta, 16)
plt.plot(original, predict(X, theta), color='yellow', label='MBGD')
print('error with mbgd', mse(X, y, theta))


# plt.plot(original, predict(X, theta), color='red')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.show()