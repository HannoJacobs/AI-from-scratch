import numpy as np

x1 = np.random.uniform(-2, 2, 10)
y1 = 2*x1 + 1 + np.random.randn()

m = np.random.randn()
c = np.random.randn()
learning_rate = 0.01

def gradient_descent(x1, y1, m, c, learning_rate):
    deriv_loss_wrt_m = 0.0
    deriv_loss_wrt_c = 0.0
    num_samples = x1.shape[0]

    loss = (y1 - (m*x1 + c))**2 # loss function for mean squared error of a linear regressor

    for x1i, y1i in zip(x1, y1):
        """
        Calculate the loss for each one of the training
        values and then make one step in the correct 
        direction when you have calculated the loss/MSE
        for all the training values

        loss = (y1 - yhat)**2 # Mean squared error
        y = mx + c # linear regression equation

        Goal: to changed the linear regression 
        equation parameters to make the loss function
        as small as possible. 
        """
        deriv_loss_wrt_m += -2*x1i*(y1i - (m*x1i + c))
        deriv_loss_wrt_c += -2*1*(y1i - (m*x1i + c))

    # make update to our 
    m -= learning_rate*(1/num_samples)*deriv_loss_wrt_m
    c -= learning_rate*(1/num_samples)*deriv_loss_wrt_c
    loss = (y1 - (m*x1 + c))**2
    total_loss = np.sum(loss, axis=0)

    return m, c, total_loss


for epoch in range(400):
    m, c, total_loss = gradient_descent(x1, y1, m, c, learning_rate)
    print(f"m = {np.round(m,2)}\tc = {np.round(c)}\tloss = {np.round(total_loss,2)}")

print(f"\nlocal_min = {m}, {c}\n")
