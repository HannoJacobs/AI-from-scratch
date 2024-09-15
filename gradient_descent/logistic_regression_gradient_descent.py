import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.uniform(-4, 4, 10)
y1 = 1 / (1 + np.exp(-1 * (2 * x1 + 1 + (1 / 100) * np.random.randn())))

# y = 1/(1 + np.exp(-1*(a*x1 + b)))
a = (1 / 10) * np.random.randn()
b = (1 / 10) * np.random.randn()
learning_rate = 0.1


def gradient_descent(x1, y1, a, b, learning_rate):
    # loss function for mean squared error of a logistic regressor
    # yhat = 1/(1 + np.exp(-1*(a*x1 + b))) # logistic regression equation
    # loss = (y1 - yhat)**2
    # loss = (y1 - 1/(1 + np.exp(-1*(a*x1 + b))))**2

    deriv_loss_wrt_a = 0.0
    deriv_loss_wrt_b = 0.0
    num_samples = x1.shape[0]

    for x1i, y1i in zip(x1, y1):
        deriv_loss_wrt_a += (
            -2 * (x1i) * np.exp(a * x1i + b) * ((y1i - 1) * np.exp(a * x1i + b) + y1i)
        ) / ((np.exp(a * x1i + b) + 1) ** 3)
        deriv_loss_wrt_b += (
            -2 * (1) * np.exp(a * x1i + b) * ((y1i - 1) * np.exp(a * x1i + b) + y1i)
        ) / ((np.exp(a * x1i + b) + 1) ** 3)

    # update the logistic regressor parameters
    a -= learning_rate * (1 / num_samples) * deriv_loss_wrt_a
    b -= learning_rate * (1 / num_samples) * deriv_loss_wrt_b

    loss = loss = (y1 - 1 / (1 + np.exp(-1 * (a * x1 + b)))) ** 2
    total_loss = np.sum(loss, axis=0)

    return a, b, total_loss


num_epochs = 2000
for epoch in range(num_epochs):
    a, b, total_loss = gradient_descent(x1, y1, a, b, learning_rate)
    print(
        f"a = {np.round(a,2)}\t b = {np.round(b,2)}\t loss = {np.round(total_loss,2)}"
    )

print(f"\nlocal_min = {a}, {b}\n")

# plotting original vs the approximated function
x1 = np.linspace(-4, 4, 100)
y1 = 1 / (1 + np.exp(-1 * (2 * x1 + 1 + (1 / 100) * np.random.randn())))
y_approx = 1 / (1 + np.exp(-1 * (a * x1 + b)))
plt.plot(x1, y1, "b", label="Original function")
plt.plot(x1, y_approx, "r", label="Approximated function")
plt.legend()
plt.title(f"The approximation after {num_epochs} epochs")
plt.show()
