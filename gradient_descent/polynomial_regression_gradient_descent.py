import numpy as np

x1 = np.random.uniform(-2, 2, 10)
y1 = 4.1 * x1**3 + 3.1 * x1**2 + 2.1 * x1 + 1.0 + (1 / 10) * np.random.randn()

# y = ax**3 + bx**2 + cx + d
a = (1 / 10) * np.random.randn()
b = (1 / 10) * np.random.randn()
c = (1 / 10) * np.random.randn()
d = (1 / 10) * np.random.randn()
learning_rate = 0.01


def gradient_descent(x1, y1, a, b, c, d, learning_rate):
    # loss function for mean squared error of a polynomial regressor
    # yhat = ax**3 + bx**2 + cx + d # polynomial regression equation
    # loss = (y1 - yhat)**2
    # loss = (y1 - (ax**3 + bx**2 + cx + d))**2

    deriv_loss_wrt_a = 0.0
    deriv_loss_wrt_b = 0.0
    deriv_loss_wrt_c = 0.0
    deriv_loss_wrt_d = 0.0
    num_samples = x1.shape[0]

    for x1i, y1i in zip(x1, y1):
        deriv_loss_wrt_a += (
            -2 * (x1i**3) * (y1i - (a * x1i**3 + b * x1i**2 + c * x1i + d))
        )
        deriv_loss_wrt_b += (
            -2 * (x1i**2) * (y1i - (a * x1i**3 + b * x1i**2 + c * x1i + d))
        )
        deriv_loss_wrt_c += (
            -2 * (x1i**1) * (y1i - (a * x1i**3 + b * x1i**2 + c * x1i + d))
        )
        deriv_loss_wrt_d += (
            -2 * (x1i**0) * (y1i - (a * x1i**3 + b * x1i**2 + c * x1i + d))
        )

    # make update to our
    a -= learning_rate * (1 / num_samples) * deriv_loss_wrt_a
    b -= learning_rate * (1 / num_samples) * deriv_loss_wrt_b
    c -= learning_rate * (1 / num_samples) * deriv_loss_wrt_c
    d -= learning_rate * (1 / num_samples) * deriv_loss_wrt_d

    loss = (y1 - (a * x1**3 + b * x1**2 + c * x1 + d)) ** 2
    total_loss = np.sum(loss, axis=0)

    return a, b, c, d, total_loss


for epoch in range(1000):
    a, b, c, d, total_loss = gradient_descent(x1, y1, a, b, c, d, learning_rate)
    print(
        f"a = {np.round(a,2)}\t b = {np.round(b,2)}\t c = {np.round(c,2)}\t d = {np.round(d,2)}\tloss = {np.round(total_loss,2)}"
    )

print(f"\nlocal_min = {a}, {b}, {c}, {d}\n")
