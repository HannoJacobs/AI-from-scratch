import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_and_plot_3d_function(f):
    x1, x2 = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    y = f(x1, x2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x1, x2, y, cmap="coolwarm", linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title("3D Plot of Function")
    plt.show()


def main():
    # define an arbitrary 3d function
    f_x1_x2 = (
        lambda x1, x2: (x1**2 + x1 - 5 + 8 * np.sin(0.9 * x1))  # x1 function
        + (x2**2 + x2 - 5 + 7 * np.sin(0.9 * x2))  # x2 function
        + np.random.randn(*x1.shape)  # add noise
    )
    generate_and_plot_3d_function(f_x1_x2)


if __name__ == "__main__":
    main()
