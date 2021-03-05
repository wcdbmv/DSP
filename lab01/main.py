#!/usr/bin/env python3

from math import exp
from sys import argv
import numpy as np
import matplotlib.pyplot as plt


def gauss(x):
    A: float = 1.0
    sigma: float = 5.0
    return A * exp(-(x / sigma) ** 2)


def rect(x):
    c: float = 5.0
    return int(abs(x) < c)


def main() -> None:
    vgauss = np.vectorize(gauss)
    vrect = np.vectorize(rect)

    # Ввод параметров дискретизации
    n = int(input('n: ')) if len(argv) < 2 else int(argv[1])
    dt = float(input('dt: ')) if len(argv) < 3 else float(argv[2])
    t_max = dt * (n - 1) / 2.0

    # Исходный сигнал
    x = np.arange(-t_max, t_max, 0.005)

    gauss_reference = vgauss(x)
    rect_reference = vrect(x)

    # Дискретизация
    t = np.arange(-t_max, t_max + dt / 2.0, dt)

    gauss_discrete = vgauss(t)
    rect_discrete = vrect(t)

    # Восстановление сигнала
    gauss_restored = np.zeros(x.size)
    rect_restored = np.zeros(x.size)
    for i in range(x.size):
        for j in range(n):
            sinc = np.sinc((x[i] - t[j]) / dt)
            gauss_restored[i] += gauss_discrete[j] * sinc
            rect_restored[i] += rect_discrete[j] * sinc

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, rect_reference, label='Исходный')
    axs[0].plot(x, rect_restored, label='Восстановленный')
    axs[0].scatter(t, rect_discrete, label='Дискретный')
    axs[0].grid(True)
    axs[1].plot(x, gauss_reference, label='Исходный')
    axs[1].plot(x, gauss_restored, label='Восстановленный')
    axs[1].scatter(t, gauss_discrete, label='Дискретный')
    axs[1].grid(True)

    # fig.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
