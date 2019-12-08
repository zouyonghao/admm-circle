import pandas as pd
import matplotlib.pyplot as plt
import math
import random

data0 = pd.read_csv("data/data4.txt", header=None, sep=" ")
data0[:5]


def g1(a, b, r, ksi1, x, y):
    return -ksi1 - r + x * x - 2 * a * x + a * a + y * y - 2 * b * y + b * b


def g2(a, b, r, ksi2, x, y):
    return -ksi2 + r - x * x + 2 * a * x - a * a - y * y + 2 * b * y - b * b


def train():

    # parameters
    ksi1 = [0.4] * len(data0)
    ksi2 = [0.4] * len(data0)
    r = 0.05
    a = 0
    b = 0

    # lagrange parameters
    u1 = [0] * len(data0)
    u2 = [0] * len(data0)
    # z = 0
    u3 = [0] * len(data0)
    u4 = [0] * len(data0)
    # uz = 0

    # step
    p = 0.1
    step = 0.1
    i = 0
    # outer loop
    while (i < 100):
        i += 1
        j = 0

        # print(ksi1)
        # print(ksi2)
        print(a)
        print(b)
        print(r)
        # print(u1, u2, u3, u4)

        # inner loop
        while (j < 100):
            j += 1
            # print(j)
            k = 0
            while (k < len(data0)):
                # rand_num = random.randint(0, len(data0) - 1)
                x = data0[0][k]
                y = data0[1][k]
                tmp_g1 = g1(a, b, r, ksi1[k], x, y)
                tmp_g2 = g2(a, b, r, ksi2[k], x, y)
                tmp_ksi1 = max(0, -ksi1[k])
                tmp_ksi2 = max(0, -ksi2[k])

                # Partial derivative
                tmp_g1_ksi1 = -1
                tmp_g2_ksi2 = -1
                tmp_g1_r = -1
                tmp_g2_r = 1
                tmp_g1_a = -2 * x + 2 * a
                tmp_g2_a = 2 * x - 2 * a
                tmp_g1_b = -2 * y + 2 * b
                tmp_g2_b = 2 * y - 2 * b
                if tmp_g1 <= 0:
                    tmp_g1 = 0
                    tmp_g1_ksi1 = 0
                    tmp_g1_r = 0
                    tmp_g1_a = 0
                    tmp_g1_b = 0

                if tmp_g2 <= 0:
                    tmp_g2 = 0
                    tmp_g2_ksi2 = 0
                    tmp_g2_r = 0
                    tmp_g2_a = 0
                    tmp_g2_b = 0

                grad_ksi1 = 0.1 + p * tmp_g1 * tmp_g1 * 2 * tmp_g1 * tmp_g1_ksi1 \
                    + p * 2 * u1[k] * tmp_g1 * tmp_g1_ksi1 \
                    + 2 * p * (max(0, -ksi1[k]) ** 3) * -1 \
                    + 2 * u3[k] * (max(0, -ksi1[k])) * -1

                grad_ksi2 = 0.1 + p * tmp_g2 * tmp_g2 * 2 * tmp_g2*tmp_g2_ksi2 \
                    + p * 2 * u2[k] * tmp_g2 * tmp_g2_ksi2 \
                    + 2 * p * (max(0, -ksi2[k]) ** 3) * -1 \
                    + 2 * u4[k] * (max(0, -ksi2[k])) * -1

                grad_r = p * tmp_g1 * tmp_g1 * 2 * tmp_g1 * tmp_g1_r \
                    + p * 2 * u1[k] * tmp_g1 * tmp_g1_r \
                    + p * tmp_g2 * tmp_g2 * 2 * tmp_g2 * tmp_g2_r \
                    + p * 2 * u2[k] * tmp_g2 * tmp_g2_r

                grad_a = p * tmp_g1 * tmp_g1 * 2 * tmp_g1 * tmp_g1_a \
                    + p * 2 * u1[k] * tmp_g1 * tmp_g1_a \
                    + p * tmp_g2 * tmp_g2 * 2 * tmp_g2 * tmp_g2_a \
                    + p * 2 * u2[k] * tmp_g2 * tmp_g2_a

                grad_b = p * tmp_g1 * tmp_g1 * 2 * tmp_g1 * tmp_g1_b \
                    + p * 2 * u1[k] * tmp_g1 * tmp_g1_b \
                    + p * tmp_g2 * tmp_g2 * 2 * tmp_g2 * tmp_g2_b \
                    + p * 2 * u2[k] * tmp_g2 * tmp_g2_b

                ksi1[k] -= step * grad_ksi1
                ksi1[k] = ksi1[k] if ksi1[k] > 0 else 0

                ksi2[k] -= step * grad_ksi2
                ksi2[k] = ksi2[k] if ksi2[k] > 0 else 0

                r -= step * grad_r
                a -= step * grad_a
                b -= step * grad_b

                # # update u1 u2
                # tmp_g1 = g1(a, b, r, ksi1[k], x, y)
                # tmp_g1 = tmp_g1 if tmp_g1 < 0 else 0
                # u1[k] += p * tmp_g1 * tmp_g1
                # tmp_g2 = g2(a, b, r, ksi2[k], x, y)
                # tmp_g2 = tmp_g2 if tmp_g2 < 0 else 0
                # u2[k] += p * tmp_g2 * tmp_g2

                k += 1
            k = 0
            while (k < len(data0)):
                # rand_num = random.randint(0, len(data0) - 1)
                x = data0[0][k]
                y = data0[1][k]
                tmp_g1 = max(0, g1(a, b, r, ksi1[k], x, y))
                tmp_g2 = max(0, g2(a, b, r, ksi2[k], x, y))
                # update u1 u2

                # tmp_g1 = tmp_g1 if tmp_g1 < 0 else 0
                u1[k] += p * tmp_g1 * tmp_g1

                # tmp_g2 = tmp_g2 if tmp_g2 < 0 else 0
                u2[k] += p * tmp_g2 * tmp_g2

                u3[k] += p * max(0, -ksi1[k])

                u4[k] += p * max(0, -ksi2[k])

                k += 1

        fig, ax = plt.subplots()
        circle = plt.Circle((a, b), math.sqrt(r), color='r', fill=False)
        # circle1 = plt.Circle((a, b), r + ksi1, color='r', fill=False)
        # circle2 = plt.Circle((a, b), r - ksi2, color='b', fill=False)
        ax.axis('equal')
        ax.axis([-3, 3, -3, 3])
        ax.add_artist(circle)
        # ax.add_artist(circle1)
        # ax.add_artist(circle2)
        ax.plot(data0[0], data0[1], 'o')
        fig.savefig("progress.png")


train()
