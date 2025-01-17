import numpy as np


class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):

        self.data = np.array([w, x, y, z])

    @classmethod
    def copy(cls, rhs):
        return Quaternion(*rhs.data)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, val):
        self.data[i] = val

    def conj(self):
        return Quaternion(self[0], -self[1], -self[2], -self[2])

    def norm(self):
        return np.linalg.norm(self.data)

    def __add__(self, q):
        return Quaternion(*(self.data + q.data))

    def __sub__(self, q):
        return Quaternion(*(self.data - q.data))

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            a1, b1, c1, d1 = self.data
            a2, b2, c2, d2 = q.data
            return Quaternion(
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            )
        else:
            return Quaternion(*(q * self.data))

    def __repr__(self):
        return "Quaternion(%s,%s,%s,%s)" % tuple(self.data)

    def dot(self, q):
        return np.inner(self.data, q.data)

    def normalize(self):
        return Quaternion(*(self.data * 1.0 / self.norm()))

    def toRotation4(self):
        a, b, c, d = self.data
        return np.array(
            [
                [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (b * d + a * c), 0],
                [2 * (b * c + a * d), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (c * d - a * b), 0],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a ** 2 - b ** 2 - c ** 2 + d ** 2, 0],
                [0, 0, 0, 1],
            ]
        )

    def toRotation3(self):
        a, b, c, d = self.data
        return np.array(
            [
                [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                [2 * (b * c + a * d), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (c * d - a * b)],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a ** 2 - b ** 2 - c ** 2 + d ** 2],
            ]
        )


def quaternion_slerp(q1, q2, t):
    q1, q2 = q1.normalize(), q2.normalize()

    prod = q1.dot(q2)
    if abs(prod) > 0.9998:
        return q1 + (q2 - q1) * t
    # picks the shorter great circle
    if prod < 0:
        q2 = q2 * (-1.0)
        prod *= -1.0

    w = np.arccos(prod)
    return (q1 * (np.sin((1.0 - t) * w) / np.sin(w))) + q2 * (np.sin(t * w) / np.sin(w))


if __name__ == "__main__":
    q1 = Quaternion(1, 0, 0, 0)
    q2 = Quaternion(0, 1, 0, 0)

    for t in np.linspace(0, 1, 10):
        q = quaternion_slerp(q1, q2, t)
        print(t, q, q.norm())
