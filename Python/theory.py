#Exact error theory for every method, p and n, no sampling involved
#For each stratum: exact heart area inside it -> inside fraction f_i -> exact variance of A_hat
import numpy as np

#Same constants as src/heart.h
X0, X1 = -1.13903, 1.13903
Y0, Y1 = -1.0, 1.23666
B = (X1 - X0) * (Y1 - Y0)
A_REF = 3.66197272582
X_WIDE = 1.1390282 #Where the boundary goes vertical (sqrt kink), used as a panel break

METHODS = ["m1", "m2v", "m2h", "m3"]


def cut(lo, hi, k, q):
    #Same edge formula as decomp.c
    if k == q:
        return hi
    return lo + (hi - lo) * k / q


def subbox(method, rank, p):
    #Python port of decomp.c subbox(), returns (x0, x1, y0, y1)
    if method == "m1":
        return (X0, X1, Y0, Y1)
    if method == "m2v":
        return (cut(X0, X1, rank, p), cut(X0, X1, rank + 1, p), Y0, Y1)
    if method == "m2h":
        return (X0, X1, cut(Y0, Y1, rank, p), cut(Y0, Y1, rank + 1, p))
    if method == "m3":
        q = int(round(p ** 0.5))
        if q * q != p:
            raise ValueError("m3 needs a perfect square p")
        i, j = rank % q, rank // q
        return (cut(X0, X1, i, q), cut(X0, X1, i + 1, q), cut(Y0, Y1, j, q), cut(Y0, Y1, j + 1, q))
    raise ValueError(method)


def heart_intervals(x):
    #y intervals inside the heart at this x, clipped to the box
    #F(x, y) = (y^2 + c)^3 - x^2 y^3 with c = x^2 - 1, a degree 6 polynomial in y
    c = x * x - 1.0
    coeffs = [1.0, 0.0, 3 * c, -x * x, 3 * c * c, 0.0, c ** 3]
    roots = np.roots(coeffs)
    ys = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and Y0 <= r.real <= Y1)
    pts = [Y0] + ys + [Y1]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            continue
        m = 0.5 * (a + b)
        if (x * x + m * m - 1.0) ** 3 - x * x * m ** 3 <= 0.0:
            out.append((a, b))
    return out


class HeartArea:
    #Composite Gauss-Legendre in x on panels that line up with every stratum edge,
    #exact vertical chord lengths at each node from the polynomial roots

    def __init__(self, ps=(1, 4, 16, 36, 64), sub=24, order=10):
        breaks = {X0, X1, 0.0, -X_WIDE, X_WIDE}
        for p in ps:
            for k in range(p + 1):
                breaks.add(cut(X0, X1, k, p))
            q = int(round(p ** 0.5))
            if q * q == p:
                for k in range(q + 1):
                    breaks.add(cut(X0, X1, k, q))
        breaks = sorted(breaks)
        gx, gw = np.polynomial.legendre.leggauss(order)
        xs, ws = [], []
        for a, b in zip(breaks[:-1], breaks[1:]):
            edges = np.linspace(a, b, sub + 1)
            for u, v in zip(edges[:-1], edges[1:]):
                xs.extend(0.5 * (v - u) * gx + 0.5 * (u + v))
                ws.extend(0.5 * (v - u) * gw)
        self.x = np.array(xs)
        self.w = np.array(ws)
        self.iv = [heart_intervals(x) for x in self.x]

    def area(self, x0, x1, y0, y1):
        #Exact heart area inside one box
        total = 0.0
        sel = np.nonzero((self.x > x0) & (self.x < x1))[0]
        for k in sel:
            chord = 0.0
            for a, b in self.iv[k]:
                lo, hi = max(a, y0), min(b, y1)
                if hi > lo:
                    chord += hi - lo
            total += self.w[k] * chord
        return total


def strata_fractions(ha, method, p):
    #Inside fraction f_i of every stratum (M1: p copies of the whole box)
    out = []
    for r in range(p):
        x0, x1, y0, y1 = subbox(method, r, p)
        out.append(ha.area(x0, x1, y0, y1) / ((x1 - x0) * (y1 - y0)))
    return np.array(out)


def n_mixed(f, tol=1e-7):
    #Strata cut by the boundary, tol keeps quadrature rounding (f = 1 - 1e-9) from counting
    return int(np.sum((f > tol) & (f < 1.0 - tol)))


def sigma_rel(f, p, n):
    #Exact std of A_hat / A for strata fractions f, each weight B/p, n points each
    w = B / p
    return np.sqrt(np.sum(w * w * f * (1.0 - f) / n)) / A_REF


if __name__ == "__main__":
    ha = HeartArea()
    print("full box area %.10f vs A_REF %.10f" % (ha.area(X0, X1, Y0, Y1), A_REF))
    for p in (16, 36, 64):
        for m in METHODS:
            f = strata_fractions(ha, m, p)
            print(p, m, "sum of strata areas %.10f" % (np.sum(f) * B / p),
                  "sigma_rel at N=5.76e8: %.4e" % sigma_rel(f, p, 576000000 // p),
                  "mixed", n_mixed(f))