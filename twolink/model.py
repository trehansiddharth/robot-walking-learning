from sympy import *
import sympy.physics.mechanics as mechanics
from sympy.core.function import AppliedUndef
import pickle
import tarfile
import io
import numpy as np

def system(m0, m1, m2, l1, l2, g):
    x0, y0, t1, t2 = mechanics.dynamicsymbols("x0 y0 t1 t2")
    dx0, dy0, dt1, dt2 = mechanics.dynamicsymbols("x0 y0 t1 t2", 1)
    kground, bground, xground, yground = symbols("kground bground xground yground")
    tau1, tau2 = symbols("tau1 tau2")

    # positions of centers of mass
    r1 = l1 / 2
    r2 = l2 / 2

    # moment of inertias (assuming solid rod, negligible thickness)
    i1 = m1 * (l1 ** 2) / 12
    i2 = m2 * (l2 ** 2) / 12

    # position offsets
    sx1 =  l1 * sin(t1)
    sy1 = -l1 * cos(t1)
    sx2 =  l2 * sin(t1 + t2)
    sy2 = -l2 * cos(t1 + t2)

    # positions
    x1 = x0 + sx1 * r1 / l1
    y1 = y0 + sy1 * r1 / l1
    x2 = x0 + sx1 + sx2 * r2 / l2
    y2 = y0 + sy1 + sy2 * r2 / l2
    xc = x0 + sx1 + sx2
    yc = y0 + sy1 + sy2

    # velocities
    dx1 = x1.diff()
    dy1 = y1.diff()
    dx2 = x2.diff()
    dy2 = y2.diff()
    dxc = xc.diff()
    dyc = yc.diff()

    # linear kinetic energy
    Tlin0 = 0.5 * m1 * ((dx0 ** 2) + (dy0 ** 2))
    Tlin1 = 0.5 * m1 * ((dx1 ** 2) + (dy1 ** 2))
    Tlin2 = 0.5 * m2 * ((dx2 ** 2) + (dy2 ** 2))

    # rotational kinetic energy
    Trot0 = 0.0
    Trot1 = 0.5 * i1 * dt1 ** 2
    Trot2 = 0.5 * i2 * (dt1 + dt2) ** 2

    # total kinetic energy
    T = (Tlin0 + Tlin1 + Tlin2) + (Trot0 + Trot1 + Trot2)

    # potential energy
    U0 = m0 * g * y0
    U1 = m1 * g * y1
    U2 = m2 * g * y2
    Uc = 0.5 * kground * ((yc - yground) ** 2 + (xc - xground) ** 2)

    # total potential energy
    U = U0 + U1 + U2 + Uc

    # lagrangian L
    L = T - U

    v = Matrix([[0, 0, 0, 0, 0, 0, tau1, tau2]]).T

    qs = [x0, y0, t1, t2]
    dqs = [dx0, dy0, dt1, dt2]

    return L, v, qs, dqs

def motion(L, qs, dqs, v):
    LM = mechanics.LagrangesMethod(L, qs)
    LM.form_lagranges_equations()

    A = LM.mass_matrix_full
    b = LM.forcing_full

    return A.inv() * (b + v)

def derive(**kwargs):
    L, v, qs, dqs = system(**kwargs)

    M = motion(L, qs, dqs, v)

    dynM = tuple(mechanics.find_dynamicsymbols(M))
    frM = tuple(M.free_symbols)
    allM = tuple([dummify_undefined_functions(x) for x in dynM + frM])
    fM = lambdify(allM, dummify_undefined_functions(M), dummify=False)

    return fM

def save(A, b, params, filename):
    fA = io.StringIO(str(A))
    fb = io.StringIO(str(b))
    fparams = io.BytesIO(pickle.dumps(params))
    with tarfile.open(filename, "w:gz") as tar:
        tar.addfile(tarfile.TarInfo("A"), fA)
        tar.addfile(tarfile.TarInfo("b"), fb)
        tar.addfile(tarfile.TarInfo("params"), fparams)

def load(filename):
    pass

def dummify_undefined_functions(expr):
    mapping = {}

    for der in expr.atoms(Derivative):
        f_name = der.expr.func.__name__
        mapping[der] = Symbol("d%s" % f_name)

    for f in expr.atoms(AppliedUndef):
        f_name = f.func.__name__
        mapping[f] = Symbol(f_name)

    return expr.subs(mapping)

if __name__ == "__main__":
    fM = derive(m0=1.0, m1=0.1, m2=0.1, l1=1.0, l2=1.0, g=9.8)
