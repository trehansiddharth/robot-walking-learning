import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import time
import cv2

class LegSystem:
    def __init__(self, q0, t0, dt):
        self.M = 0.2 # kg
        self.R = 0.3 # m
        self.m = 0.01 # kg
        self.l0 = 2 # m
        self.omega0 = 10 # Hz
        #gammma = 0.1 # Hz
        self.g = 9.8 # m / s**2

        self.k = 50 # N / m
        self.b = 5 # N s / m

        self.kground = 50000 # N / m
        self.bground = 10 # N s / m

        self.xprev = None

        self.q0 = q0
        self.t0 = t0

        self.solver = scipy.integrate.ode(self.f)
        self.solver.set_initial_value(q0, t0)
        self.solver.set_integrator("vode", method="bdf")

        self.t = t0
        self.q = q0
        self.dt = dt

    def dynamics(self, u, v):
        x, y, sx, sy, dx, dy, dsx, dsy = tuple(u)
        Ftheta = v

        theta = np.arctan2(sy, sx)

        ss = np.linalg.norm((sx, sy))
        l0x = self.l0 * sx / ss
        l0y = self.l0 * sy / ss

        if y <= 0:
            if self.xprev is None:
                self.xprev = x
            kcontact = self.kground
            xcontact = self.xprev
            bcontactx = self.bground
            if dy <= 0:
                bcontacty = self.bground
            else:
                bcontacty = 0.0
        else:
            self.xprev = None
            kcontact = 0.0
            xcontact = x
            bcontactx = 0.0
            bcontacty = 0.0

        def spring(k, b, m, x0, x, dx):
            return (k * (x - x0) + b * dx) / m

        ddx = spring(self.k, 0.0, self.m, l0x, sx, dsx) - spring(kcontact, bcontactx, self.m, xcontact, x, dx) - Ftheta * np.sin(theta) / self.m
        ddy = spring(self.k, 0.0, self.m, l0y, sy, dsy) - spring(kcontact, bcontacty, self.m, 0.0, y, dy) + Ftheta * np.cos(theta) / self.m - self.g
        ddsx = - spring(self.k, 0.0, self.M, l0x, sx, dsx) - ddx
        ddsy = - spring(self.k, 0.0, self.M, l0y, sy, dsx) - ddy - self.g

        return np.array([ddx, ddy, ddsx, ddsy])

    def f(self, t, u):
        q = u[0:4]
        dq = u[4:8]
        v = self.v
        ddq = self.dynamics(u, v)
        return np.concatenate((dq, ddq))

    def step(self, v):
        self.t += self.dt
        self.v = v
        self.q = self.solver.integrate(self.t)

    def peek(self, v):
        t = self.t + self.dt
        self.v = v
        q = self.solver.integrate(t)
        self.solver.set_initial_value(self.q, self.t)
        return q

    def isAlive(self, q=None):
        if q is None:
            q = self.q
        return q[1] + q[3] > 0

    def isInContact(self, q):
        if q is None:
            q = self.q
        return q[1] <= 0

def showVideo(qs, dt, outputFile=None):
    if outputFile is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(outputFile, fourcc, int(1/dt), (500, 500), True)
    img = np.zeros((500, 500, 3), dtype="uint8")
    img[-100:,:,1] = 255
    for q in qs:
        img[:-100,:,:] = 0
        img[-100:,:,0] = 180 * np.mod((-30 * q[0] + np.tile(np.arange(500), 100).reshape(100, -1)) / 50, 2).astype("uint8")
        img[-100:,:,2] = 0
        x = q[0]
        y = q[1]
        X = q[0] + q[2]
        Y = q[1] + q[3]
        ibody = int(400 - 30 * Y)
        jbody = int(250 - 30 * (X - X))
        img[ibody-10:ibody+10,240:260,0] = 255
        ifoot = int(400 - 30 * y)
        jfoot = int(250 - 30 * (x - X))
        img[ifoot-5:ifoot+5,jfoot-5:jfoot+5,2] = 255
        if outputFile is None:
            cv2.imshow("preview", img)
            cv2.waitKey(1)
            time.sleep(dt)
        else:
            out.write(img)
    if outputFile is not None:
        out.release()

def showPlot(qs, ts):
    plt.plot(ts, qs[:,0])
    plt.plot(ts, qs[:,0] + qs[:,2])
    plt.show()

if __name__ == "__main__":
    q0 = np.array([0.0, 10.0, 0.0, 2, 1.0, 0.0, 0.0, 0.0])
    t0 = 0.0
    dt = 0.05
    legSystem = LegSystem(q0, t0, dt)

    ts = [t0]
    qs = [q0]
    while legSystem.t < 15.0 and legSystem.isAlive():
        legSystem.step(0.001)
        ts.append(legSystem.t)
        qs.append(legSystem.q)
    ts = np.array(ts)
    qs = np.array(qs)

    showVideo(qs, dt)
    #showPlot(qs, ts)
