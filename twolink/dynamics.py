import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import twolink.model as model
import time
import cv2

class TwoLinkSystem:
    def __init__(self, z0, dt):
        self.m0 = 0.2 # kg
        self.m1 = 0.01 # kg
        self.m2 = 0.01 # kg
        self.l1 = 1.0 # m
        self.l2 = 1.0 # m
        self.g = 9.8 # m / s**2

        self.kground = 50000 # N / m
        self.bground = 10 # N s / m

        fM = model.derive(m0=self.m0, m1=self.m1, m2=self.m2, l1=self.l1,
            l2=self.l2, g=self.g)

        self.fM = fM

        self.z0 = z0

        self.solver = scipy.integrate.ode(self.f)
        self.solver.set_initial_value(z0, 0.0)
        self.solver.set_integrator("vode", method="bdf")

        self.t = 0.0
        self.z = z0
        self.dt = dt

    def reset(self, z0):
        self.z0 = z0

        self.solver = scipy.integrate.ode(self.f)
        self.solver.set_initial_value(z0, 0.0)
        self.solver.set_integrator("vode", method="bdf")

        self.t = 0.0
        self.z = z0

        self.xprev = None

    def dynamics(self, z, v):
        x0 = z[0]
        y0 = z[1]
        t1 = z[2]
        t2 = z[3]
        dx0 = z[4]
        dy0 = z[5]
        dt1 = z[6]
        dt2 = z[7]

        tau1 = v[0]
        tau2 = v[1]

        if self.isInContact(z):
            print("-")
            kground = self.kground
            bground = self.bground
            if self.xprev is None:
                xground = x0 + self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
                self.xprev = xground
            else:
                xground = self.xprev
            yground = 0.0
        else:
            kground = 0.0
            bground = 0.0
            self.xprev = None
            xground = 0.0
            yground = 0.0

        #A : (dt2_dt, dt1_dt, y0, t2, x0, t1, dx0_dt, t, xground, kground, yground)
        #b : (t1(t), t2(t), tau2, tau1, t)
        #M : (dx0, dt2, y0, x0, dy0, t1, dt1, t2, t, tau1, tau2, xground, yground, kground)

        M = self.fM(dt2=dt2, dt1=dt1, y0=y0, t2=t2, x0=x0, t1=t1, dx0=dx0, t=0,
            xground=xground, kground=kground, yground=yground, tau2=tau2,
            tau1=tau1, dy0=dy0)

        return M

    def f(self, t, z):
        v = self.v
        dz = self.dynamics(z, v)
        return dz

    def step(self, v):
        self.t += self.dt
        self.v = v
        self.z = self.solver.integrate(self.t)

    def peek(self, v):
        t = self.t + self.dt
        self.v = v
        z = self.solver.integrate(t)
        self.solver.set_initial_value(self.z, self.t)
        return z

    def isAlive(self, z=None):
        if z is None:
            z = self.z
        y0 = z[1]
        return y0 > 0

    def isInContact(self, z=None):
        if z is None:
            z = self.z
        y0 = z[1]
        t1 = z[2]
        t2 = z[3]
        yc = y0 - self.l1 * np.cos(t1) - self.l2 * np.cos(t1 + t2)
        return yc <= 0

def showVideo(sys, zs, dt):
    img = np.zeros((500, 500, 3))
    img[-100:,:,1] = 255
    for z in zs:
        # state
        x0 = z[0]
        y0 = z[1]
        t1 = z[2]
        t2 = z[3]

        # knee and foot positions
        xj = x0 + sys.l1 * np.sin(t1)
        yj = y0 - sys.l1 * np.cos(t1)
        xc = xj + sys.l2 * np.sin(t1 + t2)
        yc = yj - sys.l2 * np.cos(t1 + t2)

        # render ground
        img[:-100,:,:] = 0
        img[-100:,:,0] = 180 * np.mod((-30 * x0 + np.tile(np.arange(500), 100).reshape(100, -1)) / 50, 2).astype("uint8")
        img[-100:,:,2] = 0

        # render body
        ibody = int(400 - 30 * y0)
        jbody = int(250 - 30 * (x0 - x0))
        img[ibody-10:ibody+10,240:260,0] = 255

        # render knee
        iknee = int(400 - 30 * yj)
        jknee = int(250 - 30 * (xj - x0))
        img[iknee-5:iknee+5,jknee-5:jknee+5,:2] = 128

        # render foot
        ifoot = int(400 - 30 * yc)
        jfoot = int(250 - 30 * (xc - x0))
        img[ifoot-5:ifoot+5,jfoot-5:jfoot+5,2] = 255

        # show image
        cv2.imshow("preview", img)
        cv2.waitKey(1)
        time.sleep(dt)

def showPlot(qs, ts):
    plt.plot(ts, qs[:,0])
    plt.plot(ts, qs[:,0] + qs[:,2])
    plt.show()

if __name__ == "__main__":
    z0 = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dt = 0.03
    sys = TwoLinkSystem(z0, dt)

    ts = [0.0]
    zs = [z0]
    while sys.t < 10.0 and sys.isAlive():
        sys.step(np.array([0.0001, -0.0001]))
        ts.append(sys.t)
        zs.append(sys.z)
    ts = np.array(ts)
    zs = np.array(zs)

    showVideo(sys, zs, dt)
