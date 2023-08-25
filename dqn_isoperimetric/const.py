import numpy as np
from scipy.special import *

class Newton():
    def divided_quadrature_method(self, a, b):
        n = 100
        dqm = 0
        for i in range(n):
            theta = (np.pi*i)/(2*n)
            dqm = dqm + np.sqrt(\
                a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
            )
        dqm = (4*np.pi*0.5/n)*dqm
        return dqm

    def kdef(self, a, b):
        if a < b:
            a, b = b, a
        k = 1-b**2/a**2
        return k

    def derivative(self, a, b):
        k = self.kdef(a, b)
        if a < b:
            a, b = b, a
        der = 4*a * (ellipe(k)-ellipk(k)) / (b**2 - a**2) 
        return der

    def newton_method(self, a, x1, delta):
        convtest = 1e-2
        err = convtest+1
        dqm1 = self.divided_quadrature_method(a, x1)
        a = a + delta
        der = self.derivative(a, x1)

        while err > convtest:
            dqm2 = self.divided_quadrature_method(a, x1)
            x2 = x1 - (dqm2-dqm1) / der
            err = np.abs(x2-x1)
            x1 = x2
        return x2

