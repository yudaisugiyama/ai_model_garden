import numpy as np
from scipy.special import *

# x2 = x1 - f(x) / f'(x)
# f(x) = S0(0.78539816339745) - pi*h*w1 
# f'(x) =  pi*h

class Newton():
    def newton_method(self, w, h1, delta, S0):
        convtest = 1e-6
        err = convtest+1
        w = w + delta

        while err > convtest:
            h2 = h1 - ((S0-np.pi*h1*w) / (-np.pi*w))
            err = np.abs(S0-np.pi*h1*w) # 収束判定
            h1 = h2
        return h2

# デバッグ用
# newton = Newton()
# S0 = 0.78539816339745
# w = 0.5
# h = 0.5
# h2 = newton.newton_method(w,h, 0.1, S0)
