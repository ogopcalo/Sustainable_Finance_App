from setuptools.command.install import install

import matplotlib.pyplot as plt
import numpy as np

mu = np.array([0.05,0.12])
sigma = np.array([0.09,0.20])
rhos = np.array([0,-1,-0.2])
weights = np.linspace(0,1,1001)

def var_covar(sigma, rho):
         Var_covar = np.array([
          [sigma[0]**2,rho*sigma[0]*sigma[1]],
          [rho*sigma[0]*sigma[1],sigma[1]**2]
         ])
         return var_covar

def eff_front(mu,sigma,rhos,weights):
        port_var = var_covar(sigma,rho)
        port_ret=[]
        port_stddev = []
        for w in weights:
              w = np.array([w,(1-w)])
              ret_p = np.dot(mu,w)
              stddev_p = np.dor(w,np.dot(port_var,w))

              port_ret.append(ret_p)
              port_stddev.append(stddev_p)

        return np.array(port_ret), np.array(port_stddev)


for rho in rhos:
     ret_p, stddev_p = eff_front(mu,sigma,rhos,weights)
     plt.plot(stddev_p,ret_p,label = "hdajic")

plt.xlabel("hjbvv")
plt.ylabel("hiuwhwiw")
plt.title("bwhivbw")
plt.legend()
plt.grid(True)
plt.savefig("fbhwuifbiw.png")
plt.show()

