import matplotlib.pyplot as plt
import numpy as np

def k_gate(t,tau,s,r):
    val = phi(t,tau,s)
    res = np.zeros(val.shape)
    
    res[val < 0.5*r] = 2 * val[val < 0.5*r] / r
    res[(val > 0.5*r) & (val < r)] = 2 - 2 * val[(val > 0.5*r) & (val < r)] / r

    return res

def phi(t,tau,s):
    return ( (t-s) % tau ) / tau

tau,s = 5,1
t = np.linspace(0,3*tau,100)

i = 1

plt.rcParams["figure.figsize"] = (20,5)
for r in [0.5,1.0,2.0]:
    res = k_gate(t,tau,s,r)

    plt.subplot(1,3,i)

    plt.title(r"$s = %i$, $\tau = %i$, $r_{on} = %.2f$" % (s,tau,r))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$k^{(t)}$")
    plt.plot(t,res)
    plt.plot([tau+s,tau+s],[0,1],'--',label=r'$s + \tau$')
    plt.plot([s,s],[0,1],'--',label=r"$s$")
    plt.plot([s+tau*r/2,s+tau*r/2],[0,1],'--',label=r'$s + \frac{r_{on}}{2} \tau$')
    plt.grid()
    plt.legend()

    i += 1

plt.savefig("k_gate.png")
