### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import numpy as np
import os, sys, itertools
import iminuit as Minuit
import matplotlib.pyplot as plt
from IPython.core.display import Latex
from IPython.display import display
from scipy import stats, optimize
from sympy import *


sys.path.append('C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\AppStat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

## Change directory to current one
os.chdir('Projects\\AppStat2022\\Project')


### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def lprint(*args,**kwargs):
    """Pretty print arguments as LaTeX using IPython display system 
    
    Parameters
    ----------
    args : tuple 
        What to print (in LaTeX math mode)
    kwargs : dict 
        optional keywords to pass to `display` 
    """
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)

### MAIN ------------------------------------------------------------------------------------------------------------------------------------


# The idea is to extract the times at which the ball passes the 5 gauges. Combined with the masurements of the relative distances
# of the gauges, a fit can be performed to estimate the acceleration, where the uncertainty on t and s must be taken into account

# Now, doing this for both the direct and reverse orientation gives an estimate of the angle between the table and the ground

def main():
    # Define variables:
    g, a, v, dv, D, d = symbols("g, a, v, dv, D, d")
    dg,da,dv,ddv,dD,dd = symbols("sigma_g, sigma_a, sigma_P, sigma_A, sigma_D")

    # Perimeter:
    # Define relation, and print:
    P = 2*L + 2*W
    lprint(latex(Eq(symbols('P'),P)))

    # Calculate uncertainty and print:
    dP = sqrt((P.diff(L) * dL)**2 + (P.diff(W) * dW)**2)
    lprint(latex(Eq(symbols('sigma_P'), dP)))

    # Turn expression into numerical functions 
    fP = lambdify((L,W),P)
    fdP = lambdify((L,dL,W,dW),dP)

    # Define values and their errors
    vL, vdL = 0.1, 1.2
    vW, vdW = 0.2, 1

    # Numerically evaluate expressions and print 
    vP = fP(vL,vW)
    vdP = fdP(vL,vdL,vW,vdW)
    lprint(fr'P = ({vP:.1f} \pm {vdP:.1f})\,\mathrm{{m}}')


if __name__ == '__main__':
    main()
