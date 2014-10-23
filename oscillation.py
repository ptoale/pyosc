#!/usr/bin/env python
"""
Driver for propagating neutrinos through the Earth and calculating oscillation
probabilities.

P. Toale
5/17/2012

An intial flavor is propagated through the Earth, where the Earth density
profile is parameterized by way of the PREM (see prem.py). The evolution is
governed by the Hamiltonian:

   i (d/dt) |nu> = H |nu>

where the Hamilotonian is:

   H = M2_f/(2*E) + V

The mass matrix, M2_f, is calculated in the neutrino_mixing module. The
potential, V, has a single non-zero value V[0][0] which is proportional to
the local, position depedent, electron density.

Instead of solving 3 complex equations (one for each flavor), we solve
6 real equations, where the neutrinos and anti-neutrinos form a 6-component
state.

   (d/dt) |nu>    =  Im(H) |nu> + Re(H) |nubar>
   (d/dt) |nubar> = -Re(H) |nu> + Im(H) |nubar>

The oscillation probabilty is then,

   P(nu_x -> nu_e)   = |nu>(e)**2   + |nubar>(e)**2
   P(nu_x -> nu_mu)  = |nu>(mu)**2  + |nubar>(mu)**2
   P(nu_x -> nu_tau) = |nu>(tau)**2 + |nubar>(tay)**2

"""
from neutrino_mixing import NeutrinoMixing
from prem import PREM
import numpy as np
import sys
from scipy.integrate import odeint
from optparse import OptionParser, OptionGroup

"""Default values"""
default_flavor          = "numu"
default_zenith          = 180.0
default_energy          = 25.0
#default_sinsq_2theta_12 = 0.861
#default_sinsq_2theta_13 = 0.092
#default_sinsq_2theta_23 = 1.000
#default_delta_msq_21    = 7.59e-5
#default_delta_msq_31    = 2.43e-3
default_sinsq_2theta_12 = 0.858624
default_sinsq_2theta_13 = 0.0975
default_sinsq_2theta_23 = 0.9744
default_delta_msq_21    = 7.58e-5
default_delta_msq_31    = 2.35e-3
default_delta_cp        = 0.0
default_hierarchy       = "normal"

"""Setup the parser"""
usage = "%prog [options]"
parser = OptionParser(usage=usage)

parser.add_option("-f", "--flavor", action="store", type="string",
                  dest="flavor", metavar="[nue,numu,nutau,nuebar,numubar,nutaubar]", default=default_flavor,
                  help="Neutrino flavor [%s]" % default_flavor)
parser.add_option("-z", "--zenith", action="store", type="float",
                  dest="zenith", metavar="<zenith>", default=default_zenith,
                  help="Neutrino zenith angle in degrees [%f]" % default_zenith)
parser.add_option("-e", "--energy", action="store", type="float",
                  dest="energy", metavar="<energy>", default=default_energy,
                  help="Neutrino energy in GeV [%f]" % default_energy)

parser.add_option("-o", "--output", action="store", type="string",
                  dest="output", metavar="[long, short, nue, numu, nutau]", default="long",
                  help="Output format [long]")

nu_group = OptionGroup(parser, "Neutrino Mixing Parameters")
nu_group.add_option("--sinsq2theta12", action="store", type="float",
                    dest="sinsq_2theta_12", metavar="<sinsq_2theta_12>", default=default_sinsq_2theta_12,
                    help="The square of the sine of twice theta_12 [%f]" % default_sinsq_2theta_12)
nu_group.add_option("--sinsq2theta13", action="store", type="float",
                    dest="sinsq_2theta_13", metavar="<sinsq_2theta_13>", default=default_sinsq_2theta_13,
                    help="The square of the sine of twice theta_13 [%f]" % default_sinsq_2theta_13)
nu_group.add_option("--sinsq2theta23", action="store", type="float",
                    dest="sinsq_2theta_23", metavar="<sinsq_2theta_23>", default=default_sinsq_2theta_23,
                    help="The square of the sine of twice theta_23 [%f]" % default_sinsq_2theta_23)
nu_group.add_option("--deltamsq21", action="store", type="float",
                    dest="delta_msq_21", metavar="<delta_msq_21>", default=default_delta_msq_21,
                    help="The difference in the squares of the 2-1 mass states [%f]" % default_delta_msq_21)
nu_group.add_option("--deltamsq31", action="store", type="float",
                    dest="delta_msq_31", metavar="<delta_msq_31>", default=default_delta_msq_31,
                    help="The difference in the squares of the 3-1 mass states [%f]" % default_delta_msq_31)
nu_group.add_option("--deltacp", action="store", type="float",
                    dest="delta_cp", metavar="<deltacp>", default=default_delta_cp,
                    help="The Dirac CP phase in degrees [%f]" % default_delta_cp)
nu_group.add_option("--hierarchy", action="store", type="string",
                    dest="hierarchy", metavar="[normal,inverted]", default=default_hierarchy,
                    help="The mass hierarchy [%s]" % default_hierarchy)
parser.add_option_group(nu_group)

opts, args = parser.parse_args()

"""Setup the initial state"""
anti = 0
y_0 = np.zeros(6)
if opts.flavor == "nue":
    y_0[0] = 1
elif opts.flavor == "numu":
    y_0[1] = 1
elif opts.flavor == "nutau":
    y_0[2] = 1
elif opts.flavor == "nuebar":
    y_0[3] = 1
    anti = 1
elif opts.flavor == "numubar":
    y_0[4] = 1
    anti = 1
elif opts.flavor == "nutaubar":
    y_0[5] = 1
    anti = 1
else:
    parser.error("Invalid flavor %s" % opts.flavor)

"""Check the zenith angle"""
if opts.zenith < 0 or opts.zenith > 180:
    parser.error("Unphysical zenith angle, must be between 0 and 180 degrees")

"""Check the energy"""
if opts.energy < 0:
    parser.error("Unphysical energy, must be positive")

"""Check the mixing parameters"""
if opts.sinsq_2theta_12 < 0 or opts.sinsq_2theta_12 > 1:
    parser.error("Unphysical sinsq_2theta_12, must be between 0 and 1")
if opts.sinsq_2theta_13 < 0 or opts.sinsq_2theta_13 > 1:
    parser.error("Unphysical sinsq_2theta_13, must be between 0 and 1")
if opts.sinsq_2theta_23 < 0 or opts.sinsq_2theta_23 > 1:
    parser.error("Unphysical sinsq_2theta_23, must be between 0 and 1")
if opts.delta_msq_31 < 0:
    parser.error("Delta_msq_31 must be positive. Use --hierarchy")
if opts.delta_cp < 0 or opts.delta_cp > 180:
    parser.error("Unphysical delta_cp, must be between 0 and 180 degrees")

"""Check the mass hierarchy"""
delta_atmo = opts.delta_msq_31
if opts.hierarchy == "inverted":
    delta_atmo *= -1.0

"""
Constants: in natural units
   gfermi in units of eV**-2
   na in units of mol**-1
   ye in units of mol/g
   hbarc in units of eV m

   pot_const is the constant part of the matter potential.
   It is constructed such that when multiplied by the density in g/cm**3, the
   potential will have units of eV to match the vacuum part.
   So the constant has units of eV (cm**3/g)
"""
gfermi = 1.16637e-5/(1.e9*1.e9)
na     = 6.02214129e23
ye     = 0.5
hbarc  = 1.97326972e-7
convert_cm_to_invEV = 100.*hbarc
convert_eV_to_invKm = 1000./hbarc
pot_const = np.sqrt(2.)*gfermi*na*ye*convert_cm_to_invEV**3

"""Setup the calculation"""
mixing = NeutrinoMixing(opts.sinsq_2theta_12, opts.sinsq_2theta_13, opts.sinsq_2theta_23,
                        opts.delta_msq_21, delta_atmo, opts.delta_cp)
prem = PREM(verbose=False)

"""Get the mass matrix (units are ev**2)"""
mass_matrix = mixing.calcMassMatrix()

def derivs(y, x, p):
    """
    This is the right-hand side of the differential equation. There are three input parameters:
       p[0]: the energy of the neutrino (in GeV)
       p[1]: the zenith angle (in degrees)
       p[2]: indicates if this is an anti-neutrino (potential flips sign)
    """

    energy = p[0]
    zenith = p[1]
    anti   = p[2]

    """This is the vacuum Hamiltonian (units are eV)"""
    H = mass_matrix/(2.0*energy*1.e9)

    """
    Now get the matter potential term.
       density will be in units of g/cm**3
       pot_const has units of eV (cm**3/g)
    """    
    density = prem.getDensity(x, zenith)
    V00 = density*pot_const
    if anti > 0:
        V00 *= -1.0

    """Add in the matter potential"""
    H[0][0] += V00

    """Finally convert the whole thing from units of eV to km**-1"""
    H *= convert_eV_to_invKm

    """
    The six coupled equations
    """
    dydx_0 =  y[0]*H[0][0].imag + y[1]*H[0][1].imag + y[2]*H[0][2].imag \
            + y[3]*H[0][0].real + y[4]*H[0][1].real + y[5]*H[0][2].real
    dydx_1 =  y[0]*H[1][0].imag + y[1]*H[1][1].imag + y[2]*H[1][2].imag \
            + y[3]*H[1][0].real + y[4]*H[1][1].real + y[5]*H[1][2].real
    dydx_2 =  y[0]*H[2][0].imag + y[1]*H[2][1].imag + y[2]*H[2][2].imag \
            + y[3]*H[2][0].real + y[4]*H[2][1].real + y[5]*H[2][2].real
    dydx_3 = -y[0]*H[0][0].real - y[1]*H[0][1].real - y[2]*H[0][2].real \
            + y[3]*H[0][0].imag + y[4]*H[0][1].imag + y[5]*H[0][2].imag
    dydx_4 = -y[0]*H[1][0].real - y[1]*H[1][1].real - y[2]*H[1][2].real \
            + y[3]*H[1][0].imag + y[4]*H[1][1].imag + y[5]*H[1][2].imag
    dydx_5 = -y[0]*H[2][0].real - y[1]*H[2][1].real - y[2]*H[2][2].real \
            + y[3]*H[2][0].imag + y[4]*H[2][1].imag + y[5]*H[2][2].imag

    return [dydx_0, dydx_1, dydx_2, dydx_3, dydx_4, dydx_5]


"""Get the layers to integrate over (in units of km) and add the starting point"""
points = [0]
for point in prem.getLayers(opts.zenith):
    points.append(point)

"""Integrate the ODE"""
p = [opts.energy, opts.zenith, anti]
y = odeint(derivs, y_0, points, args=(p,))

"""Get the final state and compute the oscillation probabilities"""
y_f = y[-1]
prob_nue   = y_f[0]*y_f[0] + y_f[3]*y_f[3]
prob_numu  = y_f[1]*y_f[1] + y_f[4]*y_f[4]
prob_nutau = y_f[2]*y_f[2] + y_f[5]*y_f[5]
prob = [prob_nue, prob_numu, prob_nutau]

"""Support several types of output"""
if opts.output == "long":
    print "Flavor=%s Zenith=%f Energy=%f:" % (opts.flavor, opts.zenith, opts.energy),
    print prob
elif opts.output == "short":
    print "%f %f %f" % (prob_nue, prob_numu, prob_nutau)
elif opts.output == "nue":
    print prob_nue
elif opts.output == "numu":
    print prob_numu
elif opts.output == "nutau":
    print prob_nutau

