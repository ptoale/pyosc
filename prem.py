#!/usr/bin/env python
"""
Module for computing track lengths through the Earth and density at
arbitrary positions along the track.

P. Toale
5/17/2012

The prelimenary reference Earth model (PREM) is used to describe the structure
of the Earth. For large zenith angles, IceCube is assumed to be a point on the
surface of the Earth. In reality, the center of IceCube is defined to be at an
elevation of 2900 ft (0.884 km), which is at a depth of 6391.31 ft (1.948 km)
below the surface of the ice. Assuming the ice surface is a sphere of constant
radius, the horizontal distance from the center of IceCube to the ice surface
is 157.571 km. Under the IceCube-on-the-surface assumption, a track length of
157.571 km occurs at a zenith angle of 90.709 degrees. Therefore at this angle,
I switch between the IceCube-on-the-surface assumption and the
ice-surface-is-a-sphere assumption. Between zenith angles of 90.709 and
90 degrees, the track length is taken to be 157.571 km of ice (water). For zenith
angles less than 90 degrees, the proper distance to the surface of the ice is
used. This treatment should be no less accurate than ignoring angles above the
horizon.

The PREM paper: http://dx.doi.org/10.1016/0031-9201(81)90046-7

There are 10 layers with different densities. The density is parameterized,
indpedentely in each layer, as a third order polynomial in the fractional
radius of the position, x = r/a, where a is the Earth radius (6371.0 km).

-------------------------------------------------------------------------------------------
Layer                | Range [km]          | Density [g/cm^3]
-------------------------------------------------------------------------------------------
0) Inner Core        |    0   - 1221.5 km  | 13.0885            - 8.8381*x**2
1) Outer Core        | 1221.5 - 3480.0 km  | 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
2) Lower Mantle      | 3480.0 - 5701.0 km  |  7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3
3) Transition Zone 1 | 5701.0 - 5771.0 km  |  5.3197 - 1.4836*x
4) Transition Zone 2 | 5771.0 - 5971.0 km  | 11.2494 - 8.0298*x
5) Transition Zone 3 | 5971.0 - 6151.0 km  |  7.1089 - 3.8045*x
6) LVZ/LID           | 6151.0 - 6346.6 km  |  2.6910 + 0.6924*x
7) Crust 1           | 6346.6 - 6356.0 km  |  2.9000
8) Crust 2           | 6356.0 - 6368.0 km  |  2.6000
9) Ocean             | 6368.0 - 6371.0 km  |  1.0200
-------------------------------------------------------------------------------------------

"""
from math import sqrt, sin, cos, radians, degrees
import numpy as np

"""
PREM parameters
"""
r_earth = 6371.0
n_layers = 10
r_layers = np.array([1221.5,
                     3480.0,
                     5701.0,
                     5771.0,
                     5971.0,
                     6151.0,
                     6346.6,
                     6356.0,
                     6368.0,
                     r_earth])

c0 = np.zeros(n_layers)
c1 = np.zeros(n_layers)
c2 = np.zeros(n_layers)
c3 = np.zeros(n_layers)

c0[0] =  13.0885
c0[1] =  12.5815
c0[2] =   7.9565
c0[3] =   5.3197
c0[4] =  11.2494
c0[5] =   7.1089
c0[6] =   2.6910
c0[7] =   2.9000
c0[8] =   2.6000
c0[9] =   1.0200

c1[1] =  -1.2638
c1[2] =  -6.4761
c1[3] =  -1.4836
c1[4] =  -8.0298
c1[5] =  -3.8045
c1[6] =   0.6924

c2[0] =  -8.8381
c2[1] =  -3.6426
c2[2] =   5.5283

c3[1] =  -5.5281
c3[2] =  -3.0807

"""
IceCube and South Pole elevations for down-going directions
"""
elevation_icecube = 0.884
elevation_surface = 2.832

r_icecube = r_earth + elevation_icecube
r_surface = r_earth + elevation_surface

horizontal_distance = sqrt(r_surface**2 - r_icecube**2)
zenith_break = 180. - degrees(np.arccos(horizontal_distance/(2.*r_earth)))

class PREM(object):
    """
    Main class object. Don't really need a class, but why not.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def getTrackLength(self, zenith_in_deg):
        """
        Get the length of the neutrino track through the earth for a given zenith angle.
           For zenith angles less than 90 degrees:
              Distance between center of IecCube and spherical ice surface
           For zenith angles between 90 and 90.709 degrees:
              Fixed at the horizontal distance to the spherical ice surface (157.571 km)
           For zenith angles larger than 90.709 degrees:
              Distance between points on the Earth surface according to the PREM
        """
        if zenith_in_deg < 90:
            ic_impact = r_icecube*sin(radians(zenith_in_deg))
            return sqrt(r_surface**2 - ic_impact**2) - sqrt(r_icecube**2 - ic_impact**2)
        elif zenith_in_deg < zenith_break:
            return horizontal_distance
        else:
            return -2.0*r_earth*cos(radians(zenith_in_deg))

    def getLayers(self, zenith_in_deg):
        """
        Get the end points of each PREM layer for a given zenith angle.
        """

        """Special treatment close to the horizon"""
        if zenith_in_deg < zenith_break:
            length = self.getTrackLength(zenith_in_deg)
            if self.verbose:
                print "Zenith = %f less than break zenith (%f)  Baseline = %f" \
                      % (zenith_in_deg, zenith_break, length)
            return np.array([length])

        sin_zenith = sin(radians(zenith_in_deg))
        cos_zenith = -cos(radians(zenith_in_deg))

        """Calculate the distance of closest approach to the center of the Earth"""
        baseline = 2.0*r_earth*cos_zenith
        impact = r_earth*sin_zenith

        if self.verbose:
            print "Zenith = %f  Baseline = %f  Impact parameter = %f" % (zenith_in_deg, baseline, impact)
            print ""

        """
        Determine the number of layers that the neutrino passes through
           Find the layer that the impact parameter falls in.
           Each of the outer layers will count twice (in and out)
        """
        im = n_layers
        if self.verbose: print "Finding the number of layers traversed..."
        for i in range(n_layers):
            if self.verbose: print "   Layer %d: radius = %f" % (i, r_layers[i])
            if impact < r_layers[i]:
                im = i
                if self.verbose: print "   This is the inner-most layer"
                break

        nl = n_layers-im
        if self.verbose:
            print "   Number of Layers = %d" % nl
            print ""

        """Now find the half-length of each layer"""
        half_length = np.zeros(nl)
        tot_length = 0.0
        if self.verbose: print "Finding the length traversered in each layer..."
        for k in range(nl):
            layer = im+k
            if k == 0:
                half_length[k] = sqrt(r_layers[layer]*r_layers[layer] - impact*impact)
            else:
                half_length[k] =   sqrt(r_layers[layer]*r_layers[layer]     - impact*impact) \
                                 - sqrt(r_layers[layer-1]*r_layers[layer-1] - impact*impact)
            tot_length += half_length[k]
            if self.verbose: print "   Half length in layer %d is %f" % (layer, half_length[k])
        if self.verbose:
            print "   Total length = %f" % (2.*tot_length)
            print ""


        """Finally find the positions of the boundaries"""
        nb = 2*nl-1
        pos = np.zeros(nb)
        if self.verbose: print "Finding the positions of the boundaries..."
        for k in range(nb):
            if k < nl:
                layer = nl - k - 1 + im
            else:
                layer = k - nl + 1 + im
            if k == 0:
                pos[k] = half_length[layer-im]
                if nl == 1:
                    pos[k] *= 2.0
            elif k == nl-1:
                pos[k] = pos[k-1] + 2*half_length[layer-im]
            else:
                pos[k] = pos[k-1] + half_length[layer-im]                
            if self.verbose: print "Boundary %d at %f" % (k, pos[k])

        return pos

    def getDensity(self, x, zenith):
        """
        Get the density at position x along a track with a given zenith angle.
        For zenith angles less than the break angle, return the density of the outer
        layer (Ocean).
        """

        """This is just water"""
        if zenith < zenith_break:
            return c0[n_layers-1]

        """Calculate the radius of this position (absolute and fractional)"""
        r = sqrt(r_earth*r_earth + x*x + 2.*x*r_earth*cos(radians(zenith)))
        rf = r/r_earth

        """Find the layer that this position is in"""
        layer = n_layers-1
        for i in range(n_layers):
            if r <= r_layers[i]:
                layer = i
                break

        """Compute the density"""
        density = c0[layer] + rf*(c1[layer] + rf*(c2[layer] + rf*c3[layer]))
        return density

def main():
    """
    Plot several profiles at different zenith angles.
    Requires matplotlib.
    """
    from optparse import OptionParser
    import matplotlib.pyplot as plt

    """Setup the parser"""
    parser = OptionParser()
    parser.add_option("-z", "--zenith", action="store", type="float", dest="zenith",
                      default=-1, help="Zenith angle to use")
    opts, args = parser.parse_args()

    """Create the PREM"""
    prem = PREM(verbose=True)

    """Setup the zenith angles to sample"""
    if opts.zenith >= 0:
        nz = 1
        zs = np.array([opts.zenith])
    else:
        nz = 6
        zs = np.linspace(90, 180, nz)

    """Number of points to use along track"""
    n = 1000
    den = np.zeros((nz,n))

    """Find the density along each track"""
    for i in range(nz):
        layers = prem.getLayers(zs[i])
        pos = np.linspace(0, layers[-1], n)

        for j in range(n):
            den[i][j] = prem.getDensity(pos[j], zs[i])
        plt.plot(pos, den[i], linewidth=3)

    """Pretty up the plots"""
    plt.ylim(0,14)
    plt.xlim(-1000, 14000)
    plt.title('Density Profile')
    plt.xlabel(r'L [km]')
    plt.ylabel(r'$\rho$ [g/cm$^3$]')
    plt.legend(zs, title='Zenith Angle')
    plt.grid(True)

    plt.savefig('prem_profile.png')

    plt.show()

if __name__ == "__main__":
    main()
