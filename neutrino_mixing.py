#!/usr/bin/env python
"""
Module that provides several useful calculations related to neutrino mixing.

P. Toale
5/17/2012

The module provides a single class that represents the neutrino mixing matrix
and the mass matrix for 3 flavors of Dirac neutrinos. The class is initilized
with the three mixing angles, the two mass-squared differences, and the one
CP phase.

The mixing matrix is calculated in the standard parameterization:

   U = U(23)*I(delta)*U(13)*I(-delta)*U(12)

where the U's are rotation matrices,

   U(12) = [[ cos(theta_12),  sin(theta_12),              0],
            [-sin(theta_12),  cos(theta_12),              0],
            [             0,              0,              1]]

   U(13) = [[ cos(theta_13),              0,  sin(theta_13)],
            [             0,              1,              0],
            [-sin(theta_13),              0,  cos(theta_13)]]

   U(23) = [[             1,              0,              0],
            [             0,  cos(theta_23),  sin(theta_23)],
            [             0, -sin(theta_23),  cos(theta_23)]],

and the I's are diagonal,

   I(delta) = diag[1, 1, exp(delta)]

The mass matrix in the flavor basis is also produced. This is calculated as

   M2_f = U M2 U^dagger

where M = diag[0, delta_msq_21, delta_msq_31] is the mass matrix in the mass
basis and has units of eV**2.

"""
from math import sqrt, sin, cos, radians
import numpy as np

class NeutrinoMixing(object):
    """Neutrino mixing class.

    :param sinsq_2theta_12: The square of the sine of twice the 12 mixing angle.
    :type sinsq_2theta_12: float
    :param sinsq_2theta_13: The square of the sine of twice the 13 mixing angle.
    :type sinsq_2theta_13: float
    :param sinsq_2theta_23: The square of the sine of twice the 23 mixing angle.
    :type sinsq_2theta_23: float
    :param delta_msq_21: The difference in the squares of the masses of the
                         second and first mass eigenstates [ev**2].
    :type delta_msq_21: float
    :param delta_msq_31: The difference in the squares of the masses of the
                         third and first mass eigenstates [ev**2].
    :type delta_msq_31: float
    :param delta_cp: The Dirac CP phase [rad].
    :type delta_cp: float

    The default values are the world averaged/best as of May 2012, taken from the
    PDG: http://pdg.lbl.gov/2011/tables/rpp2011-sum-leptons.pdf

    For a normal mass hierarchy, be sure that delta_msq_31 is positive. If you
    want an inverted mass hierarchy, construct with a negative delta_msq_31.
    """

    def __init__(self,
#                 sinsq_2theta_12 = 0.861,
#                 sinsq_2theta_13 = 0.092,
#                 sinsq_2theta_23 = 1.0,
#                 delta_msq_21 = 7.59e-5,
#                 delta_msq_31 = 2.43e-3,
                 sinsq_2theta_12 = 0.858624,
                 sinsq_2theta_13 = 0.0975,
                 sinsq_2theta_23 = 0.9744,
                 delta_msq_21 = 7.58e-5,
                 delta_msq_31 = 2.35e-3,
                 delta_cp = 0.0):
        """Construct a NeutrinoMixing object with a specific set of mixing parameters."""

        self.sinsq_2theta_12 = sinsq_2theta_12
        self.sinsq_2theta_13 = sinsq_2theta_13
        self.sinsq_2theta_23 = sinsq_2theta_23

        self.sin_12 = sqrt(0.5*(1.0 - sqrt(1 - sinsq_2theta_12)))
        self.sin_13 = sqrt(0.5*(1.0 - sqrt(1 - sinsq_2theta_13)))
        self.sin_23 = sqrt(0.5*(1.0 - sqrt(1 - sinsq_2theta_23)))

        self.msq_21 = delta_msq_21
        self.msq_31 = delta_msq_31

        self.delta = radians(delta_cp)

        self.U = None
        self.S = None

    def calcMixingMatrix(self):
        """Return the mixing matrix. First calculate it if hasn't been already.

        :returns: The matrix that connects the mass eigenstates to the flavor eigenstates.
        :rtype: 2-D complex numpy array
        """

        if self.U is None:
            """Need to calculate it"""

            sin_12 = self.sin_12
            sin_13 = self.sin_13
            sin_23 = self.sin_23
            cos_12 = sqrt(1.0 - sin_12*sin_12)
            cos_13 = sqrt(1.0 - sin_13*sin_13)
            cos_23 = sqrt(1.0 - sin_23*sin_23)
        
            sin_delta = sin(self.delta)
            cos_delta = cos(self.delta)

            RU_11 =  cos_12*cos_13
            RU_21 = -sin_12*cos_23 - cos_12*sin_13*sin_23*cos_delta
            RU_31 =  sin_12*sin_23 - cos_12*sin_13*cos_23*cos_delta
            RU_12 =  sin_12*cos_13
            RU_22 =  cos_12*cos_23 - sin_12*sin_13*sin_23*cos_delta
            RU_32 = -cos_12*sin_23 - sin_12*sin_13*cos_23*cos_delta
            RU_13 =  sin_13*cos_delta
            RU_23 =  cos_13*sin_23
            RU_33 =  cos_13*cos_23

            IU_11 =  0
            IU_21 = -cos_12*sin_13*sin_23*sin_delta
            IU_31 = -cos_12*sin_13*cos_23*sin_delta
            IU_12 =  0
            IU_22 = -sin_12*sin_13*sin_23*sin_delta
            IU_32 = -sin_12*sin_13*cos_23*sin_delta
            IU_13 = -sin_13*sin_delta
            IU_23 =  0
            IU_33 =  0

            self.U = np.array([[complex(RU_11, IU_11), complex(RU_12, IU_12), complex(RU_13, IU_13)],
                               [complex(RU_21, IU_21), complex(RU_22, IU_22), complex(RU_23, IU_23)],
                               [complex(RU_31, IU_31), complex(RU_32, IU_32), complex(RU_33, IU_33)]])

        return self.U

    def calcMassMatrix(self):
        """Return the mass matrix in the flavor basis. First calculate it if hasn't been already.

        :returns: The mass matrix in the flavor basis [ev**2].
        :rtype: 2-D complex numpy array
        """

        if self.S is None:
            """Need to calculate it"""
        
            msq_21 = self.msq_21
            msq_31 = self.msq_31
        
            U = self.calcMixingMatrix()

            S_11 = msq_21*U[0][1]*U[0][1].conjugate() + msq_31*U[0][2]*U[0][2].conjugate()
            S_21 = msq_21*U[1][1]*U[0][1].conjugate() + msq_31*U[1][2]*U[0][2].conjugate()
            S_31 = msq_21*U[2][1]*U[0][1].conjugate() + msq_31*U[2][2]*U[0][2].conjugate()
            S_12 = msq_21*U[0][1]*U[1][1].conjugate() + msq_31*U[0][2]*U[1][2].conjugate()
            S_22 = msq_21*U[1][1]*U[1][1].conjugate() + msq_31*U[1][2]*U[1][2].conjugate()
            S_32 = msq_21*U[2][1]*U[1][1].conjugate() + msq_31*U[2][2]*U[1][2].conjugate()
            S_13 = msq_21*U[0][1]*U[2][1].conjugate() + msq_31*U[0][2]*U[2][2].conjugate()
            S_23 = msq_21*U[1][1]*U[2][1].conjugate() + msq_31*U[1][2]*U[2][2].conjugate()
            S_33 = msq_21*U[2][1]*U[2][1].conjugate() + msq_31*U[2][2]*U[2][2].conjugate()

            self.S = np.array([[S_11, S_12, S_13],[S_21, S_22, S_23], [S_31, S_32, S_33]])

        return self.S


    def calcOldStyleMassMatrix(self):
        """
        The following is for comparing to the original fortran code.
        It has been fixed here to match the matrix calculation.
        It can be cleaned up soon...
        """

        sin_12 = self.sin_12
        sin_13 = self.sin_13
        sin_23 = self.sin_23
        cos_12 = sqrt(1.0 - sin_12*sin_12)
        cos_13 = sqrt(1.0 - sin_13*sin_13)
        cos_23 = sqrt(1.0 - sin_23*sin_23)

        msq_21 = self.msq_21
        msq_31 = self.msq_31

        sin_delta = sin(self.delta)
        cos_delta = cos(self.delta)

        RSS_11 =  msq_31*sin_13*sin_13 \
                 + msq_21*(sin_12*sin_12*cos_13*cos_13)
        RSS_21 =  msq_31*sin_13*cos_13*sin_23*cos_delta \
                 + msq_21*sin_12*cos_13*(cos_12*cos_23 - sin_12*sin_13*sin_23*cos_delta)
        RSS_31 =  msq_31*sin_13*cos_13*cos_23*cos_delta \
                 - msq_21*sin_12*cos_13*(cos_12*sin_23 + sin_12*sin_13*cos_23*cos_delta)
        RSS_12 = RSS_21
        RSS_22 =  msq_31*cos_13*cos_13*sin_23*sin_23 \
                 + msq_21*(cos_12*cos_12*cos_23*cos_23 + sin_12*sin_12*sin_13*sin_13*sin_23*sin_23 \
                           - 2.*sin_12*cos_12*sin_13*sin_23*cos_23*cos_delta)
        RSS_32 =  msq_31*cos_13*cos_13*sin_23*cos_23 \
                 - msq_21*(sin_23*cos_23*(cos_12*cos_12 - sin_12*sin_12*sin_13*sin_13) \
                              + sin_12*cos_12*sin_13*(cos_23*cos_23 - sin_23*sin_23)*cos_delta)
        RSS_13 = RSS_31
        RSS_23 = RSS_32
        RSS_33 =  msq_31*cos_13*cos_13*cos_23*cos_23 \
                 + msq_21*(cos_12*cos_12*sin_23*sin_23 + sin_12*sin_12*sin_13*sin_13*cos_23*cos_23 \
                           + 2.*sin_12*cos_12*sin_13*sin_23*cos_23*cos_delta)

        ISS_11 = 0
        ISS_21 = msq_31*sin_13*cos_13*sin_23*sin_delta - msq_21*sin_12*sin_12*sin_13*cos_13*sin_23*sin_delta
        ISS_31 = msq_31*sin_13*cos_13*cos_23*sin_delta - msq_21*sin_12*sin_12*sin_13*cos_13*cos_23*sin_delta
        ISS_12 = -ISS_21
        ISS_22 = 0
        ISS_32 = -msq_21*sin_12*cos_12*sin_13*sin_delta
        ISS_13 = -ISS_31
        ISS_23 = -ISS_32
        ISS_33 = 0

        SS = np.array([[complex(RSS_11,ISS_11), complex(RSS_12,ISS_12), complex(RSS_13,ISS_13)],
                       [complex(RSS_21,ISS_21), complex(RSS_22,ISS_22), complex(RSS_23,ISS_23)],
                       [complex(RSS_31,ISS_31), complex(RSS_32,ISS_32), complex(RSS_33,ISS_33)]])

        print self.calcMassMatrix() - SS

        return SS

    def __str__(self):
        """Pretty printout of mixing parameters."""
        
        s = "NeutrinoMixing: Sin^2(2*theta_12) = %g\n" \
            "                Sin^2(2*theta_13) = %g\n" \
            "                Sin^2(2*theta_23) = %g\n" \
            "                Delta msq_21      = %g\n" \
            "                Delta msq_31      = %g\n" \
            "                delta_cp          = %g"   \
            % (self.sinsq_2theta_12, self.sinsq_2theta_13, self.sinsq_2theta_23,
               self.msq_21, self.msq_31, self.delta)
        return s

def main():
    """
    Just create some instances and print out the matrices.
    """

    std_mixing = NeutrinoMixing()
    print std_mixing
    print ""

    mixing_matrix = std_mixing.calcMixingMatrix()
    print "Mixing Matrix:"
    print mixing_matrix
    print ""

    mass_matrix = std_mixing.calcMassMatrix()
    print "Mass Matrix:"
    print mass_matrix
    print ""

    print "Delta between new and old calculation:"
    std_mixing.calcOldStyleMassMatrix()

if __name__ == "__main__":
    main()

