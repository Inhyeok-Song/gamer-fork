#ifndef __HELMHOLTZEOS_H__
#define __HELMHOLTZEOS_H__



#include "Macro.h"
#include "PhysicalConstant.h"


// constants for Helmholtz EOS
const double FourThirdPi     = 4.0/3.0 * M_PI;
const double Kelvin2MeV      = Const_kB_eV * 1.0e-6;
const double Const_kB_NA     = Const_kB * Const_NA;                          // constant
const double Const_sigma     = 5.670367e-5;                                  // constant
const double Const_rad_a     = 4*Const_sigma / Const_c;                      // constant
const double Const_rad_ai3   = Const_rad_a / 3.0;                            // constant
const double Const_qe        = 4.803204673e-10;                              // constant
const double Const_esqu      = Const_qe * Const_qe;                          // constant
const double Const_sioncon   = ( Const_amu * Const_kB ) /
                               ( 2.0 * M_PI * Const_Planck * Const_Planck ); // constant

const double Const_a1        = -0.898004;  // constant a1
const double Const_b1        =  0.96786;   // constant b1
const double Const_c1        =  0.220703;  // constant c1
const double Const_d1        = -0.86097;   // constant d1
const double Const_e1        =  2.5269;    // constant e1
const double Const_a2        =  0.29561;   // constant a2
const double Const_b2        =  1.9885;    // constant b2
const double Const_c2        =  0.288675;  // constant c2

const double Const_tlo       = 3.0;        // lower limit of temperature
const double Const_thi       = 13.0;       // upper limit of temperature

const double Const_dlo       = -12.0;      // lower limit of density
const double Const_dhi       = 15.0;       // upper limit of density


// variable indices for the Helmholtz EoS
#define PSI0                  0     // psi0
#define DPSI0                 1     // dpsi0
#define DDPSI0                2     // ddpsi0
#define PSI1                  3     // psi1
#define DPSI1                 4     // dpsi1
#define DDPSI1                5     // ddpsi1
#define PSI2                  6     // psi2
#define DPSI2                 7     // dpsi2
#define DDPSI2                8     // ddpsi2
#define XPSI0                 9     // xpsi0
#define XDPSI0               10     // xdpsi0
#define XPSI1                11     // xpsi1
#define XDPSI1               12     // xdpsi1


#endif // #ifndef __HELMHOLTZEOS_H__
