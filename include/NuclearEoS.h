#ifndef __NUCLEAREOS_H__
#define __NUCLEAREOS_H__



#include "CUFLU.h"


#define NUC_TABLE_NVAR       16     // number of variables in the Nuclear EoS table
#ifdef  HELMHOLTZ_EOS
#define NUC_TABLE_NPTR       19     // number of table pointers to be sent to GPU
#define HELM_TABLE_NVAR      21     // number of variables in the Helmholtz EoS table
#else
#define NUC_TABLE_NPTR       10     // number of table pointers to be sent to GPU
#endif


// auxiliary array indices
#define NUC_AUX_ESHIFT        0     // AuxArray_Flt: energy_shift
#define NUC_AUX_DENS2CGS      1     // AuxArray_Flt: convert density    to cgs
#define NUC_AUX_PRES2CGS      2     // AuxArray_Flt: convert pressure   to cgs
#define NUC_AUX_VSQR2CGS      3     // AuxArray_Flt: convert velocity^2 to cgs
#define NUC_AUX_PRES2CODE     4     // AuxArray_Flt: convert pressure   to code unit
#define NUC_AUX_VSQR2CODE     5     // AuxArray_Flt: convert velocity^2 to code unit
#define NUC_AUX_KELVIN2MEV    6     // AuxArray_Flt: convert kelvin     to MeV
#define NUC_AUX_MEV2KELVIN    7     // AuxArray_Flt: convert MeV        to kelvin
#define NUC_AUX_M_kB          8     // AuxArray_Flt: mean molecular weight*atomic mass unit/
                                    //               Bolzmann constant*(UNIT_E/UNIT_M)
#ifdef HELMHOLTZ_EOS
#define NUC_AUX_DENS_TRANS    9     // AuxArray_Flt: transition density nuclear/Helmholtz EoS
#define NUC_AUX_DENS_STOP    10     // AuxArray_Flt: density lower limit for nuclear EoS
#define NUC_AUX_ALPHA        11     // AuxArray_Flt: decline parameter in transition region
#define NUC_AUX_CSHIFT       12     // AuxArray_Flt: energy shift at transition density
#endif

#define NUC_AUX_NRHO          0     // AuxArray_Int: nrho
#define NUC_AUX_NTORE         1     // AuxArray_Int: ntemp/neps
#define NUC_AUX_NYE           2     // AuxArray_Int: nye
#define NUC_AUX_NRHO_MODE     3     // AuxArray_Int: nrho_mode
#define NUC_AUX_NMODE         4     // AuxArray_Int: nmode
#define NUC_AUX_NYE_MODE      5     // AuxArray_Int: nye_mode
#define NUC_AUX_INT_AUX       6     // AuxArray_Int: interpolation scheme for auxiliary   table
#define NUC_AUX_INT_MAIN      7     // AuxArray_Int: interpolation scheme for Nuclear EoS table
#define NUC_AUX_HELM_I        8     // AuxArray_Int: Helmholtz EoS table row    size
#define NUC_AUX_HELM_J        9     // AuxArray_Int: Helmholtz EoS table column size
#define NUC_AUX_PROG_NBIN    10     // AuxArray_Int: progenitor model column size


// table indices
#define NUC_TAB_ALL           0     // alltables
#define NUC_TAB_ALL_MODE      1     // alltables_mode
#define NUC_TAB_RHO           2     // logrho
#define NUC_TAB_TORE          3     // logtemp/logenergy
#define NUC_TAB_YE            4     // yes
#define NUC_TAB_RHO_MODE      5     // logrho_mode
#define NUC_TAB_EORT_MODE     6     // logenergy_mode/logtemp_mode
#define NUC_TAB_ENTR_MODE     7     // entr_mode
#define NUC_TAB_PRES_MODE     8     // logprss_mode
#define NUC_TAB_YE_MODE       9     // yes_mode
#ifdef HELMHOLTZ_EOS
#define NUC_TABLE_HELM       10     // Helmholtz table
#define NUC_TABLE_HELM_DD    11     // derivative of density
#define NUC_TABLE_HELM_DT    12     // derivative of temperature
#define NUC_TABLE_HELM_DENS  13     // density
#define NUC_TABLE_HELM_TEMP  14     // temperature
#define NUC_TABLE_HELM_DIFF  15     // E_nuc - E_Helm - c_shift
#define NUC_TABLE_PROG_DENS  16     // progenitor density
#define NUC_TABLE_PROG_ABAR  17     // progenitor Abar
#define NUC_TABLE_PROG_ZBAR  18     // progenitor Zbar
#endif


// EoS modes
#define NUC_MODE_ENGY         0     // energy      mode
#define NUC_MODE_TEMP         1     // temperature mode
#define NUC_MODE_ENTR         2     // entropy     mode
#define NUC_MODE_PRES         3     // pressure    mode


// variable indices in the Nuclear EoS table and auxiliary table
#define NUC_VAR_IDX_PRES      0     // pressure
#define NUC_VAR_IDX_EORT      1     // internal energy (temperature-based table)
                                    // temperature     (energy-based      table)
#define NUC_VAR_IDX_ENTR      2     // entropy
#define NUC_VAR_IDX_MUNU      3     // mu_e - mu_n + mu_p
#define NUC_VAR_IDX_CSQR      4     // sound speed squared
#define NUC_VAR_IDX_MUHAT     5     // mu_n - mu_p
#define NUC_VAR_IDX_MUE       6     // electron chemical potential (includes rest mass)
#define NUC_VAR_IDX_MUP       7     // proton   chemical potential
#define NUC_VAR_IDX_MUN       8     // neutron  chemical potential
#define NUC_VAR_IDX_XA        9     // alpha particle        mass fraction
#define NUC_VAR_IDX_XH       10     // average heavy nucleus mass fraction
#define NUC_VAR_IDX_XN       11     // neutron               mass fraction
#define NUC_VAR_IDX_XP       12     // proton                mass fraction
#define NUC_VAR_IDX_ABAR     13     // average heavy nucleus mass number
#define NUC_VAR_IDX_ZBAR     14     // average heavy nucleus atomic number
#define NUC_VAR_IDX_GAMMA    15     // adiabatic index


// Tolerance for Newton-Raphson or bisection methods in temperature driver
#ifdef FLOAT8
const real Tolerance = 1e-10;
#else
const real Tolerance = 1e-6;
#endif


#endif // __NUCLEAREOS_H__
