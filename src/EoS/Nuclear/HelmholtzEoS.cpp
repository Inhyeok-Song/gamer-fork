#include "NuclearEoS.h"
#include "HelmholtzEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
#ifdef HELMHOLTZ_EOS



GPU_DEVICE static
double hermite_poly( const double z, const int TargetIdx );

GPU_DEVICE static
double biquintic_h5( const double *fi,  const double w0t,  const double w1t,
                     const double w2t,  const double w0mt, const double w1mt,
                     const double w2mt, const double w0d,  const double w1d,
                     const double w2d,  const double w0md, const double w1md,
                     const double w2md );

GPU_DEVICE static
double bicubic_h3( const double *fi,  const double w0t,  const double w1t,
                   const double w0mt, const double w1mt, const double w0d,
                   const double w1d,  const double w0md, const double w1md );

GPU_DEVICE static
double linterp_2D_table( const double x, const double y, const double *table,
                         const int nx, const int ny, const real *xt, const real *yt );

template <typename T>
GPU_DEVICE static
T interp_from_table( const int N, const T Table_x[], const T Table_y[], const T x );



#ifdef __CUDACC__

GPU_DEVICE static
void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               const int *TargetIdx, real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt );

GPU_DEVICE static
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );

#else

void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               const int *TargetIdx, real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt );

void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );

#endif // #ifdef __CUDACC__ ... else ...




//-----------------------------------------------------------------------------------------------
// Function    :  Helmholtz_eos
// Description :  Function to find thermodynamic variables by searching
//                a pre-calculated Helmholtz equation of state table
//
// Note        :  1. It will strictly return values in cgs or MeV
//                2. Five modes are supported
//                   --> Energy      mode (0)
//                       Temperature mode (1)
//                       Entropy     mode (2)
//                       Pressure    mode (3)
//                3. Out[] must have the size of at least NTarget+1:
//                   --> Out[NTarget] stores the internal energy or temperature either
//                       from the input value or the value found in the auxiliary nuclear EoS table
//
// Parameter   :  Out            : Output array
//                In             : Input array
//                                 --> In[0] = mass density        ( rho) in g/cm^3
//                                     In[1] = internal energy     ( eps) in cm^2/s^2   (keymode = 0/NUC_MODE_ENGY)
//                                           = Temperature         (temp) in MeV        (keymode = 1/NUC_MODE_TEMP)
//                                           = entropy             (entr) in kB/baryon  (keymode = 2/NUC_MODE_ENTR)
//                                           = pressure            (pres) in dyne/cm^2  (keymode = 3/NUC_MODE_PRES)
//                                     In[2] = Ye                  (  Ye) dimensionless
//                                     In[3] = average mass number (abar) dimensionless
//                NTarget        : Number of thermodynamic variables retrieved from the nuclear EoS table
//                TargetIdx      : Indices of thermodynamic variables to be returned
//
// Return      :  Out[]
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
void Helmholtz_eos( real *Out, const real *In, const int NTarget, const int *TargetIdx,
                    real Temp_InitGuess, const real c_shift, const int imax, const int jmax,
                    const double *helmholtz_table, const double *helmholtz_dd, const double *helmholtz_dt,
                    const double *helm_dens, const double *helm_temp, const double *helm_diff,
                    const double *prog_dens, const double *prog_abar, const double *prog_zbar,
                    const int prog_nbin, const real dens_trans, const real dens_stop, const real alpha,
                    const int nrho, const int ntoreps, const int nye, const real *alltables,
                    const real *logrho, const real *logtoreps, const real *yes,
                    const int IntScheme_Aux, const int IntScheme_Main,
                    const int keymode, int *keyerr, const real rfeps )
{

   *keyerr = 0;

         real   lr = LOG10( In[0] );

         double pres;
         double ener;
         double entr;
         double entr_kb;
         double sound;

         real   abar, zbar;
         real   mu_e, mu_p, mu_n, xh, xn, xp;

   const double tstp  = ( Const_thi - Const_tlo ) / ( jmax - 1 );
   const double tstpi = 1.0 / tstp;
   const double dstp  = ( Const_dhi - Const_dlo ) / ( imax - 1 );
   const double dstpi = 1.0 / dstp;

   const double Kelvin2MeV = Const_kB_eV * 1.0e-6;
   const double MeV2Kelvin = 1.0 / Kelvin2MeV;
   const double Dens_CGS   = In[0];
   const double Ye         = In[2];
         double Temp_MeV   = NULL_REAL;
   if ( keymode == NUC_MODE_TEMP )   Temp_MeV = In[1];
   else                              Temp_MeV = Temp_InitGuess;


// enter the table with Ye*Dens_CGS
   const double YeDens    = Ye*Dens_CGS;
   const double _Dens_CGS = 1.0/Dens_CGS;

// density difference
   int    iat = int(  ( LOG10(YeDens) - Const_dlo ) * dstpi  );
          iat = MAX( 0, MIN(iat, imax-2) );

   double xd  = MAX(  ( YeDens - helm_dens[iat] ) * helmholtz_dd[iat*5 + 2], 0.0  );
   double mxd = 1.0 - xd;

// the six density basis functions
   double si0d    =  hermite_poly( xd,  PSI0   );
   double si1d    =  hermite_poly( xd,  PSI1   ) * helmholtz_dd[iat*5 + 0];
   double si2d    =  hermite_poly( xd,  PSI2   ) * helmholtz_dd[iat*5 + 1];

   double si0md   =  hermite_poly( mxd, PSI0   );
   double si1md   = -hermite_poly( mxd, PSI1   ) * helmholtz_dd[iat*5 + 0];
   double si2md   =  hermite_poly( mxd, PSI2   ) * helmholtz_dd[iat*5 + 1];

// derivatives of the weight functions
   double dsi0d   =  hermite_poly( xd,  DPSI0  ) * helmholtz_dd[iat*5 + 2];
   double dsi1d   =  hermite_poly( xd,  DPSI1  );
   double dsi2d   =  hermite_poly( xd,  DPSI2  ) * helmholtz_dd[iat*5 + 0];

   double dsi0md  = -hermite_poly( mxd, DPSI0  ) * helmholtz_dd[iat*5 + 2];
   double dsi1md  =  hermite_poly( mxd, DPSI1  );
   double dsi2md  = -hermite_poly( mxd, DPSI2  ) * helmholtz_dd[iat*5 + 0];

// now get the pressure derivative with density, chemical potential, and
// electron positron number densities
// get the interpolation weight functions
   double xsi0d   =  hermite_poly( xd,  XPSI0  );
   double xsi1d   =  hermite_poly( xd,  XPSI1  ) * helmholtz_dd[iat*5 + 0];

   double xsi0md  =  hermite_poly( mxd, XPSI0  );
   double xsi1md  = -hermite_poly( mxd, XPSI1  ) * helmholtz_dd[iat*5 + 0];

// derivatives of weight functions
   double xdsi0d  =  hermite_poly( xd,  XDPSI0 ) * helmholtz_dd[iat*5 + 2];
   double xdsi1d  =  hermite_poly( xd,  XDPSI1 );

   double xdsi0md = -hermite_poly( mxd, XDPSI0 ) * helmholtz_dd[iat*5 + 2];
   double xdsi1md =  hermite_poly( mxd, XDPSI1 );


   const int    itmax = 20;
         double dvardlt;
         double var0, var;  // temp vars for finding value
         double lt0, lt1;   // temp vars for temperature
   lt0 = log10( Temp_MeV );
   lt1 = lt0;

   int jmax_HELM = jmax*HELM_TABLE_NVAR;
   int imax_HELM = imax*HELM_TABLE_NVAR;

   switch ( keymode )
   {
      case NUC_MODE_ENGY : var0 = LOG10( In[1] );           break;
      case NUC_MODE_PRES : var0 = LOG10( In[1] );           break;
      case NUC_MODE_ENTR : var0 = In[1] * ( Const_kB_NA );  break;
   }


   int it = ( keymode != NUC_MODE_TEMP ) ? 0 : itmax-1;
   while ( it < itmax )
   {
             lt0       = lt1;
             Temp_MeV  = POW( 10.0, lt0 );
      double Temp_Kelv = Temp_MeV * MeV2Kelvin;

//    calculate Abar, Zbar from nuclear eos table, if possible
      if (  ( (real)lr  > logrho   [0]  &&  (real)lr  < logrho   [nrho-1   ] )  &&
            ( (real)lt0 > logtoreps[0]  &&  (real)lt0 < logtoreps[ntoreps-1] )  &&
            ( (real)Ye  > yes      [0]  &&  (real)Ye  < yes      [nye-1    ] )  )
      {
         const int  NTarget_Aux = 2;
               real Out_Aux[NTarget_Aux];
               int  TargetIdx_Aux[NTarget_Aux] = { NUC_VAR_IDX_ABAR, NUC_VAR_IDX_ZBAR };

         if ( IntScheme_Main == NUC_INT_LINEAR )
         {
            nuc_eos_C_linterp_some( (real)lr, (real)lt0, (real)Ye, TargetIdx_Aux, Out_Aux, alltables,
                                    nrho, ntoreps, nye, NTarget_Aux, logrho, logtoreps, yes );
         }

         else
         {
            nuc_eos_C_cubinterp_some( (real)lr, (real)lt0, (real)Ye, TargetIdx_Aux, Out_Aux, alltables,
                                      nrho, ntoreps, nye, NTarget_Aux, logrho, logtoreps, yes );
         }

         abar = Out_Aux[0];
         zbar = Out_Aux[1];
      }

//    calculate Abar, Abar from the progenitor model
      else {
         abar = interp_from_table( prog_nbin, prog_dens, prog_abar, -Dens_CGS );
         // zbar = interp_from_table( prog_nbin, prog_dens, prog_zbar, -Dens );
         zbar = abar * Ye;
         mu_e = NULL_REAL;
         mu_p = NULL_REAL;
         mu_n = NULL_REAL;
         xh   = NULL_REAL;
         xn   = NULL_REAL;
         xp   = NULL_REAL;
      }


//    radiation section
      const double ytot1      = 1.0 / abar;
      const double _Temp_Kelv = 1.0 / Temp_Kelv;

      const double kt        = Temp_Kelv * Const_kB;
      const double kt_inv    = 1.0 / kt;

      const double prad    = Const_rad_ai3 * SQR(  SQR ( Temp_Kelv )  );
      const double dpraddd = 0.0;
      const double dpraddt = 4.0 * prad * _Temp_Kelv;

      const double erad    = 3.0 * prad * _Dens_CGS;
      const double deraddt = 3.0 * dpraddt * _Dens_CGS;

      const double srad    = ( prad*_Dens_CGS + erad ) * _Temp_Kelv;
      const double dsraddt = ( dpraddt*_Dens_CGS + deraddt - srad ) * _Temp_Kelv;


//    ion section
      const double xni     = Const_NA * ytot1 * Dens_CGS;
      const double dxnidd  = Const_NA * ytot1;
      const double dxnida  = -xni * ytot1;

      const double pion    = xni * kt;
      const double dpiondd = dxnidd * kt;
      const double dpiondt = xni * Const_kB;
      const double dpionda = dxnida * kt;
      const double dpiondz = 0.0;

      const double eion    = 1.5 * pion * _Dens_CGS;
      const double deiondt = 1.5 * dpiondt * _Dens_CGS;


//    sackur-tetrode equation for the ion entropy of
//    a single ideal gas characterized by abar
      double x = SQR( abar ) * sqrt( abar ) * _Dens_CGS / Const_NA;
      double s = Const_sioncon * Temp_Kelv;
      double z = x * s * sqrt( s );
      double y = log( z );

      const double sion    = ( pion*_Dens_CGS + eion ) * _Temp_Kelv + Const_kB_NA * ytot1 * y;
      const double dsiondt = ( dpiondt*_Dens_CGS + deiondt ) * _Temp_Kelv - ( pion*_Dens_CGS + eion ) * SQR( _Temp_Kelv )
                           + 1.5 * Const_kB_NA * _Temp_Kelv * ytot1;

//    temperature difference
      int jat = int(  ( LOG10(Temp_Kelv) - Const_tlo ) * tstpi  );
          jat = MAX( 0, MIN(jat, jmax-2) );

      double fi[36];
      int arr_idx_9[9] = { 0, 2, 4, 1, 3, 5, 6, 7, 8 };

      for (int k=0; k<9; k++) {  const int kidx = arr_idx_9[k];
      for (int j=0; j<2; j++) {  const int jidx = ((jat+j)*HELM_TABLE_NVAR);
      for (int i=0; i<2; i++) {
         fi[4*k + 2*j + i] = helmholtz_table[ ((iat+i)*jmax_HELM) + jidx + kidx ];
      }}}

//    various differences
      double xt  = MAX(  ( Temp_Kelv - helm_temp[jat] ) * helmholtz_dt[jat*5 + 2], 0.0  );
      double mxt = 1.0 - xt;

//    the six temperature basis functions
      double si0t    =  hermite_poly( xt,  PSI0   );
      double si1t    =  hermite_poly( xt,  PSI1   ) * helmholtz_dt[jat*5 + 0];
      double si2t    =  hermite_poly( xt,  PSI2   ) * helmholtz_dt[jat*5 + 1];

      double si0mt   =  hermite_poly( mxt, PSI0   );
      double si1mt   = -hermite_poly( mxt, PSI1   ) * helmholtz_dt[jat*5 + 0];
      double si2mt   =  hermite_poly( mxt, PSI2   ) * helmholtz_dt[jat*5 + 1];

//    derivatives of the weight functions
      double dsi0t   =  hermite_poly( xt,  DPSI0  ) * helmholtz_dt[jat*5 + 2];
      double dsi1t   =  hermite_poly( xt,  DPSI1  );
      double dsi2t   =  hermite_poly( xt,  DPSI2  ) * helmholtz_dt[jat*5 + 0];

      double dsi0mt  = -hermite_poly( mxt, DPSI0  ) * helmholtz_dt[jat*5 + 2];
      double dsi1mt  =  hermite_poly( mxt, DPSI1  );
      double dsi2mt  = -hermite_poly( mxt, DPSI2  ) * helmholtz_dt[jat*5 + 0];

//    second derivatives of the weight functions
      double ddsi0t  =  hermite_poly( xt,  DDPSI0 ) * helmholtz_dt[jat*5 + 3];
      double ddsi1t  =  hermite_poly( xt,  DDPSI1 ) * helmholtz_dt[jat*5 + 2];
      double ddsi2t  =  hermite_poly( xt,  DDPSI2 );

      double ddsi0mt =  hermite_poly( mxt, DDPSI0 ) * helmholtz_dt[jat*5 + 3];
      double ddsi1mt = -hermite_poly( mxt, DDPSI1 ) * helmholtz_dt[jat*5 + 2];
      double ddsi2mt =  hermite_poly( mxt, DDPSI2 );

//    the free energy
      double free  = biquintic_h5( fi, si0t, si1t, si2t, si0mt, si1mt, si2mt,
                                   si0d, si1d, si2d, si0md, si1md, si2md );

//    derivative with respect to density
      double df_d  = biquintic_h5( fi, si0t, si1t,  si2t, si0mt, si1mt, si2mt,
                                   dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md );

//    derivative with respect to temperature
      double df_t  = biquintic_h5( fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
                                   si0d, si1d, si2d, si0md, si1md, si2md );


//    derivative with respect to temperature**2
      double df_tt = biquintic_h5( fi, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
                                   si0d, si1d, si2d, si0md, si1md, si2md );

//    derivative with respect to temperature and density
      double df_dt = biquintic_h5( fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
                                   dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md );

//    now get the pressure derivative with density, chemical potential, and
//    electron positron number densities
//    get the interpolation weight functions
      double xsi0t   =  hermite_poly( xt,  XPSI0  );
      double xsi1t   =  hermite_poly( xt,  XPSI1  ) * helmholtz_dt[jat*5 + 0];

      double xsi0mt  =  hermite_poly( mxt, XPSI0  );
      double xsi1mt  = -hermite_poly( mxt, XPSI1  ) * helmholtz_dt[jat*5 + 0];

//    look in the pressure derivative only once
      int arr_idx_4[4] = { 9, 11, 10, 12 };

      for (int k=0; k<4; k++) {  const int kidx = arr_idx_4[k];
      for (int j=0; j<2; j++) {  const int jidx = ((jat+j)*HELM_TABLE_NVAR);
      for (int i=0; i<2; i++) {
         fi[4*k + 2*j + i] = helmholtz_table[ ((iat+i)*jmax_HELM) + jidx + kidx ];
      }}}

//    pressure derivative with density
      double dpepdd = bicubic_h3( fi, xsi0t, xsi1t, xsi0mt, xsi1mt,
                                  xsi0d, xsi1d, xsi0md, xsi1md );
             dpepdd = MAX( Ye * dpepdd, Tolerance );


//    look in the electron chemical potential table only once
      arr_idx_4[0] = 13;  arr_idx_4[1] = 15;  arr_idx_4[2] = 14;  arr_idx_4[3] = 16;

      for (int k=0; k<4; k++) {  const int kidx = arr_idx_4[k];
      for (int j=0; j<2; j++) {  const int jidx = ((jat+j)*HELM_TABLE_NVAR);
      for (int i=0; i<2; i++) {
         fi[4*k + 2*j + i] = helmholtz_table[ ((iat+i)*jmax_HELM) + jidx + kidx ];
      }}}

//    electron chemical potnetial
      const double etaele = bicubic_h3( fi, si0t, si1t, si0mt, si1mt,
                                        si0d, si1d, si0md, si1md );

//    derivative with respect to density
      x = bicubic_h3( fi, xsi0t, xsi1t, xsi0mt, xsi1mt,
                      xdsi0d, xdsi1d, xdsi0md, xdsi1md );


//    look in the number density table only once
      arr_idx_4[0] = 17;  arr_idx_4[1] = 19;  arr_idx_4[2] = 18;  arr_idx_4[3] = 20;

      for (int k=0; k<4; k++) {  const int kidx = arr_idx_4[k];
      for (int j=0; j<2; j++) {  const int jidx = ((jat+j)*HELM_TABLE_NVAR);
      for (int i=0; i<2; i++) {
         fi[4*k + 2*j + i] = helmholtz_table[ ((iat+i)*jmax_HELM) + jidx + kidx ];
      }}}

//    the desired electron-positron thermodynamic quantities

//    dpepdd at high temperatures and low density is below the
//    floating point limit of the subtraction of two large terms.
//    since dpresdd doesn't enter the maxwell relations at all, use the
//    bicubic interpolation done above instead of the formally correct expression
             x      = SQR( YeDens );
      double pele   = x * df_d;
      double dpepdt = x * df_dt;
             s      = dpepdd/Ye - 2.0 * YeDens * df_d;

             x      = SQR( Ye );
      double sele   = -df_t * Ye;
      double dsepdt = -df_tt * Ye;

      double eele   = Ye*free + Temp_Kelv * sele;
      double deepdt = Temp_Kelv * dsepdt;


//    coulomb section:
//
//    uniform background corrections only
//    from yakovlev & shalybkov 1989
//    lami is the average ion seperation
//    plasg is the plasma coupling parameter
      const double third = 1.0/3.0;

                   z     = 4.0/3.0 * M_PI;
                   s     = z * xni;
      const double dsdd  = z * dxnidd;
      const double dsda  = z * dxnida;

      const double lami     = 1.0 / pow( s, third );
      const double inv_lami = 1.0 / lami;
                   z        = -third * lami;
      const double lamidd   = z * dsdd/s;
      const double lamida   = z * dsda/s;

      const double plasg   = SQR( zbar )*Const_esqu*kt_inv*inv_lami;
                   z       = -plasg * inv_lami;
      const double plasgdd = z * lamidd;
      const double plasgda = z * lamida;
      const double plasgdt = -plasg*kt_inv * Const_kB;
      const double plasgdz = 2.0 * plasg/zbar;

      double pcoul    = 0.0;  double dpcouldd = 0.0;  double dpcouldt = 0.0;  double dpcoulda = 0.0;
      double dpcouldz = 0.0;  double ecoul    = 0.0;  double decouldd = 0.0;  double decouldt = 0.0;
      double decoulda = 0.0;  double decouldz = 0.0;  double scoul    = 0.0;  double dscouldt = 0.0;
      double dscouldz = 0.0;

      if ( plasg >= 1.0 ) {
         x        = pow( plasg, 0.25 );
         y        = Const_kB_NA * ytot1;
         ecoul    = y * Temp_Kelv * ( Const_a1 * plasg + Const_b1 * x + Const_c1 / x + Const_d1 );
         pcoul    = third * Dens_CGS * ecoul;
         scoul    = -y * (  3.0 * Const_b1 * x - 5.0 * Const_c1 / x + Const_d1 * ( log(plasg) - 1.0 ) - Const_e1  );

         y        = Const_NA * ytot1 * kt * (  Const_a1 + 0.25 / plasg * ( Const_b1 * x - Const_c1 / x )  );
         decouldd = y * plasgdd;
         decouldt = y * plasgdt + ecoul * _Temp_Kelv;
         // decoulda = y * plasgda - ecoul/abar;
         // decouldz = y * plasgdz;

         y        = third * Dens_CGS;
         dpcouldd = third * ecoul + y * decouldd;
         dpcouldt = y * decouldt;
         // dpcoulda = y * decoulda;
         // dpcouldz = y * decouldz;

         y        = - Const_kB_NA * ytot1 / plasg * ( 0.75 * Const_b1 * x + 1.25 * Const_c1 / x + Const_d1 );
         dscouldz = y * plasgdz;
         dscouldt = y * plasgdt;
      }

      else if ( plasg < 1.0 ) {
         x        = plasg*sqrt( plasg );
         y        = pow( plasg, Const_b2 );
         z        = Const_c2 * x - third * Const_a2 * y;
         pcoul    = -pion * z;
         ecoul    = 3.0 * pcoul * _Dens_CGS;
         scoul    = -Const_kB_NA * ytot1 *(  Const_c2 * x - Const_a2 * ( Const_b2 - 1.0 ) / Const_b2 * y  );

         s        = 1.5 * Const_c2 * x / plasg - third * Const_a2 * Const_b2 * y / plasg;
         dpcouldd = -dpiondd * z - pion * s * plasgdd;
         dpcouldt = -dpiondt * z - pion * s * plasgdt;
         // dpcoulda = -dpionda * z - pion * s * plasgda;
         // dpcouldz = -dpiondz * z - pion * s * plasgdz;

         s        = 3.0 * _Dens_CGS;
         decouldd = s * dpcouldd - ecoul * _Dens_CGS;
         decouldt = s * dpcouldt;
         // decoulda = s * dpcoulda;
         // decouldz = s * dpcouldz;

         s        = - Const_kB_NA * ytot1 / plasg  * ( 1.5 * Const_c2 * x - Const_a2 * ( Const_b2 - 1.0 ) * y );
         dscouldz = s * plasgdz;
      }

//    bomb proof
      x = prad + pion + pele + pcoul;
      y = erad + eion + eele + ecoul;
      z = srad + sion + sele + scoul;

      if ( x <= 0.0  ||  y <= 0.0 ) {
         pcoul    = 0.0;  dpcouldd = 0.0;  dpcouldt = 0.0;  dpcoulda = 0.0;
         dpcouldz = 0.0;  ecoul    = 0.0;  decouldd = 0.0;  decouldt = 0.0;
         decoulda = 0.0;  decouldz = 0.0;  scoul    = 0.0;  dscouldz = 0.0;
      }


//    sum all the gas components
      const double pgas = pion + pele + pcoul;
      const double egas = eion + eele + ecoul;
      const double sgas = sion + sele + scoul;

      const double dpgasdd = dpiondd + dpepdd + dpcouldd;
      const double dpgasdt = dpiondt + dpepdt + dpcouldt;
      const double degasdt = deiondt + deepdt + decouldt;
      const double dsgasdt = dsiondt + dsepdt + dscouldt;

//    add in radiation to get the total
      pres    = prad + pgas;
      ener    = erad + egas;
      entr    = srad + sgas;
      entr_kb = entr / ( Const_kB_NA );

      const double dpresdd = dpraddd + dpgasdd;
      const double dpresdt = dpraddt + dpgasdt;
      const double denerdt = deraddt + degasdt;
      const double dentrdt = dsraddt + dsgasdt;


//    for the gas
//    the temperature and density exponents (c&g 9.81 9.82)
//    the specific heat at constant volume (c&g 9.92)
//    the third adiabatic exponent (c&g 9.93)
//    the first adiabatic exponent (c&g 9.97)
//    the second adiabatic exponent (c&g 9.105)
//    the specific heat at constant pressure (c&g 9.98)
//    and relativistic formula for the sound speed (c&g 14.29)
//    double zz        = pgas*_Dens_CGS;
//    double zzi       = Dens_CGS/pgas;
//    double chit_gas  = Temp/pgas * dpgasdt;
//    double chid_gas  = dpgasdd*zzi;
//    double cv_gas    = degasdt;
//                 x         = zz * chit_gas/(Temp * cv_gas);
//    double gam1_gas  = chit_gas*x + chid_gas;
//                 z         = 1.0 + ( egas + c*c ) * zzi;
//    double sound_gas = c * sqrt(gam1_gas/z);

//    // for the totals
      const double zz    = pres * _Dens_CGS;
      const double zzi   = Dens_CGS / pres;
      const double chit  = Temp_Kelv / pres * dpresdt;
      const double chid  = dpresdd * zzi;
            double cv    = denerdt;
                   x     = zz * chit / ( Temp_Kelv * cv );
      const double gam1  = chit * x + chid;
                   z     = 1.0 + ( ener + SQR(Const_c) ) * zzi;
                   sound = Const_c * SQRT( gam1 / z );


//    shift the energy
      if ( c_shift != NULL_REAL )   // do not shift the energy when initializes (read table)
      {
         if (  Dens_CGS <= dens_trans  &&  Dens_CGS > dens_stop  )
         {
            const double lt = log10( Temp_MeV );
            const double diff_shift = linterp_2D_table( lt, Ye, helm_diff, ntoreps, nye, logtoreps, yes );
            ener = ener + diff_shift
                 * exp(  ( log10(Dens_CGS) - log10(dens_trans) ) / alpha  ) + c_shift;
         }

         else if ( Dens_CGS <= dens_stop )
            ener = ener + c_shift;
      }


//    different modes
      switch ( keymode )
      {
         case NUC_MODE_ENGY : var = LOG10( ener );  dvardlt = denerdt * Temp_Kelv / ener;  break;
         case NUC_MODE_PRES : var = LOG10( pres );  dvardlt = dpresdt * Temp_Kelv / pres;  break;
         case NUC_MODE_ENTR : var = entr;           dvardlt = dentrdt * Temp_Kelv;         break;
      }


      if ( keymode != NUC_MODE_TEMP )
      {
         if ( FABS( var - var0 ) < rfeps*FABS( var0 ) )
            break;
         else
            lt1 = lt0 - ( var - var0 ) / dvardlt;
//       don't allow the temperature to change by more than an order of magnitude
         if      ( lt1 > lt0 + 1.0 )
            lt1 = lt0 + 1.0;
         else if ( lt1 < lt0 - 1.0 )
            lt1 = lt0 - 1.0;
      }

      it++;
   }


// calculate other variables from nuclear eos table, if possible
   if (  ( (real)lr  > logrho   [0]  &&  (real)lr  < logrho   [nrho-1   ] )  &&
         ( (real)lt0 > logtoreps[0]  &&  (real)lt0 < logtoreps[ntoreps-1] )  &&
         ( (real)Ye  > yes      [0]  &&  (real)Ye  < yes      [nye-1    ] )  )
   {
      if ( IntScheme_Main == NUC_INT_LINEAR )
      {
         nuc_eos_C_linterp_some( (real)lr, (real)lt0, (real)Ye, TargetIdx, Out, alltables,
                                 nrho, ntoreps, nye, NTarget, logrho, logtoreps, yes );
      }

      else
      {
         nuc_eos_C_cubinterp_some( (real)lr, (real)lt0, (real)Ye, TargetIdx, Out, alltables,
                                    nrho, ntoreps, nye, NTarget, logrho, logtoreps, yes );
      }

   }


// output variables
   for (int i=0; i<NTarget; i++)
   {
      switch ( TargetIdx[i] )
      {
         case NUC_VAR_IDX_PRES :
            Out[i] = pres;
            break;

         case NUC_VAR_IDX_EORT :
            Out[i] = ener;
            break;

         case NUC_VAR_IDX_ENTR :
            Out[i] = entr_kb;
            break;

         case NUC_VAR_IDX_CSQR :
            Out[i] = sound * sound;
            break;

         case NUC_VAR_IDX_ABAR :
            Out[i] = abar;
            break;

         case NUC_VAR_IDX_ZBAR :
            Out[i] = zbar;
            break;

//       unsupported output variables
         default :
            if ( !(
                   ( (real)lr  > logrho   [0]  &&  (real)lr  < logrho   [nrho-1   ] )  &&
                   ( (real)lt0 > logtoreps[0]  &&  (real)lt0 < logtoreps[ntoreps-1] )  &&
                   ( (real)Ye  > yes      [0]  &&  (real)Ye  < yes      [nye-1    ] )
                  ) )
            {  *keyerr = 203;  return;  }
      }
   }



// store the temperature in Out[NTarget]
   Out[NTarget] = Temp_MeV;


} // FUNCTION : Helmholtz_eos



//-----------------------------------------------------------------------------------------------
// Function    :  hermite_poly
// Description :  quintic/cubic hermite basis statement functions
//
// Note        :
//
// Parameter   :
//
// Return      :
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
double hermite_poly( const double z, const int TargetIdx )
{

   double res = 0.0;
   switch ( TargetIdx )
   {
   case PSI0   :
      res = CUBE( z ) * ( z * (-6.0 * z + 15.0) - 10.0 ) + 1.0;
      break;

   case DPSI0  :
      res = SQR( z ) * ( z * (-30.0 * z + 60.0 ) - 30.0 );
      break;

   case DDPSI0 :
      res = z * ( z * ( -120.0*z + 180.0 ) -60.0 );
      break;

   case PSI1:
      res = z * ( SQR( z ) * ( z * ( -3.0*z + 8.0 ) - 6.0 ) + 1.0 );
      break;

   case DPSI1  :
      res = SQR( z ) * ( z * ( -15.0*z + 32.0 ) - 18.0 ) + 1.0;
      break;

   case DDPSI1 :
      res = z * ( z * (-60.0*z + 96.0 ) - 36.0 );
      break;

   case PSI2   :
      res = 0.5 * SQR( z ) * ( z * ( z * ( -z + 3.0 ) - 3.0 ) + 1.0 );
      break;

   case DPSI2  :
      res = 0.5 * z * ( z * ( z * ( -5.0*z + 12.0 ) - 9.0 ) + 2.0 );
      break;

   case DDPSI2 :
      res = 0.5 * ( z * ( z * ( -20.0*z + 36.0 ) - 18.0 ) + 2.0 );
      break;

   case XPSI0  :
      res = SQR( z ) * ( 2.0*z - 3.0 ) + 1.0;
      break;

   case XDPSI0 :
      res = z * ( 6.0*z - 6.0 );
      break;

   case XPSI1  :
      res = z * ( z * ( z - 2.0 ) + 1.0 );
      break;

   case XDPSI1 :
      res = z * ( 3.0*z - 4.0 ) + 1.0;
      break;
   }

   return res;

} // FUNCTION : hermite_poly



//-----------------------------------------------------------------------------------------------
// Function    :  biquintic hermite polynomial statement function
// Description :  Interpolating cubic Hermite polynomial
//
// Note        :
//
// Parameter   :
//
// Return      :
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
double biquintic_h5( const double *fi,  const double w0t,  const double w1t,
                     const double w2t,  const double w0mt, const double w1mt,
                     const double w2mt, const double w0d,  const double w1d,
                     const double w2d,  const double w0md, const double w1md,
                     const double w2md )
{

   double res = 0.0;

   res = fi[0 ] * w0d * w0t   + fi[1 ] * w0md * w0t
       + fi[2 ] * w0d * w0mt  + fi[3 ] * w0md * w0mt
       + fi[4 ] * w0d * w1t   + fi[5 ] * w0md * w1t
       + fi[6 ] * w0d * w1mt  + fi[7 ] * w0md * w1mt
       + fi[8 ] * w0d * w2t   + fi[9 ] * w0md * w2t
       + fi[10] * w0d * w2mt  + fi[11] * w0md * w2mt
       + fi[12] * w1d * w0t   + fi[13] * w1md * w0t
       + fi[14] * w1d * w0mt  + fi[15] * w1md * w0mt
       + fi[16] * w2d * w0t   + fi[17] * w2md * w0t
       + fi[18] * w2d * w0mt  + fi[19] * w2md * w0mt
       + fi[20] * w1d * w1t   + fi[21] * w1md * w1t
       + fi[22] * w1d * w1mt  + fi[23] * w1md * w1mt
       + fi[24] * w2d * w1t   + fi[25] * w2md * w1t
       + fi[26] * w2d * w1mt  + fi[27] * w2md * w1mt
       + fi[28] * w1d * w2t   + fi[29] * w1md * w2t
       + fi[30] * w1d * w2mt  + fi[31] * w1md * w2mt
       + fi[32] * w2d * w2t   + fi[33] * w2md * w2t
       + fi[34] * w2d * w2mt  + fi[35] * w2md * w2mt;


   return res;

} // FUNCTION : biquintic_h5



//-----------------------------------------------------------------------------------------------
// Function    :  bicubic hermite polynomial statement function
// Description :  Find biquintic hermite polynomial statement function
//
// Note        :
//
// Parameter   :
//
// Return      :
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
double bicubic_h3( const double *fi,  const double w0t,  const double w1t,
                   const double w0mt, const double w1mt, const double w0d,
                   const double w1d,  const double w0md, const double w1md )
{

    double res = 0.0;
    const double wd[2]  = { w0d,  w1d  };
    const double wmd[2] = { w0md, w1md };
    const double wt[2]  = { w0t,  w1t  };
    const double wmt[2] = { w0mt, w1mt };

    int idx = 0;
    for (int i=0; i<2; i++)
    for (int j=0; j<2; j++) {{
      res += fi[idx++] * wd [i] * wt [j];  // fi[x] * wd  * wt
      res += fi[idx++] * wmd[i] * wt [j];  // fi[x] * wmd * wt
      res += fi[idx++] * wd [i] * wmt[j]; // fi[x] * wd  * wmt
      res += fi[idx++] * wmd[i] * wmt[j]; // fi[x] * wmd * wmt
   }}


   return res;

} // FUNCTION : bicubic_h3



//-----------------------------------------------------------------------------------------------
// Function    :  linterp_2D_table
// Description :
//
// Note        :
//
// Parameter   :
//
// Return      :
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
double linterp_2D_table( const double x, const double y, const double *table,
                         const int nx, const int ny, const real *xt, const real *yt )
{

// helper variables
   int  ix, iy;
   double a[4], fh[4];


// determine spacing parameters of equidistant (!!!) table
   const double dx = ( xt[nx-1] - xt[0] ) / (double)(nx-1);
   const double dy = ( yt[ny-1] - yt[0] ) / (double)(ny-1);

   const double dxi   = (double)1.0 / dx;
   const double dyi   = (double)1.0 / dy;

   const double dxyi  = dxi*dyi;


// determine location in table
   ix = 1 + (int)( ( x - xt[0] )*dxi );
   iy = 1 + (int)( ( y - yt[0] )*dyi );

   ix = MAX(  1, MIN( ix, nx-1 )  );
   iy = MAX(  1, MIN( iy, ny-1 )  );


// set up aux vars for interpolation
   const double delx = xt[ix] - x;
   const double dely = yt[iy] - y;


   const int idx  = ix + nx*iy;

//    set up aux vars for interpolation assuming array ordering (ix, iy)
   fh[0] = table[ idx          ]; // ( ix  , iy   )
   fh[1] = table[ idx - 1      ]; // ( ix-1, iy   )
   fh[2] = table[ idx     - nx ]; // ( ix  , iy-1 )
   fh[3] = table[ idx - 1 - nx ]; // ( ix-1, iy-1 )

   a[0] = fh[0];
   a[1] = dxi  * ( fh[1] - fh[0] );
   a[2] = dyi  * ( fh[2] - fh[0] );
   a[3] = dxyi * ( fh[0] - fh[1] - fh[2] + fh [3] );

   double e_shift = a[0]
                  + a[1] * delx
                  + a[2] * dely
                  + a[3] * delx*dely;


   return e_shift;
} // FUNCTION : linterp_2D_table



//-------------------------------------------------------------------------------------------------------
// Function    :  interp_from_table
// Description :  Assuming y=y(x), return the interpolated value of y for a given point x
//
// Note        :  1. Interpolation table Table_x must be sorted into ascending numerical order in advance
//                2. Target point x must lie in the range Table_x[0] <= x < Table_x[N-1]
//                   --> Otherwise the function returns NULL_REAL
//                3. Currently the function only supports linear interpolation
//                4. Overloaded with different types
//                5. Explicit template instantiation is put in the end of this file
//
// Parameter   :  N        : Number of elements in the interpolation tables Table_x and Table_y
//                           --> Must be >= 2
//                Table_x  : Interpolation table x
//                Table_y  : Interpolation table y
//                x        : Target point x for interpolation
//
// Return      :  y(x)      if x lies in the range Table_x[0] <= x < Table_x[N-1]
//                NULL_REAL if x lies outside the above range
//-------------------------------------------------------------------------------------------------------
template <typename T>
static GPU_DEVICE
T interp_from_table(const int N, const T Table_x[], const T Table_y[], const T x)
{

// initial check
// #  ifdef GAMER_DEBUG
//    if ( N <= 1 )           Aux_Error( ERROR_INFO, "incorrect input parameter \"N (%d) <= 1\" !!\n", N );
//    if ( Table_x == NULL )  Aux_Error( ERROR_INFO, "Table_x == NULL !!\n" );
//    if ( Table_y == NULL )  Aux_Error( ERROR_INFO, "Table_y == NULL !!\n" );
// #  endif


// check whether the target x lies within the accepted range
   if ( x < Table_x[0]  ||  x >= Table_x[N-1] )    return NULL_REAL;


// binary search
   int    IdxL, IdxR;
   T      xL, xR, yL, yR, y;
   int    Min = 0;
   int    Max = N-1;

   if ( x <  Table_x[Min] )   IdxL = Min-1;
   if ( x >= Table_x[Max] )   IdxL = Max;

   if (  IdxL != Min-1  ||  IdxL != Max  ) {

      IdxL = -2;

      while (  ( IdxL=(Min+Max)/2 ) != Min  )
      {
         if   ( Table_x[IdxL] > x )  Max = IdxL;
         else                        Min = IdxL;
      }


// check whether the found array index is correct
// #     ifdef GAMER_DEBUG
//       if ( IdxL < Min  ||  IdxL >= Max )
//          Aux_Error( ERROR_INFO, "incorrect output index (IdxL %d, Min %d, Max%d) !!\n", IdxL, Min, Max );

//       if (  Table_x[IdxL] > x  ||  Table_x[IdxL+1] <= x )
//          Aux_Error( ERROR_INFO, "incorrect output index (IdxL %d, ValueL %14.7e, ValueR %14.7e, x %14.7e) !!\n",
//                   IdxL, Table_x[IdxL], Table_x[IdxL+1], x );
// #     endif
   }


// #  ifdef GAMER_DEBUG
//    if ( IdxL < 0  ||  IdxL >= N-1 )
//       Aux_Error( ERROR_INFO, "IdxL (%d) lies outside the accepted range [%d ... %d] !!\n", IdxL, 0, N-2 );
// #  endif


   IdxR = IdxL + 1;
   xL   = Table_x[IdxL];
   xR   = Table_x[IdxR];
   yL   = Table_y[IdxL];
   yR   = Table_y[IdxR];


// linear interpolation
   y = yL + (yR-yL)/(xR-xL)*(x-xL);

   return y;

} // FUNCTION : interp_from_table



#endif // #ifdef HELMHOLTZ_EOS
#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
