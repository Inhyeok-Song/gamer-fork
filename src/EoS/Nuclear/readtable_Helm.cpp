#include "NuclearEoS.h"
#include "HelmholtzEoS.h"
#include "GAMER.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
#ifdef HELMHOLTZ_EOS



extern int     g_helm_imax;
extern int     g_helm_jmax;
extern int     g_prog_nbin;
extern double  g_helm_alpha;
extern double  g_c_shift;

extern double *g_helmholtz_table;
extern double *g_helmholtz_dd;
extern double *g_helmholtz_dt;
extern double *g_helm_dens;
extern double *g_helm_temp;
extern double *g_helm_diff;
extern double *g_prog_dens;
extern double *g_prog_abar;
extern double *g_prog_zbar;
extern double *g_prog_xn;
extern double *g_prog_xp;


extern int    g_nrho;
extern int    g_ntemp;
extern int    g_nye;
extern int    g_nrho_mode;
extern int    g_nmode;
extern int    g_nye_mode;
extern double g_energy_shift;

extern real  *g_alltables;
extern real  *g_logrho;
extern real  *g_logtemp;
extern real  *g_yes;



// prototypes
void nuc_eos_C_short( real *Out, const real *In,
                      const int NTarget, const int *TargetIdx,
                      const real energy_shift, real Temp_InitGuess,
                      const int nrho, const int ntoreps, const int nye,
                      const int nrho_Aux, const int nmode_Aux, const int nye_Aux,
                      const real *alltables, const real *alltables_Aux,
                      const real *logrho, const real *logtoreps, const real *yes,
                      const real *logrho_Aux, const real *mode_Aux, const real *yes_Aux,
                      const int IntScheme_Aux, const int IntScheme_Main,
                      const int keymode, int *keyerr, const real rfeps );
void Helmholtz_eos( real *Out, const real *In, const int NTarget, const int *TargetIdx,
                    real Temp_InitGuess, const real c_shift, const int imax, const int jmax,
                    const double *helmholtz_table, const double *helmholtz_dd, const double *helmholtz_dt,
                    const double *helm_dens, const double *helm_temp, const double *helm_diff,
                    const double *prog_dens, const double *prog_abar, const double *prog_zbar,
                    const int prog_nbin, const real dens_trans, const real dens_stop, const real alpha,
                    const int nrho, const int ntoreps, const int nye, const real *alltables,
                    const real *logrho, const real *logtoreps, const real *yes,
                    const int IntScheme_Aux, const int IntScheme_Main,
                    const int keymode, int *keyerr, const real rfeps );



//-------------------------------------------------------------------------------------
// Function    :  Helm_eos_ReadTable
// Description :  Load the Helmholtz EoS table from the disk
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//
// Parameter   :  helmeos_table_name : Filename
//
// Return      :  EoS tables
//-------------------------------------------------------------------------------------
void Helm_eos_ReadTable( char *helmeos_table_name )
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


   if ( MPI_Rank == 0 )
      Aux_Message( stdout, "   Reading Helmholtz EoS table: %s\n", helmeos_table_name );

// check file existence
   if ( !Aux_CheckFileExist(helmeos_table_name) )
      Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", helmeos_table_name );


// use these macros to allocate and free the tables
// allocate EOS tables and variables
#  define ALLOC_EOS( ptr, size )                                                 \
   {                                                                             \
      ptr = (double *)malloc( size );                                            \
      if ( !ptr )                                                                \
      {                                                                          \
         Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );   \
      }                                                                          \
   }

#  define FREE_AND_NULL( p ) do { free(p); (p) = NULL; } while (0)



   FILE *FILE = fopen( helmeos_table_name, "r" );


// set the number of row/column of the Helmholtz EoS table
   g_helm_imax = EoS.Helm_Imax;
   g_helm_jmax = EoS.Helm_Jmax;


// allocate memory for tables
   ALLOC_EOS( g_helmholtz_table, g_helm_imax*g_helm_jmax*HELM_TABLE_NVAR*sizeof(double) );
   ALLOC_EOS( g_helmholtz_dd,    (g_helm_imax-1)*5*sizeof(double)                       );
   ALLOC_EOS( g_helmholtz_dt,    (g_helm_jmax-1)*5*sizeof(double)                       );
   ALLOC_EOS( g_helm_dens,       g_helm_imax*sizeof(double)                             );
   ALLOC_EOS( g_helm_temp,       g_helm_jmax*sizeof(double)                             );
   ALLOC_EOS( g_helm_diff,       g_ntemp*g_nye*sizeof(double)                           );


// set up arrays to store the Helmholtz free energy and its derivatives
   double *f, *fd, *ft, *fdd, *ftt, *fdt, *fddt, *fdtt, *fddtt;
   double *dpdf, *dpdfd, *dpdft, *dpdfdt, *ef, *efd, *eft, *efdt, *xf, *xfd, *xft, *xfdt;
   size_t table_size = g_helm_imax * g_helm_jmax * sizeof(double);
   ALLOC_EOS( f,      table_size );   ALLOC_EOS( fd,     table_size );   ALLOC_EOS( ft,     table_size );
   ALLOC_EOS( fdd,    table_size );   ALLOC_EOS( ftt,    table_size );   ALLOC_EOS( fdt,    table_size );
   ALLOC_EOS( fddt,   table_size );   ALLOC_EOS( fdtt,   table_size );   ALLOC_EOS( fddtt,  table_size );
   ALLOC_EOS( dpdf,   table_size );   ALLOC_EOS( dpdfd,  table_size );   ALLOC_EOS( dpdft,  table_size );
   ALLOC_EOS( dpdfdt, table_size );   ALLOC_EOS( ef,     table_size );   ALLOC_EOS( efd,    table_size );
   ALLOC_EOS( eft,    table_size );   ALLOC_EOS( efdt,   table_size );   ALLOC_EOS( xf,     table_size );
   ALLOC_EOS( xfd,    table_size );   ALLOC_EOS( xft,    table_size );   ALLOC_EOS( xfdt,   table_size );

   double *dd_sav, *dd2_sav, *ddi_sav, *dd2i_sav, *dd3i_sav;
   size_t size_i = (g_helm_imax-1) * sizeof(double);
   ALLOC_EOS( dd_sav,   size_i );   ALLOC_EOS( dd2_sav,  size_i );   ALLOC_EOS( ddi_sav,  size_i );
   ALLOC_EOS( dd2i_sav, size_i );   ALLOC_EOS( dd3i_sav, size_i );

   double *dt_sav, *dt2_sav, *dti_sav, *dt2i_sav, *dt3i_sav;
   size_t size_j = (g_helm_jmax-1) * sizeof(double);
   ALLOC_EOS( dt_sav,   size_j );   ALLOC_EOS( dt2_sav,  size_j );   ALLOC_EOS( dti_sav,  size_j );
   ALLOC_EOS( dt2i_sav, size_j );   ALLOC_EOS( dt3i_sav, size_j );


   const double tstp  = (Const_thi - Const_tlo) / (g_helm_jmax - 1);
   const double tstpi = 1.0 / tstp;
   const double dstp  = (Const_dhi - Const_dlo) / (g_helm_imax - 1);
   const double dstpi = 1.0 / dstp;

   for (int j=0; j<g_helm_jmax; j++)
   {
      double tsav = Const_tlo + j*tstp;
      g_helm_temp[j] = pow( 10.0, tsav );
      for (int i=0; i<g_helm_imax; i++)
      {
         double dsav = Const_dlo + i*dstp;
         g_helm_dens[i] = pow( 10.0, dsav );
         fscanf( FILE, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                 &f[i*g_helm_jmax+j],    &fd[i*g_helm_jmax+j],   &ft[i*g_helm_jmax+j],
                 &fdd[i*g_helm_jmax+j],  &ftt[i*g_helm_jmax+j],  &fdt[i*g_helm_jmax+j],
                 &fddt[i*g_helm_jmax+j], &fdtt[i*g_helm_jmax+j], &fddtt[i*g_helm_jmax+j] );
      }
   }


   for (int j=0; j<g_helm_jmax; j++)
   for (int i=0; i<g_helm_imax; i++)
      fscanf( FILE, "%lf %lf %lf %lf",
              &dpdf[i*g_helm_jmax+j], &dpdfd[i*g_helm_jmax+j], &dpdft[i*g_helm_jmax+j], &dpdfdt[i*g_helm_jmax+j] );

   for (int j=0; j<g_helm_jmax; j++)
   for (int i=0; i<g_helm_imax; i++)
      fscanf( FILE, "%lf %lf %lf %lf",
              &ef[i*g_helm_jmax+j], &efd[i*g_helm_jmax+j], &eft[i*g_helm_jmax+j], &efdt[i*g_helm_jmax+j] );

   for (int j=0; j<g_helm_jmax; j++)
   for (int i=0; i<g_helm_imax; i++)
      fscanf( FILE, "%lf %lf %lf %lf",
              &xf[i*g_helm_jmax+j], &xfd[i*g_helm_jmax+j], &xft[i*g_helm_jmax+j], &xfdt[i*g_helm_jmax+j] );


// calculate the derivatives
   for (int i=0; i<g_helm_imax-1; i++)
   {
      double dd   = g_helm_dens[i+1] - g_helm_dens[i];
      double dd2  = dd * dd;
      double ddi  = 1.0 / dd;
      double dd2i = 1.0 / dd2;
      double dd3i = dd2i * ddi;

      dd_sav  [i] = dd;
      dd2_sav [i] = dd2;
      ddi_sav [i] = ddi;
      dd2i_sav[i] = dd2i;
      dd3i_sav[i] = dd3i;
   }

   for (int j=0; j<g_helm_jmax-1; j++)
   {
      double dth  = g_helm_temp[j+1] - g_helm_temp[j];
      double dt2  = dth * dth;
      double dti  = 1.0/dth;
      double dt2i = 1.0/dt2;
      double dt3i = dt2i * dti;

      dt_sav  [j] = dth;
      dt2_sav [j] = dt2;
      dti_sav [j] = dti;
      dt2i_sav[j] = dt2i;
      dt3i_sav[j] = dt3i;
   }


// store the variables
   for (int i=0; i<g_helm_imax; i++)
   {
      for (int j=0; j<g_helm_jmax; j++)
      {
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 0  ] = f     [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 1  ] = fd    [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 2  ] = ft    [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 3  ] = fdd   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 4  ] = ftt   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 5  ] = fdt   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 6  ] = fddt  [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 7  ] = fdtt  [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 8  ] = fddtt [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 9  ] = dpdf  [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 10 ] = dpdfd [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 11 ] = dpdft [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 12 ] = dpdfdt[i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 13 ] = ef    [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 14 ] = efd   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 15 ] = eft   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 16 ] = efdt  [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 17 ] = xf    [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 18 ] = xfd   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 19 ] = xft   [i*g_helm_jmax + j];
         g_helmholtz_table[ ( i*g_helm_jmax*HELM_TABLE_NVAR) + (j*HELM_TABLE_NVAR) + 20 ] = xfdt  [i*g_helm_jmax + j];
      }
   }

   for (int i=0; i<g_helm_imax-1; i++)
   {
      g_helmholtz_dd[ i*5 + 0 ] = dd_sav  [i];
      g_helmholtz_dd[ i*5 + 1 ] = dd2_sav [i];
      g_helmholtz_dd[ i*5 + 2 ] = ddi_sav [i];
      g_helmholtz_dd[ i*5 + 3 ] = dd2i_sav[i];
      g_helmholtz_dd[ i*5 + 4 ] = dd3i_sav[i];
   }

   for (int j=0; j<g_helm_jmax-1; j++)
   {
      g_helmholtz_dt[ j*5 + 0 ] = dt_sav  [j];
      g_helmholtz_dt[ j*5 + 1 ] = dt2_sav [j];
      g_helmholtz_dt[ j*5 + 2 ] = dti_sav [j];
      g_helmholtz_dt[ j*5 + 3 ] = dt2i_sav[j];
      g_helmholtz_dt[ j*5 + 4 ] = dt3i_sav[j];
   }


// calculate c_shift and diff(T, Ye) at Helm_Dens_Trans
   const double Dens_Trans = EoS.Helm_Dens_Trans;
   const double Dens_Stop  = EoS.Helm_Dens_Stop;


// calculate the exponential decline parameter alpha
   g_helm_alpha = (  LOG10( Dens_Stop ) - LOG10( Dens_Trans )  ) / LOG( EoS.Helm_Decrease );


// load a progenitor model
   const bool RowMajor_No  = false; // load data into the column major
   const bool AllocMem_Yes = true;  // allocate memory for CCSN_Prof

   double* CCSN_Prof = NULL;        // radial profile of initial condition
   int     CCSN_Prof_NBin;          // number of radial bins in the input profile
   int     CCSN_NCol = 4;           // number of columns read from the input profile
   int     CCSN_TargetCols[CCSN_NCol];
   CCSN_TargetCols[0] =  1;  CCSN_TargetCols[1] =  2;  CCSN_TargetCols[2] =  4;
   CCSN_TargetCols[3] =  9;

   const int CCSN_ColIdx_Dens  = 0;
   const int CCSN_ColIdx_Temp  = 1;
   const int CCSN_ColIdx_Ye    = 2;
   const int CCSN_ColIdx_Abar  = 3;
   const int CCSN_ColIdx_Zbar  = 4;

   CCSN_Prof_NBin = Aux_LoadTable( CCSN_Prof, EoS.Helm_IC_File, CCSN_NCol, CCSN_TargetCols, RowMajor_No, AllocMem_Yes );
   g_prog_nbin    = CCSN_Prof_NBin;

         double *Table_Dens = CCSN_Prof + CCSN_ColIdx_Dens*CCSN_Prof_NBin;
   const double *Table_Temp = CCSN_Prof + CCSN_ColIdx_Temp*CCSN_Prof_NBin;
   const double *Table_Ye   = CCSN_Prof + CCSN_ColIdx_Ye  *CCSN_Prof_NBin;
   const double *Table_Abar = CCSN_Prof + CCSN_ColIdx_Abar*CCSN_Prof_NBin;
   // const double *Table_Zbar = CCSN_Prof + CCSN_ColIdx_Zbar*CCSN_Prof_NBin;

// allocate memory for Abar and Zbar
   if (  ! ( g_prog_dens = (double*)malloc(CCSN_Prof_NBin*sizeof(double)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_prog_abar = (double*)malloc(CCSN_Prof_NBin*sizeof(double)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_prog_zbar = (double*)malloc(CCSN_Prof_NBin*sizeof(double)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   for (int i=0; i<CCSN_Prof_NBin; i++)
   {
      g_prog_dens[i] = -Table_Dens[i];  // g_prog_dens must be sorted into the ascending order
      g_prog_abar[i] =  Table_Abar[i];
      // g_prog_zbar[i] =  Table_Zbar[i];
      g_prog_zbar[i] =  NULL;
   }

   for (int i=0; i<CCSN_Prof_NBin; i++)
      Table_Dens[i] *= -1.0; //g_prog_dens[i];

// find temperature and Ye at the transition density in IC
   const double Kelvin2MeV = Const_kB_eV * 1.0e-6;
   const double Temp_Prog_Kelv = Mis_InterpolateFromTable( CCSN_Prof_NBin, Table_Dens, Table_Temp, -Dens_Trans );
   const double Ye_Prog        = Mis_InterpolateFromTable( CCSN_Prof_NBin, Table_Dens, Table_Ye,   -Dens_Trans );
   const double Temp_Prog_MeV  = Temp_Prog_Kelv * Kelvin2MeV;


   const int NTarget = 1;
   int TargetIdx[NTarget];
   real In_Flt[4], Out[NTarget+1];
   In_Flt[0] = Dens_Trans;    // density in g/cm^3
   In_Flt[1] = Temp_Prog_MeV; // temperature in MeV
   In_Flt[2] = Ye_Prog;
   In_Flt[3] = Temp_Prog_MeV;

   int Err = 0;

   TargetIdx[0] = NUC_VAR_IDX_EORT;

// call nuc_eos_C_short directly with temperature mode
   nuc_eos_C_short( Out, In_Flt, NTarget, TargetIdx, g_energy_shift, NULL_REAL,
                    g_nrho, g_ntemp, g_nye, NULL_INT, NULL_INT, NULL_INT, g_alltables,
                    NULL, g_logrho, g_logtemp, g_yes, NULL, NULL, NULL,
                    NUC_INT_SCHEME_AUX, NUC_INT_SCHEME_MAIN, NUC_MODE_TEMP, &Err, Tolerance );

   const real Eint_Trans_Nuc = Out[0];

// call helm_eos directly with temperature mode
   Helmholtz_eos( Out, In_Flt, NTarget, TargetIdx, NULL_REAL, NULL_REAL,
                  g_helm_imax, g_helm_jmax, g_helmholtz_table, g_helmholtz_dd,
                  g_helmholtz_dt, g_helm_dens, g_helm_temp, NULL, g_prog_dens,
                  g_prog_abar, g_prog_zbar, g_prog_nbin, Dens_Trans, Dens_Stop,
                  g_helm_alpha, g_nrho, g_ntemp, g_nye, g_alltables,
                  g_logrho, g_logtemp, g_yes, NUC_INT_SCHEME_AUX, NUC_INT_SCHEME_MAIN,
                  NUC_MODE_TEMP, &Err, Tolerance );

   const double Eint_Trans_Helm = Out[0];

   const double c_shift   = Eint_Trans_Nuc - Eint_Trans_Helm;
                g_c_shift = c_shift;

   for (int itemp=0; itemp<g_ntemp; itemp++)
   for (int iye=0;   iye<g_nye;     iye++  )
   {
      const int NTarget = 3;
      int TargetIdx[NTarget];
      real In_Flt[4], Out[NTarget+1];
      In_Flt[0] = Dens_Trans;                    // density in g/cm^3
      In_Flt[1] = POW( 10.0, g_logtemp[itemp] ); // temperature in MeV
      In_Flt[2] = g_yes[iye];
      In_Flt[3] = NULL_REAL;

      int Err = 0;

      TargetIdx[0] = NUC_VAR_IDX_EORT;
      TargetIdx[1] = NUC_VAR_IDX_ABAR;
      TargetIdx[2] = NUC_VAR_IDX_ZBAR;

//    call nuc_eos_C_short directly with temperature mode
      nuc_eos_C_short( Out, In_Flt, NTarget, TargetIdx, g_energy_shift, NULL_REAL,
                       g_nrho, g_ntemp, g_nye, NULL_INT, NULL_INT, NULL_INT, g_alltables,
                       NULL, g_logrho, g_logtemp, g_yes, NULL, NULL, NULL,
                       NUC_INT_SCHEME_AUX, NUC_INT_SCHEME_MAIN, NUC_MODE_TEMP, &Err, Tolerance );

      const real Eint_Nuc = Out[0];   // shifted internal energy in the nuclear EoS (in cm^2/s^2)

//    call helm_eos directly
      Helmholtz_eos( Out, In_Flt, NTarget, TargetIdx, NULL_REAL, NULL_REAL,
                     g_helm_imax, g_helm_jmax, g_helmholtz_table, g_helmholtz_dd,
                     g_helmholtz_dt, g_helm_dens, g_helm_temp, NULL, g_prog_dens,
                     g_prog_abar, g_prog_zbar, g_prog_nbin, Dens_Trans, Dens_Stop,
                     g_helm_alpha, g_nrho, g_ntemp, g_nye, g_alltables, 
                     g_logrho, g_logtemp, g_yes, NUC_INT_SCHEME_AUX, NUC_INT_SCHEME_MAIN,
                     NUC_MODE_TEMP, &Err, Tolerance );

      const double Eint_Helm = Out[0];

      g_helm_diff[itemp + g_ntemp*iye] = Eint_Nuc - Eint_Helm - c_shift;
   }


// set the EoS table pointers
   h_EoS_Table[NUC_TABLE_HELM     ] = g_helmholtz_table;
   h_EoS_Table[NUC_TABLE_HELM_DD  ] = g_helmholtz_dd;
   h_EoS_Table[NUC_TABLE_HELM_DT  ] = g_helmholtz_dt;
   h_EoS_Table[NUC_TABLE_HELM_DENS] = g_helm_dens;
   h_EoS_Table[NUC_TABLE_HELM_TEMP] = g_helm_temp;
   h_EoS_Table[NUC_TABLE_HELM_DIFF] = g_helm_diff;
   h_EoS_Table[NUC_TABLE_PROG_DENS] = g_prog_dens;
   h_EoS_Table[NUC_TABLE_PROG_ABAR] = g_prog_abar;
   h_EoS_Table[NUC_TABLE_PROG_ZBAR] = g_prog_zbar;


// free memory
   FREE_AND_NULL( f        ); FREE_AND_NULL( fd       ); FREE_AND_NULL( ft      );
   FREE_AND_NULL( fdd      ); FREE_AND_NULL( ftt      ); FREE_AND_NULL( fdt     );
   FREE_AND_NULL( fddt     ); FREE_AND_NULL( fdtt     ); FREE_AND_NULL( fddtt   );
   FREE_AND_NULL( dpdf     ); FREE_AND_NULL( dpdfd    ); FREE_AND_NULL( dpdft   );
   FREE_AND_NULL( dpdfdt   ); FREE_AND_NULL( ef       ); FREE_AND_NULL( efd     );
   FREE_AND_NULL( eft      ); FREE_AND_NULL( efdt     ); FREE_AND_NULL( xf      );
   FREE_AND_NULL( xfd      ); FREE_AND_NULL( xft      ); FREE_AND_NULL( xfdt    );

   FREE_AND_NULL( dd_sav   ); FREE_AND_NULL( dd2_sav  ); FREE_AND_NULL( ddi_sav );
   FREE_AND_NULL( dd2i_sav ); FREE_AND_NULL( dd3i_sav );
   FREE_AND_NULL( dt_sav   ); FREE_AND_NULL( dt2_sav  ); FREE_AND_NULL( dti_sav );
   FREE_AND_NULL( dt2i_sav ); FREE_AND_NULL( dt3i_sav );


// close the file
   fclose( FILE );


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Helm_eos_ReadTable



#endif // #ifdef HELMHOLTZ_EOS
#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
