#include "EoS.h"
#include "CUFLU.h"
#include "NuclearEoS.h"

#ifdef DELEPTONIZATION

#define SRC_AUX_ESHIFT              0
#define SRC_AUX_DENS2CGS            1
#define SRC_AUX_VSQR2CGS            2
#define SRC_AUX_KELVIN2MEV          3 
#define SRC_AUX_MEV2KELVIN          4
#define SRC_AUX_DELEP_ENU           5      
#define SRC_AUX_DELEP_RHO1          6
#define SRC_AUX_DELEP_RHO2          7
#define SRC_AUX_DELEP_YE1           8
#define SRC_AUX_DELEP_YE2           9
#define SRC_AUX_DELEP_YEC          10

#if ( MODEL == HYDRO )



// external functions and GPU-related set-up
#ifdef __CUDACC__

#include "CUAPI.h"
#include "CUFLU_Shared_FluUtility.cu"
#include "CUDA_ConstMemory.h"

extern real (*d_SrcDlepProf_Data)[SRC_DLEP_PROF_NBINMAX];
extern real  *d_SrcDlepProf_Radius;

#endif // #ifdef __CUDACC__


// local function prototypes
#ifndef __CUDACC__

void Src_SetAuxArray_Deleptonization( double [], int [] );
void Src_SetFunc_Deleptonization( SrcFunc_t & );
void Src_SetConstMemory_Deleptonization( const double AuxArray_Flt[], const int AuxArray_Int[],
                                         double *&DevPtr_Flt, int *&DevPtr_Int );
void Src_PassData2GPU_Deleptonization();

#endif



/********************************************************
1. Deleptonization source term
   --> Enabled by the runtime option "SRC_DELEPTONIZATION"

2. This file is shared by both CPU and GPU

   CUSRC_Src_Deleptonization.cu -> CPU_Src_Deleptonization.cpp

3. Four steps are required to implement a source term

   I.   Set auxiliary arrays
   II.  Implement the source-term function
   III. [Optional] Add the work to be done every time
        before calling the major source-term function
   IV.  Set initialization functions

4. The source-term function must be thread-safe and
   not use any global variable
********************************************************/

GPU_DEVICE static
real YeOfRhoFunc( const real DENS_CGS, const real DELEP_RHO1, const real DELEP_RHO2, 
                  const real DELEP_YE1, const real DELEP_YE2, const real DELEP_YEC );

// =======================
// I. Set auxiliary arrays
// =======================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetAuxArray_Deleptonization
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by Src_Init_Deleptonization()
//                2. AuxArray_Flt/Int[] have the size of SRC_NAUX_DLEP defined in Macro.h (default = 5)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_SetAuxArray_Deleptonization( double AuxArray_Flt[], int AuxArray_Int[] )
{

   AuxArray_Flt[SRC_AUX_ESHIFT            ] = EoS_AuxArray_Flt[NUC_AUX_ESHIFT];
   AuxArray_Flt[SRC_AUX_DENS2CGS          ] = UNIT_D;
   AuxArray_Flt[SRC_AUX_VSQR2CGS          ] = SQR( UNIT_V );
   AuxArray_Flt[SRC_AUX_KELVIN2MEV        ] = Const_kB_eV*1.0e-6;
   AuxArray_Flt[SRC_AUX_MEV2KELVIN        ] = 1.0  / AuxArray_Flt[NUC_AUX_KELVIN2MEV];

   AuxArray_Flt[SRC_AUX_DELEP_ENU         ] = DELEP_ENU;
   AuxArray_Flt[SRC_AUX_DELEP_RHO1        ] = DELEP_RHO1;
   AuxArray_Flt[SRC_AUX_DELEP_RHO2        ] = DELEP_RHO2;
   AuxArray_Flt[SRC_AUX_DELEP_YE1         ] = DELEP_YE1;
   AuxArray_Flt[SRC_AUX_DELEP_YE2         ] = DELEP_YE2;
   AuxArray_Flt[SRC_AUX_DELEP_YEC         ] = DELEP_YEC;

} // FUNCTION : Src_SetAuxArray_Deleptonization
#endif // #ifndef __CUDACC__



// ======================================
// II. Implement the source-term function
// ======================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Deleptonization
// Description :  Major source-term function
//
// Note        :  1. Invoked by CPU/GPU_SrcSolver_IterateAllCells()
//                2. See Src_SetAuxArray_Deleptonization() for the values stored in AuxArray_Flt/Int[]
//                3. Shared by both CPU and GPU
//
// Parameter   :  fluid             : Fluid array storing both the input and updated values
//                                    --> Including both active and passive variables
//                B                 : Cell-centered magnetic field
//                SrcTerms          : Structure storing all source-term variables
//                dt                : Time interval to advance solution
//                dh                : Grid size
//                x/y/z             : Target physical coordinates
//                TimeNew           : Target physical time to reach
//                TimeOld           : Physical time before update
//                                    --> This function updates physical time from TimeOld to TimeNew
//                MinDens/Pres/Eint : Density, pressure, and internal energy floors
//                EoS               : EoS object
//                AuxArray_*        : Auxiliary arrays (see the Note above)
//
// Return      :  fluid[]
//-----------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static void Src_Deleptonization( real fluid[], const real B[],
                                 const SrcTerms_t *SrcTerms, const real dt, const real dh,
                                 const double x, const double y, const double z,
                                 const double TimeNew, const double TimeOld,
                                 const real MinDens, const real MinPres, const real MinEint,
                                 const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );
#  endif


   const real EnergyShift  = AuxArray_Flt[SRC_AUX_ESHIFT    ];
   const real Dens2CGS     = AuxArray_Flt[SRC_AUX_DENS2CGS  ];
   const real sEint2CGS    = AuxArray_Flt[SRC_AUX_VSQR2CGS  ];
   const real MeV2Kelvin   = AuxArray_Flt[SRC_AUX_MEV2KELVIN];
   const real DELEP_ENU    = AuxArray_Flt[SRC_AUX_DELEP_ENU ];
   const real DELEP_RHO1   = AuxArray_Flt[SRC_AUX_DELEP_RHO1];
   const real DELEP_RHO2   = AuxArray_Flt[SRC_AUX_DELEP_RHO2];
   const real DELEP_YE1    = AuxArray_Flt[SRC_AUX_DELEP_YE1 ];
   const real DELEP_YE2    = AuxArray_Flt[SRC_AUX_DELEP_YE2 ];
   const real DELEP_YEC    = AuxArray_Flt[SRC_AUX_DELEP_YEC ];

// TBF
// profiles are stored in SrcTerms->Dlep_Profile_DataDevPtr/Dlep_Profile_RadiusDevPtr/Dlep_Profile_NBin
// --> see "include/SrcTerms.h"

#  if ( EOS == EOS_NUCLEAR )  &&  ( defined YeOfRhoFunc )

   const real Delep_minDens_CGS  = 1.e6; // [g/cm^3]
   const real Q = 1.293333;

   real Del_Ye   = 0.0
   real Del_Entr = 0.0;

   real xMom_Code;
   real yMom_Code;
   real zMom_Code;
 
   // output Ye
   real Yout = NULL_REAL;

   if ( EOS_POSTBOUNCE )
   {
      return;
   }

   // Deleptonization
   
   // code units
   real Dens_Code = fluid[DENS];
   real Eint_Code = fluid[ENGY];
   real Entr      = fluid[ENTR] / Dens_Code; // entropy in kb/baryon
   real Ye        = fluid[YE - NCOMP_FLUID] / Dens_Code;
   xMom_Code      = fluid[MOMX];
   yMom_Code      = fluid[MOMY];
   zMom_Code      = fluid[MOMZ];

   Eint_Code      = Eint_Code - 0.5 * ( SQR(fluid[MOMX] ) + SQR( fluid[MOMY] ) + SQR( fluid[MOMZ] ) ) / Dens_Code; // internal energy
   real sEint_CGS = ( Eint_Code * sEint2CGS / Dens_Code ) - EnergyShift; // specific internal energy


// check
#  ifdef GAMER_DEBUG
   if ( Hydro_CheckNegative(Dens_Code) )
      printf( "ERROR : invalid input density (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Dens_Code, __FILE__, __LINE__, __FUNCTION__ );

// still require Eint>0 for the nuclear EoS
   if ( Hydro_CheckNegative(Eint_Code) )
      printf( "ERROR : invalid input internal energy (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Eint_Code, __FILE__, __LINE__, __FUNCTION__ );
   if ( Eint_Code < (real)Table[NUC_TAB_YE][0]  ||  Eint_Code> (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
// check Ye              
   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG

   // input and output arrays for Nuclear EoS
   real In1[3];
   real Out1[2];
   In1[0] = Dens_Code; // density in code units
   In1[1] = Eint_Code; // internal energy in code units
   In1[2] = Ye;        // electron fraction
   Out1[0] = NULL_REAL; // Temp in MeV
   Out1[1] = NULL_REAL; // chemical potential munu

   if ( sEint_CGS <= Delep_minDens_CGS )
   {
      Del_Ye = 0.0;
   } else
   {
      yout = YeOfRhoFunc( sEint_CGS );
      Del_Ye = Yout - Ye;
      Del_Ye = MIN( 0.0, Del_Ye ); // Deleptonization cannot increase Ye
   }

#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(sEint_CGS) )
      printf( "ERROR : EoS overflow (sEint_CGS %13.7e, Eint_Code %13.7e, Dens_Code %13.7e, sEint2CGS %13.7e) in %s() !!\n",
              sEint_CGS, Eint_Code, Dens_Code, sEint2CGS, __FUNCTION__ );
#  endif // GAMER_DEBUG

   if ( Del_Ye < 0.0 )
   {
      // Nuclear EoS
      EoS_General_Nuclear( NUC_MODE_ENGY, Out1, In1, EoS_AuxArray_Flt, EoS_AuxArray_Int, 
                           EoS.table ) // energy mode 
      real Temp_Mev = Out1[0];
      real munu     = Out1[1]; 
      munu += Q;  // add chemical potential

#  ifdef GAMER_DEBUG
   if ( munu != mu_nu )
      printf( "ERROR : Couldn't get chemical potential munu (NaN) !!\n" );
#  endif // GAMER_DEBUG

      if ( ( mu_nu < DELEP_ENU ) || ( Dens_Code >= 2.e12 / Dens2CGS ) ) 
      {
         Del_Entr = 0.0;
      } else 
      {
         Del_Entr = - Del_Ye * ( mu_nu - DELEP_ENU ) / Temp_MeV;
      }

      fluid[ENTR] = Dens_Code * ( Entr + Del_Entr );
      fluid[YE]   = Dens_Code * ( Ye + Del_Ye );

      Entr = Entr + Del_Entr;
      Ye   = Ye + Del_Ye;

// check entropy and Ye
#  ifdef GAMER_DEBUG
  if ( Entr_CGS < (real)EoS_Table[NUC_TAB_ENTR_MODE][0]  ||  Entr_CGS > (real)EoS_Table[NUC_TAB_ENTR_MODE][NMode-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Entr_CGS, EoS_Table[NUC_TAB_ENTR_MODE][0], EoS_Table[NUC_TAB_ENTR_MODE][NMode-1], __FUNCTION__ );
// check Ye              
   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG

      // input and output arrays for Nuclear EoS
      real In2[3];
      real Out2[1];
      In2[0]  = Dens_Code; // density in code units
      In2[1]  = Entr;      // entropy in kb/baryon
      In2[2]  = Ye;        // electron fraction
      Out2[0] = NULL_REAL; // volume energy density in code units (with energy shift)

      // Nuclear EoS
      EoS_General_Nuclear( NUC_MODE_ENTR, Out, In, EoS_AuxArray_Flt, EoS_AuxArray_Int, 
                           EoS.table ) // entropy mode 

      Eint_Code = Out2[0]; // volume energy density in code units (with energy shift)

// trigger a *hard failure* if the EoS driver fails
   if ( Err )  Eint_Code = NAN;

   fluid[ENGY] = Eint_Code + 0.5 * ( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS];
   
// final check
#  ifdef GAMER_DEBUG
   if ( Hydro_CheckNegative(Eint_Code) )
   {
      printf( "ERROR : invalid output internal energy density (%13.7e) in %s() !!\n", Eint_Code, __FUNCTION__ );
      printf( "        Dens=%13.7e, Pres=%13.7e\n", Dens_Code, Pres_Code );
      printf( "        EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG

    } // if ( Del_Ye < 0.0 )

#  endif // # if ( EOS == EOS_NUCLEAR )


} // FUNCTION : Src_Deleptonization



// ==================================================
// III. [Optional] Add the work to be done every time
//      before calling the major source-term function
// ==================================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_WorkBeforeMajorFunc_Deleptonization
// Description :  Specify work to be done every time before calling the major source-term function
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  lv               : Target refinement level
//                TimeNew          : Target physical time to reach
//                TimeOld          : Physical time before update
//                                   --> The major source-term function will update the system from TimeOld to TimeNew
//                dt               : Time interval to advance solution
//                                   --> Physical coordinates : TimeNew - TimeOld == dt
//                                       Comoving coordinates : TimeNew - TimeOld == delta(scale factor) != dt
//                AuxArray_Flt/Int : Auxiliary arrays
//                                   --> Can be used and/or modified here
//                                   --> Must call Src_SetConstMemory_Deleptonization() after modification
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_WorkBeforeMajorFunc_Deleptonization( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                              double AuxArray_Flt[], int AuxArray_Int[] )
{

// TBF

/*
// compute profiles
// --> here is just an example; see GREP for a more efficient implementation
// --> SRC_DLEP_PROF_NVAR and SRC_DLEP_PROF_NBINMAX are defined in Macro.h (default = 6 and 4000, respectively)
// --> be careful about the issue of center drifting
   const double      Center[3]      = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };
   const double      MaxRadius      = 0.5*amr->BoxSize[0];
   const double      MinBinSize     = amr->dh[MAX_LEVEL];
   const bool        LogBin         = true;
   const double      LogBinRatio    = 1.25;
   const bool        RemoveEmptyBin = true;
   const long        TVar[]         = { _DENS, _MOMX, _ENGY, _PRES, _VELR, _EINT_DER };
   const int         NProf          = SRC_DLEP_PROF_NVAR;
   const int         SingleLv       = -1;
   const int         MaxLv          = -1;
   const PatchType_t PatchType      = PATCH_LEAF;
   const double      PrepTime       = TimeNew;

   Profile_t *Prof[SRC_DLEP_PROF_NVAR];
   for (int v=0; v<SRC_DLEP_PROF_NVAR; v++)  Prof[v] = new Profile_t();

   Aux_ComputeProfile( Prof, Center, MaxRadius, MinBinSize, LogBin, LogBinRatio, RemoveEmptyBin,
                       TVar, NProf, SingleLv, MaxLv, PatchType, PrepTime );


// check and store the number of radial bins
   if ( Prof[0]->NBin > SRC_DLEP_PROF_NBINMAX )
      Aux_Error( ERROR_INFO, "Number of radial bins (%d) exceeds the maximum size (%d) !!\n",
                 Prof[0]->NBin, SRC_DLEP_PROF_NBINMAX );

   SrcTerms.Dlep_Profile_NBin = Prof[0]->NBin;


// store profiles in the host arrays
// --> note the typecasting from double to real
   for (int v=0; v<SRC_DLEP_PROF_NVAR; v++)
   for (int b=0; b<Prof[v]->NBin; b++)
      h_SrcDlepProf_Data[v][b] = (real)Prof[v]->Data[b];

   for (int b=0; b<Prof[0]->NBin; b++)
      h_SrcDlepProf_Radius[b] = (real)Prof[0]->Radius[b];


// pass profiles to GPU
#  ifdef GPU
   Src_PassData2GPU_Deleptonization();
#  endif


// uncomment the following lines if the auxiliary arrays have been modified
//#  ifdef GPU
//   Src_SetConstMemory_Deleptonization( AuxArray_Flt, AuxArray_Int,
//                                       SrcTerms.Dlep_AuxArrayDevPtr_Flt, SrcTerms.Dlep_AuxArrayDevPtr_Int );
//#  endif


// free memory
   for (int v=0; v<SRC_DLEP_PROF_NVAR; v++)  delete Prof[v];
*/

} // FUNCTION : Src_WorkBeforeMajorFunc_Deleptonization
#endif



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_PassData2GPU_Deleptonization
// Description :  Transfer data to GPU
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc_Deleptonization()
//                2. Use synchronous transfer
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Src_PassData2GPU_Deleptonization()
{

   const long Size_Data   = sizeof(real)*SRC_DLEP_PROF_NVAR*SRC_DLEP_PROF_NBINMAX;
   const long Size_Radius = sizeof(real)*                   SRC_DLEP_PROF_NBINMAX;

// use synchronous transfer
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcDlepProf_Data,   h_SrcDlepProf_Data,   Size_Data,   cudaMemcpyHostToDevice )  );
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcDlepProf_Radius, h_SrcDlepProf_Radius, Size_Radius, cudaMemcpyHostToDevice )  );

} // FUNCTION : Src_PassData2GPU_Deleptonization
#endif // #ifdef __CUDACC__



// ================================
// IV. Set initialization functions
// ================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE SrcFunc_t SrcFunc_Ptr = Src_Deleptonization;

//-----------------------------------------------------------------------------------------
// Function    :  Src_SetFunc_Deleptonization
// Description :  Return the function pointer of the CPU/GPU source-term function
//
// Note        :  1. Invoked by Src_Init_Deleptonization()
//                2. Call-by-reference
//                3. Use either CPU or GPU but not both of them
//
// Parameter   :  SrcFunc_CPU/GPUPtr : CPU/GPU function pointer to be set
//
// Return      :  SrcFunc_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void Src_SetFunc_Deleptonization( SrcFunc_t &SrcFunc_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &SrcFunc_GPUPtr, SrcFunc_Ptr, sizeof(SrcFunc_t) )  );
}

#elif ( !defined GPU )

void Src_SetFunc_Deleptonization( SrcFunc_t &SrcFunc_CPUPtr )
{
   SrcFunc_CPUPtr = SrcFunc_Ptr;
}

#endif // #ifdef __CUDACC__ ... elif ...



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetConstMemory_Deleptonization
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by Src_Init_Deleptonizatio() and, if necessary, Src_WorkBeforeMajorFunc_Deleptonizatio()
//                3. SRC_NAUX_DLEP is defined in Macro.h
//
// Parameter   :  AuxArray_Flt/Int : Auxiliary arrays to be copied to the constant memory
//                DevPtr_Flt/Int   : Pointers to store the addresses of constant memory arrays
//
// Return      :  c_Src_Dlep_AuxArray_Flt[], c_Src_Dlep_AuxArray_Int[], DevPtr_Flt, DevPtr_Int
//---------------------------------------------------------------------------------------------------
void Src_SetConstMemory_Deleptonization( const double AuxArray_Flt[], const int AuxArray_Int[],
                                         double *&DevPtr_Flt, int *&DevPtr_Int )
{

// copy data to constant memory
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Dlep_AuxArray_Flt, AuxArray_Flt, SRC_NAUX_DLEP*sizeof(double) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Dlep_AuxArray_Int, AuxArray_Int, SRC_NAUX_DLEP*sizeof(int   ) )  );

// obtain the constant-memory pointers
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Flt, c_Src_Dlep_AuxArray_Flt )  );
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Int, c_Src_Dlep_AuxArray_Int )  );

} // FUNCTION : Src_SetConstMemory_Deleptonization
#endif // #ifdef __CUDACC__



#ifndef __CUDACC__

//-----------------------------------------------------------------------------------------
// Function    :  Src_Init_Deleptonization
// Description :  Initialize the deleptonization source term
//
// Note        :  1. Set auxiliary arrays by invoking Src_SetAuxArray_*()
//                   --> Copy to the GPU constant memory and store the associated addresses
//                2. Set the source-term function by invoking Src_SetFunc_*()
//                   --> Unlike other modules (e.g., EoS), here we use either CPU or GPU but not
//                       both of them
//                3. Invoked by Src_Init()
//                4. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_Init_Deleptonization()
{

// set the auxiliary arrays
   Src_SetAuxArray_Deleptonization( Src_Dlep_AuxArray_Flt, Src_Dlep_AuxArray_Int );

// copy the auxiliary arrays to the GPU constant memory and store the associated addresses
#  ifdef GPU
   Src_SetConstMemory_Deleptonization( Src_Dlep_AuxArray_Flt, Src_Dlep_AuxArray_Int,
                                       SrcTerms.Dlep_AuxArrayDevPtr_Flt, SrcTerms.Dlep_AuxArrayDevPtr_Int );
#  else
   SrcTerms.Dlep_AuxArrayDevPtr_Flt = Src_Dlep_AuxArray_Flt;
   SrcTerms.Dlep_AuxArrayDevPtr_Int = Src_Dlep_AuxArray_Int;
#  endif

// set the major source-term function
   Src_SetFunc_Deleptonization( SrcTerms.Dlep_FuncPtr );

} // FUNCTION : Src_Init_Deleptonization



//-----------------------------------------------------------------------------------------
// Function    :  Src_End_Deleptonization
// Description :  Release the resources used by the deleptonization source term
//
// Note        :  1. Invoked by Src_End()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_End_Deleptonization()
{

// TBF

} // FUNCTION : Src_End_Deleptonization

#endif // #ifndef __CUDACC__






//-----------------------------------------------------------------------------------------
// Function    :  YeOfRhoFunc
// Description :  Calculate electron fraction Ye from the density
//
// Note        :  1. Invoked by Src_End()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  xdens : density in CGS from which Ye is caculated
//
// Return      :  YeOfRhoFunc
//-----------------------------------------------------------------------------------------
GPU_DEVICE static
real YeOfRhoFunc( const real DENS_CGS, const real DELEP_RHO1, const real DELEP_RHO2, 
                  const real DELEP_YE1, const real DELEP_YE2, const real DELEP_YEC )
{

   real XofRho, Ye;

   XofRho = 2.0 * LOG10( DENS_CGS ) - LOG10( DELEP_RHO2 ) - LOG10( DELEP_RHO1 );
   XofRho = xofrho / ( LOG10( DELEP_RHO2 ) - LOG10( DELEP_RHO1 ) );
   XofRho = MAX( -1.0, MIN( 1.0, xofrho ) );
   Ye = 0.5 * ( DELEP_YE2 + DELEP_YE1 ) + 0.5 * XofRho * ( DELEP_YE2 - DELEP_YE1 );
   Ye = Ye + DELEP_YEC * ( 1.0 - FABS( XofRho ) );
   Ye = Ye + DELEP_YEC * 4.0 * FABS( XofRho ) * ( FABS( XofRho ) - 0.5 ) * ( FABS( XofRho ) - 1.0 );
   return Ye;

}

#endif // #if ( MODEL == HYDRO )


#endif // #ifdef DELEPTONIZATION