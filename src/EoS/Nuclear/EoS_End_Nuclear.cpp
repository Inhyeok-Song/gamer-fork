#include "GAMER.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )


#ifdef GPU
void CUAPI_MemFree_NuclearEoS();
#endif


extern real *g_alltables;
extern real *g_alltables_mode;
extern real *g_logrho;
extern real *g_yes;
extern real *g_logrho_mode;
extern real *g_entr_mode;
extern real *g_logprss_mode;
extern real *g_yes_mode;

#if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
extern real *g_logtemp;
extern real *g_logeps_mode;
#else
extern real *g_logeps;
extern real *g_logtemp_mode;
#endif


#ifdef HELMHOLTZ_EOS
extern double  *g_helmholtz_table;
extern double  *g_helmholtz_dd;
extern double  *g_helmholtz_dt;
extern double  *g_helm_dens;
extern double  *g_helm_temp;
extern double  *g_helm_diff;
extern double  *g_prog_dens;
extern double  *g_prog_abar;
extern double  *g_prog_zbar;
#endif




//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_End_Nuclear
// Description :  Free the resources used by the nuclear EoS routines
//
// Note        :  1. Invoked by EoS_End()
//                   --> Linked to the function pointer "EoS_End_Ptr"
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void EoS_End_Nuclear()
{

// CPU memory
   free( g_alltables      );  g_alltables      = NULL;
   free( g_alltables_mode );  g_alltables_mode = NULL;
   free( g_logrho         );  g_logrho         = NULL;
   free( g_yes            );  g_yes            = NULL;
   free( g_entr_mode      );  g_entr_mode      = NULL;
   free( g_logprss_mode   );  g_logprss_mode   = NULL;
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   free( g_logtemp        );  g_logtemp        = NULL;
   free( g_logeps_mode    );  g_logeps_mode    = NULL;
#  else
   free( g_logeps         );  g_logeps         = NULL;
   free( g_logtemp_mode   );  g_logtemp_mode   = NULL;
#  endif
   free( g_yes_mode       );  g_yes_mode       = NULL;

#  ifdef HELMHOLTZ_EOS
   free( g_helmholtz_table ); g_helmholtz_table = NULL;
   free( g_helmholtz_dd    ); g_helmholtz_dd    = NULL;
   free( g_helmholtz_dt    ); g_helmholtz_dt    = NULL;
   free( g_helm_dens       ); g_helm_dens       = NULL;
   free( g_helm_temp       ); g_helm_temp       = NULL;
   free( g_helm_diff       ); g_helm_diff       = NULL;
   free( g_prog_dens       ); g_prog_dens       = NULL;
   free( g_prog_abar       ); g_prog_abar       = NULL;
   free( g_prog_zbar       ); g_prog_zbar       = NULL;
#  endif



// GPU memory
#  ifdef GPU
   CUAPI_MemFree_NuclearEoS();
#  endif

} // FUNCTION : EoS_End_Nuclear



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
