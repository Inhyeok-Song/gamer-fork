#include "GAMER.h"
//#include "NuclearEos.h"


#if ( defined GRAVITY  &&  defined GREP )



//-------------------------------------------------------------------------------------------------------
// Function    :  CPU_ComputeEffPot
// Description :  Construct the effective potential
//
// Note        :  1. Enabled if macro GRAVITY and GREP are set
//                2. The profile Phi_eff store the value of -Phi_NW(r) + Phi_TOV(r)
//                   at the left edge of bins
//-------------------------------------------------------------------------------------------------------
void CPU_ComputeEffPot( Profile_t *DensAve, Profile_t *EngyAve, Profile_t *VrAve, Profile_t *PresAve,
                        Profile_t *Phi_eff )
{

   const double c2             = SQR( Const_c/UNIT_V );
   const double FourPI         = 4.0*M_PI;
   const double FourThirdPI    = FourPI/3.0;
   const double tolerance      = 1.e-5;

   int     NIter               = GREP_MAXITER;
   int     NBin                = DensAve->NBin;
   double *Radius              = DensAve->Radius;
   double  MaxRadius           = DensAve->MaxRadius;
   double  Mass_NW      [NBin] = { 0.0 };  // Newtonian mass for \bar_Phi(r)     in Eq. (7) in Marek+ (2006)
   double  Mass_TOV     [NBin] = { 0.0 };  // TOV mass       for \bar_Phi(r)_TOV in Eq. (7) in Marek+ (2006)
   double  Mass_TOV_USG [NBin] = { 0.0 };  // TOV mass before update
   double  Dens_TOV     [NBin] = { 0.0 };  // empirical TOV density                 Eq. (4) in Marek+ (2006)
   double  Gamma_TOV    [NBin] = { 1.0 };  // metric function                       Eq. (5) in Marek+ (2006)
   double  Pot_NW       [NBin] = { 0.0 };
   double  Pot_TOV      [NBin] = { 0.0 };
   double  dVol         [NBin] = { 0.0 };
   double  EdgeL        [NBin] = { 0.0 };

   for ( int i=1; i<NBin;   i++ )   EdgeL[i] = ( GREP_LOGBIN ) ? sqrt( Radius[i-1] * Radius[i] )
                                                               : 0.5*( Radius[i-1] + Radius[i] );

   for ( int i=0; i<NBin-1; i++ )   dVol[i] = FourThirdPI * ( CUBE( EdgeL[i+1] ) - CUBE( EdgeL[i] ) );

   dVol[NBin-1] = FourThirdPI * ( CUBE( MaxRadius) - CUBE( EdgeL[NBin-1] ) );


// compute Mass_NW defined at the outer edge of bin
   Mass_NW[0] = dVol[0] * DensAve->Data[0];

   for ( int i=1; i<NBin-1; i++ )
      Mass_NW[i] = Mass_NW[i-1]
                 + dVol[i] * ( 0.5*DensAve->Data[i] + 0.25*( DensAve->Data[i-1] + DensAve->Data[i+1] ) );

   Mass_NW[NBin-1] = Mass_NW[NBin-2] + dVol[NBin-1] * DensAve->Data[NBin-1];


// iteratively compute Mass_TOV and Gamma_TOV defined at the outer edge of bin
   while ( NIter-- )
   {

      for ( int i=0; i<NBin; i++ )
         Dens_TOV[i] = Gamma_TOV[i] * ( DensAve->Data[i] + EngyAve->Data[i] / c2 );


//    compute Mass_TOV
      Mass_TOV[0] = dVol[0] * Dens_TOV[0];

      for ( int i=1; i<NBin-1; i++ )
         Mass_TOV[i] = Mass_TOV[i-1]
                     + dVol[i] * ( 0.5*Dens_TOV[i] + 0.25*( Dens_TOV[i-1] + Dens_TOV[i+1] ) );

      Mass_TOV[NBin-1] = Mass_TOV[NBin-2] + dVol[NBin-1] * Dens_TOV[NBin-1];


//    compute Gamma_TOV
      for ( int i=0; i<NBin-1; i++ )
           Gamma_TOV[i] = SQRT( MAX( TINY_NUMBER,
                                     1.0 + ( 0.25*SQR( VrAve->Data[i] + VrAve->Data[i+1] )
                                           - 2.0*NEWTON_G * Mass_TOV[i]      / EdgeL[i+1]  ) / c2 ) );

      Gamma_TOV[NBin-1] = SQRT( MAX( TINY_NUMBER,
                                     1.0 + (      SQR( VrAve->Data[NBin-1] )
                                           - 2.0*NEWTON_G * Mass_TOV[NBin-1] / MaxRadius   ) / c2 ) );


//    check if tolerance is satisfied
      bool Pass = true;
      for ( int i=0; i<NBin; i++ )
      {
         double error = FABS( Mass_TOV_USG[i] - Mass_TOV[i] ) / Mass_TOV[i];

         if ( error > tolerance )
         {
            Pass = false;
            break;
         }
      }


//    for debug
      if ( !NIter  &&  !Pass )
      {
         if ( ( MPI_Rank == 0 )  &&  ( omp_get_thread_num() == 0 ) )
         {
            printf("\n# GREP_CENTER_METHOD: %d\n",                  GREP_CENTER_METHOD);
            printf("# Center              : %.15e\t%.15e\t%.15e\n", DensAve->Center[0], DensAve->Center[1], DensAve->Center[2]);
            printf("# MaxRadius           : %.15e\n",               DensAve->MaxRadius);
         //   printf("# MinBinSize          : %.15e\n",               MinBinSize);
            printf("# LogBin              : %d\n",                  DensAve->LogBin);
            printf("# LogBinRatio         : %.15e\n",               DensAve->LogBinRatio);
            printf("# Num of Iteration    : %d\n",                  GREP_MAXITER - NIter);
            printf("# NBin                : %d\n",                  NBin);
            printf("# ============================================================\n");
            printf("#  Bin     NCell     RADIUS       DENS       ENGY         Vr   Pressure    Mass_NW   Mass_TOV  Mass_TOV_old  Error_rel  Gamma_TOV\n");
            for ( int i=0; i<NBin; i++ )
               printf("%6d  %8ld  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e     %9.2e  %9.2e  %9.2e\n",
                      i, DensAve->NCell[i], Radius[i],
                      DensAve->Data[i], EngyAve->Data[i], VrAve->Data[i], PresAve->Data[i],
                      Mass_NW[i], Mass_TOV[i], Mass_TOV_USG[i], FABS( Mass_TOV_USG[i] - Mass_TOV[i] ) / Mass_TOV[i], Gamma_TOV[i]);
         }

         Aux_Error( ERROR_INFO, "Too many iterations in computing effective potential\n" );
      }


      if ( Pass )
         break;
      else
         for ( int i=0; i<NBin; i++ )   Mass_TOV_USG[i] = Mass_TOV[i];

   } // while ( NIter-- )


// copy bin information into Phi_eff
   Phi_eff->NBin        = DensAve->NBin;
   Phi_eff->LogBin      = DensAve->LogBin;
   Phi_eff->LogBinRatio = DensAve->LogBinRatio;
   Phi_eff->MaxRadius   = DensAve->MaxRadius;
   Phi_eff->AllocateMemory();
   for ( int d=0; d<3; d++ )               Phi_eff->Center[d] = DensAve->Center[d];
   for ( int b=0; b<Phi_eff->NBin; b++ )   Phi_eff->Radius[b] = DensAve->Radius[b];


// compute effective potential at the bin center
// set the outer boundary condition to be -M_grid / r_outer
   Phi_eff->Data[NBin-1] = -NEWTON_G * ( Mass_TOV[NBin-1] - Mass_NW[NBin-1] ) / Radius[NBin-1];

   for ( int i=NBin-2; i>0; i-- )
   {
      double dr          = Radius[i] - Radius[i+1];
      double RadiusCubed = CUBE( Radius[i] );

      double Mass_NW_rad  = 0.5*( Mass_NW[i] + Mass_NW[i-1] );

      double Mass_TOV_rad = ( 0.5*( Mass_TOV[i] + Mass_TOV[i-1] ) + FourPI * RadiusCubed * PresAve->Data[i] / c2 )
                          * ( 1.0 + ( EngyAve->Data[i] + PresAve->Data[i] ) / ( DensAve->Data[i] * c2 ) )
                          / SQR( 0.5 * ( Gamma_TOV[i] + Gamma_TOV[i-1] ) );

      Phi_eff->Data[i] = Phi_eff->Data[i+1] - dr * NEWTON_G * ( Mass_NW_rad - Mass_TOV_rad ) / SQR( Radius[i] );
   }

   double dr           = Radius[0] - Radius[1];
   double RadiusCubed  = CUBE( Radius[0] );
   double Mass_NW_rad  = Mass_NW[0];
   double Mass_TOV_rad = ( Mass_TOV[0] + FourPI * RadiusCubed * PresAve->Data[0] / c2 )
                       * ( 1.0 + ( EngyAve->Data[0] + PresAve->Data[0] ) / ( DensAve->Data[0] * c2 ) )
                       / SQR( Gamma_TOV[0] );


   Phi_eff->Data[0] = Phi_eff->Data[1] - dr * NEWTON_G * ( Mass_NW_rad - Mass_TOV_rad ) / SQR( Radius[0] ) ;


#ifdef GREP_DEBUG
   if ( ( MPI_Rank == 0 )  &&  ( omp_get_thread_num() == 0 ) )
   {
      printf("\n# GREP_CENTER_METHOD: %d\n",                  GREP_CENTER_METHOD);
      printf("# Center              : %.15e\t%.15e\t%.15e\n", Phi_eff->Center[0], Phi_eff->Center[1], Phi_eff->Center[2]);
      printf("# MaxRadius           : %.15e\n",               Phi_eff->MaxRadius);
      printf("# LogBin              : %d\n",                  Phi_eff->LogBin);
      printf("# LogBinRatio         : %.15e\n",               Phi_eff->LogBinRatio);
      printf("# Num of Iteration    : %d\n",                  GREP_MAXITER - NIter);
      printf("# NBin                : %d\n",                  NBin);
      printf("# ============================================================\n");
      printf("#  Bin     NCell     RADIUS       DENS       ENGY         Vr   Pressure    Mass_NW   Mass_TOV  Gamma_TOV    Eff_Pot\n");
      for ( int i=0; i<NBin; i++ )
         printf("%6d  %8ld  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e\n",
                i, DensAve->NCell[i], Radius[i],
                DensAve->Data[i], EngyAve->Data[i], VrAve->Data[i], PresAve->Data[i],
                Mass_NW[i], Mass_TOV[i], Gamma_TOV[i], Phi_eff->Data[i]);
   }
#endif

} // FUNCTION : CPU_ComputeEffPot


#endif // #if ( defined GRAVITY  &&  defined GREP )
