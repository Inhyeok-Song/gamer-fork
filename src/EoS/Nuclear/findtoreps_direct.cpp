#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



//-------------------------------------------------------------------------------------
// Function    :  BinarySearch
// Description :  Use the binary method to find the index
//
// Note        :  1. Invoked by findtoreps_direct()
//
// Return      :  Mid
//-------------------------------------------------------------------------------------
GPU_DEVICE
int BinarySearch( const real *table, const real target, int Min, int Max,
                  const int nx, const int ip )
{

   int Mid = -1;

   while (  ( Mid = ( Min + Max ) / 2 ) != Min  )
   {
      if   ( table[ip + nx * Mid] > target )   Max = Mid;
      else                                     Min = Mid;
   }

   return Mid;

} // FUNCTION : BinarySearch



//-------------------------------------------------------------------------------------
// Function    :  Bilinear
// Description :  Bilinear interpolation
//
// Note        :  1. Invoked by findtoreps_direct()
//
// Return      :  interpolated value
//-------------------------------------------------------------------------------------
GPU_DEVICE
real Bilinear( const real *table, const real frac_x, const real frac_y,
               const int ifs, const int nxy )
{

   const real left  = table[ifs      ] + ( table[ifs + 1      ] - table[ifs      ] ) * frac_x;
   const real right = table[ifs + nxy] + ( table[ifs + 1 + nxy] - table[ifs + nxy] ) * frac_x;

   return left + ( right - left ) * frac_y;

} // FUNCTION : Bilinear



//-------------------------------------------------------------------------------------
// Function    :  findtoreps_direct
// Description :  Finding temperature of internal energy from different modes
//                --> Energy      mode (0)
//                    Temperature mode (1)
//                    Entropy     mode (2)
//                    Pressure    mode (3)
//
// Note        :  1. Use 3D Lagrange linear interpolation or Catmull-Rom cubic interpolation formulae
//                   to search the corresponding temperature/energy given (rho, eps/temp/entr/pres, Ye)
//                2. Invoked by nuc_eos_C_short()
//
// Parameter   :  x             : Input vector of first  variable (rho)
//                var           : Input vector of second variable (eps/temp/entr/pres)
//                z             : Input vector of third  variable (Ye)
//                found_ltoreps : Output log(temp)/log(eps) of interpolated function values
//                table_main    : The main      tabulated nuclear EoS table
//                table_Aux     : The auxiliary tabulated nuclear EoS table
//                nx            : X-dimension of table
//                ny            : Y-dimension of table
//                nz            : Z-dimension of table
//                nvar          : Size of input quantity in the auxiliary table
//                xt            : Vector of x-coordinates of table
//                yt            : Vector of y-coordinates of table
//                zt            : Vector of z-coordinates of table
//                vart          : Vector of the input quantity
//                keymode       : Which mode to be used
//                keyerr        : Output error
//
// Return      :  found_ltoreps
//-------------------------------------------------------------------------------------
GPU_DEVICE
void findtoreps_direct( const real x, const real var, const real z,
                        real *found_ltoreps, const real *table_main, const real *table_aux,
                        const int nx, const int ny, const int nz, const int nvar,
                        const real *xt, const real *yt, const real *zt, const real *vart,
                        const int keymode, int *keyerr )
{

   int  iy_bdry_1, iy_bdry_2, ip, ifs;
   real var_bdry_1, var_bdry_2;

   const int nxy  = nx  * ny;
   const int nxyz = nxy * nz;


// determine spacing parameters and location in an equidistant table
   const real dx = (   xt[nx  -1] -   xt[0] ) / (real)(nx  -1);  // dens
   const real dz = (   zt[nz  -1] -   zt[0] ) / (real)(nz  -1);  // ye
   const real dv = ( vart[nvar-1] - vart[0] ) / (real)(nvar-1);  // var

   const real dxi = (real)1.0 / dx;
   const real dzi = (real)1.0 / dz;
   const real dvi = (real)1.0 / dv;

   const int  ix = (int)(  (   x -   xt[0] ) * dxi  );
   const int  iz = (int)(  (   z -   zt[0] ) * dzi  );
   const int  iv = (int)(  ( var - vart[0] ) * dvi  );  // index in the auxiliary table


// (0) check if the target variable is in the degeneracy region
   const int iv_left  =     (int)table_aux[ix + nx *  iv    + nxy * iz];  // index in the main table
   const int iv_right = 1 + (int)table_aux[ix + nx * (iv+1) + nxy * iz];  // index in the main table

   if (  ( iv_left < 0 )  ||  ( iv_right < 0 )  )   {  *keyerr = 660;  return;  }

// compute the fraction and the corresponding location in the main table
   if      ( keymode == NUC_MODE_ENGY )   ip = NUC_VAR_IDX_EORT;  // energy mode
   else if ( keymode == NUC_MODE_ENTR )   ip = NUC_VAR_IDX_ENTR;  // entropy mode
   else if ( keymode == NUC_MODE_PRES )   ip = NUC_VAR_IDX_PRES;  // pressure mode

   ip = ip * nxyz + ix + nxy * iz;

   const real fracx = ( x - xt[ix] ) * dxi;
   const real fracz = ( z - zt[iz] ) * dzi;


// (1-a) find the index `iy_bdry_1` s.t. var(ix, iy_bdry_1, iz) <= var < var(ix, iy_bdry_1+1, iz)
   iy_bdry_1 = BinarySearch( table_main, var, iv_left, iv_right, nx, ip );

   if ( iy_bdry_1 < 0 )   {  *keyerr = 660;  return;  }

// (1-b) find the value at the given density and Ye, and the input quantity at `iy_bdry_1`
   ifs        = ip + nx * iy_bdry_1;
   var_bdry_1 = Bilinear( table_main, fracx, fracz, ifs, nxy );


// (2) find the lower and upper boundaries iteratively
   bool bdry_found = false;

   if ( var_bdry_1 > var )
   {
//    (2-a) guess the solution located at index < iy_bdry_1, and find the lower boundary
      while ( iy_bdry_1 > iv_left )
      {
         iy_bdry_2  = iy_bdry_1 - 1;
         ifs        = ip + nx * iy_bdry_2;
         var_bdry_2 = Bilinear( table_main, fracx, fracz, ifs, nxy );

         if   ( var_bdry_2 < var )   {  bdry_found = true;  break;  }
         else                        {  iy_bdry_1 = iy_bdry_2;  var_bdry_1 = var_bdry_2;  }
      }
   }

   else
   {
//    (2-b) guess the solution located at index > iy_bdry_1, and find the upper boundary
      while ( iy_bdry_1 < iv_right )
      {
         iy_bdry_2  = iy_bdry_1 + 1;
         ifs        = ip + nx * iy_bdry_2;
         var_bdry_2 = Bilinear( table_main, fracx, fracz, ifs, nxy );

         if   ( var_bdry_2 > var )   {  bdry_found = true;  break;  }
         else                        {  iy_bdry_1 = iy_bdry_2;  var_bdry_1 = var_bdry_2;  }
      }
   } // if ( var_bdry_1 > var ) ... else ...


// (3) apply linear interpolation to compute the temperature / energy, if found
   if ( bdry_found == false )   {  *keyerr = 660;  return;  }

   const real fracy = ( var - var_bdry_1 ) / ( var_bdry_2 - var_bdry_1 );

   *found_ltoreps = yt[iy_bdry_1] + ( yt[iy_bdry_2] - yt[iy_bdry_1] ) * fracy;


   return;

} // FUNCTION : findtoreps_direct



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
