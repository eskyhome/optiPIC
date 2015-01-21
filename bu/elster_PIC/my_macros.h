/***********************************************************************/
/*****************  HEADER FILE "my_macros.h" *******************************/
/***********************************************************************/

/*	Author: Anne C. Elster				July 1991
 *
 *	"my_macros.h" is a C header file with preprocessor (macro) definitions
 *	that aids in making C more compact and readable from a FORTRAN
 *	point of view. Also includes useful macros such as MIN, MAX,
 *	and SWAP.
 *
 *	This header file is used extensively in the authors C BLAS and
 *	LINPACK-like libraries (based on efforts by Dongarra, Moler et. al.)
 *	upon which her Parallel (distributed) BLAS are being built.
 *	
 *  	Several of the definitions can be found in "C Tools for Scientists
 *	and Engineers" by L. Baker. Additions are by A.C. Elster.
 *
 *	PURPOSE: perform in-line a number of useful chores including loops,
 *		 array subscripting, and finding minimum, maximum, and
 *		 absolute value.
 *
 *		 SWAP and MALLOC added July 1992.
 *
 * 	DEPENDENCIES: none.
 *
 *	USAGE:
 * 
 *		invoke with preprocessor (macro) directive:
 *
 *			#include "ftoc.h"  [search source directory first] 
 *				or
 *			#include <ftoc.h>  [implementation dependant]
 *
 *		near the beggining of your program.
 *
 *	DISCLAIMER/WARNING:
 *
 *		To paraphrase netlib: These functions are free, but 
 *					come with no guarantee!
 */

/* in-line macro for declaring a 2D matrix as a 1-D array (to be used
 * with L0INDEX or LINDEX for column-storage */

#define LMATRIX(X,n,m,type) \
  type X[m*n]

#define MATRIX(X,i,j,type) \
  type X[i*j]           /* def. for creation of 1-D matrix arrays */


/* in-line indexing function simulating 2-D storage A(0,0) - A(m-1, n-1): */

#define L0INDEX(X,m,i,j) \
  A[((j)*m) + (i)]                   
                              

#define A0INDEX(X,m,i,j) \
  X[((i)*m) + (j)]                            /* m = leading row dimension */
                                        /* starting from (0,0) */


/* in-line indexing function simulating 2-D storage A(1,1) - A(m, n): */

#define LINDEX(A,m,i,j) \
  A[((j-1)*m) + (i) - 1]

#define ACOLINDEX(X,m,i,j) \
  X[((j-1)*m) + (i) - 1]                    /* m = leading column dim */

#define AINDEX(X,m,i,j) \
  X[((i - 1)*m) + (j)-1]                    /* m = leading row dimension */
                                        /* starting from (1,1) */



/*in-line functions for use with 2D arrays: */

/* row-major order as in C, indecies run 0 .. n-1 as in C: */
#define INDEX(i,j) [j+(i)*coln]

/* row-major order as in , indecies run 1..n: */
/*	#define INDEX1(i,j) [(j-1)+(i-1)*n]
*/

/* column major order, as in fortran: */
#define INDEXC(i,j) [i-1+(j-1)*rown]

/* Usage: if A(30,30) is matrix, then (i,j) in C will be
	#define INDEX(i,j), if n = 30. */

/* various loop constructors: */
#define D0FOR(i,to) for(i=0; i<to; i++)
#define D1FOR(i,to) for(i=1; i<=to; i++)

#define DOFOR(i,to) for(i=0;i<to;i++)
#define DFOR(i,from,to) for(i=from-1;i<to;i++)
#define DOBY(i,from,to,by) for(i=from-1;i<to;i+=by)
#define DOBYY(i,from,to,by) for(i=from;i<to;i+=by)
#define DOBYYY(i,from,to,by) for(i=from;i<to;i++)
#define DOV(i,to,by) for(i=0;i<to;i+=by)

/* to index vectors starting with 1 */
#define VECTOR(i) [i-1]


/* some useful macro-functions: */

//#define MAX(a, b) ( ((a) > (b)) ? (a) : (b) )
//#define MIN(a, b) ( ((a) < (b)) ? (a) : (b) )
#define ABS(x)	( ((x) > 0.) ? (x) : -(x))

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr  /* used by 1D-fft routine */

#define MALLOC(type,nn) (type *)malloc(sizeof(type) * nn)



/* The following are defined in f2c.h provided by the f2c translator: 
#define max(a, b) ( ((a) > (b)) ? (a) : (b) )
#define min(a, b) ( ((a) < (b)) ? (a) : (b) )
#define abs(x)	( ((x) > 0.) ? (x) : -(x))

*/



