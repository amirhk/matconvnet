/*
calculates MMD 
*/

#include <stdlib.h>
#include <mex.h>
#include <math.h>

/* 32 bit unsigned integers [on Pentium Linux PC] */
typedef unsigned long uint32;

#define PREALLOC 50000

#if !defined(max)
#define max(A, B) ((A) > (B) ? (A) : (B))
#endif
#if !defined(min)
#define min(A, B) ((A) < (B) ? (A) : (B))
#endif


int getDouble(const mxArray *arg, double *x) {
  int m, n;

  if (!arg) return -1;
  m = mxGetM(arg);
  n = mxGetN(arg);
  if (!mxIsNumeric(arg) || mxIsComplex(arg) ||
      mxIsSparse(arg)  || !mxIsDouble(arg) ||
      (m != 1) || (n != 1)) {
    *x = 0.0;
    return -1;
  }
  *x = mxGetScalar(arg);
  return 0;
}


int getDoubleArray(const mxArray *arg, double **x, uint32 *m, uint32 *n) {
  if (!arg) return -1;
  *m = (uint32) mxGetM(arg);
  *n = (uint32) mxGetN(arg);
  if (!mxIsNumeric(arg) || mxIsComplex(arg) ||
      mxIsSparse(arg)  || !mxIsDouble(arg) ) {
    *x = NULL;
    return -1;
  }
  *x = mxGetPr(arg);
  return 0;
}



void getmom4(double *U , int n, double * U_m4) {
  /* calculates the 4th moment (only first term!)*/

  uint32 a,b,c,d;

  double intc=0,intd = 0,intab=0;
  double ntmp;
  double nd;

  /* delete diag*/
  for (a=0;a<n*n; a+= n+1) {
    U[a] = 0;
  }

  nd = (double) n;

  intab=0;
  for (a=0;a<n;a++) {
    for (b=a+1 ; b<n; b++) {
      intc = 0;
      for (c=0;c<n;c++) {
	intd = 0;
	for (d=0 ; d<n; d++) {
          intd += U[n*a + d] * U[n*c + d];
	}
	intc += intd/(nd-2) * U[n*b + c];
      }
      intab += U[n*a + b] * intc/(nd-1);
    }
  }


  ntmp = (nd*(nd-1));

  U_m4[0] = 64*(nd-2)*(nd-3)*intab/ntmp/ntmp/ntmp/ntmp;

  /* times 2; somehow...*/
  U_m4[0] = U_m4[0] * 2;


}


void printUsage(void) {
   mexPrintf("Usage: [U_m4] = U4thmoment(U);\n\n");
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  uint32 n,dim;
  double * U;
  double mom4;

  /*if (sizeof(uint32) != 4)
    mexErrMsgTxt("There is something wrong with uint32!!");*/


  if ((nlhs != 1) || (nrhs != 1)) {
    printUsage();
    return;
  }


  if (getDoubleArray(prhs[0], &U, &n, &dim)) {
    printUsage();
    mexPrintf("\tU is not a double array!\n");
    return;
  }

  if (n!=dim) {
    printUsage();
    mexPrintf("\t square matrix expected!\n");
    return;
  }
  
  getmom4(U , n, &mom4);
  
  plhs[0] = mxCreateNumericMatrix(1,1, mxDOUBLE_CLASS, mxREAL);

  memcpy(mxGetPr(plhs[0]), &mom4, sizeof(double));


}

