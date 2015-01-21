/* HDF.H */
/* Declaration for NSCA HDF routines */
/* See anonymous FTP address 128.174.20.50 for details */

#ifndef HDF_H
#define HDF_H

int DFR8setpalette (unsigned char* palette_array);

int DFR8putimage (char* filename, unsigned char* image, 
                  long xdim, long ydim, int compress);

int DFR8getimage (char* filename, char* image, 
                  long xdim, long ydim, unsigned char* palette);

int DFR8getdims
   (char *filename, long *pxdim, long *pydim, int *pispal);


#endif /* HDF_H */

