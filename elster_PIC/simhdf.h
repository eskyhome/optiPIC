
/* SIMHDF.H */

#ifndef SIMHDF_H
#define SIMHDF_H

#include "hdf.h"

/* Keep max and min annotations in this structure */
#define ANNOT_BUF_LEN 1024
typedef struct {
  char buf[ANNOT_BUF_LEN];
  char *next_buf;
} Annot;

void white_palette (unsigned char* palette);
/* Color all 256 colors of the palette white.
 * palette should point to a character array of length 768.
 */

void black_palette (unsigned char* palette);
/* Color all 256 colors of the palette black */

void rainbow_palette (unsigned char* palette);
/* Define a rainbow-colored palette */

void palette_entry (unsigned char* palette,
		    int entry_index,int r,int g,int b);
/* Change color number entry_index to the color
 * red=r, green=g, blue=b. Leave the rest of the colors alone.
 */

void grayscale (unsigned char* palette);
/* Define a grayscale palette */

void maxminplot (Annot *a, double *x, int no_of_points,
		 int stride, char *xlabel, int pixel_color);
/* Write maximum and minimum values in the array x into 
 * the Annot structure, in a format compatible with put_annot. 
 * When viewed with Xhdf, the maximum and minimum values of 
 * the array x will appear in the plot, labeled by xlabel,
 * and in the color designated by pixel color.
 */

void maxminplot_over (Annot *a, double *x, int no_of_points,
		 int stride, char *xlabel, int pixel_color);
/* Same as maxminplot, but retain annotations from previous
 * calls to maxminplot or maxminplot_over.
 */

void scatterplot(unsigned char* image, int nxpix,int nypix,
     double *x, double *y, int no_of_points, int stride,
     double xmin,double xmax,double ymin,double ymax,int pixel_color);
/* Plot (x[i],y[i]) in the HDF image array image, where i runs
 * from 0 to no_of_points-1. Only every stride-th point will
 * be plotted. nxpix and nypix are the dimensions of the 
 * image array in pixels.  It is assumed that the image array
 * has dimension at least nxpix*nypix. xmin, xmax, ymin, ymax
 * correspond to the edges of the plot. If any point (x[i],y[i]) 
 * lies outside this range, it will not appear on the plot.
 */

void scatterplot_over(unsigned char* image, int nxpix,int nypix,
     double *x, double *y, int no_of_points, int stride,
     double xmin,double xmax,double ymin,double ymax,int pixel_color);
/* Same as scatterplot, but the array image is not pre-cleared, so
 * points are drawn over any existing image.
 */

void scatterplot_select (unsigned char* image, int nxpix,int nypix,
     double *x, double *y, double *z, int no_of_points, int stride,
     double xmin, double xmax, double ymin, double ymax,
     double zmin, double zmax, int pixel_color_base, int ncolors);
/* Scatterplot routine which plots (x[i],y[i]) only if z[i] is
 * between zmin and zmax. If so, the pixel used to plot the point
 * will be between pixel_color_base and pixel_color_base+ncolors-1
 * with each pixel in this range representing a value of z[i]
 * in one of ncolors equally divided intervals between zmin and
 * zmax.
 */

void scatterplot_shift (unsigned char* image, int nxpix,int nypix,
     double *x, double *y, double *z, int no_of_points, int stride,
     double xmin, double xmax, double ymin, double ymax,
     double zmin, double zmax, double shift, 
     int pixel_color_base, int ncolors);
/* Same as scatterplot_select, except that each color is shifted
 * from the base color in the y-direction by multiples of shift.
 */

void lineplot(unsigned char* image, int nxpix, int nypix,
     double *x, double *y, int no_of_points, int stride,
     double xmin,double xmax,double ymin,double ymax,int pixel_color);
/* Same as scatterplot, except the points are connected with lines. */

void lineplot_over(unsigned char* image, int nxpix, int nypix,
     double *x, double *y, int no_of_points, int stride,
     double xmin,double xmax,double ymin,double ymax,int pixel_color);
/* Same as lineplot, but the array image is not pre-cleared, so
 * lines are drawn over any existing image.
 */
 
void color_plot (unsigned char* image, int nxpix, int nypix,
 		 double *v, int ngx, int ngy,
	          double vmin, double vmax, int swap);
/* Assigns the palette entry no.s 1 through 254 inclusive to the
 * image according to the values in the array v, assumed to be
 * organized as a ngx by ngy two-dimensional array.
 * If swap is non-zero, the axes are swapped.
 */

void put (char* filename, unsigned char* image,
          int nxpix, int nypix, unsigned char* palette);
/* Put the image stored in the array image[nypix][nxpix] and
 * the palette stored in palette into an HDF file named filename.
 */
 
 void put_annot (char* filename, Annot *a);
 /* Put the annotation in Annot into the HDF file */

/****************************************************************
 * Special plotting functions for the sudden cardiac death	*
 * syndrome simulation.						*
 ****************************************************************/

void color_polar_palette (unsigned char *palette);
/* Produces a color palette which paints the different
 * phases theta=arctan(w/v) with different color and the
 * the different magnitudes r=sqrt(v*v+w*w) with different
 * intensities when used with vwonxy_plot.
 */

void vwonxy_plot (unsigned char* image, int nxpix, int nypix,
         	  double *v, double *w, int ngx, int ngy,
		  double rmax);
/* Plots v and w as functions of x and y when used with the
 * the palette produced by the routine color_polar_palette.
 */

void xyonvw_plot(unsigned char* image, int nvpix, int nwpix,
                 double *v, double *w, int ngx, int ngy,
	         double rmax, int pixel_color); 
/* Plots the x-y grid in v-w space according to the values
 * of v and w, assumed arrange in a two-dimensional ngx-by-ngy
 * array.
 */

void xyonvw_plot_over (unsigned char* image, int nvpix, int nwpix,
                 double *v, double *w, int ngx, int ngy,
	         double rmax, int pixel_color); 
/* Same as xyonvw_plot except the image is not pre-cleared.
 */

#endif /* SIMHDF_H */

