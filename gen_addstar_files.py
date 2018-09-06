#!/usr/bin/env

# PYTHON 2.X or 3.X


import argparse
import parser
import numpy as np
import bisect
from math import floor
from scipy.optimize import fsolve
from astropy.io import fits


def map_XY_coord(guess_coord, params):
	"""
	Use the DAOPHOT .mch file coefficients
	to invert the DAOMASTER transformation equations
	and find where stars in one images should lie in another

	This function is combined with the fsolve routine to 
	numerically approximate the values x and y that solve
	the two non-linear transformation equations.
	"""

	x, y = guess_coord

	# Reference coord, XY dim of images, .mch coefficients

	x0, y0, RCOL, RROW, coeffs = params

	# Variables defined in daomaster.f

	XS = 2. * (x - 1.)/(RCOL - 1.) - 1.
	YS = 2. * (y - 1.)/(RROW - 1.) - 1.
	XY = XS * YS
	X2 = 1.5 * XS**2. - 0.5
	Y2 = 1.5 * YS**2. - 0.5

	# Transformation terms defined in daomaster.f

	LINEAR_TERMS_X = coeffs[0] + coeffs[2] * x + coeffs[4] * y
	LINEAR_TERMS_Y = coeffs[1] + coeffs[3] * x + coeffs[5] * y
	QUAD_TERMS_X = coeffs[6] * X2 + coeffs[8] * XY + coeffs[10] * Y2
	QUAD_TERMS_Y = coeffs[7] * X2 + coeffs[9] * XY + coeffs[11] * Y2
	CUBIC_TERMS_X = ( coeffs[12] * XS + coeffs[14] * YS ) * X2 + ( coeffs[16] * XS + coeffs[18] * YS ) * Y2
	CUBIC_TERMS_Y = ( coeffs[13] * XS + coeffs[15] * YS ) * X2 + ( coeffs[17] * XS + coeffs[19] * YS ) * Y2 

	# Simultaneously solve the system of equations and return the result

	return ( -x0 + LINEAR_TERMS_X + QUAD_TERMS_X + CUBIC_TERMS_X, 
		 -y0 + LINEAR_TERMS_Y + QUAD_TERMS_Y + CUBIC_TERMS_Y )
	

def check_args_input(args):
	"""
	Force some of the command-line inputs
	to adhere to certain boundary conditions.
	"""

	# Check existence of luminosity functions

	pass



def collect_args():
	"""
	Parse command-line arguments and store 
	as variable.
	"""

	parser = argparse.ArgumentParser(description='Inputs from command line.')
	parser.add_argument("-v", "--verbose", action="store_true",
		help="Increase output verbosity")
	parser.add_argument("-nstar", "--nartstar", type=int, default = 2000, 
		help="number of artificial stars in the luminosity function")
	parser.add_argument("-d", "--imagedim", type=float, nargs='+',
                help="The X and Y dimensions of the images.",required=True)
	parser.add_argument("-r", "--RGB_AGB_ratio", type=float,
                help="RGB:AGB at TRGB interface",default=1.0)
	parser.add_argument("-tr", "--trgbrange", type=float,
                help="Magnitude range around TRGB considered part of 'TRGB interface'",default=0.25)
	parser.add_argument("-tm", "--trgbmag", type=float,
                help="Magnitude of TRGB", required=True)
	parser.add_argument("-f", "--lfs", type=argparse.FileType('r'), nargs='+',
                help="The RGB and AGB luminosity functions",required=True)
	parser.add_argument("-mf", "--matchfile", type=argparse.FileType('r'),
                help="DAOMASTER match file",required=True)
	parser.add_argument("-c", "--colors", type=float, nargs='+',
                help="Color range for the artifical stars",required=True)

	args = parser.parse_args()

	check_args_input(args)
	                                                                            
	return args

def count_stars_in_range(lum_fcn, start_range, end_range):
	"""
	Since the start_range will always be brighter than
	the TRGB (start of RGB), the RGB should always start
	at index 0.	
	"""

	num_stars=0.

	# Get the first index that appears that matches
	# the input value. If not match exists, return
	# the next element (which will be the next highest
	# element since the list is monotonically increasing) 

	curr_position = bisect.bisect_left(lum_fcn,start_range)

	while lum_fcn[curr_position] <= end_range:
		num_stars += 1.
		curr_position += 1

	return num_stars

def RGB_AGB_ratio_at_TRGB(args,rgb_lum_fcn,agb_lum_fcn):
	"""
	"""

	start_range = args.trgbmag - args.trgbrange
	end_range   = args.trgbmag + args.trgbrange

	# Count the number of stars in magnitude range around the TRGB for each class

	num_RGB = count_stars_in_range(rgb_lum_fcn, start_range, end_range)
	num_AGB = count_stars_in_range(agb_lum_fcn, start_range, end_range)

	if num_RGB == 0 or num_AGB == 0:
		print("WARNING: All RGB or AGB stars at the TRGB interface have been removed!")
		exit(1)
	
	return num_RGB/num_AGB


def remove_star(lum_fcn):
	"""
	Want to remove stars from the entire 
	luminosity function instead of just at the
	TRGB interface. In some instances, the
	RGB:AGB will therefore be unchanged.

	Written as separate function to generalize the 
	same operation for the RGB and AGB components

	"""

	# Always round down (floor) so that there is
        # never an index error at len(agb_lum_fcn) 

	pos_to_remove = floor( np.random.uniform()*len(lum_fcn) )
                                                                      
	lum_fcn.pop( pos_to_remove )

	return lum_fcn


def rebalance_lfs(args,rgb_lum_fcn,agb_lum_fcn):
	"""
	If current RGB:AGB ratio does not match
	the specified value, re-balance ratio of RGB:AGB
	                                                                  	
	If too few RGB, need to remove AGB
	Keep removing AGB stars until ratio is met

	"""

	# desired RGB:AGB > current RGB:AGB

	while args.RGB_AGB_ratio > RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn):
		
		# Reduce AGB count
		
		agb_lum_fcn = remove_star( agb_lum_fcn )

	# desired RGB:AGB < current RGB:AGB

	while args.RGB_AGB_ratio < RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn):

		# Reduce RGB count

		rgb_lum_fcn = remove_star( rgb_lum_fcn )

	return [rgb_lum_fcn, agb_lum_fcn]


def adjust_RGB_AGB_ratio(args,rgb_lum_fcn,agb_lum_fcn):
	"""
	Decrease the total number of artificial stars
        until the desired number is reached.
        By default from gen_lf.py, there will be
        more than enough artficial stars sampled
        that one can reduce one population to meet
        the specified ratio.
                                                      
        At each iteration of reducing the number of
        artificial stars, try to 
                                                      
        Adjust the number of RGB:AGB number to the
        desired input: RGB_AGB_ratio
        From gen_lf.py, populations are 1:1, so
        decrease one population or the other to 
        achieve the correct balance

	Currently not optimized for speed since each element
	removal requires reallocation of memory for the entire
	luminosity function. In the case of small luminosity functions,
	say at most 2000 stars, this performs reasonably quickly.
	"""


	# Sort with increasing magnitude for some computation 
        # speed when adjusting for the RGB:AGB ratio
                                                               
	rgb_lum_fcn.sort()
	agb_lum_fcn.sort()	

	# Get initial ratio of RGB:AGB to match
	# the specified ratio
	
	rgb_lum_fcn, agb_lum_fcn = rebalance_lfs(args, rgb_lum_fcn, agb_lum_fcn)

	# Enforce max nartstar condition

	while len(rgb_lum_fcn) + len(agb_lum_fcn) > args.nartstar:
	
		# Too many artificial stars, so remove a pair
		# and keep approx. the same ratio RGB:AGB

		rgb_lum_fcn = remove_star( rgb_lum_fcn )
		agb_lum_fcn = remove_star( agb_lum_fcn )

		rgb_lum_fcn, agb_lum_fcn = rebalance_lfs(args, rgb_lum_fcn, agb_lum_fcn)

	return [rgb_lum_fcn, agb_lum_fcn]



def load_lfs(args):
	"""
	"""

	# Luminosity functions are written as single column, 
	# so for simplicity use built in Python line reader

	with args.lfs[0] as f:
		rgb_lum_fcn = [float(line) for line in f.readlines()]
	with args.lfs[1] as f:
        	agb_lum_fcn = [float(line) for line in f.readlines()]

	# Re-balance the relative number of RGB:AGB stars and ensure 
	# that the total number of stars does not exceed nartstar

	return adjust_RGB_AGB_ratio(args, rgb_lum_fcn, agb_lum_fcn) 


def extract_mch_file_contents(args):
	"""
	DAOMASTER .mch file has a rigid structure.
	Extracting information from the file therefore
	has a specific set of steps or components
	to ignore.
	"""

	# Read .mch file contents into a list

	with args.matchfile as f:  match_file_contents = f.readlines()


	# For convenience, combine the filter and exptime into a dictionary object
	# Otherwise store filename and transformation coefficients in lists

	coeffs    = list()
	filebasenames = list()
	filter_and_ref_exptime = {}	

	# Go line by line and parse the .mch file contents

	for line in match_file_contents:
	
		parts = line.split()

		# Remove apostrophe and extension from filename 

		filename_split = parts[0].split('\'')
		phot_filename = filename_split[1]
		filename_split = phot_filename.split('.')
		filebasename = filename_split[0]

		filebasenames.append(filebasename)

		# Transformation coefficients 

		coeff  = [ float(x) for x in parts[2:] ]

		# Remove 6th and 7th values: average mag offset 
		# and other non-essential info

		coeff.pop(6) # shifts list left
		coeff.pop(6)	

		coeffs.append(coeff)

		# Get unique filters, keep track of exptime of first instance of new filter
		# These will be used to set every artstar to the same magnitude, i.e.
		# account for exptime differences between exposures

		fits_filename = filebasename+'.fits'
		with fits.open(fits_filename) as hdu:

			curr_filter = (hdu[0].header['FILTER'])
		
			# If first instance of a filter in the .mch file

			if curr_filter not in filter_and_ref_exptime: 

				curr_exptime = (hdu[0].header['EXPTIME'])
				
				filter_and_ref_exptime[curr_filter] = curr_exptime


	return [filebasenames, coeffs, filter_and_ref_exptime] 


def calc_exptime_mag_diff(filebasename, filter_and_ref_exptime):
	"""
	Adjust the input magnitude so that the proper
	S/N of the star is given for each image
	"""

	fits_filename = filebasename+'.fits'
	with fits.open(fits_filename) as hdu:
	
		curr_filter = (hdu[0].header['FILTER'])
		curr_exptime = (hdu[0].header['EXPTIME'])
		
		# Will subtract off this value before writing artstar mag

		exptime_mag_diff = np.log10( curr_exptime/filter_and_ref_exptime[curr_filter] )

	return [curr_filter, exptime_mag_diff]


def gen_artstar_properties(args, rgb_lum_fcn, agb_lum_fcn):
	"""
	"""

	min_color, max_color = args.colors[0], args.colors[1]
	min_x_pos, max_x_pos = 0, args.imagedim[0]
	min_y_pos, max_y_pos = 0, args.imagedim[1]

	colors = [ min_color + np.random.uniform()*(max_color-min_color) for x in range( len(rgb_lum_fcn) + len(agb_lum_fcn) ) ]
	x_pos  = [ min_x_pos + np.random.uniform()*(max_x_pos-min_x_pos) for x in range( len(rgb_lum_fcn) + len(agb_lum_fcn) ) ]
	y_pos  = [ min_y_pos + np.random.uniform()*(max_y_pos-min_y_pos) for x in range( len(rgb_lum_fcn) + len(agb_lum_fcn) ) ]

	return [colors, x_pos, y_pos]



def gen_addstar_files(args, rgb_lum_fcn, agb_lum_fcn):
	"""
	Read in .mch file and 
		(1) add headers to .add files
		(2) store image transformation coefficients
		(3) record the image exposure times

	For each artificial star, assign it a color (using
	the base magnitude from the LFs plus some additional 
	value that is assigned randomly)

	Loop through each image and write the star to 
	its respective .add file. When second filter
	images are used, use the other color.

	"""

	# Assign random colors and positions 
	# Option to include own position input
	
	colors, x_pos, y_pos = gen_artstar_properties(args, rgb_lum_fcn, agb_lum_fcn)	
	
	# Read in the .mch file filenames, coefficients, and filter/exptime info

	filebasenames, coeffs, filter_and_ref_exptime = extract_mch_file_contents(args)

	# Primary filter is designated as the first one present in the mch file
	# The primary filter is assumed to be the longer wavelength passband,
	# and colors should be input with that notation in mind.

	for key in filter_and_ref_exptime:
		primary_filter = key
		break

	# Loop through each file, transform coordinates
	# for each images and add artificial stars to .add files

	for file_loop in range( len(filebasenames) ):

		# Adjust for exposure time differences to ensure the correct S/N
		# is given to each star. For the reference image (first appearance of 
		# the filter in the .mch file), this value is zero.

		curr_filter, exptime_mag_diff = calc_exptime_mag_diff( filebasenames[file_loop], filter_and_ref_exptime ) 

		# For each image, assign the correct star magnitudes and positions 

		for star_loop in range( len( rgb_lum_fcn) + len( agb_lum_fcn )  ):

			# Colors are assigned to the non-primary filter (any
			# filter that follows the filter for the first image
			# in the .mch file)			

			if curr_filter != primary_filter: 
				color_offset = colors[star_loop]
			else:	
				color_offset = 0.


			# Assign large ID numbers (permissible by DAOPHOT) to distinguish
                        # between real and artificial stars. RGB are set to 300k+
                        # and AGB are 400k+.

			if star_loop < len( rgb_lum_fcn): 
				id_el = int(3e5+star_loop)
				star_mag = rgb_lum_fcn[star_loop] + color_offset - exptime_mag_diff
			else: 
				id_el = int(4e5+star_loop)	
				star_mag = agb_lum_fcn[star_loop - len( rgb_lum_fcn )] + color_offset - exptime_mag_diff


			# Use the reference coordinates x0, y0 as a first		
                	# guess for the mapped coordinates. From coeffs,
			# these are the first two values

			guess_x, guess_y = coeffs[file_loop][0], coeffs[file_loop][1]
	
			guess_coord = (guess_x, guess_y)

			# Solve for the star coordinates in the another image
			# using the DAOMASTER .mch coefficients

			params = [ x_pos[star_loop], y_pos[star_loop], args.imagedim[0], args.imagedim[1], coeffs[file_loop] ]
			
			x, y = fsolve(map_XY_coord, guess_coord, params)


			
			print( '{:9d} {:<8.3f} {:8.3f} {:7.3f}'.format(id_el,x,y,star_mag) )





if __name__ == "__main__":
	"""
	Generate
	"""

	# Parse command line input

	args = collect_args()

	# Load the luminosity functions derived in gen_lf.py
	# Based on the desired ratio RGB:AGB, adjust the populations 

	rgb_lum_fcn, agb_lum_fcn = load_lfs(args)	

	# Generate addstar files with the luminosity functions
	# and the specified (uniform) color range

	gen_addstar_files(args, rgb_lum_fcn, agb_lum_fcn)



