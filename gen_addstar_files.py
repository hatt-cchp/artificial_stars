#!/usr/bin/env

# PYTHON 2.X or 3.X


import argparse
import parser
import numpy as np
import bisect
from math import floor
from scipy.optimize import fsolve


def map_XY_coord(guess_coeff, params):
	"""
	Use the DAOPHOT .mch file coefficients
	to invert the DAOMASTER transformation equations
	and find where stars in one images should lie in another

	This function is combined with the fsolve routine to 
	numerically approximate the values x and y that solve
	the two non-linear transformation equations.
	"""

	x, y = guess_coeff

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

	#assert os.path.isfile(args.lfs[0]) 
	#assert os.path.isfile(args.lfs[1]) 



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
	parser.add_argument("-c", "--colors", type=float, nargs='+',
                help="Color range for the artifical stars",required=True)


	args = parser.parse_args()

	check_args_input(args)
	                                                                            
	return args

def count_stars_in_range(lum_fcn, start_range, end_range):

	num_stars=0.

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
	Written as separate function to generalize the 
	same operation for the RGB and AGB components
	"""

	# Always round down (floor) so that there is
        # never an index error at len(agb_lum_fcn) 

	pos_to_remove = floor( np.random.uniform()*len(lum_fcn) )
                                                                      
	lum_fcn.pop( pos_to_remove )

	return lum_fcn


def adjust_RGB_AGB_ratio(args,rgb_lum_fcn,agb_lum_fcn):
	"""

	Currently not optimized for speed since each element
	removal requires reallocation of memory for the entire
	luminosity function. In the case of small luminosity functions,
	say at most 2000 stars, this performs reasonably quickly.

	"""

	 # Decrease the total number of artificial stars
	 # until the desired number is reached.
	 # By default from gen_lf.py, there will be
	 # more than enough artficial stars sampled
	 # that one can reduce one population to meet
	 # the specified ratio.
	 
	 # At each iteration of reducing the number of
	 # artificial stars, try to 
	 
	 # Adjust the number of RGB:AGB number to the
	 # desired input: RGB_AGB_ratio
	 # From gen_lf.py, populations are 1:1, so
	 # decrease one population or the other to 
	 # achieve the correct balance

	while len(rgb_lum_fcn) + len(agb_lum_fcn) > args.nartstar:
	
		# Too many artificial stars, so remove a pair
		# and keep approx. the same ratio RGB:AGB

		rgb_lum_fcn = remove_star( rgb_lum_fcn )
		agb_lum_fcn = remove_star( agb_lum_fcn )

		current_ratio_at_TRGB = RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn)

		# Re-balance ratio of RGB:AGB
		                                                                    
		# Too few RGB, need to remove AGB
		if current_ratio_at_TRGB < args.RGB_AGB_ratio:

			print("AGB",current_ratio_at_TRGB)

			while current_ratio_at_TRGB < args.RGB_AGB_ratio:
				
				# Reduce AGB count by star_incr at random
				
				agb_lum_fcn = remove_star( agb_lum_fcn )

				# Calculate updated ratio

				current_ratio_at_TRGB = RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn)	
		               
			print("removing AGB",current_ratio_at_TRGB)                                                     

		# Too few AGB, need to move RGB
		elif current_ratio_at_TRGB > args.RGB_AGB_ratio: 

			print("RGB",current_ratio_at_TRGB)

			while current_ratio_at_TRGB > args.RGB_AGB_ratio:

				rgb_lum_fcn = remove_star( rgb_lum_fcn )
				                                                                                 
				current_ratio_at_TRGB = RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn)	

			print("removing RGB",current_ratio_at_TRGB)                                                     

		else: pass # Ratio is perfect, continue removing equal numbers of RGB and AGB until desired count is reached

	print(RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn),len(rgb_lum_fcn)+len(agb_lum_fcn))
	exit(0)

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

	# Re-balance the relative number of RGB:AGB stars
	# and ensure that the total number of stars does not 
	# exceed nartstar

	# Sort with increasing magnitude for some computation 
	# speed when adjusting for the RGB:AGB ratio

	rgb_lum_fcn.sort()
	agb_lum_fcn.sort()	

	rgb_lum_fcn, agb_lum_fcn = adjust_RGB_AGB_ratio(args, rgb_lum_fcn, agb_lum_fcn)	

	# Now that the luminosity functions have the right balance
        # of RGB and AGB populations, we need to adjust each luminosity
        # function such that n_rgb+n_agb <= nartstar, i.e.
        # we do not exceed the maximum we want within the images.
	# Remove both in equal numbers to ensure that the ratio
	# of RGB to AGB remains fixed
                                                                    
	#while len(rgb_lum_fcn) + len(agb_lum_fcn) > args.nartstar:
        #	
        #	rgb_lum_fcn = remove_star(rgb_lum_fcn)
        #	agb_lum_fcn = remove_star(agb_lum_fcn)

	# NEED TO SIMULTANEOUSLY APPROACH THE DESIRED NUMBER OF AGB STARS
	print(len(rgb_lum_fcn) + len(agb_lum_fcn),RGB_AGB_ratio_at_TRGB(args, rgb_lum_fcn, agb_lum_fcn))

	return [rgb_lum_fcn, agb_lum_fcn]


def gen_addstar_files(args, rgb_lum_fcn, agb_lum_fcn):
	"""
	"""

	# Read in the .mch file coefficients


	# Temp
	x0, y0 = 5,50

	# Use the reference coordinates x0, y0 as a first
	# guess for the mapped coordinates
	
	guess_coord = (x0, y0)

	# Coeffs are a list within the list of params
	#coeffs=[-3.5902,-3.6476,1.000055786,0.000057921,0.000120263,1.000246524,-0.16793649,-0.20200588,-0.14784137,0.156522376,-0.10951879,0.514530480,-0.17102246,0.297778067,0.354187830,0.152300631,0.220135105,-0.08607051,0.021139762,-1.08764398]
	coeffs=[0.,0.,1.,0.,0.,1.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] 

	# Solve for the star coordinates in the another image
	# using the DAOMASTER .mch coefficients

	params = [ x0, y0, args.imagedim[0], args.imagedim[1], coeffs ]
	
	x, y = fsolve(map_XY_coord, guess_coord, params)

	print(x,y)
	


if __name__ == "__main__":

	"""
	"""

	# Parse command line input

	args = collect_args()

	# Load the luminosity functions derived in gen_lf.py
	# Based on the desired ratio RGB:AGB, adjust the populations 

	rgb_lum_fcn, agb_lum_fcn = load_lfs(args)	

	# Generate addstar files with the luminosity functions
	# and the specified (uniform) color range

	gen_addstar_files(args, rgb_lum_fcn, agb_lum_fcn)



