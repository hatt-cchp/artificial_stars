#!/usr/bin/env

# PYTHON 2.X or 3.X


import argparse
import parser
import numpy as np
from scipy.optimize import fsolve
#import os.path


def map_XY_coord(guess_coeff, params):
	"""
	Use the DAOPHOT .mch file coefficients
	to invert the DAOMASTER transformation equations
	and find where stars in one images should lie in another
	"""

	x, y = guess_coeff

	# Reference coord, XY dim of images, .mch coefficients

	x0, y0, RCOL, RROW, coeffs = params

	XS = 2. * (x - 1.)/(RCOL - 1.) - 1.
	YS = 2. * (y - 1.)/(RROW - 1.) - 1.
	XY = XS * YS
	X2 = 1.5 * XS**2. - 0.5
	Y2 = 1.5 * YS**2. - 0.5

	LINEAR_TERMS_X = coeffs[0] + coeffs[2] * x + coeffs[4] * y
	LINEAR_TERMS_Y = coeffs[1] + coeffs[3] * x + coeffs[5] * y
	QUAD_TERMS_X = coeffs[6]*X2 + coeffs[8]*XY + coeffs[10]*Y2
	QUAD_TERMS_Y = coeffs[7]*X2 + coeffs[9]*XY + coeffs[11]*Y2
	CUBIC_TERMS_X = ( coeffs[12]*XS + coeffs[14]*YS )*X2 + ( coeffs[16]*XS + coeffs[18]*YS )*Y2
	CUBIC_TERMS_Y = ( coeffs[13]*XS + coeffs[15]*YS )*X2 + ( coeffs[17]*XS + coeffs[19]*YS )*Y2 


	return ( -x0 + LINEAR_TERMS_X + QUAD_TERMS_X + CUBIC_TERMS_X, -y0 + LINEAR_TERMS_Y + QUAD_TERMS_Y + CUBIC_TERMS_Y )
	

	


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
	parser.add_argument("-d", "--imagedim", type=float, nargs='+',
                help="The X and Y dimensions of the images.",required=True)
	parser.add_argument("-r", "--RGB_AGB_ratio", type=float,
                help="RGB:AGB at TRGB interface",default=1.0)
	parser.add_argument("-t", "--trgbrange", type=float,
                help="Magnitude range around TRGB considered part of 'TRGB interface'",default=0.25)
	parser.add_argument("-f", "--lfs", type=argparse.FileType('r'), nargs='+',
                help="The RGB and AGB luminosity functions",required=True)
	parser.add_argument("-c", "--colors", type=float, nargs='+',
                help="Color range for the artifical stars",required=True)


	args = parser.parse_args()

	check_args_input(args)
	                                                                            
	return args

def count_stars_at_TRGB(args,rgb_lum_fcn,agb_lum_fcn):
	"""
	"""
	pass

def adjust_RGB_AGB_ratio(args,rgb_lum_fcn,agb_lum_fcn):
	"""
	"""

	# Adjust the number of RGB:AGB number to the
	# desired input: RGB_AGB_ratio
	# From gen_lf.py, populations are 1:1, so
	# decrease one population or the other to 
	# achieve the correct balance
	
	temp_star_list = list() 	# To hold downsampled LF
	star_incr = 5 			# The step size to rebalance the LF
	current_ratio_at_TRGB = 9999.
	                                                                    
	if args.RGB_AGB_ratio > 1:	# RGB:AGB > 1, more RGB than AGB
		while current_ratio_at_TRGB < args.RGB_AGB_ratio:
			pass
			
	                                                                    
	else:				# RGB:AGB < 1, more AGB than RGB
		while current_ratio_at_TRGB > args.RGB_AGB_ratio:
			pass

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
	# if anythig other than the default 1.0 is specified	

	if args.RGB_AGB_ratio != 1.0: 
		rgb_lum_fcn, agb_lum_fcn = adjust_RGB_AGB_ratio(args, rgb_lum_fcn, agb_lum_fcn)	

	return [rgb_lum_fcn, agb_lum_fcn]


def gen_addstar_files(args, rgb_lum_fcn, agb_lum_fcn):

	
	# Temp
	x0, y0 = 5,50

	# Use the reference coordinates x0, y0 as a first
	# guess for the mapped coordinates
	
	guess_coord = (x0, y0)

	# Coeffs are a list within the list of params
	#coeffs=[-3.5902,-3.6476,1.000055786,0.000057921,0.000120263,1.000246524,-0.16793649,-0.20200588,-0.14784137,0.156522376,-0.10951879,0.514530480,-0.17102246,0.297778067,0.354187830,0.152300631,0.220135105,-0.08607051,0.021139762,-1.08764398]
	#coeffs=[0.,0.,1.,0.,0.,1.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] 

	params = [ x0, y0, args.imagedim[0], args.imagedim[1], coeffs ]
	
	x, y = fsolve(map_XY_coord, guess_coord, params)

	print(x,y)
	


if __name__ == "__main__":

	"""
	"""

	# Parse command line input

	args = collect_args()

	# Based on luminosity functions and the desired ratio RGB:AGB, adjust the populations
	# Load the luminosity functions derived in gen_lf.py

	rgb_lum_fcn, agb_lum_fcn = load_lfs(args)	

	# Generate addstar files with the luminosity functions
	# and the specified (uniform) color range

	gen_addstar_files(args, rgb_lum_fcn, agb_lum_fcn)

	# Write the luminosity function file

	#write_lf(args, lum_fcn)


