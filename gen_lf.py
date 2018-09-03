#!/usr/bin/env

# PYTHON 2.X or 3.X


import argparse
import parser
import numpy as np
import bisect
import matplotlib.pyplot as plt




def check_args_input(args):
	"""
	Force some of the command-line inputs
	to adhere to certain boundary conditions.

	Any condition not satisfied will produce an
	assertion error and exit the program.
	"""

	if args.nartstar > 2000: print("Warning: A large number of artificial stars can impact the accuracy of the photometry by introducing crowding effects.")
	assert args.mrange[0] < args.mrange[1], "The magnitude range must start with the lower bound. The lower bound also cannot be equal to the upper bound."


def collect_args():
	"""
	Parse command-line arguments and store as variable.
	"""

	parser = argparse.ArgumentParser(description='Inputs from command line.')
	parser.add_argument("-v", "--verbose", action="store_true",
		help="Increase output verbosity")
	parser.add_argument("-p", "--plots", action="store_true",
        	help="Show plots of probability mass function, CDFs, and the luminosity function.")
	parser.add_argument("-s","--slope", type=float, default=0.0, 
		help='Slope of luminosity function (dex). Default slope is zero (uniform distribution)')
	parser.add_argument("-nstar", "--nartstar", type=int, default = 2000, 
		help="number of artificial stars in the luminosity function")
	parser.add_argument("-mrange", "--mrange", type=float, nargs='+',
		help="Min then Max value of the luminosity function",required=True)
	parser.add_argument("-o", "--outfile", type=argparse.FileType('w'),
                help="filename to contain the output luminosity function", default="lum_fcn.dat")	

	args = parser.parse_args()

	check_args_input(args)
	                                                                            
	return args


def gen_prob_mass_fcn(args, star_mags):
	"""
	The relative abdunance of stars for a given slope
	can be evaluated by simplying evaluating the
	model luminosity function over the magnitude 
	range given as an input. This relative abundance,
	once normalized, is the probability mass function (PMF).
	
	Input: Command-line arguments, list of all possible star magnitudes
	Output: Probability mass function for the list of star magnitudes
	"""

	# Probability mass function follows the pre-defined slope or by default
	# a uniform distribution that can later (after photometered) be crafted 
	# to a luminosity function
	
	# The shape of this function is what matters, i.e.
	# the relative probabilities. The function's exponential 
	# is scaled down to avoid any possible floating point errors
	# if the magnitude range is massive, for example

	scale_fact = np.max(star_mags)

	prob_mass_fcn = [10.**( args.slope * x/scale_fact ) for x in star_mags]
	                                                                                       
	# Normalize the probability mass function
	# In general, this step is not necessary
	# but it is a good step for understanding 
	# how the CDF is produced.

	total_prob_mass = np.sum(prob_mass_fcn)

	norm_prob_mass_fcn = [ x/total_prob_mass for x in prob_mass_fcn ]

	return norm_prob_mass_fcn


def gen_lf_from_CDF(args, CDF, star_mags):
	"""
	From the CDF, one can construct the appropriate
	luminosity function for a random subset of stars
	for the specified input parameters through
	inverse transform sampling.

	From a random uniform distribution, find the
	closest location in the CDF. Find the position
	in star magnitudes that corresponds to that 
	location in the CDF and record the value.

	The list of retreived star magnitudes will
	follow the abundances that were determined
	in the PMF.

	Input: command-line arguments, cumulative distribution function
		of the specified luminosity funciton, and the list of
		possible star magnitudes.
	Output: A randomly sampled luminosity function that follows
		the specifications provided by the user with size
		equal to nartstar.
	
	"""

	# Array to hold the obtained star magnitudes
	# that model the luminosity function
	lum_fcn = np.zeros( args.nartstar )
	
	# Loop through and obtain star magnitudes for
	# the specified number of times

	for el in range( args.nartstar ):

		# Position in CDF, 0 to 1
		pos_in_CDF = np.random.uniform()

		# Get index of nearest position in CDF
		# that is greater than pos_in_CDF
		# CDF is sorted, so bisect (binary search)
		# will work faster on average

		nearby_index_in_CDF = bisect.bisect_left(CDF, pos_in_CDF)

		# Compare index_in_CDF-1 and index_in_CDF and use the index			
		# that returns the value closest to pos_in_CDF					
		
		# Cannot compare index < 0, so compare with index+1				
		if nearby_index_in_CDF == 0: nearby_index_in_CDF += 1				 

		dist_from_left  = np.abs( CDF[nearby_index_in_CDF - 1] - pos_in_CDF )
		dist_from_right = np.abs( CDF[nearby_index_in_CDF] - pos_in_CDF )

		if dist_from_left < dist_from_right:

			closest_index_in_CDF = nearby_index_in_CDF - 1
		else:
			closest_index_in_CDF = nearby_index_in_CDF

		# Add star with matching index to luminosity function

		lum_fcn[ el ] = star_mags[ closest_index_in_CDF ] 

	return lum_fcn


def gen_lum_fcn(args):
	"""
	Generates the luminosity function.
	
	In the case of a non-uniform distribution,
	the probability mass function is first defined, 
	from which the CDF is obtained. The luminosity
	function is then generated through inverse transform
	sampling from the random uniform distribution.

	In the case of a uniform distribution, the star
	magnitudes are sampled directly from the random
	uniform distribution. 

	Input:	command line arguments to build the luminosity function
	Output: the specified luminosity function
	"""

	# Generate array with all possible star magnitudes
	# 0.001 mag spacing is assumed for more-than-adequate
	# coverage of the spacing between magnitudes.

	star_mags = np.arange( args.mrange[0], args.mrange[1], 0.001) 

	# Obtain the probability mass function for star mags

	norm_prob_mass_fcn = gen_prob_mass_fcn(args, star_mags)

	# Cumulative distribution function for a given position x
	# is defined as the sum over the probability mass function
	# up to and including x, norm_prob_mass_fcn[0:x]

	CDF = np.cumsum( norm_prob_mass_fcn )
	
	# Construct luminosity function from inverse transforms sampling from a 
	# uniform random distribution

	lum_fcn = gen_lf_from_CDF( args, CDF, star_mags )

	# Show plots if requested

	if args.plots:
        	plt.scatter(star_mags,norm_prob_mass_fcn)
        	plt.ylim(0,0.001)
        	plt.show()

        	plt.scatter(star_mags,CDF)
        	plt.show()

        	plt.hist(lum_fcn)
        	plt.show()

	return lum_fcn


def write_lf(args, lum_fcn):
	"""
	Write the derived luminosity function to file.
	The filename is provided by the -o command line
	command.

	Input: The derived luminosity function
	Output: The luminosity function written to the
		working directory.
	"""

	with args.outfile as f:
		for el in lum_fcn:
			f.write( "{:15.3f}\n".format(el) )


if __name__ == "__main__":

	"""
	Returns an array of star magnitudes that
	follow the properties of the specificied 
	luminosity function (default uniform with
	slope zero). 

	Input: Min and Max magnitude range
	Output: File with random sample of star magnitudes
		 that follow the specified input.
	"""

	# Parse command line input

	args = collect_args()

	# Proceed to generate the luminosity function

	lum_fcn = gen_lum_fcn(args)

	# Write the luminosity function file

	write_lf(args, lum_fcn)


