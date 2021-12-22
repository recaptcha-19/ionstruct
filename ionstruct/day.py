import numpy as np
import pandas as pd
import glob
import re
import itertools
import os

pattern1 = r'[0-9]+'
pattern = r'[0-9]{3}'

def day(files):
	"""Combines hourly information on TEC into blocks of information corresponding to one day
	
	Args:
		files (list): List of strings of file names
		
	Returns:
		dict: A dictionary whose keys are day and year, and values are Pandas dataframes corresponding to information from that day and year.
	"""
	titles = ['week', 'TOW', 'SVID', 'azimuth', 'elevation', 'TEC', 'locktime']
	years = []	#list of years in the files
	all_dfs = {}	#dictionary with all dataframes 
	all_year_days = {}	#nested list with all days in each year
	
	for file in files:
		x = re.findall(pattern1, file)
		#print(x)
		years.append(x[1])

	years = list(set(years))
	years.sort()
	year_files = {}	#list of all files in a year
	for i in range(len(years)):
		year_file = [file for file in files if file[-8:-6] == years[i]]
		year_files[years[i]] = year_file
	
	for i in year_files:
		year_days = []	#list of all days in a year
		for file in year_files[i]:
			x = re.findall(pattern, file)
			year_days.append(x)
		year_days = np.unique(np.asarray(year_days))
		#year_days.sort()
		all_year_days[i] = year_days

	#print(all_year_days)
		
	for year in all_year_days:
		days = all_year_days[year]
		for day in days:
			day_files = [file for file in year_files[year] if day in file]	#list of all files corresponding to a particular day
			list_day = []	#list of dataframes in a day
			for x in day_files:
				sat_data = np.loadtxt(x, delimiter = ',', usecols = [0,1,2,4,5,22,41])
				df = pd.DataFrame(sat_data, columns = titles)
				list_day.append(df)	
			#print(len(list_day))
			df_day = pd.concat(list_day)	#single dataframe for an entire day
			all_dfs["{}_{}".format(day, year)] = df_day	#master dataframe for the entire dataset	

	return all_dfs
	
	
def block_creator(path, block_size):
	"""Combines hourly information on TEC into blocks of information based on user preference
	
	Args:
		path (str): Full path of files 
		block_size (int): Block size (can be 10 or 100)
		
	Returns:
		dict: A dictionary whose key has two parts: the first and second parts represent the block and year respectively. Values are Pandas dataframes corresponding to that block and year.
	"""
	files = glob.glob(path)
	if block_size == 10:
		block = [file_name[-13:-11]+'_'+file_name[-8:-6] for file_name in files]
	if block_size == 100:
		block = [file_name[-13:-12]+'_'+file_name[-8:-6] for file_name in files]
	block = list(set(block))
	block.sort()
	#print(block)
	
	comb_dict = {}
	for days in block:
		day_comb, year = days.split('_')
		label = path.format(day_comb, year)
		#print(label)
		dfs = day(glob.glob(label))
		#print(dfs)
		dfs_list = []
		for st in dfs:
			dfs_list.append(dfs[st])
		block_df = pd.concat(dfs_list)
		comb_dict[days] = block_df
	return comb_dict
	

#files = glob.glob("*.ismr")
#block_list = block_creator(files)
#parent_dir = '/Data/rpriyadarshan/ismr/gcd_mean_el_plots'
#for directory in block_list:
#	path = os.path.join(parent_dir, directory)
#	os.mkdir(path)

		
path = "/Data/rpriyadarshan/ismr/ismr_files/PUNE3???.17_.ismr"
df = block_creator(path, block_size = 100)
print(df)

