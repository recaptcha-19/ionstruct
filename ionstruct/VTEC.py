"""Functions to provide visualizations of Vertical Total Electron Content (VTEC)"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import glob
from day import day, block_creator
import datetime
import mapping_functions as mapf
from itertools import combinations
from iri2016 import timeprofile

#rc("text", usetex = True)
R_earth = 6371
h_ion = 350
d = 100
lat_GMRT = 19.0919   #latitude of GMRT
long_GMRT = 74.0506  #longitude of GMRT

def TOW2UT(TOW):
	"""Time Of Week (TOW) to Universal Time (UT)
	
	Converts time in seconds measured since the start of the week to time of the day in Universal Time.
	
	Args:
		TOW (int): Time of Week in seconds
		
	Returns:
		str: Time in Universal Time Coordinates
	"""
	TOW = TOW%86400
	x = str(datetime.timedelta(seconds = TOW))
	return x
	
def UT2TOW(UT):
	"""Universal Time (UT) to Time Of Week (TOW)
	
	Converts time of the day in Universal Time to time in seconds measured since the start of the week.
	
	Args:
		UT (str): Universal Time
		
	Returns:
		int: Time Of Week (TOW)
	"""
	hrs, mins, sec = UT.split(':')
	hrs = int(hrs)
	mins = int(mins)
	sec = int(sec)
	TOW = hrs*3600 + mins*60 + sec
	return TOW

def rms(array):
	return np.sqrt(np.mean(array**2))

def VTEC(STEC, elevation, map_func):
	"""Vertical Total Electron Content (VTEC)
	
	Vertical Total Electron Content (VTEC) given a Slant Total Electron Content (STEC) value, angle of elevation and choice of mapping function.
	
	Args:
		STEC (float): Slant Total Electron Content measurement in TEC units
		elevation (float): Angle of elevation in degrees
		map_func (str): Choice of mapping function
		
	Returns:
		float: Vertical Total Electron Content measurement in TEC units
	"""
	map_method = getattr(mapf, map_func)
	VTEC = STEC/map_method(elevation)
	return VTEC

def clean(df, GPS = False, elevation = False, TEC = False, VTEC = False, locktime = False):
	"""Clean
	
	Removes erroneous values of VTEC
	
	Args:
		df (dataframe): Master dataframe containing TEC measurements
		GPS (bool): (default: False) If True, only GPS satellite data is included and vice-versa
		elevation (bool): (default: False) If True, only data for which elevation is greater than 30 degrees is included
		TEC (bool): (default: False) If True, only data with positive TEC values are included and vice-versa
		VTEC (bool): (default: False) If True, only data with positive VTEC values are included and vice-versa
		locktime (bool): (default: False) If True, only data with locktime greater than 3 minutes are included and vice-versa
	"""
	if elevation == True:
		df = df[df['elevation'] > 30]
	if TEC == True:
		df = df[df['TEC'] > 0]
	if VTEC == True:
		df = df[df['VTEC'] > 0]
	if locktime == True:
		df = df[df['locktime'] > 180]
	if GPS == True:
		df = df[(df['SVID'] >= 1) & (df['SVID'] <= 37)]
	return df	

def VTEC(STEC, elevation, map_func):
	"""Estimates VTEC from STEC provided an elevation and mapping function
	
	Args:
		STEC (float): Slant Total Electron Content in TEC units
		elevation (float): Angle of elevation
		map_func (str): String describing choice of mapping function
		
	Returns:
		float: Vertical Total Electron Content in TEC units
	
	"""
	map_method = getattr(mapf, map_func)
	VTEC = STEC/map_method(elevation)
	return VTEC

def great_circle_distance(el1, el2, az1, az2):
	"""Great Circle Distance
	
	Returns the great circle distance between two points
	
	Args:
		el1 (float): Elevation of 1st point
		el2 (float): Elevation of 2nd point
		az1 (float): Azimuth of 1st point
		az2 (float): Azimuth of 2nd point
		
	Returns:
		float: Great circle distance in degrees
	"""
	el1 = np.deg2rad(el1)
	el2 = np.deg2rad(el2)
	az1 = np.deg2rad(az1)
	az2 = np.deg2rad(az2)
	delta_el = el1 - el2
	delta_az = az1 - az2
	gcd = 2*np.arcsin(np.sqrt((np.sin(delta_el/2))**2 + np.cos(el1)*np.cos(el2)*(np.sin(delta_az/2))**2))
	return gcd

def VTEC_time(all_dfs, map_fn):
	"""Displays scatter plot(s) showing variation of VTEC with time for a set of day(s)
	
	Args:
		all_dfs (dict): Master dataframe containing TEC information for the given set of days. Obtained by running day.day()
		map_fn (str): String describing choice of mapping function
		  
	"""
	
	for day in all_dfs:
		print(day)
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])
		el = np.unique(df_day['elevation'])
		
		l = 0
		for satellite in SVID:
			df_sat = df_day.loc[df_day['SVID'] == satellite, ['TOW', 'elevation', 'TEC', 'locktime']]
			#df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']] 
			df_p = clean(df_sat, elevation = True, TEC = True)
			
			minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation', 'locktime']]
			minlock['VTEC'] = VTEC(minlock['TEC'], minlock['elevation'], map_func = map_fn)
			minlock_VTEC = clean(minlock, VTEC = True)
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation', 'locktime']]
			maxlock['VTEC'] = VTEC(maxlock['TEC'], maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = clean(maxlock, VTEC = True)
			
			plt.figure(0)
			plt.scatter(minlock_VTEC['TOW'], minlock_VTEC['VTEC'], marker = 's', c = minlock_VTEC['elevation'], label = "Locktime $\leq$ 3 min" if l == 0 else "")
			plt.scatter(maxlock_VTEC['TOW'], maxlock_VTEC['VTEC'], marker = 'o', s = 3, c = maxlock_VTEC['elevation'], label = "Locktime $>$ 3 min" if l == 0 else "")
			l += 1
			
		h = (np.max(TOW) - np.min(TOW))/5
		t0 = np.min(TOW)
		t1 = np.min(TOW) + h
		t2 = np.min(TOW) + 2*h
		t3 = np.min(TOW) + 3*h
		t4 = np.min(TOW) + 4*h
		t5 = np.max(TOW)
		
		plt.figure(0)
		x = plt.clim(30,90)
		plt.colorbar()
		plt.xlabel("Time (UT)")
		plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
		plt.ylabel("Vertical TEC (TECU)")
		d, y = day.split("_")
		plt.title("Vertical TEC ({}-{})".format(d, y))
		plt.legend()
		plt.grid()
		plt.show()
		#plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_VTEC_{}.png".format(day, map_fn, day))
		#print("Saved")
		#plt.close()
		
	
def VTEC_STEC(all_dfs, map_fn):
	"""Displays scatter plot(s) showing variation of VTEC with STEC for a set of day(s)
	
	Args:
		all_dfs (dict): Master dataframe containing TEC information for the given set of days. Obtained by running day.day()
		map_fn (str): String describing choice of mapping function
		
	"""

	for day in all_dfs:
		print(day)
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])
		el = np.unique(df_day['elevation'])
		
		l = 0
		for satellite in SVID:
			df_sat = df_day.loc[df_day['SVID'] == satellite, ['TOW', 'elevation', 'TEC', 'locktime']]
			#df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']] 
			df_p = clean(df_sat, elevation = True, TEC = True)

			minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation']]
			minlock['VTEC'] = VTEC(minlock['TEC'], minlock['elevation'], map_func = map_fn)
			minlock_VTEC = clean(minlock, VTEC = True)
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock['VTEC'] = VTEC(maxlock['TEC'], maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = clean(maxlock, VTEC = True)

			plt.figure(1)
			plt.scatter(minlock_VTEC['TEC'], minlock_VTEC['VTEC'], marker = 's', c = minlock_VTEC['elevation'], label = "Locktime $\leq$ 3 min" if l == 0 else "")
			plt.scatter(maxlock_VTEC['TEC'], maxlock_VTEC['VTEC'], marker = 'o', s = 3, c = maxlock_VTEC['elevation'], label = "Locktime $>$ 3 min" if l == 0 else "") 
			l += 1
	
		plt.figure(1)
		x = plt.clim(30,90)
		plt.colorbar()
		plt.xlabel("Slant TEC (TECU)")
		plt.ylabel("Vertical TEC (TECU)")
		d, y = day.split("_")
		plt.title("VTEC vs STEC ({}-{})".format(d, y))
		plt.legend()
		plt.grid()
		plt.show()
		#plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_VTEC_STEC_{}.png".format(day, map_fn, day))
		#print("Saved")
		#plt.close()
	
	
def VTEC_averaged(all_dfs, map_fn):	
	"""Displays plot(s) showing variation of mean, median and RMS VTEC with time for a set of day(s)
	
	Args:
		all_dfs (dict): Master dataframe containing TEC information for the given set of days. Obtained by running day.day()
		map_fn (str): String describing choice of mapping function
		
	"""
	for day in all_dfs:
		print(day)
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])
		el = np.unique(df_day['elevation'])
		
		start_t = TOW[2]
		end_t = TOW[-2]
		iri_time = np.arange(start_t, end_t, 1*60)
		start_time = TOW2UT(start_t)
		end_time = TOW2UT(end_t)
		dayOfYear, year = day.split("_")
		Year = '20' + year
		d = datetime.datetime.strptime('{} {}'.format(dayOfYear, Year),'%j %Y')
		calendar_day = d.strftime('%Y-%m-%d')
		start = calendar_day + "T{}".format(start_time)
		end = calendar_day + "T{}".format(end_time)
		print(start_time)
		print(end_time)
		time_prof = timeprofile((start, end), datetime.timedelta(minutes = 1), [0, 2000, 10], lat_GMRT, long_GMRT)
		iri_TEC = time_prof.TEC/10**16
		plt.figure()
		plt.plot(iri_time, iri_TEC, '--', c = 'g', label = 'IRI')

		mean_VTEC = np.array([])
		median_VTEC = np.array([])
		RMS_VTEC = np.array([])
		for time in TOW:
			
			df_sat = df_day.loc[df_day['TOW'] == time, ['TOW', 'elevation', 'TEC', 'locktime']]
			#maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock = clean(df_sat, elevation = True, TEC = True, locktime = True)
			#maxlock['VTEC'] = maxlock['TEC']/map_method(maxlock['elevation'])
			maxlock['VTEC'] = VTEC(maxlock['TEC'], maxlock['elevation'], map_func = map_fn)
			#maxlock_VTEC = maxlock.loc[maxlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]
			maxlock_VTEC = clean(maxlock, VTEC = True)

			m_VTEC = np.mean(maxlock_VTEC['VTEC'])
			mean_VTEC = np.append(mean_VTEC, m_VTEC)
			med_VTEC = np.median(maxlock_VTEC['VTEC'])
			median_VTEC = np.append(median_VTEC, med_VTEC)
			R_VTEC = rms(maxlock_VTEC['VTEC'])
			RMS_VTEC = np.append(RMS_VTEC, R_VTEC)
	
		plt.plot(TOW, mean_VTEC, c = 'blue', label = "$VTEC_{mean}$")
		plt.plot(TOW, median_VTEC, c = 'red', label = "$VTEC_{median}$")
		plt.plot(TOW, RMS_VTEC, c = 'black', label = "$VTEC_{RMS}$")
		h = (np.max(TOW) - np.min(TOW))/5
		t0 = np.min(TOW)
		t1 = np.min(TOW) + h
		t2 = np.min(TOW) + 2*h
		t3 = np.min(TOW) + 3*h
		t4 = np.min(TOW) + 4*h
		t5 = np.max(TOW)
		plt.xlabel("Time (UT)")
		plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
		plt.ylabel("Vertical TEC (TECU)")
		d, y = day.split("_")
		plt.title("Vertical TEC ({}-{})".format(d, y))
		plt.legend()
		plt.grid()
		plt.show()
		#plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/IRI_GPS_only_{}_VTEC_averaged_{}.png".format(day, map_fn, day))
		#print("Saved")
		#plt.close()
		

def VTEC_comparison(all_dfs, map_fn, print_output = False):

	for day in all_dfs:
		year = day[4:]
		df_day = all_dfs[day]
		
		SVID = np.unique(df_day['SVID'])

		#df = df_day.loc[(df_day['elevation'] > 30) & (df_day['elevation'] < 35) & (df_day['locktime'] > 180) & (df_day['TOW'] == 600), ['SVID', 'TOW', 'TEC', 'elevation']]
		#df = df_day.loc[(df_day['locktime'] > 180) & (df_day['elevation'] > 30) & (df_day['TEC'] > 0), ['SVID', 'TOW', 'TEC', 'elevation']]
		df = clean(df_day, elevation = True, TEC = True, locktime = True)
		el = np.unique(df['elevation'])
		el = el[~np.isnan(el)]
		df['VTEC'] = VTEC(df['TEC'], df['elevation'], map_func = map_fn)
		bin_angles = np.arange(np.min(el), np.max(el) + 2, 2)
		bin_labels = ['{}-{}'.format(bin_angles[i], bin_angles[i+1]) for i in range(len(bin_angles)-1)]
		df['elevation_bins'] = pd.cut(df['elevation'], bins = bin_angles, labels = bin_labels)
		
		delta_VTEC_info = {}
		for bin in bin_labels:
			#print(bin)
			df_bin = df.loc[df['elevation_bins'] == bin, ['SVID', 'TOW', 'TEC', 'VTEC', 'elevation', 'azimuth']]
			#print("Bin size: {}".format(df_bin.size))
			TOW = np.asarray(df_bin['TOW'])

			idx_sort = np.argsort(TOW)
			sorted_TOW = TOW[idx_sort]
			vals, idx_start, count = np.unique(sorted_TOW, return_counts = True, return_index = True)
			res = np.split(idx_sort, idx_start[1:])
			vals = vals[count > 1]
			res = filter(lambda x: x.size > 1, res)
			list_indices = list(res)
			#print(list_indices)

			mean_els = []
			delta_VTECs = []
			delta_azs = []
			gcds = []
			SVIDs = []
			for indices in list_indices:
				comb = list(combinations(indices, 2))
				#print(comb)
				for index in comb:
					a,b = index
					el1 = df_bin['elevation'].iloc[a]
					el2 = df_bin['elevation'].iloc[b]
					mean_el = (el1 + el2)/2
					mean_els.append(mean_el)
					az1 = df_bin['azimuth'].iloc[a]
					az2 = df_bin['azimuth'].iloc[b]
					delta_az = abs(az1 - az2)
					delta_azs.append(delta_az)
					VTEC1 = df_bin['VTEC'].iloc[a]
					VTEC2 = df_bin['VTEC'].iloc[b]
					delta_VTEC = abs(VTEC1 - VTEC2)
					delta_VTECs.append(delta_VTEC)
					gcd = great_circle_distance(el1, el2, az1, az2)
					gcd = np.rad2deg(gcd)
					gcds.append(gcd)
					SVID1 = df_bin['SVID'].iloc[a]
					SVID2 = df_bin['SVID'].iloc[b]
					SVID_pair = '{}-{}'.format(SVID1, SVID2)
					SVIDs.append(SVID_pair)
					#print("Mean elevation: {}".format(mean_el))
					#print("Difference in azimuth: {}".format(delta_az))
					#print("Difference in VTEC: {}".format(delta_VTEC))
					#print("Great circle distance: {}".format(gcd))

			delta_VTECs = np.asarray(delta_VTECs)
			mean_els = np.asarray(mean_els)
			delta_azs = np.asarray(delta_azs)
			gcds = np.asarray(gcds)
			data_dict = {'Mean elevation': mean_els, 'Azimuth difference': delta_azs, 'VTEC difference': delta_VTECs, 'Great circle distance': gcds, 'Satellite ids': SVIDs}
			df_el_bin = pd.DataFrame(data_dict)
			delta_VTEC_info[bin] = df_el_bin
			#delta_VTEC_info['{}_delta_VTECs'.format(bin)] = delta_VTECs
			#delta_VTEC_info['{}_mean_els'.format(bin)] = mean_els

		np.save("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_delta_VTEC_info.npy".format(day, day), delta_VTEC_info)
		#print(delta_VTEC_info)
		print(day + "Saved!")



def VTEC_min_comparison(all_dfs, map_fn):
	"""Displays plot(s) showing great circle distances between pairs of satellites seen in similar elevation bins for a block of day(s)
	
	Args:
		all_dfs (dict): Master dataframe containing TEC information for the given set of days. Obtained by running day.day()
		map_fn (str): String describing choice of mapping function
	"""
	
	main_df_titles = ['elevations', 'mean_elevation', 'VTECs', 'STECs', 'delta_VTEC', 'great_circle_distance', 'day', 'time', 'SVIDs', 'azimuthal_angles'] 
	main_dict = {}
	
	for day in all_dfs:
	
		day_block, year = day.split('_')
		els = []
		VTECs = []
		STECs = []
		delta_VTECs = []
		gcds = []
		days = []
		TOWs = []
		SVIDs = []
		azs = []
		mean_els = []
	
		#year = day[4:]
		df_day = all_dfs[day]
		SVID = np.unique(df_day['SVID'])
		TOW = np.unique(df_day['TOW'])
		
		df = clean(df_day, elevation = True, TEC = True, locktime = True)
		el = np.unique(df['elevation'])
		el = el[~np.isnan(el)]
		df['VTEC'] = VTEC(df['TEC'], df['elevation'], map_func = map_fn)
		bin_angles = np.arange(np.min(el), np.max(el) + 2, 2)
		bin_labels = ['{}-{}'.format(bin_angles[i], bin_angles[i+1]) for i in range(len(bin_angles)-1)]
		df['elevation_bins'] = pd.cut(df['elevation'], bins = bin_angles, labels = bin_labels)
		
		for time in TOW:
			df_time = df.loc[df['TOW'] == time, ['SVID', 'TOW', 'TEC', 'VTEC', 'elevation', 'azimuth', 'elevation_bins']]
			el = np.asarray(df_time['elevation'])
			
			idx_sort = np.argsort(el)
			sorted_el = el[idx_sort]
			vals, idx_start, count = np.unique(sorted_el, return_counts = True, return_index = True)
			res = np.split(idx_sort, idx_start[1:])
			vals = vals[count > 1]
			res = filter(lambda x: x.size > 1, res)
			list_indices = list(res)
			#print(list_indices)
						
			for indices in list_indices:
				comb = list(combinations(indices, 2))
				#print(comb)
				for index in comb:
					a,b = index
					el1 = df_time['elevation'].iloc[a]
					el2 = df_time['elevation'].iloc[b]
					el_array = np.array([el1, el2])
					els.append(el_array)
					mean_el = (el1 + el2)/2
					mean_els.append(mean_el)
					
					az1 = df_time['azimuth'].iloc[a]
					az2 = df_time['azimuth'].iloc[b]
					delta_az = abs(az1 - az2)
					az_array = np.array([az1, az2])
					azs.append(az_array)
					#delta_azs.append(delta_az)
					
					STEC1 = df_time['TEC'].iloc[a]
					STEC2 = df_time['TEC'].iloc[b]
					STEC_array = np.array([STEC1, STEC2])
					STECs.append(STEC_array)
					
					VTEC1 = df_time['VTEC'].iloc[a]
					VTEC2 = df_time['VTEC'].iloc[b]
					VTEC_array = np.array([VTEC1, VTEC2])
					VTECs.append(VTEC_array)
					delta_VTEC = (VTEC1 - VTEC2)
					delta_VTECs.append(delta_VTEC)
					
					gcd = great_circle_distance(el1, el2, az1, az2)
					gcd = np.rad2deg(gcd)
					gcds.append(gcd)
					
					SVID1 = df_time['SVID'].iloc[a]
					SVID2 = df_time['SVID'].iloc[b]
					SVID_array = [SVID1, SVID2]
					SVIDs.append(SVID_array)
					
					TOW_converted = TOW2UT(time)
					TOWs.append(TOW_converted)
					
					days.append(day)
		
	
		main_list = [els, mean_els, VTECs, STECs, delta_VTECs, gcds, days, TOWs, SVIDs, azs]
		for i in range(len(main_list)):
			main_dict[main_df_titles[i]] = main_list[i]
			#print(len(main_list[i]))
		df_main = pd.DataFrame(main_dict)
		SVID = np.unique(df['SVID'])
		#print(df_main[['elevations', 'azimuthal_angles', 'STECs', 'time', 'SVIDs']].head(20))
		#print(SVID)
		
		for no in SVID:
			plt.figure()
			df_main['SVID_presence'] = [True if no in SVID_list else False for SVID_list in df_main['SVIDs']]
			df_SVID = df_main.loc[df_main['SVID_presence'] == True]
			for i in range(len(df_SVID)):
				SVID_list = df_SVID['SVIDs'].iloc[i]
				if no == SVID_list[1]:
					df_SVID['delta_VTEC'].iloc[i] *= -1		
			plt.scatter(df_SVID['mean_elevation'], df_SVID['great_circle_distance'], c = df_SVID['delta_VTEC'], cmap = 'seismic', alpha = 0.4)
			x = plt.clim(-15,15)
			plt.colorbar()
			plt.xlabel("Mean elevation")
			plt.ylabel("Great circle distance")
			plt.title("{}, year {}, SVID: {}".format(day_block, year, no))
			plt.show()
			#plt.savefig("/Data/rpriyadarshan/ismr/gcd_mean_el_plots/{}/GPS_only_no_abs_{}_SVID_{}.png".format(day, day, no))
			#plt.close()
		print("{} done!".format(day))

'''
filestring = "PUNE1???.18_.ismr"
all_dfs = block_creator(glob.glob(filestring), block_size = 100)
for st in all_dfs:
	all_dfs[st] = clean(all_dfs[st], GPS = True)
VTEC_min_comparison(all_dfs, map_fn = "map3")
print("done!")
'''

'''
all_dfs = day(glob.glob("*.ismr"))
for day in all_dfs:
	all_dfs[day] = clean(all_dfs[day], GPS = True)
print("Cleaned")
VTEC_averaged(all_dfs, map_fn = "map3")
#print(UT2TOW('23:59:41'))
'''
'''
for st in all_dfs:
	all_dfs[st] = clean(all_dfs[st], GPS = True)
VTEC_time(all_dfs, map_fn = "map3")
'''
#df = day(glob.glob("PUNE323?.17_.ismr"))
#VTEC_time(df, map_fn = "map3")
