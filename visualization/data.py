# This file contains the Data extractor and visualization class

import numpy as np
import pandas as pd
import os
import shutil
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

class Data:

	# Class variables >>>>
	filename = "student-por.csv"

	# Initial data path
	initial_directory_path = os.path.join("visualization","student","initial")
	initial_data_path = os.path.join(initial_directory_path,filename)

	# Boolean Filtered data path
	boolean_filter_directory_path = os.path.join("visualization","student","filtered")
	boolean_filter_data_path = os.path.join(boolean_filter_directory_path,filename)

	# NaN value & directory path
	nan_directory_path = os.path.join("visualization","student","nan")
	nan_data_path = os.path.join(nan_directory_path,filename)

	# Cleaned data & directory path
	cleaned_directory_path = os.path.join("visualization","student","cleaned")
	cleaned_data_path = os.path.join(cleaned_directory_path,filename)

	# Path of folder to save plots
	plot_directory_path = os.path.join("visualization","student","Plots")
	
	# Percentage NaN
	percentage_nan = [0.03, 0.015, 0.010] # The variable to keep track of what percent of data is modified to NaN

	# Columns list 
	column_list1 = ['Pstatus','Medu','Fedu','Mjob','Fjob','reason','failures','romantic','famrel','Dalc','Walc']
	column_list2 = ['famsize','traveltime','studytime','goout','health','famsup']
	column_list3 = ['internet','freetime','guardian']

	#Dividing the columns based on what operation needs to be applied
	column_mean = ['Medu','Fedu','famrel','freetime','goout','health','traveltime']
	column_median = ['studytime','famsup','failures','Dalc','Walc']
	column_mode = ['famsize','internet','guardian','Pstatus','reason','romantic','Fjob','Mjob']





	# Init method
	def __init__(self):
		self.boolean_filter()
		#pass

	# Function to save modified dataset as a new version of DataFrame
	def save_file(self, df, directory_path, data_path):

		if os.path.exists(directory_path):
			shutil.rmtree(directory_path)

		os.mkdir(directory_path) # making new directory to save newversion
		df.to_csv(data_path,index = False)
		print("done")

	# Function to replace yes with 1 and no with 0 respectively
	def boolean_filter(self):

		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))

		# Replacing values
		df.replace(to_replace ="yes", value = 1, inplace = True)
		df.replace(to_replace ="no", value = 0, inplace = True)
		df.replace(to_replace ="YES", value = 1, inplace = True)
		df.replace(to_replace ="NO", value = 0, inplace = True)
		df.replace(to_replace ="Yes", value = 1, inplace = True)
		df.replace(to_replace ="No", value = 0, inplace = True)
		# saving file
		self.save_file(df, self.boolean_filter_directory_path, self.boolean_filter_data_path)


# Introducing NaNs

	# Function to add NaN
	def add_nan(self,df,column_list,percentage):

		# Adding NaN
		for col in column_list:
		    ori_rat = df[col].isna().mean()

		    if ori_rat >= percentage: continue

		    add_miss_rat = (percentage - ori_rat) / (1 - ori_rat)
		    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
		    df.loc[vals_to_nan, col] = np.NaN



	# Function to introduce NaN values
	def introduce_nan(self):
	
		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))
		
		# Calling add_nan function to modify 3 lists of columns
		self.add_nan(df,self.column_list1,self.percentage_nan[0])
		self.add_nan(df,self.column_list2,self.percentage_nan[1])
		self.add_nan(df,self.column_list3,self.percentage_nan[2])

		# Filling NaN
		df.fillna("NaN", inplace = True)

		# Saving the new dataset
		self.save_file(df, self.nan_directory_path, self.nan_data_path)
        
#Replacing NaNs
    
    #Functions to replace NaNs
	def rep_NaN_mean(self,df,col):
		df.fillna(df.mean()[col].round(0), inplace = True)
        
	def rep_NaN_median(self,df,col):
		df.fillna(df.median()[col].round(0), inplace = True)
        
	def rep_NaN_mode(self,df,col):
		df.fillna(df.mode()[col], inplace = True)
        
    # Function to remove the NaNs
	def replace_nan(self):
        
		# Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.nan_data_path))
        
        #Calling rem_NaN functions 
		self.rep_NaN_mean(df,self.column_mean)
		self.rep_NaN_median(df,self.column_median)
		self.rep_NaN_mode(df,self.column_mode)
        
    	# Saving the new dataset
		self.save_file(df, self.cleaned_directory_path, self.cleaned_data_path)

	# Function to return Q1 Q2 or Q3
	def get_quartile_value(self,data_list, n, quartile_ratio):

		if (n+1)*quartile_ratio == int((n+1)*quartile_ratio):
			return data_list[(n+1)*quartile_ratio]

		else:
			return ( data_list[math.ceil((n+1)*quartile_ratio)] + data_list[math.floor((n+1)*quartile_ratio)] ) / 2

	# Function to find median
	def get_IQR(self, data_list):

		# finding number of elements
		num_of_elements = len(data_list)

		# fetching Q1, Q2 & Q3 values
		Q1 = self.get_quartile_value(data_list,num_of_elements,0.25)
		Q3 = self.get_quartile_value(data_list,num_of_elements,0.75)

		return Q3 - Q1


	# Function to find bin_size using Freedman-Diaconis formula
	def get_bin_size(self,data_list, n):

		bin_size = (2 * self.get_IQR(data_list)) / (math.pow(n, 1/3))
		return math.ceil(bin_size)

	# Function to calculate num of classes
	def get_count_classes(self, data_list):

		num_of_classes = (max(data_list)-min(data_list)) / self.get_bin_size(data_list, len(data_list))
		return math.ceil(num_of_classes)


	# Pre-requisites for plotters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.

	# Functon to save the visulizaton graphs
	def save_plot(self):
		pass

	# Function to fetch column as a list
	def fetch_col(self, column_title):
		
		#change >>>> filepath
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))

		# Convertingthe df column to list
		store_list = list(df[column_title].tolist())
		store_list.sort() 
		return store_list

	# Functon to extract labels and their counts from a column
	def structure_data(self,data_list):
		
		# Extract unique fields
		label_set = list(set(data_list))

		# Get count for each field
		label_count = [data_list.count(i) for i in label_set]

		return label_set, label_count

	# Graph plotters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	# Function to plot pie-chart
	def plot_piechart(self, column_title, title):

		# Pre processing
		data_list = self.fetch_col(column_title)

		# Extracting labels and respective values
		label_set, label_count = self.structure_data(data_list)

		# Plottng
		plt.pie(label_count, labels=label_set, shadow=False, startangle=90, autopct='%.1f%%')
		plt.title(title)
		#plt.tight_layout()
		plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
		plt.show()

	# Functon to plot bar graph
	def plot_bargraph(self, column_title, title, xlabel, ylabel, isVertical = False):

		# Pre processing
		data_list = self.fetch_col(column_title)

		# Extracting labels and respective values
		label_set, label_count = self.structure_data(data_list)
		
		# To use template styling
		plt.style.use('seaborn')

		if not isVertical:
			plt.barh(label_set, label_count)
			plt.ylabel(xlabel)
			plt.xlabel(ylabel)
		else:
			plt.bar(label_set, label_count)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
		
		plt.title(title)
		# some padding it seems
		plt.tight_layout()
		plt.show()

	# Function to plot histogram
	def plot_histogram(self, column_title, title, xlabel, ylabel, plotCurve=False):
		
		# plotCurve variable is 1 if we wanna plot curve above histogram

		# Pre processing
		data_list = self.fetch_col(column_title)
		
		# Fnding num of bins
		num_of_bins = self.get_count_classes(data_list)

		# To use template styling
		plt.style.use('seaborn-whitegrid')

		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

		if plotCurve:
			pass #todo


		plt.hist(data_list, bins = num_of_bins, normed=True)
		plt.show()

	
	# Functon to plot Scatter Plot
	def plot_scatterPlot(self, column_title1, column_title2, title, xlabel, ylabel):

		# Pre processing
		data_list1 = self.fetch_col(column_title1)
		data_list2 = self.fetch_col(column_title2)


		# Plotting
		plt.scatter(data_list1, data_list2, color='r')
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
			

	def plot_boxPlot(self, fieldList, title, xlabel, ylabel, isVertical=True):

		#Test
		print(plt.style.available)

		# Empty list to hold column data
		dataList = []

		# Loop to fetch column data and append in dataList
		for i in range(len(fieldList)):
			dataList.append(self.fetch_col(fieldList[i]))
		
		"""
		showfliers : boolean, true to show outliers
		flierprops : styling for outliers markers
		notch : boolean, true to show Notch at Q2
		vert : boolean, true for vertical box plot
		"""

		# Plot related stuff
		plt.style.use("seaborn-dark")
		fig1, myPlot = plt.subplots()
		#myPlot.set_title(title)
		plt.title(title)

		# Label management
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

		if not isVertical:
			plt.xlabel(ylabel)
			plt.ylabel(xlabel)
		
		# Plotting and showing
		myPlot.boxplot(dataList, notch=False, showfliers=True, vert=isVertical, labels=fieldList)
		plt.show()
		# Save the figure
		#fig.savefig('fig1.png', bbox_inches='tight')



"""
Todo

0. Change file path from intial to cleaned once cleaning is done
2. Define save_plot, to be done at the end
3. Pie charts
4. Histograms
	- Add line plot feature on top of hist based on plotCurve's value
	- Add a feature to plot it separately too if needed
	- Add legends

5. Bar Graphs
	- Fix y axis scale representaton for horzontal and vice versa
	- Add legends

6. Defne Scatter Plot function

7. Defne path to the folder to save plots
8. Refactor code to overwrte files instead of deleting folders
9. Make box-plots
10. Normalize the data
"""





