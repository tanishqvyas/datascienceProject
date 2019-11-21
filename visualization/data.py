# This file contains the Data extractor and visualization class

def debug(x="debugging"):
	print(x)


import numpy as np
import pandas as pd
import os
import shutil
import math
from math import sqrt
import matplotlib.pyplot as plt
import scipy.stats as ss
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle

import statistics as stat
from scipy.stats import zscore, norm, binom, poisson


class Data:

	# Variable to state how much % of data is to be taken as testing data
	testSize = 0.1

	# Variable to keep track of accuracy
	current_accuracy = 0;

	# Class variables >>>>
	filename = "student-por.csv"

	# Regression Model Path
	model_directory_path = os.path.join("models")

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

	#Normalized data & directory path
	normalized_directory_path = os.path.join("visualization","student","normalized")
	normalized_data_path = os.path.join(normalized_directory_path,filename)

	standardized_directory_path = os.path.join("visualization","student","standardized")
	standardized_data_path = os.path.join(standardized_directory_path,filename)

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
	fill_columns = ['famsize','internet','guardian','Pstatus','reason','romantic','Fjob','Mjob']
	column_normalize = ['Medu','Fedu','traveltime','studytime','failures','famrel','famsup','freetime','Dalc','Walc','absences','G1','G2','G3']


	# List of columns on which analysis is to be performed
	#analysis_list1 = ['G3','G2','G1','traveltime','studytime','freetime']
	#analysis_list2 = ['G3','G2','G1','Dalc','Walc','goout']



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



	# Function to scale down data to lie between 0 and 1
	def normalize(self, column_list, max_value = 20):
	
		# reading the dataset
		df = pd.DataFrame(pd.read_csv(self.cleaned_data_path))

		# code to do the normalization of all columns in the list
		for column in column_list:

			# Setting max marks
			max_marks = max_value

			# scaling the values to a range of 0 - 1
			scaled_values = []# list to store scaled values

			# looping through the values and finding what % are they of max marks 
			for item in range(len(df[column])):
				scaled_values.append(df[column][item] / max_marks)

			# updating the column with scaled_values,    here df[column] will be saved back in case first argument is 0 
			df[column] = np.where(1,scaled_values, df[column])

			
		# saving normalized dataset
		self.save_file(df,self.normalized_directory_path,self.normalized_data_path)


	# Standardize data by replacing with z scores such that mean = 0 and variance = 1
	def standardize(self, column_list):
		
		# reading the dataset
		df = pd.DataFrame(pd.read_csv(self.cleaned_data_path))

		# code to do the standardization of all columns in the list
		for column in column_list:

			# updating the column with zscores 
			df[column] = np.where(1, zscore(df[column]), df[column])

		#Saving standardized dataset
		self.save_file(df,self.standardized_directory_path,self.standardized_data_path)


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
        
	#def rep_NaN_mode(self,df,col):
	#	df.fillna(df.mode()[col], inplace = True)

	def fill_data(self,df,col):
		
		df[col] = df[col].ffill(axis = 0)

        
    # Function to remove the NaNs
	def replace_nan(self):

		#Creating Data Frame
		df = pd.DataFrame(pd.read_csv(self.nan_data_path))

		#Calling rem_NaN functions
		self.rep_NaN_mean(df,self.column_mean)
		self.fill_data(df,self.fill_columns)
		self.rep_NaN_median(df,self.column_median)


		# Saving the new dataset
		self.save_file(df,self.cleaned_directory_path,self.cleaned_data_path)




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
		

	# TOOOOOOOOOOOOOOOOOOOOOOooo-DOOOOOOOOOOOOOOOOOOOOOOOOOOO
	def plot_normalProbabilityPlot(self,column_list, title, xlabel, ylabel):

		for column in column_list:

			data = self.fetch_col(column)

			# Sorting the data
			data = sorted(data);

			# finding mean and standard deviation
			col_mean = np.mean(data)
			col_std = np.std(data)


			# Finding the P value i.e.  P = (position - 0.5)/len of data
			# modified list essentially contains values randing from 0 - 1
			modified_list = [(data.index(i) + 1 -0.5)/ len(data) for i in data]

			# getting zscore list for the list of P
			zscore_list = zscore(modified_list)

			# Finding theoretical Quantile for each value in zscore_list
			theoretical_quantile_list = [ (col_std*z) + col_mean  for z in zscore_list]

			# Plotting labels and stuff
			plt.title(title)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)

			# plotting scatter plot
			plt.scatter(theoretical_quantile_list, zscore_list)

			plt.show()


		

	def binomial_distribution(self, n, p):

		pmf = []

		for x in range(n+1):

			pmf.append(binom.pmf(x, n, p))

		for i in range(n+1):

			print(i, "\t\t", pmf[i])

		plt.bar(range(n+1), pmf, width=1)
		plt.show()


	def poisson(self, rate, a, b, x1, x2):

		# rate at which events are happening
		# a, b is the min and max range
		# x1 and x2 k beech me hone ki probab kitti hai


		x = np.arange(a, b, 1) # step len = 1

		z = np.arange(a, b, 1/20)

		plt.bar(x, poisson.pmf(x, rate), fill = False)
		plt.plot(z, norm.pdf(z, rate, math.sqrt(rate)))

		section = np.arange(x1, x2, 1.)
		plt.fill_between(section, poisson.pmf(section, rate))
		plt.show()




	def hypothesis(self):

		df = pd.DataFrame(pd.read_csv(self.initial_data_path))

		f_min = 0; f_max = 19 ;m_min = 0 ;m_max = 19

		df_rank = df.groupby('sex')
		df_rank_size = df_rank.size()


		number_of_males = df_rank_size[0]
		number_of_females = df_rank_size[1]

		print("no: of males:",number_of_males,"no: of females:",number_of_females)
		print(" ")

		group_parameters = df_rank['G3'].agg(['mean', 'median', 
		                                  'std','var', 'min', 'max']).reset_index()
		print(group_parameters)
		print(" ")


		df_rank_means = df_rank['G3'].agg(['mean'])
		df_rank_std = df_rank['G3'].agg(['std'])
		df_rank_var = df_rank['G3'].agg(['var'])
		df_rank_min = df_rank['G3'].agg(['min'])
		df_rank_max = df_rank['G3'].agg(['max'])




		f_mean = df_rank_means['mean'][0]
		m_mean = df_rank_means['mean'][1]

		f_std = df_rank_std['std'][0]
		m_std = df_rank_std['std'][1]

		f_variance = df_rank_var['var'][0]
		m_variance = df_rank_var['var'][1]

		print("Chosen hypothesis")
		print("H0 : f_mean - m_mean >= 0")
		print("H1 : f_mean - m_mean < 0 - so left tailed")
		print(" ")

		#ppf(x) gives the z score of the area x. cdf is the opposite, so it gives area for z-score
		alpha = 0.05
		print("Our chosen significance level is 5% and our confidence level is 95%")

		diff_mean = f_mean - m_mean
		diff_variance = (f_variance)/number_of_females + m_variance/number_of_males
		diff_std = sqrt(diff_variance)

		#What should be taken as n? Total number of students?
		z1 = (0-diff_mean)/diff_std

		area1 = norm.cdf(z1)
		print(area1)
		print(" ")
		print("Since P is found to be <<0.05, H0 can be rejected and H1 can be accepted")










	def buildTrainSavePredictModel(self, analysis_list, predict, model_name, wannaTrain=0):

		# Reading the data from csv
		df = pd.DataFrame(pd.read_csv(self.initial_data_path))

		# Trimming the columns which aren't required
		# analysis_list is the list of columns taken for analysis
		df = df[analysis_list]

		# Separating our target field and features
		attributes = np.array(df.drop([predict], 1))
		target = np.array(df[predict])

		# Splitting the dataset into testing and training data
		train_attributes, test_attributes, train_target, test_target = sklearn.model_selection.train_test_split(attributes,target,test_size=self.testSize)


		if wannaTrain:		

			for i in range(1000000):
				
				# Creating the Linear Regression model
				predictorModel = linear_model.LinearRegression()

				# Getting the least square fit line
				predictorModel.fit(train_attributes, train_target) # obtaining least square line as model

				# Predicting the target values for the testing data once we have obtained our least square line
				# The function returns the accuracy score based on the generated values
				accuracy = predictorModel.score(test_attributes, test_target)

				# printing the accuracy, y-intercept and (slope for all the fields since its not a 2D plot)
				#print(accuracy)
				#print("Slope : ",predictorModel.coef_)
				#print("Y-Intercept :",predictorModel.intercept_)

				
				# SAVING OUR MODEL
				# Although it doesn't really makes sense to save it because it doesn't really take more than a second to train
				# But we are gonna save the best one we find for future use
				
				if accuracy > self.current_accuracy:
					self.current_accuracy = accuracy
					# wb stating writing bytes
					pickle_file_obj = open(model_name,"wb")
					pickle.dump(predictorModel, pickle_file_obj)
					pickle_file_obj.close()


		# Loading our model from our pickle file
		load_model = open("gradePredictorModel.pickle","rb")
		predictorModel = pickle.load(load_model)

		# Printing predicted value, Input data, Actual Value>>>>>>>>>>>>>>>>

		#Getting the predictions value
		predicted_values = predictorModel.predict(test_attributes)

		for i in range(len(predicted_values)):

			#print("Predicted : ", math.round(predicted_values[i])," Input data :", test_attributes[i], "   Actual Value : ",test_target[i])
			print("Predicted : ",int(round(predicted_values[i])), "   Actual Value : ",test_target[i])

		print(self.current_accuracy) #activate this line when wannaTrain is 1

		# Getting the corelation graph
		plt.style.use("ggplot")
		plt.scatter(test_target, predicted_values)
		plt.xlabel("G3 Actual")
		plt.ylabel("G3 Predicted")
		plt.show()


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
	- Fix y axis scale representaton for horizontal and vice versa
	- Add legends

6. Defne Scatter Plot function

7. Defne path to the folder to save plots
8. Refactor code to overwrte files instead of deleting folders
9. Make box-plots
10. Normalize the data
11. Styling and outliers in boxplot got fucked cause of order fix that
"""

	




