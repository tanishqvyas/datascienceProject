import visualization.data as manipulator

"""
# Creating object for Data class 

The class contains following functions

	~ introduce_nan()
	~ replace_nan()

"""
# Creating classobject to perform the above stated tasks
data = manipulator.Data()

# Introducing and replacing NaNs
data.introduce_nan()
data.replace_nan()

#data.plot_boxPlot(["G1","G2"],"title ye rha bhai","xlabel bhi lele", "ha y bhi le hi le", True)
#data.plot_piechart("guardian","Skadoosh")
#data.plot_histogram("G1","Marks Distro", "Marks", "Num of Students")
#data.plot_bargraph("traveltime","traveltme", "time", "Num of Students")
#data.plot_scatterPlot("G1","G2","This is dotted broo","G1","G2")


# Graphs
data.plot_piechart("sex","Gender Distribution")
#data.plot_scatterPlot("sex","G3","Marks vs Gender","Sex","G3")
data.plot_histogram("G3","Marks Distribution", "Marks For G3", "Num of Students")
data.plot_boxPlot(["G1","G2", "G3"],"Box plot for Marks for different tests","Exams G1, G2 and G3", "Marks", True)
data.plot_scatterPlot("traveltime","G3","Travel time in Hrs vs Marks","Travel Time in Hrs","Marks in G3")






# Normalizing columns
data.normalize(["G1", "G2", "G3"], 20)

# Standardizing columns
data.standardize(["G1", "G2"])

# Plottting Normal Probability Plot
#data.plot_normalProbabilityPlot(["G1", "G2"],"Normal Probability Plot for Test Scores","Theoretical Quantile","Z-score")

# Building simple model and predicting the Final grades
#data.buildTrainSavePredictModel(['G2'],"G3", "TravelStudyFreeModel", 0)


# Get correlation coefficient for 2 columns
data.get_correlation_coefficient()

# Get results for hypothesis testing
data.hypothesis()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LEAVE THE SECTION BELOW>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#####data.plot_normalProbabilityPlot("G2")
#####data.binomial_distribution(50,0.9)
#####data.poisson(27, 5, 100, 20, 60)

