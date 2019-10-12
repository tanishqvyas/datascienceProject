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

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LEAVE THE SECTION BELOW>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#data.buildTrainSavePredictModel(['G3','G2','traveltime','studytime','freetime'],"G3",1)
#####data.plot_normalProbabilityPlot("G2")
#####data.binomial_distribution(50,0.9)
#####data.poisson(27, 5, 100, 20, 60)

# Normalizing columns
data.normalize(["G1","G2"], 20)

# Standardizing columns
data.standardize(["G1","G2"])
