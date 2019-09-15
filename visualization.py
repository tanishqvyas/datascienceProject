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
#data.plot_piechart("guardian","Skadoosh")
#data.plot_histogram("G1","Marks Distro", "Marks", "Num of Students")
data.plot_bargraph("traveltime","traveltme", "time", "Num of Students")

#data.plot_scatterPlot("Fjob","G1","This is dotted broo","Mjob","G1")