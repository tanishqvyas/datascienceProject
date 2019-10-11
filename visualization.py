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

# visualization
data.plot_boxPlot(["G1","G2"],"Test 1 marks vs Test 2 marks","Test 1 marks", "Test 2 marks", True)

data.plot_piechart("guardian","Guardians")

data.plot_histogram("G2","Marks Distribution", "Test 2 marks", "Number of Students")

data.plot_histogram("G1","Marks Distribution", "Test 1 marks", "Number of Students")

data.plot_bargraph("traveltime","traveltime", "time", "Number of Students")

data.plot_bargraph("reason","reason for joining school","reasons","number of students")

data.plot_scatterPlot("G1","G2","Sactter Plot for marks for both tests","Test 1 marks","Test 2 marks")

data.plot_scatterPlot("Mjob", "G2", "Scatter Plot for marks vs Mother's occupation","Occupations", "Test 2 marks")

data.plot_scatterPlot("Fjob", "G2", "Scatter Plot for marks vs Father's occupation","Occupations", "Test 2 marks")

data.plot_scatterPlot("romantic","G2","Scatter Plot for marks vs Realationship status","Is in a Realationship?","Test 2 Marks")

data.plot_scatterPlot("traveltime", "G2", "Scatter Plot for marks vs traveltime","traveltime in hours", "Test 2 marks")

data.plot_scatterPlot("Walc","G2","Weekly alcohol consumption vs Marks in Test 2","Weekly alcohol consumption","Test 2 marks")

data.plot_scatterPlot("Dalc","G2","Daily alcohol consumption vs Marks in Test 2","Daily alcohol consumption","Test 2 marks")

data.plot_scatterPlot("studytime","G2","Study time vs marks","Studytime in hrs","Test 2 marks")
#data.plot_normalProbabilityPlot("G1")