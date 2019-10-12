import visualization.data as manipulator

"""
# Creating object for Data class 

The class contains following functions

	~ introduce_nan()
	~ replace_nan()

"""
# Creating classobject to perform the above stated tasks
data = manipulator.Data()

# Introducing NaN Values
data.introduce_nan()

# replacing NaNs
data.replace_nan()

# visualization
""" visualize """
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

data.plot_scatterPlot("reason","G2","Reason of joining vs Marks","Reason for joining","Test 2 Marks")

data.plot_scatterPlot("guardian","G2","Guardian vs Marks for Test 2","Guardian","Test 2 Marks")

data.plot_scatterPlot("age","G2","Age vs Test 2 Marks","Age","Test 2 Marks")

data.plot_scatterPlot("internet","G2","Internet access vs Marks","Internet access?","Test 2 Marks")

data.plot_scatterPlot("goout","G2","Going out vs Marks","Going out","Test 2 Marks")

data.plot_scatterPlot("famrel","G2","Family relationship vs Test 2 Marks","Family relationship quality","Test 2 Marks")

data.plot_scatterPlot("goout","famrel","Going out vs Family Realationship quality","Going Out","relationship quality")



#data.plot_normalProbabilityPlot("G1")



"""
# Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:
1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
2 sex - student's sex (binary: "F" - female or "M" - male)
3 age - student's age (numeric: from 15 to 22)
4 address - student's home address type (binary: "U" - urban or "R" - rural)
5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
12 guardian - student's guardian (nominal: "mother", "father" or "other")
13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
16 schoolsup - extra educational support (binary: yes or no)
17 famsup - family educational support (binary: yes or no)
18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19 activities - extra-curricular activities (binary: yes or no)
20 nursery - attended nursery school (binary: yes or no)
21 higher - wants to take higher education (binary: yes or no)
22 internet - Internet access at home (binary: yes or no)
23 romantic - with a romantic relationship (binary: yes or no)
24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29 health - current health status (numeric: from 1 - very bad to 5 - very good)
30 absences - number of school absences (numeric: from 0 to 93)

# these grades are related with the course subject, Math or Portuguese:
31 G1 - first period grade (numeric: from 0 to 20)
31 G2 - second period grade (numeric: from 0 to 20)
32 G3 - final grade (numeric: from 0 to 20, output target)

Additional note: there are several (382) students that belong to both datasets . 
These students can be identified by searching for identical attributes
that characterize each student, as shown in the annexed R file.
"""