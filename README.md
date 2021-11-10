# PythonBasicDataAnalysis

The open source dataset gives various metrics for shipping in the South China Sea. I created functions to anaylze the region, timeframe, sparse variables, and metrics for specific ship's identifiers.

Requirements:
-Python 3.8
-Python Dependencies Installed:
	-pandas
	-json
	-datetime
	-matplotlib
	-sklearn
	-numpy
	-kneed

Instructions:

1. Execute Command from main.py directory to Run: python {main.py} {path to ocean json directory} {OPTIONAL: Iterations}

Ex. If I am in the directory where main.py is located and the path to the directory where all
the json files are located is "./json/" relative to my path, then I will run the command:
	python main.py ./json/
The default cut-off is at 10000 iterations. If the user wants more, you must specify it in the third argument.
Ex.
	python main.py ./json/ 100000
