# MSc
Matlab scripts to perform pairwise comparisons of average N transformation rates and flow rates for different treatments.

The script performs pairwise t-tests of average N rates of different treatments, and multiple, seperate t-tests at each time point for flow rates.

The script is configured by the configuration_template.xlsx 

sheet 'Configuration' contains configurations for
"Ntrace Model": Specify which model is used in comparison testing.
"Number of treatments": Number of treatments being compared.
"Output Filename": Name of output file for comparison test results.
"P Value Adjustment": Specify an algorithm for adjusting for multiplicity.
"Significance Level": Test significance level.

sheet "Treatment_Info" contains configurations for
"Treatment": Names of treatments to compare
"Filename": .mat file that contains Ntrace output data for the corresponding treatment

sheet "Pool" contains configurations for pool names

sheet "Parameters" contains configurations for parameter names

sheet "Combined Parameters" contains configurations for combined parameters; 
Linear combinations must be formatted as follows: "T1 + T2 + ... + Tp - Tp+1 - Tp+2 - ... - Tn". 
  There must be a space between + / -. 
  Transformation names must be exactly as they appear in the model
