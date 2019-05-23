# Welcome to your NumEconCPH repository

We have provided you with the following repository that contains the required structure for this course. **You should not change any folders in the root directory**.

## Dataproject

We model the day-ahead electricity prices as a function of forecasted wind power penetration

We use data from nordpoolgroup.com.
The data file "DA_prices.csv" contains hourly observations of wind power prognosis, electricity consumption prognosis, wind power penetration and Day-Ahead prices of electricity.
Wind power penetration is calculated as: wind power production / electricity consumption * 100%

## Modelproject

We start by solving the basic Solow-model.
- Finding the FOC
- Finding steady state values for k and y
- Plotting the transition diagram
- Using a widget to plot an interactive transition diagram in which the user can change parameter values
  These sliders does not work in GitHub, but by opening the notebook in Anaconda-Navigator as a Jupyter Notebook the sliders      
  work fine.
  
Next, we solve the Solow-model with land using SymPy
- Finding the FOC
- Finding steady state values for z and y

Finally, we use a bisection method to calculate the steady state level of capital.

A short description of the files and folders:

* [README.md](/README.md): gives a short introduction to your project. You should change this file so it gives an introduction to what your repository consists of, and how to run the code to get your output. The present README file is always present on [github](https://www.github.com/numeconcopenhagen/numeconcopenhagen-2018/blob/master/README.md).
* [/binder](/binder/): The folder is used by mybinder.org to setup an interactive version of your repository. For details see this [guide](https://numeconcopenhagen.netlify.com/guides/mybinder/).
* [/dataproject](/dataproject): The structure is as follows: it contains a jupyter notebook where all the results should be presented. Furthermore, there is a python module named the dataproject where you can write and structure all your code.
* [/examproject](/examproject): Same structure as above.
* [/modelproject](/modelproject): Same structure as above.
* [.gitignore](/.gitignore): A textfile specifying files and folder that will not be uploaded to github, and will not be tracked by git.  
