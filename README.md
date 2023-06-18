# Smart-Nutrition-Tracker-Web-app

A significant portion of people in the UK experience mental health problems. It is estimated that at least 1 in 4 individuals in the UK have mental health problems. A multitude of causes are responsible for mental health problem. Environmental factors which include diet, sleep and nutrition are one set of them.

This project looks at the development of a smart voice-based food and nutrition data logger web app which serves as a tool to save daily nutrition information from people. 

The above objective will be done in two stages. The first stage will be the implementation of the individualized voice-based word classifier. The second stage of the project would consist of porting the machine learning project into a web interface so that it can become more convenient for users to log
their nutrition values


In addition, This file contains instructions on how to run the program and the web interface

1.Python program
    1.1 Firstly install a anaconda environment using the environment.yml file and by entering the command 
    "conda env create -f  environment.yml" into a windows or any other suitable terminal.

    1.2 Open the "python project.ipynb" in a code editor like VS code and select the kernel to be of the name 
    of the environment created in the previous step.

    1.3 Run all the code cells to view the output from all the models and from the simple program snippet at 
    the end which depicts an example nutritional logger.

2. Web interface
    2.1 Open the conda prompt application and  run commands "conda activate <environment_name>" and "python backend.py" 
    make sure the terminal directory is changed to the project folder. 
    2.2 Open up a web browser and type "http://localhost:5000/" into the url box of the web browser and hit the Enter key
    2.3 You should now see a web page with nutritional calculator as its title.
