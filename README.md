# MGS-pipeline-distribution
model for Zylka mgs


## Getting Started and Installations

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Download contents of https://github.com/BenjaminCorvera/MGS-pipeline-distribution/ via zip file. Unzip to desired location.

Visit-

https://www.anaconda.com/download/

to download the most recent version of anaconda (python 3.6 version), which includes python and number of excellent packages for python data science. Follow steps for easy installation. When prompted, make sure to set pip as a PATH enviroment variable. Once downloaded, open the anaconda navigator. Select the "environments" tab. Click on "create" button, and name your new environment "ZylkaEnvironment." 

Open the "ZylkaEnvironment" terminal. 

Navigate to your local, downloaded directory  "MGS-pipeline-distribution" and install dependencies to new enviroment. This can be con via terminal commands

**cd PathToMGS-pipeline-distribution (will vary based on location of directory)

**pip install -r req.txt

This will intall all dependencies necessary for deployment. There may be more packages downloaded than necessary, but this isn't the end of the world.



## Running the script "label.py"

create a few folders in the main directory. The names of these files must be identical to what is in quotes below. 

"images" - Place any images you wish to test agains the model in this folder. Images of white mice will work the best.

"labeled" - This can remain empty

"review" - This too can remain empty

Now run main script via command line using command:

**python label.py**

Images which fall below the .75 confidence threshold will be placed in the "review" folder. Images which meet the threshold will be placed in the "labeled" folder. Check the log file to see which images were labeled what and what their confidence score was.

## Authors

* **Mark Molinaro**
* **Ben Corvera**


## Acknowledgments

* InceptionV3 team
