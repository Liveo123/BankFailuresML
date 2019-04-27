INSTRUCTIONS TO GET DATA:


list of failed banks:
https://www.fdic.gov/bank/individual/failed/banklist.csv

This list includes banks which have failed since October 1, 2000. 

The raw data is in excellent shape, listing the names of the bank, location, acquiring institution, and closing date–with no holes in the data (it is well-maintained probably because the list is so important, if you remember 2008).  



bank data:
https://www5.fdic.gov/sdi/main.asp?formname=customddownload

Annoyingly, you have to custom download the data for each quarter.  Since we used data from 200-2018, this is a lot of downloads and will take half an hour.  

The link lists many variables, make sure to click the "Select All" button, then click "Next".  These are 56 features based on Assets & Liabilities, which are fairly good at describing a bank and predicting future failure.  Once you go to the next page, select the quarter you would like (the option is displayed at the top next to 'Information as of'), then ignore all of the other filtering options and hit "Find" at the bottom, where it will forward you to the final page from which you can slowly download the .csv file.  



Congratulations, now you have ~80 separate files.  You could read them systematically using Python, then combine it into a numpy matrix and go from there, or you can combine it into one csv using R, which is what I did using the following code:


data <- read.csv("/Users/Daniel/Downloads/SDI_Download_Data.csv", header=T, skip=6)
for(i in 2:80)
{
  newfile <- read.csv(paste("/Users/Daniel/Downloads/SDI_Download_Data-", i, ".csv", sep=""), header=T, skip=6)
  data <- rbind(data, newfile)
}

data <- data[order(as.Date(data[,10], format="%m/%d/%Y"), data[,1]),]
write.csv(data, file = "/Users/Daniel/Desktop/Schoolwork/GT/CSE 6242/Project/combined_data_v2.csv", row.names = FALSE)


80 different files, hundreds of thousands of rows and 56 columns–this will take some time to run.  


Anyways, that's how you get the data. Good luck with it!