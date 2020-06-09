# DataFest@UofT: COVID-19 Virtual Challenge Submission Instructions

Congratulations.  If your reading this then your team is on it's way to submitting to DataFest. Be sure, that **there is only one Github repository per team**.

## What should be submitted to your team's repository?

### Slide Deck or Interactive App

EITHER a slide deck (maximum 3 content slides + title slide in pdf format) **OR** an interactive app / dashboard (do not submit both).  If your team submits an interactive app then it should be deployed somewhere such as https://www.shinyapps.io/ or https://www.heroku.com/ .

### Video or Write-up

EITHER a 5 minute video or screencast **OR** a one-page write up (single spaced) (do not submit both)

### Code

The code that you developed to create slide deck/app and video/write-up. 

### Data

- Link to data source(s) or a file containg the data in your repository. 

- NB:  [Review Github's file and repository size limitations](https://help.github.com/en/github/managing-large-files/what-is-my-disk-quota#file-and-repository-size-limitations) before storing large data sets in your project's repo. For example, if your project requires a file > 100MB then store the file in another place (e.g., Google Drive, Dropbox) or if it already has a url then read in the file directly: 

```
# R using tidyverse
library(tidyverse)
df <- read_csv("https://mydatasource.org/mydata.csv")
```

```
# Python using pandas
import pandas as pd
df = pd.read_csv('https://mydatasource.org/mydata.csv')
```

### Submission Details `team_submission.md`

Create a file called `team_submission.md` with the following information:
  + A slide deck/app (select one) was created, and is available XXX (add links/file names as appropriate).
  + A slide video/write-up (select one) was created, and is available XXX (add links/file names as appropriate).
   
