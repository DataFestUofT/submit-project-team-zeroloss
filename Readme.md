# DataFest@UofT: COVID-19 Virtual Challenge Submission Instructions

Congratulations.  If your reading this then your team is :running: to submit to DataFest. Be sure, that **there is only one Github repository per team**.

:clock12:  **Submissions are due:** June 14, 2020 at 23:59 EDT.

## What should be submitted to your team's repository?

Each :white_check_mark: below indicates something that your team must include.

### Slide Deck or Interactive App

:white_check_mark:  EITHER a slide deck (maximum :three: content slides + title slide in pdf format) **OR** an interactive app / dashboard (do not submit both).  If your team submits an interactive app then it should be deployed somewhere such as (shinyapps.io)[https://www.shinyapps.io/] or (heroku.com)[https://www.heroku.com/] so that you can submit the :link: as a url. 

:x: do not submit both a slide-deck **AND** an app.

### Video or Write-up

:white_check_mark:  EITHER a :five: minute video or screencast **OR** a :one:-page write up (single spaced) 

:x: do not submit both a video **AND** a write-up.

### Code

:white_check_mark:  The code that you developed to create slide deck/app and video/write-up. 

### Data

:white_check_mark:  :link: to data source(s) or a file containg the data in your repository. 

NB:  [Review Github's file and repository size limitations](https://help.github.com/en/github/managing-large-files/what-is-my-disk-quota#file-and-repository-size-limitations) before storing large data sets in your project's repo. For example, if your project requires a file > 100MB then store the file in another place (e.g., Google Drive, Dropbox) or if it already has a url then read in the file directly using it's :link: :

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

:white_check_mark:  Create a file called `team_submission.md` with the following information:
  + Each team member's name, and UofT email (i.e., yourname@mail.utoronto.ca)
  + A slide deck/app (select one) was created, and is available XXX (add links/file names as appropriate).
  + A slide video/write-up (select one) was created, and is available XXX (add links/file names as appropriate).
