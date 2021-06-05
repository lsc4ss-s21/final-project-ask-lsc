# Large-Scale Computing Final Project
Kaylah Thomas, Sruti Kanthan, and Angelica Bosko

### Research Topic

Tracking key topics and sentiment towards women during a 12-month
period on the subreddit r/TheRedPill. 

### Research Design 

Reddit spaces can house large communities where like-minded people are able to engage 
in discussion about relevant or interesting topics. On certain occasions, these communities can harbor
prejudiced and violent opinions about groups of people. One particular subreddit, r/TheRedPill, is a place 
where many users, particularly men, share an overwhelmingly negative sentiment about women. In our research,
we intend to analyze trends in discussion about women within the subreddit r/TheRedPill, as well as changes
in sentiment over time. This may offer some insight into how dangerous group think may have a negative 
impact on marginalized communities both online and offline.

### Corpus

In this project, we will be using text data gathered from the comment section 
of posts under the subreddit r/TheRedPill in 2016. In order to gather this text data,
we first use a database containing all reddit data from a particular month in 2016
(https://files.pushshift.io/reddit/comments/) and filter only for the particular
subreddit data we choose to examine. For our data collection, we will be looking at
r/TheRedPill, r/Feminism, and r/technews.
Each monthly raw database contains around 8GB of data.

## Using Large-Scale Computing Methods

### Data Collection

For our analysis, we wanted to use information from the comment section of three 
different subreddits in 2016: r/TheRedPill, r/Feminism, and r/technews. We used the 
subreddit r/TheRedPill to track the overall sentiment and topics discussed on the subreddit
month-by-month in 2016. We used the subreddit r/Feminism as a secondary source to understand 
sentiment and topics discussed in a subreddit with opposing ideology from r/TheRedPill. We also 
decided to use r/technews as a good "control" subreddit. We believe that r/technews should contain mostly
neutral sentiment, as opposed to the sentiment of either r/TheRedPill or r/Feminism.

In order to retrieve month-by-month comment data for each subreddit in 2016, we downloaded 12 files
containing all Reddit comments for one particular month. The 12 monthly files 
(located on https://files.pushshift.io/reddit/comments/), were then downloaded to our local 
machines, where we pre-process the data to only contain comment information and time information for 
comments in our target subreddits. Each monthly raw data file for 2016 ranged between 6GB of data and
8GB of data.

In the data_collection.ipynb file (https://github.com/lsc4ss-s21/final-project-ask-lsc/blob/main/data_collection.ipynb),
we include information about how to parse data files from PushShift.io for selected subreddits using PyWren.
Due to the large data size of these files, we were unable to successfully use PyWren and used our local computers in
order to pickle the data necessary. In the data_collection.ipynb file, we also included code in order to run
each of the raw data files locally. After successfully pickling each of the individual monthly files, 
we store all the comments into one corpus, which can be seen at the end of the notebook.

After pickling together all the individual files, the final file (comments_corpus_final.pickle)
was 182.6 MB in size. 

### Data Cleaning

### Word2Vec

### Sentiment Analysis

### Topic Analysis

## Visualizations

## Works Cited

Raw Reddit Data Retrieved from Pushshift.io:

https://files.pushshift.io/reddit/comments/

How to extract subreddit data from raw Reddit data (pushshift.io):

https://github.com/AhmedSoli/Reddit-Politics/blob/master/01_Content_Analysis/PreProcessing/.ipynb_checkpoints/010_ExtractCommentsTextCorpus-checkpoint.ipynb

