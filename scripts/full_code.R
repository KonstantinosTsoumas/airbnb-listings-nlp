library(dplyr)
library(tm)
library(syuzhet)
library(textcat)
library(sf)
library(terra)
library(mapview)
library(stringr)
library(qdap)
library(qdapTools)
library(cld2)
library(stargazer)
library(texreg)
library(tm)
library(geosphere)
library(tidyr)
library(maps)
library(lubridate)
library(car)
library(SnowballC)
library(LDAvis)   
library(RSentiment)
library(ggplot2) 
library(rworldmap)
library(sp)
library(readr)
library(RColorBrewer)
library(lubridate)
library(dplyr)
library(topicmodels)
library(slam)
library(syuzhet)
library("IRdisplay")
library("DataExplorer")
library(stringr)
library("tidyverse")
library("tidytext")

#Import data
data <- read.csv("/Users/Tsoumas/Dessubset_regtop/Data Science and Marsubset_regeting Analytics/Thesis/Analysis/Data.csv", na.strings=c(NA,"NA",NA))

#subset_regeep necessary columns
columns_to_remove <- c("id", "host_id", "host_picture_url", "picture_url", "listing_url", "host_acceptance_rate", "host_neighbourhood", "scrape_id", "host_url", 
                       "host_location", "host_response_time", "host_thumbnail_url", "last_scraped", "neighbourhood", "host_verifications", "calendar_last_scraped", 
                       "number_of_reviews_l30d", "first_review", "last_review", "host_name", "property_type", "license", "neighbourhood_group_cleansed", "calendar_updated", 
                       "accommodates", "bathrooms", "bathrooms_text", "availability_30", "availability_60", "availability_90", "bathrooms_text", "host_total_listings_count", 
                       "maximum_minimum_nights", "bathrooms_text", "minimum_maximum_nights", "has_availability", "review_scores_accuracy")
data <- data %>%
  select(-columns_to_remove)

#Delete rows which contain mostly/completely missing values
data <- data[complete.cases(data[c("bedrooms", "beds", "minimum_minimum_nights","latitude","longitude", "review_scores_rating",
                                    "review_scores_cleanliness", "review_scores_checsubset_regin", "review_scores_communication", "review_scores_location",
                                    "review_scores_value", "reviews_per_month", "host_response_rate")]), ]

#Transform variables 
data <- data %>%
  mutate(number_of_reviews = as.numeric(number_of_reviews),
         reviews_per_month = as.factor(reviews_per_month),
         review_scores_rating = as.numeric(review_scores_rating),
         review_scores_communication = as.numeric(review_scores_communication),
         minimum_nights = as.numeric(minimum_nights),
         availability_365 = as.numeric(availability_365),
         room_type = as.factor(room_type),
         amenities <- as.factor(amenities))

#Transform days
endDate <- as.Date("2021-12-06","%Y-%m-%d") #define the end date to count number of days passed
data$host_since <- as.Date(data$host_since)
data$host_since <- difftime(endDate, data$host_since, units = "days") #count days since host created an account
data$description <- paste(data$name, data$description) #concatenating description and names

#Replace NA value with zeros
data <- data %>%
  mutate(number_of_reviews = coalesce(number_of_reviews, 0))

#Remove special characters, punctuatio
data <- data %>%
  mutate(price = parse_number(price)) %>%
  mutate(price = price / 100)
data$description <- removePunctuation(data$description) #punctuation
data$host_about <- removePunctuation(data$host_about) #punctuation

#Calculating Occupancy Rate
#subset_regeep only listings which are 'widely' available
data <- data %>%
  mutate(number_of_reviews_ltm = number_of_reviews_ltm / 0.5,
         number_of_reviews = number_of_reviews / 0.5) %>% 
  filter(availability_365 >= 30)

#Determine the dependent variable based on the 'San Fransisco' model's assumptions
data$occupancy_rate <- ifelse(data$minimum_nights >= 3.9, 
                               (data$minimum_nights * data$number_of_reviews_ltm)/365,
                               (3.9 * data$number_of_reviews_ltm))/365
data$occupancy_rate <- ifelse(data$occupancy_rate > 1, 1, data$occupancy_rate) 

#subset_regeep only english
data <- data %>%
  filter(detect_language(host_about) == "en",
         detect_language(description) == "en",
         detect_language(comments) == "en")

#Delete the word 'br' as it only adds noise
data$description <- gsub("br", " ", data$description, ignore.case = TRUE)
data$comments <- gsub("br", " ", data$comments, ignore.case = TRUE)

#Exploratory Data Analysis
ggplot(data, aes(x = occupancy_rate)) +
  geom_histogram(bins = 30)

#Doing the same (as below) for comments by changing description to comments, will not paste the code here to increase readability
#Bigram
raw2 <- data %>%
raw2<-raw2 %>% mutate(text = str_replace_all(host_about, "[:digit:]", " "))
raw2 <- raw2 %>%
  unnest_tosubset_regens(bigram, text, tosubset_regen = "ngrams", n = 2)
bigrams_separated <- raw2 %>%
  separate(bigram, c("word1", "word2"), sep = " ")
bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)
# new bigram counts:
bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)
bigram_counts %>% mutate(bigram = paste(word1,word2)) %>% 
  slice_head(n=30) %>%mutate(word = reorder(bigram, n)) %>% filter(bigram!="NA NA") %>% ggplot(aes(x=n,y=word))+geom_col()+theme_light()+
  labs(y = NULL)+ggtitle("Top 30 most common BIgrams") + theme(text=element_text(size=18))

#Trigram
raw3 <- data
raw3<-raw3 %>% mutate(text = str_replace_all(host_about, "[:digit:]", " "))
raw3<-raw3 %>% mutate(words=str_count(text,"\\w+")) %>% filter(words>2)
trigrams<-raw3 %>%
  unnest_tosubset_regens(trigram, text, tosubset_regen = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  count(word1, word2, word3, sort = TRUE)
trigrams %>% mutate(trigram = paste(word1,word2,word3)) %>% 
  slice_head(n=30) %>%mutate(word = reorder(trigram, n)) %>% filter(trigram!="NA NA NA") %>% ggplot(aes(x=n,y=word))+geom_col()+theme_light()+
  labs(y = NULL)+ggtitle("Top 30 most common trigrams") + theme(text=element_text(size=18))

#Convert variables to binary
data <- data %>%
  mutate(smosubset_rege = case_when(str_detect(amenities, "Smosubset_rege alarm") ~ 1, TRUE ~ 0),
         first_aid = case_when(str_detect(amenities, "First aid subset_regit") ~ 1, TRUE ~ 0),
         carbon = case_when(str_detect(amenities, "Carbon monoxide alarm") ~ 1, TRUE ~ 0),
         Fire = case_when(str_detect(amenities, "Fire extinguisher") ~ 1, TRUE ~ 0),
         BedroomLocsubset_reg = case_when(str_detect(amenities, "Locsubset_reg on bedroom door") ~ 1, TRUE ~ 0))

#Construct Aristoteles persuasion theories (Logos, Ethos, Pathos)

#Construct safety features for 'Logos' persuasion theory
data$safety_features <- data$smosubset_rege + data$first_aid + data$carbon + data$Fire + data$BedroomLocsubset_reg

#Delete safety features from amenities, not to be double counted
amenities_to_remove <- c("Smosubset_rege alarm", "First aid subset_regit", "Carbon monoxide alarm", "Fire extinguisher", "Locsubset_reg on bedroom door")
data$amenities <- str_replace_all(data$amenities, amenities_to_remove, "")

#Construct reviews from the six categories, on average, per listing
data$reviews <- (data$review_scores_rating + data$review_scores_checsubset_regin + data$review_scores_cleanliness + 
                   data$review_scores_communication + data$review_scores_location + data$review_scores_value)/6

#Delete unnecessary columns from dataset
data <- data %>%
  select(-amenities, -name)

#Construct Ethos persuasion theory
data$host_response_rate <- as.numeric(sub("%", "", data$host_response_rate)) 
data$host_response_rate <- ifelse(is.na(data$host_response_rate), 0, data$host_response_rate) #masubset_rege it binary
data$host_is_superhost <- ifelse(data$host_is_superhost == 't', 1, 0) #masubset_rege it binary
data$host_identity_verified <- ifelse(data$host_identity_verified == 't', 1, 0) #masubset_rege it binary
data$host_response_rate <- (data$host_response_rate)/100


#Construct Pathos persuasion theory using Sentiment Analysis
sent_subset <- get_nrc_sentiment(data$description)
sent_subset <- as.data.frame(sent_subset)
sent_subset <- subset(sent_subset, select = -(9:10))

#sent_subset$overall <- sent_subset$positive - sent_subset$negative

sent_subset <- get_nrc_sentiment(data$host_about)
sent_subset <- as.data.frame(sent_subset)
sent_subset <- subset(sent_subset, select = -(9:10))

data$description_pol <- ifelse(rowSums(sent_subset) != 0, "Emotional", "Not Emotional")
data$description_pol <- as.factor(data$description_pol)
data$host_about_pol <- ifelse(rowSums(sent_subset) != 0, "Emotional", "Not Emotional")
data$host_about_pol <- as.factor(data$host_about_pol)

#WORD BASED SENTIMENT ANALYSIS
#Pre processing text
data$description <- tolower(data$description)                                         # Transform all text to lower case
data$description <- gsub("[^0-9a-z,.:;' ]", "", data$description, ignore.case = TRUE) # subset_regeep a-z, 0-9, ",", ".", ":", ";" and "'", remove any other characters
data$description <- gsub(":", ",", data$description, ignore.case = TRUE)              # Replace : by ,
data$description <- gsub(";", ",", data$description, ignore.case = TRUE)              # Replace ; by ,
data$description <- gsub("\\.", ",", data$description, ignore.case = TRUE)            # Replace . by ,

data$host_about <- tolower(data$host_about)                                           # Transform all text to lower case
data$host_about <- gsub("[^0-9a-z,.:;' ]", "", data$host_about, ignore.case = TRUE)   # subset_regeep a-z, 0-9, ",", ".", ":", ";" and "'", remove any other characters
data$host_about <- gsub(":", ",", data$host_about, ignore.case = TRUE)                # Replace : by ,
data$host_about <- gsub(";", ",", data$host_about, ignore.case = TRUE)                # Replace ; by ,
data$host_about <- gsub("\\.", ",", data$host_about, ignore.case = TRUE)              # Replace . by ,

data$comments <- tolower(data$comments)                                           # Transform all text to lower case
data$comments <- gsub("[^0-9a-z,.:;' ]", "", data$host_about, ignore.case = TRUE)   # subset_regeep a-z, 0-9, ",", ".", ":", ";" and "'", remove any other characters
data$comments <- gsub(":", ",", data$host_about, ignore.case = TRUE)                # Replace : by ,
data$comments <- gsub(";", ",", data$host_about, ignore.case = TRUE)                # Replace ; by ,
data$comments <- gsub("\\.", ",", data$host_about, ignore.case = TRUE)

#unnesting description column and putting them all togheter in all_words
all_words <- data[,] %>%
  unnest_tosubset_regens("comments", output = "word") %>%
  anti_join(stop_words, by = "word") %>%
  count(word, sort = TRUE)

#sentiment based on nrc lexicon
all_words$sentiment <- sentiment(all_words$word, polarity_dt = lexicon::hash_sentiment_nrc)$sentiment

#Contribution to total sentiment plot
all_words %>%
  filter(n > 50) %>%
  filter(sentiment != 0) %>%
  mutate(n = ifelse(sentiment < 0, -n, n)) %>%
  mutate(word = reorder(word, n)) %>%
  mutate(Sentiment = ifelse(sentiment >0 , "Postive","Negative")) %>%
  ggplot(aes(word, n, fill = Sentiment)) +
  geom_col() +
  coord_flip() +
  theme(text = element_text(size = 16))+
  labs(y = "Contribution to \"total\" sentiment (NRC dictionary)", x = "Word (min freq = 250)")

#SENTENCE BASED SENTIMENT ANALYSIS
#Description
sentences <- get_sentences(data[,"comments"])
sentence_scores <- sentiment(sentences, polarity_dt = lexicon::hash_sentiment_nrc)

all_sentences <- as.data.frame( unlist(sentences[]) )
colnames(all_sentences) = "sentence"
all_sentences$sentiment <- sentence_scores$sentiment

all_sentences$sentence_id <- c(1:dim(all_sentences)[1])
all_pos_sentences <- all_sentences %>% filter(sentiment>0)
all_neg_sentences <- all_sentences %>% filter(sentiment<0)

#words in positive sentences deleting stop words
all_pos_sentences_words<- all_pos_sentences  %>%
  unnest_tosubset_regens(word,sentence) %>%
  anti_join(stop_words, by = "word")

all_neg_sentences_words <- all_neg_sentences  %>%
  unnest_tosubset_regens(word,sentence) %>%
  anti_join(get_stopwords(language = "en"), by = "word")

##DESCRIPTION
#Deleting 'br' and numbers for description
all_pos_sentences_words$word <- gsub('[[:digit:]]+', NA, all_pos_sentences_words$word) #transforming to NA
all_pos_sentences_words$word <- gsub("br",NA,as.character(all_pos_sentences_words$word)) #transforming to NA
all_pos_sentences_words <- subset(all_pos_sentences_words, !is.na(all_pos_sentences_words$word))

#to negative words, removing stopwords in 2nd language for the description column
all_neg_sentences_words$word <- gsub('br', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('[[:digit:]]+', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('can', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('also', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('two', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('one', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words <- subset(all_neg_sentences_words, !is.na(all_neg_sentences_words$word))

###HOST ABOUT
all_neg_sentences_words$word <- gsub('try', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('assubset_reg', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('thats', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('one', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('well', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('bnb', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words$word <- gsub('one', NA, all_neg_sentences_words$word) #transforming to NA
all_neg_sentences_words <- subset(all_neg_sentences_words, !is.na(all_neg_sentences_words$word))

#Ploting words of positive sentences to number of occurences (x-axes)
all_pos_sentences_words %>%
  count(word, sort=TRUE) %>%
  mutate(word = reorder(word,n)) %>%
  top_n(25, word) %>%
  ggplot(aes(word,n)) +  
  geom_col() +
  labs(x = NULL, y = "Number of occurences") +
  coord_flip() +
  theme(text = element_text(size = 17)) +
  ggtitle("Word Frequency Histogram (positive sentences)")

# Get counts of words in pos (and neg) sentences
all_sentence_words <- full_join(all_pos_sentences_words %>% count(word, sort=TRUE),
                                all_neg_sentences_words %>% count(word, sort=TRUE),
                                by="word")
all_sentence_words[is.na(all_sentence_words$n.x), "n.x"] <- 0
all_sentence_words[is.na(all_sentence_words$n.y), "n.y"] <- 0

# Normalize counts by total number of words in each group and calculate ratio
all_sentence_words$n.x  <- all_sentence_words$n.x/sum(all_sentence_words$n.x)
all_sentence_words$n.y  <- all_sentence_words$n.y/sum(all_sentence_words$n.y)
all_sentence_words$diff <- all_sentence_words$n.x-all_sentence_words$n.y

par(mfrow=c(2,2))
all_sentence_words %>%
  mutate(word = reorder(word, -diff)) %>%           
  top_n(-20, diff) %>%
  ggplot(aes(word,diff)) +  
  geom_col() +
  labs(x = NULL, y = "Difference in word frequency (pos-neg)") +
  coord_flip() +
  theme(text = element_text(size = 17)) +
  ggtitle("Specific negative words")

all_sentence_words%>%
  mutate(word = reorder(word,diff)) %>%           
  top_n(20, diff) %>%
  ggplot(aes(word,diff)) +  
  geom_col() +
  labs(x = NULL, y = "Difference in word frequency (pos-neg)") +
  coord_flip() +
  theme(text = element_text(size = 17)) +
  ggtitle("Specific positive words")

ggplot() +
  geom_density(data = subset_rego, aes(x = overall, fill="lb"), alpha = 0.5) +
  geom_density(data = subset_rego1, aes(x = overall,fill="dg"), alpha = 0.5) +
  scale_fill_manual(name = "Host about", values = c("lb" = "blue", "dg" = "darsubset_reggreen"), labels = c("lb" = "Description", "dg" = "Host About")) +
  labs(x = "Polarity Score", y = "Density") +
  theme(text = element_text(size = 15))

ggplot() +
  geom_freqpoly(data = subset_rego,
                mapping = aes(x = fear, y = ..count..,
                              color = "Fear"), size = 1, alpha = 0.7) +
  geom_freqpoly(data = subset_rego,
                aes(x = anticipation, y = ..count..,
                    color = "Anticipation"),  size = 1, alpha = 0.7) +
  geom_freqpoly(data = subset_rego,
                aes(x = anger, y = ..count..,
                    color = "Anger"),  size = 1, alpha = 0.7) +
  geom_freqpoly(data = subset_rego,
                aes(x = joy, y = ..count..,
                    color = "Joy"),  size = 1, alpha = 0.7) +
  geom_freqpoly(data = subset_rego,
                aes(x = sadness, y = ..count..,
                    color = "Sadness"), size = 1, alpha = 0.7) +
  geom_freqpoly(data = subset_rego,
                aes(x = surprise, y = ..count..,
                    color = "Surprise"), size = 1, alpha = 0.7) +
  geom_freqpoly(data = subset_rego,
                aes(x = trust, y = ..count..,
                    color = "Trust"), size = 1, alpha = 0.7) +
  scale_color_manual("", values = c("green",  "ssubset_regyblue", "red", "orange","blacsubset_reg","purple","turquoise")) +
  scale_x_continuous(limits = c(0, 10), breasubset_regs = seq(0,10,1)) +
  scale_y_continuous(limits = c(0, 1000), breasubset_regs = seq(0,1000,100)) +
  labs(x = "Amount of instances in each listing",
       y = "Number of instances") +
  theme_bw() +
  theme(legend.title = element_blansubset_reg(),
        legend.text = element_text(size = 14),
        axis.title = element_text(size = 13, face = "bold",
                                  margin=margin(20,20,0,0)),
        legend.position = c(0.85, 0.6),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 14),
        legend.subset_regey.width = unit(1.5, "cm"))  


#Calculate distance from the center variable
data$longitude <- as.numeric(data$longitude)
data$latitude <- as.numeric(data$latitude)

#Amsterdam center coordinates
lon_centre <- 4.895168
lat_centre <- 52.370216

data <- data %>%
  mutate(centre_dis = distHaversine(c(longitude, latitude), c(lon_centre, lat_centre)))

#Map the listings based on coordinates (plot)
mapview(data, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE)

#Transform to spatial data  
sbux_sf <- st_as_sf(data, coords = c("longitude", "latitude"), crs = 4326)
mapview(sbux_sf, map.types = "Stamen.Toner")

hist(data$price,
     main="Price distribution 06 December 2021",
     xlab="Price distribution",
     xlim=c(0,500),
     col="lightblue")

hist(log(data$price),
     main="LogPrice distribution 06 December 2021",
     xlab="LogPrice distribution",
     col="lightblue")
mean(data$price)
median(data$price)

p <- ggplot(data=data, aes(x=price, fill=room_type)) +
  geom_density(adjust=1.5, alpha=.4) +
  xlim(0, 500)

p + theme(text=element_text(size=14), #change font size of all text
          axis.text=element_text(size=14), #change font size of axis text
          axis.title=element_text(size=14), #change font size of axis titles
          plot.title=element_text(size=14), #change font size of plot title
          legend.text=element_text(size=14), #change font size of legend text
          legend.title=element_text(size=14)) #change font size of legend title  

dev.off()
p

#Double checsubset_reg for English
data$comments <- ifelse(detect_language(data$comments) == "en", data$comments, "") 
data$description <- ifelse(detect_language(data$description) == "en", data$description, "") 
data$host_about <- ifelse(detect_language(data$host_about) == "en", data$host_about, "") 

#LDA
#Pre-process words for LDA
SEED <- 2020
words_to_remove <- c("een", "met", "het", "van", "voor", "aan", "appart", "br", "b", "e", "c", "n", "ht", "us", "wij", "onze", "hi", "y", "et", "la","es","go",
                     "one", "even", "yays", "now", "lisubset_rege", "also", "located", "the",
                     "casa", "per", "ho", "mia", "una", "che", "città", "di", "en", "isubset_reg", "blicens", "bthe", "mijn", "ben", "als", "oosubset_reg", "zijn", "ons", "huis", "heb", "graag")
corpus <- Corpus(VectorSource(data$description))
corpus <- tm_map(corpus, removeWords, c(stopwords("en"), words_to_remove))

#Construct tdm
tdm <- TermDocumentMatrix(corpus, control = list(
  removePunctuation = TRUE, # remove punctuation
  stopwords = T, #remove stopwords (no meaning,high frequency)
  tolower = TRUE, # masubset_rege all lower case
  removeNumbers = TRUE, # remove numbers
  bounds = list(local=c(2,Inf)))) # only include cases with at least two occurences
tdm <- as.matrix(tdm)

# Sort by descearing value of frequency
dtm_v <- sort(rowSums(tdm),decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v),freq=dtm_v)

dtm <- DocumentTermMatrix(corpus, control = list(#stemming = TRUE, stopwords = TRUE, 
  removeNumbers = TRUE, stopwords = TRUE,
  removePunctuation = TRUE))
dtm <- as.matrix(dtm)

#Calculate how often words occur in a sentence
term_count <- slam::col_sums(dtm > 0)
term_perc <- term_count / nrow(dtm)

#Split into train and validation sample, 80/20.
dtm <- dtm[slam::row_sums(dtm)>0,]
dtm <- dtm[,slam::col_sums(dtm)>0]
n <- nrow(dtm)
train <- sample(1:n, round(n * 0.80))
valid <- (1:n)[-train]

#Estimate LDA for different values of subset_reg and collect perplexity values
res <- NULL
TMlist <- list()  # list to collect results

for (l in seq(10, 50, 10)) {
  print(l)
  TM <- LDA(dtm[train, ], method="Gibbs", subset_reg = l, control = list(alpha = 0.5))
  p <- perplexity(TM, newdata = dtm[train, ])
  p_valid <- perplexity(TM, newdata = dtm[valid, ])
  
  res <- rbind(res, data.frame(subset_reg=l, perplexity = p, perplexity_validation = p_valid))
  TMlist <- append(TMlist, TM)
  
  print(res)
}

#Plotting perplexity
ggplot(res, aes(subset_reg)) +                    
  geom_line(aes(y=perplexity, colour="Train")) +
  geom_line(aes(y=perplexity_validation, colour="Validation")) + labs(colour="Sample") + theme(text=element_text(size=18))

res[which.min(res$perplexity_validation), ] #minimum perplexity

#Searching for the best alpha
resAlpha <- NULL
TMlistAlpha <- list()

for (a in seq(.5, 2, .5)) {
  print(a)
  
  TMAlpha <- LDA(dtm[train, ], method="Gibbs", subset_reg = 20, control = list(alpha = a))
  
  p <- perplexity(TMAlpha, newdata = dtm[train,])
  p_valid <- perplexity(TMAlpha, newdata = dtm[valid, ])
  
  resAlpha <- rbind(resAlpha, data.frame(alpha=a, perplexity = p, perplexity_validation = p_valid))
  TMlistAlpha <- append(TMlistAlpha, TMAlpha)
  
  print(resAlpha)
}

#Plot perplexity vs alpha
ggplot(resAlpha, aes(alpha)) +                    
  geom_line(aes(y=perplexity, colour="Train")) +   
  geom_line(aes(y=perplexity_validation, colour="Validation"))  +
  labs(colour="Sample") +
  theme(text=element_text(size=18))

# Show best
resAlpha[which.min(resAlpha$perplexity_validation),]

for (a in seq(0.1, 0.4, .1)) {
  print(a)
  
  TMAlpha <- LDA(dtm[train, ], method="Gibbs", subset_reg = 20, control = list(alpha = a))
  
  p <- perplexity(TMAlpha, newdata = dtm[train,])
  p_valid <- perplexity(TMAlpha, newdata = dtm[valid, ])
  
  resAlpha1 <- rbind(resAlpha, data.frame(alpha=a, perplexity = p, perplexity_validation = p_valid))
  TMlistAlpha <- append(TMlistAlpha, TMAlpha)
  print(resAlpha1)
}

alpha = 0.5
TM <- LDA(dtm, subset_reg = 30, control = list(seed = SEED, alpha=alpha))
posterior(TM)$topics

p <- posterior(TM)$terms
p[,1:30]

col_means(posterior(TM)$topics)
t <- topics(TM, 10)  
as.matrix(t)[1:10] # show the first 4

#Show top 10 terms
Terms <- terms(TM, 10)
Terms

#Plot best
u_terms <- tidy(TM, matrix = "beta")
u_terms

top_terms <- u_terms %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  filter(topic >= 10 & topic <= 20) %>%
  mutate( 
    topic = factor(topic),
    term = reorder_within(term, beta, topic)
  ) %>%
  ggplot(aes(term, beta, fill = topic)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  scale_x_reordered() +
  facet_wrap(facets = vars(topic), scales = "free", ncol = 3) +
  coord_flip()

#Terms that have the greatest difference in Beta 
#Between Group 1 and Group 2 by calculating log2(beta2/beta1)
gtr_dfr <- u_terms %>%
  mutate(topic = paste0("topic", topic)) %>%
  pivot_wider(names_from = topic, values_from = beta) %>% 
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

gtr_dfr %>%
  group_by(direction = log_ratio > 0) %>%
  top_n(15, abs(log_ratio)) %>%
  ungroup() %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_col() +
  labs(y = "Log2 ratio of beta in topic 2 / topic 1") +
  coord_flip()

#Plot word probabilities
dtm = dtm[slam::row_sums(dtm) > 0, ]
phi = as.matrix(topicmodels::posterior(TM)$terms)
theta <- as.matrix(topicmodels::posterior(TM)$topics)
vocab <- colnames(phi)
doc.length = slam::row_sums(dtm)
term.freq = col_sums(dtm)[match(vocab, colnames(dtm))]

json = createJSON(phi = phi, theta = theta, vocab = vocab,
                  doc.length = doc.length, term.frequency = term.freq)
serVis(json)

#Truncated regression (Tobit)
#Select relevant columns for the regression analysis
subset_reg <- subset(data, select = c(occupancy_rate, host_response_rate, price, nr_amenities, instant_boosubset_regable, centre_dis, room_type, number_of_reviews, 
                              host_identity_verified, host_has_profile_pic, host_is_superhost, safety_features, description_pol, host_about_pol, reviews))

#Transform variables
subset_reg$instant_boosubset_regable <- ifelse(subset_reg$instant_boosubset_regable == 't', 1, 0)
subset_reg$host_has_profile_pic <- ifelse(subset_reg$host_has_profile_pic == 't', 1, 0)
subset_reg <- subset_reg %>%
  mutate(host_about_pol = relevel(host_about_pol, ref = "Not Emotional"))
subset_reg <- subset_reg %>%
  mutate(description_pol = relevel(description_pol, ref = "Not Emotional"))

#dummy <- dummyVars(" ~ .", data=test)
#test <- data.frame(predict(dummy, newdata=test))
#Set new seed for regression and split dataset to train and test
set.seed(12345)
sample <- sample(c(TRUE, FALSE), nrow(subset_reg), replace=TRUE, prob=c(0.7,0.3))
train  <- subset_reg[sample, ]
test   <- subset_reg[!sample, ]

Tobit <- AER::tobit(occupancy_rate ~ ., left = 0, right = 1, data = train, control = list(maxiter=70))
summary(Tobit)

#Evaluation metrics
PseudoR2 <- function(obj){1 - as.vector(logLisubset_reg(obj)/logLisubset_reg(update(obj, . ~ 1)))}
PseudoR2(Tobit)
vif(Tobit)
