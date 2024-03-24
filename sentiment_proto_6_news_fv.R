
# here we upload data from BBC as TRUE data

# set library location
# .libPaths(new=personal_lib)

# allows to avoid permission issues in case default location cant be accessed

personal_lib <- "C:\\Users\\cecetov\\Documents\\R\\win-library\\3.5"
.libPaths(new=personal_lib)
.libPaths()

# .libPaths(new=personal_lib)

# ** Setting location of Library ensures
# control in env with limited access, in case
# default R library location is not available


# ** Each Step of CRISP DM can be associated with 
# set of functions(methods) provided by set of libraries

library(dplyr) # data understanding / preparation
library(readr) # import export / data collection
library(readxl) # import export / data collection
library(tibble) #  data understanding / preparation
library(tidyr) #  data understanding / preparation
library(forcats) #  data understanding / preparation
library(data.table) # large datasets / data collection
library(lubridate) # dates manipulations /  data preparation
library(ggplot2) # visualization / data understanding /  modelling

library(tidytext)
library(rvest)

library(imputeTS) # replace missing data /  data preparation

# library(ranger) # modelling
# library(caret) # data preparation
library(corrplot) # visualisation /  data understanding /  modelling

library(xgboost) # classification
library(topicmodels)
library(jsonlite)


setwd("\\\\H1516\\Users\\cecetov\\Documents\\R\\R_Proto")
list.files()

#### Next, develop draft for sentiment

.udf_text_to_frame <- function( text = "" ) {
  
  buf <- strsplit( text , "\\.|,", fixed=FALSE , perl = TRUE )[[1]] 
  buf <- purrr::map_chr( buf, .f= function(x) 
  { gsub("\n","",x,fixed=TRUE) } )
  
  return (buf %>% '%d%'() )
  
}


##### Next, test approach on Tweets ----

d1_tweets <- read.csv("fake_news\\train.csv\\train.csv")
# glimpse(d1_tweets)
# nrow(d1_tweets)

# change encoding
d1_tweets$text <- iconv(d1_tweets$text,
                                from = "ISO-8859-1",
                                to = "UTF-8")

# glimpse(d1_tweets)

d1_tweets_abr <- d1_tweets %>% 
           select( label , text) %>% as_tibble()

# d1_tweets_abr %>% slice_sample(n=20)



#### Next add data from bbc ----

# "\\H1516\Users\cecetov\Documents\R\R_Proto\scrap_data\news.csv"


d1_bbc <- read.csv("scrap_data\\news2.csv", sep=",")

# glimpse(d1_bbc)

d1_bbc <- data.frame(label = rep(1,nrow(d1_bbc)),
                           text =  d1_bbc$text   )

d1_bbc$text <- iconv(d1_bbc$text,
                        from = "ISO-8859-1",
                        to = "UTF-8")

# glimpse(d1_bbc)

d1_tweets_abr <- d1_tweets_abr %>% 
        bind_rows(d1_bbc)

# debug
# nrow(d1_tweets_abr)
# tail(d1_tweets_abr,20)

.udf_calc_na_val( d1_tweets_abr )

# col_name na null emp_str
# 1    label  0    0       0
# 2     text  0    0      39


##  Next , clean data before tokenisation ----

d1_tweets_abr$text_clean <- gsub( "[^a-zA-Z]", " ", 
                                    d1_tweets_abr$text )

### remove twits with less than 5 words
loc <- lapply (  gregexpr( " " , d1_tweets_abr$text_clean ) , length )
d1_tweets_abr <- d1_tweets_abr[loc>5,]
# View(d1_tweets_abr)


#### Continue, proceed with tokenisation

d1_tweets_abr$sentence <- 1:nrow(d1_tweets_abr)

text_df <- tibble( name = d1_tweets_abr$sentence , 
                   value = d1_tweets_abr$text_clean ) 

##  Continue tokenisation of data, one token per row
tokens_data <- text_df %>% 
  unnest_tokens( word , value , to_lower = TRUE )

tokens_data_clean <- tokens_data %>% 
  anti_join( stop_words ) # identical column name

# remove numbers
tokens_data_clean <- tokens_data_clean %>%
  filter( !(grepl("[0-9]" , word) ) ) 

tokens_data_clean <- tokens_data_clean %>% rename( sentence = name )


# debug
tokens_data_clean %>% 
  count( word, sort=TRUE )


# AFINN sentiment score data
d1_afinn <- get_sentiments( "afinn" ) %>% 
      rename( sent_afinn = value )


#### Verify aligment of classification

tokens_data_clean_sentiment_afinn <- tokens_data_clean %>%
  
  inner_join( d1_afinn , by = c("word")) %>% 
    group_by( sentence  )  %>%
     summarize( sentiment = sum(sent_afinn) ) %>%
       mutate( method = "AFINN" )

quantile( tokens_data_clean_sentiment_afinn$sentiment , 
          probs = seq(0, 1, 0.1))

#   0%   10%   20%   30%   40%   50%   60%   70%   80%   90%  100% 
# -1041   -46   -27   -17   -10    -5    -1     3     8    19   297 

# all(tokens_data_clean_sentiment_afinn$sentence %in% tokens_data_clean$sentence)
# must be TRUE

d1_tweets_abr <- d1_tweets_abr %>% left_join( 
  
  tokens_data_clean_sentiment_afinn  , by = join_by("sentence") 
  
  )

.udf_calc_na_val( d1_tweets_abr ) # 20626
#     col_name   na null emp_str
# 1 airline_sentiment    0    0       0
# 2            method 4897    0      NA
# 3          sentence    0    0       0
# 4         sentiment 4897    0       0
# 5              text    0    0       0
# 6        text_clean    0    0       0
# 26% are too short or neutral

# keep only non missing values

d1_tweets_abr <- d1_tweets_abr %>%  
         filter( !(is.na(sentiment)) )


########  Next , determine sentiment

d1_tweets_abr_2 <- d1_tweets_abr 

quantile(d1_tweets_abr_2$sentiment)

# 0%   25%   50%   75%  100% 
# -1041   -21    -5     5   297 


##### Continue, filter on IQR interval
### Decreases dataset by 60%

d1_tweets_abr_2 <- d1_tweets_abr_2 %>% 
                     filter( sentiment>=-5 & sentiment<=5)

# range(d1_tweets_abr_2$sentiment)


### Next , assign dummy for Fake probability

# Generate 10 random numbers from 0 to 1

random_probabilities <- runif(nrow(d1_tweets_abr_2), min = 0, max = 1)

d1_tweets_abr_2$reliability <- random_probabilities


# Define the number of gradient steps and the range of sentiment values for the gradient

data <- d1_tweets_abr_2 %>% slice_sample(n=50)

data$article <- paste( "art" , 1:nrow(data), sep = "_" )


# Define the number of gradient steps and the range of sentiment values for the gradient
num_steps <- 50

min_sentiment_red <- -5
max_sentiment_red <- -1.5
min_sentiment_blue <- 1.5
max_sentiment_blue <- 5

# Create the plot with a gradient background for negative and positive sentiment values
plot <- ggplot(data, aes(x = reliability, y = sentiment)) +
  geom_point() +  # Add points
  geom_jitter(width = 0.02, height = 0.2) + 
  geom_text(aes(label = article), nudge_y = 0.25, check_overlap = TRUE) +
  labs(title = "Draft , News Reliability vs. Sentiment",
       x = "Reliability (Probability of Being Fake)",
       y = "Sentiment Score") +
  theme_minimal()  + scale_y_continuous(breaks = seq(-5, 5, by = 1)) +
  theme(plot.title = element_text(size = 15),  # Increase plot title size
        axis.title.x = element_text(size = 15),  # Increase x axis title size
        axis.title.y = element_text(size = 15))  # Increase y axis title size

# Add gradient rectangles for the negative sentiment range (red)
for (i in 1:num_steps) {
  min_y <- min_sentiment_red + (max_sentiment_red - min_sentiment_red) * (i - 1) / num_steps
  max_y <- min_sentiment_red + (max_sentiment_red - min_sentiment_red) * i / num_steps
  alpha_value <- 0.5 * (num_steps - i + 1) / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = Inf, 
                          ymin = min_y, ymax = max_y, fill = "red", alpha = alpha_value)
}

# Add gradient rectangles for the positive sentiment range (blue)
for (i in 1:num_steps) {
  min_y <- min_sentiment_blue + (max_sentiment_blue - min_sentiment_blue) * (i - 1) / num_steps
  max_y <- min_sentiment_blue + (max_sentiment_blue - min_sentiment_blue) * i / num_steps
  alpha_value <- 0.5 * i / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = Inf, 
                          ymin = min_y, ymax = max_y, fill = "blue", alpha = alpha_value)
}

# Display the plot
# print(plot)




#####  Next to Gradient Boosting ----

#### Sentiment is detected, now we need to train model to 
#### estimate probability of Fake


# debug
d1_tweets_abr_2 <- d1_tweets_abr_2 %>% rename(fake_var_target = label)
prop.table(table(d1_tweets_abr_2$fake_var_target))

# 0         1 
# 0.4056552 0.5943448

# glimpse(d1_tweets_abr_2)


# rename to avoid duplication of names
d1_tweets_abr_2 <- d1_tweets_abr_2 %>% 
     rename( sentiment_var=sentiment )
# glimpse(d1_tweets_abr_2)

d1_tweets_abr_2$reliability <- NULL


# Continue, estimate occurances
word_counts <- tokens_data_clean %>%
  count(word, sort = TRUE)  # Sort TRUE ensures the most frequent words are first

# Calculate the cumulative percentage
word_counts <- word_counts %>%
  mutate(total = sum(n),
         cumulative_n = cumsum(n),
         cumulative_percent = (cumulative_n / total) * 100)

# Filter for words that make up the first 80% of occurrences
selected_words <- word_counts %>%
  filter(cumulative_percent <= 80)

# Use these selected words to filter your original dataset
tokens_data_filtered <- tokens_data_clean %>%
  filter(word %in% selected_words$word)


# Next create wide format 
# Count the number of occurrences of each word in each sentence
word_counts <- tokens_data_filtered %>%
                  count(sentence, word)

# column to implement join
word_counts <- word_counts %>% rename(sentence_id = sentence)

# Reshape the data so each word becomes a column and the values are the counts of the words in each sentence
wide_data <- word_counts %>%
  pivot_wider(names_from = word, values_from = n, values_fill = list(n = 0))

tokens_data_clean_xg <- wide_data

# debug

# dim(tokens_data_clean_xg)
# 20527  3949

# min(tokens_data_clean_xg$sentence)

d1_tweets_abr_2 <- d1_tweets_abr_2 %>% 
                    rename(sentence_id = sentence)


# add sentiment_var as models feature
tokens_data_clean_xg <- tokens_data_clean_xg %>% 
  left_join(
    
    select(d1_tweets_abr_2, sentence_id, fake_var_target ,sentiment_var ) , 
        by = c("sentence_id") ) %>% 
  
  relocate( sentence_id, fake_var_target , sentiment_var, everything() ) 

# glimpse(tokens_data_clean_xg)
# names(tokens_data_clean_xg[1:10])
# names(tokens_data_clean_xg[1:2000])


#### Next , Proceed to modelling

### Join preprocessed data
restored <- tokens_data_clean_xg %>% 
          filter( !(is.na(fake_var_target)) )

id_store <- restored$sentence_id

restored$sentence_id <- NULL
# glimpse(restored)
# dim(restored)
# names(restored[1:100])


set.seed(123)

## Generate Folds
inFolds <- caret::createFolds( restored$fake_var_target , k=5 )

inTraining <- list()

for( i in 1:length(inFolds) ) {
  
  inTraining[[i]] <- unname(unlist( inFolds[-i] ))
  
}

training <- lapply(inTraining, function(x) { restored[x,] } ) 
valid <- lapply(inTraining, function(x) { restored[-x,] } ) 

loop_sens_score <- rep(0,length(inTraining))
loop_spec_score <- rep(0,length(inTraining))
loop_ac_score <- rep(0,length(inTraining))
loop_f1_score <- rep(0,length(inTraining))


  for( i in 1:(length(inTraining)  ) ) { 
    
    training_buf <- training[[i]]
    valid_buf <- valid[[i]]
    
    print( paste("loop" , i , sep="_") )
    flush.console()
    
    label <- as.integer( training_buf$fake_var_target )   # training_buf$y[1:2]    label[1:2]
    
    dtrain <- xgb.DMatrix( data = as.matrix( training_buf[, -which(names(training_buf) == "fake_var_target")] ),
                           label = label,   
                           nthread = 4)

    xgb_params <- list(
      nthread = 22 , 
      booster = "gbtree",
      eta = 0.01, 
      max_depth = 7,
      gamma = 0,
      subsample = 0.5, 
      colsample_bytree=1,
      objective = "binary:logistic",
      eval_metric = "error" 
    )
    
    xgb_model <- xgb.train(
      params = xgb_params,
      data = dtrain,
      nrounds = 300,  
      verbose = 1
    )
    
    # validate model
    dnew = xgb.DMatrix( data = as.matrix( valid_buf[, -which(names(valid_buf) == "fake_var_target")] ) )
    
    xgb_preds <- predict( xgb_model , dnew , reshape = TRUE)
    xgb_preds <- as.data.frame( xgb_preds )
    
    xgb_preds$y_hat <- dplyr::if_else( xgb_preds$xgb_preds>=0.5,1,0 )
    
    xgb_preds$fake_var_target <- as.integer(valid_buf$fake_var_target)
    
    # estimate confusion matrix
    conf_matrix <- caret::confusionMatrix( factor( xgb_preds$y_hat ) , 
                                           factor(xgb_preds$fake_var_target)  )
    
    loop_sens_score[i] <- conf_matrix$byClass[1] # Sensitivity_val
    loop_spec_score[i] <- conf_matrix$byClass[2] # Specificity_val
    
    loop_ac_score[i] <- conf_matrix$overall[1]
    
    precision <- conf_matrix$byClass['Pos Pred Value']
    recall <- conf_matrix$byClass['Sensitivity']
    loop_f1_score[i] <- 2 * (precision * recall) / (precision + recall)
    
  } # loop i cross validation


# quantile( loop_f1_score )

#     0%       25%       50%       75%      100% 
# 0.8701595 0.8741259 0.8783943 0.8793103 0.8908189 


# quantile( loop_ac_score )
#     0%       25%       50%       75%      100% 
# 0.8967391 0.9020852 0.9067029 0.9111514 0.9202176 





###  Continue train model on full dataset

label <- as.integer( restored$fake_var_target ) # training_buf$y[1:2]    label[1:2]

dtrain <- xgb.DMatrix( data = as.matrix( restored[, -which(names(training_buf) == "fake_var_target")] ),
                       label = label,   
                       nthread = 4)

xgb_params <- list(
  nthread = 27 , 
  booster = "gbtree",
  eta = 0.01, 
  max_depth = 7,
  gamma = 0,
  subsample = 0.5, 
  colsample_bytree=1,
  objective = "binary:logistic",
  eval_metric = "error" 
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 300,  verbose = 1
)



#### Visualisation of final output ----

# restored
# id_store 
# d1_tweets_abr_2

## generate prediction for whole dataset and add to 


dnew = xgb.DMatrix( data = as.matrix( restored[, -which(names(restored) == "fake_var_target")] ) )

xgb_preds <- predict( xgb_model , dnew , reshape = TRUE)
xgb_preds <- as.data.frame( xgb_preds )

# debug 
# nrow( xgb_preds )== length(id_store) #must be TRUE

restored_v2 <- data.frame( reliability = xgb_preds , 
                           sentence_id = id_store )
# glimpse(restored_v2)

restored_v2 <- restored_v2 %>% rename( reliability = xgb_preds )

d1_tweets_abr_3 <- d1_tweets_abr_2 %>% left_join( restored_v2 , 
                                                  by=c("sentence_id") )

# glimpse(d1_tweets_abr_3)



#### Next proceed to forecast of news data ----

data <- d1_tweets_abr_3 %>% filter( abs( sentiment_var ) <= 5 )
# glimpse(data)

set.seed(1234)

data <- data %>% slice_sample(n=50)

data$article <- paste( "art" , 1:nrow(data), sep = "_" )


# Define the number of gradient steps and the range of sentiment values for the gradient
num_steps <- 50

min_sentiment_red <- -5.1
max_sentiment_red <- -1.5
min_sentiment_blue <- 1.5
max_sentiment_blue <- 5.1

min_sentiment_green <- -1.5
max_sentiment_green <- 1.5

# Create the plot with a gradient background for negative and positive sentiment_var values
plot <- ggplot(data, aes(x = reliability, y = sentiment_var)) +
  geom_point() +  # Add points
  geom_jitter(width = 0.02, height = 0.2) + 
  geom_text(aes(label = article), nudge_y = 0.25, check_overlap = TRUE) +
  labs(title = "News Reliability vs. Sentiment",
       x = "Probability of Fake News",
       y = "Sentiment Score") +
  theme_minimal()  + scale_y_continuous(breaks = seq(-5, 5, by = 1)) +
  theme(plot.title = element_text(size = 15),  # Increase plot title size
        axis.title.x = element_text(size = 15),  # Increase x axis title size
        axis.title.y = element_text(size = 15))  # Increase y axis title size

# Add gradient rectangles for the negative sentiment_var range (red)
for (i in 1:num_steps) {
  min_y <- min_sentiment_red + (max_sentiment_red - min_sentiment_red) * (i - 1) / num_steps
  max_y <- min_sentiment_red + (max_sentiment_red - min_sentiment_red) * i / num_steps
  alpha_value <- 0.5 * (num_steps - i + 1) / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = Inf, 
                          ymin = min_y, ymax = max_y, fill = "red", alpha = alpha_value)
}

# Add gradient rectangles for the positive sentiment_var range (blue)
for (i in 1:num_steps) {
  min_y <- min_sentiment_blue + (max_sentiment_blue - min_sentiment_blue) * (i - 1) / num_steps
  max_y <- min_sentiment_blue + (max_sentiment_blue - min_sentiment_blue) * i / num_steps
  alpha_value <- 0.5 * i / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = Inf, 
                          ymin = min_y, ymax = max_y, fill = "blue", alpha = alpha_value)
}

# Add gradient rectangles for tneutral
for (i in 1:num_steps) {
  min_y <- min_sentiment_green + (max_sentiment_green - min_sentiment_green) * (i - 1) / num_steps
  max_y <- min_sentiment_green + (max_sentiment_green - min_sentiment_green) * i / num_steps
  alpha_value <- 0.5 * i / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = 0.5, 
                          ymin = min_y, ymax = max_y, fill = "green", alpha = alpha_value)
}

plot <- plot +
  theme(plot.title = element_text(size = 15),  # Increase plot title size
        axis.title.x = element_text(size = 15),  # Increase x axis title size
        axis.title.y = element_text(size = 15, 
                                    margin = margin(t = 0, r = 0, b = 10, l = 10, unit = "pt")))  # Increase y axis title size and padding

# Display the plot
print(plot)


saveRDS( xgb_model, "class_model_fake" )

saveRDS( selected_words, "selected_words" )




####  Next Topic Modelling

word_counts_dtm <- word_counts %>% 
       cast_dtm( sentence_id , word  , n )

word_counts_dtm

sentences_lda <- LDA( word_counts_dtm , k=3 , 
                           control = list(seed=1234) )


ap_topics <- tidy( sentences_lda, matrix = "beta")

ap_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)





###### Docker Function ----


# selected_words
xgboost_model <- readRDS("class_model_fake")

# models features
selected_words <- xgboost_model$feature_names

# select bbc news sample
set.seed(123)

sample <-  read.csv("scrap_data\\news2.csv", sep=",")[[1]][8]

# change encoding
sample <- iconv(sample,from = "ISO-8859-1" , to = "UTF-8")

text_df <- tibble( name = 1 , 
                   value = sample ) 


##  Continue tokenisation of data, one token per row
tokens_data <- text_df %>% 
  unnest_tokens( word , value , to_lower = TRUE )

tokens_data_clean <- tokens_data %>% 
  anti_join( stop_words ) # identical column name

# remove numbers
tokens_data_clean <- tokens_data_clean %>%
  filter( !(grepl("[0-9]" , word) ) ) 

tokens_data_clean <- tokens_data_clean %>% rename( sentence = name )

d1_afinn <- get_sentiments( "afinn" ) %>% rename( sent_afinn = value )

#### Verify aligment of classification

tokens_data_clean_sentiment_afinn <- tokens_data_clean %>%
  
  inner_join( d1_afinn , by = c("word")) %>% 
  group_by( sentence  )  %>%
  summarize( sentiment = sum(sent_afinn) ) %>%
  mutate( method = "AFINN" )

quantile( tokens_data_clean_sentiment_afinn$sentiment ,
          probs = seq(0, 1, 0.1))

#   0%   10%   20%   30%   40%   50%   60%   70%   80%   90%  100% 
# -1041   -46   -27   -17   -10    -5    -1     3     8    19   297 

# all(tokens_data_clean_sentiment_afinn$sentence %in% tokens_data_clean$sentence)
# must be TRUE


## bound_scale

tokens_data_clean_sentiment_afinn$sentiment <- if_else( tokens_data_clean_sentiment_afinn$sentiment > 5 , 5 , 
                                                        ifelse( tokens_data_clean_sentiment_afinn$sentiment < -5 , -5 ) )
             
             
text_df <- text_df %>% rename(sentence = name)

text_df <- text_df %>% left_join( 
  
  tokens_data_clean_sentiment_afinn  , by = join_by("sentence") 
  
)


### Hardest Part, fit features names

## available in test data
tokens_data_clean <- tokens_data_clean %>%
           filter( word %in% selected_words )

word_counts <- tokens_data_clean %>%
  count(sentence, word)

### all( word_counts$word %in% selected_words )

# not in test data
buf <- selected_words[ !(selected_words %in% word_counts$word) ]

buf <- data.frame( sentence = rep(1, length(buf) )  ,
                  word = buf , 
                  n = rep(0, length(buf) ) )

### unite data 
word_counts <- word_counts %>% bind_rows(buf)

word_counts <- word_counts %>% rename(sentence_id = sentence)


# Reshape the data so each word becomes a column, and the values are the counts of the words in each sentence
wide_data <- word_counts %>%
  pivot_wider(names_from = word, values_from = n, values_fill = list(n = 0))

tokens_data_clean_xg <- wide_data

# prepare data for modelling
id_store <- tokens_data_clean_xg$sentence_id

tokens_data_clean_xg$sentence_id <- NULL

# tokens_data_clean_xg$sentiment_var <- text_df$sentiment

# all( names(tokens_data_clean_xg) %in%  xgboost_model$feature_names )
# all( xgboost_model$feature_names %in%  names(tokens_data_clean_xg) )

tokens_data_clean_xg <- tokens_data_clean_xg %>% 
             select( xgboost_model$feature_names )

tokens_data_clean_xg$sentiment_var <- text_df$sentiment

dnew = xgb.DMatrix( data = as.matrix( tokens_data_clean_xg[ , ] ) )

xgb_preds <- predict( xgboost_model , dnew , reshape = TRUE)
xgb_preds <- as.data.frame( xgb_preds )

# debug 
# nrow( xgb_preds )== length(id_store) #must be TRUE

restored_v2 <- data.frame( fake_prob = xgb_preds , 
                           sentence_id = id_store )
# glimpse(restored_v2)

restored_v2 <- restored_v2 %>% rename( fake_prob = xgb_preds )

d1_tweets_abr_3 <- text_df %>% left_join( restored_v2 , 
                                                  by=c("sentence" = "sentence_id") )


#### Next proceed point visualisation single point ----

data <- d1_tweets_abr_3 %>% rename( sentiment_var=sentiment )

# glimpse(data)

set.seed(1234)

data <- data %>% slice_sample(n=50)

data$article <- paste( "art" , 1:nrow(data), sep = "_" )


# Define the number of gradient steps and the range of sentiment values for the gradient
num_steps <- 50

min_sentiment_red <- -5.1
max_sentiment_red <- -1.5
min_sentiment_blue <- 1.5
max_sentiment_blue <- 5.1

min_sentiment_green <- -1.5
max_sentiment_green <- 1.5

# Create the plot with a gradient background for negative and positive sentiment_var values
plot <- ggplot(data, aes(x = fake_prob, y = sentiment_var)) +
  geom_point( size = 3) +  # Add points
  # geom_jitter(width = 0.02, height = 0.2) + 
  scale_x_continuous( limits = c(0,1)) + 
  geom_text(aes(label = article), nudge_y = 0.25, check_overlap = TRUE) +
  labs(title = "News Reliability vs. Sentiment",
       x = "Probability of Fake News",
       y = "Sentiment Score") +
  theme_minimal()  + scale_y_continuous(breaks = seq(-5, 5, by = 1)) +
  theme(plot.title = element_text(size = 15),  # Increase plot title size
        axis.title.x = element_text(size = 15),  # Increase x axis title size
        axis.title.y = element_text(size = 15))  # Increase y axis title size

# Add gradient rectangles for the negative sentiment_var range (red)
for (i in 1:num_steps) {
  min_y <- min_sentiment_red + (max_sentiment_red - min_sentiment_red) * (i - 1) / num_steps
  max_y <- min_sentiment_red + (max_sentiment_red - min_sentiment_red) * i / num_steps
  alpha_value <- 0.5 * (num_steps - i + 1) / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = Inf, 
                          ymin = min_y, ymax = max_y, fill = "red", alpha = alpha_value)
}

# Add gradient rectangles for the positive sentiment_var range (blue)
for (i in 1:num_steps) {
  min_y <- min_sentiment_blue + (max_sentiment_blue - min_sentiment_blue) * (i - 1) / num_steps
  max_y <- min_sentiment_blue + (max_sentiment_blue - min_sentiment_blue) * i / num_steps
  alpha_value <- 0.5 * i / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = Inf, 
                          ymin = min_y, ymax = max_y, fill = "blue", alpha = alpha_value)
}

# Add gradient rectangles for tneutral
for (i in 1:num_steps) {
  min_y <- min_sentiment_green + (max_sentiment_green - min_sentiment_green) * (i - 1) / num_steps
  max_y <- min_sentiment_green + (max_sentiment_green - min_sentiment_green) * i / num_steps
  alpha_value <- 0.5 * i / num_steps
  plot <- plot + annotate("rect", xmin = -Inf, xmax = 0.5, 
                          ymin = min_y, ymax = max_y, fill = "green", alpha = alpha_value)
}

plot <- plot +
  theme(plot.title = element_text(size = 15),  # Increase plot title size
        axis.title.x = element_text(size = 15),  # Increase x axis title size
        axis.title.y = element_text(size = 15, 
                                    margin = margin(t = 0, r = 0, b = 10, l = 10, unit = "pt")))  # Increase y axis title size and padding

# Display the plot
print(plot)








