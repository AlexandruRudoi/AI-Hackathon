

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
# library(rvest)

# library(imputeTS) # replace missing data /  data preparation

# library(ranger) # modelling
# library(caret) # data preparation
# library(corrplot) # visualisation /  data understanding /  modelling

library(jsonlite)
library(xgboost) # classification
# library(topicmodels)




###### Docker Function ----

## Upload text
# Path to the .txt file
file_path <- "demo\\input.txt"

# Read the file
sample <- readLines(file_path)

# Join all lines into one text, preserving line breaks
sample <- paste(sample, collapse = "\n")

sample <- iconv(sample,from = "ISO-8859-1" , to = "UTF-8")



# selected_words
xgboost_model <- readRDS("class_model_fake")

# models features
selected_words <- xgboost_model$feature_names

# select bbc news sample
set.seed(123)



####  Next, proceed to tokenisation

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

# quantile( tokens_data_clean_sentiment_afinn$sentiment ,
#           probs = seq(0, 1, 0.1))

#   0%   10%   20%   30%   40%   50%   60%   70%   80%   90%  100% 
# -1041   -46   -27   -17   -10    -5    -1     3     8    19   297 

# all(tokens_data_clean_sentiment_afinn$sentence %in% tokens_data_clean$sentence)
# must be TRUE


## bound_scale

tokens_data_clean_sentiment_afinn$sentiment <- if_else( tokens_data_clean_sentiment_afinn$sentiment > 5 , 3 ,
                                                        ifelse( tokens_data_clean_sentiment_afinn$sentiment < -5 , -3 ) )


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
  geom_point( size = 4) +  # Add points
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


# Convert the data frame to JSON
json_data <- toJSON(data, pretty = TRUE)

# Print the JSON output
print(json_data)












