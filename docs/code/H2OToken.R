# This implementation of H2O's GBM using their word2vec function. 
# Example provided by H2O's website https://github.com/h2oai/h2o-3/blob/master/h2o-r/demos/rdemo.word2vec.craigslistjobtitles.R

library(h2o)
library(utils)
library(stringr)
library(tidyr)
library(readr)

corp <-read_csv("/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/corp.csv")

#Removing punctuation
corp <-as.data.frame(sapply(corp, function(x) as.character(gsub('[[:punct:]]+', " ",x))))

#Extracting the first two terms
corp$plain_fw <- word(corp$plain, 1,2, sep = " ")

#Reordering the dataframe columns (putting response in front) and dropping plain text column
corp <- corp[,c(3,2)]

x <- as.data.frame(table(corp$plain_fw))

head(x[order(-x$Freq),])

names <- c("KING HENRY", "GLOUCESTER", "HAMLET", "BRUTUS", "QUEEN MARGARET", "MARK ANTHONY", "PORTIA", "FALSTAFF", "DUKE VINCENTIO", "KING LEAR",
           "PROSPERO", "TITUS ANDRONICUS", "IMOGEN", "ROSALIND", "MACBETH", "HELENA", "CORIOLANUS", "BIRON", "PRINCE HENRY")

#Filtering for the parts above
corp_names <- subset(corp, plain_fw %in% names) #sorting by the reduced plaintext terms

#Saving the new dataset
write_csv(corp_names, "corpH2o.csv", col_names = FALSE) #Saving the file to be automatically uploaded into H2O


h2o.init()

cipher_corp <- h2o.importFile("/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/corpH2o.csv", destination_frame = "corp", col.names = c("plain", "cipher"),
                              col.types = c("Enum", "String"), header = FALSE, )

tokenize <- function(sentances){
  tokens <- h2o.tokenize(sentances, "\\\\W+")
}

decipher <- function(cipher, w2v, gbm){
  words <- tokenize(as.character(as.h2o(cipher)))
  cipher_vec <- h2o.transform(w2v, words, aggregate_method = "Average")
  h2o.predict(gbm, cipher_vec)
}

#Tokenize the ciphertext
c_words <- tokenize(cipher_corp$cipher)

#Use the H2O word to vector function for word embeddings
w2v_model <- h2o.word2vec(c_words, sent_sample_rate = 0, epochs = 10) 

#Transforming the prediction into vectors to use in the GMB model
cipher_vecs <- h2o.transform(w2v_model, c_words, aggregate_method = "AVERAGE")


valid_cipher <- ! is.na(cipher_vecs$C1) #Checking for valid characters
data <- h2o.cbind(cipher_corp[valid_cipher, "plain"], cipher_vecs[valid_cipher, ])
data.split <- h2o.splitFrame(data, ratios = 0.8) #splitting the set into train and test

#Creating the GBM model 
gbm <- h2o.gbm(x = names(cipher_vecs), y = "plain",
                     training_frame = data.split[[1]], validation_frame = data.split[[2]])

#Make Predictions using a line from Duke V
decipher("HKJP AXMNIDSTS  Cx exhdykjg ned ftat xay  jgpr", w2v_model, gbm)

h2o.shutdown()
 