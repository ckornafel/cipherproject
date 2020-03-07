#Creates a Document Term Matrix given a dataframe
#Used to generate wordclouds from Corpus
Doc_Mtx <- function(df){
  
  names(df)[1] <- "doc_id"
  names(df)[2] <- "text"
  c <- Corpus(DataframeSource(df[,1:2]))
  
  #Creating the Document Term Matrix - a matrix of words and their frequencies
  d <- DocumentTermMatrix(c) %>%
    as.matrix()
  
  w <- sort(colSums(d), decreasing = TRUE)
  rm(d)
  d_sort <- data.frame(Term = names(w), Frequency=w)
  rm(w)
  
  return(d_sort)
}

#Creates a Wordcloud graphic of most frequent terms in a dataframe (DTM)
#Default argument of 100 most frequent terms
WC <- function(df, x=100){
  return( wordcloud(words = df$Term, freq = df$Frequency, min.freq = 1,
            max.words = x, random.order = FALSE, rot.per = 0.2,
            colors = brewer.pal(8, "Dark2")))
}


WordCount <- function(x){
  return(length(unlist(strsplit(as.character(x), "\\W+"))))
  
}

LtrCount <- function(x){
  return(length(unlist(strsplit(as.character(x), ""))))
  
}

ChrFreqTbl <- function(df, col){
  r <- df %>%
    group_by(chr_cnt) %>%
    summarise(counts = n())
  r$source <- deparse(substitute(df))
  return(r)
}

RndUp100 <- function(x){
  return(ceiling(x/100)*100)
}


PunctCnt <- function(x){
  y<- (str_count(x,"[[:punct:]][[:space:]]"))
  z<- (str_count(x,"[[:punct:]]$"))
  return(y+z)
}

AllCapCnt <- function(x){
  return(str_count(x, "\\b[A-Z]{2,}\\b"))
}

AllLowerCnt <- function(x){
  return(str_count(x, "\\b[a-z]{2,}\\b"))
}

AllNbrCnt <- function(x){
  return(str_count(x, "[[:digit:]]"))
}


PunctWordCnt <- function(x){
  l<-as.vector((unlist(strsplit(as.character(x), "[[:punct:]][[:space:]]"))))
  l<-as.vector((unlist(str_trim(l, side = "both"))))
  r<-as.numeric(unname(sapply(l,WordCount)))
  return(r)
}

AddZeros <- function(x){
  if(is.factor(x))
    return(factor(x, c(levels(x), "None")))
  return(x)
}

