#Framing the Problem
#"No one should ever claim to be a data analyst until he or she has done string manipulation” - Gaston Sanchez
library(readr)
library(stringr)  #String Manipulation Functions
library(tm) #Corpus Functions
library(wordcloud) #For Wordcloud Graphic/Plot
library(ggplot2) #Plotting Functions
library(plyr) #Functional Shortcuts
library(dplyr) #Load after plyr to prioritize on stack for speed of functions
library(gridExtra) #Display multiple ggplot on a single page

#Loading the Data 
##Datasets have been copied to the website
train <- read.csv("https://raw.githubusercontent.com/ckornafel/cyphertext/master/ciphertext-challenge-iii/train.csv",
                 stringsAsFactors = FALSE)
test <- read.csv("https://github.com/ckornafel/cyphertext/raw/master/ciphertext-challenge-iii/test.csv",
                 stringsAsFactors = FALSE)

#Exploring the Training (Plaintext) Dataset

head(train[order(train$index),],15)
#As expected, the training dataset contains three columns, the plaintext id, the Shakespearian plaintext, and the index key. 
##The plaintext sample shows that each line/row of text could be partial examples of play lines, including randomly included titles. It does appear that
##some of the text can be grouped by certain whole Shakespearian works (King Henry) and broken out by scene line.
##Since the first part of King Henry IV contains 33 lines, I wanted to know what the next block of text might include. 

train[which(train$index > 33 & train$index < 43),]

head(train$text)
##Westmoreland is the next section of King Henry IV after King Henry's part (scene I). 

#Word Stemming (reducing words to their basic parts) - ommited due to reducing the predictive power of the original word 

#Validating Uniqueness in Text
length(unique(train$text))

#108755 unique lines of text matches that total number of rows in the training data set. Therefore, it signifies that there are no repeated lines which
##could complicate dicphering the cyphertext, given that there are multiple cyphers applied to the entire set. 

##Exploring the Test (Cyphertext) Dataset
str(test)#C
#The test (cyphertext) data set also contains three variables: ciphertext id, ciphertext, and difficulty. The difficulty value indicates the number
##of ciphers used on the plaintext. E.g. level one indicates that a single cipher was applied, level two indicates that an additional cipher was applied
##to the first ciphertext, etc. 

#Validating Levels of Difficulty
table(test$difficulty)
#Appears that the four different cyphertext groups represent a quarter of the training population (roughly 25% each)

#Splitting the Test (ciphertext) Dataset into its Four Difficulty Levels
test1<- test[test$difficulty==1,]
test2<- test[test$difficulty==2,]
test3<- test[test$difficulty==3,]
test4<- test[test$difficulty==4,]

rm(test) #Conserving memory by removing unused variables 

#Taking a closer look at the Different Levels of Ciphertext
head(test1$ciphertext) 
#The level 1 cyphertext sample indicates a type of cypher that is similar to a substitution cypher given that there appears
#to be space separations between "words" 
head(test2$ciphertext)
head(test3$ciphertext)
head(test4$ciphertext)

#Word Frequency

#Creating a Language Corpus for the Shakespeare Text
names(train)[1] <- "doc_id" #tm Corpus function requires specific named columns
corp <- Corpus(DataframeSource(train[,1:2]))

#Creating the Document Term Matrix - a matrix of words and their frequencies
dtm <- DocumentTermMatrix(corp) %>%
  as.matrix()

words <- sort(colSums(dtm), decreasing = TRUE)
dtm_sort <- data.frame(Term = names(words), Frequency=words)
rm(words)

nrow(dtm_sort) #Number of unique Shakespearean words

head(table(dtm_sort$Frequency),10)

ggplot(data = as.data.frame(head(table(dtm_sort$Frequency),25)), aes(x=Var1, y=Freq))+
  geom_bar(stat = "identity")+
  labs(title = "Term Frequency", x= "Term Rep", y= "Term Frequency")+
  theme_minimal()
#It appears that there are only a small subset of terms (approx 7) which appear with high frequency. One term appears over 25,000 times
#but the next most frequent term drops to below 10,000 occurances in the complete Shapkespearean text. 

nrow(select(filter(dtm_sort,Frequency == 1), Term))
nrow(select(filter(dtm_sort,Frequency == 1), Term))/nrow(dtm_sort)
#There are 27937 terms that appear only once in the Shakespear text. This highlights that slightly more than half (51.33%) of the entire
#set of terms are unique combinations of letters/punctuation. 

#Creating a wordcloud of the 500 most frequent terms
set.seed(123)
wordcloud(words = dtm_sort$Term, freq = dtm_sort$Frequency, min.freq = 1,
          max.words = 500, random.order = FALSE, rot.per = 0.2,
          colors = brewer.pal(8, "Dark2"))
#The most frequent Shakespeare terms are similar to those found in modern English (e.g. and, the, not, etc.). Normally, in text mining exercises, 
#these common (stop) words would be removed in order to focus on the more impactful words in the corpus. However, since I am working with cyphertext
#every word (and punctuation) could be represented in the ciphertext and therefore needs to remain in the plaintext. 


#Test Set Terms and Wordclouds
names(test1)[1] <- "doc_id"
d_test1<-Doc_Mtx(test1)
WC(d_test1,250) #Viewing the 250 most frequent terms in test set level 1

d_test2<-Doc_Mtx(test2)
WC(d_test2,250)

d_test3<-Doc_Mtx(test3)
WC(d_test3,250)

table(d_test3$Frequency)[1:10]
#There are no terms in test 3 which appear only once. The lowest frequency of occurance are two terms which appear twice and three times in the cypher text
#Given the large volume of individual number combinations, it may indicate that they represent letter pairs or phonetic sounds instead of whole terms. 



rm(dtm, uniq_terms) #Conserving memory by removing unused variables 


train$num_term<-sapply(train$text, WordCount)
table(train$num_term)
test1$num_term<-sapply(test1$ciphertext, WordCount)
table(test1$num_term)
test2$num_term<-sapply(test2$ciphertext, WordCount)
table(test2$num_term)
test3$num_term<-sapply(test3$ciphertext, WordCount)
table(test3$num_term)
#It does not appear that the count of terms for each section of Shakespeare Text correlates well with the count of terms from 
##each of the cypher texts. Those term counts that do align (e.g. one instance of a 49-term length Shakespeare text) also appear in
##multiple cypher texts (e.g one instance of a 49-term Test 1 text and one instance of a 49-term Test 2 text). Given the assumption 
##that there is no overlap of cypher to plain texts, it would appear that spaces may not term separators in the cypher texts. 

#Additionally, there are 416 instances of a single-term Shakespeare text but the smallest number of cypher text terms is seven in 
##both test1 and test2 sets. Either these single-term texts are hidden in test3 and test4, or another indication that spaces are not
##term separators in the cypher texts. 


term_freq <- train %>%
  group_by(num_term) %>%
  summarise(counts = n())

ggplot(head(term_freq,20), aes(x=num_term, y = counts ))+
  geom_bar(fill = "steelblue", stat = "identity")+
  theme_minimal()

#Individual Letters Patterns
all_text_train <- paste(train$text, collapse= "")
all_text_test1 <- paste(test1$ciphertext, collapse = "")
all_text_test2 <- paste(test2$ciphertext, collapse = "")
all_text_test3 <- paste(test3$ciphertext, collapse = "")
all_text_test4 <- paste(test4$ciphertext, collapse = "")

uniq_chr_train <-as.vector(unique(strsplit(all_text_train, "")[[1]]))
uniq_chr_test1 <-as.vector(unique(strsplit(all_text_test1, "")[[1]]))
uniq_chr_test2 <-as.vector(unique(strsplit(all_text_test2, "")[[1]]))
uniq_chr_test3 <-as.vector(unique(strsplit(all_text_test3, "")[[1]]))
uniq_chr_test4 <-as.vector(unique(strsplit(all_text_test4, "")[[1]]))

length(uniq_chr_train)
length(uniq_chr_test1)
length(uniq_chr_test2)
length(uniq_chr_test3)
length(uniq_chr_test4)
#Of the four cypher sets, test1 and test2 appear to use the same characters as the plain text set. However, the all-numeric cypher text
##set (test3) uses 11 characters (e.g. 0-9 and whitespaces) and the non-whitespace cypher text set (test4) uses only 65 unique characters

#A quick check to verify that the same characters are used for train and test1 / test2
sort(uniq_chr_train) == sort(uniq_chr_test1)
sort(uniq_chr_train) == sort(uniq_chr_test2)

#Characters in test4 (cypher) set which are not in the train (plain) set
setdiff(uniq_chr_test4,uniq_chr_train)

#Characters in train (plain) set which are not in the test4 (cypher) set
setdiff(uniq_chr_train,uniq_chr_test4)

#There are actually 17 different characters in total that are different between the two sets (train / test4 )


train_chr <- as.data.frame(table(strsplit(all_text_train, "")[[1]]))
train_chr$Prop <- (train_chr$Freq/sum(train_chr$Freq))
levels(train_chr$Var1)[match(" ", levels(train_chr$Var1))]<-"spc"


test_chr<- data.frame(Char =as.vector(strsplit(all_text_test1, "")[[1]]),
                                       Source = "test1" )
test_chr$Source <- as.character(test_chr$Source)
test_chr<- bind_rows(test_chr,data.frame(Char =as.vector(strsplit(all_text_test2, "")[[1]]),
                                       Source = as.character("test2" )))
levels(test_chr$Char)[match(" ", levels(test_chr$Char))]<-"spc"

str(train_chr)
str(test_chr)

comp_test <- test_chr %>%
  count(Char, Source) %>%
  mutate(Prop = prop.table(n))

train_plot_top <- ggplot(head(train_chr[order(-train_chr$Prop),],15), 
                          aes(x=reorder(Var1,-Prop), y=Prop)) +
  geom_bar(stat = "identity", fill = "steelblue")+
  labs(title = "Most Frequent Characters in Train-Set", 
       x ="Character",
       y= "Percentage")+
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))+
  theme_minimal()

test_plot_top <- ggplot(head(comp_test[order(-comp_test$Prop),],30),
                         aes(x=reorder(Char,-Prop), y=Prop, fill = Source)) +
  geom_bar(stat = "identity", position = "dodge")+
  labs(title = "Most Frequent Characters in Test-Sets 1", 
       x ="Character",
       y= "Percentage")+
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))+
  theme_minimal()


grid.arrange(train_plot_top,test_plot_top, nrow = 2)


#Focusing only on Punctuation
train_punct<-train_chr[2:13,]

test_punct<-test_chr[test_chr$Char == train_punct$Var1,]
comp_punct <- test_punct %>%
  count(Char, Source)


train_plot_pnt <- ggplot(train_punct, 
                         aes(x=reorder(Var1,-Freq), y=Freq)) +
  geom_bar(stat = "identity", fill = "steelblue")+
  labs(title = "Punctuation Counts in Train-Set", 
       x ="Character",
       y= "Percentage")+
  scale_y_continuous()+
  theme_minimal()

test_plot_pnt <- ggplot(comp_punct,
                        aes(x=reorder(Char,-n), y=n, fill = Source)) +
  geom_bar(stat = "identity", position = "dodge")+
  labs(title = "Punctuation Counts in Test-Sets 1 & 2", 
       x ="Character",
       y= "Percentage")+
  scale_y_continuous()+
  theme_minimal()


grid.arrange(train_plot_pnt,test_plot_pnt, nrow = 2)

rm(comp_test, comp_punct, train_punct,train_plot_pnt,train_plot_top, test_plot_pnt, test_plot_top,test_chr, test_punct)

#Adding Numeric Text Patterns and Preparing the Datasets for Predictive Analysis

#Removing Non-needed Columns from Test Datasets
test1$difficulty <- NULL
test2$difficulty <- NULL
test3$difficulty <- NULL
test4$difficulty <- NULL

#Adding Total Character Counts
train$chr_cnt <- sapply(train$text, LtrCount)
test1$chr_cnt <- sapply(test1$ciphertext, LtrCount)

#Plotting the Frequency of Character Counts 
chr_freq<- rbind(ChrFreqTbl(train), ChrFreqTbl(test1))

ggplot(chr_freq[chr_freq$chr_cnt<150,], aes(chr_cnt, counts, fill = source))+
  geom_bar(stat = "identity")+
  labs(title = "Character Counts (<150) by Row: \nTrain and Test1 Sets", y= "Character Counts Per Line", x= "Frequency")+
  theme_minimal()

rm(chr_freq)

#Adjusting Character Count for Train To Accomodate Padding
train$achr_cnt <- sapply(train$chr_cnt, RndUp100)

#Calculating the Amount of Padding 
train$pad_amt <- train$achr_cnt-train$chr_cnt

pad_plot <- ggplot(train, aes(x= pad_amt, y=achr_cnt, col = as.factor(achr_cnt)))+
  geom_point(stat = "identity")+
  labs(title = "Extra Characters Padded In Train", x = "Extra Characters", y= "Adjusted Char Length")+
  theme_minimal()

pad_plot + theme(legend.position = "none")


pad_box <- ggplot(train, aes(x="", y=pad_amt, col = as.factor(achr_cnt)))+
  geom_boxplot()+
  labs(title = "Extra Characters Padded In Train", y= "Adjusted Char Length", x = "", col = "Padded Char \nGroup")+
  theme_minimal()

pad_box+theme(legend.position = "bottom")

#The point and box plots for the padded character amounts show that the majority of the padding occurs within the 0 - 100 character rows. 
##This group (100) has an average of approx. 60 additional characters added to the Shakespeare text. However, it also has a range of up to 99 
##additional characters - having the largest spread of padding. The 700, 900, and 1100 (largest) groups have the fewest memebers and consist of approx
## 27, 60, and 72 (respectively) additional characters. I assume that the low number of these larger text blocks will compensate for the additional characters
## when predicting the plain-text. 


#Replacing Char Cnt Column with the Adjusted "Padded" Count
train$chr_cnt <- train$achr_cnt
train$achr_cnt <- NULL
train$pad_amt <- NULL




#I saw in the previous plot that the Train1 set had a large number of 100 term rows; therefore, I focused on the rows in which
##the padded character counts exceeded 100 (since 100 would be the minimum). It appears that the train set has text in excess of 600
##padded characters. However, Train1 looks like it peaks at 500 characters. The majority of the Train1 set consists of 100 character lengths
##followed by 200 characters, dropping to a minimum of cases over 300 characters. 

##Unfortunately, this means that the majority of Test1 text includes a large amount of padding which could obsure the predictive patterns.


#Attempt Number 1 Dataset Creation 
#Adding the Counts of All Capital Terms Greater than 2 Characters
train$cap_cnt <- sapply(train$text,AllCapCnt)
test1$cap_cnt <- sapply(test1$ciphertext, AllCapCnt)

#Adding the Counts of All LowerCase Terms Greater than 2 Characters
train$lwr_cnt <- sapply(train$text, AllLowerCnt)
test1$lwr_cnt <- sapply(test1$ciphertext, AllLowerCnt)

#Adding the Counts of Numeric Digits 
train$nbr_cnt <- sapply(train$text, AllNbrCnt)
test1$nbr_cnt <- sapply(test1$ciphertext, AllNbrCnt)

#Adding the Counts of Punctuation followed by Space or End of String
train$pnt_cnt <- sapply(train$text,PunctCnt)
test1$pnt_cnt <- sapply(test1$ciphertext, PunctCnt)







xxx<-as.vector((unlist(strsplit(as.character(s), "[[:punct:]]"))))
xxx<-as.vector((unlist(str_trim(xxx, side = "both"))))
ddd<-unname(sapply(xxx,WordCount))



table(train$pnt_cnt)
table(test1$pnt_cnt)


##Adding Character Frequency to Datasets
#Train Dataset

ltr_freq <- train[,2:3] #Creating a temp df to load char frequencies 
ltr_freq[,uniq_chr_train]<-uniq_chr_train #Adding the doc_id and Text to df
                    
#Counting Each Character and Storing the Counts in the DataFrame
for(i in 1:length(uniq_chr_train)){
  ltr_freq[,i+2]<-str_count(ltr_freq$text,coll(uniq_chr_train[i]))
}

#Since the Cypher text changes the character value, having the frequency for each character will not provide
##much benefit. Therefore, sorting the values of each character (most to least) and storing them as general counts
##could help illuminate frequency patterns regardless of any substituions 

#Sorting each Character Frequency to Most->Least
freq_mat <- ltr_freq[,3:76]
ltr_freq_sorted<-as.data.frame(t(apply(freq_mat,1,sort,decreasing = TRUE)))
#ltr_freq_sorted <- ltr_freq_sorted[,which(colSums(ltr_freq_sorted) !=0)] #Removing columns of all zeros
ltr_freq_sorted$index <- ltr_freq$index


pred_train<-merge(train,ltr_freq_sorted, by="index")

#Verifying that the char sums matches the previously calculated char counts
pred_train$chr_check <- rowSums(pred_train[,10:83])
sum(ifelse(pred_train$chr_cnt==pred_train$chr_check,0,1)) 
pred_train$chr_check <- NULL


#Test1 Dataset 
ltr_freq1 <- test1[,1:2] #Creating a temp df to load char frequencies 
ltr_freq1[,uniq_chr_train]<-uniq_chr_train #Adding the doc_id and Text to df

#Counting Each Character and Storing the Counts in the DataFrame
for(i in 1:length(uniq_chr_train)){
  ltr_freq1[,i+2]<-str_count(ltr_freq1$ciphertext,coll(uniq_chr_train[i]))
}

#Sorting each Character Frequency to Most->Least
freq_mat1 <- ltr_freq1[,3:76]
ltr_freq_sorted1<-as.data.frame(t(apply(freq_mat1,1,sort,decreasing = TRUE)))
#ltr_freq_sorted <- ltr_freq_sorted[,which(colSums(ltr_freq_sorted) !=0)] #Removing columns of all zeros
ltr_freq_sorted1$ciphertext_id <- ltr_freq1$ciphertext_id


pred_test1<-merge(test1,ltr_freq_sorted1, by="ciphertext_id")


#Verifying that the char sums matches the previously calculated char counts
pred_test1$chr_check <- rowSums(pred_test1[,8:81])
sum(ifelse(pred_test1$chr_cnt==pred_test1$chr_check,0,1)) 
pred_test1$chr_check <- NULL



rm(i,n,temp,temp2,freq_mat,ltr_freq2,ltr_freq3,ltr_matrix,ltr4)

#Replacing Char Cnt Column with the Adjusted "Padded" Count
pred_train$chr_cnt <- pred_train$achr_cnt
pred_train$achr_cnt <- NULL
pred_train$pad_amt <- NULL

#There are no text samples that utilize every character. Therefore, when the count rows are 
#resorted by frequency, it produces several columns of zeroes. In order to keep an eye on overfitting
#the predictive model, I will remove any columns in both test and train that have zero valued columns
#Checking to see if there are any columns that I could remove 
length(pred_train[,colSums(pred_train[9:82] !=0)>0])
length(pred_test1[,colSums(pred_test1[8:81] !=0)>0])

#The Test1 dataset has more character frequencies than the Train dataset. The additional character frequencies are 
##most likely a result of padded characters. To reduce the negative impact of the padded characters, I will remove 
##the columns based on the training set. 

pred_train[,62:83]<-NULL
pred_test1[,60:81]<-NULL


#Adding the level 1 key to guage predictive performance
key_lvl1 <- read_csv("https://www.kaggleusercontent.com/kf/18776315/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0.._ZItym4yN2BhBFuu0kFheQ.TApDQcDet_1m8WaG_XbKJKp0YZW1VZf8AtQ_mO09uoDQlaHmlSxt5-g7Ycd-VD9bM20w8GjEuUqNcY7vMuQn0mK8uXUTygj0b3LKgL2qYpgScki8SOq4oADWpiddtN4YnrBUcFToF4VNmE3_Ggyw4KDeG_34lBEIYsa0sHDUxx8.N8KelF68PXg7XWJwSN-UKw/submit-level-1.csv")
pred_test1<-merge(pred_test1, key_lvl1, by="ciphertext_id")

pred_test1 <- pred_test1[,c(60,1:59)] #Moving the index column to the front of the dataframe


#Changing the plaintext / ciphertext ids to rownames
rownames(pred_train) <- pred_train[,2]
pred_train[,2] <- NULL

rownames(pred_test1) <- pred_test1[,2]
pred_test1[,2] <- NULL


#Finally, change the name of the text column of  test1 to match train
names(pred_test1)[names(pred_test1)=="ciphertext"]<-"text"

#Saving modified dataframes to .csv file so that I can work on them in another IDE 
write.csv(pred_train,"pred_train.csv")
write.csv(pred_test1, "pred_test1.csv")

head(pred_test1$text)


##Further Modifications

df<- read.csv("cipher_corp.csv", stringsAsFactors = FALSE)
#I noticed there were quite a few missing values in the csv file
df <-df[complete.cases(df),]

colnames(df)[2] <- "cipherwd"

#Keeping only one instance of each row and adding the total count of each instance
df_dedup <- ddply(df, .(plainwd,cipherwd), nrow)
colnames(df_dedup)[3] <- "occur"


df_dedup <- df_dedup[-1,] #Removing the "space" character

df_dedup$p_wd_len <- sapply(df_dedup$cipherwd, str_count, "")


df_ltr<-as.data.frame(str_split_fixed(df_dedup$cipherwd,"",max(df_dedup$p_wd_len)), stringsAsFactors = FALSE)

str(df_ltr)



df_ltr[(df_ltr=="")] <- "none"
df_ltr <- cbind(df_dedup, df_ltr)

str(df_ltr)

df_ltr$V3[8]

write_csv(df_ltr[,-2], "cipher_ltr.csv")

#Creating a dataset for the H2O implentation: less than 1000 classes. 
#Sorting the dataframe will group similar classes together
H2O_df <- df_ltr[order(df_ltr$plainwd ),] 

str(H2O_df )
#Removing the Ciphertext
H2O_df$cipherwd <- NULL

#Writing the dataset to csv 
write_csv(H2O_df[46642:48142,], "c_ltr_h2o.csv") #Saving the last 1500 items from the dataframe












