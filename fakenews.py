import pandas as pd
#data preprocess- data exploring/retrive
df = pd.read_csv('Cleaned_Data_spell.csv', encoding='latin1')
df.head(5)
df.shape
df.isnull().sum()
df['Processed_Title'].nunique()
df.info()
df.describe()
type(df['News_Headline'])
pd.set_option('display.max_columns', None)
df.drop(to_drop, inplace=True, axis=1)
df = df.drop(['column_name1', 'column_name2', 'column_name3'], axis=1)
# Importing Libraries
import unidecode
import pandas as pd
import re
import time
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import timeit
stoplist = stopwords.words('english') 
stoplist = set(stoplist)
spell = Speller(lang='en')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

#----------------------------craw website content----------------------------------
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
from requests.packages.urllib3.util.retry import Retry
from time import sleep
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('punkt')

CIPHERS = (
    'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:'
    'ECDHE+AES:DHE+AES:!AES128:!aNULL:!MD5:!DSS'
)

class DESAdapter(HTTPAdapter):
    """
    A TransportAdapter that re-enables 3DES support in Requests.
    """
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=CIPHERS)
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=CIPHERS)
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).proxy_manager_for(*args, **kwargs)

s = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])
s.mount('https://', DESAdapter(max_retries=retries))

def extract_content(url):
    sleep(1) # delay between requests
    response = s.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = ''
    for paragraph in soup.find_all('p'):
        content += paragraph.text
    return content

df['Link_content'] = df['Link_Of_News'].apply(extract_content)

df.to_csv('Cleaned_Data_ML.csv', index = False)
df
# clean the content
df2.head(5)
# Define the starting phrase and ending phrase of the required content
start_phrase = "More Info"
end_phrase = "The Principles of the Truth-O-Meter"

# Make sure to escape any special characters in start_phrase and end_phrase
start_phrase = re.escape(start_phrase)
end_phrase = re.escape(end_phrase)

# Create the regex pattern
pattern = f"{start_phrase}.*{end_phrase}"

# Function to extract the required content
def extract_content(text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group()
    else:
        return None  # or some other value indicating no match

# Apply the function to each row in 'column_name'
df2['clean_linkcontent'] = df2['Link_content'].apply(extract_content)

# clean more info and last sentence

def clean_link(text):
    a = re.sub("More Info", '', text)
    b = re.sub("The Principles of the Truth-O-Meter", '5', a)
    return b

df2['clean_linkcontent'] = df2['clean_linkcontent'].astype(str)
df2['clean_linkcontent'] = df2['clean_linkcontent'].apply(clean_link)

#-------------------Data Preprocessing----------------------------
# remove newlines & tabs

def remove_newlines_tabs(text):
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_text

df['Clean_Headline'] = df['News_Headline'].astype(str)
df['clean_linkcontent'] = df['clean_linkcontent'].astype(str)
df['Clean_Headline'] = df['Clean_Headline'].apply(remove_newlines_tabs)
df['clean_linkcontent'] = df['clean_linkcontent'].apply(remove_newlines_tabs)


#Strip the link of news
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

df['Clean_Headline'] = df['Clean_Headline'].apply(strip_html_tags)
df['clean_linkcontent'] = df['clean_linkcontent'].apply(remove_newlines_tabs)

# strip link-- remove link
def remove_links(text):
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

df['Clean_Headline'] = df['Clean_Headline'].apply(remove_links)
df['clean_linkcontent'] = df['clean_linkcontent'].apply(remove_links)

# Remove WhiteSpaces

def remove_whitespace(text):
     pattern = re.compile(r'\s+') 
     Without_whitespace = re.sub(pattern, ' ', text)
     text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
     return text    
 
df['Clean_Headline'] = df['Clean_Headline'].apply(remove_whitespace)
df['clean_linkcontent'] = df['clean_linkcontent'].apply(remove_whitespace)
#------------------
# Step1: Remove Accented Characters & number
import unidecode
def accented_characters_removal(text):
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = unidecode.unidecode(text)
    return text

df['Processed_content_link'] = df['clean_linkcontent'].apply(accented_characters_removal)
len(df['Processed_content_link'])
#Step2: Case Conversion

def lower_casing_text(text):
    text = text.lower()
    return text

df['Processed_content_link'] = df['Processed_content_link'].apply(lower_casing_text)
#Step3: Reduce repeated characters and punctuations
def reducing_incorrect_character_repeatation(text):
     # Pattern matching for all case alphabets
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    
    # Limiting all the  repeatation to two characters.
    Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    
    # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    return Final_Formatted


df['Processed_content_link'] = df['Processed_content_link'].apply(reducing_incorrect_character_repeatation)
# Step4: Expand contraction words
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
}


def expand_contractions(text, contraction_mapping =  CONTRACTION_MAP):
    list_Of_tokens = text.split(' ')
    String_Of_tokens = text  # initialize the variable with original text
    for Word in list_Of_tokens: 
        if Word in CONTRACTION_MAP: 
            list_Of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_Of_tokens]
            String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens) 
    return String_Of_tokens

df['Processed_content_link'] = df['Processed_content_link'].apply(expand_contractions)
#Step5: Remove special characters
def removing_special_characters(text): 
    Formatted_Text = re.sub(r"[^a-zA-Z0-9:$-,%.?!]+", ' ', text)
    return Formatted_Text

df['Processed_content_link'] = df['Processed_content_link'].apply(removing_special_characters)
# Step6: Remove stopwords
def removing_stopwords(text):
    text = repr(text)
    No_StopWords = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
    words_string = ' '.join(No_StopWords)    
    return words_string
df['Processed_content_link'] = df['Processed_content_link'].apply(removing_stopwords)
# Step7: Correct mis-spelled words in text
def spelling_correction(text):
    spell = Speller(lang='en')
    Corrected_text = spell(text)
    return Corrected_text

df['Processed_content_link'] = df['Processed_content_link'].apply(spelling_correction)
len(df['Processed_content_link'] )
#Step8: Lemmatization
def lemmatization(text):
    lemma = [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]
    return lemma

df['Processed_content_link'] = df['Processed_content_link'].apply(lemmatization)
df.to_csv('Cleaned_Data_Spell.csv', index = False)
#Step9: Putting all in single function

def text_preprocessing(text, accented_chars=True, contractions=True, lemma = True,
                        extra_whitespace=True, newlines_tabs=True, repeatition=True, 
                       lowercase=True, punctuations=True, mis_spell=True,
                       remove_html=True, links=True,  special_chars=True,
                       stop_words=False):
    if newlines_tabs == True: #remove newlines & tabs.
        Data = remove_newlines_tabs(text)

    if remove_html == True: #remove html tags
        Data = strip_html_tags(Data)

    if links == True: #remove links
        Data = remove_links(Data)

    if extra_whitespace == True: #remove extra whitespaces
        Data = remove_whitespace(Data)

    if accented_chars == True: #remove accented characters
        Data = accented_characters_removal(Data)

    if lowercase == True: #convert all characters to lowercase
        Data = lower_casing_text(Data)

    if repeatition == True: #Reduce repeatitions   
        Data = reducing_incorrect_character_repeatation(Data)

    if contractions == True: #expand contractions
        Data = expand_contractions(Data)

    if punctuations == True: #remove punctuations
        Data = removing_special_characters(Data)

    stoplist = stopwords.words('english') 
    stoplist = set(stoplist)
    
    if stop_words == True: #Remove stopwords
        Data = removing_stopwords(Data)

    spell = Speller(lang='en')
    
    if mis_spell == True: #Check for mis-spelled words & correct them.
        Data = spelling_correction(Data)

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
     
    if lemma == True: #Converts words to lemma form.
        Data = lemmatization(Data)

    if isinstance(Data, list): # ensure Data is a single string
        Data = ' '.join(Data)
        
    return Data

def list_content(Data):
    if isinstance(Data, list):  # ensure Data is a list
        Data = ' '.join(str(i) for i in Data)  # convert all elements to string before joining
    return Data

df['Processed_content_link'] = df['Processed_content_link'].apply(list_content)
df['Processed_content_link']
print(type(df['Processed_content_link'].iloc[0]))
# Pre-processing for Newsheadline Content
List_Content = df['Clean_Headline'].to_list()
Final_Article = []
Complete_Content = []
for article in List_Content:
    Processed_Content = text_preprocessing(str(article)) #Cleaned text of Content attribute after pre-processing
    Final_Article.append(Processed_Content)
Complete_Content.extend(Final_Article)
df['Processed_Title'] = Complete_Content

len(Final_Article)
len(Processed_Content)
len(Complete_Content)
len(df['Sentiment'])
print(Processed_Content)
df
# preprocess link_content
List_Content1 = df['clean_linkcontent'].to_list()
Final_Article1 = []
Complete_Content1 = []
for article in List_Content1:
    Processed_Content1 = text_preprocessing(str(article)) #Cleaned text of Content attribute after pre-processing
    Final_Article1.append(Processed_Content1)
Complete_Content1.extend(Final_Article1)
df['Processed_content_link'] = Final_Article1

len(df['clean_linkcontent'])
len(Final_Article1)
len(Processed_Content1)
len(Complete_Content1)
len(df['Sentiment'])
len(df['Processed_content_link'])
print(Processed_Content1)
print(Final_Article1)
print(df['clean_linkcontent'])
#Saved
df.to_csv('Cleaned_Data_Spell.csv', index = False)


#---------------------Text Feature extraction------------------------
#Tokenization: This is the process of breaking up the text into individual words or tokens. This can be done using the nltk library:
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def tokenize_text(text):
    return word_tokenize(text)

df['Token_Headline'] = df['Clean_Headline'].apply(tokenize_text)
df['Token_Content'] = df['clean_linkcontent'].apply(tokenize_text)
# feature extraction-Word Frequence
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['Clean_Headline'])

word_freq_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(word_freq_df)

# N-grams

def get_ngrams(corpus, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(df['Clean_Headline'])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

N_gram = get_ngrams(df['Clean_Headline'],1)
print(N_gram)


# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


TF_IDF = get_tfidf(df['Processed_Title'])
TF_IDF_cont = get_tfidf(df['Processed_content_link'])
print(TF_IDF_cont)
df
# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

def get_cosin(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Clean_Headline'])
    cos_sim = cosine_similarity(X, X)
    return cos_sim

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Clean_Headline'])
cosine_sim = cosine_similarity(TF_IDF,TF_IDF)
list(cosine_sim)

Cosine1 = get_cosin(df['Clean_Headline'])
print(Cosine1)

#Sentinet feature- TExtBlob

from textblob import TextBlob


def get_sentiment_polar(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['Sentiment_blob'] = df['Processed_Title'].apply(get_sentiment_polar)
df['Sentiment_link_blob'] = df['Processed_content_link'].apply(get_sentiment_polar)
df['Sentiment_link_blob']

def get_sentiment_sub(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity

df['Sentiment_blo_sub'] = df['Processed_Title'].apply(get_sentiment_sub)
df['Sentiment_link_blob_sub'] = df['Processed_content_link'].apply(get_sentiment_sub)
df['Sentiment_link_blob_sub']

# Sentinet feature NLTK
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['Sentiment_NLTK'] = df['Processed_Title'].apply(lambda title: sia.polarity_scores(title)['compound'])
df['Sentiment_link_NLTK'] = df['Processed_content_link'].apply(lambda title: sia.polarity_scores(title)['compound'])
df['Sentiment_link_NLTK'].describe()

# sentiment: flair
from flair.models import TextClassifier
from flair.data import Sentence
from segtok.segmenter import split_single
import nltk
nltk.download('punkt')

classifier = TextClassifier.load('en-sentiment')
df['Sentiment_Flair'] = ''
for index, row in df.iterrows():
    sentence = Sentence(row['Processed_Title'])
    classifier.predict(sentence)
    df.at[index, 'Sentiment_Flair'] = sentence.labels[0].value
classifier.predict(sentence)
print(sentence)


def detect_flair_sentiment(text):
    # Split the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    
    scores = []
    for sentence in sentences:
        # Create a Sentence object
        sentence = Sentence(sentence)
        
        # Predict the sentiment
        classifier.predict(sentence)

        # Get the sentiment score
        value = sentence.labels[0].value 
        score = sentence.labels[0].score
        if value == 'NEGATIVE':
            score = -score
        scores.append(score)
    
    # Return the average sentiment score
    return sum(scores) / len(scores)


df['flair_sentiment_score'] = df['Processed_Title'].apply(detect_flair_sentiment)
df['Sentiment_link_flair'] = df['Processed_content_link'].apply(detect_flair_sentiment)
df.drop('Sentiment_Flair', inplace=True, axis=1)

df['flair_sentiment_score']
df['flair_sentiment_score'].describe()
df['Sentiment_link_flair'].describe()
#--------------other text feature-------------
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import pos_tag
from collections import Counter
nltk.download('averaged_perceptron_tagger')

# Number of stop words
def count_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stop_words_in_text = [w for w in word_tokens if w in stop_words]
    return len(stop_words_in_text)

df['StopW_NH'] = df['News_Headline'].apply(count_stop_words)
df['StopW_Cont'] = df['clean_linkcontent'].apply(count_stop_words)

# Number of uppercase words
def count_uppercase_words(text):
    return sum(map(str.isupper, text.split()))

df['NumUpper_NH'] = df['News_Headline'].apply(count_uppercase_words)
df['NumUpper_Cont'] = df['clean_linkcontent'].apply(count_uppercase_words)

# Number of lowercase words
def count_lowercase_words(text):
    return sum(map(str.islower, text.split()))

df['NumLow_NH'] = df['News_Headline'].apply(count_lowercase_words)
df['NumLow_Cont'] = df['clean_linkcontent'].apply(count_lowercase_words)

# Number of numerics
def count_numerics(text):
    return len([word for word in text.split() if word.isdigit()])

df['Number_NH'] = df['News_Headline'].apply(count_numerics)
df['NUmber_Cont'] = df['clean_linkcontent'].apply(count_numerics)

# Word count
def count_words(text):
    return len(text.split())

df['WordC_NH'] = df['News_Headline'].apply(count_words)
df['WordC_Cont'] = df['clean_linkcontent'].apply(count_words)

# Character count
def count_characters(text):
    return len(text)

df['Chrac_NH'] = df['News_Headline'].apply(count_characters)
df['Chrac_Cont'] = df['clean_linkcontent'].apply(count_characters)

# Sentence count
def count_sentences(text):
    return len(sent_tokenize(text))

df['Sentence_NH'] = df['News_Headline'].apply(count_sentences)
df['SSentenc_Cont'] = df['clean_linkcontent'].apply(count_sentences)

# Average sentence length
def average_sentence_length(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    average = len(words) / len(sentences)
    return average

df['AvgSent_NH'] = df['News_Headline'].apply(average_sentence_length)
df['AvgSent_Cont'] = df['clean_linkcontent'].apply(average_sentence_length)

# lexical count 

def count_lexical_categories(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Tag each token with its part of speech
    tagged_tokens = pos_tag(tokens)

    # Dictionary to store the counts
    counts = { 'CC': 0, 'CD': 0, 'DT': 0, 'EX': 0, 'FW': 0, 'IN': 0, 'JJ': 0, 'JJR': 0,
               'JJS': 0, 'LS': 0, 'MD': 0, 'NN': 0, 'NNS': 0, 'NNP': 0, 'NNPS': 0, 'PDT': 0, 
               'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 'RP': 0, 'TO': 0,
               'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 
               'WP': 0, 'WP$': 0, 'WRB': 0 }

    for word, tag in tagged_tokens:
        if tag in counts:
            counts[tag] += 1

    return pd.Series(counts)

# Apply the function to the 'Headline' column
df3 = df['News_Headline'].apply(count_lexical_categories)
df3.columns = ['NH_' + col for col in df3.columns]
df4 = df['Processed_Title'].apply(count_lexical_categories)
df4.columns = ['NH_process' + col for col in df4.columns]
# Apply the function to the 'Link_content' column
df5 = df['clean_linkcontent'].apply(count_lexical_categories)
df5.columns = ['Cont_' + col for col in df5.columns]
df6=df['Processed_content_link'].apply(count_lexical_categories)
df6.columns = ['Cont_process' + col for col in df6.columns]
# Concatenate the original DataFrame with the new DataFrames
df = pd.concat([df,df3,df4,df5,df6], axis=1)
df= pd.concat([df,TF_IDF_cont],axis=1)
df= pd.concat([df,TF_IDF],axis=1)
# group into headline and content


#drop content
columns_to_drop1 = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
               'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO',
               'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 
               'WP','WP$','WRB']
columns_to_drop2 = [col for col in df.columns if col.endswith('_Cont')]
columns_to_drop3 = [col for col in df.columns if col.startswith('Cont_')]
columns_to_drop4 = ['Sentiment_link_blob','Sentiment_link_flair','Sentiment_link_NLTK','Sentiment_link_blob_sub']
#drop headline
columns_to_drop5 = [col for col in df.columns if col.startswith('NH_')]
columns_to_drop6 = [col for col in df.columns if col.endswith('_NH')]
columns_to_drop7 = [col for col in df.columns if col.startswith('NH_process_')]
columns_to_drop8 = ['Sentiment_blob','Sentiment_NLTK','flair_sentiment_score','Sentiment_blo_sub']
#drop whole Sentimet
columns_to_drop9 = ['Sentiment_link_blob','Sentiment_link_flair','Sentiment_link_NLTK','Sentiment_link_blob_sub','Sentiment_blob','Sentiment_NLTK','flair_sentiment_score','Sentiment_blo_sub']
#drop Headline Sentimet
columns_to_drop8 = ['Sentiment_blob','Sentiment_NLTK','flair_sentiment_score','Sentiment_blo_sub']
#drop Content Sentimet
columns_to_drop4 = ['Sentiment_link_blob','Sentiment_link_flair','Sentiment_link_NLTK','Sentiment_link_blob_sub']
#drop whole Sentimet
#apply
df = df.drop(columns_to_drop9, axis=1)
df
#---- concat TF-IDF----------
df5.to_csv('Cleaned_Data_TF_IDF.csv', index = False)
df5.describe

# --------------transform fake label-------------------
def Fake_label(text):
    a = re.sub("TRUE", '6', text)
    b = re.sub("mostly-true", '5', a)
    c = re.sub("half-true", '4', b)
    d = re.sub("barely-true", '3', c)
    e = re.sub("FALSE", '2', d)
    f = re.sub("full-flop", '3', e)
    g = re.sub("no-flip", '3', f)
    h = re.sub("half-flip", '3', g)
    label = re.sub("pants-fire", '1', h)
    return label

df['Label'] = df['Label'].astype(str)
df['Label'] = df['Label'].apply(Fake_label)


#-------indentified all the entities----------
import spacy

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

social_media = ["Facebook","posts", "Bloggers", "Tweets"]

def ner(text):
    doc = nlp(text)
    entities = [ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
    entities += ["SOCIAL_MEDIA" for word in text.split() if word in social_media]
    return entities[0] if entities else 'PERSON'


df['Sorted_source_3'] = df['Source'].apply(ner)
df_grouped = df.groupby('Sorted_source_3').size().reset_index(name='Counts')
df_grouped = df_grouped.sort_values('Counts', ascending=False)
print(df_grouped)


# group in two
def Sorted_source_num(text):
    a = re.sub('PERSON', '1', text)
    b = re.sub('ORG', '2', a)
    c = re.sub('SOCIAL_MEDIA','3',b)
    return c

df['Sorted_source_3']=df['Sorted_source_3'].apply(Sorted_source_num)
df['Sorted_source_3']=df['Sorted_source_3'].astype(int)
df['Sorted_source_3']
#-------Prepare for machune learning-------------

def remove_useless(df):
    df.drop('News_Headline', inplace=True, axis=1)
    df.drop('Link_Of_News', inplace=True, axis=1)
    df.drop('Source', inplace=True, axis=1)
    df.drop('Stated_On', inplace=True, axis=1)
    df.drop('flair_sentiment', inplace=True, axis=1)
    df.drop('Date', inplace=True, axis=1)
    df.drop('New_Link', inplace=True, axis=1)
    df.drop('Processed_Title', inplace=True, axis=1)
    df.drop('clean_linkcontent', inplace=True, axis=1)
    df.drop('Processed_content_link', inplace=True, axis=1)
    df.drop('Sentiment_Flair', inplace=True, axis=1)
    df.drop('Clean_Headline', inplace=True, axis=1)
    df.drop('Link_content', inplace=True, axis=1)
    df.drop('Label_normal', inplace=True, axis=1)
    df.drop('Sorted_source', inplace=True, axis=1)
    return df

df=remove_useless(df)
df['Label']
df
df5['cosin-score']=list(cosine_sim)
df5['cosin-score'] = df5['cosin-score'].apply(np.mean)

df.to_csv('Cleaned_Data_Spell.csv', index = False)


#------------------Read machine learning dataset---------
df = pd.read_csv('Cleaned_Data_ML.csv', encoding='latin1')
#------------------ANOVA test----------------------

import scipy.stats as stats

group_a = df['Sentiment_blob']
group_b = df["Sentiment_NLTK"]
group_c = df["flair_sentiment_score"] 
group_d = df['Sentiment_link_blob']
group_e = df["Sentiment_link_NLTK"]
group_f = df["Sentiment_link_flair"] 
group_g = df['Sentiment_blo_sub'] 
group_h = df["Sentiment_link_blob_sub"] 

f_statistic_blob, p_value_blob = stats.f_oneway(group_a, group_d)
f_statistic_VADER, p_value_VADER = stats.f_oneway(group_b, group_e)
f_statistic_flair, p_value_flair = stats.f_oneway(group_c, group_f)
f_statistic_blob_sub, p_value_blob_sub = stats.f_oneway(group_g, group_h)
print("F Statistic:", f_statistic_blob)
print("P Value:", p_value_blob)
                    
print("F Statistic:", f_statistic_VADER)
print("P Value:", p_value_VADER)

print("F Statistic:", f_statistic_flair)
print("P Value:", p_value_flair)

print("F Statistic:", f_statistic_blob_sub)
print("P Value:", p_value_blob_sub)
df['Sentiment_blob'].mean()   
df['Sentiment_link_blob'].mean()  
df["Sentiment_NLTK"].mean()   
df["Sentiment_link_NLTK"].mean()   
df["flair_sentiment_score"].mean()    
df["Sentiment_link_flair"].mean()   
df['Sentiment_blo_sub'].mean()   
df['Sentiment_link_blob_sub'].mean()      

#------------------sort data--------------

df = pd.concat([df,TF_IDF],axis=1)
df = pd.concat([df,TF_IDF_cont],axis=1)  

def Fake_label_binary(text):
    a = re.sub("0", '0', text)
    b = re.sub("1", '0', a)
    c = re.sub("2", '1', b)
    d = re.sub("3", '2', c)
    e = re.sub("4", '3', d)
    f = re.sub("5", '4', e)
    g = re.sub("6", '5', f)
    return g

df['Label'] = df['Label'].astype(str)
df['Label'] = df['Label'].astype(int)
df['Label'] = df['Label'].apply(Fake_label_binary) 

df['Label']
df
#----------Test Correlation-----------------------------

# correlation
import seaborn as sns
import matplotlib.pyplot as plt
Plot_Corr_sentiment = df[['Sentiment_blob', 'Sentiment_NLTK', 'flair_sentiment_score',
                          'Sentiment_link_blob','Sentiment_link_NLTK','Sentiment_link_flair',
                          'Label','Sentiment_blo_sub','Sentiment_link_blob_sub']]
group_a = df['Sentiment_blob']
group_b = df["Sentiment_NLTK"]
group_c = df["flair_sentiment_score"] 
group_d = df['Sentiment_link_blob']
group_e = df["Sentiment_link_NLTK"]
group_f = df["Sentiment_link_flair"] 
corr_matrix = Plot_Corr_sentiment.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.1)
plt.show()


plt.savefig('Corr.png') 


#normalisation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['Label_normal'] = scaler.fit_transform(df[['Label']])

#------------------Apply Random forest(RF) in random forest  using DF6----------------
pd.read_csv('Cleaned_Data_Link.csv', encoding='latin1')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
import matplotlib.pyplot as plt

# Load the iris dataset
# Assuming df is your DataFrame and it has a 'target' column for the label

# Separate the features from the target
X = df.drop('Label', axis=1)  
y = df['Label']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# establsih parmeter interval
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

forest2 = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(estimator = forest2, param_distributions=random_grid,n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)



#try best parameter
rf_random.fit(X_train,y_train)
rf_random.best_params_

# Using train model
forest3 = RandomForestClassifier(bootstrap=True,
                                 max_depth=90, 
                                 max_features='sqrt', 
                                 min_samples_leaf=4, 
                                 min_samples_split=2,
                                 n_estimators=800)

# using best parameter to build model
forest3 = forest3.fit(X_train, y_train)
# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=(10))

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = forest3.predict(X_test)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
0#--------- CV
scores = cross_val_score(clf, X, y, cv=5)
print('Scores from each iteration:', scores)
print('Average score:', scores.mean())
df
#----------------------------feature importance--------------------------------------
# Get feature importance
importances = forest3.feature_importances_

# Convert the importances into a DataFrame
feature_names = X.columns.tolist()
feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

len(importances)
len(feature_names)
# Sort by importance
feature_importances = feature_importances.sort_values("importance", ascending=False)

# Display
print(top_20_features)

# Get top 20
top_20_features = feature_importances.head(40)

# Plot
plt.figure(figsize=(10,8))
plt.barh(top_20_features['feature'], top_20_features['importance'], align='center', alpha=0.8)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
plt.show()
#------------Bayes Classification gassianNB------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


X = df.drop('Label', axis=1)  
y = df['Label']
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Gaussian Classifier
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = gnb.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

scores = cross_val_score(gnb, X, y, cv=5)
print('Scores from each iteration:', scores)
print('Average score:', scores.mean())
#-------- multi NB------
X = df['Processed_content_link']
y = df['Label_bi']
# Convert the text data into a matrix of token counts
count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['Processed_content_link'])

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(counts, df['Label'], test_size=0.3, random_state=1)

# Create a MultinomialNB object
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels on the test set
predictions = clf.predict(X_test)

# Print the accuracy score
print("Accuracy:", metrics.accuracy_score(y_test, predictions))



#---------------LSTM-----------------
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from gensim.models import KeyedVectors
import numpy as np
df

#-----------------clear-------------------------
def remove_useless(df):
    df.drop('News_Headline', inplace=True, axis=1)
    df.drop('Link_Of_News', inplace=True, axis=1)
    df.drop('Source', inplace=True, axis=1)
    df.drop('Stated_On', inplace=True, axis=1)
    df.drop('flair_sentiment', inplace=True, axis=1)
    df.drop('Date', inplace=True, axis=1)
    df.drop('New_Link', inplace=True, axis=1)
    df.drop('Processed_Title', inplace=True, axis=1)
    df.drop('clean_linkcontent', inplace=True, axis=1)
    df.drop('Sentiment_Flair', inplace=True, axis=1)
    df.drop('Clean_Headline', inplace=True, axis=1)
    df.drop('Link_content', inplace=True, axis=1)
    df.drop('Label_normal', inplace=True, axis=1)
    return df

df=remove_useless(df)

# Separate out the features and labels
texts = df['Processed_content_link']
features = df.drop(['Label', 'Processed_Title'], axis=1)
labels = df['Label']
# Load Word2Vec model. Note you can download Google's pretrained Word2Vec model
# from: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  

# Tokenize the text data
tokenized_texts = [text.split() for text in texts]


# Generate a sequence of word vectors for each text
text_sequences = [[word2vec_model.wv[word] for word in text] for text in tokenized_texts]

# Pad the sequences so they all have the same length
padded_sequences = pad_sequences(text_sequences, maxlen=500)

# Normalize the additional features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Define the inputs
sequence_input = Input(shape=(500, 100), dtype='float32')  # input for the sequences
features_input = Input(shape=(scaled_features.shape[1],), dtype='float32')  # input for the additional features

# Then when defining your model you use this embedding matrix as weights for the embedding layer
model = Sequential()

model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                    output_dim=word2vec_model.vector_size, 
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=50,
                    trainable=False))  # We do not want to update the learned word weights in this model

model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))

# Convert the labels to 0's and 1's
df['Label'] = df['Label'].map({'fake': 0, 'real': 1})
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Processed_content_link'], df['Label'], test_size=0.2)

# Initialize the Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Processed_content_link'])

# Use the tokenizer to convert the text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences so they're all the same length
X_train_seq_padded = pad_sequences(X_train_seq, 50)
X_test_seq_padded = pad_sequences(X_test_seq, 50)

# Create a weight matrix for words in training docs
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, word2vec_model.vector_size))

for word, i in tokenizer.word_index.items():
    if word in word2vec_model:
        embedding_matrix[i] = word2vec_model[word]

# Then when defining your model you use this embedding matrix as weights for the embedding layer
model = Sequential()

model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                    output_dim=word2vec_model.vector_size, 
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=50,
                    trainable=False))  # We do not want to update the learned word weights in this model

model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train_seq_padded, y_train, validation_data=(X_test_seq_padded, y_test), epochs=5)

# Step 4: Make Predictions

# The predict method gives us probabilities, we convert to binary predictions
predictions = model.predict(X_test)
predictions = [1 if p > 0.5 else 0 for p in predictions]


#------------------------LSTM multi feature-----------------------------
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l1_l2
from keras.optimizers import Adam

# Separate out the features and labels
texts = df['Processed_content_link']
features = df.drop(['Label', 'Processed_content_link'], axis=1)
labels = df['Label']

# Tokenize the text data
tokenized_texts = [text.split() for text in texts]

# Train a Word2Vec model on the tokenized texts
word2vec_model = Word2Vec(tokenized_texts, vector_size=300, min_count=1)

# Generate a sequence of word vectors for each text
text_sequences = [[word2vec_model.wv[word] for word in text] for text in tokenized_texts]

# Pad the sequences so they all have the same length
padded_sequences = pad_sequences(text_sequences, maxlen=500)


# Normalize the additional features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

#---Build Model-------
# Define the inputs
sequence_input = Input(shape=(500, 300), dtype='float32')  # input for the sequences
features_input = Input(shape=(scaled_features.shape[1],), dtype='float32')  # input for the additional features

# Define the LSTM layer for the sequences
lstm = LSTM(256)(sequence_input)

# Define a Dense layer for the additional features
dense_features = Dense(128, activation='relu')(features_input)

# Concatenate the outputs of the LSTM and Dense layers
concatenated = Concatenate()([lstm, dense_features])

# Add a dropout layer
dropped = Dropout(0.5)(concatenated)

# Define the output layer
output = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(dropped)


# Create the model
model = Model(inputs=[sequence_input, features_input], outputs=output)

#Step 6: Compile and Train the Model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

model.fit([padded_sequences, scaled_features], labels, epochs=5, batch_size=32)

# The predict method gives us probabilities, we convert to binary predictions
predictions = model.predict(X_test)
predictions = [1 if p > 0.5 else 0 for p in predictions]


#-------------------XG BOOST---------------------------
# import necessary libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# assuming df is your DataFrame and you're trying to predict 'target'
X = df.drop('Label', axis=1)
y = df['Label']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# create a XGBoost classifier
xg_cl = xgb.XGBClassifier(objective='binarmulity:logistic', n_estimators=10, seed=123)
param = {
    'objective': 'multi:softmax',  # for multiclass classification
    'num_class': 6,  # number of classes
    'learning_rate': 0.1,  # learning rate
    'max_depth': 6,  # maximum depth of a tree
    'min_child_weight': 1,  # minimum sum of instance weight needed in a child
    'gamma': 0.1,  # minimum loss reduction required to make a further partition on a leaf node of the tree
    'subsample': 0.8,  # fraction of observations to be randomly sampled for each tree
    'colsample_bytree': 0.8,  # fraction of columns to be randomly sampled for each tree
    'n_estimators': 100,
    'seed': 123# number of trees to build
}
xg_cl = xgb.XGBClassifier(**param)
# fit the classifier to the training data
xg_cl.fit(X_train, y_train)

# predict the labels of the test set
preds = xg_cl.predict(X_test)

# compute the accuracy of the predictions
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %f" % accuracy)


scores = cross_val_score(xg_cl, X, y, cv=5)
print('Scores from each iteration:', scores)
print('Average score:', scores.mean())
#----------------------------feature importance--------------------------------------
# Get feature importance
importances_xgboost = xg_cl.feature_importances_

# Convert the importances into a DataFrame
feature_names_xg = X.columns.tolist()
feature_importances_xg = pd.DataFrame({"feature": feature_names_xg, "importance": importances_xgboost})

# Sort by importance
feature_importances_xg = feature_importances_xg.sort_values("importance", ascending=False)

# Display
print(top_20_features)

# Get top 20
top_20_features_xg = feature_importances_xg.head(40)

# Plot
plt.figure(figsize=(10,8))
plt.barh(top_20_features_xg['feature'], top_20_features_xg['importance'], align='center', alpha=0.8)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
plt.show()
#-------------------CROSS VALID TEST----------------
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=1000, centers=2, n_features=20, random_state=1)

model=xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
model = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(model, X, y, cv=5)

print('Scores from each iteration:', scores)
print('Average score:', scores.mean())

#-------------------KNN algorithm---------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix

# load iris dataset as an example
X = df.drop('Label', axis=1)  
y = df['Label']

# split the dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# feature scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# train the model with K=5
knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski')
knn_model.fit(X_train, y_train)

# make predictions on our test data
y_pred = knn_model.predict(X_test)

# evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

scores = cross_val_score(knn_model, X, y, cv=5)
print('Scores from each iteration:', scores)
print('Average score:', scores.mean())
#----------------SVM ALgorithm ------------------------
from sklearn import svm
from sklearn import metrics

# Assuming you have a DataFrame df with 'label' as the target column
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Target variable

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a svm Classifier
SVM_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale')  # Linear Kernel

# Train the model using the training sets
SVM_model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = SVM_model.predict(X_test)
# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

scores = cross_val_score(SVM_model, X, y, cv=5)
print('Scores from each iteration:', scores)
print('Average score:', scores.mean())