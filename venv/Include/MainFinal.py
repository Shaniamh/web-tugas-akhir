import glob
import pandas as pd
import spacy
import nltk
import pprint
import re

from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from spacy.tokens import Doc
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag

from collections import defaultdict


from Model.LangStyle import Sentence
from Model.LangStyle import TweetRow
from Model.LangStyle import UserTweet
from Model.LangStyle import Feature
from Model.TrialLabel import TrialLabel as TrialLabel
from Model.LangStyle import Result as Result
from Model.ResultPerUser import Result as ResultUser
from Model.ResultPerUser import UserTweet as UsernameTweet
from DBRepository.UserTweetRepository import UserTweetRepository
from DBRepository.WordList_sentic_Repository import WordList_sentic_Repository
from DBRepository.WordList_depression_Repository import WordList_depressionRepository as wordlist_dep_repo
from DBRepository.TrialLabel_Repository import TrialLabel_Repository as TrialLabelRepo
from DBRepository.DataTrainPerUserRepository import DataTrainPerUserRepository as DataTrainPerUser
from DBRepository.ResultPerUserRepository import ResultPerUser as ResultPerUser
from Model.WordList_sentic import WordList_sentic
from senticnet.senticnet import SenticNet
from Model.User import User

import time
start_time = time.time()

tag_map = defaultdict(lambda : wordnet.NOUN)
tag_map['J'] = wordnet.ADJ
tag_map['V'] = wordnet.VERB
tag_map['R'] = wordnet.ADV
wn_lemmater = WordNetLemmatizer()
pStemmer = PorterStemmer()
sent_analyzer = SentimentIntensityAnalyzer()
wpt = WordPunctTokenizer()
nlp = spacy.load('en_core_web_sm')
stop_words =set(stopwords.words('english'))


negation_words = ['no', 'not', 'mustn', "wouldn't", "aren't", "hasn't", 'wasn', 'don',
                      "isn't", 'won', "won't", "didn't", "couldn't", "weren't", 'nor', 'neither', "'t"]
self_words = ['my', 'myself', 'my%20self','i', "i'", 'am', 'me', 'id', "i'd", "'d", "ain", "ain't",
                  "i'll", 'im', "i'm", "ive","i've",
                  "mine", "own", 'myselves', 'ourselves', "'ve"]

#membaca file csv dari direktori
def readFolderCSV(dir):
    csvDir = glob.glob(dir + '/*.csv');
    filenamecsv = []
    i = 0;
    for file in csvDir:
        i = i + 1
        nameOfFile = file
        space = nameOfFile.find(" ")
        file = nameOfFile[54:]
        file = nameOfFile[61:]
        num = nameOfFile[54:space]
        nameCSV = nameOfFile[space + 1:]
        name = nameCSV[:-11]
        # print("{} | {} || : {} | {} | {}" .format(i, file, num, nameCSV, name))
        filenamecsv.append(nameOfFile)
    print("jumlah CSV : {}".format(i))
    # 2214 records
    return filenamecsv


#mencari username pada file csv
def findUname(file):
    nameOfFile = file
    space = nameOfFile.find(" ")
    num = nameOfFile[0:space]
    username = nameOfFile[space + 1:-11]
    return username

regex_str = [
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
def tokenize(s):
    return tokens_re.findall(s)
def preprocess(param, lowercase=False):
    tokens = tokenize(param)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens

def getNumberRemoval(rowText):
    # PREPROCESS - NUMBER REMOVAL
    rowText = re.sub(r'\d+', '', rowText)
    # print("Number removal: {}".format(text))
    return rowText


def getMentionLinkHashtagRemoval(rowText):
    # PREPROCESS - MENTION REMOVAL , LINK, HASHTAG sign REMOVAL
    rowText = re.sub(r'@\w+ ?|http\S+|#', '', rowText)
    # print("Mention, Link, hashtag sign removal: {}".format(text))
    return rowText


def getNTConversion(rowText):
    # PREPROCESS - n't conversion
    # text  = re.sub('n''t+$', " not", text)
    rowText = re.sub("n't\s*|don$", " not ", rowText)
    # print(" n't conversion: {}".format(text))
    return rowText


def getFivePreprocess(rowText):
    rowText = getNumberRemoval(rowText)  # PREPROCESS - NUMBER REMOVAL
    # PREPROCESS - PUNCTUATION REMOVAL (have done at prev preprocess)
    # text = text.translate(string ("", ""), string.punctuation)
    rowText = getMentionLinkHashtagRemoval(rowText)  # PREPROCESS - MENTION REMOVAL , LINK, HASHTAG sign REMOVAL
    rowText = getNTConversion(rowText)  # PREPROCESS - n't conversion
    # PREPROCESS - OVERWRITE (data dari DB sudah recognize by wordnet and corrected by textblob)
    # text = ''.join(''.join(s)[:] for _, s in itertools.groupby(text))
    return rowText

def updateStopWords(stop_words):
    to_extend = ['x', 'y', 'r', 'e', 's', 'm', 'hi', 'yet', 'may', 'oh', 'due', 'to',
                 'day', 'days', 'weeks', 'week',
                 'being', 'months', 'way', ]
    stop_words = stop_words.union(to_extend)

    # print(stop_words)
    to_remove = ['instead']
    stop_words = stop_words.difference(to_remove)
    stop_words = stop_words.difference(self_words)
    stop_words = stop_words.difference(negation_words)
    # print(stop_words)
    return  stop_words

def sentiment_scores(doc):
    return sent_analyzer.polarity_scores(doc.text)


Doc.set_extension("sentimenter", getter=sentiment_scores)

def getSentence(date, rowText):
    # membentuk objek sentence berisi info csv dan sentimen
    # =================Sentence sentimen
    # 1. sentence info from csv
    doc = nlp(rowText)
    created_at = date
    text = rowText

    # 2. tokensize
    tokenSize = len(doc)

    # 3. sentimen with spacy
    negSentimen = doc._.sentimenter['neg']
    posSentimen = doc._.sentimenter['pos']
    neuSentimen = doc._.sentimenter['neu']

    uniSentimen = ''
    multiSentimen = []
    if (neuSentimen > (negSentimen + posSentimen)):  # jika skor netral lebih banyak dari kecederungan
        multiSentimen.append('neutral')
        uniSentimen = 'neutral'
    elif negSentimen > posSentimen:  # jika skor kecenderungan lebih besar, dan cenderung negative
        multiSentimen.append('negative')
        uniSentimen = 'negative'
    elif posSentimen > negSentimen:  # jika skor kecenderungan lebih besar, dan cenderung positif
        multiSentimen.append('positive')
        uniSentimen = 'positive'
    else:  # else jika skor kecenderungan lebih besar, tapi skor kecenderungan + - sama
        uniSentimen = 'neutral'
    if negSentimen > posSentimen:  # jika skor kecenderungan lebih besar, dan cenderung negative
        multiSentimen.append('negative')
    if posSentimen > negSentimen:  # jika skor kecenderungan lebih besar, dan cenderung positif
        multiSentimen.append('positive')

    # 4. return objek sentence
    sentenceIns = Sentence(None, created_at, text, tokenSize,
                           posSentimen, negSentimen, neuSentimen, uniSentimen, multiSentimen)
   # print("2. sentence result : {}".format(sentenceIns.__dict__))
    return sentenceIns


def initArrayNol(length):
    arrayNol = []
    for i in range(length):
        arrayNol.append(0)

    print("panjang array : {}".format(len(arrayNol)))
    return arrayNol

def sendArrayOfTrialLabelToDB(arrayTrialLabel):
    trialLabelRepo = TrialLabelRepo()
    counter = 0
    for trialLabel in arrayTrialLabel:
        trialLabelRepo.create(trialLabel)
        counter = counter + 1
        print("berhasil masukkan ke db trial {} kata".format(counter))


if __name__ == '__main__':

    #TEMPAT DEKLARASI

    sumLemmaToken = 0
    filtered_sentence = []
    ALL_filtered_sentence = []
    sumToken = 0
    sumHasil = 0
    NegativityScore = 0
    temNegScore = 0
    arrayCleanTweet = []

    wordList_depRepo = wordlist_dep_repo()
    sumMDScore = 0
    arraySelfScore = []
    arraySentiScore = []
    arrayMDScore = []
    arrayMDScore2 = []
    arrayNegativityScore = []
    arrayNegativityScore2 = []
    arrayAbsolutistScore = []
    arrayText = []
    sum = 0
    nAScore = 0
    lowScore = 0
    moderateScore = 0
    highScore = 0
    arrayResult = []
    arrayObjFeature = []
    arrayTweet = []
    arrayTweetRowIns = []  # untuk menyimpan sejumlah tweet dalam 1 CSV user
    arrayTweetRowIns2 = []
    arrayTweet2 = []
    arrayObjTriallabel =[]
    arrayObjResult = []

    #Memanggil fungsi updateStopWords
    stop_words = updateStopWords(stop_words)

    #Senticnet membaca wordlist
    wordSenticRepo = WordList_sentic_Repository()
    sn = SenticNet()
    existWordlistDepression = wordList_depRepo.read()
    wordSenticListDB = wordSenticRepo.read()

    wordSenticList = set()
    for words in wordSenticListDB:
        wordSenticList.add(words['word'])


    dir = "C:\\userDepClean\\testing"
    # read file-file csv di dalam 1 folder untuk bisa diakses
    fileNameCSVs = []
    fileNameCSVs = readFolderCSV(dir)
    counterDepUser = 0

    for file in fileNameCSVs:
        print(file)
        # prepare for saving DB
        repoUserTweet = DataTrainPerUser()
        repoResultPerUser = ResultPerUser()

        arrayTweet = []
        username = findUname(file)
        print("username : {}".format(username))

        df = pd.read_csv(file, encoding='latin-1')
        print(df.shape[0])
        arrayTweetRowIns = []  # untuk menyimpan sejumlah tweet dalam 1 CSV user
        count = 159
        for count in range(df.shape[0]):
            print("count : {}".format(count))
            rowText = str(df['text'][count])
            timeRowText = df['created_at'][count]
            # 1. koreksi
            blo = TextBlob(rowText)
            corrected = blo.correct()
            #membaca text tweet
            rowText = str(corrected)
           # print("text belum bersih: {}".format(rowText))
            arrayTweet.append(rowText)                      #array tweet belum bersih
            #2nd - SENTIMENT CHECK
            sentiScore = getSentence(timeRowText,rowText).negSentiment  #mengambil nilai negSentiment
            arraySentiScore.append(sentiScore)
           # print("sentiment negatif : {}".format(sentiScore))

            #preprocessing NUMBER, LINK, MENTION, # sign (removal)
            text = getFivePreprocess(rowText)
           # print("text sudah bersih: {}".format(text))

            # PREPROCESS - lemmatization by wordnet NLTK
            wn_lemmater = WordNetLemmatizer()
            # text = "@coralineada i spoke with a number of developer from underrepresented groups and almost all of them said they"
            tokens = word_tokenize(text)
            token_2 = []
            lemma_2 = []
            for token, tag in pos_tag(tokens):
                lemma = wn_lemmater.lemmatize(token, tag_map[tag[0]])
                # print("heyo")
                # print(tag_map[tag[0]])
                token_2.append(token)
                lemma_2.append(lemma)
            sumLemmaToken = sumLemmaToken + len(tokens)
            # print("hasil token      : {}".format(token_2))
            # print("hasil lemma     {} : {}".format(len(lemma_2),lemma_2))
            text = ' '.join(lemma_2)
           # print("text lemma : {}".format(text))

            # PREPROCESS - #LOWERCASE ALL
            # TOKEN NEEDED TYPE (word)
            word_tokens = cleanByRegex = preprocess(text, True)
            # print("clean by regex : {}".format(cleanByRegex))
            # word_tokens = lemma_2   #lemma 2 - digunakan jika tidak menggunakan tahap preprocess by regEx

            filtered_sentence = []
            for w in word_tokens:
                # PREPROCESS - STOPWORD REMOVAL
                if w not in stop_words:
                    filtered_sentence.append(w)  # untuk tampilan per tweet
                    ALL_filtered_sentence.append(w)  # untuk mencari kata dari seluruh data training

            # ALL_filtered_sentence.append(filtered_sentence)
            if (len(lemma_2) != len(word_tokens)):
                print("FIND ME - membuktikan proses cleanbyregex telah mengurangi kata yang tidak jelas(noise)")
           # print("{} token : {}".format(len(word_tokens), word_tokens))
           # print("{} hasil : {} ".format(len(filtered_sentence),filtered_sentence))
            sumToken = sumToken + len(word_tokens)
            sumHasil = sumHasil + len(filtered_sentence)

            # 1st - SELF REFERENCES - check
            selfScore = False
            for word in filtered_sentence:
                if word in self_words:
                    selfScore = True;
                    break
            arraySelfScore.append(selfScore)

            # 5 - ABSOLUTIST - check
            absolutistScore = False
            absoluteWord = ['entirely', 'full', 'fully', 'absolutely',
                            'all', 'always', 'complete', 'completely', 'constant',
                            'constantly', 'definitely', 'entire', 'ever', 'every',
                            'everyone', 'everything', 'totally', 'whole', 'suicide',
                            'suicidal', 'total', 'tired', 'fault', 'alone', 'sadness',
                            'sad', 'death', 'die', 'upset', 'angry', 'stressed', 'pain',
                            'loneliness', 'painful', 'worst', 'depressed', 'depression',
                            'sick', 'lonely', 'kill']
            for word in filtered_sentence:
                if word in absoluteWord:
                    absolutistScore = True;
                    break
            MDScore = 0
            NegativityScore = 0
            temNegScore = 0

            print("filtered sentence (clean tweet): {}".format(filtered_sentence))
            arrayCleanTweet.append(filtered_sentence)

            for word in filtered_sentence:
                print("word : {}".format(word))
                # if len(word) != 0:
                if word in wordSenticList:
                    findWord = wordSenticRepo.searchWord(word)
                    MDScore = MDScore + 1
                    print("mdscore non abs : {}".format(MDScore))
                    if word in absoluteWord:
                        MDScore = MDScore * 2
                        print("mdscore with abs : {}".format(MDScore))

                    for fw in findWord:
                        sentics = fw['senticnet']['sentics']
                        print("kata : {}, nilai senticsnya : {}".format(word, sentics))
                        if (float(sentics['sensitivity']) < 0):
                            print("sentics negativity : {}".format(abs(float(sentics['sensitivity']))))
                            temNegScore = temNegScore + (-1 * abs(float(sentics['sensitivity'])))
                        elif (float(sentics['pleasantness']) < 0):
                            print("sentics pleasantness : {}".format(sentics['pleasantness']))
                            temNegScore = temNegScore + float(sentics['pleasantness'])
                        elif (float(sentics['aptitude']) < 0):
                            print("sentics aptitude : {}".format(sentics['aptitude']))
                            temNegScore = temNegScore + float(sentics['aptitude'])
                        else:
                            temNegScore = 0
                # else:
                #     MDScore = 0
                #     temNegScore = 0

            arrayAbsolutistScore.append(absolutistScore)
            print("temNegScore : {}".format(temNegScore))
            print("MD SCORE : {}".format(MDScore))
            arrayNegativityScore.append(temNegScore)
            arrayMDScore.append(MDScore)  # array untuk menyimpan semua MD score dari tweet2 user
            print("negativity : {}".format(arrayNegativityScore))
            print("ARRAY MDSCORE : {}".format(arrayMDScore))
            print("len array md score : {}".format(len(arrayMDScore)))
            if (MDScore != 0):
                arrayMDScore2.append(MDScore)
            if (NegativityScore != 0):
                arrayNegativityScore2.append(NegativityScore)
            sumMDScore = sumMDScore + MDScore
            print("sumMDScore : {}".format(sumMDScore))

    print("============= Outlier removal=============")

    arrayMDScoreSorted = arrayMDScore2;
    arrayMDScoreSorted.sort()

    print(arrayMDScoreSorted)
    # arrayMDScoreSorted arrayNegativityScore2

    lenMDscore = len(arrayMDScoreSorted)
    print("len : {}".format(lenMDscore))
    print("Q1 len array of MD Score : {} ".format(arrayMDScoreSorted[int((lenMDscore + 1) * 1 / 4)]))
    print("Q2 len array of MD Score : {} ".format(arrayMDScoreSorted[int((lenMDscore + 1) / 2)]))
    print("Q3 len array of MD Score : {} ".format(arrayMDScoreSorted[int((lenMDscore + 1) * 3 / 4)]))

    q1 = arrayMDScoreSorted[int((lenMDscore + 1) * 1 / 4)]
    q2 = arrayMDScoreSorted[int((lenMDscore + 1) / 2)]
    q3 = arrayMDScoreSorted[int((lenMDscore + 1) * 3 / 4)]

    interquartile = q3 - q1
    batasOutlierBawah = q1 - (1.5 * interquartile)
    batasOutlierAtas = q3 + (1.5 * interquartile)

    print(" Batas outlier bawah : {}".format(batasOutlierBawah))
    print(" Batas outlier atas : {}".format(batasOutlierAtas))

    if (batasOutlierBawah < arrayMDScoreSorted[0]):
        batasBawah = arrayMDScoreSorted[0]
    else:
        batasBawah = batasOutlierBawah
    if (batasOutlierAtas > arrayMDScoreSorted[len(arrayMDScoreSorted) - 1]):
        batasAtas = arrayMDScoreSorted[len(arrayMDScoreSorted) - 1]
    else:
        batasAtas = batasOutlierAtas

    #print("RULE penggolongan ===============================================")

    maxRangeMD = batasAtas - batasBawah
    arrayLabel = initArrayNol(len(arraySelfScore))  # akan diisi {NA,least, moderate, most}
    level = ["NA", "low", "moderate", "high"]
    abc = 'tweets'
    # negEmo dg MDcek dulu
    for i in range(len(arrayMDScore)):
        # versi dengan SELF REF
        if arraySelfScore[i] != True:
            # versi tanpa self ref
            # if arraySelfScore[i] == None:
            arrayLabel[i] = level[0]
            nAScore +=1
        #  print(arrayLabel[i])
        else:
            if arraySentiScore[i] > 0:  # kalimat bersentimen Negatif
                if arrayNegativityScore[i] != 0:  # tingkat negatifnya ADA
                    if arrayAbsolutistScore[i] != True:
                        if arrayMDScore[i] < (maxRangeMD / 3):
                            arrayLabel[i] = level[2]  # MODERATE 1
                            moderateScore +=1
                        else:
                            arrayLabel[i] = level[3]  # HIGH 2
                            highScore +=1
                    else:
                        if arrayMDScore[i] < (maxRangeMD / 3*2):
                            arrayLabel[i] = level[2]  # MODERATE 1
                            moderateScore +=1
                        else:
                            arrayLabel[i] = level[3]  # HIGH 2
                            highScore +=1
                else:  # tingkat sadnessnya tidak ada
                    # print("kalimatnya tidak mengandung sadness")
                    arrayLabel[i] = level[1]  # low
                    lowScore +=1
            else:  # kalimat bersentimen positif
                # print("kalimat nya positif")
                if arrayNegativityScore[i] != 0:  # tingkat sadnessnya ada
                    if arrayAbsolutistScore[i] != True:
                        if arrayMDScore[i] < (maxRangeMD / 3):
                            arrayLabel[i] = level[1]  # LOW 6
                            lowScore += 1
                        else:
                            arrayLabel[i] = level[2]  # MODERATE 7
                            moderateScore += 1
                    else:
                        if arrayMDScore[i] < (maxRangeMD / 3*2):
                            arrayLabel[i] = level[1]  # LOW 6
                            lowScore += 1
                        else:
                            arrayLabel[i] = level[2]  # MODERATE 7
                            moderateScore += 1
                else:  # tingkat sadness nya tidak ada
                    #  print("kalimatnya tidak mengandung sadness")
                    arrayLabel[i] = level[0]  # na
                    nAScore +=1

    print("================================================================")

    print("Array Self Score : {}".format(arraySelfScore))
    print("Array Senti Score : {}".format(arraySentiScore))
    print("Array AbsolutistScore : {}".format(arrayAbsolutistScore))
    print("Array MDscore : {}".format(arrayMDScore))
    print("Array Negative Emotion Score : {}".format(arrayNegativityScore))
    print("Array Label : {}".format(arrayLabel))
    print("==========================FINAL RESULT===============================")
    print("USERNAME : {}".format(username))
    print("================================================================")
    print("NA Score : {}".format(nAScore))
    print("Low Score : {}".format(lowScore))
    print("Moderate Score : {}".format(moderateScore))
    print("High Score : {}".format(highScore))
    print("================================================================")


    # ================================MENYIMPAN KE DB===================================
    # print("===============================SAVING DATA TO DB RESULT PERUSER===============================")
    #
    # for file in fileNameCSVs:
    #     df = pd.read_csv(file, encoding='latin-1')
    #     print(df.shape[0])
    #     count = 159
    #     for count in range(df.shape[0]):
    #         print("count : {}".format(count))
    #         rowText = str(df['text'][count])
    #         timeRowText = df['created_at'][count]
    #         # 1. koreksi
    #         blo = TextBlob(rowText)
    #         corrected = blo.correct()
    #         # membaca text tweet
    #         rowText = str(corrected)
    #         objResult = ResultUser(None, nAScore, lowScore, moderateScore, highScore)
    #         # print("objResult : {}".format(objResult.__dict__))
    #         userTweetIns = UsernameTweet(None, username, objResult.__dict__)
    #
    #     repoResultPerUser.create(userTweetIns)  # insert ke db datatrainperuser
    #     # sendArrayOfTrialLabelToDB(arrayObjTriallabel)    #insert ke db trial label untuk mengambil labelnya

    print("===============================SAVING DATA TO DB DATATRAINPERUSER===============================")
    for file in fileNameCSVs:
        df = pd.read_csv(file, encoding='latin-1')
        print(df.shape[0])
        count = 159
        for count in range(df.shape[0]):
            print("count : {}".format(count))
            rowText = str(df['text'][count])
            timeRowText = df['created_at'][count]
            # 1. koreksi
            blo = TextBlob(rowText)
            corrected = blo.correct()
            # membaca text tweet
            rowText = str(corrected)
            objResult = Result(None, nAScore, lowScore, moderateScore, highScore)
            # print("objResult : {}".format(objResult.__dict__))
            userTweetIns = UserTweet(None, username, objResult.__dict__)
            objSentence = getSentence(timeRowText, rowText)  # make objek sentence
            objTweet = TweetRow(None,                          #bikin obj tweetrow sementara
                                objSentence.__dict__)
            arrayTweetRowIns.append(objTweet)
            for i in range (len(arrayTweetRowIns)):       #dilooping pada range tweet row
                # print("count feature : {}".format(i))
                objFeature = Feature(None, arrayCleanTweet[i], arraySelfScore[i], arraySentiScore[i], arrayNegativityScore[i],
                                 arrayAbsolutistScore[i], arrayMDScore[i], arrayLabel[i])
                tempObjTrial = TrialLabel(None, username, arrayTweet[i], arrayLabel[i])         #untuk mengambil tweet dan label

            arrayObjTriallabel.append(tempObjTrial)
           # print("arrayTrialLabel : {}".format(arrayObjTriallabel))

            objTweet2 = TweetRow(None,
                                 objSentence.__dict__, objFeature.__dict__)  # make object tweets final
            arrayTweetRowIns2.append(objTweet2)

            for userTweet in arrayTweetRowIns2:
                userTweetIns.inputTweet(userTweet.__dict__)


        repoUserTweet.create(userTweetIns)               #insert ke db datatrainperuser
        sendArrayOfTrialLabelToDB(arrayObjTriallabel)    #insert ke db trial label untuk mengambil labelnya

    print("--- FINISHED in %s seconds ---" % (time.time() - start_time))