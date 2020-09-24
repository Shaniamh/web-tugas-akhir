import json

from bson.objectid import ObjectId
import pprint

class Emotion(object):
    #7 emotion
    def __init__(self,id=None, positivity=None, joy=None, fear=None, sadness=None, angry=None, surprise=None, disgust=None):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.positivity = positivity
        self.joy = joy
        self.fear = fear
        self.sadness = sadness
        self.angry = angry
        self.surprise = surprise
        self.disgust = disgust

    # def __str__(self):

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    # method ini untuk membuat 1 instance baru, paramnya dari data json, di pecah dibikinkan objek
    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return Annotation(json_data.get('_id', None),
                                 json_data['positivity'],
                                 json_data['joy'],
                                  json_data['fear'],
                                  json_data['sadness'],
                                  json_data['angry'],
                                  json_data['surprise'],
                                  json_data['disgust'],
                             )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")

class Annotation(object):
    def __init__(self,id=None, negAnno=None, posAnno=None):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.negAnno = negAnno
        self.posAnno = posAnno

    # def __str__(self):

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    # method ini untuk membuat 1 instance baru, paramnya dari data json, di pecah dibikinkan objek
    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return Annotation(json_data.get('_id', None),
                                 json_data['negAnno'],
                                 json_data['posAnno']
                             )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")

class Sentence(object):
    #8 Params
    def __init__(self,id=None, createdAt=None, text=None, tokenSize=None,
                 posSentimen=None, negSentimen=None, neuSentimen=None, uniSentimen=None, multiSentimen=None, feature=None):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.createdAt = createdAt          #1
        self.text = text                    #2
        self.tokenSize = tokenSize          #3
        self.posSentiment = posSentimen     #4
        self.negSentiment = negSentimen     #5
        self.neuSentiment = neuSentimen     #6
        self.uniSentiment = uniSentimen     #7
        self.multiSentiment = multiSentimen #8
        self.feature = feature              #9

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    # method ini untuk membuat 1 instance baru, paramnya dari data json, di pecah dibikinkan objek
    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return Sentence(json_data.get('_id', None),
                                 json_data['createdAt'],
                                 json_data['text'],
                                 json_data['tokenSize'],
                                 json_data['posSentiment'],
                                 json_data['negSentiment'],
                                 json_data['neuSentiment'],
                                 json_data['uniSentiment'],
                                 json_data['multiSentiment'],
                                 json_data['feature']
                             )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")


class Feature(object):
    """A class for storing Project related information"""
    # constructor isi 2 parameter
    def __init__(self, id=None, cleanTweet=None, selfScore=None, negSentiScore=None,
                 negEmo=None, absolutistScore=None, MDscore=None, label=None):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.cleanTweet = cleanTweet
        self.selfScore = selfScore
        self.negSentiScore = negSentiScore
        self.negEmo = negEmo
        self.absolutistScore = absolutistScore
        self.MDscore = MDscore
        self.label = label


    #konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    # method ini untuk membuat 1 instance baru, paramnya dari data json, di pecah dibikinkan objek
    @staticmethod
    def build_from_json(json_data):
        """ Method used to build WordList objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return Feature(json_data.get('_id', None),
                               json_data['cleanTweet'],
                                 json_data['selfScore'],
                                 json_data['negSentiScore'],
                                 json_data['negEmo'],
                                 json_data['absolutistScore'],
                                 json_data['MDscore'],
                                 json_data['label']
                             )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")


class Result(object):
    def __init__(self, id=None, NA=None, low=None, moderate=None, high=None):
        if id is None:
            self._id = ObjectId()
            self.NA = NA
            self.low = low
            self.moderate = moderate
            self.high = high
        else:
            self._id = id
            self.NA = NA
            self.low = low
            self.moderate = moderate
            self.high = high


    #konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    # method ini untuk membuat 1 instance baru, paramnya dari data json, di pecah dibikinkan objek
    @staticmethod
    def build_from_json(json_data):
        """ Method used to build WordList objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return Result(json_data.get('_id', None),
                               json_data['NA'],
                               json_data['low'],
                               json_data['moderate'],
                               json_data['high']
                             )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")




class POSTag(object):

    #8 params
    def __init__(self, id=None, aux=[], conj=[], adv=[], impronoun=[], perpronoun=[],
                 prep=[], func = [], negation = [], filler =[]):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.aux = aux
        self.conj = conj
        self.adv = adv
        self.impronoun = impronoun
        self.perpronoun = perpronoun
        self.prep = prep
        self.func = func
        self.negation = negation
        self.filler = filler

        if aux!=None: self.countAux = len(aux)
        else : self.countAux = 0
        if conj: self.countConj = len(conj)
        else: self.countConj = 0
        if adv: self.countAdv = len(adv)
        else : self.countAdv = 0
        if impronoun: self.countImpronoun = len(impronoun)
        else : self.counImpronoun = 0
        if perpronoun: self.countPerpronoun = len(perpronoun)
        else : self.countPerpronoun = 0
        if prep: self.countPrep= len(prep)
        else : self.countPrep = 0
        if func!=None: self.countFunc= len(func)
        else : self.countFunc = 0
        if negation: self.countNegation= len(negation)
        else : self.countNegation = 0
        if filler: self.countFiller= len(filler)
        else : self.countFiller = 0

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    # method ini untuk membuat 1 instance baru, paramnya dari data json, di pecah dibikinkan objek
    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        # json_data = json.load(json_data)
        if json_data is not None:
            if 'aux' in json_data:
                try:
                    return POSTag(json_data.get('_id', None),
                                 json_data['aux'],
                                 json_data['conj'],
                                 json_data['adv'],
                                 json_data['impronoun'],
                                 json_data['perpronoun'],
                                 json_data['prep'],
                                 json_data['func'],
                                 json_data['negation'],
                                json_data['filler'],
                              json_data['countAux'],
                              json_data['countConj'],
                              json_data['countAdv'],
                              json_data['countImpronoun'],
                              json_data['countPerpronoun'],
                              json_data['countPrep'],
                              json_data['countFunc'],
                              json_data['countNegation'],
                              json_data['countFiller'],
                             )
                except KeyError as e:
                    raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")

class TweetRow(object):
    #3 Params
    def __init__(self, id=None, sentence=None):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        # self.emotion = emotion
        self.sentence = sentence

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return TweetRow(json_data.get('_id', None),
                              # json_data['emotion'],
                              json_data['sentence']
                              )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")

class UserTweet(object):
    def __init__(self, id=None, username=None, result=None, tweets=None):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.username = username
        self.result = result
        self.tweets = []

        # for tweet in tweets:
        #     self.inputTweet(tweet)


    def inputTweet(self, tweet):
        self.tweets.append(tweet)

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return UserTweet(json_data.get('_id', None),
                                 json_data['username'],
                                 json_data['result'],
                                 json_data['tweets']

                              )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")

class TweetRow4Param(object):
    #3 Params
    def __init__(self, id=None, emotion=None, sentence=None, POSTag=None, annotation=None ):
        if id is None:
            self._id = ObjectId()
        else:
            self._id = id
        self.emotion = emotion
        self.sentence= sentence
        self.POSTag = POSTag
        self.annotation = annotation

    # konversi project itself into JSON form
    def get_as_json(self):
        """ Method returns the JSON representation of the Project object, yang mau disave di mongoDB"""
        return self.__dict__

    @staticmethod
    def build_from_json(json_data):
        """ Method used to build Project objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return POSTag(json_data.get('_id', None),
                              json_data['emotion'],
                              json_data['sentence'],
                              json_data['POSTag'],
                              json_data['annotation']
                              )
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e))
        else:
            raise Exception("Tidak ada data untuk dijadikan Data Entry")


def printTry():
    pprint.pprint(sentence1.__dict__)
    pprint.pprint(POSTag1.__dict__)
    pprint.pprint(annotation1.__dict__)

if __name__ == '__main__':

    sentence1 = Sentence(None, '2018-10-10 20:10', "also the irs functions to audit 10 poor people for every millionaire thanks to ronald reagan so it's a money pit me https://t.co/edpyx3jllw",
                         24, 0.112, 0.119, 0.769, "neutral", ['neutral', 'negative'])
    POSTag1 = POSTag(None, ['to', 'to'],[],['also', 'so'],[],['it', 'me'], ['for', 'to', 'so'],
                     ['the', 'to', 'to', 'for', 'every', 'to', 'to', 'so', 'it', 'a', 'me'], [])
    annotation1 = Annotation(None, ['poor'],['thanks'])
    annotation2 = Annotation(None, ['bad'], ['love'])
    # printTry()

    rowTweet1= TweetRow(None, sentence1.__dict__, POSTag1.__dict__, annotation1.__dict__)
    rowTweet2= TweetRow(None, sentence1.__dict__, POSTag1.__dict__, annotation2.__dict__)
    # pprint.pprint(rowTweet1.__dict__)

    rowTweets = []
    rowTweets.append(rowTweet1.__dict__)
    rowTweets.append(rowTweet2.__dict__)
    # pprint.pprint(rowTweets)

    #OBJECT untuk disave di mongodb
    # userDepTweets1 = UserTweet(None, 'hchazan', rowTweets)
    # pprint.pprint(userDepTweets1.__dict__)



