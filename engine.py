import nltk
import math
import string
from pathlib import Path
from nltk.util import bigrams, trigrams, ngrams
from textblob import TextBlob
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.data.path.append('/var/www/nltk_data')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

unnecessary_words = ['hed', 'best', 'date', "what'll", 'self', 'first', 'home', 'nt', 'b', 'ask', 'yj', 'seriously', 'beginnings', "don't", 'xv', 'H', 'il', 'approximately', 'arise', 'isn', 'possible', '3d', 'becomes', 'section', 'seen', 'nowhere', 'always', 'ls', 'empty', 'ay', 'believe', 'fi', 'hopefully', 'needn', 'same', 'h', 'ob', "a's", 'relatively', 'av', 'au', 'all', 'resulting', 'od', 'mr', '3', 'ran', 'ix', 'hu', 'bill', 'hr', 'truly', 'ml', 'bl', "t's", 'c2', 'take', 'otherwise', 'might', 'hi', "i've", 'due', 'ever', 'provides', 'r2', 'wonder', 'vt', 'make', 'my', 'when', 'its', 'J', 'together', 'within', 'predominantly', 'ke', "let's", "it's", 'didn', 'auth', 'un', 'last', 'di', 'o', 'shes', 'else', "ain't", 'ones', 'other', 'yt', "they're", 'A', 'well-b', 'used', 'further', 'act', 'pn', 'de', "they'd", 'full', '24', 'changes', 'somebody', 'added', 'readily', 'pq', 'thus', 'etc', 'come', 'cq', "needn't", "they've", 't', 'each', 'pl', 'useful', 'normally', 's', 'top', 'wo', 'go', 'should', 'i2', 'nl', 'vj', 'cx', 'P', 'is', 'her', 'given', 'on', 'q', 'j', 'known', 'instead', 'next', 'end', 'h3', 'U', 'afterwards', 'ef', 'until', 'ours', 'hasnt', 'much', 'poorly', 'ex', 'everything', 'widely', 'le', 'e3', 'ap', 'ee', 'help', 'i8', 'since', 'being', "there'll", 'way', 'she', 'cf', 'i3', 'perhaps', 'tq', 'wheres', 'think', 'okay', 'er', 'abst', "that's", 'zz', 'z', 'shan', 'did', 'hid', 'will', 'ending', 'via', 'tip', 'dr', 'significantly', 'nj', 'ups', 'shows', 'considering', 'c1', 'if', 'various', 'beginning', '29', 'nd', 'hers', '12', 'V', 'pk', 'obtain', '6o', 'call', 'b2', 'xl', 'e2', 'wants', "we'll", 'p', 'sure', 'anyways', 's2', 'ns', 'than', 'tc', 'rf', 'know', 'million', 'hereafter', 'cannot', 'obtained', 'enough', 'them', 'B', 'tx', 'yours', 'ij', 'bu', 'gave', 'meanwhile', 'dk', 'lj', 'respectively', 'et', 'fify', 'described', 'br', 'vd', 'reasonably', 'were', 'probably', 'v', 'specifying', 'uj', 'bd', 'often', 'Q', 'po', 'quite', 'kj', 'edu', 'whod', 'inasmuch', 'took', 'regarding', 'certain', 'however', 'al', 'awfully', 'somewhat', 'qu', 'or', 'about', 'a', 'mg', 'usefully', 'makes', 'qv', 'move', 'hes', 'taken', 'cy', 'lt', 'rr', 'u201d', 'theres', 'ro', 'then', 'therefore', 'th', 'cause', 'example', 'fifth', 'x', 'n', 'toward', 'rj', 'entirely', 'yet', 'rm', 'thereto', 'ab', 'wherein', 'maybe', 'mine', 'pu', 'f', 'gy', 'oj', 'really', 'side', 'xx', 'cj', 'none', 'haven', 'index', 'somehow', 'Y', 'because', 'promptly', 'dj', 'est', "i'm", 'myself', "won't", 'aren', 'amongst', 'except', 'having', 'tries', 'themselves', 'dd', 'keeps', 'gets', 'who', 'n2', 'anyhow', 'I', 'would', 'obviously', 'begin', 'lets', 'non', 'describe', 'se', 'os', 'specified', 'ra', 'fa', 'ot', 'nor', 'merely', 'tf', 'que', 'io', 'mustn', 'the', 'of', 'whereupon', 'km', 'slightly', 'rq', 'en', 'concerning', 'more', 'someone', 'ko', 'ut', '6b', 'want', 'affects', 'ignored', 'hadn', 'xt', 'ui', 'ol', 'nonetheless', 'seeming', 'zero', 'it', 'elsewhere', 'shown', 'lately', 'jr', 'done', "who's", 'e', 'successfully', 'throug', 'sl', 'M', 'results', 'never', 'pf', 'consider', 'pp', 'somewhere', 'himself', 'secondly', 'seems', 'apparently', 'in', 'what', 'exactly', 'anymore', 'had', 'thoughh', 'um', 'below', 'ae', 'at', 'immediate', 'unfortunately', 'regardless', 'whereby', 'previously', 'sc', 'beyond', 'www', 'wouldnt', 'to', 'sent', 'sec', "hadn't", 'sm', 'el', 'sorry', 'xk', 'x3', "they'll", 'no', '32', 'important', 'liked', 'you', 'related', 'over', 'wasnt', 'according', 'thin', 'm', "c's", "couldn't", 'xj', 'name', 'hasn', "mightn't", 'us', 'whomever', 'a3', 'use', 'latter', 'once', 'invention', "shouldn't", 'nay', 'te', 'especially', 'now', 'thousand', 'h2', 'everyone', 'while', 'xo', 'why', 'nr', 'ce', 't1', 'saying', 'along', 'not', 'inc', 'N', 'indicates', '7', 'apart', 'already', 'besides', 'importance', 'ur', 'anybody', 'plus', 'ej', 'moreover', 'indicate', 'js', '13', 'fn', 'ms', "that've", '22', 'showns', "wouldn't", 'ni', 'became', 'cg', 'miss', 'til', 'st', 'most', 'but', 'cry', 'needs', 'bottom', 'sup', 'either', 'u', 'two', 'ic', 'oc', 'stop', 'lc', 'mrs', 'b1', 'similar', 'therein', "shan't", 'og', 'c3', 'so', 'ou', 'been', "she'll", 'ti', 'again', 'wasn', 'strongly', 'comes', 'rn', 'specify', 'twelve', 'almost', 'for', 'between', 'thou', 'tell', 'insofar', "should've", 'throughout', 'appreciate', 'trying', 'ju', 'everybody', 'ax', 'run', 'ph', 'bx', 'primarily', '5', 'werent', 'inner', 'volumtype', 'lf', 'whats', 'therere', 'pe', 'ds', 'me', 'hello', 'vq', 'whenever', 'va', 'using', 'd2', "aren't", 'necessarily', 'may', 'against', 'ten', 'specifically', 'fo', 'that', 'course', "you're", 'sj', '26', 'kept', 'ln', 'thence', 'doesn', 'above', 'fl', 'id', 'rd', 'et-al', 'under', 'E', "mustn't", 'gotten', 'jt', "isn't", 'aj', '8', 'wherever', '3b', 'six', 'yes', 'yourself', 'through', 'without', '27', 'also', 'found', 'par', '0s', 'ox', 'doing', 'cc', 'indicated', 'tb', "you'll", 'ip', 'meantime', 'pm', 'vol', 'following', 'seeing', 'uo', 'whether', 'off', 'noone', "she's", 'largely', 'followed', 'try', 'anyone', 'ain', 'pd', 'oi', 'wont', "it'd", 'couldn', 'tt', 'nine', 'accordance', 'ny', 'unlike', 'research', 'herself', 'despite', 'tp', 'ought', '4', 'ru', 'vs', "you'd", 'ci', 'nc', 'their', 'please', 'was', 'aw', 'sz', 'outside', 'iz', 'wish', 'cp', 'i', 'lr', 'ng', '28', 'another', 'sometime', 'ie', 'willing', 'affecting', 'thoroughly', 'cv', 'value', 'thank', "he's", 'get', 'mn', 'F', 'ltd', 'his', 'seem', 'follows', 'detail', 'accordingly', 'y', 'which', 'something', 'a2', 'eu', 'dt', "i'll", 'onto', 'new', 'y2', 'latterly', 'near', 'fifteen', 'able', 'thereby', 'em', 'sixty', 'could', 'td', 'system', 'ss', 'contain', 'and', '25', 'going', 'wi', 'too', 'tl', 'before', 'rv', 'unlikely', 'p3', 'a4', 'cn', 'by', 'every', "he'll", 'theyd', 'tm', 'shall', 'heres', 'up', 'this', 'serious', 'hereupon', '21', 'ri', 'regards', 'weren', 'S', 'ow', 'O', 'sa', 'fire', 'one', 'tr', 'resulted', 'vo', 'rs', 'ao', "haven't", 'ii', "wasn't", 'sd', 'whereas', 'least', '18', "who'll", 'well', 'five', 'particularly', 'upon', 'ac', "doesn't", 'mt', 'arent', 'allow', 'tv', 'W', 'back', 'after', 'though', 'biol', 'twice', '9', 'own', 'says', 'out', 'like', 'xf', 'hundred', 'ey', 'only', 'xi', 'gives', 'gs', 'nevertheless', '11', 'a1', 'youre', 'omitted', '31', 'f2', "we'd", 'mean', 'con', 'made', 'tends', 'just', 'pages', 'ourselves', 'taking', 'actually', 'whither', 'welcome', "she'd", 've', 'thereof', 'ct', 'ea', 'third', 'com', 'does', 'iq', 'interest', 'rc', 'neither', 'pi', 'pj', 'from', 'showed', 'oz', 'x1', 'tried', 'sometimes', 'oh', 'certainly', 'oa', 'sincere', 'fy', 'pc', 'affected', 'kg', 'dc', 'bc', 'far', 'contains', 'hardly', 'wed', 'shouldn', 'where', 'si', 'here', 'there', 'allows', 'hj', 'noted', "hasn't", "i'd", 'i4', 'pr', "where's", 'ib', 'az', 'vols', 'others', 'even', 'downwards', 'present', 'ep', '14', 'former', 'how', 'itself', 'xs', 're', 'cs', 'whence', 'greetings', 'r', 'we', 'adj', 'fix', 'usually', 'lo', 'corresponding', 'still', 'eo', "we've", 'dp', 'thanks', 'substantially', 'are', 'ry', 'hence', 'likely', 'ad', 'unto', 't3', 'anywhere', 'furthermore', 'got', 'presumably', 'couldnt', "he'd", 'bt', 'somethan', 'bi', '10', 'ir', 'iy', 'qj', '1', 'rh', 'seven', 'overall', 'brief', 'associated', 'towards', 'cm', 'means', 'ff', 'need', 'cant', 'eleven', 'seemed', 'dx', 'any', 'find', 'ys', 'pt', 'knows', 'around', 'l2', 'later', 'better', 'beside', 'mo', 'page', 'necessary', 'nearly', 'less', 'potentially', 'jj', 'recently', 'he', 'thereafter', 'gl', '3a', 'past', 'cz', 'appear', '20', '16', "how's", 'py', 'have', 'nn', 'sf', 'oo', 'theirs', 'briefly', 'iv', '6', 'ec', 'owing', 'thorough', "it'll", 'whom', 'thru', 'yourselves', 'bj', 'refs', 'fu', 'w', 'appropriate', 'pagecount', 'anything', 'whereafter', 'won', 'ord', 'few', 'gj', 'cu', 'announce', 'they', 'bs', 'gr', 'look', 'ag', 'old', 'fill', '0o', 'sufficiently', 'fj', 'theyre', 'forty', 'hy', 'several', 'sy', 'yr', '19', 'eq', 'per', 'whole', 'novel', 'don', 'fr', 'everywhere', 'Z', 'mostly', 'inward', 'whatever', 'immediately', 'quickly', 'ig', 'viz', 'whose', 'goes', 'recent', 'getting', 'd', 'clearly', 'went', 'hither', 'definitely', 'indeed', 'some', 'tn', 'i7', 'happens', 'i6', "why's", 'do', '15', 'looking', "there's", "didn't", 'im', 'right', 'ca', "'ll", "when's", 'hereby', 'possibly', "what's", 'ibid', 'rl', 'be', 'research-articl', 'thereupon', 'amoungst', 'across', 'saw', 'say', 'ed', 'youd', 'G', 't2', "c'mon", 'thered', 'ba', 'T', 'fs', 'eighty', 'herein', 'significant', 'different', 'na', 'uses', 'ninety', 'b3', 'les', 'ref', 'l', 'particular', 'cit', 'la', 'K', 'these', "that'll", 'ih', '17', 'among', 'although', 'bp', 'ue', 'ei', '2', 'world', 'lest', 'during', 'can', 'uk', 'asking', '30', 'C', '23', 'p2', 'es', 'twenty', 'as', 'line', 'give', 'g', 'ge', 'X', 'yl', 'sp', 'suggest', 'R', "here's", 'pas', 'our', 'ne', 'available', 'placed', 'om', 'proud', 'whoever', 'forth', 'cd', 'came', 'eg', 'mu', 'dy', 'x2', 'c', 'usefulness', 'part', 'lb', 'ts', 'm2', 'nobody', 'op', 'wa', 'ho', 'words', 'mill', 'mightn', 'du', 'zi', 'causes', 'with', 'selves', 'both', 'second', "can't", 'information', "weren't", "'ve", 'co', 'anyway', 'ev', 'am', 'three', 'bk', 'into', 'sr', 'soon', 'him', 'alone', 'whos', 'rather', 'thats', 'has', 'formerly', 'front', 'see', 'those', 'begins', 'containing', 'an', 'gone', 'af', 'keep', 'sn', "there've", 'nos', 'los', 'put', 'looks', 'ft', 'p1', 'thanx', 'namely', 'unless', 'vu', 'aside', 'xn', 'let', 'gi', 'must', 'bn', 'eight', 'away', 'amount', 'down', 'cl', 'ok', 'ps', 'nothing', 'mainly', 'itd', "we're", 'ch', 'rt', 'k', 'ia', 'df', 'sensible', 'hh', 'four', 'mug', 'whim', 'http', 'your', 'oq', 'very', 'becoming', 'behind', 'similarly', 'D', 'many', 'wouldn', 'sub', 'said', 'ah', 'shed', 'thickv', 'show', 'effect', 'sq', 'll', 'little', 'giving', 'L', 'dl', 'ar', 'ma', "you've", 'currently', 'beforehand', 'cr', 'howbeit', 'ga', 'fc', 'such', 'da', 'hs', 'tj', 'consequently', 'become']

# Set Document as an articel to pass it and check the Keyword Generator.

document1 = "Ram@ is a good@#$ guy, worst negetive i hate bishal i love nyima bishal is a bad@#$ guy@#$. good@#$ or bad@#$ hello sarkar. Hello world, this is a try for the Keyword Generation which is working on the dependency function of remove_punctuation, token_generation, term_frequency and then with the inverse_document_frequency. This now sentence is to check the term frequency by the sentence: Hello world i love coding, hello world i love coding, hello world i hate coding, hello world i hate codiing, hello sir how is my project going on ! helo sir how is my project going on!. now the sentence ends here. and then finally the keywords_generation function works. Bikkikaushal@_ Bikkikaushal@_ Bikkikaushal@_ Bikkikaushal!_ Bikkikaushal@_ fuck that i hate coding fuck that i hate coding fuck that i hate coding fuck that i hate coding oh no i love coding actually"

# Removing Punctuations.
 
def remove_punctuation(document):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~`”'''

    punctuation_less_document = ""

    for d in document:
        if d not in punctuations:
            punctuation_less_document = punctuation_less_document + d

    return punctuation_less_document


def remove_punctuation_ends(document):
    punctuations = '''!()-[]{};:'"\,<>/@#$%^&*_~`”'''

    punctuation_less_document = ""

    for d in document:
        if d not in punctuations:
            punctuation_less_document = punctuation_less_document + d

    return punctuation_less_document



# Token Generator.

def token_generation(document):
    token1 = word_tokenize(document)

    stop_words = stopwords.words('english')

    kt1 = []
    
    for keyword in token1:
        if keyword not in stop_words:
            kt1.append(keyword)

    tokens = list(set(kt1))
    return tokens


# Term frequency calculator.

def term_frequency(term, sentence):
    repetition_count = 0
    tokens_in_term = word_tokenize(term)
    uni_tokens_in_sentence = word_tokenize(sentence)
    term_length = len(tokens_in_term)
    tokens_in_sentence = []
    for token in list(ngrams(uni_tokens_in_sentence, term_length)):
        tokens_in_sentence.append(' '.join(token))
    for token in tokens_in_sentence:
        if term == token:
            repetition_count += 1
    tokens_count = len(list(tokens_in_sentence))
    if tokens_count == 0:
        term_freq = 0
    else:
        term_freq = repetition_count/tokens_count
    return float(term_freq)


# Inverse Document Frequency

def inverse_document_frequency(word, document):
    sentences_in_document = sent_tokenize(document)
    sentences_count = len(sentences_in_document)

    sentences_with_word_count = 0

    for s in sentences_in_document:
        if word in s:
            sentences_with_word_count += 1

    if sentences_with_word_count == 0:
        idf = 0
    else:
        idf = math.log(sentences_count/sentences_with_word_count)

    return float(idf)



# Keywords Generator.

def keyword_generation(document):
    stop_words = set(stopwords.words("english"))
    document_without_stopwords = " "
    lowercase_document = document.lower()
    for word in word_tokenize(lowercase_document):
        if word not in stop_words:
            document_without_stopwords = document_without_stopwords + " " + word
    document_with_remove_punctuation = remove_punctuation(document_without_stopwords)
    tokens_generated = token_generation(document_with_remove_punctuation)
    term_frequencies = []
    inverse_document_frequencies = []
    document_with_ends = remove_punctuation_ends(document_without_stopwords)
    document_with_ends = document_with_ends.replace("  ", " ")
    document_with_ends = document_with_ends.replace(" .", ".")

    array_position_tf = 0

    tokenized_sentences_with_ends = sent_tokenize(document_with_ends)

    for token in tokens_generated:
        term_frequencies.append([])
        for sentence in tokenized_sentences_with_ends:
            term_frequencies[array_position_tf].append(term_frequency(token, remove_punctuation(sentence)))
        if array_position_tf <= len(tokens_generated):
            array_position_tf += 1

    for token in tokens_generated:
        inverse_document_frequencies.append(inverse_document_frequency(token, document_with_ends))

    term_frequencies_summations = []
    for term_frequencies_per_token in term_frequencies:
        term_frequencies_summations.append(sum(term_frequencies_per_token))

    keywords_importance = {}
    for i in range(0, len(tokens_generated)):
        if inverse_document_frequencies[i] != 0:
            keywords_importance[tokens_generated[i]] = term_frequencies_summations[i] * inverse_document_frequencies[i]

    keywords_importance_sorted = dict(reversed(sorted(keywords_importance.items(), key=lambda x: x[1])))

    tags = []

    for tag in list(keywords_importance_sorted):
        if tag not in unnecessary_words:
            tags.append(tag)

    # return tags[0:200]

    listToStrFinalWords = ' '.join(map(str, tags[0:200]))
    return listToStrFinalWords


# print(keyword_generation(document1))


def sentiment_intensity_analyzer(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    return score


def final_analysis(message):
    # print(sentiment_intensity_analyzer(keyword_generation(message)))
    result = sentiment_intensity_analyzer(keyword_generation(message))
    return result


# print(final_analysis("hello world its amazing"))



