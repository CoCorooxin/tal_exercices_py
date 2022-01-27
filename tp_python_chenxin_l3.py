"""
TP python
L3 - Linguistique Informatique
Xin Chen
n.etudiant : 22105822
EXERCISES FOR NLP
How to evaluate a tagger using python
"""

import json


def read_output(fichier_output):
    with open(fichier_output, encoding= 'utf-8' ) as f:
        corpus = json.load(f)
    return corpus
corpus = read_output("output.json")
"""alter : corpus = json.load(open("output.json") """

def sum_words(corpus):
    """the sum of all the words contained in the corpus"""
    s = 0
    for example in corpus:
        s += len(example["gold_labels"])
    return s
sum_tag = sum_words(corpus)

def get_train_word(train_conllu):
    """get the dict of all words and their possible tags in the train corpus"""
    train_word = {}
    with open(train_conllu, "r", encoding="utf-8") as train_corpus:
        for data_string in train_corpus:
            if data_string.startswith("#") or not data_string.strip():
                continue
            word = data_string.split()[1]
            tag = data_string.split()[3]
            if tag == "_":
                continue

            if word in train_word and tag not in train_word[word]:
                train_word[word].append(tag)
            else:
                train_word[word] = [tag]
    return train_word

train_word= get_train_word("train.conllu")

"""
EX1
(the percentage of words predicted wrong)
"""

def count_error_rate(corpus): # the fonction take a list of dicts as argument
    count_bad_tag = 0
    global sum_tag
    for example in corpus:
        for gl, pl in zip(example["gold_labels"], example["predicted_labels"]):
            if pl != gl:
                count_bad_tag += 1
    error_rate = count_bad_tag/sum_tag
    return f"error_rate: {round(error_rate*100,2)}%"
#print(count_error_rate(corpus))
"""error_rate: 3.37%"""

"""
EX2
"""

def count_perfect_sentence(corpus): #corpus is a list, the fonction returns a string
    count_ps = 0
    count_sentence = 0
    for sentence in corpus:
        count_sentence += 1
        if sentence["gold_labels"] == sentence["predicted_labels"] :
            count_ps+=1
    return f"perfect_sentence_rate: {round(count_ps/count_sentence*100,2)}%"
print(count_perfect_sentence(corpus))
"""perfect_sentence_rate: 51.92%"""


"""
EX3
"""
def w_not_in_train_corpus(corpus, train_word ):   #prend une liste en parametre, retourne un string
    """take into account only words that are not in the training corpus"""
    count_errors = 0
    count_OOV= 0
    for example in corpus:
        for word, gl, pl in zip(example["words"], example["gold_labels"], example["predicted_labels"]):
            if word not in train_word:
                count_OOV += 1
                if gl != pl:
                    count_errors += 1
    return f"error_rate_for_OOV: {round(count_errors/count_OOV*100,2)}% (OOV stands for out of vocabulary)"
#print(w_not_in_train_corpus(corpus, train_word))

"""error_rate_for_OOV: 15.31% (OOV stands for out of vocabulary) """


"""
EX4
"""

def w_ambigu(dicto):
    """sum of words that have multiple possible tags"""
    result = []
    for word, tag in dicto.items():
        if len(tag) != 1:
            result.append(word)
    return result

def rate_ambigu(corpus, w_ambigu):   #prend une liste en parametre, retourne un string
    """the error rate for ambiguous words"""
    count_errors = 0
    count_ambig= 0
    for example in corpus:
        for word, gl, pl in zip(example["words"], example["gold_labels"], example["predicted_labels"]):
            if word in w_ambigu:
                count_ambig += 1
                if gl != pl:
                    count_errors += 1
    return f"error_rate_for_ambiguous_words: {round(count_errors/count_ambig*100,2)}%"

list_word_ambigu =  w_ambigu(get_train_word("train.conllu"))
#print(rate_ambigu(corpus,list_word_ambigu))
"""error_rate_for_ambiguous_words: 14.88%"""


"""
EX5
the confusion matrix for the tagger's result of prediction 
"""
def confusion_matrix(corpus):   #la matrice de confusion pour les étiquetes dans le corpus
    confusion_matrix = {}
    for example in corpus:
        for gl, pl in zip(example["gold_labels"], example["predicted_labels"]):
            if gl not in confusion_matrix:
                confusion_matrix[gl] = {gl: 0}
            if gl != pl and pl not in confusion_matrix[gl]:
                confusion_matrix[gl][pl] = 1
            elif gl != pl and pl in confusion_matrix[gl]:
                confusion_matrix[gl][pl] += 1
            else:
                confusion_matrix[gl][gl] += 1

    return confusion_matrix

#print(confusion_matrix(corpus))
confusion_matrix = confusion_matrix(corpus)
#print(len(confusion_matrix))
"""
{'PRON': {'PRON': 535, 'SCONJ': 3, 'ADJ': 2, 'DET': 2, 'ADV': 3, 'PROPN': 2, 'VERB': 1}, 'VERB': {'VERB': 776, 'ADJ': 9, 'AUX': 19, 'NOUN': 11, 'ADV': 6, 'PROPN': 4, 'ADP': 1}, 'SCONJ': {'SCONJ': 117, 'PRON': 7, 'ADP': 3, 'ADV': 1}, 'ADP': {'ADP': 1478, 'NOUN': 1, 'DET': 2, 'ADV': 1}, 'CCONJ': {'CCONJ': 247, 'PROPN': 1}, 'DET': {'DET': 1468, 'ADJ': 5, 'ADP': 4, 'PROPN': 2, 'VERB': 1, 'PRON': 1}, 'NOUN': {'NOUN': 1781, 'PROPN': 41, 'VERB': 15, 'ADJ': 21, 'ADV': 6, 'X': 1, 'NUM': 3, 'PRON': 3, 'DET': 1}, 'ADJ': {'ADJ': 551, 'DET': 4, 'PROPN': 10, 'VERB': 27, 'NOUN': 10, 'ADV': 3}, 'AUX': {'AUX': 347, 'VERB': 7, 'ADV': 1}, 'ADV': {'ADV': 473, 'VERB': 3, 'SCONJ': 4, 'NOUN': 4, 'ADJ': 2, 'PRON': 1, 'CCONJ': 1, 'PROPN': 1, 'ADP': 3}, 'PUNCT': {'PUNCT': 1186}, 'PROPN': {'PROPN': 459, 'NOUN': 30, 'VERB': 1, 'X': 6, 'NUM': 3, 'ADJ': 3, 'DET': 1}, 'NUM': {'NUM': 215, 'ADJ': 1, 'VERB': 1, 'DET': 3, 'PROPN': 1, 'NOUN': 3}, 'SYM': {'SYM': 35, 'NOUN': 2, 'ADJ': 1}, 'X': {'X': 2, 'NOUN': 1, 'PROPN': 12, 'NUM': 2}, 'INTJ': {'INTJ': 6, 'PROPN': 2}}
"""
