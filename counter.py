"""
TP python
L3 - Linguistique Informatique
Xin Chen
n.etudiant : 22105822
EXERCISES FOR NLP
How to evaluate a tagger using python
"""

import json

"""on ouvre les fichiers(output.json et train.conllu) qu'on va utiliser"""
file = open("output.json", encoding= 'utf-8' )
corpus = json.load(file)
"""corpus est lu comme une liste"""

train_corpus = open("train.conllu", "r", encoding="utf-8")
read_conllu = list(train_corpus)

""" on regroupe les fonctions utilitares: ce sont des petites toolkits."""

def sum_t(corpus):
    """calculer l'ensemble de mots contenant dans le corpus"""
    s = 0
    for example in corpus:
        s += len(example["gold_labels"])
    return s
sum_tag = sum_t(corpus)

def train_word_list():
    """la fonction pour récupérer les mots contenant dans le train corpus"""
    train_word = []
    for data_string in read_conllu:
        try:
            data_string = data_string.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            continue

        """J'ai du mal à comprendre pourquoi on a besoin de ce processus d'encodage et de décodage, mais si je l'ignore
        mon programme va retourner des mauvais résultats """

        line = data_string.strip().split()
        if len(line) >= 10 and "#" not in data_string:
            train_word.append(line[1])
    return train_word

train_word= train_word_list()

"""
EX1
(pourcentage de mots dont les etiquettes predite 
ne correspond pas aux etiquettes de reference).
"""

def count_error_rate(corpus): # prendre en argument une liste composee des dicts, retourn une string
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
def count_perfect_sentence(corpus): #prendre en argument une liste, retourne un string
    count_ps = 0
    global sum_tag
    for example in corpus:
        if example["gold_labels"] == example["predicted_labels"] :
            count_ps+=1
    return f"perfect_sentence_rate: {round(count_ps/sum_tag*100,2)}%"
#print(count_perfect_sentence(corpus))
"""perfect_sentence_rate: 2.16%"""


"""
EX3
"""
def w_not_in_train_corpus(corpus, train_word ):   #prend une liste en parametre, retourne un string
    """la fonction va retourner le taux d'érreur en prenant en compte que les mots hors de train corpus"""
    count_errors = 0
    count_OOV= 0
    for example in corpus:
        for word, gl, pl in zip(example["words"], example["gold_labels"], example["predicted_labels"]):
            if word not in train_word:
                count_OOV += 1
                if gl != pl:
                    count_errors += 1
    return f"error_rate_for_OOV: {round(count_errors/count_OOV*100,2)}% (OOV stands for out of vocabulary)"
print(w_not_in_train_corpus(corpus, train_word))

"""error_rate_for_OOV: 7.17% (OOV stands for out of vocabulary) """

"""
EX4
"""
def train_w_dict(read_conllu):
    """la fonction pour recuperer tous les mots ambigus et leurs étiquetes dans le train corpus """
    dict_word = {}
    for data_string in read_conllu:
        """le boucle for itère sur le fichier contenant train corpus pour récuperer les mots dedans"""
        try:
            data_string = data_string.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            continue

        if len(data_string) <= 1 or data_string[0] == "#":
            continue
        words_l = data_string.strip().split()
        if words_l[1] in dict_word and words_l[3] not in dict_word[words_l[1]]:
            dict_word[words_l[1]].append(words_l[3])
        else:
            dict_word[words_l[1]] = [words_l[3]]
    return dict_word


def w_ambigu(dicto):
    """la fonction recupère une liste de mots ambigus depuis le dicto crée par la fonction train_word_dict"""
    result = []
    for word, tag in dicto.items():
        if len(tag) != 1:
            result.append(word)
    return result
#print(w_ambigu(train_w_dict(read_conllu)))
#print(len(w_ambigu(train_w_dict(read_conllu))))

def rate_ambigu(corpus, w_ambigu):   #prend une liste en parametre, retourne un string
    """la fonction va retourner le taux d'erreur pour les mots qui ont plusieurs étiquetes possibles"""
    count_errors = 0
    count_ambig= []
    for example in corpus:
        for word, gl, pl in zip(example["words"], example["gold_labels"], example["predicted_labels"]):
            if word in w_ambigu:
                count_ambig.append(word)
                if gl != pl:
                    count_errors += 1
    return f"error_rate_for_ambiguous_words: {round(count_errors/len(count_ambig)*100,2)}%"

list_word_ambigu =  w_ambigu(train_w_dict(read_conllu))
print(rate_ambigu(corpus,list_word_ambigu))
"""error_rate_for_ambiguous_words: 15.79%"""

"""C'est bien plus élévé que le taux d'erreur sur l'ensemble de corpus, parce que ce sont des mots que l'analyseur n'a jamais rencontré"""

"""
EX5
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
#confusion_matrix = confusion_matrix(corpus)
#print(len(confusion_matrix))
"""
{'PRON': {'PRON': 535, 'SCONJ': 3, 'ADJ': 2, 'DET': 2, 'ADV': 3, 'PROPN': 2, 'VERB': 1}, 'VERB': {'VERB': 776, 'ADJ': 9, 'AUX': 19, 'NOUN': 11, 'ADV': 6, 'PROPN': 4, 'ADP': 1}, 'SCONJ': {'SCONJ': 117, 'PRON': 7, 'ADP': 3, 'ADV': 1}, 'ADP': {'ADP': 1478, 'NOUN': 1, 'DET': 2, 'ADV': 1}, 'CCONJ': {'CCONJ': 247, 'PROPN': 1}, 'DET': {'DET': 1468, 'ADJ': 5, 'ADP': 4, 'PROPN': 2, 'VERB': 1, 'PRON': 1}, 'NOUN': {'NOUN': 1781, 'PROPN': 41, 'VERB': 15, 'ADJ': 21, 'ADV': 6, 'X': 1, 'NUM': 3, 'PRON': 3, 'DET': 1}, 'ADJ': {'ADJ': 551, 'DET': 4, 'PROPN': 10, 'VERB': 27, 'NOUN': 10, 'ADV': 3}, 'AUX': {'AUX': 347, 'VERB': 7, 'ADV': 1}, 'ADV': {'ADV': 473, 'VERB': 3, 'SCONJ': 4, 'NOUN': 4, 'ADJ': 2, 'PRON': 1, 'CCONJ': 1, 'PROPN': 1, 'ADP': 3}, 'PUNCT': {'PUNCT': 1186}, 'PROPN': {'PROPN': 459, 'NOUN': 30, 'VERB': 1, 'X': 6, 'NUM': 3, 'ADJ': 3, 'DET': 1}, 'NUM': {'NUM': 215, 'ADJ': 1, 'VERB': 1, 'DET': 3, 'PROPN': 1, 'NOUN': 3}, 'SYM': {'SYM': 35, 'NOUN': 2, 'ADJ': 1}, 'X': {'X': 2, 'NOUN': 1, 'PROPN': 12, 'NUM': 2}, 'INTJ': {'INTJ': 6, 'PROPN': 2}}
"""

read_conllu = list(open("train.conllu", "r", encoding="utf-8"))
word_corpus = set()
for data_string in read_conllu:
    words_l = data_string.strip().split()
    if len(words_l) >= 10 and "#" not in data_string:
        word_corpus.add(words_l[1])
#print(word_l)
"""
J'ai un doute sur le choix de vocabulaire ici, cad est-ce que je prend en compte les variables des mots
dans leurs formes finis (conjuguées, masculin, féminin, sing, pl, etc...) ;
ou je choisis que les mots de racine? Ici, j'ai fait le premier choix qui est de prendre en compte tous les variables des mots dans
le train corpus,
qui a donné un taux d'erreur très élévé à la fin.
"""
def w_only_train_corpus(corpus, word_corpus):
    count_errors = 0
    count_OOV= 0
    for example in corpus:
        for word, gl, pl in zip(example["words"], example["gold_labels"],example["predicted_labels"]):
            if word not in word_corpus:
                count_OOV +=1
                if gl != pl:
                    count_errors += 1
# print(len(l_n)) -> 601 all together 601 words out of train corpus
    return f"error_rate_for_OOV: {round(count_errors/count_OOV*100,2)}% (OOV stands for out of vocabulary)"

print(w_only_train_corpus(corpus, word_corpus))
"""error_rate_for_OOV: 15.31% (OOV stands for out of vocabulary)"""

def rate_ambigu(corpus, list_w_ambigu):
    count_errors = 0
    count_ambig= 0
    for example in corpus:
        for word, gl, pl in zip(example["words"], example["gold_labels"],example["predicted_labels"]):
            if word in list_w_ambigu:
                count_ambig +=1
                if gl != pl:
                    count_errors += 1
    return f"error_rate_for_ambiguous_words: {round(count_errors/count_ambig*100,2)}%"
print(rate_ambigu(corpus, w_ambigu()))
"""error_rate_for_ambiguous_words: 1.94%"""


file.close()
train_corpus.close()

"""
apparemment on peut egalement utiliser with open(), mais je ne sais pas comment l'agencer d'une maniere moins laborieuse
"""