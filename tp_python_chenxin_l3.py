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

def read_output(fichier_output):
    with open(fichier_output, encoding= 'utf-8' ) as f:
        corpus = json.load(f)
    return corpus
corpus = read_output("output.json")
"""corpus est lu comme une liste"""

""" on regroupe les fonctions utilitares: ce sont des petites toolkits."""

def sum_t(corpus):
    """calculer l'ensemble de mots contenant dans le corpus"""
    s = 0
    for example in corpus:
        s += len(example["gold_labels"])
    return s
sum_tag = sum_t(corpus)

def get_train_word(fichier_train):
    """la fonction pour récupérer les mots contenant dans le train corpus"""
    train_word = {}
    with open(fichier_train, "r", encoding="utf-8") as read_conllu:
        for data_string in read_conllu:
            line = data_string.strip().split()
            if len(line) >= 10 and "#" not in data_string:
                if line[1] in train_word and line[3] not in train_word[line[1]]:
                    train_word[line[1]].append(line[3])
                else:
                    train_word[line[1]] = [line[3]]
    return train_word

train_word= get_train_word("train.conllu")

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
print(count_error_rate(corpus))
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
print(count_perfect_sentence(corpus))
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

"""error_rate_for_OOV: 15.31% (OOV stands for out of vocabulary) """
"""C'est bien plus élévé que le taux d'erreur sur l'ensemble de corpus, parce que ce sont des mots que l'analyseur n'a jamais rencontré"""

"""
EX4
"""

def w_ambigu(dicto):
    """la fonction recupère une liste de mots ambigus depuis le dicto crée par la fonction train_word_dict"""
    result = []
    for word, tag in dicto.items():
        if len(tag) != 1:
            result.append(word)
    return result

def rate_ambigu(corpus, w_ambigu):   #prend une liste en parametre, retourne un string
    """la fonction va retourner le taux d'erreur pour les mots qui ont plusieurs étiquetes possibles"""
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
print(rate_ambigu(corpus,list_word_ambigu))
"""error_rate_for_ambiguous_words: 14.88%"""


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

print(confusion_matrix(corpus))
confusion_matrix = confusion_matrix(corpus)
#print(len(confusion_matrix))
"""
{'PRON': {'PRON': 535, 'SCONJ': 3, 'ADJ': 2, 'DET': 2, 'ADV': 3, 'PROPN': 2, 'VERB': 1}, 'VERB': {'VERB': 776, 'ADJ': 9, 'AUX': 19, 'NOUN': 11, 'ADV': 6, 'PROPN': 4, 'ADP': 1}, 'SCONJ': {'SCONJ': 117, 'PRON': 7, 'ADP': 3, 'ADV': 1}, 'ADP': {'ADP': 1478, 'NOUN': 1, 'DET': 2, 'ADV': 1}, 'CCONJ': {'CCONJ': 247, 'PROPN': 1}, 'DET': {'DET': 1468, 'ADJ': 5, 'ADP': 4, 'PROPN': 2, 'VERB': 1, 'PRON': 1}, 'NOUN': {'NOUN': 1781, 'PROPN': 41, 'VERB': 15, 'ADJ': 21, 'ADV': 6, 'X': 1, 'NUM': 3, 'PRON': 3, 'DET': 1}, 'ADJ': {'ADJ': 551, 'DET': 4, 'PROPN': 10, 'VERB': 27, 'NOUN': 10, 'ADV': 3}, 'AUX': {'AUX': 347, 'VERB': 7, 'ADV': 1}, 'ADV': {'ADV': 473, 'VERB': 3, 'SCONJ': 4, 'NOUN': 4, 'ADJ': 2, 'PRON': 1, 'CCONJ': 1, 'PROPN': 1, 'ADP': 3}, 'PUNCT': {'PUNCT': 1186}, 'PROPN': {'PROPN': 459, 'NOUN': 30, 'VERB': 1, 'X': 6, 'NUM': 3, 'ADJ': 3, 'DET': 1}, 'NUM': {'NUM': 215, 'ADJ': 1, 'VERB': 1, 'DET': 3, 'PROPN': 1, 'NOUN': 3}, 'SYM': {'SYM': 35, 'NOUN': 2, 'ADJ': 1}, 'X': {'X': 2, 'NOUN': 1, 'PROPN': 12, 'NUM': 2}, 'INTJ': {'INTJ': 6, 'PROPN': 2}}
"""
