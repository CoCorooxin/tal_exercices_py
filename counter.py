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
corpus = json.load(file) "corpus est lu comme une liste"

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
        words_brut = data_string.strip().split()
        if len(data_string) <= 1 or data_string[0] == "#":
            continue
        train_word.append(words_brut[1])
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
        """apparement il faut decoder le fichier en 'latin1' et ensuite l'encoder a nouveau en utf-8... mais j'ai aucune idee pourquoi on fait ca
        mais ça a donné le bon résultat(si j'ignore ce processus j'obtienne un taux d'erreur pour les mots ambigus moins élévé que le taux d'erreur pour l'ensemble de corpus )"""
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

print(confusion_matrix(corpus))
#confusion_matrix = confusion_matrix(corpus)
#print(len(confusion_matrix))
"""
{'PRON': {'PRON': 535, 'SCONJ': 3, 'ADJ': 2, 'DET': 2, 'ADV': 3, 'PROPN': 2, 'VERB': 1}, 'VERB': {'VERB': 776, 'ADJ': 9, 'AUX': 19, 'NOUN': 11, 'ADV': 6, 'PROPN': 4, 'ADP': 1}, 'SCONJ': {'SCONJ': 117, 'PRON': 7, 'ADP': 3, 'ADV': 1}, 'ADP': {'ADP': 1478, 'NOUN': 1, 'DET': 2, 'ADV': 1}, 'CCONJ': {'CCONJ': 247, 'PROPN': 1}, 'DET': {'DET': 1468, 'ADJ': 5, 'ADP': 4, 'PROPN': 2, 'VERB': 1, 'PRON': 1}, 'NOUN': {'NOUN': 1781, 'PROPN': 41, 'VERB': 15, 'ADJ': 21, 'ADV': 6, 'X': 1, 'NUM': 3, 'PRON': 3, 'DET': 1}, 'ADJ': {'ADJ': 551, 'DET': 4, 'PROPN': 10, 'VERB': 27, 'NOUN': 10, 'ADV': 3}, 'AUX': {'AUX': 347, 'VERB': 7, 'ADV': 1}, 'ADV': {'ADV': 473, 'VERB': 3, 'SCONJ': 4, 'NOUN': 4, 'ADJ': 2, 'PRON': 1, 'CCONJ': 1, 'PROPN': 1, 'ADP': 3}, 'PUNCT': {'PUNCT': 1186}, 'PROPN': {'PROPN': 459, 'NOUN': 30, 'VERB': 1, 'X': 6, 'NUM': 3, 'ADJ': 3, 'DET': 1}, 'NUM': {'NUM': 215, 'ADJ': 1, 'VERB': 1, 'DET': 3, 'PROPN': 1, 'NOUN': 3}, 'SYM': {'SYM': 35, 'NOUN': 2, 'ADJ': 1}, 'X': {'X': 2, 'NOUN': 1, 'PROPN': 12, 'NUM': 2}, 'INTJ': {'INTJ': 6, 'PROPN': 2}}
"""

file.close()
train_corpus.close()

"""
apparemment on peut egalement utiliser with open(), mais je ne sais pas comment l'agencer d'une maniere moins laborieuse
"""

""" 
les résultats obtenus pour toutes les fonctions:
error_rate: 3.37%
perfect_sentence_rate: 2.16%
error_rate_for_OOV: 7.17% (OOV stands for out of vocabulary)
['annonce', 'Grand', 'Tourisme', 'pouvoir', 'final', 'avant', 'ont', 'Ouest', 'peu', 'femelles', 'finlandais', 'reste', 'coucher', 'leur', 'tout', 'existant', "D'", 'Internationale', 'recherche', 'tous', 'innocents', 'politiques', 'uniforme', 'vis', 'seconde', 'Est', 'gauche', 'Clash', 'Titans', 'politique', 'sourire', 'prime', 'quelle', '55', 'seul', 'devoir', 'responsable', 'A', 'sierra', 'dure', 'Jedi', 'certains', 'escorte', 'sorties', 'Heavy', 'Metal', 'Hard', 'Normand', 'x', 'mort', 'cadets', 'compte', 'nouvelles', 'flamand', 'savoir', 'touche', 'Or', '550', 'jusque', 'produit', 'probiotiques', 'Cours', 'vide', 'Pistol', 'Brigade', 'Trio', 'Man', 'Army', 'Waters', 'Quelque', 'Alliance', 'complice', 'NBA', 'Me', 'super', 'marque', 'contente', 'M.', 'second', 'maternelle', 'pire', 'large', 'Dr.', 'pionnier', 'fera', 'X', 'vivant', 'synonymes', 'N4', 'Certains', 'frais', 'BE', 'Billy', '68', '02', 'voisin', 'Exhibition', 'Blazers', 'envers', 'missionnaires', 'protestants', 'Esprit', 'anciens', 'pose', 'mobile', 'Comme', 'Chez', 'fermes', 'locaux', 'National', 'bimoteur', 'ravageur', 'di', 'folles', 'Freedom', 'Group', 'sportifs', 'colonial', 'tranche', 'Alaska', 'comprise', 'issue', 'auront', 'rouge', 'pair', 'Quels', 'Chess', 'Mer', 'PV', 'missionnaire', 'combattant', 'collecte', 'secondes', 'loge', 'partis', 'Justice', 'repoussant', 'composant', 'emporos', 'dorique', 'occupant', 'fort', 'mal', 'incident', '+5', 'e', 'championne', 'total', 'complexes', 'Mission', 'estime', 'souper', 'Augustine', 'ibn', 'Your', 'observateurs', 'contemporains', 'AITA', '300', 'USB', 'interdite', 'grec', 'Ceroxylon', 'marchands', 'mondiaux', 'Philippines', 'c', 'diverses', 'automobile', 'mi', 'kommune', 'primaire', 'troubles', 'mordant', 'immeubles', 'change', 'offensive', 'majeure', 'Suivant', 'logique', 'Direction', 'relance', 'gags', 'White', 'Universal', 'versant', 'Out', 'juste', 'tribal', 'cyclistes', 'Quel', 'Hors', 'victime', 'blanc', 'noble', 'malheureux', 'sous-marin', 'identifiant', 'frits', 'Challenge', 'Programme', 'semblant', 'patriote', 'mineurs', 'avez', 'bulgare', 'exemplaire', 'Concert', "Qu'", 'DJ', 'C.', 'indiens', 'Nuits', 'vif', 'dessert', 'Bassin', 'primaires', 'public', 'rituels', 'pauvres', 'Bourguignons', 'Peuple', 'Royal', 'POSITIF', 'inconscient', 'noir', 'coupable', 'Princesse', 'Sterling', 'Rural', 'ressort', 'pieux', 'Society', 'RAF', 'Heures', 'Invalides', 'crus', 'Discovery', '400', 'OTAN', 'Landing', 'natifs', 'Body', 'Journal', 'rire', 'Village', 'XVI', 'City', '150', 'demi', 'complexe', 'suivantes', 'Mgr', 'ton', 'Indiens', 'Red', 'LNH', 'Commons', 'manifeste', 'BD', '147', 'Blue', 'extrait', 'excuse', 'Lancaster', 'Vierge', 'Gundam', 'guerriers', 'Dragon', 'Baron', '3D', 'Sweet', 'dynamique', 'Kingdom', 'Chapelle', 'Echelon', 'Film', 'Ricky', 'gagnant', 'Ensemble', 'marocains', 'Four', 'pa', 'Hospital', 'Lost', 'esclave', 'PME', 'Zone', 'Garde', 'Town', 'VS', 'Mr', 'massif', 'Argentina', 'Marine', '17h', 'Entente', 'perpendiculaires', 'calme', 'Sportive', 'conservateur', 'impressionnistes', 'riches', 'militant', 'Avant', 'conduite', 'Sans', 'Indians', 'Football', 'alternatives', 'lorrains', 'infini', 'Oz', 'P2', 'Olympiades', 'Birds', 'incluse', '*', 'Airways', 'secrets', 'veille', 'ovale', 'basses', 'UAI', 'Canadien', 'Rapport', 'Chemins', 'Fer', 'Pucelle', 'albanais', 'Nature', 'CNC', 'Capitaine', 'Technologies', 'adulte', 'riverains', 'porteurs', 'Deutsche', 'House', 'botanique', 'No', 'ARNm', 'Week-end', 'Secret', 'Forces', 'First', 'Services', 'Annales', 'Ayant', 'porcine', 'Quatorze', 'As', 'citoyens', 'Central', 'Tejo', 'inversible', 'parodie', 'industriel', 'francophones', 'Chances', 'Lady', 'creux', 'effectif', 'Vert', 'hacker', 'Australian', 'Essai', 'domestiques', 'gage', 'portable', 'Golfe', 'BP', 'Perse', 'Bureau', 'Octobre', 'fins', 'nickel', 'Lie', 'speedrun', 'K', 'Caisse', 'Dr', 'Web', 'calcaire', 'Sahraouie', 'parent', 'lituanien', 'Drifters', 'Forges', 'Honneur', 'Bridge', 'fossiles', 'Research', 'Project', 'RRP', 'Projet', 'G8', 'menace', 'Second', 'souffle', 'Council', 'AC', 'N', 'Lords', 'Chemin', 'correspondante', 'Territoire', 'PDG', 'L', 'faste', 'Old', 'inconnu', 'collectifs', 'Produit', 'perse', 'master', 'Joie', 'Damariscotta', 'Industries', 'extraits', 'EHPAD', 'hippique', 'illustre', 'commande', 'Illusion', "'06", 'Riverside', 'humain', 'PC', 'European', 'Acte', 'mexicaine', 'Version', 'Florida', 'Heavyweight', 'fasciste', 'alsaciens', 'PAN', 'fantasy', 'SFIO', 'Sound', 'Burn', 'Restauration', 'Know', 'So', 'Bretonne', 'Noirs', 'Advanced', 'Sudistes', 'mutuelle', 'Graves', 'finance', 'G3', 'circulaires', 'Biographie', 'minima', 'demandes', 'Clair', 'bordure', 'Russes', 'Documentation', '1833', 'approfondie', 'Voix', '520', 'suspect', 'Affaire', 'liquides', 'Magazine', 'plastiques', 'Sonobe', 'mutant', 'Prince', 'British', 'Doctrine', 'solides', 'soutenue', 'Ville', '>', 'Corps', 'Belges', 'couverte', 'Allemands', 'S.A.', 'Social', 'Driver', 'Trust', 'Andrews', 'Las', 'Information', 'Jeunesse', 'amphibies', 'Shadow', 'Euros', 'Concept', 'Sport', 'Historic', 'augure', 'CEDH', 'domestique', 'remplies', 'sage', 'ferait', 'africains', 'DA', 'Sahraouis', 'ter', 'manquant', 'nomade', 'Pistons', 'Note', 'LP', 'Acton', 'Shropshire', 'reporter', 'Porte', 'interdits', 'brise', 'candidat', 'BDE', 'Pack', 'Jaune', 'Transport', 'Spider-Man', 'Testament', 'atteinte', 'Top', 'Carrefour', 'Viking', 'moral', 'Guru', 'Bataille', 'synonyme', 'provinciales', 'SI', 'rivales', 'U.S.', 'AA', 'AAA', 'Blood', 'Thraces', 'Travail', 'Paton', 'Opera', 'Base', 'Gray', 'Pacific', 'Monastir', 'Concernant', 'Annonciation', 'Professeur', 'Roman', 'Mondial', 'console', 'Bar', 'Photo', 'solitaires', 'PMU', 'traversiers', 'Train', 'bilingue', 'hebdomadaires', 'brillant', 'fier', 'assassin', 'CE', 'Tutsi', 'contraste', 'instructeur', 'manufacturiers', 'Acadiens', 'gmina', 'Vos', 'Allo', 'Stop', 'Chroniques', 'Communication', '149', 'aspirant', 'Dynastor', 'macrosiris', 'Manifeste', 'Agriculture', 'RAS', 'orange', 'persan', 'file', 'Saints', 'tabous', 'Fables', 'DES', 'castor', 'fine', 'tentant', 'amorce', 'afrikaans', 'nulles', 'Scorpion', 'j', 'majeurs', 'capture', 'mat', 'Premiers', 'fluides', 'naturalistes', 'Seigneur', 'personnels', 'Sounds', 'it', 'roses', 'Quartier', 'Latin', 'abrupt', 'patrouille', 'Roma', 'infra', 'Autruche', 'SD', 'Go', 'Mode', 'Salut', 'portables', 'Market', 'Songs', 'Cartel', 'death', 'Fabricants', 'Six', 'revenu', 'souterrains', 'Coney', 'voisine', 'radical', 'abbatiale', 'Girl', 'Session', 'avertis', 'ap', 'parachutistes', 'Earth', 'rivaux', 'Tel', 'cartographie', 'suis', '251', 'Tentation', 'rose', 'Institution', 'Analysis', 'media', 'Wales', 'China', 'Dream', 'han', 'Agrias', 'Uptown', 'dirigeantes', 'Of', 'Juin', 'souterrain', 'O', '4B', 'IIHF', 'Robot', 'guerrier', 'corrompus', 'Jeu', 'phares', 'Grands', 'Nickel', 'District', 'rondes', 'Business', 'lisse', 'Avenue', 'Parc', 'poster', 'Jenna', 'Jameson', 'Cavalier', 'homosexuels', 'grecs', 'con', 'Program', 'Guide', 'Recordings', 'logos', 'decumanus', 'Bourg', 'BoJ', 'seyssel', 'factice', 'codex', 'Right', 'Wild', 'Bois', 'Zeitung', 'protecteur', 'Jefferson', 'Bone', 't.', 'Telle', 'Rosa', 'NT', 'Tchan', 'Brother', 'prairial', 'Vega', 'discontinu', 'Courant', 'Rennais', 'Nuit', 'perdant', 'priori', 'A.L.F.A.', 'attrayant', 'valide', 'doublement', 'Winter', 'Fiction', 'Album', 'immortel', 'enduit', 'Arabes', 'Palermo', 'engageant', 'Phaeriemagick', 'Sugar', 'Irlandais', 'AG', 'Life', 'Act', 'Lyonnais', 'antibiotiques', 'Scout', 'no', 'Ju', 'Tag', 'QUE', 'kendayan', 'Patrie', 'Handicap', 'Eau']
743
error_rate_for_ambiguous_words: 15.79%
{'PRON': {'PRON': 535, 'SCONJ': 3, 'ADJ': 2, 'DET': 2, 'ADV': 3, 'PROPN': 2, 'VERB': 1}, 'VERB': {'VERB': 776, 'ADJ': 9, 'AUX': 19, 'NOUN': 11, 'ADV': 6, 'PROPN': 4, 'ADP': 1}, 'SCONJ': {'SCONJ': 117, 'PRON': 7, 'ADP': 3, 'ADV': 1}, 'ADP': {'ADP': 1478, 'NOUN': 1, 'DET': 2, 'ADV': 1}, 'CCONJ': {'CCONJ': 247, 'PROPN': 1}, 'DET': {'DET': 1468, 'ADJ': 5, 'ADP': 4, 'PROPN': 2, 'VERB': 1, 'PRON': 1}, 'NOUN': {'NOUN': 1781, 'PROPN': 41, 'VERB': 15, 'ADJ': 21, 'ADV': 6, 'X': 1, 'NUM': 3, 'PRON': 3, 'DET': 1}, 'ADJ': {'ADJ': 551, 'DET': 4, 'PROPN': 10, 'VERB': 27, 'NOUN': 10, 'ADV': 3}, 'AUX': {'AUX': 347, 'VERB': 7, 'ADV': 1}, 'ADV': {'ADV': 473, 'VERB': 3, 'SCONJ': 4, 'NOUN': 4, 'ADJ': 2, 'PRON': 1, 'CCONJ': 1, 'PROPN': 1, 'ADP': 3}, 'PUNCT': {'PUNCT': 1186}, 'PROPN': {'PROPN': 459, 'NOUN': 30, 'VERB': 1, 'X': 6, 'NUM': 3, 'ADJ': 3, 'DET': 1}, 'NUM': {'NUM': 215, 'ADJ': 1, 'VERB': 1, 'DET': 3, 'PROPN': 1, 'NOUN': 3}, 'SYM': {'SYM': 35, 'NOUN': 2, 'ADJ': 1}, 'X': {'X': 2, 'NOUN': 1, 'PROPN': 12, 'NUM': 2}, 'INTJ': {'INTJ': 6, 'PROPN': 2}}
"""