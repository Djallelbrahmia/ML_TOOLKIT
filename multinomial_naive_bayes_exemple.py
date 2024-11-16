from Models.none_linear_models.multinomial_naive_bayes import MultinomialNB

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Exemple de données : critiques de films 
critiques = [
    # Critiques positives
    "Un chef-d'œuvre absolu du cinéma contemporain",
    "La performance des acteurs est tout simplement exceptionnelle",
    "Une réalisation magistrale qui restera dans les annales",
    "Des effets spéciaux époustouflants et une histoire captivante",
    "Une mise en scène parfaitement maîtrisée du début à la fin",
    "Scénario intelligent et dialogues percutants",
    "Une expérience cinématographique inoubliable",
    "Les émotions sont palpables tout au long du film",
    "Un film qui repousse les limites du genre",
    "Bande sonore remarquable qui sublime chaque scène",
    "Direction artistique impeccable, chaque plan est magnifique",
    "Un divertissement de haute volée",
    "Casting parfait, chaque acteur brille dans son rôle",
    "Un film qui marque les esprits durablement",
    "Rythme parfait, on ne s'ennuie pas une seconde",
    "Une histoire originale et bien ficelée",
    "Des personnages attachants et bien développés",
    "Un film qui fait réfléchir tout en divertissant",
    "Visuellement sublime et émotionnellement fort",
    "Une réussite sur tous les plans",
    "Suspense haletant jusqu'à la dernière minute",
    "Un film qui mérite tous les éloges",
    "Techniquement irréprochable et artistiquement brillant",
    "Une œuvre qui restera dans les mémoires",
    "Brillamment écrit et magnifiquement réalisé",
    "Un voyage cinématographique exceptionnel",

    
    # Critiques négatives

    "Une réalisation amateur qui fait peur",
    "Deux heures de ma vie que je ne récupérerai jamais",
    "Un film qui s'égare dans tous les sens",
    "Aucun rythme, on s'ennuie fermement",
    "Des personnages mal développés et peu crédibles",
    "Une histoire prévisible du début à la fin",
    "Budget visiblement insuffisant pour un tel projet",
    "Montage décousu qui donne le tournis",
    "Un film qui ne sait pas où il va",
    "Bande sonore irritante et mal utilisée",
    "Des dialogues qui font grincer des dents",
    "Une déception totale sur tous les plans",
    "Réalisation bâclée et sans ambition",
    "Un film qui n'aurait jamais dû sortir",
    "Casting mal choisi et mal dirigé",
    "Histoire sans queue ni tête",
    "Production de très mauvaise qualité",
    "Un film qui accumule les clichés",
    "Effets spéciaux dignes d'un film étudiant",
    "Rythme soporifique du début à la fin",
    "Une catastrophe cinématographique",
    "Aucune émotion, aucune profondeur",
    "Un film vide de sens et d'intérêt",
    "Techniquement faible et artistiquement nul",
    "Une expérience cinématographique pénible",
    "À éviter même gratuitement"
]

# Création des labels (1 pour positif, 0 pour négatif)
labels = np.array([1] * 26 + [0] * 26)

# Préparation des données
vectorizer = CountVectorizer(min_df=1) 
X = vectorizer.fit_transform(critiques).toarray()


# Création et entraînement du modèle
mnb = MultinomialNB(alpha=1.0)
mnb.fit( X, labels)


# Exemple d'utilisation pour de nouvelles critiques
nouvelles_critiques = [
    "Un film extraordinaire avec des effets spéciaux impressionnants et un scénario captivant",
    "Une déception totale, des acteurs médiocres et une histoire sans intérêt",
    "Une expérience cinématographique unique, à voir absolument",
    "Production amateur avec des effets spéciaux ratés et des dialogues ridicules"
]

X_nouveau = vectorizer.transform(nouvelles_critiques).toarray()
predictions = mnb.predict(X_nouveau)
print(predictions)
print("\nPrédictions pour les nouvelles critiques:")
for i, (critique, pred) in enumerate(zip(nouvelles_critiques, predictions)):
    print(f"\nCritique: {critique}")
    print(f"Prédiction: {'positif' if pred == 1 else 'négatif'}")
