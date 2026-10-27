import streamlit as st
import numpy as np
import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    GlobalAveragePooling1D
)
from sklearn.preprocessing import Normalizer

# =====================================
# (n) Personnalisation de Streamlit
# =====================================

st.set_page_config(
    page_title="Word2Vec - DataScientest",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Modèle Word2Vec")
st.write(
    "Exploration des relations sémantiques "
    "entre les mots grâce au Deep Learning."
)

# =====================================
# (h) Création de l'application Streamlit
# (j) Chargement des poids du modèle
# =====================================

vocab_size = 10000
embedding_dim = 300

@st.cache_resource
def charger_modele():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(vocab_size, activation='softmax'))

    model.build(input_shape=(None, 4))
    model.load_weights("word2vec.h5")

    return model

model = charger_modele()

st.success("Modèle Word2Vec chargé avec succès !")

# =====================================
# Chargement du Tokenizer
# =====================================

@st.cache_resource
def charger_tokenizer():
    with open("tokenizer_word2vec.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

tokenizer = charger_tokenizer()

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

# =====================================
# (k) Extraction des embeddings
# =====================================

vectors = model.layers[0].trainable_weights[0].numpy()

def dot_product(vec1, vec2):
    return np.sum((vec1 * vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(
        dot_product(vec1, vec1) *
        dot_product(vec2, vec2)
    )

def find_closest(word_index, vectors, number_closest):
    list1 = []
    query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])

    return np.asarray(
        sorted(list1, reverse=True)[:number_closest]
    )

def compare(
    index_word1,
    index_word2,
    index_word3,
    vectors,
    number_closest
):
    list1 = []

    query_vector = (
        vectors[index_word1]
        - vectors[index_word2]
        + vectors[index_word3]
    )

    normalizer = Normalizer()

    query_vector = normalizer.fit_transform(
        [query_vector], 'l2'
    )

    query_vector = query_vector[0]

    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])

    return np.asarray(
        sorted(list1, reverse=True)[:number_closest]
    )

def print_closest(word, number=10):
    index_closest_words = find_closest(
        word2idx[word],
        vectors,
        number
    )

    for index_word in index_closest_words:
        print(
            idx2word[int(index_word[1])],
            " -- ",
            index_word[0]
        )

# =====================================
# (l) Recherche de mots similaires
# =====================================

st.header("1. Recherche de mots similaires")

st.write(
    "Saisissez un mot anglais pour découvrir "
    "les mots les plus proches."
)

mot = st.text_input(
    "Mot à rechercher",
    value="zombie"
)

nombre = st.slider(
    "Nombre de mots proches",
    min_value=1,
    max_value=20,
    value=10
)

if st.button("Rechercher les mots proches"):

    mot = mot.lower().strip()

    if mot not in word2idx:
        st.warning("Ce mot n'existe pas dans le dictionnaire.")

    elif word2idx[mot] >= vocab_size:
        st.warning(
            "Ce mot ne fait pas partie des 10 000 "
            "mots utilisés pendant l'entraînement."
        )

    else:
        index_closest_words = find_closest(
            word2idx[mot],
            vectors,
            nombre
        )

        st.subheader("Résultats")

        for index_word in index_closest_words:
            index = int(index_word[1])
            score = index_word[0]

            st.write(
                f"**{idx2word[index]}** : {score:.4f}"
            )

# =====================================
# (m) Arithmétique entre les mots
# =====================================

st.divider()

st.header("2. Arithmétique des mots")

st.write(
    "Effectuez une opération entre trois mots "
    "pour découvrir leurs relations sémantiques."
)

st.info("Opération : Mot 1 − Mot 2 + Mot 3")

col1, col2, col3 = st.columns(3)

with col1:
    mot1 = st.text_input("Mot 1", value="king")

with col2:
    mot2 = st.text_input("Mot 2", value="man")

with col3:
    mot3 = st.text_input("Mot 3", value="woman")

nombre_compare = st.slider(
    "Nombre de résultats",
    min_value=1,
    max_value=20,
    value=10
)

if st.button("Comparer les mots"):

    mot1 = mot1.lower().strip()
    mot2 = mot2.lower().strip()
    mot3 = mot3.lower().strip()

    mots = [mot1, mot2, mot3]

    if any(mot not in word2idx for mot in mots):
        st.warning(
            "Un ou plusieurs mots sont absents "
            "du dictionnaire."
        )

    elif any(word2idx[mot] >= vocab_size for mot in mots):
        st.warning(
            "Un ou plusieurs mots ne font pas "
            "partie du vocabulaire entraîné."
        )

    else:
        resultats = compare(
            word2idx[mot1],
            word2idx[mot2],
            word2idx[mot3],
            vectors,
            nombre_compare
        )

        st.subheader("Résultats de la comparaison")

        st.write(
            f"**{mot1} − {mot2} + {mot3}**"
        )

        for resultat in resultats:
            score = resultat[0]
            index = int(resultat[1])

            st.write(
                f"**{idx2word[index]}** : {score:.4f}"
            )

# =====================================
# (n) Informations sur l'application
# =====================================

st.divider()

st.caption(
    "Projet Word2Vec CBOW - Formation DataScientest"
)

st.caption(
    "Modèle entraîné sur 25 000 critiques de films "
    "avec des embeddings de 300 dimensions."
)
