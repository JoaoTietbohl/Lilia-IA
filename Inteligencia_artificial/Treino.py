import pandas as pd
import nltk as nl
from dados.Categorias import categorias
from dados.Textos import textos
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import RSLPStemmer

cat = []
cat = cat + categorias

y = [x for x in categorias for _ in range(len(textos) // len(categorias))]

tex = []
tex = tex + textos

#corte
genericas = [
    'a', 'o', 'no', 'na', 'em', 'para', 'de', 'um', 'das', 'dos', 'do', 'da', 'deu',
    'com', 'se', 'por', 'uma', 'os', 'as', 'mais', 'foi', 'ser', 'tem', 'ter',
    'favor', 'solicito', 'precisamos', 'necessario', 'realizar', 'enviar',
    'esta', 'estao', 'pode', 'pode', 'querendo', 'nao', 'sim', 'ainda',
]

vectorizer = TfidfVectorizer(stop_words=genericas,ngram_range=(1, 2))

#o fit le todos os textos e cria um dicioanrio a parte do trasnforme conta quantas vezes cada dado apareceu
matriz_numerica = vectorizer.fit_transform(tex)

vocabulario = vectorizer.vocabulary_
colunas = vectorizer.get_feature_names_out()

df_resultado = pd.DataFrame(matriz_numerica.toarray(), columns=colunas)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn.fit(matriz_numerica, y)

def realizar_predicao(texto_do_usuario):

    novo_matriz = vectorizer.transform([texto_do_usuario])

    previsao = knn.predict(novo_matriz)

    probabilidades = knn.predict_proba(novo_matriz)

    porcentagem = max(probabilidades[0]) * 100

    return previsao[0], porcentagem






