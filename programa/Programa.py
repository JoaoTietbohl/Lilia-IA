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

y = [x for x in cat for _ in range(16)]

tex = []
tex = tex + textos

#Funçao da magia
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

resultado = []
problema = input('Digite seu problema: ')

resultado.append(problema)

novo_matriz = vectorizer.transform(resultado)

previsao = knn.predict(novo_matriz)
print(f"O sistema classificou seu problema como: {previsao[0]}")

probabilidades = knn.predict_proba(novo_matriz)

maior_probabilidade = max(probabilidades[0])

porcentagem = maior_probabilidade * 100

print(f"Certeza do modelo: {porcentagem:.2f}%")








