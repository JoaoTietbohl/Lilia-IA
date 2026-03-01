import streamlit as st
from Inteligencia_artificial.Treino import realizar_predicao

st.set_page_config(page_title="Lilia IA", page_icon="🤖")

st.title("🤖 Lilia - Classificador de Chamados")
st.markdown("Digite o problema abaixo para que a IA identifique a categoria.")

prompt = st.text_input("Descreva o problema:", placeholder="Ex: Meu monitor não liga...")

if st.button("Classificar Agora"):
    if prompt:

        categoria, confianca = realizar_predicao(prompt)

        st.divider()
        st.subheader(f"Categoria sugerida: :blue[{categoria}]")


        st.write(f"Confiança do modelo: {confianca:.2f}%")
        st.progress(confianca / 100)
    else:
        st.warning("Por favor, digite algo antes de classificar.")

if st.button("Sugerir frase"):
    if prompt:
        print('mensagem')
    else:
        st.subheader("Sugerir frase")
