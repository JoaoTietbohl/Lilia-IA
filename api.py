from fastapi import FastAPI
from Inteligencia_artificial.Treino import realizar_predicao
from pydantic import BaseModel
app = FastAPI()

"""
# ROTAS DA API

# 1 -- GET / (Boas-vindas)
# 2 -- POST /classificar (Recebe o prompt e retorna o resultado)
# 3 -- POST /feedback (Recebe se clicou em OK ou Tentar de Novo)
"""

class Chamado(BaseModel):
    texto: str

@app.post("/classificar")
async def classificar_chamado(item: Chamado):
    resultado, certeza = realizar_predicao(item.texto)

    return {"categoria": resultado, "certeza": certeza}
