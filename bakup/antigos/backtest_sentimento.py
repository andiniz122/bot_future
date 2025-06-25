import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIGURAÇÃO DO BACKTEST ====
ALAVANCAGEM = 125
RISCO_POR_TRADE = 0.10        # 10% do capital por operação
TAKE_PROFIT_PERCENT = 0.02    # 2% de lucro sobre o valor alocado
SENTIMENTO_MINIMO = 85        # Compra quando sentimento >= 85%
CAPITAL_INICIAL = 300
PERIODO_DIAS = 90             # Histórico em dias
INTERVALO = "1h"              # Intervalo do candle (1h)

# ==== COLETAR DADOS DO BTC ====
df = yf.download("BTC-USD", period=f"{PERIODO_DIAS}d", interval=INTERVALO)
df.dropna(inplace=True)
df.reset_index(inplace=True)

# Foco nos últimos 30% do período
df = df.tail(int(len(df) * 0.30)).copy()

# Simular sentimento com valores entre 70% e 95%
np.random.seed(42)
df['sentimento'] = np.random.uniform(70, 95, size=len(df))

# ==== LÓGICA DO BACKTEST ====
capital = CAPITAL_INICIAL
historico = []

for i in range(len(df) - 1):
    candle = df.iloc[i]
    sentimento = df.loc[i, 'sentimento']
    preco_entrada = candle['Close']

    if sentimento >= SENTIMENTO_MINIMO:
        valor_usdt = capital * RISCO_POR_TRADE
        tamanho_btc = (valor_usdt * ALAVANCAGEM) / preco_entrada
        alvo_preco = preco_entrada * (1 + (TAKE_PROFIT_PERCENT / ALAVANCAGEM))

        # Verifica se no próximo candle atinge o alvo
        candle_prox = df.iloc[i + 1]
        if candle_prox['High'] >= alvo_preco:
            lucro = valor_usdt * TAKE_PROFIT_PERCENT
            capital += lucro
            historico.append({
                'data': candle.name,
                'preco_entrada': preco_entrada,
                'alvo_preco': alvo_preco,
                'lucro': lucro,
                'capital_total': capital,
                'sentimento': sentimento
            })

# ==== RESULTADOS ====
resultados = pd.DataFrame(historico)
print(f"\n📊 Total de operações: {len(resultados)}")
print(f"💰 Capital final: {capital:.2f} USDT")
print(f"📈 Lucro total: {capital - CAPITAL_INICIAL:.2f} USDT")

# ==== VISUALIZAÇÃO ====
plt.plot(resultados['capital_total'])
plt.title("Evolução do Capital com Backtest")
plt.xlabel("Operações")
plt.ylabel("Capital USDT")
plt.grid(True)
plt.show()
