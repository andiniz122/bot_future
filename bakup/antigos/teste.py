#!/usr/bin/env python3
"""
Script Corrigido - Gate.io Futures API
Processa corretamente o formato JSON da API
"""

import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime
import numpy as np

async def test_gateio_futures_corrected():
    """
    Testa a API Gate.io com o formato correto dos dados
    """
    
    print("🚀 Teste CORRIGIDO da Gate.io Futures API")
    print("=" * 60)
    
    base_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
    symbol = "BTC_USDT"
    interval = "1m"
    limit = 50
    
    params = {
        "contract": symbol,
        "interval": interval, 
        "limit": limit
    }
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'TradingDashboard-Test/1.0'
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            print(f"🔍 Buscando dados de {symbol}...")
            
            async with session.get(base_url, params=params, headers=headers) as response:
                if response.status != 200:
                    print(f"❌ Erro: Status {response.status}")
                    return False
                
                raw_data = await response.json()
                print(f"✅ Recebidos {len(raw_data)} candles")
                print("=" * 60)
                
                # Mostrar formato real dos dados
                print("📋 Exemplo de 1 candle (formato real):")
                print(json.dumps(raw_data[0], indent=2))
                print("=" * 60)
                
                # Processar dados corretamente
                print("🔄 Processando dados com formato correto...")
                
                processed_data = []
                for candle in raw_data:
                    processed_candle = {
                        'timestamp': int(candle['t']),
                        'open': float(candle['o']),
                        'high': float(candle['h']),
                        'low': float(candle['l']),
                        'close': float(candle['c']),
                        'volume': int(candle['v']),
                        'volume_value': float(candle['sum'])
                    }
                    processed_data.append(processed_candle)
                
                # Criar DataFrame com dados corretos
                df = pd.DataFrame(processed_data)
                
                print("📊 DataFrame criado:")
                print(df.head())
                print(f"Shape: {df.shape}")
                print(f"Colunas: {list(df.columns)}")
                print("=" * 60)
                
                # Converter timestamp para datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
                
                # Renomear para padrão OHLCV
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Ordenar por data (mais antigo primeiro)
                df = df.sort_index()
                
                print("✅ DataFrame processado:")
                print(df[['Open', 'High', 'Low', 'Close', 'Volume']].head())
                print("=" * 60)
                
                # Calcular cores do volume
                print("🎨 Calculando cores do volume...")
                df['price_direction'] = 'neutral'
                
                for i in range(1, len(df)):
                    current_close = df.iloc[i]['Close']
                    previous_close = df.iloc[i-1]['Close']
                    
                    if current_close > previous_close:
                        df.iloc[i, df.columns.get_loc('price_direction')] = 'up'
                    elif current_close < previous_close:
                        df.iloc[i, df.columns.get_loc('price_direction')] = 'down'
                
                volume_colors = df['price_direction'].value_counts()
                print(f"✅ Cores calculadas: {dict(volume_colors)}")
                print("=" * 60)
                
                # Simular MACD (precisaria do talib para o real)
                print("📈 Calculando MACD simulado...")
                if len(df) >= 26:
                    df['EMA12'] = df['Close'].ewm(span=12).mean()
                    df['EMA26'] = df['Close'].ewm(span=26).mean()
                    df['MACD'] = df['EMA12'] - df['EMA26']
                    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
                    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
                    
                    print("✅ MACD calculado!")
                    print(df[['Close', 'MACD', 'MACD_Signal', 'MACD_Hist']].tail())
                else:
                    print(f"⚠️ Poucos dados para MACD: {len(df)} (precisa >= 26)")
                
                print("=" * 60)
                
                # Preparar dados para o frontend (formato final)
                frontend_data = {
                    "symbol": symbol,
                    "interval": interval,
                    "data_points": len(df),
                    "dates": [dt.isoformat() for dt in df.index],
                    "price_data": df['Close'].tolist(),
                    "volume_data": df['Volume'].tolist(),
                    "volume_colors": df['price_direction'].tolist(),
                    "open_data": df['Open'].tolist(),
                    "high_data": df['High'].tolist(),
                    "low_data": df['Low'].tolist(),
                    "latest_price": float(df['Close'].iloc[-1]),
                    "latest_volume": int(df['Volume'].iloc[-1]),
                    "price_change": float(df['Close'].iloc[-1] - df['Close'].iloc[-2]) if len(df) > 1 else 0,
                    "price_change_pct": float(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100) if len(df) > 1 else 0
                }
                
                # Adicionar MACD se calculado
                if 'MACD' in df.columns:
                    frontend_data.update({
                        "macd_data": df['MACD'].fillna(0).tolist(),
                        "macd_signal_data": df['MACD_Signal'].fillna(0).tolist(),
                        "macd_hist_data": df['MACD_Hist'].fillna(0).tolist()
                    })
                else:
                    frontend_data.update({
                        "macd_data": [],
                        "macd_signal_data": [],
                        "macd_hist_data": []
                    })
                
                print("📤 Dados finais para frontend:")
                print(f"📊 Pontos de dados: {frontend_data['data_points']}")
                print(f"💰 Preço atual: ${frontend_data['latest_price']:,.2f}")
                print(f"📈 Volume atual: {frontend_data['latest_volume']:,}")
                print(f"📈 Variação: {frontend_data['price_change']:+.2f} ({frontend_data['price_change_pct']:+.2f}%)")
                print(f"🎨 Cores: {dict(pd.Series(frontend_data['volume_colors']).value_counts())}")
                
                # Mostrar sample dos dados (primeiros 3 pontos)
                print("\n📋 Sample dos dados (primeiros 3 pontos):")
                sample_data = {
                    "dates": frontend_data["dates"][:3],
                    "prices": frontend_data["price_data"][:3],
                    "volumes": frontend_data["volume_data"][:3],
                    "colors": frontend_data["volume_colors"][:3]
                }
                print(json.dumps(sample_data, indent=2))
                
                print("=" * 60)
                print("🎉 SUCESSO! Dados processados corretamente!")
                print(f"✅ {len(df)} candles válidos de {symbol}")
                print(f"💰 BTC: ${frontend_data['latest_price']:,.2f}")
                print(f"📊 Volume: {frontend_data['latest_volume']:,}")
                
                return frontend_data
                
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

async def test_integration_format():
    """
    Testa o formato que será usado na integração
    """
    print("\n🔧 TESTANDO FORMATO DE INTEGRAÇÃO")
    print("=" * 60)
    
    # Simular a função que será usada no backend
    async def fetch_gateio_ohlcv_corrected(symbol_pair: str, interval: str, limit: int = 200):
        base_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
        params = {
            "contract": symbol_pair,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'TradingDashboard/1.0'
        }

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(base_url, params=params, headers=headers) as response:
                    if response.status != 200:
                        print(f"❌ API Error: Status {response.status}")
                        return pd.DataFrame()

                    raw_data = await response.json()
                    
                    if not raw_data:
                        print(f"⚠️ No data returned for {symbol_pair}")
                        return pd.DataFrame()

                    # Processar dados corretamente
                    processed_data = []
                    for candle in raw_data:
                        processed_data.append({
                            'Open': float(candle['o']),
                            'High': float(candle['h']),
                            'Low': float(candle['l']),
                            'Close': float(candle['c']),
                            'Volume': int(candle['v']),
                            'timestamp': pd.to_datetime(int(candle['t']), unit='s')
                        })

                    df = pd.DataFrame(processed_data)
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    
                    print(f"✅ Processados {len(df)} candles para {symbol_pair}")
                    return df

        except Exception as e:
            print(f"❌ Error fetching {symbol_pair}: {e}")
            return pd.DataFrame()
    
    # Testar a função corrigida
    test_symbols = ["BTC_USDT"]
    test_intervals = ["1m", "1h"]
    
    for symbol in test_symbols:
        for interval in test_intervals:
            print(f"\n🧪 Testando {symbol} - {interval}")
            df = await fetch_gateio_ohlcv_corrected(symbol, interval, 20)
            
            if not df.empty:
                print(f"   ✅ {len(df)} candles recebidos")
                print(f"   💰 Último preço: ${df['Close'].iloc[-1]:,.2f}")
                print(f"   📊 Último volume: {df['Volume'].iloc[-1]:,}")
            else:
                print(f"   ❌ Falha ao buscar dados")
    
    return True

if __name__ == "__main__":
    print("🧪 TESTE CORRIGIDO DA GATE.IO FUTURES API")
    print("Usando o formato real dos dados da API...")
    print()
    
    # Teste principal
    result = asyncio.run(test_gateio_futures_corrected())
    
    if result:
        # Teste de integração
        asyncio.run(test_integration_format())
        
        print("\n" + "="*60)
        print("🎯 PRÓXIMOS PASSOS:")
        print("1. ✅ API funcionando com formato correto")
        print("2. 🔧 Atualizar função fetch_gateio_ohlcv no backend")
        print("3. 🧪 Testar endpoint /api/precos/1d")
        print("4. 🎨 Verificar gráficos no frontend")
        print("5. 📊 Integrar MACD real (talib)")
        print("="*60)
    else:
        print("\n❌ Teste falhou.")