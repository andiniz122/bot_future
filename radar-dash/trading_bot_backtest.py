#!/usr/bin/env python3
"""
🤖 BOT DE TRADING COM IA LEVE v2.1
Integração: Machine Learning + Análise de Sentimento + Padrões Históricos + Calendário FRED + CryptoPanic
Estratégia: SuperTrend + VWAP + RSI + Volume + Multi-Timeframe + Filtros IA
"""

import os
import time
import json
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
import logging
import ta
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from collections import defaultdict, deque

# ============================================================================
# 🧠 IMPORTAÇÕES DE MACHINE LEARNING E DADOS
# ============================================================================

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

load_dotenv()

# ============================================================================
# 🔧 CONFIGURAÇÕES DA API GATE.IO, NEWSAPI, FRED E CRYPTOPANIC
# ============================================================================

API_KEY = os.getenv('GATE_TESTNET_API_KEY') or os.getenv('GATE_API_KEY')
SECRET = os.getenv('GATE_TESTNET_API_SECRET') or os.getenv('GATE_API_SECRET')
ENVIRONMENT = os.getenv('GATE_ENVIRONMENT', 'testnet')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Importa FRED_API_KEY e CRYPTOPANIC_API_KEY do config.py
try:
    from config import FRED_API_KEY
except ImportError:
    FRED_API_KEY = "DUMMY_KEY_FRED"
    logging.getLogger(__name__).warning("FRED_API_KEY não encontrada em config.py. Funcionalidade do FRED será limitada/inativa.")

try:
    from config import CRYPTOPANIC_API_KEY
except ImportError:
    CRYPTOPANIC_API_KEY = "DUMMY_KEY_CRYPTOPANIC"
    logging.getLogger(__name__).warning("CRYPTOPANIC_API_KEY não encontrada em config.py. Funcionalidade da CryptoPanic será limitada/inativa.")


def get_base_urls():
    """Retorna URLs baseado no ambiente"""
    if ENVIRONMENT == 'testnet':
        return {
            'rest': 'https://api-testnet.gateapi.io',
            'ws': 'wss://ws-testnet.gate.io/v4/ws/futures/usdt'
        }
    else:
        return {
            'rest': 'https://fx-api.gateio.ws',
            'ws': 'wss://fx-ws.gateio.ws/v4/ws/usdt'
        }

def sign_request(method: str, endpoint: str, query_string: str = '', body: str = '') -> dict:
    """Gera headers assinados para autenticação"""
    timestamp = str(int(time.time()))
    
    if not endpoint.startswith('/api/v4'):
        endpoint_for_signature = '/api/v4' + endpoint
    else:
        endpoint_for_signature = endpoint
    
    message = f"{method}\n{endpoint_for_signature}\n{query_string}\n{hashlib.sha512(body.encode('utf-8')).hexdigest()}\n{timestamp}"
    
    signature = hmac.new(
        SECRET.encode('utf-8'), 
        message.encode('utf-8'), 
        hashlib.sha512
    ).hexdigest()
    
    headers = {
        'Content-Type': 'application/json',
        'KEY': API_KEY,
        'Timestamp': timestamp,
        'SIGN': signature
    }
    
    return headers

def _convert_gateio_symbol_to_yfinance(symbol: str) -> str:
    """Converts Gate.io symbol (e.g., BTC_USDT) to yfinance format (e.g., BTC-USD)."""
    parts = symbol.split('_')
    if len(parts) == 2:
        return f"{parts[0]}-USD" # yfinance usually uses USD, not USDT
    return symbol # Return as is if format is unexpected

# Símbolos para trading
TRADING_SYMBOLS = ['BTC_USDT', 'ETH_USDT']

# ============================================================================
# ⚙️ CONFIGURAÇÕES DO SISTEMA
# ============================================================================

# Configurações específicas para IA
AI_CONFIG = {
    'model_retrain_interval_hours': 24,    # Retreinar modelo a cada 24h
    'min_training_samples': 100,           # Mínimo de trades para treinar
    'feature_window': 20,                  # Janela de features (20 candles)
    'prediction_threshold': 0.65,          # Confiança mínima da IA (65%)
    'pattern_memory_size': 1000,           # Quantos padrões manter em memória
    'sentiment_weight': 0.15,              # Peso do sentimento (15%)
    'model_save_path': 'models/',          # Pasta para salvar modelos
    'fred_impact_penalty_factor': 0.20,    # Penalidade na confiança se houver evento FRED alto impacto
    'fred_cooldown_minutes_high_impact': 60, # Cooldown em minutos após evento de alto impacto FRED
}

# Configurações de risco por ambiente
RISK_CONFIG = {
    'testnet': {
        'position_size_usdt': 15,
        'max_open_positions': 2,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 4.0,
        'max_daily_trades': 10,
        'cooldown_minutes': 10,
        'trailing_stop_percent': 1.2,
        'reversal_fall_pct': 1.0,
        'reversal_rise_pct': 1.0,
        'reversal_volume_multiplier': 2.5,
        'reversal_cooldown_minutes': 45,
        'max_correlation': 0.8,
        'min_volume_ratio': 1.5,
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'ai_confidence_boost': 0.20          # Boost de confiança da IA
    },
    'live': {
        'position_size_usdt': 30,
        'max_open_positions': 1,
        'stop_loss_atr_multiplier': 1.5,
        'take_profit_atr_multiplier': 3.0,
        'max_daily_trades': 6,
        'cooldown_minutes': 20,
        'trailing_stop_percent': 0.8,
        'reversal_fall_pct': 1.5,
        'reversal_rise_pct': 1.5,
        'reversal_volume_multiplier': 3.0,
        'reversal_cooldown_minutes': 90,
        'max_correlation': 0.6,
        'min_volume_ratio': 2.0,
        'rsi_oversold': 20,
        'rsi_overbought': 80,
        'ai_confidence_boost': 0.15          # Mais conservador em live
    }
}

# ============================================================================
# 🧠 CLASSES DE MACHINE LEARNING
# ============================================================================

@dataclass
class AIFeatures:
    """Features para o modelo de ML"""
    # Indicadores técnicos
    rsi: float
    supertrend_dir: int  # 1 ou -1
    vwap_distance: float  # % distância do VWAP
    volume_ratio: float
    atr_normalized: float
    bollinger_position: float # Posição em relação às bandas de Bollinger (0-1)
    macd_signal_diff: float # Diferença entre MACD e linha de sinal

    # Features de candles
    candle_size: float
    upper_shadow: float
    lower_shadow: float
    
    # Features de contexto
    trend_strength: float
    volatility_regime: int  # 0=baixa, 1=normal, 2=alta
    time_of_day: int  # Hora do dia (0-23)
    
    # Features de momentum
    price_momentum_5: float    # Momentum 5 períodos
    price_momentum_10: float  # Momentum 10 períodos
    volume_momentum: float
    
    # Features de padrão
    pattern_score: float  # Score do pattern matcher
    
    def to_array(self) -> np.ndarray:
        """Converte para array numpy para ML"""
        return np.array([
            self.rsi, self.supertrend_dir, self.vwap_distance,
            self.volume_ratio, self.atr_normalized, self.bollinger_position,
            self.macd_signal_diff, self.candle_size, self.upper_shadow,
            self.lower_shadow, self.trend_strength, self.volatility_regime,
            self.time_of_day, self.price_momentum_5, self.price_momentum_10,
            self.volume_momentum, self.pattern_score
        ])

class PatternMatcher:
    """Detecção de padrões históricos com scoring"""
    
    def __init__(self, memory_size: int = 1000):
        self.patterns = defaultdict(list)  # {pattern_hash: [outcomes]}
        self.memory_size = memory_size
        
    def extract_pattern(self, df: pd.DataFrame) -> str:
        """Extrai padrão dos últimos candles"""
        if len(df) < 5:
            return "insufficient_data"
        
        recent = df.tail(5)
        
        # Padrão baseado em direção dos candles e volume
        directions = []
        volumes = []
        
        for i in range(len(recent)):
            candle = recent.iloc[i]
            direction = 'up' if candle['close'] > candle['open'] else 'down'
            vol_level = 'high' if candle['volume'] > recent['volume'].mean() else 'low'
            directions.append(direction)
            volumes.append(vol_level)
        
        pattern = f"{'_'.join(directions)}__{'_'.join(volumes)}"
        return pattern
    
    def add_outcome(self, pattern: str, outcome: bool):
        """Adiciona resultado de um padrão"""
        self.patterns[pattern].append(1 if outcome else 0)
        
        # Limitar memória
        if len(self.patterns[pattern]) > self.memory_size:
            self.patterns[pattern] = self.patterns[pattern][-self.memory_size:]
    
    def get_pattern_score(self, pattern: str) -> float:
        """Retorna score de sucesso do padrão (0-1)"""
        if pattern not in self.patterns or len(self.patterns[pattern]) < 5:
            return 0.5  # Neutro para padrões novos
        
        outcomes = self.patterns[pattern]
        success_rate = sum(outcomes) / len(outcomes)
        
        # Ajustar confiança baseado no número de amostras
        confidence_factor = min(1.0, len(outcomes) / 20)  # Máxima confiança com 20+ amostras
        
        return 0.5 + (success_rate - 0.5) * confidence_factor

class SentimentAnalyzer:
    """Análise de sentimento simplificada, agora com múltiplas fontes de notícias."""
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_expiry = timedelta(minutes=15) # Cache sentiment for 15 minutes to be more responsive
        self.last_update = {}
        self.logger = logging.getLogger('SentimentAnalyzer')
        self.cryptopanic_api_key = CRYPTOPANIC_API_KEY
        
    def get_crypto_news_sentiment(self, symbol: str) -> float:
        """
        Obtém sentimento das notícias de várias fontes relevantes.
        Prioriza Wall Street Journal, notícias de negócios dos EUA e CryptoPanic.
        """
        cache_key = f"{symbol}_sentiment_multi_source"
        now = datetime.now()
        
        # Verificar cache
        if (cache_key in self.sentiment_cache and 
            cache_key in self.last_update and
            (now - self.last_update[cache_key]) < self.cache_expiry):
            self.logger.debug(f"Sentimento para {symbol} do cache.")
            return self.sentiment_cache[cache_key]
        
        sentiment_scores = []
        
        # 1. Sentimento da CryptoPanic (focado em cripto) - Alta prioridade
        if self.cryptopanic_api_key and self.cryptopanic_api_key != "DUMMY_KEY_CRYPTOPANIC":
            cryptopanic_text = self._fetch_news_from_cryptopanic(symbol.split('_')[0])
            if cryptopanic_text:
                cryptopanic_sentiment = self.analyze_text_sentiment_real(cryptopanic_text)
                self.logger.info(f"Sentimento CryptoPanic para {symbol}: {cryptopanic_sentiment:.3f}")
                sentiment_scores.append(cryptopanic_sentiment * 1.2) # Give slightly more weight to direct crypto news
            else:
                self.logger.debug(f"Nenhuma notícia da CryptoPanic encontrada para {symbol} ou falha na API.")


        # 2. Sentimento do Wall Street Journal (WSJ)
        search_query_wsj = symbol.split('_')[0] if symbol in TRADING_SYMBOLS else "crypto"
        wsj_text = self._fetch_news_from_newsapi(q=search_query_wsj, domains="wsj.com")
        if wsj_text:
            wsj_sentiment = self.analyze_text_sentiment_real(wsj_text)
            self.logger.info(f"Sentimento WSJ para {symbol}: {wsj_sentiment:.3f}")
            sentiment_scores.append(wsj_sentiment)
        else:
            self.logger.debug(f"Nenhuma notícia do WSJ encontrada para {symbol} ou falha na API.")

        # 3. Sentimento de Negócios dos EUA
        search_query_us_biz = symbol.split('_')[0] if symbol in TRADING_SYMBOLS else "crypto"
        us_business_text = self._fetch_news_from_newsapi(country="us", category="business", q=search_query_us_biz)
        if us_business_text:
            us_business_sentiment = self.analyze_text_sentiment_real(us_business_text)
            self.logger.info(f"Sentimento de Negócios EUA para {symbol}: {us_business_sentiment:.3f}")
            sentiment_scores.append(us_business_sentiment)
        else:
            self.logger.debug(f"Nenhuma notícia de negócios dos EUA encontrada para {symbol} ou falha na API.")

        # 4. Sentimento geral de cripto (como fallback/complemento)
        general_crypto_text = self._fetch_news_from_newsapi(q=symbol.split('_')[0] + " crypto")
        if general_crypto_text:
            general_crypto_sentiment = self.analyze_text_sentiment_real(general_crypto_text)
            self.logger.info(f"Sentimento Geral Cripto para {symbol}: {general_crypto_sentiment:.3f}")
            sentiment_scores.append(general_crypto_sentiment)
        else:
            self.logger.debug(f"Nenhuma notícia geral de cripto encontrada para {symbol} ou falha na API.")

        # Calcular sentimento médio, ou padrão se não houver dados
        final_sentiment = 0.5
        if sentiment_scores:
            final_sentiment = np.mean(sentiment_scores)
        else:
            self.logger.warning(f"Nenhuma notícia de sentimento real capturada para {symbol}. Usando sentimento padrão (0.5).")
            # Adicionar uma pequena aleatoriedade para simular variação mesmo sem dados
            import random
            final_sentiment = 0.5 + random.uniform(-0.05, 0.05) 
        
        self.sentiment_cache[cache_key] = final_sentiment
        self.last_update[cache_key] = now
        
        return final_sentiment
    
    def _fetch_news_from_cryptopanic(self, currency_code: str, limit: int = 10) -> str:
        """
        Busca notícias de criptomoedas da CryptoPanic API.
        currency_code deve ser um código de moeda (ex: "BTC", "ETH").
        """
        if not self.cryptopanic_api_key or self.cryptopanic_api_key == "DUMMY_KEY_CRYPTOPANIC":
            self.logger.warning("CRYPTOPANIC_API_KEY não configurada. Não é possível buscar notícias da CryptoPanic.")
            return ""

        base_url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.cryptopanic_api_key,
            "currencies": currency_code,
            "public": "true",
            "kind": "news",        # Pode ser: news, media ou all
            "filter": "important", # Pode usar: bullish, bearish, hot, rising, etc.
            "limit": limit
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            all_text = " ".join([
                item.get("title", "") + " " + (item.get("url", "") or "") # CryptoPanic descriptions are often just URLs in API
                for item in data.get("results", [])
                if item.get("title")
            ])
            return all_text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao conectar com CryptoPanic API: {e}")
            return ""
        except json.JSONDecodeError:
            self.logger.error("Erro ao decodificar JSON da CryptoPanic API.")
            return ""
        except Exception as e:
            self.logger.error(f"Erro inesperado ao buscar notícias da CryptoPanic: {e}")
            return ""


    def _fetch_news_from_newsapi(self, q: str = None, domains: str = None, country: str = None, category: str = None) -> str:
        """
        Função auxiliar para buscar notícias da NewsAPI com base em parâmetros.
        Retorna uma string concatenada de títulos e descrições.
        """
        if not NEWS_API_KEY:
            self.logger.warning("NEWS_API_KEY não configurada. Não é possível buscar notícias reais.")
            return ""
        
        url = "https://newsapi.org/v2/everything" # Default to everything endpoint
        params = {
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt', # Always get most recent
            'pageSize': 20 # Get a reasonable number of articles
        }
        
        if q:
            params['q'] = q
        if domains:
            params['domains'] = domains
            url = "https://newsapi.org/v2/everything" # Explicitly use 'everything' endpoint for domains
        
        # If country and category are provided, use 'top-headlines' endpoint
        if country and category:
            params['country'] = country
            params['category'] = category
            url = "https://newsapi.org/v2/top-headlines"
            # For top-headlines, 'q' is optional. If only country/category, NewsAPI uses it.
            # If 'q' is also present, it's applied as a filter.
        elif country and not category: # If only country is specified (for top-headlines)
            params['country'] = country
            url = "https://newsapi.org/v2/top-headlines"

        # NewsAPI requires either 'q' or 'sources' or 'country' + 'category' for top-headlines
        # For 'everything', it requires 'q' or 'sources' or 'domains'.
        # Ensure at least one required parameter is present based on the chosen URL/endpoint.
        if url == "https://newsapi.org/v2/top-headlines":
            if not (country or params.get('q')): # For top-headlines, need country OR q
                 self.logger.warning("Parâmetros insuficientes para top-headlines NewsAPI.")
                 return ""
        elif url == "https://newsapi.org/v2/everything":
            if not (q or domains): # For everything, need q OR domains OR sources (sources not implemented here)
                self.logger.warning("Parâmetros insuficientes para everything NewsAPI.")
                return ""


        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            all_text = " ".join([
                article.get('title', '') + " " + (article.get('description', '') or '')
                for article in articles
                if article.get('title') or article.get('description')
            ])
            return all_text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao conectar com NewsAPI (URL: {url}, Params: {params}): {e}")
            return ""
        except json.JSONDecodeError:
            self.logger.error("Erro ao decodificar JSON da NewsAPI.")
            return ""
        except Exception as e:
            self.logger.error(f"Erro inesperado ao buscar notícias da NewsAPI: {e}")
            return ""
    
    def analyze_text_sentiment_real(self, text: str) -> float:
        """Analisa sentimento de texto usando TextBlob."""
        if not SENTIMENT_AVAILABLE or not text:
            return 0.5
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 a 1
            return (polarity + 1) / 2  # Converter para 0-1 (0 = muito negativo, 1 = muito positivo)
        except Exception as e:
            self.logger.error(f"Erro ao analisar sentimento de texto: {e}")
            return 0.5

class MLPredictor:
    """Sistema de Machine Learning para predição de sinais"""
    
    def __init__(self, environment: str):
        self.environment = environment
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [] # This could be populated dynamically later
        self.training_data = []
        self.last_retrain = None
        self.model_accuracy = 0.0
        
        # Configurar logger
        self.logger = logging.getLogger('AIPredictor')
        
        # Criar diretório para modelos
        os.makedirs(AI_CONFIG['model_save_path'], exist_ok=True)
        
        # Tentar carregar modelo existente
        self.load_model()
    
    def extract_features(self, df: pd.DataFrame, pattern_score: float = 0.5) -> Optional[AIFeatures]:
        """Extrai features dos dados OHLCV"""
        if len(df) < 20: # Ensure enough data for indicators
            return None
        
        try:
            current = df.iloc[-1]
            
            # Indicadores técnicos
            rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
            supertrend_dir = self.calculate_supertrend_direction(df)
            
            # VWAP
            vwap = self.calculate_vwap(df).iloc[-1]
            vwap_distance = ((current['close'] - vwap) / vwap) * 100 if vwap > 0 else 0.0
            
            # Volume
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current['volume'] / volume_ma if volume_ma > 0 else 1.0
            
            # ATR normalizado
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14).iloc[-1]
            atr_normalized = atr / current['close'] * 100 if current['close'] > 0 else 0.0
            
            # Bollinger Bands Position
            window = 20
            window_dev = 2
            bb = ta.volatility.BollingerBands(df['close'], window=window, window_dev=window_dev)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            
            bollinger_position = 0.5 # Default to middle if bands are too narrow or invalid
            if not pd.isna(bb_upper) and not pd.isna(bb_lower) and (bb_upper - bb_lower) > 0:
                bollinger_position = (current['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD Signal Difference
            macd = ta.trend.macd(df['close']).iloc[-1]
            macd_signal = ta.trend.macd_signal(df['close']).iloc[-1]
            macd_signal_diff = macd - macd_signal if not pd.isna(macd) and not pd.isna(macd_signal) else 0.0
            
            # Features de candle
            candle_size = abs(current['close'] - current['open']) / current['open'] * 100 if current['open'] > 0 else 0.0
            body_size = abs(current['close'] - current['open'])
            upper_shadow = (current['high'] - max(current['open'], current['close'])) / body_size if body_size > 0 else 0
            lower_shadow = (min(current['open'], current['close']) - current['low']) / body_size if body_size > 0 else 0
            
            # Trend strength
            ma10 = df['close'].rolling(10).mean()
            if len(ma10) >= 5 and ma10.iloc[-5] > 0:
                trend_strength = (ma10.iloc[-1] - ma10.iloc[-5]) / ma10.iloc[-5] * 100
            else:
                trend_strength = 0.0
            
            # Regime de volatilidade
            volatility_pct = df['close'].pct_change().std() * 100 if len(df['close'].pct_change().dropna()) > 0 else 0.0
            if volatility_pct < 1.0:
                volatility_regime = 0  # Baixa
            elif volatility_pct > 3.0:
                volatility_regime = 2  # Alta
            else:
                volatility_regime = 1  # Normal
            
            # Hora do dia
            time_of_day = datetime.now().hour
            
            # Momentum
            price_momentum_5 = ((current['close'] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100) if len(df) >= 6 and df['close'].iloc[-6] > 0 else 0
            price_momentum_10 = ((current['close'] - df['close'].iloc[-11]) / df['close'].iloc[-11] * 100) if len(df) >= 11 and df['close'].iloc[-11] > 0 else 0
            volume_momentum = ((current['volume'] - df['volume'].iloc[-6]) / df['volume'].iloc[-6] * 100) if len(df) >= 6 and df['volume'].iloc[-6] > 0 else 0
            
            return AIFeatures(
                rsi=rsi if not pd.isna(rsi) else 50.0,
                supertrend_dir=supertrend_dir,
                vwap_distance=vwap_distance if not pd.isna(vwap_distance) else 0.0,
                volume_ratio=volume_ratio,
                atr_normalized=atr_normalized if not pd.isna(atr_normalized) else 1.0,
                bollinger_position=bollinger_position if not pd.isna(bollinger_position) else 0.5,
                macd_signal_diff=macd_signal_diff if not pd.isna(macd_signal_diff) else 0.0,
                candle_size=candle_size,
                upper_shadow=upper_shadow,
                lower_shadow=lower_shadow,
                trend_strength=trend_strength,
                volatility_regime=volatility_regime,
                time_of_day=time_of_day,
                price_momentum_5=price_momentum_5,
                price_momentum_10=price_momentum_10,
                volume_momentum=volume_momentum,
                pattern_score=pattern_score
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair features: {e}", exc_info=True)
            return None
    
    def calculate_supertrend_direction(self, df: pd.DataFrame) -> int:
        """Calcula direção do SuperTrend"""
        try:
            st_indicator = ta.trend.supertrend(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=10,
                multiplier=3
            )
            
            direction_col = [col for col in st_indicator.columns if 'SUPERT_D' in col]
            
            if direction_col and not st_indicator[direction_col[0]].empty and not pd.isna(st_indicator[direction_col[0]].iloc[-1]):
                return int(st_indicator[direction_col[0]].iloc[-1])
            else:
                # Fallback to MA-based trend if SuperTrend cannot be calculated or is neutral
                if len(df) >= 20:
                    ma20 = df['close'].rolling(20).mean().iloc[-1]
                    if not pd.isna(ma20) and df['close'].iloc[-1] > ma20:
                        return 1
                    elif not pd.isna(ma20) and df['close'].iloc[-1] < ma20:
                        return -1
                return 0
        except Exception as e:
            self.logger.error(f"Erro ao calcular SuperTrend direction: {e}")
            return 0
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calcula VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        # Ensure volume is not zero to prevent division by zero
        if df['volume'].sum() == 0:
            return pd.Series([0.0] * len(df), index=df.index)
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def add_training_sample(self, features: AIFeatures, outcome: bool):
        """Adiciona amostra de treinamento"""
        self.training_data.append({
            'features': features.to_array(),
            'outcome': 1 if outcome else 0,
            'timestamp': datetime.now()
        })
        
        # Limitar tamanho dos dados de treinamento
        if len(self.training_data) > 5000:
            self.training_data = self.training_data[-5000:]
    
    def should_retrain(self) -> bool:
        """Verifica se deve retreinar o modelo"""
        if self.model is None:
            return len(self.training_data) >= AI_CONFIG['min_training_samples']
        
        if self.last_retrain is None:
            return True
        
        hours_since_retrain = (datetime.now() - self.last_retrain).total_seconds() / 3600
        return hours_since_retrain >= AI_CONFIG['model_retrain_interval_hours']
    
    def train_model(self) -> bool:
        """Treina o modelo de ML"""
        if not ML_AVAILABLE or len(self.training_data) < AI_CONFIG['min_training_samples']:
            self.logger.warning(f"Dados insuficientes para treinar: {len(self.training_data)} (mínimo: {AI_CONFIG['min_training_samples']})")
            return False
        
        try:
            self.logger.info(f"🧠 Iniciando treinamento com {len(self.training_data)} amostras...")
            
            # Preparar dados
            X = np.array([sample['features'] for sample in self.training_data])
            y = np.array([sample['outcome'] for sample in self.training_data])
            
            # Verificar se há variabilidade nos dados
            if len(np.unique(y)) < 2:
                self.logger.warning("Dados de treinamento sem variabilidade (apenas um tipo de resultado)")
                return False
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class balance
            )
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar múltiplos modelos e escolher o melhor
            models = {}
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            models['rf'] = {
                'model': rf_model,
                'accuracy': accuracy_score(y_test, rf_pred)
            }
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_test_scaled)
            models['gb'] = {
                'model': gb_model,
                'accuracy': accuracy_score(y_test, gb_pred)
            }
            
            # XGBoost (se disponível)
            if XGB_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    use_label_encoder=False, # Suppress warning
                    eval_metric='logloss'    # Suppress warning
                )
                xgb_model.fit(X_train_scaled, y_train)
                xgb_pred = xgb_model.predict(X_test_scaled)
                models['xgb'] = {
                    'model': xgb_model,
                    'accuracy': accuracy_score(y_test, xgb_pred)
                }
            
            # Escolher melhor modelo
            best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
            self.model = models[best_model_name]['model']
            self.model_accuracy = models[best_model_name]['accuracy']
            
            self.last_retrain = datetime.now()
            
            self.logger.info(f"✅ Modelo treinado ({best_model_name}): Acurácia {self.model_accuracy:.3f}")
            
            # Salvar modelo
            self.save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no treinamento: {e}", exc_info=True)
            return False
    
    def predict(self, features: AIFeatures) -> Tuple[float, float]:
        """Prediz probabilidade de sucesso do trade"""
        if self.model is None:
            return 0.5, 0.0  # Neutro se não há modelo
        
        try:
            # Preparar features
            X = features.to_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predição
            probabilities = self.model.predict_proba(X_scaled)[0]
            prob_success = probabilities[1] if len(probabilities) > 1 else 0.5
            
            # Confiança baseada na distância do threshold
            confidence = abs(prob_success - 0.5) * 2  # 0 a 1
            
            return prob_success, confidence
            
        except Exception as e:
            self.logger.error(f"Erro na predição: {e}")
            return 0.5, 0.0
    
    def save_model(self):
        """Salva modelo treinado"""
        try:
            model_path = os.path.join(AI_CONFIG['model_save_path'], f'model_{self.environment}.pkl')
            scaler_path = os.path.join(AI_CONFIG['model_save_path'], f'scaler_{self.environment}.pkl')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Salvar metadados
            metadata = {
                'accuracy': self.model_accuracy,
                'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
                'training_samples': len(self.training_data)
            }
            
            metadata_path = os.path.join(AI_CONFIG['model_save_path'], f'metadata_{self.environment}.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            self.logger.info(f"✅ Modelo salvo: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")
    
    def load_model(self):
        """Carrega modelo salvo"""
        try:
            model_path = os.path.join(AI_CONFIG['model_save_path'], f'model_{self.environment}.pkl')
            scaler_path = os.path.join(AI_CONFIG['model_save_path'], f'scaler_{self.environment}.pkl')
            metadata_path = os.path.join(AI_CONFIG['model_save_path'], f'metadata_{self.environment}.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_accuracy = metadata.get('accuracy', 0.0)
                    last_retrain_str = metadata.get('last_retrain')
                    if last_retrain_str:
                        self.last_retrain = datetime.fromisoformat(last_retrain_str)
                
                self.logger.info(f"✅ Modelo carregado: Acurácia {self.model_accuracy:.3f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
        
        return False

# ============================================================================
# 📊 SINAL DE TRADING ENHANCED
# ============================================================================

@dataclass
class EnhancedTradingSignal:
    """Sinal de trading com predições da IA"""
    symbol: str
    action: str
    base_confidence: float
    ai_probability: float
    ai_confidence: float
    final_confidence: float
    pattern_score: float
    sentiment_score: float
    features: AIFeatures
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    timestamp: datetime
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'action': self.action,
            'base_confidence': round(self.base_confidence, 3),
            'ai_probability': round(self.ai_probability, 3),
            'ai_confidence': round(self.ai_confidence, 3),
            'final_confidence': round(self.final_confidence, 3),
            'pattern_score': round(self.pattern_score, 3),
            'sentiment_score': round(self.sentiment_score, 3),
            'entry_price': round(self.entry_price, 4),
            'stop_loss': round(self.stop_loss, 4),
            'take_profit': round(self.take_profit, 4),
            'risk_reward': round(self.risk_reward, 2),
            'timestamp': self.timestamp.isoformat()
        }

# ============================================================================
# 📅 CALENDÁRIO ECONÔMICO FRED
# ============================================================================

@dataclass
class EconomicEvent:
    """Classe para representar um evento econômico com dados mais ricos."""
    release_id: str
    name: str
    date: datetime # Data do release
    time: Optional[str] # Horário estimado do release (ex: "08:30")
    importance: str  # HIGH, MEDIUM, LOW
    frequency: Optional[str] = None # Mensal, Semanal, etc.
    series_id: Optional[str] = None # ID da série FRED associada ao evento
    previous_value: Optional[float] = None
    forecast: Optional[float] = None # Difícil de obter do FRED, pode ser N/A
    actual: Optional[float] = None # Última observação da série
    impact_score: float = 0.0 # 0-100
    currency: str = "USD"
    category: str = "GENERAL"
    metadata: Dict[str, Any] = field(default_factory=dict) # Usa default_factory para mutáveis

class FREDEconomicCalendar:
    """
    Integração profissional com a API do FRED para calendário econômico.
    Foca em releases de alto impacto e busca as últimas observações.
    """
    
    def __init__(self, api_key: str):
        if not api_key or api_key == "DUMMY_KEY_FRED":
            logging.getLogger(__name__).warning("FRED API Key não fornecida ou é dummy. Calendário FRED inativo.")
            self.api_key = None # Desativa a API se a chave não for válida
        else:
            self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # Cache de releases e eventos para evitar múltiplas requisições da mesma informação
        self.cache: Dict[str, Any] = {
            "upcoming_events": [],
            "last_full_update": None, # Timestamp da última atualização completa
            "cache_duration_seconds": 21600 # Cache por 6 horas (21600 segundos) para eventos futuros
        }
        self.logger = logging.getLogger('FREDCalendar')

        # Para backtesting, permite sobrescrever datetime.now()
        self.datetime_now_override: Optional[datetime] = None
    
        # Mapeamento de releases críticos com seus FRED series IDs e scores de impacto
        # Estes IDs são os que usaremos para buscar os valores anteriores/atuais.
        self.critical_release_series: Dict[str, Dict[str, Any]] = {
            # Non-Farm Payrolls - NFP (Muitas séries para NFP, usando um proxy comum)
            "PAYEMS": { 
                "name": "Non-Farm Payrolls", "series_id": "PAYEMS", # Total Nonfarm Payrolls: All Employees, Seasonally Adjusted
                "impact_score": 95, "importance": "HIGH", "category": "EMPLOYMENT", "volatility_factor": 2.5
            },
            # Consumer Price Index (CPI)
            "CPIAUCSL": { 
                "name": "Consumer Price Index", "series_id": "CPIAUCSL", # CPI for All Urban Consumers: All Items, Seasonally Adjusted
                "impact_score": 90, "importance": "HIGH", "category": "INFLATION", "volatility_factor": 2.0
            },
            # Federal Funds Rate (Taxa de Juros)
            "FEDFUNDS": { 
                "name": "Federal Funds Rate", "series_id": "FEDFUNDS", # Federal Funds Effective Rate
                "impact_score": 98, "importance": "HIGH", "category": "MONETARY_POLICY", "volatility_factor": 3.0
            },
            # Gross Domestic Product (GDP)
            "GDP": { 
                "name": "Gross Domestic Product", "series_id": "GDP", # Gross Domestic Product
                "impact_score": 85, "importance": "HIGH", "category": "GROWTH", "volatility_factor": 1.8
            },
            # Unemployment Rate
            "UNRATE": { 
                "name": "Unemployment Rate", "series_id": "UNRATE", # Civilian Unemployment Rate
                "impact_score": 80, "importance": "HIGH", "category": "EMPLOYMENT", "volatility_factor": 1.5
            },
            # Personal Consumption Expenditures (PCE Price Index)
            "PCEPI": { 
                "name": "PCE Price Index", "series_id": "PCEPI", # Personal Consumption Expenditures Price Index
                "impact_score": 85, "importance": "HIGH", "category": "INFLATION", "volatility_factor": 1.7
            },
            # Produção Industrial
            "INDPRO": {
                "name": "Industrial Production Index", "series_id": "INDPRO", # Industrial Production Index, Seasonally Adjusted
                "impact_score": 75, "importance": "MEDIUM", "category": "MANUFACTURING", "volatility_factor": 1.4
            },
            # Vendas no Varejo (Retail Sales)
            "RSXFS": {
                "name": "Retail Sales: Total", "series_id": "RSXFS", # Retail Sales: Total, Seasonally Adjusted
                "impact_score": 70, "importance": "MEDIUM", "category": "CONSUMPTION", "volatility_factor": 1.3
            }
            # Adicione mais conforme necessário, consultando o site do FRED para os `series_id`
        }
    
    def _get_current_datetime(self) -> datetime:
        """Retorna o datetime atual, ou o datetime de override para backtesting."""
        return self.datetime_now_override if self.datetime_now_override else datetime.now()

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Faz requisição para a API do FRED com tratamento de erros."""
        if not self.api_key:
            self.logger.debug(f"FRED API Key não configurada. Não é possível fazer requisições FRED para {endpoint}.")
            return None
        
        try:
            params.update({
                "api_key": self.api_key,
                "file_type": "json"
            })
            
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=10) # Adiciona timeout
            response.raise_for_status() # Lança HTTPError para 4xx/5xx responses
            
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Requisição FRED excedeu o tempo limite para {endpoint}.")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro de requisição FRED para {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar JSON da resposta FRED para {endpoint}: {e}. Resposta: {response.text[:200]}...")
            return None
        except Exception as e:
            self.logger.error(f"Erro inesperado em _make_request para {endpoint}: {e}")
            return None

    def _get_series_observations(self, series_id: str, limit: int = 2) -> List[Dict]:
        """
        Busca as últimas observações para uma série específica do FRED.
        Args:
            series_id: O ID da série FRED (ex: "CPIAUCSL").
            limit: Número de observações mais recentes a buscar (para previous e actual).
        Returns:
            Lista de dicionários de observações.
        """
        params = {
            "series_id": series_id,
            "sort_order": "desc", # Ordem decrescente de data
            "limit": limit
        }
        data = self._make_request("series/observations", params)
        if data and "observations" in data:
            # Filter out '.' values (no data)
            valid_observations = [obs for obs in data["observations"] if obs["value"] != "."]
            return valid_observations
        return []

    def _get_release_info(self, release_id: str) -> Optional[Dict]:
        """Busca informações detalhadas de um release (ex: frequência)."""
        params = {"release_id": release_id}
        data = self._make_request("release", params)
        if data and "releases" in data and data["releases"]:
            return data["releases"][0]
        return None
    
    def get_upcoming_releases(self, days_ahead: int = 14) -> List[EconomicEvent]:
        """
        Obtém próximos releases econômicos importantes, incluindo dados anteriores/atuais.
        Utiliza cache para otimização.
        """
        current_dt = self._get_current_datetime()

        # Verifica o cache com base no current_dt
        if self.cache["last_full_update"] and \
           (current_dt - self.cache["last_full_update"]).total_seconds() < self.cache["cache_duration_seconds"]:
            self.logger.info("Retornando eventos FRED do cache.")
            return self.cache["upcoming_events"]

        if not self.api_key:
            self.logger.warning("FRED API Key não configurada. Não é possível obter releases.")
            self.cache["upcoming_events"] = []
            self.cache["last_full_update"] = current_dt
            return []

        self.logger.info(f"Buscando novos eventos FRED para os próximos {days_ahead} dias.")
        try:
            start_date_str = current_dt.strftime("%Y-%m-%d")
            end_date_str = (current_dt + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            params = {
                "realtime_start": start_date_str,
                "realtime_end": end_date_str,
                "include_release_dates_with_no_data": "false"
            }
            
            data = self._make_request("releases/dates", params)
            
            if not data or "release_dates" not in data:
                self.logger.warning("Nenhum dado de 'releases/dates' recebido do FRED.")
                self.cache["upcoming_events"] = [] 
                self.cache["last_full_update"] = current_dt
                return []
            
            events = []
            
            # Mapeia nomes de releases críticos para seus metadados (para classificação rápida)
            critical_names_map = {v["name"].lower(): k for k, v in self.critical_release_series.items()}

            for release in data["release_dates"]:
                release_id = release.get("release_id")
                release_name = release.get("release_name", "Unknown")
                release_date_str = release.get("date", "1970-01-01")
                
                try:
                    release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                except ValueError:
                    self.logger.warning(f"Data de release inválida: {release_date_str}. Pulando evento.")
                    continue

                importance = "LOW"
                impact_score = 30
                category = "GENERAL"
                series_id = None
                
                matched_series_key = critical_names_map.get(release_name.lower())
                if matched_series_key and matched_series_key in self.critical_release_series:
                    critical_info = self.critical_release_series[matched_series_key]
                    importance = critical_info["importance"]
                    impact_score = critical_info["impact_score"]
                    category = critical_info["category"]
                    series_id = critical_info["series_id"]
                
                actual_val: Optional[float] = None
                previous_val: Optional[float] = None
                forecast_val: Optional[float] = None 
                
                if series_id:
                    observations = self._get_series_observations(series_id, limit=2)
                    if observations:
                        try:
                            actual_val = float(observations[0]["value"]) # Most recent valid observation
                        except (ValueError, TypeError):
                            actual_val = None
                        
                        if len(observations) > 1:
                            try:
                                previous_val = float(observations[1]["value"]) # Second most recent valid observation
                            except (ValueError, TypeError):
                                previous_val = None

                frequency = "Unknown"
                release_details = self._get_release_info(release_id)
                if release_details and "frequency" in release_details:
                    frequency = release_details["frequency"]

                event_obj = EconomicEvent(
                    release_id=release_id,
                    name=release_name,
                    date=release_date,
                    time=self._estimate_release_time(category),
                    importance=importance,
                    frequency=frequency,
                    series_id=series_id,
                    previous_value=previous_val,
                    forecast=forecast_val, 
                    actual=actual_val,
                    impact_score=impact_score,
                    category=category
                )
                
                events.append(event_obj)
            
            # Ordena por data e impacto
            events.sort(key=lambda x: (x.date, -x.impact_score))
            
            self.logger.info(f"Coletados e enriquecidos {len(events)} eventos econômicos próximos do FRED.")
            self.cache["upcoming_events"] = events
            self.cache["last_full_update"] = current_dt
            return events
            
        except Exception as e:
            self.logger.error(f"Erro ao obter e processar releases futuros do FRED: {e}")
            return []
    
    def _classify_release(self, release_name: str, release_id: str) -> tuple:
        """
        Classifica a importância e categoria de um release com base no nome e IDs críticos pré-definidos.
        Esta função é uma fallback/complemento para a lógica em `get_upcoming_releases`
        que já tenta usar `self.critical_release_series`.
        """
        name_lower = release_name.lower()
        
        # Procura na lista de releases críticos pelo nome (chave) ou nome (dentro do dict)
        for key, info in self.critical_release_series.items():
            if key.lower() in name_lower or info["name"].lower() in name_lower:
                return info["importance"], info["impact_score"], info["category"]
        
        # Fallback para classificação genérica se não for um dos críticos principais
        if any(keyword in name_lower for keyword in [
            "employment", "payroll", "jobs", "unemployment",
            "inflation", "cpi", "pce", "price index",
            "federal funds", "fomc", "fed", "interest rate",
            "gdp", "gross domestic"
        ]):
            return "HIGH", 80, self._get_category(name_lower)
        
        elif any(keyword in name_lower for keyword in [
            "housing", "retail", "consumer", "industrial",
            "durable goods", "trade", "treasury", "bond"
        ]):
            return "MEDIUM", 55, self._get_category(name_lower)
        
        else:
            return "LOW", 25, "GENERAL"
    
    def _get_category(self, name_lower: str) -> str:
        """Determina a categoria do evento."""
        if any(word in name_lower for word in ["employment", "payroll", "jobs", "unemployment", "labor"]):
            return "EMPLOYMENT"
        elif any(word in name_lower for word in ["inflation", "cpi", "pce", "price"]):
            return "INFLATION"
        elif any(word in name_lower for word in ["fed", "fomc", "funds", "rate", "monetary"]):
            return "MONETARY_POLICY"
        elif any(word in name_lower for word in ["gdp", "growth", "economic", "production"]):
            return "GROWTH"
        elif any(word in name_lower for word in ["housing", "home", "building"]):
            return "HOUSING"
        elif any(word in name_lower for word in ["trade", "export", "import", "balance"]):
            return "TRADE"
        elif any(word in name_lower for word in ["manufacturing", "industrial", "ism"]):
            return "MANUFACTURING"
        elif any(word in name_lower for word in ["consumer", "retail", "sales"]):
            return "CONSUMPTION"
        else:
            return "GENERAL"
    
    def _estimate_release_time(self, category: str) -> str:
        """Estima horário típico de release baseado na categoria (Horário de Nova York - ET)."""
        time_map = {
            "EMPLOYMENT": "08:30 ET",  
            "INFLATION": "08:30 ET",   
            "MONETARY_POLICY": "14:00 ET", 
            "GROWTH": "08:30 ET",      
            "HOUSING": "10:00 ET",     
            "TRADE": "08:30 ET",       
            "MANUFACTURING": "10:00 ET",
            "CONSUMPTION": "08:30 ET",
            "GENERAL": "10:00 ET"      
        }
        return time_map.get(category, "10:00 ET")
    
    def get_high_impact_events_today(self) -> List[EconomicEvent]:
        """Obtém eventos de alto impacto para hoje."""
        current_dt = self._get_current_datetime()
        today_date = current_dt.date()
        # Busca releases até amanhã para pegar os de hoje, atualizando o cache
        upcoming = self.get_upcoming_releases(days_ahead=1) 
        
        return [
            event for event in upcoming 
            if event.date.date() == today_date and event.importance == "HIGH"
        ]
    
    def get_next_critical_event(self) -> Optional[EconomicEvent]:
        """Obtém o próximo evento crítico (impact_score >= 80)."""
        current_dt = self._get_current_datetime()
        upcoming = self.get_upcoming_releases(days_ahead=30)
        critical_events = [
            event for event in upcoming 
            if event.impact_score >= 80 and event.date.date() >= current_dt.date() # A partir de hoje
        ]
        # Retorna o primeiro evento crítico futuro, se houver
        return critical_events[0] if critical_events else None
    
    def generate_pre_event_alerts(self, hours_before: int = 24) -> List[Dict]:
        """Gera alertas preventivos antes de eventos importantes."""
        alerts = []
        current_dt = self._get_current_datetime()
        upcoming = self.get_upcoming_releases(days_ahead=7) # Busca eventos para os próximos 7 dias
        
        for event in upcoming:
            # Considera eventos de alta ou média importância
            if event.importance in ["HIGH", "MEDIUM"]:
                
                # Se o evento é hoje ou no futuro próximo
                if event.date.date() >= current_dt.date():
                    
                    # Constrói o datetime completo para o evento (data + hora estimada)
                    event_datetime_str = f"{event.date.strftime('%Y-%m-%d')} {event.time.split(' ')[0]}" # Remove 'ET'
                    try:
                        event_datetime = datetime.strptime(event_datetime_str, "%Y-%m-%d %H:%M")
                    except ValueError:
                        self.logger.warning(f"Não foi possível parsear o tempo do evento {event.name}: {event_datetime_str}. Ignorando horário.")
                        event_datetime = event.date # Usa só a data

                    time_until_release = event_datetime - current_dt
                    hours_until = time_until_release.total_seconds() / 3600

                    # Gera alerta se o evento está dentro da janela `hours_before` e ainda não passou
                    if 0 < hours_until <= hours_before:
                        
                        alert = {
                            "type": "PRE_EVENT_ALERT",
                            "title": f"⏰ Evento Econômico Importante: {event.name}",
                            "message": f"Divulgação de {event.name} ({event.importance} Impacto) em {hours_until:.1f} horas ({event_datetime.strftime('%d/%m %H:%M %Z')}).",
                            "severity": "MEDIUM" if hours_until > 6 else "HIGH", # Mais perto, maior a severidade
                            "timestamp": current_dt.isoformat(),
                            "event_data": {
                                "event_name": event.name,
                                "event_date": event.date.isoformat(),
                                "event_time": event.time,
                                "importance": event.importance,
                                "impact_score": event.impact_score,
                                "category": event.category,
                                "hours_until": hours_until,
                                "series_id": event.series_id,
                                "previous_value": event.previous_value,
                                "actual_value": event.actual, # O 'actual' será o último disponível antes do release
                                "forecast_value": event.forecast # Será None
                            },
                            "recommendations": self._generate_recommendations(event, hours_until)
                        }
                        
                        alerts.append(alert)
        
        return alerts
    
    def _generate_recommendations(self, event: EconomicEvent, hours_until: float) -> List[str]:
        """Gera recomendações de trading baseadas no tipo de evento e iminência."""
        recommendations = []
        
        if event.category == "EMPLOYMENT":
            recommendations.extend([
                "🔍 Monitorar DXY/BTC próximo ao NFP (Non-Farm Payrolls).",
                "📈 Esperar alta volatilidade em todos os ativos de risco.",
                "⚠️ Possível reversão de tendências se dados surpreenderem."
            ])
        elif event.category == "INFLATION":
            recommendations.extend([
                "🥇 Ouro pode ter movimento forte com dados de inflação (CPI/PCE).",
                "💵 DXY sensível a surpresas nos índices de preços.",
                "📊 Observar padrões de divergência DXY/BTC em torno da inflação."
            ])
        elif event.category == "MONETARY_POLICY":
            recommendations.extend([
                "🏛️ Decisões do FOMC (FED) podem gerar padrões angulares extremos.",
                "⚡ Volatilidade máxima esperada em todos os ativos.",
                "🎯 Monitorar 'perfect divergence' DXY/BTC com declarações do FED."
            ])
        elif event.category == "GROWTH":
            recommendations.extend([
                "📊 Dados de PIB (GDP) impactam diretamente o sentimento de risco.",
                "📈 Abertura de novas posições com cautela após o release."
            ])
        
        if hours_until <= 0.5: # 30 minutos antes
            recommendations.append("🚨 ALERTA CRÍTICO: Evento iminente em menos de 30 minutos - MÁXIMA ATENÇÃO!.")
        elif hours_until <= 2: # 2 horas antes
            recommendations.append("⚠️ ALERTA: Evento importante se aproxima - preparar para volatilidade.")
        elif hours_until <= 12: # 12 horas antes
            recommendations.append("🔔 Aviso: Evento de alto impacto nas próximas 12 horas.")
        
        return recommendations
    
    def correlate_with_angular_data(self, events: List[EconomicEvent], 
                                   angular_cache: Dict) -> Dict:
        """
        Correlaciona eventos econômicos com movimentos angulares históricos.
        Ainda em desenvolvimento, para um radar profissional real, esta lógica seria mais complexa.
        """
        correlations = {
            "event_impact_analysis": [],
            "volatility_predictions": {}, # Ex: prever se o evento causará alta ou baixa volatilidade
            "pattern_likelihood": {} # Ex: probabilidade de um certo padrão angular após o evento
        }
        
        try:
            # angular_history será uma lista de dicionários com 'timestamp', 'btc', 'gold', 'dxy'
            angular_history = angular_cache.get("angular_data", []) 

            # Converte a lista de dicionários para DataFrame para facilitar a busca por datas
            if angular_history:
                angular_df = pd.DataFrame(angular_history)
                # Garante que o timestamp é datetime e define como índice
                angular_df['timestamp'] = pd.to_datetime(angular_df['timestamp'])
                angular_df.set_index('timestamp', inplace=True)
                angular_df.sort_index(inplace=True)
            else:
                angular_df = pd.DataFrame()
            
            for event in events:
                event_date = event.date.date()
                
                if not angular_df.empty:
                    # Filtra dados angulares para o dia do evento ou nas 24h anteriores/posteriores
                    # Para simplificar, pegamos o dia do evento
                    event_day_data = angular_df.loc[
                        angular_df.index.date == event_date
                    ]
                    
                    if not event_day_data.empty:
                        # Analisa volatilidade angular no dia do evento (amplitude ou desvio padrão dos ângulos)
                        max_angle_change = 0
                        
                        for idx, data_row in event_day_data.iterrows():
                            for asset_key in ["btc", "gold", "dxy"]:
                                if asset_key in data_row and isinstance(data_row[asset_key], dict) and "angle" in data_row[asset_key]:
                                    angle_val = data_row[asset_key]["angle"]
                                    # Pega o ângulo absoluto como uma medida simples de volatilidade angular
                                    if abs(angle_val) > max_angle_change:
                                        max_angle_change = abs(angle_val)
                        
                        correlations["event_impact_analysis"].append({
                            "event_name": event.name,
                            "date": event_date.isoformat(),
                            "max_angular_volatility_observed": max_angle_change,
                            "event_impact_score": event.impact_score,
                            "event_category": event.category
                        })
            
            self.logger.info(f"Correlação angular concluída para {len(events)} eventos.")
            
        except Exception as e:
            self.logger.error(f"Erro na correlação angular: {e}")
        
        return correlations

# ============================================================================
# 🔄 ANALISADOR COM IA INTEGRADA
# ============================================================================

class AIEnhancedAnalyzer:
    """Analisador com IA integrada"""
    
    def __init__(self, environment: str = 'testnet'):
        self.environment = environment
        self.risk_config = RISK_CONFIG[environment]
        self.logger = logging.getLogger('AIAnalyzer')
        
        # Componentes de IA
        self.ml_predictor = MLPredictor(environment)
        self.pattern_matcher = PatternMatcher(AI_CONFIG['pattern_memory_size'])
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Calendário FRED (agora integrado)
        self.fred_calendar = FREDEconomicCalendar(FRED_API_KEY)
        self.last_fred_update = None # Usado para controlar a frequência de atualização do cache de eventos FRED
        self.fred_events_cache: List[EconomicEvent] = [] # Cache para eventos FRED
        
        # Cache de dados (OHLCV)
        self.data_cache = {}
        self.last_cache_update = {}
        
        self.logger.info(f"🧠 AIEnhancedAnalyzer inicializado para {environment}")
    
    def _update_fred_events_cache(self):
        """Atualiza o cache de eventos do FRED periodicamente."""
        current_dt = datetime.now() # Usa datetime.now() para o ambiente real
        if self.environment == 'simulate_backtest' and hasattr(self.fred_calendar, 'datetime_now_override') and self.fred_calendar.datetime_now_override:
            current_dt = self.fred_calendar.datetime_now_override

        if self.last_fred_update is None or \
           (current_dt - self.last_fred_update).total_seconds() > self.fred_calendar.cache["cache_duration_seconds"]:
            try:
                self.fred_events_cache = self.fred_calendar.get_upcoming_releases(days_ahead=7)
                self.last_fred_update = current_dt
                self.logger.info(f"Cache de eventos FRED atualizado. {len(self.fred_events_cache)} eventos encontrados.")
            except Exception as e:
                self.logger.error(f"Falha ao atualizar cache de eventos FRED: {e}")
                self.fred_events_cache = [] # Limpa cache se houver erro
    
    def _get_current_fred_impact(self) -> Tuple[bool, float, Optional[EconomicEvent]]:
        """
        Verifica se há um evento FRED de alto impacto iminente e calcula a penalidade na confiança.
        Retorna (is_high_impact_imminent, penalty_factor, next_high_impact_event).
        """
        self._update_fred_events_cache() # Garante que o cache de eventos FRED está atualizado
        
        current_dt = datetime.now()
        if self.environment == 'simulate_backtest' and hasattr(self.fred_calendar, 'datetime_now_override') and self.fred_calendar.datetime_now_override:
            current_dt = self.fred_calendar.datetime_now_override
        
        # Busca o próximo evento de alto impacto na janela relevante (ex: próximas 2-4 horas)
        next_high_impact_event: Optional[EconomicEvent] = None
        for event in self.fred_events_cache:
            if event.importance == "HIGH":
                # Tenta criar um datetime completo para o evento
                try:
                    event_datetime = event.date.replace(hour=int(event.time.split(':')[0]), minute=int(event.time.split(':')[1].split(' ')[0]))
                except (ValueError, AttributeError):
                    # Se o tempo não puder ser parseado, assume o início do dia do evento
                    event_datetime = event.date 
                    self.logger.warning(f"Erro ao parsear tempo para evento FRED '{event.name}'. Usando início do dia.")

                time_until_release_hours = (event_datetime - current_dt).total_seconds() / 3600
                
                # Considera evento iminente se estiver a 4 horas ou menos (antes ou depois)
                if -AI_CONFIG['fred_cooldown_minutes_high_impact']/60 <= time_until_release_hours <= 4.0: # Ex: 1h depois a 4h antes
                    next_high_impact_event = event
                    break # Pega o mais próximo
                

        if next_high_impact_event:
            hours_diff = (next_high_impact_event.date.replace(hour=int(next_high_impact_event.time.split(':')[0]), minute=int(next_high_impact_event.time.split(':')[1].split(' ')[0])) - current_dt).total_seconds() / 3600
            
            # Penalidade máxima perto do evento, diminuindo com a distância
            # Aplica penalidade mesmo se o evento já passou há pouco tempo (dentro do cooldown)
            if abs(hours_diff) <= 0.5: # 30 min antes/depois
                penalty = AI_CONFIG['fred_impact_penalty_factor'] * 1.0 # Penalidade total
            elif abs(hours_diff) <= AI_CONFIG['fred_cooldown_minutes_high_impact']/60: # Dentro da janela de cooldown FRED
                penalty = AI_CONFIG['fred_impact_penalty_factor'] * (1 - (abs(hours_diff) / (AI_CONFIG['fred_cooldown_minutes_high_impact']/60))) # Reduz gradualmente
            else:
                penalty = 0.0 # Nenhuma penalidade se estiver fora da janela de impacto/cooldown
            
            self.logger.info(f"🚨 Evento FRED de ALTO IMPACTO iminente/recente: {next_high_impact_event.name} em {hours_diff:.1f}h. Penalidade na confiança: {penalty:.2f}")
            return True, penalty, next_high_impact_event
        
        return False, 0.0, None

    def generate_enhanced_signal(self, symbol: str, active_positions: List) -> Optional[EnhancedTradingSignal]:
        """Gera sinal com todas as análises de IA, incluindo o impacto do calendário FRED."""
        try:
            # Verificar dados
            if symbol not in self.data_cache or '5m' not in self.data_cache[symbol]:
                self.logger.debug(f"Dados insuficientes para {symbol} no cache.")
                return None
            
            df = self.data_cache[symbol]['5m']
            if df is None or len(df) < 50:
                self.logger.debug(f"DataFrame insuficiente para {symbol} ({len(df)} candles).")
                return None
            
            # === Avaliar impacto do FRED ===
            is_fred_imminent, fred_penalty, fred_event_info = self._get_current_fred_impact()
            
            # Se um evento de alto impacto FRED está iminente, pode aplicar um filtro inicial
            if is_fred_imminent and fred_event_info:
                # Se o evento é de altíssima importância (e.g., FED Funds Rate) e estamos muito perto
                try:
                    event_datetime_fred = fred_event_info.date.replace(hour=int(fred_event_info.time.split(':')[0]), minute=int(fred_event_info.time.split(':')[1].split(' ')[0]))
                except (ValueError, AttributeError):
                    event_datetime_fred = fred_event_info.date # Fallback if time parsing fails

                current_dt_check = datetime.now()
                if self.environment == 'simulate_backtest' and hasattr(self.fred_calendar, 'datetime_now_override') and self.fred_calendar.datetime_now_override:
                    current_dt_check = self.fred_calendar.datetime_now_override

                hours_until_fred = (event_datetime_fred - current_dt_check).total_seconds() / 3600
                
                # Se estamos muito perto (ex: 30 minutos antes/depois) E o impacto é CRÍTICO (e.g., score >= 90)
                if abs(hours_until_fred) <= 0.5 and fred_event_info.impact_score >= 90: 
                    self.logger.warning(f"❌ Evento FRED CRÍTICO '{fred_event_info.name}' iminente/recente. PAUSANDO GERAÇÃO DE SINAL.")
                    return None # Não gera sinal
            
            # === 1. ANÁLISE TÉCNICA TRADICIONAL ===
            base_signal = self.generate_base_signal(df, symbol)
            if not base_signal:
                self.logger.debug(f"Nenhum sinal base gerado para {symbol}.")
                return None
            
            # === 2. ANÁLISE DE PADRÕES ===
            pattern = self.pattern_matcher.extract_pattern(df)
            pattern_score = self.pattern_matcher.get_pattern_score(pattern)
            
            # === 3. EXTRAÇÃO DE FEATURES PARA IA ===
            features = self.ml_predictor.extract_features(df, pattern_score)
            if not features:
                self.logger.debug(f"Falha ao extrair features para IA para {symbol}.")
                return None
            
            # === 4. PREDIÇÃO DE MACHINE LEARNING ===
            ai_probability, ai_confidence = self.ml_predictor.predict(features)
            
            # === 5. ANÁLISE DE SENTIMENTO ===
            sentiment_score = self.sentiment_analyzer.get_crypto_news_sentiment(symbol)
            
            # === 6. COMBINAR TODAS AS ANÁLISES ===
            final_confidence = self.calculate_final_confidence(
                base_confidence=base_signal['confidence'],
                ai_probability=ai_probability,
                ai_confidence=ai_confidence,
                pattern_score=pattern_score,
                sentiment_score=sentiment_score,
                action=base_signal['action']
            )
            
            # Aplicar penalidade do FRED na confiança final (mesmo se não pausar, reduz confiança)
            if is_fred_imminent:
                original_confidence = final_confidence
                final_confidence = max(0.0, final_confidence - fred_penalty)
                self.logger.info(f"Confiança ajustada por FRED: {original_confidence:.3f} -> {final_confidence:.3f} devido a {fred_event_info.name}")
            
            # === 7. FILTROS FINAIS ===
            if final_confidence < AI_CONFIG['prediction_threshold']:
                self.logger.debug(f"Confiança final insuficiente para {symbol}: {final_confidence:.3f}")
                return None
            
            # Verificar se a IA contradiz fortemente o sinal base
            if ((base_signal['action'] == 'BUY' and ai_probability < 0.3 and ai_confidence > 0.7) or
                (base_signal['action'] == 'SELL' and ai_probability > 0.7 and ai_confidence > 0.7)):
                self.logger.info(f"IA contradiz sinal base para {symbol} com alta confiança, cancelando.")
                return None
            
            return EnhancedTradingSignal(
                symbol=symbol,
                action=base_signal['action'],
                base_confidence=base_signal['confidence'],
                ai_probability=ai_probability,
                ai_confidence=ai_confidence,
                final_confidence=final_confidence,
                pattern_score=pattern_score,
                sentiment_score=sentiment_score,
                features=features,
                entry_price=base_signal['entry_price'],
                stop_loss=base_signal['stop_loss'],
                take_profit=base_signal['take_profit'],
                risk_reward=base_signal['risk_reward'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal enhanced para {symbol}: {e}", exc_info=True)
            return None
    
    def generate_base_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Gera sinal base usando análise técnica tradicional"""
        try:
            if len(df) < 30:
                return None
            
            # Calcular indicadores
            df = df.copy()
            
            # SuperTrend
            supertrend_dir = self.ml_predictor.calculate_supertrend_direction(df)
            
            # VWAP
            vwap = self.ml_predictor.calculate_vwap(df).iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # RSI
            rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
            
            # Volume
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / volume_ma if volume_ma > 0 else 1.0
            
            # ATR
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14).iloc[-1]
            
            # Lógica de sinal simplificada
            action = None
            confidence = 0.0
            
            # Condições para BUY
            if (supertrend_dir == 1 and 
                current_price > vwap and
                rsi < self.risk_config['rsi_overbought'] and # Use overbought for exit condition
                volume_ratio >= self.risk_config['min_volume_ratio']):
                action = 'BUY'
                confidence = 0.7
            
            # Condições para SELL
            elif (supertrend_dir == -1 and 
                  current_price < vwap and
                  rsi > self.risk_config['rsi_oversold'] and # Use oversold for exit condition
                  volume_ratio >= self.risk_config['min_volume_ratio']):
                action = 'SELL'
                confidence = 0.7
            
            if not action:
                return None
            
            # Calcular níveis
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.002
            
            if action == 'BUY':
                stop_loss = current_price - (atr * self.risk_config['stop_loss_atr_multiplier'])
                take_profit = current_price + (atr * self.risk_config['take_profit_atr_multiplier'])
            else:
                stop_loss = current_price + (atr * self.risk_config['stop_loss_atr_multiplier'])
                take_profit = current_price - (atr * self.risk_config['take_profit_atr_multiplier'])
            
            risk_reward = abs(take_profit - current_price) / abs(stop_loss - current_price) if abs(stop_loss - current_price) > 0.0001 else 0.0
            
            return {
                'action': action,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal base: {e}", exc_info=True)
            return None
    
    def calculate_final_confidence(self, base_confidence: float, ai_probability: float, 
                                   ai_confidence: float, pattern_score: float, 
                                   sentiment_score: float, action: str) -> float:
        """Combina todas as análises em uma confiança final"""
        try:
            # Pesos para cada componente
            weights = {
                'base': 0.40,        # 40% - Análise técnica tradicional
                'ai': 0.30,          # 30% - Predição de ML
                'pattern': 0.15,     # 15% - Padrões históricos
                'sentiment': 0.15    # 15% - Sentimento
            }
            
            # Normalizar ai_probability baseado na ação
            if action == 'BUY':
                ai_score = ai_probability  # Para BUY, queremos alta probabilidade
            else:
                ai_score = 1 - ai_probability  # Para SELL, queremos baixa probabilidade (prob de BUY ser baixa)
            
            # Ajustar pattern_score para a ação
            # Assume que pattern_score > 0.5 é bullish, < 0.5 bearish
            pattern_adjusted = pattern_score
            if action == 'SELL':
                pattern_adjusted = 1 - pattern_score
            
            # Ajustar sentiment para a ação
            if action == 'BUY':
                sentiment_adjusted = sentiment_score  # Sentimento positivo favorece BUY
            else:
                sentiment_adjusted = 1 - sentiment_score  # Sentimento negativo favorece SELL
            
            # Calcular confiança ponderada
            final_confidence = (
                weights['base'] * base_confidence +
                weights['ai'] * ai_score * ai_confidence +  # Modular por ai_confidence
                weights['pattern'] * pattern_adjusted +
                weights['sentiment'] * sentiment_adjusted
            )
            
            # Aplicar boost se IA estiver muito confiante
            if ai_confidence > 0.8:
                boost = self.risk_config['ai_confidence_boost'] * ai_confidence
                final_confidence += boost
            
            # Limitar entre 0 e 1
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular confiança final: {e}")
            return base_confidence
    
    def update_training_data(self, signal: Any, outcome: bool): # Changed to Any for compatibility
        """Atualiza dados de treinamento com resultado do trade"""
        try:
            # Adicionar amostra para o modelo ML
            if ML_AVAILABLE:
                if hasattr(signal, 'features') and signal.features is not None:
                    self.ml_predictor.add_training_sample(signal.features, outcome)
                else:
                    self.logger.warning("Sinal sem features para treinamento de ML. Pulando.")
            
            # Adicionar resultado para pattern matching
            if signal.symbol in self.data_cache and '5m' in self.data_cache[signal.symbol]:
                df = self.data_cache[signal.symbol]['5m']
                pattern = self.pattern_matcher.extract_pattern(df)
                self.pattern_matcher.add_outcome(pattern, outcome)
            
            # Log do aprendizado
            result_text = "GANHOU" if outcome else "PERDEU"
            self.logger.info(f"📚 Aprendizado: {signal.symbol} {signal.action} {result_text}")
            
            # For logging purposes only, convert to dict if not already
            signal_dict = signal.to_dict() if hasattr(signal, 'to_dict') else signal 

            self.logger.info(f"   Base: {signal_dict.get('base_confidence', 0.0):.3f} | IA: {signal_dict.get('ai_probability', 0.5):.3f} | Final: {signal_dict.get('final_confidence', 0.0):.3f}")
            
            # Verificar se deve retreinar
            if ML_AVAILABLE and self.ml_predictor.should_retrain():
                self.logger.info("🔄 Iniciando retreinamento do modelo...")
                self.ml_predictor.train_model()
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dados de treinamento: {e}", exc_info=True)

# ============================================================================
# 🤖 BOT PRINCIPAL COM IA
# ============================================================================

class AIEnhancedTradingBot:
    """Bot de trading com IA integrada"""
    
    def __init__(self, environment: str = 'testnet'):
        self.environment = environment
        self.urls = get_base_urls()
        self.analyzer = AIEnhancedAnalyzer(environment)
        
        # Estado do bot
        self.active_positions = []
        self.signals_history = []
        self.trade_history = [] # Stores detailed trade outcomes for Kelly criterion
        self.last_trade_time = {}
        self.last_reversal_time = {}
        self.daily_trades = 0
        self.daily_reset_date = datetime.now().date()
        self.running = False
        self.fred_last_checked_for_cooldown = None # Para gerenciar o cooldown do FRED

        # Performance tracking com métricas de IA
        self.performance = {
            "current_balance": 0.0,
            "start_balance": 0.0,
            "total_pnl": 0.0,
            "roi_percentage": 0.0,
            "winning_trades": 0,
            "total_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0, # To be calculated more robustly
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "last_update": None,
            "session_start_time": datetime.now().isoformat(),
            "ai_accuracy": 0.0,
            "ai_predictions": 0,
            "ai_correct_predictions": 0,
            "pattern_accuracy": 0.0, # Not directly calculated yet, but kept for future use
            "sentiment_impact": 0.0 # Not directly calculated yet, but kept for future use
        }
        
        # Configurar logging
        log_level = logging.DEBUG if self.environment == 'testnet' else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'ai_bot_{self.environment}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AIBot')
        
        self.logger.info(f"🤖 AIEnhancedTradingBot inicializado - {environment.upper()}")
        if ML_AVAILABLE:
            self.logger.info("✅ Machine Learning disponível")
        else:
            self.logger.warning("⚠️ Machine Learning NÃO disponível")
        if YFINANCE_AVAILABLE:
            self.logger.info("✅ YFinance disponível para dados históricos")
        else:
            self.logger.warning("⚠️ YFinance NÃO disponível. Backtesting pode ser limitado.")
        if FRED_API_KEY and FRED_API_KEY != "DUMMY_KEY_FRED":
            self.logger.info("✅ FRED API configurada. Calendário econômico ativo.")
        else:
            self.logger.warning("⚠️ FRED API Key não configurada/inválida. Calendário econômico inativo.")
        if CRYPTOPANIC_API_KEY and CRYPTOPANIC_API_KEY != "DUMMY_KEY_CRYPTOPANIC":
            self.logger.info("✅ CryptoPanic API configurada. Análise de sentimento de cripto ativa.")
        else:
            self.logger.warning("⚠️ CryptoPanic API Key não configurada/inválida. Análise de sentimento de cripto limitada.")

            
    def get_account_balance(self) -> float:
        """Obtém saldo da conta"""
        try:
            endpoint = "/futures/usdt/accounts"
            headers = sign_request("GET", endpoint)
            full_url = f"{self.urls['rest']}/api/v4{endpoint}"
            
            response = requests.get(full_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                balance = float(data.get('available', 0))
                
                if self.performance["start_balance"] == 0.0:
                    self.performance["start_balance"] = balance
                self.performance["current_balance"] = balance
                return balance
            
            self.logger.error(f"Falha ao obter saldo: {response.status_code}, {response.text}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Erro ao obter saldo: {e}")
            return 0.0
    
    def get_current_price(self, symbol: str) -> float:
        """Obtém preço atual"""
        try:
            endpoint = "/futures/usdt/tickers"
            query_string = f"contract={symbol}"
            full_url = f"{self.urls['rest']}/api/v4{endpoint}?{query_string}"
            
            response = requests.get(full_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return float(data[0].get('last', 0))
            self.logger.error(f"Falha ao obter preço {symbol}: {response.status_code}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Erro ao obter preço {symbol}: {e}")
            return 0.0
    
    def get_ohlcv_data(self, symbol: str, interval: str = '5m', limit: int = 200) -> pd.DataFrame:
        """
        Busca dados OHLCV.
        Para live/testnet, usa a API da Gate.io.
        Para backtesting (simulação), pode usar yfinance ou dados locais.
        """
        if self.environment == 'simulate_backtest' and YFINANCE_AVAILABLE:
            try:
                yf_symbol = _convert_gateio_symbol_to_yfinance(symbol)
                # yfinance 'interval' maps: '5m' -> '5m', '15m' -> '15m', '1h' -> '60m'
                yf_interval = {'5m': '5m', '15m': '15m', '1h': '60m'}.get(interval, '5m')
                
                # yfinance 'period' argument for historical depth.
                if yf_interval == '5m':
                    # For 5m, yfinance typically has max 7 days historical data
                    period = "7d"
                elif yf_interval == '15m':
                    # For 15m, yfinance typically has max 60 days historical data
                    period = "60d"
                elif yf_interval == '60m':
                    # For 60m, yfinance typically has max 730 days historical data (2 years)
                    period = "2y"
                else:
                    period = "1mo" # Default for other intervals

                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=period, interval=yf_interval)
                
                if df.empty:
                    self.logger.warning(f"YFinance returned empty data for {yf_symbol} with interval {yf_interval} and period {period}")
                    return pd.DataFrame()

                # Rename columns to match Gate.io/common format
                df.columns = [col.lower() for col in df.columns]
                df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'}, inplace=True)
                df.index.name = 'timestamp'
                df = df[['open', 'high', 'low', 'close', 'volume']] # Reorder columns
                df = df.dropna().sort_index()
                return df.tail(limit) # Ensure we return max 'limit' candles

            except Exception as e:
                self.logger.error(f"Erro ao buscar OHLCV com YFinance para {symbol} {interval}: {e}", exc_info=True)
                return pd.DataFrame()
        else: # Use Gate.io API for live and testnet environments
            try:
                endpoint = f"/futures/usdt/candlesticks"
                query_string = f"contract={symbol}&interval={interval}&limit={limit}"
                full_url = f"{self.urls['rest']}/api/v4{endpoint}?{query_string}"
                
                response = requests.get(full_url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        return pd.DataFrame()
                    
                    df = pd.DataFrame(data, columns=['timestamp', 'volume', 'close', 'high', 'low', 'open'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df.set_index('timestamp', inplace=True)
                    return df.dropna().sort_index()
                
                self.logger.error(f"Falha ao buscar OHLCV {symbol} {interval}: {response.status_code}, {response.text}")
                return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Erro ao buscar OHLCV {symbol} {interval}: {e}", exc_info=True)
                return pd.DataFrame()
    
    def update_data_cache(self):
        """Atualiza cache de dados"""
        try:
            for symbol in TRADING_SYMBOLS:
                if symbol not in self.analyzer.data_cache:
                    self.analyzer.data_cache[symbol] = {}
                
                for timeframe in ['5m', '15m', '1h']:
                    cache_key = f"{symbol}_{timeframe}"
                    now = datetime.now()
                    
                    # Update cache every 2 minutes for 5m, 5 minutes for 15m, 15 minutes for 1h
                    update_interval = timedelta(minutes=2) if timeframe == '5m' else \
                                      timedelta(minutes=5) if timeframe == '15m' else \
                                      timedelta(minutes=15)
                                      
                    if (cache_key not in self.analyzer.last_cache_update or 
                        now - self.analyzer.last_cache_update[cache_key] > update_interval):
                        
                        # Use the environment-aware get_ohlcv_data
                        df = self.get_ohlcv_data(symbol, timeframe, 200)
                        
                        if not df.empty:
                            self.analyzer.data_cache[symbol][timeframe] = df
                            self.analyzer.last_cache_update[cache_key] = now
                            self.logger.debug(f"Cache atualizado: {symbol} {timeframe} ({len(df)} candles)")
                        else:
                            self.logger.warning(f"Falha ao atualizar cache: {symbol} {timeframe}")
                            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar cache de dados: {e}", exc_info=True)
    
    def calculate_optimal_position_size(self, current_price: float, risk_factor: float = 0.25) -> float:
        """
        Calcula o tamanho ótimo da posição usando uma versão simplificada do Kelly Criterion.
        Limita o risco por trade a uma fração do saldo disponível.
        
        Args:
            current_price (float): Preço atual do ativo.
            risk_factor (float): Fator de risco (e.g., 0.25 para 25% do Kelly fraction).
                                 Isso limita o risco por trade para evitar alta volatilidade.
        Returns:
            float: Número de contratos a serem negociados.
        """
        total_trades = self.performance["total_trades"]
        winning_trades = self.performance["winning_trades"]
        
        if total_trades < AI_CONFIG['min_training_samples']: # Need enough trades for statistics
            self.logger.info("Ainda não há trades suficientes para calcular tamanho de posição otimizado. Usando tamanho fixo.")
            return self.risk_config['position_size_usdt'] / current_price if current_price > 0 else 0
            
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.5 # Default to 0.5 if no trades
        
        # Calculate average win and average loss from trade history
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.01 # Avoid division by zero
        avg_loss = sum(losses) / len(losses) if losses else -0.01 # Avoid division by zero, convert to positive for Kelly
        avg_loss = abs(avg_loss) # Kelly uses positive values for losses
        
        if avg_win == 0 or avg_loss == 0:
            self.logger.warning("Average win or loss is zero, cannot apply Kelly Criterion. Using fixed size.")
            return self.risk_config['position_size_usdt'] / current_price if current_price > 0 else 0
            
        if win_rate * avg_win - (1 - win_rate) * avg_loss <= 0:
            self.logger.warning("Edge is not positive (Kelly criterion suggests not to bet). Using fixed size.")
            return self.risk_config['position_size_usdt'] / current_price if current_price > 0 else 0

        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Limit Kelly fraction to prevent over-leveraging
        # Common practice is to use a fraction of Kelly (e.g., 0.25)
        # And also an absolute maximum (e.g., 5% of balance for a single trade)
        
        # Current balance is needed for absolute position sizing
        current_balance = self.get_account_balance()
        if current_balance == 0:
            self.logger.warning("Current balance is zero. Cannot calculate position size.")
            return 0
            
        # Maximum capital to risk per trade (e.g., 5% of total balance, or a fixed USDT amount if balance is very low)
        max_capital_per_trade_usdt = max(self.risk_config['position_size_usdt'], current_balance * 0.05)
        
        # Convert Kelly fraction to USDT amount
        calculated_usdt_size = kelly_fraction * risk_factor * current_balance
        
        # Use the minimum of calculated_usdt_size and max_capital_per_trade_usdt
        position_usdt = min(calculated_usdt_size, max_capital_per_trade_usdt)
        
        contracts = max(1, int(position_usdt / current_price))
        
        self.logger.info(f"Kelly Calc: Win Rate: {win_rate:.2f}, Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
        self.logger.info(f"Kelly Fraction: {kelly_fraction:.3f}, Desired USDT: ${position_usdt:.2f}, Contracts: {contracts}")
        return contracts

    def place_market_order(self, symbol: str, side: str, contracts: Optional[float] = None) -> Optional[dict]:
        """Coloca ordem a mercado"""
        try:
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                self.logger.error(f"Preço inválido para {symbol} ao tentar colocar ordem.")
                return None
            
            # Use Kelly Criterion for position sizing if not provided
            if contracts is None:
                contracts = self.calculate_optimal_position_size(current_price)
            
            if contracts <= 0:
                self.logger.warning(f"Tamanho de contratos inválido ({contracts}) para {symbol}.")
                return None

            endpoint = "/futures/usdt/orders"
            order_data = {
                "contract": symbol,
                "size": contracts if side == "buy" else -contracts,
                "text": f"ai_bot_{int(time.time())}"
            }
            
            body = json.dumps(order_data)
            headers = sign_request("POST", endpoint, "", body)
            full_url = f"{self.urls['rest']}/api/v4{endpoint}"
            
            response = requests.post(full_url, headers=headers, data=body, timeout=15)
            if response.status_code in [200, 201]:
                self.logger.info(f"✅ Ordem {side.upper()} {symbol}: {contracts} contratos @ ${current_price:.4f}")
                return response.json()
            
            self.logger.error(f"Falha na ordem {side} {symbol}: {response.status_code}, {response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao colocar ordem: {e}", exc_info=True)
            return None
    
    def can_trade_symbol(self, signal_symbol: str) -> bool:
        """Verificações para trading"""
        today = datetime.now().date()
        if today != self.daily_reset_date:
            self.daily_trades = 0
            self.daily_reset_date = today
            self.performance["daily_trades"] = 0
            self.performance["daily_pnl"] = 0.0
        
        risk_config = RISK_CONFIG[self.environment]
        
        if self.daily_trades >= risk_config['max_daily_trades']:
            self.logger.warning(f"Limite diário de trades atingido: {self.daily_trades}/{risk_config['max_daily_trades']}")
            return False
        
        if len(self.active_positions) >= risk_config['max_open_positions']:
            self.logger.info(f"Limite de posições abertas atingido: {len(self.active_positions)}/{risk_config['max_open_positions']}")
            return False
        
        # Cooldown por trade no mesmo símbolo
        if signal_symbol in self.last_trade_time:
            cooldown_end = self.last_trade_time[signal_symbol] + timedelta(minutes=risk_config['cooldown_minutes'])
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).total_seconds() / 60
                self.logger.debug(f"Cooldown para {signal_symbol}: {remaining:.1f}min restantes.")
                return False
        
        # Cooldown por evento FRED de alto impacto
        # Verifica se há um evento FRED de alto impacto que justifique o cooldown
        if FRED_API_KEY and FRED_API_KEY != "DUMMY_KEY_FRED":
            self.analyzer._update_fred_events_cache() # Garante cache FRED atualizado
            
            current_dt_fred_check = datetime.now()
            if self.environment == 'simulate_backtest' and hasattr(self.analyzer.fred_calendar, 'datetime_now_override') and self.analyzer.fred_calendar.datetime_now_override:
                current_dt_fred_check = self.analyzer.fred_calendar.datetime_now_override

            for event in self.analyzer.fred_events_cache:
                if event.importance == "HIGH":
                    try:
                        event_datetime = event.date.replace(hour=int(event.time.split(':')[0]), minute=int(event.time.split(':')[1].split(' ')[0]))
                    except (ValueError, AttributeError):
                        event_datetime = event.date # Fallback if time parsing fails
                    
                    time_diff_hours = (event_datetime - current_dt_fred_check).total_seconds() / 3600
                    
                    # Se o evento está dentro da janela de cooldown (antes ou depois)
                    if abs(time_diff_hours) * 60 <= AI_CONFIG['fred_cooldown_minutes_high_impact']:
                        self.logger.warning(f"Trade bloqueado devido a evento FRED de alto impacto '{event.name}'. Cooldown ativo.")
                        return False

        if any(pos['symbol'] == signal_symbol for pos in self.active_positions):
            self.logger.debug(f"Posição já existe para {signal_symbol}. Não abrir nova.")
            return False
        
        return True
    
    def execute_enhanced_signal(self, signal: EnhancedTradingSignal) -> bool:
        """Executa sinal enhanced com IA"""
        if not signal or not self.can_trade_symbol(signal.symbol):
            return False
        
        side = "buy" if signal.action == "BUY" else "sell"
        
        # Get optimal contracts using Kelly criterion
        contracts_to_trade = self.calculate_optimal_position_size(signal.entry_price)
        
        if contracts_to_trade <= 0:
            self.logger.warning(f"Não foi possível determinar um tamanho de posição válido para {signal.symbol}. Cancelando trade.")
            return False

        order = self.place_market_order(signal.symbol, side, contracts=contracts_to_trade)
        
        if order:
            time.sleep(2) # Give some time for order to execute and price to update
            actual_price = self.get_current_price(signal.symbol)
            
            if actual_price == 0:
                self.logger.error(f"Preço de execução inválido para {signal.symbol} após a ordem.")
                return False
            
            position = {
                'symbol': signal.symbol,
                'side': signal.action.lower(),
                'size': float(order.get('size', 1)),
                'entry_price': actual_price,
                'current_price': actual_price, # Will be updated by manage_positions
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'signal': signal.to_dict(), # Store the signal details as dict
                'order_id': order.get('id'),
                'timestamp': datetime.now(),
                'pnl': 0.0,
                'pnl_percent': 0.0,
                'ai_prediction': signal.ai_probability,
                'pattern_score': signal.pattern_score,
                'sentiment_score': signal.sentiment_score
            }
            
            self.active_positions.append(position)
            self.last_trade_time[signal.symbol] = datetime.now()
            self.daily_trades += 1
            self.performance["total_trades"] += 1
            self.performance["daily_trades"] += 1
            self.performance["ai_predictions"] += 1 # Count AI-driven trades
            self.signals_history.append(signal)
            
            self.logger.info(f"🎯 POSIÇÃO ABERTA COM IA: {signal.action} {signal.symbol}")
            self.logger.info(f"  💰 Entrada: ${actual_price:.4f} | Contratos: {contracts_to_trade:.2f}")
            self.logger.info(f"  🧠 IA Prob: {signal.ai_probability:.3f} | Conf Final: {signal.final_confidence:.3f}")
            self.logger.info(f"  📊 Pattern: {signal.pattern_score:.3f} | Sentiment: {signal.sentiment_score:.3f}")
            self.logger.info(f"  🎲 Base: {signal.base_confidence:.3f} | R/R: {signal.risk_reward:.2f}")
            
            return True
            
        return False
    
    def manage_positions(self):
        """Gestão de posições com feedback para IA"""
        positions_to_close = []
        
        for position in list(self.active_positions): # Iterate over a copy to allow modification
            try:
                current_price = self.get_current_price(position['symbol'])
                if current_price == 0:
                    self.logger.warning(f"Não foi possível obter o preço atual para {position['symbol']}. Pulando gerenciamento.")
                    continue
                
                position['current_price'] = current_price
                
                # Calcular PnL
                if position['side'] == 'buy':
                    pnl = (current_price - position['entry_price']) * abs(position['size'])
                else: # 'sell' (short)
                    pnl = (position['entry_price'] - current_price) * abs(position['size'])
                
                position['pnl'] = pnl
                position['pnl_percent'] = (pnl / (position['entry_price'] * abs(position['size']))) * 100 if position['entry_price'] * abs(position['size']) != 0 else 0
                
                # Verificar condições de saída
                should_close = False
                close_reason = ""
                
                # Stop Loss / Take Profit
                if position['side'] == 'buy':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif current_price >= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                else: # 'sell'
                    if current_price >= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif current_price <= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                
                if should_close:
                    positions_to_close.append((position, close_reason, pnl))
                    
            except Exception as e:
                self.logger.error(f"Erro ao gerenciar posição {position.get('symbol', 'N/A')}: {e}", exc_info=True)
        
        # Close positions
        for position, reason, pnl in positions_to_close:
            if position in self.active_positions: 
                self.close_position_with_ai_feedback(position, reason, pnl)
                self.active_positions.remove(position)
    
    def close_position_with_ai_feedback(self, position: dict, reason: str, pnl: float):
        """Fecha posição e fornece feedback para IA"""
        try:
            close_side = "sell" if position['side'] == "buy" else "buy"
            order = self.place_market_order(position['symbol'], close_side, contracts=abs(position['size'])) # Ensure closing the full size
            
            if order:
                outcome = pnl > 0
                roi_trade = (pnl / (position['entry_price'] * abs(position['size']))) * 100 if position['entry_price'] * abs(position['size']) != 0 else 0
                
                self.logger.info(f"🔚 POSIÇÃO FECHADA: {position['symbol']} - {reason}")
                self.logger.info(f"  💰 PnL: ${pnl:.2f} ({roi_trade:.2f}%)")
                self.logger.info(f"  🧠 IA estava certa: {'SIM' if outcome else 'NÃO'}")
                
                # Registrar trade com métricas de IA
                trade_record = {
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': position['current_price'],
                    'pnl': pnl,
                    'pnl_percent': roi_trade,
                    'reason': reason,
                    'duration_minutes': (datetime.now() - position['timestamp']).total_seconds() / 60,
                    'ai_prediction': position.get('ai_prediction', 0.5),
                    'pattern_score': position.get('pattern_score', 0.5),
                    'sentiment_score': position.get('sentiment_score', 0.5),
                    'ai_was_correct': outcome,
                    'timestamp': datetime.now().isoformat()
                }
                self.trade_history.append(trade_record)
                
                # Atualizar performance
                self.performance["total_pnl"] += pnl
                self.performance["daily_pnl"] += pnl
                
                if pnl > 0:
                    self.performance["winning_trades"] += 1
                    # Check if AI's prediction aligns with winning outcome
                    if (position['side'] == 'buy' and position.get('ai_prediction', 0.5) >= AI_CONFIG['prediction_threshold']) or \
                       (position['side'] == 'sell' and position.get('ai_prediction', 0.5) < (1 - AI_CONFIG['prediction_threshold'])):
                        self.performance["ai_correct_predictions"] += 1
                
                # Recalcular métricas
                if self.performance["total_trades"] > 0:
                    self.performance["win_rate"] = (self.performance["winning_trades"] / self.performance["total_trades"]) * 100
                
                if self.performance["ai_predictions"] > 0:
                    self.performance["ai_accuracy"] = (self.performance["ai_correct_predictions"] / self.performance["ai_predictions"]) * 100
                
                if self.performance["start_balance"] > 0:
                    self.performance["roi_percentage"] = (self.performance["total_pnl"] / self.performance["start_balance"]) * 100
                
                # Feedback para IA
                if 'signal' in position and isinstance(position['signal'], dict):
                    original_signal_dict = position['signal']
                    features_for_feedback = None
                    if position['symbol'] in self.analyzer.data_cache and '5m' in self.analyzer.data_cache[position['symbol']]:
                        df_current = self.analyzer.data_cache[position['symbol']]['5m']
                        pattern_score_from_pos = position.get('pattern_score', 0.5)
                        features_for_feedback = self.analyzer.ml_predictor.extract_features(df_current, pattern_score=pattern_score_from_pos)

                    signal_for_feedback = EnhancedTradingSignal(
                        symbol=original_signal_dict['symbol'],
                        action=original_signal_dict['action'],
                        base_confidence=original_signal_dict['base_confidence'],
                        ai_probability=original_signal_dict['ai_probability'],
                        ai_confidence=original_signal_dict['ai_confidence'],
                        final_confidence=original_signal_dict['final_confidence'],
                        pattern_score=original_signal_dict['pattern_score'],
                        sentiment_score=original_signal_dict['sentiment_score'],
                        features=features_for_feedback or AIFeatures(**{k:0.0 for k in AIFeatures.__annotations__.keys()}), # Provide a dummy if re-extraction fails
                        entry_price=original_signal_dict['entry_price'],
                        stop_loss=original_signal_dict['stop_loss'],
                        take_profit=original_signal_dict['take_profit'],
                        risk_reward=original_signal_dict['risk_reward'],
                        timestamp=datetime.fromisoformat(original_signal_dict['timestamp'])
                    )
                    self.analyzer.update_training_data(signal_for_feedback, outcome)
                else:
                    self.logger.warning("No signal data found in position for AI feedback.")
                
                self.performance["last_update"] = datetime.now().isoformat()
                
        except Exception as e:
            self.logger.error(f"Erro ao fechar posição: {e}", exc_info=True)
    
    def scan_markets_with_ai(self):
        """Escaneia mercados usando IA"""
        for symbol in TRADING_SYMBOLS:
            try:
                if any(pos['symbol'] == symbol for pos in self.active_positions):
                    continue
                
                # Gerar sinal enhanced com IA
                signal = self.analyzer.generate_enhanced_signal(symbol, self.active_positions)
                
                if signal:
                    self.logger.info(f"🔔 SINAL IA DETECTADO: {signal.action} {signal.symbol}")
                    self.logger.info(f"  🧠 IA Prob: {signal.ai_probability:.3f} | Conf Final: {signal.final_confidence:.3f}")
                    self.logger.info(f"  📊 Pattern: {signal.pattern_score:.3f} | Sentiment: {signal.sentiment_score:.3f}")
                    self.logger.info(f"  🎲 Base: {signal.base_confidence:.3f} | R/R: {signal.risk_reward:.2f}")
                    
                    success = self.execute_enhanced_signal(signal)
                    if success:
                        self.logger.info(f"✅ Sinal IA executado com sucesso para {signal.symbol}")
                    else:
                        self.logger.warning(f"❌ Falha ao executar sinal IA para {signal.symbol}")
                        
            except Exception as e:
                self.logger.error(f"Erro ao escanear {symbol} com IA: {e}", exc_info=True)
    
    def run(self):
        """Loop principal com IA"""
        self.running = True
        self.get_account_balance()
        
        self.logger.info(f"🚀 BOT COM IA v2.1 INICIADO ({self.environment.upper()})")
        self.logger.info(f"💰 Saldo inicial: ${self.performance['start_balance']:.2f}")
        self.logger.info(f"🧠 ML Disponível: {'SIM' if ML_AVAILABLE else 'NÃO'}")
        self.logger.info(f"📊 XGBoost: {'SIM' if XGB_AVAILABLE else 'NÃO'}")
        self.logger.info(f"💭 Sentiment: {'SIM' if SENTIMENT_AVAILABLE else 'NÃO'}")
        self.logger.info(f"📈 YFinance: {'SIM' if YFINANCE_AVAILABLE else 'NÃO'}")
        if FRED_API_KEY and FRED_API_KEY != "DUMMY_KEY_FRED":
            self.logger.info("📅 Calendário FRED: ATIVO")
        else:
            self.logger.warning("⚠️ Calendário FRED: INATIVO (chave API ausente/inválida)")
        if CRYPTOPANIC_API_KEY and CRYPTOPANIC_API_KEY != "DUMMY_KEY_CRYPTOPANIC":
            self.logger.info("📰 CryptoPanic: ATIVO (Sentimento de Cripto)")
        else:
            self.logger.warning("⚠️ CryptoPanic: INATIVO (chave API ausente/inválida)")
        
        self.performance["session_start_time"] = datetime.now().isoformat()
        
        cycle_count = 0
        
        try:
            while self.running:
                cycle_start = time.time()
                cycle_count += 1
                
                try:
                    # 0. Atualizar eventos FRED (ocorre dentro do analyzer)
                    # A cada ciclo, o analisador vai verificar se o cache do FRED precisa ser atualizado.
                    
                    # 1. Atualizar cache de dados (OHLCV)
                    self.update_data_cache()
                    
                    # 2. Gerenciar posições (com feedback para IA)
                    if self.active_positions:
                        self.manage_positions()
                    
                    # 3. Escanear com IA
                    self.scan_markets_with_ai()
                    
                    # 4. Verificar se precisa retreinar modelo
                    if ML_AVAILABLE and cycle_count % 10 == 0 and self.analyzer.ml_predictor.should_retrain():
                        self.logger.info("🔄 Retreinamento de modelo necessário...")
                        self.analyzer.ml_predictor.train_model()
                    
                    # 5. Log periódico com IA
                    if cycle_count % 20 == 0:
                        self.log_ai_periodic_status()
                    
                    # 6. Aguardar próximo ciclo
                    elapsed = time.time() - cycle_start
                    sleep_time = max(1, 30 - elapsed) # Aim for a cycle every 30 seconds
                    time.sleep(sleep_time)
                    
                except Exception as cycle_error:
                    self.logger.error(f"Erro no ciclo {cycle_count}: {cycle_error}", exc_info=True)
                    time.sleep(10) # Pause longer in case of error
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Interrupção pelo usuário")
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no bot: {e}", exc_info=True)
        finally:
            self.running = False
            self.shutdown_ai_bot()
    
    def log_ai_periodic_status(self):
        """Log periódico com métricas de IA"""
        try:
            current_balance = self.get_account_balance()
            
            self.logger.info("=" * 70)
            self.logger.info(f"🧠 STATUS IA - {datetime.now().strftime('%H:%M:%S')}")
            self.logger.info(f"💰 Saldo: ${current_balance:.2f} | PnL: ${self.performance['total_pnl']:.2f} ({self.performance['roi_percentage']:.2f}%)")
            self.logger.info(f"📈 Trades: {self.performance['total_trades']} | Win Rate: {self.performance['win_rate']:.1f}%")
            self.logger.info(f"🧠 IA Accuracy: {self.performance['ai_accuracy']:.1f}% | Predições: {self.performance['ai_predictions']}")
            self.logger.info(f"🎯 Modelo Accuracy: {self.analyzer.ml_predictor.model_accuracy:.3f} | Samples: {len(self.analyzer.ml_predictor.training_data)}")
            
            # Adicionar status do FRED
            if FRED_API_KEY and FRED_API_KEY != "DUMMY_KEY_FRED":
                self.analyzer._update_fred_events_cache() # Atualiza cache para exibição no log
                fred_high_impact_today = [e for e in self.analyzer.fred_events_cache if e.importance == "HIGH" and e.date.date() == datetime.now().date()]
                if fred_high_impact_today:
                    self.logger.info(f"📅 Eventos FRED HOJE (Alto Impacto):")
                    for event in fred_high_impact_today:
                        hours_until_release = (event.date.replace(hour=int(event.time.split(':')[0]), minute=int(event.time.split(':')[1].split(' ')[0])) - datetime.now()).total_seconds() / 3600
                        self.logger.info(f"    - {event.name} ({event.time}): {event.actual} (Prev: {event.previous_value}) - Em {hours_until_release:.1f}h")
                else:
                    self.logger.info("📅 Nenhum evento FRED de Alto Impacto hoje.")

            if self.active_positions:
                self.logger.info(f"🔄 Posições Ativas ({len(self.active_positions)}):")
                for pos in self.active_positions:
                    duration = (datetime.now() - pos['timestamp']).total_seconds() / 60
                    ai_pred = pos.get('ai_prediction', 0.5)
                    self.logger.info(f"   📍 {pos['symbol']} {pos['side'].upper()}: PnL ${pos.get('pnl', 0):.2f} ({pos.get('pnl_percent', 0):.2f}%) | IA: {ai_pred:.3f} ({duration:.1f}min)")
            
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Erro no log de IA: {e}", exc_info=True)
    
    def shutdown_ai_bot(self):
        """Encerramento com estatísticas de IA"""
        try:
            final_balance = self.get_account_balance()
            
            self.logger.info("🔴 ENCERRANDO BOT COM IA")
            self.logger.info(f"💰 Saldo Final: ${final_balance:.2f}")
            self.logger.info(f"📊 Performance Final:")
            self.logger.info(f"   • Total PnL: ${self.performance['total_pnl']:.2f}")
            self.logger.info(f"   • ROI: {self.performance['roi_percentage']:.2f}%")
            self.logger.info(f"   • Win Rate: {self.performance['win_rate']:.1f}%")
            self.logger.info(f"   • Total Trades: {self.performance['total_trades']}")
            
            self.logger.info(f"🧠 Estatísticas de IA:")
            self.logger.info(f"   • IA Accuracy: {self.performance['ai_accuracy']:.1f}%")
            self.logger.info(f"   • Predições: {self.performance['ai_predictions']}")
            if ML_AVAILABLE:
                self.logger.info(f"   • Modelo Accuracy: {self.analyzer.ml_predictor.model_accuracy:.3f}")
                self.logger.info(f"   • Samples de Treino: {len(self.analyzer.ml_predictor.training_data)}")
            self.logger.info(f"   • Padrões Aprendidos: {len(self.analyzer.pattern_matcher.patterns)}")
            
            # Salvar modelo final
            if ML_AVAILABLE and self.analyzer.ml_predictor.model is not None:
                self.analyzer.ml_predictor.save_model()
                self.logger.info("💾 Modelo de IA salvo")
            
            self.performance["current_balance"] = final_balance
            self.performance["last_update"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Erro no encerramento: {e}", exc_info=True)
    
    def stop(self):
        """Para o bot"""
        self.running = False
        self.logger.info("🛑 Sinal de parada recebido")

# ============================================================================
# ⚙️ BACKTESTING INTEGRADO
# ============================================================================

class Backtester:
    """
    Ferramenta de Backtesting para simular o desempenho do bot com dados históricos.
    Este é um esqueleto. A implementação completa exigiria:
    1. Carregamento eficiente de dados históricos (possivelmente de um banco de dados local).
    2. Simulação detalhada de ordens (slippage, taxas).
    3. Gerenciamento de estado do bot ao longo do tempo (posições, saldo).
    4. Geração de relatórios de métricas de performance (drawdown, Sharpe ratio, etc.).
    """
    
    def __init__(self, environment: str = 'testnet'):
        self.environment = environment
        # Initialize the bot in a 'simulate_backtest' environment to trigger yfinance and mock API calls
        self.bot = AIEnhancedTradingBot(environment='simulate_backtest') 
        self.logger = logging.getLogger('Backtester')
        self.logger.info(f"🔄 Backtester inicializado para ambiente: {environment.upper()}")
        
        # Override bot's real API calls with simulation mocks
        self.bot.get_current_price = self._get_historical_price
        self.bot.place_market_order = self._simulate_order_execution
        self.bot.get_account_balance = self._get_simulated_balance
        # self.bot.update_data_cache is now designed to handle yfinance or real API
        
        self.historical_data_frames = {} # Store fetched yfinance data
        self.simulated_balance = 0.0 # Initial capital for backtest
        self.simulated_positions = []
        self.current_simulation_time = None # Will track the current time in backtest
        self.trade_logs = [] # To store simulated trades
        
    def _load_initial_historical_data(self, start_date: datetime, end_date: datetime, symbols: List[str], interval: str = '5m'):
        """Loads all necessary historical data for the backtest period using yfinance."""
        self.logger.info(f"Carregando dados históricos via YFinance para backtest ({start_date.date()} a {end_date.date()})...")
        for symbol in symbols:
            yf_symbol = _convert_gateio_symbol_to_yfinance(symbol)
            
            yf_interval = {'5m': '5m', '15m': '15m', '1h': '60m'}.get(interval, '5m')
            
            # Request period slightly larger to ensure enough data for indicators
            # YFinance's 5m interval is typically limited to 7 days
            if yf_interval == '5m':
                if (end_date - start_date).days > 7:
                    self.logger.warning(f"Período de backtest para {symbol} ({start_date.date()} a {end_date.date()}) é maior que 7 dias para intervalo de 5m. O YFinance pode não fornecer dados completos. Considere usar intervalos maiores ou dados locais.")
                period_str = "7d" 
            elif yf_interval == '15m':
                period_str = "60d" 
            elif yf_interval == '60m':
                period_str = "2y" 
            else:
                period_str = "1mo" 

            try:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start_date - timedelta(days=30), end=end_date + timedelta(days=1), interval=yf_interval) 
                
                if df.empty:
                    self.logger.warning(f"Não foi possível carregar dados para {yf_symbol} de {start_date} a {end_date}. Pulando.")
                    continue
                
                df.columns = [col.lower() for col in df.columns]
                df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'}, inplace=True)
                df.index.name = 'timestamp'
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                # Resample to ensure consistent intervals (YFinance might have gaps)
                resampled_5m = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                resampled_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                resampled_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

                self.historical_data_frames[symbol] = {
                    '5m': resampled_5m,
                    '15m': resampled_15m,
                    '1h': resampled_1h
                }
                self.logger.info(f"Dados carregados para {symbol} (5m: {len(self.historical_data_frames[symbol]['5m'])} candles)")
                
            except Exception as e:
                self.logger.error(f"Erro ao carregar dados históricos para {yf_symbol} com YFinance: {e}", exc_info=True)
        self.logger.info("Concluído o carregamento de dados históricos.")


    def _get_historical_price(self, symbol: str) -> float:
        """Returns the closing price at the current simulation time."""
        if symbol in self.bot.analyzer.data_cache and '5m' in self.bot.analyzer.data_cache[symbol] and not self.bot.analyzer.data_cache[symbol]['5m'].empty:
            # Find the most recent candle data BEFORE or AT current_simulation_time
            df = self.bot.analyzer.data_cache[symbol]['5m']
            recent_candles = df[df.index <= self.current_simulation_time]
            if not recent_candles.empty:
                return recent_candles['close'].iloc[-1]
        return 0.0

    def _simulate_order_execution(self, symbol: str, side: str, contracts: float) -> Optional[dict]:
        """Simulates the execution of a market order."""
        price = self._get_historical_price(symbol)
        if price == 0:
            self.logger.error(f"Backtest: Preço inválido para {symbol} na simulação de ordem.")
            return None
        
        simulated_order = {
            "id": f"sim_order_{int(self.current_simulation_time.timestamp())}_{np.random.randint(1000,9999)}",
            "contract": symbol,
            "size": contracts if side == "buy" else -contracts,
            "price": price,
            "create_time": self.current_simulation_time.timestamp()
        }
        self.logger.info(f"Backtest: Simulating {side.upper()} order for {contracts} {symbol} @ ${price:.4f} at {self.current_simulation_time}")
        
        cost_usdt = contracts * price
        if side == "buy":
            self.simulated_balance -= cost_usdt 
        else: 
            self.simulated_balance += cost_usdt
        
        return simulated_order

    def _get_simulated_balance(self) -> float:
        """Returns the simulated balance."""
        return self.simulated_balance

    def run_backtest(self, start_date: datetime, end_date: datetime, initial_capital: float):
        """
        Executa o backtest da estratégia.
        Avança o tempo candle por candle para simular o ambiente real.
        """
        self.logger.info(f"🚀 Iniciando Backtest de {start_date.date()} a {end_date.date()} com capital inicial ${initial_capital:.2f}")
        self.simulated_balance = initial_capital
        self.bot.performance["start_balance"] = initial_capital
        self.bot.performance["current_balance"] = initial_capital
        
        # Reset bot's internal state for backtesting
        self.bot.active_positions = []
        self.bot.trade_history = []
        self.bot.signals_history = []
        self.bot.last_trade_time = {}
        self.bot.daily_trades = 0
        self.bot.daily_reset_date = start_date.date() # Start daily reset from backtest start date
        self.bot.fred_last_checked_for_cooldown = None # Reset FRED cooldown for backtest

        self.bot.performance["total_trades"] = 0
        self.bot.performance["winning_trades"] = 0
        self.bot.performance["total_pnl"] = 0.0
        self.bot.performance["ai_predictions"] = 0
        self.bot.performance["ai_correct_predictions"] = 0
        
        self.bot.analyzer.ml_predictor.training_data = [] # Reset ML training data for fresh start or re-load
        self.bot.analyzer.ml_predictor.last_retrain = None
        self.bot.analyzer.ml_predictor.load_model() # Attempt to load pre-trained model for backtest if available

        # Load all necessary historical data at once
        self._load_initial_historical_data(start_date, end_date, TRADING_SYMBOLS, '5m')
        
        # Determine the earliest start time considering the feature window for indicators
        earliest_candle_needed = start_date - timedelta(days=max(AI_CONFIG['feature_window'] * 5 // (24*60), 5)) 
        
        all_timestamps = []
        for symbol in TRADING_SYMBOLS:
            if symbol in self.historical_data_frames and '5m' in self.historical_data_frames[symbol]:
                df_symbol = self.historical_data_frames[symbol]['5m']
                filtered_timestamps = df_symbol.index[(df_symbol.index >= earliest_candle_needed) & (df_symbol.index <= end_date)].tolist()
                all_timestamps.extend(filtered_timestamps)
        all_timestamps = sorted(list(set(all_timestamps)))

        if not all_timestamps:
            self.logger.error("Data histórica insuficiente ou período inválido para o backtest. Verifique datas e disponibilidade de dados.")
            return
            
        self.logger.info(f"Iniciando simulação candle a candle a partir de {all_timestamps[0]} até {all_timestamps[-1]}...")
        
        log_interval_steps = (len(all_timestamps) // 10) or 1 # Log progress 10 times

        for i, current_timestamp in enumerate(all_timestamps):
            self.current_simulation_time = current_timestamp
            self.bot.performance["last_update"] = self.current_simulation_time.isoformat()

            # Simulate data update for the current timestamp for all relevant timeframes
            for symbol in TRADING_SYMBOLS:
                if symbol in self.historical_data_frames:
                    for tf, df_data in self.historical_data_frames[symbol].items():
                        recent_df = df_data[df_data.index <= self.current_simulation_time].tail(max(AI_CONFIG['feature_window'], 50))
                        if not recent_df.empty:
                            if symbol not in self.bot.analyzer.data_cache:
                                self.bot.analyzer.data_cache[symbol] = {}
                            self.bot.analyzer.data_cache[symbol][tf] = recent_df
                            self.bot.analyzer.last_cache_update[f"{symbol}_{tf}"] = self.current_simulation_time
            
            # === Backtest FRED integration ===
            # Override datetime.now() for FRED calendar to match simulation time
            self.bot.analyzer.fred_calendar.datetime_now_override = self.current_simulation_time
            
            # Manually trigger FRED cache update for backtesting
            self.bot.analyzer._update_fred_events_cache()


            # 2. Manage simulated positions (check for SL/TP, close if hit)
            self.bot.manage_positions()

            # 3. Scan for new signals
            self.bot.scan_markets_with_ai()
            
            # 4. Simulate retraining (less frequent than every candle)
            if ML_AVAILABLE and self.bot.performance["total_trades"] >= AI_CONFIG['min_training_samples'] and \
               (self.bot.analyzer.ml_predictor.last_retrain is None or \
               (self.current_simulation_time - self.bot.analyzer.ml_predictor.last_retrain).total_seconds() / 3600 >= AI_CONFIG['model_retrain_interval_hours']):
                self.logger.info(f"Backtest: Retraining model at {self.current_simulation_time}...")
                self.bot.analyzer.ml_predictor.train_model()

            # Periodically log status during backtest
            if (i % log_interval_steps == 0 and i > 0) or (i == len(all_timestamps) - 1):
                self.logger.info(f"Backtest Progress: {self.current_simulation_time.strftime('%Y-%m-%d %H:%M')}")
                self.bot.log_ai_periodic_status() 
        
        self.logger.info("✅ Backtest concluído!")
        self._generate_backtest_report()

    def _generate_backtest_report(self):
        """Gera um relatório de performance detalhado do backtest."""
        self.logger.info("=" * 70)
        self.logger.info("📊 RELATÓRIO DE BACKTEST")
        self.logger.info("=" * 70)
        
        # Recalculate final balance based on initial capital and total PnL from trade_history
        final_balance = self.bot.performance["start_balance"] + self.bot.performance["total_pnl"]
        
        self.logger.info(f"Capital Inicial: ${self.bot.performance['start_balance']:.2f}")
        self.logger.info(f"Capital Final: ${final_balance:.2f}")
        self.logger.info(f"PnL Total: ${self.bot.performance['total_pnl']:.2f}")
        self.logger.info(f"ROI: {self.bot.performance['roi_percentage']:.2f}%")
        self.logger.info(f"Total de Trades: {self.bot.performance['total_trades']}")
        self.logger.info(f"Trades Vencedores: {self.bot.performance['winning_trades']}")
        self.logger.info(f"Taxa de Vitória: {self.bot.performance['win_rate']:.2f}%")
        
        self.logger.info(f"Acurácia da IA (Trades no Backtest): {self.bot.performance['ai_accuracy']:.2f}%")
        
        if ML_AVAILABLE:
            self.logger.info(f"Acurácia do Modelo ML Treinado: {self.bot.analyzer.ml_predictor.model_accuracy:.3f}")
            self.logger.info(f"Amostras de Treino Utilizadas: {len(self.bot.analyzer.ml_predictor.training_data)}")

        self.logger.info("=" * 70)

# ============================================================================
# 🚀 WRAPPER PARA INTEGRAÇÃO
# ============================================================================

class CombinedAITradingBot(AIEnhancedTradingBot):
    """Wrapper para compatibilidade com FastAPI"""
    
    def __init__(self, environment: str = 'testnet'):
        super().__init__(environment=environment)
        self.logger.info(f"🤖 CombinedAITradingBot v2.1 inicializado - Ambiente: {self.environment.upper()}")
        
        # Validar credenciais
        if not API_KEY or not SECRET:
            self.logger.error("❌ Credenciais API da Gate.io não configuradas!")
        
        self.logger.info(f"🔑 API Gate.io configurada: {'...' + API_KEY[-5:] if API_KEY else 'N/A'}")
        
        if not NEWS_API_KEY and SENTIMENT_AVAILABLE:
            self.logger.warning("⚠️ NEWS_API_KEY não configurada. A análise de sentimento usará dados simulados ou limitados.")
        elif NEWS_API_KEY and SENTIMENT_AVAILABLE:
            self.logger.info("✅ NEWS_API_KEY configurada. Análise de sentimento real ativada.")

        # Testar conectividade apenas se não for simulação
        if environment not in ['simulate_backtest']: 
            try:
                balance = self.get_account_balance()
                self.logger.info(f"✅ Conectividade Gate.io testada - Saldo: ${balance:.2f}")
            except Exception as e:
                self.logger.error(f"❌ Falha no teste de conectividade Gate.io: {e}")
    
    def run_trading_loop(self):
        """Método para execução em thread separada"""
        try:
            self.run()
        except Exception as e:
            self.logger.error(f"Erro na thread de trading: {e}", exc_info=True)
    
    def get_detailed_ai_status(self) -> Dict:
        """Status detalhado com métricas de IA"""
        try:
            current_balance = self.get_account_balance() 
            
            # Recalculate metrics
            if self.performance["start_balance"] > 0:
                self.performance["roi_percentage"] = (self.performance["total_pnl"] / self.performance["start_balance"]) * 100
            
            if self.performance["total_trades"] > 0:
                self.performance["win_rate"] = (self.performance["winning_trades"] / self.performance["total_trades"]) * 100
            
            if self.performance["ai_predictions"] > 0:
                self.performance["ai_accuracy"] = (self.performance["ai_correct_predictions"] / self.performance["ai_predictions"]) * 100
            
            # Format active positions
            simple_active_positions = []
            for pos in self.active_positions:
                try:
                    simple_active_positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': pos['size'],
                        'entry_price': round(pos['entry_price'], 4),
                        'current_price': round(pos.get('current_price', 0.0), 4),
                        'pnl': round(pos.get('pnl', 0.0), 2),
                        'pnl_percent': round(pos.get('pnl_percent', 0.0), 2),
                        'duration_minutes': (datetime.now() - pos.get('timestamp', datetime.now())).total_seconds() / 60,
                        'ai_prediction': round(pos.get('ai_prediction', 0.5), 3),
                        'pattern_score': round(pos.get('pattern_score', 0.5), 3),
                        'sentiment_score': round(pos.get('sentiment_score', 0.5), 3),
                        'timestamp': pos.get('timestamp').isoformat() if pos.get('timestamp') else None
                    })
                except Exception as e:
                    self.logger.error(f"Erro ao formatar posição: {e}")
                    continue
            
            return {
                "status": "running" if self.running else "stopped",
                "environment": self.environment,
                "version": "2.1_ai_enhanced",
                
                # Performance Principal
                "current_balance": round(current_balance, 2),
                "start_balance": round(self.performance["start_balance"], 2),
                "total_pnl": round(self.performance["total_pnl"], 2),
                "roi_percentage": round(self.performance["roi_percentage"], 2),
                "winning_trades": self.performance["winning_trades"],
                "total_trades": self.performance["total_trades"],
                "win_rate": round(self.performance["win_rate"], 2),
                "daily_pnl": round(self.performance["daily_pnl"], 2),
                "daily_trades": self.performance["daily_trades"],
                
                # Métricas de IA
                "ai_accuracy": round(self.performance["ai_accuracy"], 2),
                "ai_predictions": self.performance["ai_predictions"],
                "ai_correct_predictions": self.performance["ai_correct_predictions"],
                "ml_model_accuracy": round(self.analyzer.ml_predictor.model_accuracy, 3) if ML_AVAILABLE and self.analyzer.ml_predictor.model_accuracy else 0,
                "training_samples": len(self.analyzer.ml_predictor.training_data),
                
                # Timestamps
                "last_update": self.performance["last_update"] or datetime.now().isoformat(),
                "session_start_time": self.performance["session_start_time"],
                
                # Posições com dados de IA
                "active_positions_count": len(self.active_positions),
                "active_positions": simple_active_positions,
                
                # Histórico
                "recent_signals_count": len(self.signals_history),
                "recent_signals": [s.to_dict() for s in self.signals_history[-10:]],
                "trade_history_count": len(self.trade_history),
                "trade_history": self.trade_history[-20:], # Only show recent
                
                # Configurações
                "config": {
                    "ai_prediction_threshold": AI_CONFIG['prediction_threshold'],
                    "model_retrain_interval_hours": AI_CONFIG['model_retrain_interval_hours'],
                    "pattern_memory_size": AI_CONFIG['pattern_memory_size'],
                    "sentiment_weight": AI_CONFIG['sentiment_weight'],
                    "position_size_usdt": RISK_CONFIG[self.environment]['position_size_usdt'],
                    "max_open_positions": RISK_CONFIG[self.environment]['max_open_positions'],
                    "ai_confidence_boost": RISK_CONFIG[self.environment]['ai_confidence_boost']
                },
                
                # Status dos Componentes de IA
                "ai_system_status": {
                    "ml_available": ML_AVAILABLE,
                    "xgb_available": XGB_AVAILABLE,
                    "sentiment_available": SENTIMENT_AVAILABLE,
                    "model_trained": self.analyzer.ml_predictor.model is not None,
                    "last_retrain": self.analyzer.ml_predictor.last_retrain.isoformat() if self.analyzer.ml_predictor.last_retrain else None,
                    "pattern_database_size": len(self.analyzer.pattern_matcher.patterns),
                    "sentiment_cache_size": len(self.analyzer.sentiment_analyzer.sentiment_cache),
                    "fred_calendar_active": (FRED_API_KEY is not None and FRED_API_KEY != "DUMMY_KEY_FRED"),
                    "fred_events_cached": len(self.analyzer.fred_events_cache),
                    "fred_last_update": self.analyzer.last_fred_update.isoformat() if self.analyzer.last_fred_update else None,
                    "cryptopanic_active": (CRYPTOPANIC_API_KEY is not None and CRYPTOPANIC_API_KEY != "DUMMY_KEY_CRYPTOPANIC")
                },
                
                # Features da versão IA
                "ai_features": {
                    "machine_learning": ML_AVAILABLE,
                    "pattern_matching": True,
                    "sentiment_analysis": SENTIMENT_AVAILABLE,
                    "xgboost_available": XGB_AVAILABLE,
                    "automatic_retraining": True,
                    "feature_engineering": True,
                    "multi_model_ensemble": True,
                    "kelly_criterion_sizing": True,
                    "fred_economic_calendar_integration": True,
                    "cryptopanic_news_integration": True # New feature
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório de IA: {e}", exc_info=True)
            return {"error": f"Erro no relatório: {str(e)}"}
    
    def force_model_retrain(self) -> bool:
        """Força retreinamento do modelo"""
        if not ML_AVAILABLE:
            self.logger.warning("Machine Learning não disponível. Não é possível forçar retreinamento.")
            return False
        try:
            self.logger.info("🔄 Forçando retreinamento do modelo...")
            return self.analyzer.ml_predictor.train_model()
        except Exception as e:
            self.logger.error(f"Erro ao forçar retreinamento: {e}")
            return False

# ============================================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    # Banner do bot
    print("=" * 80)
    print("🤖 BOT DE TRADING COM IA LEVE v2.1")
    print("🧠 Machine Learning + Análise de Sentimento + Padrões Históricos")
    print("✨ Melhorias: Mais Indicadores, Gerenciamento de Risco Kelly, Backtesting (Esqueleto)")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description='Bot de Trading com IA v2.1')
    parser.add_argument('--env', choices=['testnet', 'live'], default='testnet',
                        help='Ambiente de execução (testnet ou live)')
    parser.add_argument('--train-only', action='store_true',
                        help='Apenas treinar modelo sem fazer trades')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Forçar retreinamento do modelo existente')
    parser.add_argument('--backtest', action='store_true',
                        help='Rodar backtesting em vez de live trading')
    parser.add_argument('--backtest-start', type=str,
                        help='Data de início para backtesting (YYYY-MM-DD)')
    parser.add_argument('--backtest-end', type=str,
                        help='Data de fim para backtesting (YYYY-MM-DD)')
    parser.add_argument('--backtest-capital', type=float, default=1000.0,
                        help='Capital inicial para backtesting')
    parser.add_argument('--debug', action='store_true',
                        help='Modo debug com logs detalhados')
    
    args = parser.parse_args()
    
    # Configurar nível de log
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Modo DEBUG ativado")
    
    # Verificar dependências críticas
    print("\n🔍 Verificando dependências...")
    
    missing_deps = []
    if not ML_AVAILABLE:
        missing_deps.append("scikit-learn")
    if not XGB_AVAILABLE:
        missing_deps.append("xgboost")
    if not SENTIMENT_AVAILABLE:
        missing_deps.append("textblob")
    if not YFINANCE_AVAILABLE:
        missing_deps.append("yfinance")
    
    if missing_deps:
        print(f"⚠️ ATENÇÃO: Dependências opcionais não disponíveis: {', '.join(missing_deps)}")
        print("📦 Para funcionalidade completa, instale:")
        print(f"    pip install {' '.join(missing_deps)}")
        
        if not ML_AVAILABLE:
            print("❌ ERRO: Machine Learning é obrigatório para esta versão!")
            print("💡 Instale com: pip install scikit-learn joblib numpy pandas")
            sys.exit(1)
        
        print("⚡ Continuando com funcionalidades limitadas...\n")
    else:
        print("✅ Todas as dependências estão disponíveis\n")
    
    # Verificar credenciais Gate.io apenas se não for backtest
    if not args.backtest:
        if not API_KEY or not SECRET:
            print("❌ ERRO: Credenciais da API da Gate.io não configuradas!")
            print("📝 Configure no arquivo .env:")
            print("    GATE_TESTNET_API_KEY=sua_key")
            print("    GATE_TESTNET_API_SECRET=seu_secret")
            sys.exit(1)
        
        print(f"🔑 Credenciais Gate.io configuradas: ...{API_KEY[-5:] if API_KEY else 'N/A'}")
    else:
        print("💡 Modo Backtest: Credenciais da API da Gate.io não são estritamente necessárias para a simulação.")

    # Verificar credenciais NewsAPI
    if SENTIMENT_AVAILABLE:
        if not NEWS_API_KEY:
            print("⚠️ ATENÇÃO: NEWS_API_KEY não configurada. A análise de sentimento usará dados simulados ou limitados.")
            print("📝 Para análise de sentimento real, adicione no arquivo .env:")
            print("    NEWS_API_KEY=sua_key_newsapi")
        else:
            print(f"🔑 Credenciais NewsAPI configuradas: ...{NEWS_API_KEY[-5:]}")

    # Verificar credenciais FRED API
    if FRED_API_KEY == "DUMMY_KEY_FRED":
        print("⚠️ ATENÇÃO: FRED_API_KEY não configurada. O calendário econômico do FRED não será funcional.")
        print("📝 Para usar o FRED, adicione no arquivo config.py:")
        print("    FRED_API_KEY='SUA_CHAVE_API_FRED_AQUI'")
    else:
        print(f"🔑 Credenciais FRED API configuradas: ...{FRED_API_KEY[-5:] if FRED_API_KEY else 'N/A'}")

    # Verificar credenciais CryptoPanic API
    if CRYPTOPANIC_API_KEY == "DUMMY_KEY_CRYPTOPANIC":
        print("⚠️ ATENÇÃO: CRYPTOPANIC_API_KEY não configurada. A análise de sentimento da CryptoPanic não será funcional.")
        print("📝 Para usar a CryptoPanic, adicione no arquivo config.py:")
        print("    CRYPTOPANIC_API_KEY='SUA_CHAVE_API_CRYPTOPANIC_AQUI'")
    else:
        print(f"🔑 Credenciais CryptoPanic API configuradas: ...{CRYPTOPANIC_API_KEY[-5:] if CRYPTOPANIC_API_KEY else 'N/A'}")


    # Aviso para ambiente live
    if args.env == 'live' and not args.backtest:
        print("\n" + "🚨" * 30)
        print("⚠️  ATENÇÃO: MODO LIVE SELECIONADO!")
        print("💰 Este bot irá operar com DINHEIRO REAL!")
        print("🚨" * 30)
        
        confirm = input("\nDigite 'CONFIRMO' para continuar com dinheiro real: ")
        if confirm != 'CONFIRMO':
            print("❌ Operação cancelada pelo usuário")
            sys.exit(0)
        
        print("💰 MODO LIVE CONFIRMADO - Iniciando em 5 segundos...")
        time.sleep(5)
    
    bot = None # Initialize bot to None
    backtester = None # Initialize backtester to None

    try:
        if args.backtest:
            if not args.backtest_start or not args.backtest_end:
                print("❌ ERRO: Para backtesting, --backtest-start e --backtest-end são obrigatórios (YYYY-MM-DD).")
                sys.exit(1)
            
            try:
                start_date = datetime.strptime(args.backtest_start, '%Y-%m-%d')
                end_date = datetime.strptime(args.backtest_end, '%Y-%m-%d').replace(hour=23, minute=59, second=59) # End of day
            except ValueError:
                print("❌ ERRO: Formato de data inválido. Use%Y-%m-%d.")
                sys.exit(1)
            
            print(f"\n🎮 Inicializando Backtester para o período: {start_date.date()} a {end_date.date()}")
            backtester = Backtester(environment=args.env) # Pass original env, Backtester will set its internal bot env to simulate_backtest
            backtester.run_backtest(start_date, end_date, args.backtest_capital)

        # Modo apenas treinamento
        elif args.train_only:
            print("🧠 Modo treinamento apenas...")
            bot = CombinedAITradingBot(environment=args.env)

            if len(bot.analyzer.ml_predictor.training_data) < AI_CONFIG['min_training_samples']:
                print(f"❌ Dados insuficientes para treinar: {len(bot.analyzer.ml_predictor.training_data)}")
                print(f"📊 Necessário pelo menos: {AI_CONFIG['min_training_samples']} amostras")
                print("💡 Execute o bot normal primeiro para coletar dados")
                sys.exit(1)
            
            success = bot.analyzer.ml_predictor.train_model()
            if success:
                print("✅ Treinamento concluído com sucesso!")
                print(f"📊 Accuracy: {bot.analyzer.ml_predictor.model_accuracy:.3f}")
                print(f"📈 Amostras de treino: {len(bot.analyzer.ml_predictor.training_data)}")
                bot.analyzer.ml_predictor.save_model()
                print("💾 Modelo salvo")
            else:
                print("❌ Falha no treinamento")
                sys.exit(1)
            
        # Forçar retreinamento
        elif args.force_retrain:
            print("🔄 Forçando retreinamento do modelo...")
            bot = CombinedAITradingBot(environment=args.env)
            success = bot.force_model_retrain()
            if success:
                print("✅ Retreinamento concluído!")
            else:
                print("❌ Falha no retreinamento")
                sys.exit(1)
            
        # Modo normal de trading
        else:
            print(f"\n🚀 Iniciando bot de trading com IA...")
            print(f"🌍 Ambiente: {args.env.upper()}")
            print(f"🧠 IA ativada: {'SIM' if ML_AVAILABLE else 'NÃO'}")
            print(f"⚙️ Configurações:")
            print(f"    • Posições máx: {RISK_CONFIG[args.env]['max_open_positions']}")
            print(f"    • Trades/dia máx: {RISK_CONFIG[args.env]['max_daily_trades']}")
            print(f"    • Tamanho posição (Base): ${RISK_CONFIG[args.env]['position_size_usdt']}")
            print(f"    • Threshold IA: {AI_CONFIG['prediction_threshold']}")
            
            print(f"\n⏰ Iniciando em 3 segundos...")
            time.sleep(3)
            
            # Executar bot
            bot = CombinedAITradingBot(environment=args.env)
            bot.run()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupção pelo usuário detectada")
        
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        
    finally:
        try:
            if bot is not None and bot.running: # Only shutdown if it was actively running
                print("🔄 Encerrando bot de forma segura...")
                bot.stop()
                
                # Exibir estatísticas finais
                if bot.performance["total_trades"] > 0:
                    print("\n📊 ESTATÍSTICAS FINAIS:")
                    print(f"    💰 PnL Total: ${bot.performance['total_pnl']:.2f}")
                    print(f"    📈 ROI: {bot.performance['roi_percentage']:.2f}%")
                    print(f"    🎯 Win Rate: {bot.performance['win_rate']:.1f}%")
                    print(f"    🤖 IA Accuracy: {bot.performance['ai_accuracy']:.1f}%")
                    print(f"    📊 Trades Total: {bot.performance['total_trades']}")
                
                print("✅ Bot encerrado com segurança")
            elif backtester is not None:
                print("\n✅ Backtest finalizado.")
                # The backtester already prints its report
                
        except Exception as shutdown_error:
            print(f"⚠️ Erro no encerramento: {shutdown_error}")
            
        print("\n👋 Obrigado por usar o Bot de Trading com IA!")
        print("🔗 GitHub: https://github.com/seu-usuario/ai-trading-bot")
        print("📧 Suporte: support@seudominio.com")