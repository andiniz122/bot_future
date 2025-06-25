from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Deque # Added Deque import
import math
from scipy import stats
import pytz
import requests
import json
from dataclasses import dataclass
import threading
import time
import random
import asyncio
import aiohttp
import talib
import websockets # Added websockets import
from collections import deque # Added deque import

# ===============================================================================
# 🔧 CONFIGURAÇÕES (GLOBAL VARIABLES)
# ===============================================================================

# Símbolos dos ativos - CORRIGIDO para corresponder ao frontend
SYMBOLS = {
    'gold': 'GC=F',
    'btc': 'BTC-USD',
    'dxy': 'DX-Y.NYB'
}

# Cores dos ativos
ASSET_COLORS = {
    'gold': '#FFD700',
    'btc': '#FF8C00',
    'dxy': '#00CC66'
}

# Chave da API FRED
FRED_API_KEY = "4533c6f5e65f2377d74e594577b3eae9"
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Configurações de cache
CACHE_CONFIG = {
    "price_data_duration": 30,
    "fred_data_duration": 1800,
    "max_alerts": 100,
    "max_price_history": 1000,
    "max_angular_history": 500
}

# Intervalos de atualização
UPDATE_INTERVALS = {
    "angular_analysis": 60,
    "fred_data": 1800
}

# Thresholds para alertas angulares
ANGULAR_ALERT_THRESHOLDS = {
    "perfect_divergence": {
        "dxy_min_angle": 15,
        "btc_max_angle": -15,
        "min_strength": 0.6
    },
    "bullish_convergence": {
        "btc_min_angle": 10,
        "gold_min_angle": 5,
        "dxy_max_angle": -5
    },
    "bearish_avalanche": {
        "btc_max_angle": -10,
        "gold_max_angle": -5,
        "dxy_min_angle": 10
    },
    "trend_reversal": {
        "min_angle_change": 20
    },
    "extreme_momentum": {
        "min_angle": 30,
        "min_strength": 0.7
    }
}

# Configurações do servidor
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "log_level": "info"
}

# Configurações CORS
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

# Informações do sistema
SYSTEM_INFO = {
    "version": "6.0-compatible",
    "description": "Sistema de Trading Bot Compatível com Frontend v6.0",
    "features": [
        "📊 Dados de mercado em tempo real",
        "📐 Análise angular avançada",
        "🚨 Sistema de alertas inteligente",
        "📅 Calendário econômico FRED integrado",
        "🎯 Detecção de padrões complexos",
        "🤖 Trading Bot Simulado Completo"
    ]
}

# ===============================================================================
# 🔧 CONFIGURAÇÃO DE LOGGING
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# 💾 CACHE PRINCIPAL E ESTADOS GLOBAIS
# ===============================================================================

cache = {
    "data": None,
    "timestamp": None,
    "cache_duration": CACHE_CONFIG["price_data_duration"],
    "price_history": [],
    "alerts": [],
    "angular_data": [],
    "trend_analysis": None,
    "last_alert_check": None,
    "last_angular_analysis": None,
    "fred_data": {
        "upcoming_events": [],
        "pre_event_alerts": [],
        "last_fred_update": None,
        "fred_cache_duration": CACHE_CONFIG["fred_data_duration"]
    }
}

# Cache específico para sentimento de mercado
sentiment_cache = {
    "btc_sentiment": {
        "buyers": 50.0,
        "sellers": 50.0,
        "total_bids": 0.0,
        "total_asks": 0.0,
        "last_update": None,
        "trend": "NEUTRAL",
        "volume_24h": "$0B",
        "bid_ask_ratio": 1.0
    },
    "paxg_sentiment": {
        "buyers": 50.0,
        "sellers": 50.0,
        "total_bids": 0.0,
        "total_asks": 0.0,
        "last_update": None,
        "trend": "NEUTRAL",
        "volume_24h": "$0B",
        "bid_ask_ratio": 1.0
    },
    "fear_greed_index": 50,
    "market_mood": "NEUTRAL",
    "websocket_connected": False,
    "sentiment_history": []
}

# Cache para as recomendações do backtest
backtest_recommendations_cache = {
    "recommendations": [],
    "last_update": None,
    "update_interval_minutes": 20
}

# Estados do Trading Bot (compatível com frontend)
trading_bot_state = {
    "running": False,
    "last_update": None,
    "active_positions": [],
    "active_signals": [],
    "performance": {
        "current_balance": 10000.0,
        "start_balance": 10000.0,
        "total_pnl": 0.0,
        "winning_trades": 0,
        "total_trades": 0,
        "max_drawdown": 0.0,
        "daily_pnl": 0.0,
        "daily_trades": 0,
        "last_update": None
    }
}

# Lista de conexões WebSocket ativas
active_sentiment_websocket_connections: List[WebSocket] = []

# ===============================================================================
# 📊 CACHE PARA DADOS TEMPO REAL (NOVO)
# ===============================================================================

# Cache para dados OHLCV em tempo real
realtime_ohlcv_cache = {
    "btc": {
        "candles": deque(maxlen=200),  # Últimos 200 candles para MACD
        "current_candle": None,
        "macd_data": {
            "macd": [],
            "signal": [],
            "histogram": [],
            "last_crossover": None,
            "crossover_alerts": []
        },
        "last_update": None,
        "websocket_connected": False
    },
    "volume_realtime": deque(maxlen=100),
    "price_updates": deque(maxlen=50)
}

# Lista de conexões WebSocket para dados OHLCV
active_ohlcv_websocket_connections: List[WebSocket] = []

# ===============================================================================
# FastAPI APP INITIALIZATION
# ===============================================================================

app = FastAPI(
    title=f"Trading Dashboard API {SYSTEM_INFO['version']} - Compatible",
    description=SYSTEM_INFO["description"],
    version=SYSTEM_INFO["version"],
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_CONFIG["allow_origins"],
    allow_credentials=CORS_CONFIG["allow_credentials"],
    allow_methods=CORS_CONFIG["allow_methods"],
    allow_headers=CORS_CONFIG["allow_headers"],
)

# APIRouter para as rotas do Trading Bot
trading_bot_router = APIRouter(prefix="/api/trading-bot")

# ===============================================================================
# HELPER FUNCTIONS AND CLASSES
# ===============================================================================

def calculate_volume_colors_and_macd(hist_data):
    """
    Calcula cores do volume baseado em alta/baixa e melhora dados MACD
    CORRIGIDO para funcionar com dados reais da Gate.io
    """
    try:
        if hist_data.empty or len(hist_data) < 2:
            logger.warning("DataFrame vazio ou insuficiente para calculate_volume_colors_and_macd")
            return hist_data
        
        # Verificar se as colunas necessárias existem
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in hist_data.columns]
        if missing_cols:
            logger.error(f"Colunas faltando no DataFrame: {missing_cols}")
            return hist_data
        
        # Adicionar coluna de direção do preço (para colorir volume)
        hist_data = hist_data.copy()  # Evitar modificar o DataFrame original
        hist_data['price_direction'] = 'neutral'
        
        # Comparar preço de fechamento atual com anterior
        for i in range(1, len(hist_data)):
            try:
                current_close = hist_data.iloc[i]['Close']
                previous_close = hist_data.iloc[i-1]['Close']
                
                if pd.isna(current_close) or pd.isna(previous_close):
                    continue
                
                if current_close > previous_close:
                    hist_data.iloc[i, hist_data.columns.get_loc('price_direction')] = 'up'
                elif current_close < previous_close:
                    hist_data.iloc[i, hist_data.columns.get_loc('price_direction')] = 'down'
                else:
                    hist_data.iloc[i, hist_data.columns.get_loc('price_direction')] = 'neutral'
            except Exception as e:
                logger.warning(f"Erro processando cor do volume no índice {i}: {e}")
                continue
        
        # Primeiro candle sempre neutral
        hist_data.iloc[0, hist_data.columns.get_loc('price_direction')] = 'neutral'
        
        # Log de estatísticas das cores
        if 'price_direction' in hist_data.columns:
            color_stats = hist_data['price_direction'].value_counts()
            logger.debug(f"Volume colors calculated: {dict(color_stats)}")
        
        return hist_data
        
    except Exception as e:
        logger.error(f"Erro ao calcular cores do volume: {e}")
        return hist_data

# 📅 FRED CALENDAR SYSTEM
class FREDEconomicCalendar:
    """Calendário econômico FRED simplificado"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = FRED_BASE_URL

    def get_upcoming_releases(self, days_ahead: int = 30) -> List[Dict]:
        """Busca próximos releases econômicos"""
        try:
            if self.api_key == "YOUR_FRED_API_KEY_HERE" or not self.api_key:
                logger.warning("FRED API Key is not configured. Using simulated FRED data.")
                return self._generate_simulated_events(days_ahead)

            # Aqui você pode implementar a integração real com FRED
            # Por enquanto, usar dados simulados
            return self._generate_simulated_events(days_ahead)

        except Exception as e:
            logger.warning(f"FRED API error: {e}. Using simulated data.")
            return self._generate_simulated_events(days_ahead)

    def _generate_simulated_events(self, days_ahead: int) -> List[Dict]:
        """Gera eventos econômicos simulados"""
        events = []
        base_date = datetime.now()

        event_templates = [
            {"name": "Non-Farm Payrolls", "category": "EMPLOYMENT", "importance": "HIGH", "impact_score": 95},
            {"name": "CPI Inflation", "category": "INFLATION", "importance": "HIGH", "impact_score": 90},
            {"name": "Federal Funds Rate", "category": "MONETARY_POLICY", "importance": "HIGH", "impact_score": 100},
            {"name": "GDP Growth", "category": "GDP", "importance": "HIGH", "impact_score": 85},
            {"name": "Unemployment Rate", "category": "EMPLOYMENT", "importance": "MEDIUM", "impact_score": 75},
            {"name": "Retail Sales", "category": "CONSUMER", "importance": "MEDIUM", "impact_score": 70},
            {"name": "Industrial Production", "category": "INDUSTRIAL", "importance": "MEDIUM", "impact_score": 65},
            {"name": "Housing Starts", "category": "HOUSING", "importance": "LOW", "impact_score": 50}
        ]

        for i in range(min(days_ahead * 2, 20)):
            event = random.choice(event_templates).copy()
            event_date = base_date + timedelta(days=random.randint(1, days_ahead))

            forecast_val = round(random.uniform(1.0, 5.0), 1)
            actual_val = forecast_val + random.uniform(-0.5, 0.5)
            actual_val = round(actual_val, 1)

            event.update({
                "date": event_date.isoformat(),
                "forecast": forecast_val,
                "previous": round(random.uniform(1.0, 5.0), 1),
                "actual": actual_val if random.random() > 0.3 else None,
                "volatility_factor": random.uniform(0.8, 2.0),
                "market_impact": random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"]),
                "affected_assets": random.sample(["BTC", "GOLD", "DXY"], k=random.randint(1, 3))
            })

            events.append(event)

        return sorted(events, key=lambda x: x["date"])

def update_fred_data():
    """Atualiza o cache de dados FRED com eventos econômicos."""
    try:
        fred_calendar = FREDEconomicCalendar(FRED_API_KEY)
        fred_data_result = integrate_fred_calendar(FRED_API_KEY, cache["data"])
        cache["fred_data"]["upcoming_events"] = fred_data_result["upcoming_events"]
        cache["fred_data"]["pre_event_alerts"] = fred_data_result["pre_event_alerts"]
        cache["fred_data"]["last_fred_update"] = datetime.now().isoformat()
        cache["alerts"].extend(fred_data_result["pre_event_alerts"])
        logger.info(f"📅 FRED data updated: {len(cache['fred_data']['upcoming_events'])} upcoming events")
    except Exception as e:
        logger.error(f"❌ Error updating FRED data: {e}")

def integrate_fred_calendar(api_key: str, current_data: Dict) -> Dict:
    """Integra calendário FRED com análise atual"""
    try:
        fred_calendar = FREDEconomicCalendar(api_key)
        upcoming_events = fred_calendar.get_upcoming_releases()

        pre_event_alerts = []
        current_time = datetime.now()

        for event in upcoming_events:
            try:
                event_date = datetime.fromisoformat(event["date"])
                hours_until = (event_date - current_time).total_seconds() / 3600

                if 0 < hours_until <= 25 and event.get("impact_score", 0) >= 75:
                    alert = {
                        "type": "PRE_EVENT_ALERT",
                        "title": f"⚠️ {event['name']} em {round(hours_until, 1)}h",
                        "message": f"Evento de alto impacto se aproximando: {event['name']} (impacto: {event.get('impact_score', 0)})",
                        "severity": "HIGH" if hours_until <= 4 else "MEDIUM",
                        "timestamp": current_time.isoformat(),
                        "event": event,
                        "hours_until": round(hours_until, 1)
                    }
                    pre_event_alerts.append(alert)
            except Exception as e:
                logger.warning(f"Error processing FRED event: {e}")
                continue

        return {
            "status": "active",
            "upcoming_events": upcoming_events,
            "pre_event_alerts": pre_event_alerts,
            "total_events": len(upcoming_events),
            "high_impact_events": len([e for e in upcoming_events if e.get("importance") == "HIGH"])
        }

    except Exception as e:
        logger.error(f"Error integrating FRED calendar: {e}")
        return {
            "status": "error",
            "error": str(e),
            "upcoming_events": [],
            "pre_event_alerts": []
        }

# 📐 ANGULAR ANALYSIS SYSTEM
def calculate_trend_angle(prices: List[float], time_window: int = 5) -> Dict:
    """Calcula o ângulo de tendência usando regressão linear."""
    if len(prices) < time_window:
        return {
            "angle": 0,
            "strength": 0,
            "r_squared": 0,
            "trend": "NEUTRAL",
            "velocity": 0
        }

    try:
        recent_prices = prices[-time_window:]
        x = np.arange(len(recent_prices))

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
        angle = math.degrees(math.atan(slope))
        r_squared = r_value ** 2

        if angle > 30 and r_squared > 0.7:
            trend = "STRONG_UP"
        elif angle > 10 and r_squared > 0.4:
            trend = "WEAK_UP"
        elif angle < -30 and r_squared > 0.7:
            trend = "STRONG_DOWN"
        elif angle < -10 and r_squared > 0.4:
            trend = "WEAK_DOWN"
        else:
            trend = "NEUTRAL"

        return {
            "angle": round(angle, 2),
            "strength": round(r_squared, 3),
            "r_squared": round(r_squared, 3),
            "trend": trend,
            "velocity": round(slope, 6),
            "p_value": round(p_value, 4)
        }

    except Exception as e:
        logger.error(f"❌ Error calculating trend angle: {e}")
        return {
            "angle": 0,
            "strength": 0,
            "r_squared": 0,
            "trend": "NEUTRAL",
            "velocity": 0
        }

def analyze_angular_patterns(angular_history: List[Dict]) -> Dict:
    """Analisa padrões angulares complexos."""
    if len(angular_history) < 3:
        return {"patterns": [], "confidence": 0, "total_patterns": 0}

    try:
        patterns = []
        latest = angular_history[-1]
        prev = angular_history[-2] if len(angular_history) > 1 else latest

        required_keys = ["btc", "gold", "dxy"]
        for key in required_keys:
            if key not in latest or not isinstance(latest[key], dict):
                logger.warning(f"Missing or invalid '{key}' data in angular analysis")
                return {"patterns": [], "confidence": 0, "total_patterns": 0}

        # Perfect Divergence (DXY ⬆️ + BTC ⬇️)
        if (latest["dxy"]["angle"] > 15 and latest["dxy"]["strength"] > 0.6 and
            latest["btc"]["angle"] < -15 and latest["btc"]["strength"] > 0.6):
            patterns.append({
                "name": "PERFECT_DIVERGENCE",
                "title": "🎯 Perfect Divergence Detected",
                "description": f"DXY rising {latest['dxy']['angle']:.1f}° while BTC falling {abs(latest['btc']['angle']):.1f}°",
                "severity": "HIGH",
                "confidence": min(latest["dxy"]["strength"], latest["btc"]["strength"]),
                "type": "DIVERGENCE"
            })

        # Bullish Convergence
        if (latest["btc"]["angle"] > 10 and latest["gold"]["angle"] > 5 and latest["dxy"]["angle"] < -5):
            patterns.append({
                "name": "BULLISH_CONVERGENCE",
                "title": "🚀 Bullish Convergence Pattern",
                "description": f"BTC +{latest['btc']['angle']:.1f}°, Gold +{latest['gold']['angle']:.1f}°, DXY {latest['dxy']['angle']:.1f}°",
                "severity": "MEDIUM",
                "confidence": (latest["btc"]["strength"] + latest["gold"]["strength"]) / 2,
                "type": "CONVERGENCE"
            })

        # Bearish Avalanche
        if (latest["btc"]["angle"] < -10 and latest["gold"]["angle"] < -5 and latest["dxy"]["angle"] > 10):
            patterns.append({
                "name": "BEARISH_AVALANCHE",
                "title": "📉 Bearish Avalanche Detected",
                "description": f"Risk assets falling: BTC {latest['btc']['angle']:.1f}°, Gold {latest['gold']['angle']:.1f}°",
                "severity": "HIGH",
                "confidence": (latest["btc"]["strength"] + latest["gold"]["strength"] + latest["dxy"]["strength"]) / 3,
                "type": "AVALANCHE"
            })

        avg_confidence = sum([p["confidence"] for p in patterns]) / len(patterns) if patterns else 0

        return {
            "patterns": patterns,
            "total_patterns": len(patterns),
            "confidence": round(avg_confidence, 3),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error in angular pattern analysis: {e}")
        return {"patterns": [], "confidence": 0, "total_patterns": 0}

def update_angular_analysis(current_data: Dict):
    """Atualiza análise angular com dados atuais"""
    try:
        current_time = datetime.now()

        if len(cache["price_history"]) < 5:
            return

        if not all(asset_key in current_data and 'current_price' in current_data[asset_key] for asset_key in SYMBOLS.keys()):
            logger.warning("Current data is incomplete for angular analysis")
            return

        # Garante que há pelo menos 10 pontos para a tendência
        points_for_trend = 10
        if len(cache["price_history"]) < points_for_trend:
            logger.warning(f"Insufficient historical data ({len(cache['price_history'])} points) for robust angular analysis. Skipping.")
            return

        btc_prices = [point["btc_price"] for point in cache["price_history"][-points_for_trend:]]
        gold_prices = [point["gold_price"] for point in cache["price_history"][-points_for_trend:]]
        dxy_prices = [point["dxy_price"] for point in cache["price_history"][-points_for_trend:]]

        # Adiciona o preço mais atual
        btc_prices.append(current_data["btc"]["current_price"])
        gold_prices.append(current_data["gold"]["current_price"])
        dxy_prices.append(current_data["dxy"]["current_price"])

        angular_point = {
            "timestamp": current_time.isoformat(),
            "btc": calculate_trend_angle(btc_prices, 5),
            "gold": calculate_trend_angle(gold_prices, 5),
            "dxy": calculate_trend_angle(dxy_prices, 5)
        }

        cache["angular_data"].append(angular_point)

        if len(cache["angular_data"]) > CACHE_CONFIG["max_angular_history"]:
            cache["angular_data"] = cache["angular_data"][-CACHE_CONFIG["max_angular_history"]:]

        cache["last_angular_analysis"] = current_time.isoformat()

        logger.debug(f"📐 Angular analysis updated: {len(cache['angular_data'])} points")

    except Exception as e:
        logger.error(f"❌ Error updating angular analysis: {e}")

# 📊 MARKET DATA COLLECTION SYSTEM
def format_volume(volume: int) -> str:
    """Formata volume para exibição (K, M, B)"""
    if volume >= 1_000_000_000:
        return f"{volume/1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.1f}K"
    else:
        return str(volume)

def get_current_market_data() -> Dict:
    """Coleta dados atuais de mercado usando múltiplas estratégias."""
    current_data = {}

    for name, symbol in SYMBOLS.items():
        try:
            logger.debug(f"📊 Fetching data for {name} ({symbol})")

            current_price = 0
            current_volume = 0
            market_open = 0
            day_high = 0
            day_low = 0
            previous_close = 0

            # Strategy 1: ticker.info
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                price_keys = ['regularMarketPrice', 'currentPrice', 'previousClose', 'price']
                for key in price_keys:
                    if info.get(key) is not None and info.get(key) > 0:
                        current_price = float(info.get(key))
                        break

                volume_keys = ['regularMarketVolume', 'volume', 'averageVolume']
                for key in volume_keys:
                    if info.get(key) is not None and info.get(key) > 0:
                        current_volume = int(info.get(key))
                        break

                market_open = float(info.get('regularMarketOpen', 0) or info.get('open', 0) or current_price)
                day_high = float(info.get('dayHigh', 0) or info.get('regularMarketDayHigh', 0) or current_price)
                day_low = float(info.get('dayLow', 0) or info.get('regularMarketDayLow', 0) or current_price)
                previous_close = float(info.get('previousClose', 0) or info.get('regularMarketPreviousClose', 0) or current_price)

            except Exception as e:
                logger.warning(f"⚠️ Info method failed for {symbol}: {e}")

            # Strategy 2: ticker.history (fallback)
            if current_price == 0 or current_volume == 0:
                try:
                    hist_data = yf.download(symbol, period="2d", progress=False, threads=False)
                    if not hist_data.empty:
                        if current_price == 0 and 'Close' in hist_data.columns:
                            current_price = float(hist_data['Close'].iloc[-1])
                        if current_volume == 0 and 'Volume' in hist_data.columns:
                            current_volume = int(hist_data['Volume'].iloc[-1])
                        if market_open == 0 and 'Open' in hist_data.columns:
                            market_open = float(hist_data['Open'].iloc[-1])
                        if day_high == 0 and 'High' in hist_data.columns:
                            day_high = float(hist_data['High'].iloc[-1])
                        if day_low == 0 and 'Low' in hist_data.columns:
                            day_low = float(hist_data['Low'].iloc[-1])
                        if previous_close == 0 and len(hist_data) > 1:
                            previous_close = float(hist_data['Close'].iloc[-2])

                except Exception as e:
                    logger.warning(f"⚠️ History method failed for {symbol}: {e}")

            # Final fallbacks
            if current_price == 0:
                if name == 'btc': current_price = 70000.0
                elif name == 'gold': current_price = 2300.0
                elif name == 'dxy': current_price = 100.0
                logger.warning(f"Using fallback price for {name}: ${current_price:.2f}")

            if current_volume == 0: current_volume = random.randint(100000, 1000000)
            if market_open == 0: market_open = current_price * random.uniform(0.99, 1.01)
            if day_high == 0: day_high = current_price * random.uniform(1.01, 1.05)
            if day_low == 0: day_low = current_price * random.uniform(0.95, 0.99)
            if previous_close == 0: previous_close = current_price * random.uniform(0.99, 1.01)

            current_data[name] = {
                'current_price': current_price,
                'current_volume': current_volume,
                'market_open': market_open,
                'day_high': day_high,
                'day_low': day_low,
                'previous_close': previous_close
            }

            logger.debug(f"📊 {name}: ${current_price:.2f} | Vol: {current_volume:,}")

        except Exception as e:
            logger.error(f"❌ Critical error getting data for {symbol}: {e}")
            current_data[name] = {
                'current_price': 1.0,
                'current_volume': 0,
                'market_open': 1.0,
                'day_high': 1.0,
                'day_low': 1.0,
                'previous_close': 1.0
            }

    return current_data

def update_price_history(current_data: Dict):
    """Atualiza histórico de preços e dispara análises"""
    try:
        current_time = datetime.now()

        if not all(asset in current_data and 'current_price' in current_data[asset] for asset in SYMBOLS.keys()):
            logger.error("current_data is incomplete for price history")
            return

        price_point = {
            "timestamp": current_time.isoformat(),
            "gold_price": current_data["gold"]["current_price"],
            "btc_price": current_data["btc"]["current_price"],
            "dxy_price": current_data["dxy"]["current_price"],
            "gold_volume": current_data["gold"]["current_volume"],
            "btc_volume": current_data["btc"]["current_volume"],
            "dxy_volume": current_data["dxy"]["current_volume"]
        }

        cache["price_history"].append(price_point)

        if len(cache["price_history"]) > CACHE_CONFIG["max_price_history"]:
            cache["price_history"] = cache["price_history"][-CACHE_CONFIG["max_price_history"]:]

        update_angular_analysis(current_data)

        # Update FRED data periodically
        last_fred_update = cache["fred_data"]["last_fred_update"]
        if (last_fred_update is None or 
            (current_time - datetime.fromisoformat(last_fred_update)).total_seconds() >= UPDATE_INTERVALS["fred_data"]):
            update_fred_data()

        logger.debug(f"📈 History updated: {len(cache['price_history'])} price points")

    except Exception as e:
        logger.error(f"❌ Error updating price history: {e}")

# 💼 BACKTEST ENGINE SYSTEM (Melhorado)
class BacktestEngine:
    """Engine de backtest para geração de sinais de trading"""

    def __init__(self):
        self.patterns_db = [
            {
                "name": "Golden Cross BTC",
                "description": "MA50 cruza acima MA200 no BTC",
                "success_rate": 0.73,
                "avg_return": 0.15,
                "max_drawdown": 0.08,
                "timeframe": "4h",
                "asset": "BTC",
                "conditions": ["ma_50_above_ma_200", "volume_above_average"]
            },
            {
                "name": "DXY Rejection Gold Rally",
                "description": "DXY rejeitado em resistência + Gold breakout",
                "success_rate": 0.68,
                "avg_return": 0.12,
                "max_drawdown": 0.06,
                "timeframe": "1d",
                "asset": "GOLD",
                "conditions": ["dxy_resistance_rejection", "gold_breakout"]
            },
            {
                "name": "Perfect Divergence Short",
                "description": "DXY alta + BTC baixa = short BTC",
                "success_rate": 0.65,
                "avg_return": 0.10,
                "max_drawdown": 0.12,
                "timeframe": "1h",
                "asset": "BTC",
                "conditions": ["dxy_uptrend", "btc_downtrend"]
            },
            {
                "name": "Risk-Off Gold Surge",
                "description": "Eventos macro negativos + fuga para Gold",
                "success_rate": 0.71,
                "avg_return": 0.18,
                "max_drawdown": 0.05,
                "timeframe": "1d",
                "asset": "GOLD",
                "conditions": ["macro_negative_surprise", "vix_spike"]
            },
            {
                "name": "BTC Momentum Continuation",
                "description": "BTC break acima resistência com volume",
                "success_rate": 0.69,
                "avg_return": 0.22,
                "max_drawdown": 0.15,
                "timeframe": "1h",
                "asset": "BTC",
                "conditions": ["resistance_breakout", "volume_confirmation"]
            }
        ]

    def get_backtest_recommendations(self, top_n: int = 5) -> List[Dict]:
        """Retorna top N recomendações baseadas em backtest."""
        try:
            logger.info(f"🔎 Running backtest analysis for top {top_n} recommendations")

            current_market_conditions = self._analyze_current_conditions()
            scored_patterns = []

            for pattern in self.patterns_db:
                condition_score = self._evaluate_pattern_conditions(pattern, current_market_conditions)

                final_score = (
                    pattern["success_rate"] * 0.4 +
                    condition_score * 0.3 +
                    (pattern["avg_return"] / max(pattern["max_drawdown"], 0.01)) * 0.2 +
                    random.uniform(0.8, 1.2) * 0.1
                )

                final_score = min(1.0, final_score)

                # Determine trade type
                trade_type = "LONG"
                if "Short" in pattern["name"] or pattern["avg_return"] < 0:
                    trade_type = "SHORT"

                # Simulate entry price
                entry_price = 0
                asset_key = pattern["asset"].lower()
                if asset_key == "btc":
                    entry_price = round(random.uniform(60000, 100000), 2)
                elif asset_key == "gold":
                    entry_price = round(random.uniform(2300, 2800), 2)
                else:
                    entry_price = round(random.uniform(100, 110), 2)

                # Simulated backtest details
                simulated_trades = random.randint(50, 200)
                simulated_win_rate = pattern["success_rate"] * random.uniform(0.9, 1.1)
                simulated_total_pnl = pattern["avg_return"] * random.uniform(0.8, 1.2) * 10000

                backtest_details = {
                    "strategy_tested": pattern["name"],
                    "timeframe_backtested": f"Last {random.choice(['3 months', '6 months', '1 year'])} ({pattern['timeframe']} data)",
                    "total_trades": simulated_trades,
                    "winning_trades": int(simulated_trades * simulated_win_rate),
                    "losing_trades": simulated_trades - int(simulated_trades * simulated_win_rate),
                    "win_rate": f"{simulated_win_rate*100:.1f}%",
                    "total_pnl": f"${simulated_total_pnl:,.2f}",
                    "avg_pnl_per_trade": f"${simulated_total_pnl/simulated_trades:,.2f}",
                    "max_drawdown": f"{pattern['max_drawdown']*100:.1f}%",
                    "sharpe_ratio": round(random.uniform(1.0, 2.5), 2),
                    "alpha": round(random.uniform(0.05, 0.20), 2),
                    "key_conditions_met": [cond.replace('_', ' ') for cond in pattern["conditions"]],
                    "recommendation_reasoning": f"Based on historical performance of '{pattern['name']}' in {pattern['timeframe']} data"
                }

                recommendation = {
                    "pattern_name": pattern["name"],
                    "description": pattern["description"],
                    "symbol": SYMBOLS.get(pattern["asset"].lower(), "UNKNOWN"),
                    "asset": pattern["asset"],
                    "timeframe": pattern["timeframe"],
                    "score": round(final_score, 3),
                    "confidence": round(final_score, 3),
                    "success_rate": pattern["success_rate"],
                    "expected_return": f"{pattern['avg_return']*100:.1f}%",
                    "max_drawdown": f"{pattern['max_drawdown']*100:.1f}%",
                    "trade_type": trade_type,
                    "entry_price": entry_price,
                    "pattern_type": self._get_pattern_type(pattern),
                    "trend_context": current_market_conditions["market_trend"],
                    "trend_confidence": round(random.uniform(0.5, 0.95), 2),
                    "timestamp": datetime.now().isoformat(),
                    "backtest_details": backtest_details
                }

                scored_patterns.append(recommendation)

            top_recommendations = sorted(scored_patterns, key=lambda x: x["score"], reverse=True)[:top_n]

            logger.info(f"✅ Generated {len(top_recommendations)} backtest recommendations")
            return top_recommendations

        except Exception as e:
            logger.error(f"❌ Error in backtest engine: {e}")
            return []

    def _analyze_current_conditions(self) -> Dict:
        """Analisa condições atuais do mercado"""
        sentiment_level = sentiment_cache["market_mood"]
        trend = "NEUTRAL"
        if sentiment_level in ["EXTREME_GREED", "GREED"]:
            trend = "BULLISH"
        elif sentiment_level in ["EXTREME_FEAR", "FEAR"]:
            trend = "BEARISH"

        return {
            "market_trend": trend,
            "volatility": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "volume_profile": random.choice(["INCREASING", "DECREASING", "STABLE"]),
            "sentiment": "RISK_ON" if sentiment_level in ["EXTREME_GREED", "GREED"] else "RISK_OFF" if sentiment_level in ["EXTREME_FEAR", "FEAR"] else "NEUTRAL",
            "macro_backdrop": random.choice(["SUPPORTIVE", "CHALLENGING", "MIXED"])
        }

    def _evaluate_pattern_conditions(self, pattern: Dict, market_conditions: Dict) -> float:
        """Avalia condições do padrão"""
        base_score = 0.5

        if pattern["asset"] == "BTC":
            if market_conditions["sentiment"] == "RISK_ON":
                base_score += 0.2
            elif market_conditions["sentiment"] == "RISK_OFF":
                base_score -= 0.2

        elif pattern["asset"] == "GOLD":
            if market_conditions["sentiment"] == "RISK_OFF":
                base_score += 0.2
            elif market_conditions["sentiment"] == "RISK_ON":
                base_score -= 0.1

        if market_conditions["volatility"] == "HIGH":
            base_score += 0.1

        return max(0, min(1, base_score + random.uniform(-0.1, 0.1)))

    def _get_pattern_type(self, pattern: Dict) -> str:
        """Determina tipo de padrão"""
        name = pattern["name"].lower()
        if "breakout" in name:
            return "BREAKOUT"
        elif "momentum" in name:
            return "TREND_FOLLOW"
        elif "rejection" in name:
            return "REVERSAL"
        elif "divergence" in name:
            return "DIVERGENCE"
        else:
            return "GENERAL_PATTERN"

# Initialize BacktestEngine
backtest_engine_instance = BacktestEngine()

def run_backtest_periodically():
    """Executa backtest e armazena recomendações no cache"""
    logger.info("Starting periodic backtest run")
    try:
        recommendations = backtest_engine_instance.get_backtest_recommendations(top_n=5)
        backtest_recommendations_cache["recommendations"] = recommendations
        backtest_recommendations_cache["last_update"] = datetime.now().isoformat()
        logger.info(f"✅ Backtest completed. {len(recommendations)} recommendations cached")
    except Exception as e:
        logger.error(f"❌ Error during periodic backtest run: {e}")

def start_backtest_scheduler():
    """Inicia scheduler do backtest em thread separado"""
    logger.info(f"Scheduling backtest to run every {backtest_recommendations_cache['update_interval_minutes']} minutes")

    def scheduler_loop():
        run_backtest_periodically()
        while True:
            time.sleep(backtest_recommendations_cache["update_interval_minutes"] * 60)
            run_backtest_periodically()

    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Backtest scheduler started successfully")

# 🎭 MARKET SENTIMENT SYSTEM - REAL GATE.IO INTEGRATION
async def fetch_gateio_real_sentiment():
    """Busca dados reais de sentimento do Gate.io via API REST"""
    try:
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not found. Please install it: pip install aiohttp")
            return False

        # Endpoints de spot para sentimento
        btc_orderbook_url = "https://api.gateio.ws/api/v4/spot/order_book?currency_pair=BTC_USDT&limit=20"
        paxg_orderbook_url = "https://api.gateio.ws/api/v4/spot/order_book?currency_pair=PAXG_USDT&limit=20"
        btc_ticker_url = "https://api.gateio.ws/api/v4/spot/tickers?currency_pair=BTC_USDT"

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        success_count = 0

        async with aiohttp.ClientSession() as session:
            # Fetch BTC Order Book
            try:
                async with session.get(btc_orderbook_url, headers=headers, timeout=5) as response:
                    response.raise_for_status()
                    btc_orderbook = await response.json()

                    bids = btc_orderbook.get('bids', [])
                    asks = btc_orderbook.get('asks', [])

                    total_bid_value = sum(float(bid[1]) * float(bid[0]) for bid in bids)
                    total_ask_value = sum(float(ask[1]) * float(ask[0]) for ask in asks)
                    total_ob_value = total_bid_value + total_ask_value

                    if total_ob_value > 0:
                        btc_buyers_pct = (total_bid_value / total_ob_value) * 100
                        btc_sellers_pct = (total_ask_value / total_ob_value) * 100
                        btc_bid_ask_ratio = total_bid_value / max(total_ask_value, 1)

                        sentiment_cache["btc_sentiment"]["buyers"] = btc_buyers_pct
                        sentiment_cache["btc_sentiment"]["sellers"] = btc_sellers_pct
                        sentiment_cache["btc_sentiment"]["total_bids"] = total_bid_value
                        sentiment_cache["btc_sentiment"]["total_asks"] = total_ask_value
                        sentiment_cache["btc_sentiment"]["bid_ask_ratio"] = round(btc_bid_ask_ratio, 2)
                        sentiment_cache["btc_sentiment"]["last_update"] = datetime.now().isoformat()
                        success_count += 1

                        logger.info(f"📊 BTC Sentiment: {btc_buyers_pct:.1f}% buyers, {btc_sellers_pct:.1f}% sellers")

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch BTC orderbook: {e}")

            # Fetch PAXG Order Book
            try:
                async with session.get(paxg_orderbook_url, headers=headers, timeout=5) as response:
                    response.raise_for_status()
                    paxg_orderbook = await response.json()

                    bids = paxg_orderbook.get('bids', [])
                    asks = paxg_orderbook.get('asks', [])

                    total_bid_value = sum(float(bid[1]) * float(bid[0]) for bid in bids)
                    total_ask_value = sum(float(ask[1]) * float(ask[0]) for ask in asks)
                    total_ob_value = total_bid_value + total_ask_value

                    if total_ob_value > 0:
                        paxg_buyers_pct = (total_bid_value / total_ob_value) * 100
                        paxg_sellers_pct = (total_ask_value / total_ob_value) * 100
                        paxg_bid_ask_ratio = total_bid_value / max(total_ask_value, 1)

                        sentiment_cache["paxg_sentiment"]["buyers"] = paxg_buyers_pct
                        sentiment_cache["paxg_sentiment"]["sellers"] = paxg_sellers_pct
                        sentiment_cache["paxg_sentiment"]["total_bids"] = total_bid_value
                        sentiment_cache["paxg_sentiment"]["total_asks"] = total_ask_value
                        sentiment_cache["paxg_sentiment"]["bid_ask_ratio"] = round(paxg_bid_ask_ratio, 2)
                        sentiment_cache["paxg_sentiment"]["last_update"] = datetime.now().isoformat()
                        success_count += 1

                        logger.info(f"📊 PAXG Sentiment: {paxg_buyers_pct:.1f}% buyers, {paxg_sellers_pct:.1f}% sellers")

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch PAXG orderbook: {e}")

            # Fetch BTC Ticker for Fear & Greed calculation
            try:
                async with session.get(btc_ticker_url, headers=headers, timeout=5) as response:
                    response.raise_for_status()
                    ticker_data = await response.json()
                    
                    if ticker_data and len(ticker_data) > 0:
                        ticker = ticker_data[0]
                        change_pct_str = ticker.get('change_percentage', '0')
                        change_pct = float(change_pct_str) if change_pct_str else 0.0
                        btc_volume_24h = float(ticker.get('base_volume', '0'))

                        sentiment_cache["btc_sentiment"]["volume_24h"] = f"${btc_volume_24h / 1_000_000_000:.1f}B" if btc_volume_24h >= 1_000_000_000 else f"${btc_volume_24h / 1_000_000:.1f}M"
                        
                        # Calculate Fear & Greed Index (simplificado)
                        # Combina a porcentagem de variação do BTC com o percentual de compradores.
                        # Isso é uma simulação, um índice real de medo/ganância seria mais complexo.
                        base_fgi = 50 
                        
                        # Ajuste baseado na variação do preço
                        if change_pct > 5: base_fgi += 20
                        elif change_pct > 2: base_fgi += 10
                        elif change_pct < -5: base_fgi -= 20
                        elif change_pct < -2: base_fgi -= 10

                        # Ajuste baseado no percentual de compradores (ponderado)
                        buyers_influence = (sentiment_cache["btc_sentiment"]["buyers"] - 50) * 0.5 # Ex: 70% buyers -> (20) * 0.5 = +10
                        base_fgi += buyers_influence

                        sentiment_cache["fear_greed_index"] = max(0, min(100, int(base_fgi)))

                        # Update market mood
                        if sentiment_cache["fear_greed_index"] > 75:
                            sentiment_cache["market_mood"] = "EXTREME_GREED"
                        elif sentiment_cache["fear_greed_index"] > 55:
                            sentiment_cache["market_mood"] = "GREED"
                        elif sentiment_cache["fear_greed_index"] < 25:
                            sentiment_cache["market_mood"] = "EXTREME_FEAR"
                        elif sentiment_cache["fear_greed_index"] < 45:
                            sentiment_cache["market_mood"] = "FEAR"
                        else:
                            sentiment_cache["market_mood"] = "NEUTRAL"

                        success_count += 1

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch BTC ticker: {e}")

        return success_count >= 2 # Retorna True se BTC e PAXG orderbook/ticker foram bem sucedidos

    except Exception as e:
        logger.error(f"❌ General error in fetch_gateio_real_sentiment: {e}")
        return False

async def simulate_and_send_sentiment_data():
    """Busca dados reais de sentimento do Gate.io e envia via WebSocket"""
    logger.info("🎭 Starting REAL Gate.io sentiment monitoring and WebSocket broadcast")

    while True:
        try:
            real_data_successfully_fetched = await fetch_gateio_real_sentiment()
            current_time = datetime.now()

            # Update trend classifications
            btc_buyers_current = sentiment_cache["btc_sentiment"]["buyers"]
            paxg_buyers_current = sentiment_cache["paxg_sentiment"]["buyers"]

            if btc_buyers_current > 75:
                sentiment_cache["btc_sentiment"]["trend"] = "STRONGLY_BULLISH"
            elif btc_buyers_current > 60:
                sentiment_cache["btc_sentiment"]["trend"] = "BULLISH"
            elif btc_buyers_current < 25:
                sentiment_cache["btc_sentiment"]["trend"] = "STRONGLY_BEARISH"
            elif btc_buyers_current < 40:
                sentiment_cache["btc_sentiment"]["trend"] = "BEARISH"
            else:
                sentiment_cache["btc_sentiment"]["trend"] = "NEUTRAL"

            if paxg_buyers_current > 70:
                sentiment_cache["paxg_sentiment"]["trend"] = "STRONGLY_BULLISH"
            elif paxg_buyers_current > 55:
                sentiment_cache["paxg_sentiment"]["trend"] = "BULLISH"
            elif paxg_buyers_current < 30:
                sentiment_cache["paxg_sentiment"]["trend"] = "STRONGLY_BEARISH"
            elif paxg_buyers_current < 45:
                sentiment_cache["paxg_sentiment"]["trend"] = "BEARISH"
            else:
                sentiment_cache["paxg_sentiment"]["trend"] = "NEUTRAL"

            # Update history if real data fetched
            if real_data_successfully_fetched:
                history_point = {
                    "timestamp": current_time.isoformat(),
                    "btc_buyers": round(sentiment_cache["btc_sentiment"]["buyers"], 2),
                    "btc_sellers": round(sentiment_cache["btc_sentiment"]["sellers"], 2),
                    "paxg_buyers": round(sentiment_cache["paxg_sentiment"]["buyers"], 2),
                    "paxg_sellers": round(sentiment_cache["paxg_sentiment"]["sellers"], 2),
                    "fear_greed": sentiment_cache["fear_greed_index"],
                    "volume_estimate": sentiment_cache["btc_sentiment"]["volume_24h"],
                    "market_mood": sentiment_cache["market_mood"],
                    "data_source": "REAL"
                }
                sentiment_cache["sentiment_history"].append(history_point)
                
                # Keep only last 144 points (24h at 10min intervals)
                if len(sentiment_cache["sentiment_history"]) > 144:
                    sentiment_cache["sentiment_history"] = sentiment_cache["sentiment_history"][-144:]

            sentiment_cache["websocket_connected"] = real_data_successfully_fetched

            # Prepare WebSocket data
            sentiment_data_for_ws = {
                "timestamp": current_time.isoformat(),
                "status": "live" if real_data_successfully_fetched else "api_failed",
                "data_source": "Gate.io API" if real_data_successfully_fetched else "Gate.io API Failed",
                "btc": {
                    "buyers": round(sentiment_cache["btc_sentiment"]["buyers"], 2),
                    "sellers": round(sentiment_cache["btc_sentiment"]["sellers"], 2),
                    "trend": sentiment_cache["btc_sentiment"]["trend"],
                    "bid_ask_ratio": sentiment_cache["btc_sentiment"]["bid_ask_ratio"],
                    "total_volume": round(sentiment_cache["btc_sentiment"]["total_bids"] + sentiment_cache["btc_sentiment"]["total_asks"], 2)
                },
                "paxg": {
                    "buyers": round(sentiment_cache["paxg_sentiment"]["buyers"], 2),
                    "sellers": round(sentiment_cache["paxg_sentiment"]["sellers"], 2),
                    "trend": sentiment_cache["paxg_sentiment"]["trend"],
                    "bid_ask_ratio": sentiment_cache["paxg_sentiment"]["bid_ask_ratio"],
                    "total_volume": round(sentiment_cache["paxg_sentiment"]["total_bids"] + sentiment_cache["paxg_sentiment"]["total_asks"], 2)
                },
                "market_mood": sentiment_cache["market_mood"],
                "fear_greed_index": sentiment_cache["fear_greed_index"],
                "volume_24h": sentiment_cache["btc_sentiment"]["volume_24h"],
                "websocket_status": {
                    "connected": True,
                    "real_data": real_data_successfully_fetched,
                    "api_status": "connected" if real_data_successfully_fetched else "failed"
                }
            }

            # Send to all connected WebSocket clients
            disconnected_clients = []
            for connection in list(active_sentiment_websocket_connections):
                try:
                    await connection.send_json(sentiment_data_for_ws)
                except Exception as e:
                    logger.warning(f"Error sending to WebSocket client: {e}")
                    disconnected_clients.append(connection)

            # Remove disconnected clients
            for client in disconnected_clients:
                if client in active_sentiment_websocket_connections:
                    active_sentiment_websocket_connections.remove(client)

        except Exception as e:
            logger.error(f"Error in sentiment monitoring main loop: {e}")

        await asyncio.sleep(10)

# 🤖 TRADING BOT SYSTEM (SIMULADO COMPATÍVEL COM FRONTEND)
def update_trading_bot_state():
    """Atualiza estado do trading bot com dados simulados"""
    try:
        if not trading_bot_state["running"]:
            return

        current_time = datetime.now()

        # Simulate position updates
        if random.random() > 0.8 and len(trading_bot_state["active_positions"]) < 3:
            new_position = {
                "symbol": random.choice(["BTC/USDT", "ETH/USDT", "GOLD/USDT"]),
                "side": random.choice(["long", "short"]),
                "size": round(random.uniform(0.001, 0.1), 3),
                "entry_price": round(random.uniform(30000, 70000), 2),
                "current_price": 0,
                "pnl": 0,
                "pnl_percent": 0,
                "strategy_name": random.choice([
                    "Angular Momentum (Simulated)",
                    "Perfect Divergence (Simulated)",
                    "Golden Cross (Simulated)"
                ]),
                "timestamp": current_time.isoformat()
            }
            new_position["current_price"] = new_position["entry_price"] * random.uniform(0.98, 1.02)
            new_position["pnl"] = (new_position["current_price"] - new_position["entry_price"]) * new_position["size"]
            new_position["pnl_percent"] = ((new_position["current_price"] - new_position["entry_price"]) / new_position["entry_price"]) * 100
            if new_position["pnl_percent"] < -5: new_position["pnl_percent"] = -5 # Limita queda para visualização
            if new_position["pnl_percent"] > 5: new_position["pnl_percent"] = 5 # Limita subida para visualização

            trading_bot_state["active_positions"].append(new_position)

        # Simulate signal updates
        if random.random() > 0.7 and len(trading_bot_state["active_signals"]) < 5:
            new_signal = {
                "symbol": random.choice(["BTC/USDT", "ETH/USDT", "GOLD/USDT"]),
                "signal_type": random.choice(["LONG", "SHORT"]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "entry_price": round(random.uniform(30000, 70000), 2),
                "risk_reward_ratio": round(random.uniform(2.0, 5.0), 1),
                "timestamp": current_time.isoformat(),
                "strategy_name": random.choice([
                    "Backtest Pattern A (Simulated)",
                    "Angular Analysis (Simulated)",
                    "Sentiment Reversal (Simulated)"
                ])
            }
            trading_bot_state["active_signals"].append(new_signal)

        # Update performance metrics
        if random.random() > 0.9:
            trading_bot_state["performance"]["total_trades"] += 1
            if random.random() > 0.4:  # 60% win rate
                trading_bot_state["performance"]["winning_trades"] += 1
                pnl_change = random.uniform(50, 500)
            else:
                pnl_change = -random.uniform(20, 200)

            trading_bot_state["performance"]["total_pnl"] += pnl_change
            trading_bot_state["performance"]["daily_pnl"] += pnl_change
            trading_bot_state["performance"]["current_balance"] = trading_bot_state["performance"]["start_balance"] + trading_bot_state["performance"]["total_pnl"]

        # Keep lists manageable
        if len(trading_bot_state["active_positions"]) > 5:
            trading_bot_state["active_positions"] = trading_bot_state["active_positions"][-5:]
        if len(trading_bot_state["active_signals"]) > 8:
            trading_bot_state["active_signals"] = trading_bot_state["active_signals"][-8:]

        trading_bot_state["last_update"] = current_time.isoformat()
        trading_bot_state["performance"]["last_update"] = current_time.isoformat()

    except Exception as e:
        logger.error(f"Error updating trading bot state: {e}")

def start_trading_bot_simulation():
    """Inicia simulação do trading bot em thread separada"""
    def bot_simulation_loop():
        while True:
            if trading_bot_state["running"]:
                update_trading_bot_state()
            time.sleep(30)  # Update every 30 seconds

    bot_thread = threading.Thread(target=bot_simulation_loop, daemon=True)
    bot_thread.start()
    logger.info("✅ Trading bot simulation started")

# Nova função para buscar OHLCV da Gate.io
async def fetch_gateio_ohlcv(symbol_pair: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Busca dados OHLCV (candlesticks) da Gate.io para FUTUROS PERPÉTUOS usando API pública.
    CORRIGIDO para processar o formato JSON real da API.
    
    symbol_pair: Ex: "BTC_USDT" (contrato futuro na Gate.io)
    interval: Ex: "1m", "5m", "1h", "1d"
    limit: Número de candles a retornar (máximo 1000)
    
    Retorna: DataFrame pandas com colunas [Open, High, Low, Close, Volume] e index datetime
    """
    # URL correta para API pública de futuros da Gate.io
    base_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
    
    # Parâmetros corretos para a API
    params = {
        "contract": symbol_pair,
        "interval": interval,
        "limit": min(limit, 1000)  # Gate.io limita a 1000 candles
    }
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'TradingDashboard/1.0'
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=15)  # Timeout mais alto
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"🔍 Fetching Gate.io Futures data: {symbol_pair} ({interval}, limit {limit})")
            
            async with session.get(base_url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"❌ Gate.io API returned status {response.status} for {symbol_pair}")
                    error_text = await response.text()
                    logger.error(f"Response: {error_text[:500]}")
                    return pd.DataFrame()
                
                raw_data = await response.json()
                logger.info(f"✅ Gate.io Futures: Received {len(raw_data)} candles for {symbol_pair}")
                
                if not raw_data or len(raw_data) == 0:
                    logger.warning(f"⚠️ No candles returned for {symbol_pair}")
                    return pd.DataFrame()
                
                # CORREÇÃO PRINCIPAL: Processar formato JSON real da Gate.io
                # Formato real: {"o": "105400.2", "h": "105400.2", "l": "105330.1", "c": "105330.1", "v": 433270, "t": 1750033860}
                processed_data = []
                for candle in raw_data:
                    try:
                        processed_candle = {
                            'Open': float(candle['o']),      # open
                            'High': float(candle['h']),      # high
                            'Low': float(candle['l']),       # low
                            'Close': float(candle['c']),     # close
                            'Volume': int(candle['v']),      # volume
                            'timestamp': pd.to_datetime(int(candle['t']), unit='s')  # timestamp
                        }
                        processed_data.append(processed_candle)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Skipping invalid candle for {symbol_pair}: {e}")
                        continue
                
                if not processed_data:
                    logger.error(f"❌ No valid candles processed for {symbol_pair}")
                    return pd.DataFrame()
                
                # Criar DataFrame com dados processados
                df = pd.DataFrame(processed_data)
                
                # Configurar index temporal
                df.set_index('timestamp', inplace=True)
                
                # Ordenar por data (mais antigo primeiro, como yfinance/talib espera)
                df = df.sort_index()
                
                # Verificar se há dados válidos
                rows_before_dropna = len(df)
                df = df.dropna(subset=['Close', 'Volume'])
                rows_after_dropna = len(df)
                
                if rows_before_dropna != rows_after_dropna:
                    logger.warning(f"Dropped {rows_before_dropna - rows_after_dropna} rows due to NaN in 'Close' or 'Volume' for {symbol_pair}")
                
                if df.empty:
                    logger.error(f"❌ DataFrame is empty after cleaning for {symbol_pair}")
                    return pd.DataFrame()
                
                logger.info(f"✅ Gate.io Futures OHLCV processed for {symbol_pair}: {len(df)} data points")
                logger.debug(f"Price range: ${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}")
                logger.debug(f"Latest price: ${df['Close'].iloc[-1]:,.2f}")
                
                return df

    except aiohttp.ClientTimeout:
        logger.error(f"❌ Timeout fetching Gate.io Futures OHLCV for {symbol_pair}")
        return pd.DataFrame()
    except aiohttp.ClientError as e:
        logger.error(f"❌ AIOHTTP client error fetching Gate.io Futures OHLCV for {symbol_pair}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ General error fetching Gate.io Futures OHLCV for {symbol_pair}: {e}")
        return pd.DataFrame()

# ===============================================================================
# 🔄 SISTEMA DE DETECÇÃO DE CRUZAMENTOS MACD (NOVO)
# ===============================================================================

class MACDCrossoverDetector:
    """Detecta cruzamentos MACD em tempo real"""
    
    def __init__(self):
        self.last_macd_signal = {}
        self.crossover_history = deque(maxlen=50)
    
    def detect_crossover(self, asset: str, macd_current: float, signal_current: float, 
                         macd_previous: float, signal_previous: float) -> Dict:
        """
        Detecta cruzamento de linhas MACD
        Returns: {'type': 'bullish'/'bearish'/'none', 'strength': float, 'alert': Dict}
        """
        try:
            # Verificar se há cruzamento
            crossover_type = "none"
            
            # Cruzamento Bullish: MACD cruza acima da Signal
            if macd_previous <= signal_previous and macd_current > signal_current:
                crossover_type = "bullish"
            
            # Cruzamento Bearish: MACD cruza abaixo da Signal  
            elif macd_previous >= signal_previous and macd_current < signal_current:
                crossover_type = "bearish"
            
            if crossover_type != "none":
                # Calcular força do cruzamento
                divergence = abs(macd_current - signal_current)
                strength = min(divergence * 100, 1.0)  # Normalizar 0-1
                
                # Criar alerta
                alert = {
                    "type": "MACD_CROSSOVER",
                    "asset": asset.upper(),
                    "crossover_type": crossover_type,
                    "strength": round(strength, 3),
                    "macd_value": round(macd_current, 4),
                    "signal_value": round(signal_current, 4),
                    "divergence": round(divergence, 4),
                    "timestamp": datetime.now().isoformat(),
                    "title": f"🎯 MACD {crossover_type.upper()} Crossover - {asset.upper()}",
                    "message": f"MACD line crossed {'above' if crossover_type == 'bullish' else 'below'} Signal line",
                    "severity": "HIGH" if strength > 0.5 else "MEDIUM",
                    "trading_signal": "BUY" if crossover_type == "bullish" else "SELL",
                    "confidence": round(strength * 100, 1)
                }
                
                # Armazenar histórico
                self.crossover_history.append(alert)
                
                # Adicionar aos alertas globais
                cache["alerts"].append(alert)
                
                logger.info(f"🎯 MACD {crossover_type.upper()} crossover detected for {asset}: strength {strength:.3f}")
                
                return {
                    "type": crossover_type,
                    "strength": strength,
                    "alert": alert
                }
            
            return {"type": "none", "strength": 0, "alert": None}
            
        except Exception as e:
            logger.error(f"Error in MACD crossover detection: {e}")
            return {"type": "none", "strength": 0, "alert": None}
    
    def get_recent_crossovers(self, limit: int = 10) -> List[Dict]:
        """Retorna cruzamentos recentes"""
        return list(self.crossover_history)[-limit:]

# Instância global do detector
macd_detector = MACDCrossoverDetector()

# ===============================================================================
# 📊 PROCESSADOR DE DADOS TEMPO REAL (NOVO)
# ===============================================================================

def update_realtime_macd(asset: str, new_candle: Dict):
    """
    Atualiza MACD em tempo real com novo candle
    """
    try:
        if asset not in realtime_ohlcv_cache:
            return
        
        # Adicionar novo candle
        realtime_ohlcv_cache[asset]["candles"].append(new_candle)
        realtime_ohlcv_cache[asset]["last_update"] = datetime.now().isoformat()
        
        # Extrair preços de fechamento para MACD
        candles = list(realtime_ohlcv_cache[asset]["candles"])
        if len(candles) < 34:  # Precisa de pelo menos 34 candles para MACD
            logger.debug(f"Insufficient candles for MACD ({len(candles)}/34)")
            return
        
        close_prices = np.array([float(c["close"]) for c in candles])
        
        # Calcular MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices.astype(np.float64),
            fastperiod=12,
            slowperiod=26,  
            signalperiod=9
        )
        
        # Armazenar resultados (sem NaNs)
        macd_clean = pd.Series(macd).fillna(0).tolist()
        signal_clean = pd.Series(macd_signal).fillna(0).tolist()
        hist_clean = pd.Series(macd_hist).fillna(0).tolist()
        
        # Atualizar cache
        realtime_ohlcv_cache[asset]["macd_data"] = {
            "macd": macd_clean,
            "signal": signal_clean,
            "histogram": hist_clean,
            "last_update": datetime.now().isoformat()
        }
        
        # Detectar cruzamentos (comparar últimos 2 valores)
        if len(macd_clean) >= 2 and len(signal_clean) >= 2:
            crossover = macd_detector.detect_crossover(
                asset=asset,
                macd_current=macd_clean[-1],
                signal_current=signal_clean[-1],  
                macd_previous=macd_clean[-2],
                signal_previous=signal_clean[-2]
            )
            
            # Se houve cruzamento, atualizar cache
            if crossover["type"] != "none":
                realtime_ohlcv_cache[asset]["macd_data"]["last_crossover"] = crossover
        
        logger.debug(f"📊 MACD updated for {asset}: {len(macd_clean)} points")
        
    except Exception as e:
        logger.error(f"Error updating realtime MACD for {asset}: {e}")

# ===============================================================================
# 🔌 WEBSOCKET PARA DADOS OHLCV TEMPO REAL (NOVO)
# ===============================================================================

async def fetch_realtime_ohlcv_gateio():
    """
    Conecta ao WebSocket da Gate.io para dados OHLCV em tempo real
    """
    logger.info("🔌 Starting Gate.io OHLCV WebSocket connection...")
    
    while True:
        try:
            # Gate.io WebSocket URL para SPOT
            ws_url = "wss://api.gateio.ws/ws/v4/"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ Connected to Gate.io OHLCV WebSocket")
                realtime_ohlcv_cache["btc"]["websocket_connected"] = True
                
                # Subscrever candles de 1 minuto para BTC
                subscribe_message = {
                    "time": int(time.time()),
                    "channel": "spot.candlesticks",
                    "event": "subscribe",
                    "payload": ["1m", "BTC_USDT"]
                }
                
                await websocket.send(json.dumps(subscribe_message))
                logger.info("📊 Subscribed to BTC_USDT 1m candlesticks")
                
                # Loop principal para receber dados
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Verificar se é dado de candlestick
                        if (data.get("channel") == "spot.candlesticks" and 
                            data.get("event") == "update"):
                            
                            result = data.get("result", {})
                            if result:
                                # Processar novo candle
                                # Gate.io format: {"t": timestamp, "v": volume, "c": close, "h": high, "l": low, "o": open}
                                new_candle = {
                                    "timestamp": result.get("t"),
                                    "open": float(result.get("o", 0)),
                                    "high": float(result.get("h", 0)), 
                                    "low": float(result.get("l", 0)),
                                    "close": float(result.get("c", 0)),
                                    "volume": float(result.get("v", 0)),
                                    "datetime": datetime.fromtimestamp(int(result.get("t", 0))).isoformat()
                                }
                                
                                # Atualizar MACD em tempo real
                                update_realtime_macd("btc", new_candle)
                                
                                # Preparar dados para broadcast
                                broadcast_data = {
                                    "type": "ohlcv_update",
                                    "asset": "BTC",
                                    "candle": new_candle,
                                    "macd_data": realtime_ohlcv_cache["btc"]["macd_data"],
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                # Enviar para todos os clientes WebSocket conectados
                                await broadcast_ohlcv_update(broadcast_data)
                                
                                logger.debug(f"📊 New BTC candle: ${new_candle['close']:,.2f}, Vol: {new_candle['volume']:,.0f}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing WebSocket message: {e}")
                        continue
                
        except Exception as e:
            logger.error(f"Gate.io OHLCV WebSocket error: {e}")
            realtime_ohlcv_cache["btc"]["websocket_connected"] = False
            
            # Aguardar antes de reconectar
            await asyncio.sleep(5)
            logger.info("🔄 Attempting to reconnect OHLCV WebSocket...")

async def broadcast_ohlcv_update(data: Dict):
    """Envia atualização OHLCV para todos os clientes conectados"""
    if not active_ohlcv_websocket_connections:
        return
    
    disconnected_clients = []
    for connection in list(active_ohlcv_websocket_connections):
        try:
            await connection.send_json(data)
        except Exception:
            disconnected_clients.append(connection)
    
    # Remover clientes desconectados
    for client in disconnected_clients:
        if client in active_ohlcv_websocket_connections:
            active_ohlcv_websocket_connections.remove(client)

# ===============================================================================
# 🌐 ENDPOINT WEBSOCKET PARA DADOS OHLCV (NOVO)
# ===============================================================================

@app.websocket("/ws/ohlcv")
async def websocket_endpoint_ohlcv(websocket: WebSocket):
    """WebSocket para dados OHLCV e MACD em tempo real"""
    await websocket.accept()
    active_ohlcv_websocket_connections.append(websocket)
    logger.info(f"OHLCV WebSocket client connected: {websocket.client}")
    
    try:
        # Enviar dados iniciais
        initial_data = {
            "type": "initial_data",
            "btc_macd": realtime_ohlcv_cache["btc"]["macd_data"],
            "websocket_status": "connected",
            "message": "OHLCV real-time data stream active"
        }
        await websocket.send_json(initial_data)
        
        # Manter conexão ativa
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        if websocket in active_ohlcv_websocket_connections:
            active_ohlcv_websocket_connections.remove(websocket)
        logger.info(f"OHLCV WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        if websocket in active_ohlcv_websocket_connections:
            active_ohlcv_websocket_connections.remove(websocket)
        logger.error(f"OHLCV WebSocket error: {e}")

# ===============================================================================
# 📊 ENDPOINTS PARA DADOS MACD TEMPO REAL (NOVO)
# ===============================================================================

@app.get("/api/macd/realtime/{asset}")
def get_realtime_macd(asset: str):
    """Endpoint para MACD em tempo real"""
    try:
        if asset.lower() not in realtime_ohlcv_cache:
            raise HTTPException(status_code=404, detail=f"Asset {asset} not found")
        
        asset_data = realtime_ohlcv_cache[asset.lower()]
        
        return {
            "asset": asset.upper(),
            "macd_data": asset_data["macd_data"],
            "websocket_connected": asset_data["websocket_connected"],
            "last_update": asset_data["last_update"],
            "candles_count": len(asset_data["candles"]),
            "recent_crossovers": macd_detector.get_recent_crossovers(5),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting realtime MACD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crossovers/recent")
def get_recent_crossovers():
    """Endpoint para cruzamentos MACD recentes"""
    try:
        return {
            "recent_crossovers": macd_detector.get_recent_crossovers(20),
            "total_crossovers": len(macd_detector.crossover_history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting crossovers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================================================================
# API ENDPOINTS (EXISTING)
# ===============================================================================

@app.get("/")
def read_root():
    """Endpoint raiz com informações da API"""
    return {
        "message": f"🚀 Trading Dashboard API {SYSTEM_INFO['version']} - Compatible",
        "description": SYSTEM_INFO["description"],
        "version": SYSTEM_INFO["version"],
        "current_time": datetime.now().isoformat(),
        "status": "🟢 ONLINE",
        "endpoints": {
            "📊 Market Data": {
                "/api/current": "Real-time prices + angular analysis + sentiment",
                "/api/precos/{period}": "Historical prices and volume",
            },
            "🎭 Market Sentiment": {
                "/api/sentiment": "Real-time Gate.io sentiment (via REST)",
                "/ws/sentiment": "Real-time sentiment WebSocket stream",
                "/api/sentiment/history": "24h sentiment history",
            },
            "📐 Angular Analysis": {
                "/api/angular": "Complete angular analysis",
                "/api/patterns": "Detected angular patterns",
            },
            "📅 Economic Calendar": {
                "/api/calendar": "FRED economic calendar",
                "/api/events": "Upcoming economic events",
            },
            "🚨 Alert System": {
                "/api/alerts": "All alerts (macro + angular + FRED + trades)",
            },
            "💰 Trade Signals (Backtest)": {
                "/api/backtest-recommendations": "Top N backtest recommendations"
            },
            "🤖 Trading Bot (Simulated)": {
                "/api/trading-bot/status": "Current status of the trading bot",
                "/api/trading-bot/positions": "List of active simulated positions",
                "/api/trading-bot/signals": "Recently detected simulated trade signals",
                "/api/trading-bot/performance": "Performance metrics of the simulated bot",
                "/api/trading-bot/start": "Start the simulated trading bot",
                "/api/trading-bot/stop": "Stop the simulated trading bot",
            },
            "⚙️ System": {
                "/api/status": "API status and health check",
                "/api/debug/gateio": "Gate.io API connectivity test",
                "/docs": "API documentation",
            },
            "📈 Real-time MACD": { # New Endpoint Group
                "/ws/ohlcv": "Real-time OHLCV and MACD WebSocket stream for BTC",
                "/api/macd/realtime/{asset}": "Latest real-time MACD data for a given asset",
                "/api/crossovers/recent": "Recent MACD crossover alerts"
            }
        },
        "features": SYSTEM_INFO["features"]
    }

@app.get("/api/current")
def get_current_data():
    """Endpoint principal para dados atuais - CORRIGIDO para compatibilidade total com frontend"""
    try:
        current_data = get_current_market_data()
        update_price_history(current_data)
        cache["data"] = current_data

        # Angular info with fallbacks
        angular_info = {
            "btc_angle": 0, "btc_strength": 0, "btc_trend": "NEUTRAL",
            "gold_angle": 0, "gold_strength": 0, "gold_trend": "NEUTRAL",
            "dxy_angle": 0, "dxy_strength": 0, "dxy_trend": "NEUTRAL"
        }
        
        if cache["angular_data"]:
            latest_angular = cache["angular_data"][-1]
            for asset_key in ["btc", "gold", "dxy"]:
                asset_data = latest_angular.get(asset_key)
                if isinstance(asset_data, dict):
                    angular_info[f"{asset_key}_angle"] = asset_data.get("angle", 0)
                    angular_info[f"{asset_key}_strength"] = asset_data.get("strength", 0)
                    angular_info[f"{asset_key}_trend"] = asset_data.get("trend", "NEUTRAL")

        # FRED info
        fred_info = {
            "today_events": 0,
            "today_high_impact": 0,
            "next_critical_event": None,
            "total_upcoming": 0,
            "pre_alerts_active": 0
        }
        
        if cache["fred_data"]["upcoming_events"]:
            today_events = [
                e for e in cache["fred_data"]["upcoming_events"]
                if "date" in e and datetime.fromisoformat(e["date"]).date() == datetime.now().date()
            ]

            critical_events = [e for e in cache["fred_data"]["upcoming_events"] if e.get("importance") == "HIGH"]
            next_critical = None
            if critical_events:
                next_critical_candidate = critical_events[0]
                if all(k in next_critical_candidate for k in ["name", "date", "impact_score"]):
                    next_critical = {
                        "name": next_critical_candidate["name"],
                        "date": next_critical_candidate["date"],
                        "impact_score": next_critical_candidate["impact_score"]
                    }
                    try:
                        event_date = datetime.fromisoformat(next_critical_candidate["date"])
                        next_critical["days_until"] = (event_date.date() - datetime.now().date()).days
                    except:
                        next_critical["days_until"] = 0

            fred_info = {
                "today_events": len(today_events),
                "today_high_impact": len([e for e in today_events if e.get("importance") == "HIGH"]),
                "next_critical_event": next_critical,
                "total_upcoming": len(cache["fred_data"]["upcoming_events"]),
                "pre_alerts_active": len(cache["fred_data"]["pre_event_alerts"])
            }

        # Sentiment info
        sentiment_info = {
            "btc": {
                "buyers_percent": sentiment_cache["btc_sentiment"]["buyers"],
                "sellers_percent": sentiment_cache["btc_sentiment"]["sellers"],
                "market_mood": sentiment_cache["market_mood"],
                "fear_greed_index": sentiment_cache["fear_greed_index"],
                "trend": sentiment_cache["btc_sentiment"]["trend"],
            },
            "paxg": {
                "buyers_percent": sentiment_cache["paxg_sentiment"]["buyers"],
                "sellers_percent": sentiment_cache["paxg_sentiment"]["sellers"],
                "trend": sentiment_cache["paxg_sentiment"]["trend"],
            },
            "websocket_connected": sentiment_cache["websocket_connected"],
            "last_update": sentiment_cache["btc_sentiment"]["last_update"]
        }

        # Real-time MACD Info (NEW)
        realtime_macd_info = {
            "btc_websocket_connected": realtime_ohlcv_cache["btc"]["websocket_connected"],
            "btc_last_update": realtime_ohlcv_cache["btc"]["last_update"],
            "btc_crossovers_count": len(macd_detector.crossover_history)
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "market_status": "Real-time data available",
            "data_quality": {
                "price_data_valid": all(current_data[asset].get("current_price", 0) > 0 for asset in SYMBOLS),
                "angular_analysis_active": len(cache["angular_data"]) > 0,
                "fred_calendar_active": len(cache["fred_data"]["upcoming_events"]) > 0,
                "sentiment_monitoring_active": sentiment_cache["websocket_connected"],
                "realtime_ohlcv_active": realtime_ohlcv_cache["btc"]["websocket_connected"] # NEW
            },
            "assets": {
                "gold": {
                    "name": "Gold Futures",
                    "symbol": SYMBOLS['gold'],
                    "current_price": current_data["gold"]["current_price"],
                    "current_volume": current_data["gold"]["current_volume"],
                    "volume_formatted": format_volume(current_data["gold"]["current_volume"]),
                    "day_high": current_data["gold"]["day_high"],
                    "day_low": current_data["gold"]["day_low"],
                    "previous_close": current_data["gold"]["previous_close"],
                    "change": current_data["gold"]["current_price"] - current_data["gold"]["previous_close"],
                    "change_percent": ((current_data["gold"]["current_price"] - current_data["gold"]["previous_close"]) / current_data["gold"]["previous_close"] * 100) if current_data["gold"]["previous_close"] > 0 else 0,
                    "color": ASSET_COLORS.get('gold', '#FFD700'),
                    "angular": {
                        "angle": angular_info["gold_angle"],
                        "strength": angular_info["gold_strength"],
                        "trend": angular_info["gold_trend"]
                    }
                },
                "btc": {
                    "name": "Bitcoin",
                    "symbol": SYMBOLS['btc'],
                    "current_price": current_data["btc"]["current_price"],
                    "current_volume": current_data["btc"]["current_volume"],
                    "volume_formatted": format_volume(current_data["btc"]["current_volume"]),
                    "day_high": current_data["btc"]["day_high"],
                    "day_low": current_data["btc"]["day_low"],
                    "previous_close": current_data["btc"]["previous_close"],
                    "change": current_data["btc"]["current_price"] - current_data["btc"]["previous_close"],
                    "change_percent": ((current_data["btc"]["current_price"] - current_data["btc"]["previous_close"]) / current_data["btc"]["previous_close"] * 100) if current_data["btc"]["previous_close"] > 0 else 0,
                    "color": ASSET_COLORS.get('btc', '#FF8C00'),
                    "angular": {
                        "angle": angular_info["btc_angle"],
                        "strength": angular_info["btc_strength"],
                        "trend": angular_info["btc_trend"]
                    }
                },
                "dxy": {
                    "name": "US Dollar Index",
                    "symbol": SYMBOLS['dxy'],
                    "current_price": current_data["dxy"]["current_price"],
                    "current_volume": current_data["dxy"]["current_volume"],
                    "volume_formatted": format_volume(current_data["dxy"]["current_volume"]),
                    "day_high": current_data["dxy"]["day_high"],
                    "day_low": current_data["dxy"]["day_low"],
                    "previous_close": current_data["dxy"]["previous_close"],
                    "change": current_data["dxy"]["current_price"] - current_data["dxy"]["previous_close"],
                    "change_percent": ((current_data["dxy"]["current_price"] - current_data["dxy"]["previous_close"]) / current_data["dxy"]["previous_close"] * 100) if current_data["dxy"]["previous_close"] > 0 else 0,
                    "color": ASSET_COLORS.get('dxy', '#00CC66'),
                    "angular": {
                        "angle": angular_info["dxy_angle"],
                        "strength": angular_info["dxy_strength"],
                        "trend": angular_info["dxy_trend"]
                    }
                }
            },
            "analysis": {
                "total_alerts": len(cache["alerts"]),
                "angular_alerts": len([a for a in cache["alerts"] if a.get("type", "").startswith("ANGULAR_")]),
                "fred_alerts": len([a for a in cache["alerts"] if a.get("type", "") == "PRE_EVENT_ALERT"]),
                "trade_signals": len([a for a in cache["alerts"] if a.get("type", "").startswith("BTC_")]),
                "macd_crossovers": len([a for a in cache["alerts"] if a.get("type", "") == "MACD_CROSSOVER"]), # NEW
                "price_history_points": len(cache["price_history"]),
                "angular_history_points": len(cache["angular_data"]),
                "last_angular_analysis": cache["last_angular_analysis"],
                "economic_calendar": fred_info,
                "market_sentiment": sentiment_info,
                "realtime_macd_status": realtime_macd_info # NEW
            },
            "system_health": {
                "cache_size": len(cache["price_history"]),
                "uptime_status": "healthy",
                "last_data_refresh": datetime.now().isoformat(),
                "api_endpoints_active": 13,
                "websocket_status": { # Updated to reflect both sentiment and ohlcv
                    "sentiment": "connected" if sentiment_cache["websocket_connected"] else "disconnected_or_failed_api",
                    "ohlcv": "connected" if realtime_ohlcv_cache["btc"]["websocket_connected"] else "disconnected_or_failed_api"
                }
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting current data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting current data: {str(e)}")

@app.get("/api/precos/{period}")
async def get_financial_data_by_period(period: str):
    """Endpoint para dados históricos, incluindo MACD e Volume colorido - Gate.io para BTC, yfinance para outros"""
    logger.info(f"📊 Fetching financial data for period: {period}")
    
    # Mapeamento de períodos para intervalos
    gateio_interval_map = {
        '1d': '1m',   # Para "hoje", pegue dados de 1 minuto
        '5d': '30m', # Para 5 dias, dados de 30 minutos
        '1mo': '4h', # Para 1 mês, dados de 4 horas
        '3mo': '1d', # Para 3 meses, dados diários
        '6mo': '1d',
        '1y': '1d',
        '2y': '1d',
        '5y': '1d',
        'max': '1d'
    }
    
    yfinance_interval_map = {
        '1d': '1m',    # Para 1 dia, usar 1m se disponível
        '5d': '15m',   # Para 5 dias, usar 15m
        '1mo': '1h',   # Para 1 mês, usar 1h
        '3mo': '1d',   # Para 3 meses, usar 1d
        '6mo': '1d',
        '1y': '1d',
        '2y': '1d',
        '5y': '1d',
        'max': '1d'
    }
    
    # Limites de dados baseados no período para Gate.io
    gateio_limit_map = {
        '1d': 1000,   # Suficiente para 1 dia em 1m
        '5d': 480,    # 5 dias * 24h * 2 (30m) = 240, um pouco mais para margem
        '1mo': 180,   # 1 mês * ~30d * 6 (4h) = 180.
        '3mo': 200,   # 3 meses * ~30d = 90, 200 é seguro
        '6mo': 200,
        '1y': 200,
        '2y': 200,
        '5y': 200,
        'max': 200
    }

    try:
        data = {}
        
        for asset_name, symbol in SYMBOLS.items():
            prices = []
            volumes = []
            volume_colors = []  # NOVO: para cores do volume
            asset_dates = []
            macd_values = []
            macd_signal_values = []
            macd_hist_values = []
            opens = []  # NOVO: preços de abertura
            highs = []  # NOVO: preços máximos
            lows = []   # NOVO: preços mínimos

            try:
                hist = pd.DataFrame()
                # Processamento especial para BTC usando Gate.io Futuros
                if asset_name == 'btc':
                    gateio_interval = gateio_interval_map.get(period, '1d')
                    limit = gateio_limit_map.get(period, 200)
                    
                    logger.info(f"DEBUG: Fetching BTC from Gate.io Futures (Contract: BTC_USDT, Interval: {gateio_interval}, Limit: {limit})")
                    hist = await fetch_gateio_ohlcv("BTC_USDT", gateio_interval, limit)
                    
                    if hist.empty:
                        logger.warning("⚠️ No Gate.io data received for BTC, or DataFrame is empty after cleaning.")
                        continue # Pula para o próximo ativo se não houver dados válidos

                # Processamento para Gold e DXY usando yfinance
                else:
                    yf_interval = yfinance_interval_map.get(period, '1d')
                    logger.info(f"DEBUG: Fetching {asset_name} from yfinance (Symbol: {symbol}, Period: {period}, Interval: {yf_interval})")
                    
                    # Busca dados históricos
                    hist = yf.download(
                        symbol, 
                        period=period, 
                        interval=yf_interval, 
                        progress=False, 
                        threads=False
                    )
                    
                    if hist.empty:
                        logger.warning(f"⚠️ No yfinance data received for {symbol}, or DataFrame is empty after download.")
                        continue # Pula para o próximo ativo se não houver dados

                # Pré-processar dados para todos os ativos (Gate.io ou yfinance)
                if (not hist.empty and 
                    'Close' in hist.columns and not hist['Close'].empty and
                    'Volume' in hist.columns and not hist['Volume'].empty and
                    'Open' in hist.columns and not hist['Open'].empty and
                    'High' in hist.columns and not hist['High'].empty and
                    'Low' in hist.columns and not hist['Low'].empty):
                    
                    # Garantir que os tipos são numéricos, forçando erros para NaN
                    for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
                        hist[col] = pd.to_numeric(hist[col], errors='coerce')

                    # Preencher NaNs remanescentes (forward-fill, depois backward-fill)
                    hist['Close'] = hist['Close'].fillna(method='ffill').fillna(method='bfill')
                    hist['Volume'] = hist['Volume'].fillna(0) # Volume pode ser 0
                    hist['Open'] = hist['Open'].fillna(method='ffill').fillna(method='bfill')
                    hist['High'] = hist['High'].fillna(method='ffill').fillna(method='bfill')
                    hist['Low'] = hist['Low'].fillna(method='ffill').fillna(method='bfill')
                    
                    # Remover linhas que ainda possam ter NaNs em Close após preenchimento (caso todo o dataset seja NaN)
                    hist_cleaned = hist.dropna(subset=['Close', 'Volume', 'Open', 'High', 'Low'])

                    if not hist_cleaned.empty:
                        # NOVO: Calcular cores do volume baseado na direção do preço
                        hist_cleaned = calculate_volume_colors_and_macd(hist_cleaned)
                        
                        # Remover timezone para consistência antes de converter para ISO
                        hist_cleaned.index = hist_cleaned.index.tz_localize(None)
                        
                        # Extrair todos os dados OHLCV
                        prices = hist_cleaned['Close'].tolist()
                        volumes = hist_cleaned['Volume'].tolist()
                        opens = hist_cleaned['Open'].tolist()
                        highs = hist_cleaned['High'].tolist()
                        lows = hist_cleaned['Low'].tolist()
                        volume_colors = hist_cleaned['price_direction'].tolist()
                        asset_dates = hist_cleaned.index.map(lambda x: x.isoformat()).tolist()
                        
                        logger.info(f"DEBUG: {symbol} - Successfully processed {len(prices)} data points for chart with OHLCV.")

                        # Calcular MACD usando talib
                        # MACD requer pelo menos 34 pontos de dados (26 + 9 para período padrão)
                        logger.debug(f"DEBUG: {symbol} - Number of data points for MACD calculation: {len(hist_cleaned['Close'])}")
                        if len(hist_cleaned['Close']) >= 34: 
                            macd, macdsignal, macdhist = talib.MACD(
                                hist_cleaned['Close'].values.astype(np.float64), # Assegura float64
                                fastperiod=12, 
                                slowperiod=26, 
                                signalperiod=9
                            )
                            
                            # Converter para listas e preencher NaNs com 0 para o JSON
                            # talib retorna NaNs para os primeiros pontos que não podem ser calculados.
                            # Usamos `fillna(0)` para garantir que o JSON não tenha `null` para os gráficos.
                            macd_values_raw = pd.Series(macd).fillna(0).tolist()
                            macd_signal_values_raw = pd.Series(macdsignal).fillna(0).tolist()
                            macd_hist_values_raw = pd.Series(macdhist).fillna(0).tolist()

                            # Assegura que todas as listas (preço, volume, datas, MACD) têm o mesmo comprimento.
                            # MACD pode ter menos pontos devido aos NaNs iniciais.
                            # Alinhamos as listas pelo final (os dados mais recentes).
                            min_len_data = min(len(prices), len(macd_values_raw), len(macd_signal_values_raw), len(macd_hist_values_raw))
                            
                            if min_len_data > 0:
                                prices = prices[-min_len_data:]
                                volumes = volumes[-min_len_data:]
                                opens = opens[-min_len_data:]
                                highs = highs[-min_len_data:]
                                lows = lows[-min_len_data:]
                                volume_colors = volume_colors[-min_len_data:]
                                asset_dates = asset_dates[-min_len_data:]
                                macd_values = macd_values_raw[-min_len_data:]
                                macd_signal_values = macd_signal_values_raw[-min_len_data:]
                                macd_hist_values = macd_hist_values_raw[-min_len_data:]
                                
                                logger.info(f"DEBUG: {symbol} - MACD calculated and aligned. Prices length: {len(prices)}, MACD length: {len(macd_values)}")
                            else:
                                logger.warning(f"DEBUG: {symbol} - MACD calculation resulted in no valid points after alignment for {period}. MACD lists will be empty.")
                                macd_values, macd_signal_values, macd_hist_values = [], [], []

                        else:
                            logger.warning(f"DEBUG: {symbol} - NOT enough data ({len(hist_cleaned['Close'])} points) for MACD calculation for {period}. Need >= 34. MACD lists will be empty.")
                            macd_values, macd_signal_values, macd_hist_values = [], [], []
                    else:
                        logger.warning(f"DEBUG: {symbol} - Cleaned history is empty after dropping NaNs for {period}. No price/volume data.")
                else:
                    logger.warning(f"DEBUG: {symbol} - No required OHLCV columns found, or history is empty for {period}.")

                avg_volume = sum(volumes) / len(volumes) if volumes else 0

                data[asset_name] = {
                    'name': asset_name.capitalize(),
                    'symbol': symbol, # Usar o 'symbol' do SYMBOLS para consistência
                    'price_data': prices,
                    'volume_data': volumes,
                    'volume_colors': volume_colors,  # NOVO
                    'open_data': opens,              # NOVO
                    'high_data': highs,              # NOVO
                    'low_data': lows,                # NOVO
                    'volume_avg_formatted': format_volume(int(avg_volume)),
                    'dates': asset_dates,
                    'macd_data': macd_values,          # Adicionado MACD
                    'macd_signal_data': macd_signal_values, # Adicionado MACD Signal
                    'macd_hist_data': macd_hist_values # Adicionado MACD Histogram
                }

            except Exception as e:
                logger.error(f"❌ Error processing {asset_name} data: {e}")
                data[asset_name] = {
                    'name': asset_name.capitalize(),
                    'symbol': symbol,
                    'price_data': [], 
                    'volume_data': [], 
                    'volume_colors': [],
                    'open_data': [],
                    'high_data': [],
                    'low_data': [],
                    'volume_avg_formatted': '0', 
                    'dates': [],
                    'macd_data': [],
                    'macd_signal_data': [],
                    'macd_hist_data': []
                }

        # Obter datas comuns para todos os ativos que tenham dados
        # A escolha de max(all_asset_dates, key=len) significa que as datas serão do ativo com mais pontos.
        # O frontend precisará lidar com o alinhamento se os tamanhos das listas forem diferentes.
        all_asset_dates_lists = [asset_data['dates'] for asset_data in data.values() if asset_data['dates']]
        if all_asset_dates_lists:
            common_dates = max(all_asset_dates_lists, key=len)
            # Para garantir que todos os dados retornados tenham o mesmo comprimento que common_dates,
            # ou um comprimento consistente com seus próprios dados, você pode precisar truncar/preencher.
            # No entanto, a lógica de `min_len_data` dentro do loop já alinha as listas.
            # Apenas garantimos que o `data_points` e `dates` do nível superior sejam coerentes.
        else:
            common_dates = []

        response = {
            "period": period,
            "data_points": len(common_dates),
            "dates": common_dates,
            "assets": {
                asset_name: {
                    "name": asset_data['name'],
                    "symbol": asset_data['symbol'],
                    "price_data": asset_data['price_data'],
                    "volume_data": asset_data['volume_data'],
                    "volume_colors": asset_data['volume_colors'],  # NOVO
                    "open_data": asset_data['open_data'],          # NOVO
                    "high_data": asset_data['high_data'],          # NOVO
                    "low_data": asset_data['low_data'],            # NOVO
                    "volume_avg_formatted": asset_data['volume_avg_formatted'],
                    "macd_data": asset_data['macd_data'],
                    "macd_signal_data": asset_data['macd_signal_data'],
                    "macd_hist_data": asset_data['macd_hist_data']
                } for asset_name, asset_data in data.items()
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ Data fetched for {period}: {len(common_dates)} common points.")
        return response

    except Exception as e:
        logger.error(f"❌ Critical error fetching financial data for {period}: {e}")
        # Retorna um JSON de erro mais robusto para o frontend
        return {
            "error": f"Failed to fetch financial data for {period}: {str(e)}",
            "period": period,
            "data_points": 0,
            "dates": [],
            "assets": {
                'gold': {'name': 'Gold', 'symbol': 'GC=F', 'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [], 'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': []},
                'btc': {'name': 'Btc', 'symbol': 'BTC-USD', 'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [], 'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': []},
                'dxy': {'name': 'Dxy', 'symbol': 'DX-Y.NYB', 'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [], 'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': []}
            },
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/sentiment")
def get_market_sentiment():
    """Endpoint REST para sentimento de mercado em tempo real do Gate.io"""
    try:
        current_time = datetime.now()
        is_real_data = sentiment_cache["websocket_connected"]

        btc_buyers_pct = sentiment_cache["btc_sentiment"]["buyers"]
        paxg_buyers_pct = sentiment_cache["paxg_sentiment"]["buyers"]

        estimated_btc_volume_24h_display = sentiment_cache["btc_sentiment"]["volume_24h"]

        # Interpretations
        if btc_buyers_pct > 80:
            btc_sentiment_interpretation = "EXTREMELY_BULLISH"
            btc_strength = "VERY_STRONG"
            btc_recommendation = "Strong bullish momentum - consider long positions"
        elif btc_buyers_pct > 65:
            btc_sentiment_interpretation = "BULLISH"
            btc_strength = "STRONG"
            btc_recommendation = "Bullish sentiment - favorable for longs"
        elif btc_buyers_pct > 35:
            btc_sentiment_interpretation = "NEUTRAL"
            btc_strength = "MODERATE"
            btc_recommendation = "Mixed signals - wait for clearer direction"
        elif btc_buyers_pct > 20:
            btc_sentiment_interpretation = "BEARISH"
            btc_strength = "STRONG"
            btc_recommendation = "Bearish sentiment - consider short positions"
        else:
            btc_sentiment_interpretation = "EXTREMELY_BEARISH"
            btc_strength = "VERY_STRONG"
            btc_recommendation = "Strong bearish momentum - avoid longs"

        if paxg_buyers_pct > 60:
            paxg_sentiment_interpretation = "BULLISH"
            paxg_strength = "MODERATE"
        elif paxg_buyers_pct < 40:
            paxg_sentiment_interpretation = "BEARISH"
            paxg_strength = "MODERATE"
        else:
            paxg_sentiment_interpretation = "NEUTRAL"
            paxg_strength = "WEAK"

        response = {
            "timestamp": current_time.isoformat(),
            "status": "🟢 LIVE" if is_real_data else "🔴 API_FAIL",
            "data_source": "Gate.io Real API" if is_real_data else "Gate.io API Offline/Failed",
            "btc": {
                "buyers": round(sentiment_cache["btc_sentiment"]["buyers"], 2),
                "sellers": round(sentiment_cache["btc_sentiment"]["sellers"], 2),
                "total_volume": round(sentiment_cache["btc_sentiment"]["total_bids"] + sentiment_cache["btc_sentiment"]["total_asks"], 2),
                "trend": sentiment_cache["btc_sentiment"]["trend"],
                "currency_pair": "BTC_USDT",
                "bid_ask_ratio": sentiment_cache["btc_sentiment"]["bid_ask_ratio"]
            },
            "paxg": {
                "buyers": round(sentiment_cache["paxg_sentiment"]["buyers"], 2),
                "sellers": round(sentiment_cache["paxg_sentiment"]["sellers"], 2),
                "total_volume": round(sentiment_cache["paxg_sentiment"]["total_bids"] + sentiment_cache["paxg_sentiment"]["total_asks"], 2),
                "trend": sentiment_cache["paxg_sentiment"]["trend"],
                "currency_pair": "PAXG_USDT",
                "bid_ask_ratio": sentiment_cache["paxg_sentiment"]["bid_ask_ratio"]
            },
            "market_mood": sentiment_cache["market_mood"],
            "fear_greed_index": sentiment_cache["fear_greed_index"],
            "volume_24h": estimated_btc_volume_24h_display,
            "websocket_status": {
                "connected": True,
                "real_data": is_real_data,
                "api_status": "connected" if is_real_data else "failed"
            }
        }
        return response

    except Exception as e:
        logger.error(f"❌ Error getting market sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting market sentiment: {str(e)}")

@app.websocket("/ws/sentiment")
async def websocket_endpoint_sentiment(websocket: WebSocket):
    """WebSocket para sentimento de mercado"""
    await websocket.accept()
    active_sentiment_websocket_connections.append(websocket)
    logger.info(f"WebSocket client connected: {websocket.client}")
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_sentiment_websocket_connections:
            active_sentiment_websocket_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        if websocket in active_sentiment_websocket_connections:
            active_sentiment_websocket_connections.remove(websocket)
        logger.error(f"WebSocket error for client {websocket.client}: {e}")

@app.get("/api/sentiment/history")
def get_sentiment_history():
    """Endpoint para histórico de sentimento (últimas 24h)"""
    try:
        current_time = datetime.now()
        history = [h for h in sentiment_cache["sentiment_history"] if h.get("data_source") == "REAL"]
        
        if len(history) > 144:
            history = history[-144:]

        if not history:
            return {
                "timestamp": current_time.isoformat(),
                "period": "24h",
                "data_points": 0,
                "history": [],
                "summary": {
                    "btc": {"avg_buyers": 0, "max_buyers": 0, "min_buyers": 0, "current_buyers": sentiment_cache["btc_sentiment"]["buyers"], "trend_1h": "N/A", "trend_6h": "N/A", "volatility": 0},
                    "paxg": {"avg_buyers": 0, "max_buyers": 0, "min_buyers": 0, "current_buyers": sentiment_cache["paxg_sentiment"]["buyers"], "trend_1h": "N/A", "trend_6h": "N/A", "volatility": 0},
                    "overall_fear_greed": sentiment_cache["fear_greed_index"]
                },
                "analysis": {
                    "btc_dominant_sentiment": "N/A", "paxg_dominant_sentiment": "N/A", "market_stability": "N/A",
                    "combined_momentum_btc": "N/A", "combined_momentum_paxg": "N/A"
                }
            }

        btc_buyers_values = [h["btc_buyers"] for h in history if "btc_buyers" in h]
        paxg_buyers_values = [h["paxg_buyers"] for h in history if "paxg_buyers" in h]

        summary = {
            "btc": {
                "avg_buyers": round(sum(btc_buyers_values) / max(1, len(btc_buyers_values)), 2) if btc_buyers_values else 0,
                "max_buyers": max(btc_buyers_values) if btc_buyers_values else 0,
                "min_buyers": min(btc_buyers_values) if btc_buyers_values else 0,
                "current_buyers": sentiment_cache["btc_sentiment"]["buyers"],
                "trend_1h": "INCREASING" if len(btc_buyers_values) >= 7 and btc_buyers_values[-1] > btc_buyers_values[-7] else "DECREASING" if len(btc_buyers_values) >= 7 and btc_buyers_values[-1] < btc_buyers_values[-7] else "NEUTRAL",
                "trend_6h": "INCREASING" if len(btc_buyers_values) >= 37 and btc_buyers_values[-1] > btc_buyers_values[-37] else "DECREASING" if len(btc_buyers_values) >= 37 and btc_buyers_values[-1] < btc_buyers_values[-37] else "NEUTRAL",
                "volatility": round(max(btc_buyers_values) - min(btc_buyers_values), 2) if btc_buyers_values else 0,
            },
            "paxg": {
                "avg_buyers": round(sum(paxg_buyers_values) / max(1, len(paxg_buyers_values)), 2) if paxg_buyers_values else 0,
                "max_buyers": max(paxg_buyers_values) if paxg_buyers_values else 0,
                "min_buyers": min(paxg_buyers_values) if paxg_buyers_values else 0,
                "current_buyers": sentiment_cache["paxg_sentiment"]["buyers"],
                "trend_1h": "INCREASING" if len(paxg_buyers_values) >= 7 and paxg_buyers_values[-1] > paxg_buyers_values[-7] else "DECREASING" if len(paxg_buyers_values) >= 7 and paxg_buyers_values[-1] < paxg_buyers_values[-7] else "NEUTRAL",
                "trend_6h": "INCREASING" if len(paxg_buyers_values) >= 37 and paxg_buyers_values[-1] > paxg_buyers_values[-37] else "DECREASING" if len(paxg_buyers_values) >= 37 and paxg_buyers_values[-1] < paxg_buyers_values[-37] else "NEUTRAL",
                "volatility": round(max(paxg_buyers_values) - min(paxg_buyers_values), 2) if paxg_buyers_values else 0,
            },
            "overall_fear_greed": sentiment_cache["fear_greed_index"]
        }

        return {
            "timestamp": current_time.isoformat(),
            "period": "24h",
            "data_points": len(history),
            "history": history,
            "summary": summary,
            "analysis": {
                "btc_dominant_sentiment": "BULLISH" if summary["btc"]["avg_buyers"] > 60 else "BEARISH" if summary["btc"]["avg_buyers"] < 40 else "NEUTRAL",
                "paxg_dominant_sentiment": "BULLISH" if summary["paxg"]["avg_buyers"] > 55 else "BEARISH" if summary["paxg"]["avg_buyers"] < 45 else "NEUTRAL",
                "market_stability": "HIGH" if (summary["btc"]["volatility"] < 15 and summary["paxg"]["volatility"] < 10) else "MODERATE" if (summary["btc"]["volatility"] < 30 and summary["paxg"]["volatility"] < 20) else "LOW",
                "combined_momentum_btc": "STRONG" if history and abs(history[-1]["btc_buyers"] - summary["btc"]["avg_buyers"]) > 15 else "WEAK" if history else "N/A",
                "combined_momentum_paxg": "STRONG" if history and abs(history[-1]["paxg_buyers"] - summary["paxg"]["avg_buyers"]) > 10 else "WEAK" if history else "N/A"
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting sentiment history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting sentiment history: {str(e)}")

@app.get("/api/angular")
def get_angular_analysis():
    """Endpoint para análise angular completa"""
    try:
        if not cache["angular_data"]:
            return {
                "message": "Insufficient angular data - need at least 5 price points",
                "angular_points": 0,
                "status": "warming_up"
            }

        latest_angular = cache["angular_data"][-1]
        patterns = analyze_angular_patterns(cache["angular_data"])

        trend_strength = {}
        for asset in ["btc", "gold", "dxy"]:
            if asset in latest_angular and isinstance(latest_angular[asset], dict):
                angle = latest_angular[asset].get("angle", 0)
                strength = latest_angular[asset].get("strength", 0)

                if strength > 0.8: confidence = "VERY_HIGH"
                elif strength > 0.6: confidence = "HIGH"
                elif strength > 0.4: confidence = "MEDIUM"
                else: confidence = "LOW"

                trend_strength[asset] = {
                    "confidence": confidence,
                    "direction": "UP" if angle > 0 else "DOWN",
                    "magnitude": "STRONG" if abs(angle) > 30 else "MODERATE" if abs(angle) > 15 else "WEAK"
                }
            else:
                trend_strength[asset] = {
                    "confidence": "N/A", "direction": "N/A", "magnitude": "N/A"
                }

        btc_angle = latest_angular.get("btc", {}).get("angle", 0)
        gold_angle = latest_angular.get("gold", {}).get("angle", 0)
        dxy_angle = latest_angular.get("dxy", {}).get("angle", 0)

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "current_angles": {
                "btc": latest_angular.get("btc", {"angle":0,"strength":0,"trend":"NEUTRAL"}),
                "gold": latest_angular.get("gold", {"angle":0,"strength":0,"trend":"NEUTRAL"}),
                "dxy": latest_angular.get("dxy", {"angle":0,"strength":0,"trend":"NEUTRAL"})
            },
            "trend_strength": trend_strength,
            "patterns": patterns,
            "historical_points": len(cache["angular_data"]),
            "analysis_summary": {
                "market_direction": "RISK_ON" if (btc_angle > 0 and gold_angle < 10) else "RISK_OFF" if (btc_angle < 0 and gold_angle > 0) else "MIXED",
                "dollar_dominance": "STRONG" if dxy_angle > 20 else "WEAK" if dxy_angle < -10 else "NEUTRAL",
                "average_strength": round(sum([
                    latest_angular[asset]["strength"] for asset in ["btc", "gold", "dxy"] 
                    if asset in latest_angular and isinstance(latest_angular[asset], dict) and 'strength' in latest_angular[asset]
                ]) / 3, 3) if latest_angular and len(cache["angular_data"]) >= 3 else 0,
                "volatility_score": round(sum([
                    abs(latest_angular[asset]["angle"]) for asset in ["btc", "gold", "dxy"] 
                    if asset in latest_angular and isinstance(latest_angular[asset], dict) and 'angle' in latest_angular[asset]
                ]) / 3, 1) if latest_angular and len(cache["angular_data"]) >= 3 else 0
            }
        }

    except Exception as e:
        logger.error(f"❌ Error in angular analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in angular analysis: {str(e)}")

@app.get("/api/patterns")
def get_patterns():
    """Endpoint para padrões angulares detectados"""
    try:
        if not cache["angular_data"]:
            return {
                "patterns": [],
                "message": "Insufficient data for pattern analysis",
                "status": "waiting_for_data"
            }

        patterns = analyze_angular_patterns(cache["angular_data"])

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "total_patterns": patterns["total_patterns"],
            "overall_confidence": patterns["confidence"],
            "all_patterns": patterns["patterns"],
            "pattern_statistics": {
                "by_severity": {
                    "HIGH": len([p for p in patterns["patterns"] if p.get("severity") == "HIGH"]),
                    "MEDIUM": len([p for p in patterns["patterns"] if p.get("severity") == "MEDIUM"]),
                    "LOW": len([p for p in patterns["patterns"] if p.get("severity") == "LOW"])
                },
                "high_confidence": len([p for p in patterns["patterns"] if p.get("confidence", 0) > 0.7]),
                "medium_confidence": len([p for p in patterns["patterns"] if 0.4 <= p.get("confidence", 0) <= 0.7]),
                "low_confidence": len([p for p in patterns["patterns"] if p.get("confidence", 0) < 0.4])
            }
        }

    except Exception as e:
        logger.error(f"❌ Error in pattern analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in pattern analysis: {str(e)}")

@app.get("/api/calendar")
def get_economic_calendar():
    """Endpoint para calendário econômico FRED completo"""
    try:
        update_fred_data()

        upcoming_events = cache["fred_data"]["upcoming_events"]
        pre_event_alerts = cache["fred_data"]["pre_event_alerts"]

        today = datetime.now().date()
        today_events = [
            e for e in upcoming_events
            if "date" in e and datetime.fromisoformat(e["date"]).date() == today
        ]

        critical_events = sorted(
            [e for e in upcoming_events if e.get("importance") == "HIGH"],
            key=lambda x: x.get("impact_score", 0), reverse=True
        )[:10]

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "fred_api_status": "simulated",
            "data_freshness": cache["fred_data"]["last_fred_update"],
            "total_events": len(upcoming_events),
            "upcoming_events": upcoming_events[:20],
            "critical_events": critical_events,
            "today_events": today_events,
            "pre_event_alerts": pre_event_alerts,
            "summary": {
                "total_upcoming": len(upcoming_events),
                "high_impact_events": len([e for e in upcoming_events if e.get("importance") == "HIGH"]),
                "medium_impact_events": len([e for e in upcoming_events if e.get("importance") == "MEDIUM"]),
                "low_impact_events": len([e for e in upcoming_events if e.get("importance") == "LOW"]),
                "events_today": len(today_events),
                "pre_event_alerts": len(pre_event_alerts)
            },
            "statistics": {
                "total_events_processed": len(upcoming_events),
                "avg_impact_score": round(sum([e.get("impact_score", 0) for e in upcoming_events]) / max(len(upcoming_events), 1), 1),
                "categories": {
                    "EMPLOYMENT": len([e for e in upcoming_events if e.get("category") == "EMPLOYMENT"]),
                    "INFLATION": len([e for e in upcoming_events if e.get("category") == "INFLATION"]),
                    "MONETARY_POLICY": len([e for e in upcoming_events if e.get("category") == "MONETARY_POLICY"]),
                    "GDP": len([e for e in upcoming_events if e.get("category") == "GDP"]),
                    "CONSUMER": len([e for e in upcoming_events if e.get("category") == "CONSUMER"]),
                    "INDUSTRIAL": len([e for e in upcoming_events if e.get("category") == "INDUSTRIAL"]),
                    "HOUSING": len([e for e in upcoming_events if e.get("category") == "HOUSING"])
                }
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting economic calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting economic calendar: {str(e)}")

@app.get("/api/events")
def get_upcoming_events():
    """Endpoint para eventos econômicos próximos"""
    try:
        upcoming_events = cache["fred_data"]["upcoming_events"]
        current_time = datetime.now()
        
        # Filter events for next 7 days
        next_week_events = []
        for event in upcoming_events:
            try:
                event_date = datetime.fromisoformat(event["date"])
                days_until = (event_date.date() - current_time.date()).days
                if 0 <= days_until <= 7:
                    event_copy = event.copy()
                    event_copy["days_until"] = days_until
                    event_copy["formatted_date"] = event_date.strftime("%d/%m/%Y %H:%M")
                    next_week_events.append(event_copy)
            except:
                continue
        
        # Sort by date
        next_week_events.sort(key=lambda x: x["date"])
        
        return {
            "timestamp": current_time.isoformat(),
            "status": "active",
            "events_count": len(next_week_events),
            "period": "next_7_days",
            "upcoming_events": next_week_events[:20],
            "high_impact_today": len([e for e in next_week_events if e.get("days_until") == 0 and e.get("importance") == "HIGH"]),
            "critical_this_week": len([e for e in next_week_events if e.get("importance") == "HIGH"])
        }

    except Exception as e:
        logger.error(f"❌ Error getting upcoming events: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting upcoming events: {str(e)}")

@app.get("/api/alerts")
def get_all_alerts():
    """Endpoint para todos os tipos de alertas"""
    try:
        all_alerts = cache["alerts"].copy()
        fred_alerts = cache["fred_data"]["pre_event_alerts"].copy()
        
        # Add FRED alerts to main alerts
        all_alerts.extend(fred_alerts)
        
        # Generate some angular pattern alerts if angular data exists
        angular_alerts = []
        if cache["angular_data"]:
            patterns = analyze_angular_patterns(cache["angular_data"])
            for pattern in patterns.get("patterns", []):
                angular_alert = {
                    "type": "ANGULAR_PATTERN",
                    "title": f"📐 {pattern['title']}",
                    "message": pattern["description"],
                    "severity": pattern["severity"],
                    "timestamp": datetime.now().isoformat(),
                    "pattern_type": pattern["type"],
                    "confidence": pattern["confidence"]
                }
                angular_alerts.append(angular_alert)
        
        all_alerts.extend(angular_alerts)
        
        # Add MACD Crossover alerts (NEW)
        all_alerts.extend(list(macd_detector.crossover_history))

        # Sort by severity and timestamp
        def alert_priority(alert):
            severity_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            return severity_weight.get(alert.get("severity", "LOW"), 1)
        
        all_alerts.sort(key=lambda x: (alert_priority(x), x.get("timestamp", "")), reverse=True)
        
        # Limit to most recent alerts
        all_alerts = all_alerts[:50]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "total_alerts": len(all_alerts),
            "alerts": all_alerts,
            "alerts_by_type": {
                "angular_patterns": len([a for a in all_alerts if a.get("type") == "ANGULAR_PATTERN"]),
                "fred_events": len([a for a in all_alerts if a.get("type") == "PRE_EVENT_ALERT"]),
                "trading_signals": len([a for a in all_alerts if a.get("type", "").startswith("BTC_") or a.get("type", "").startswith("TRADING_")]),
                "macd_crossovers": len([a for a in all_alerts if a.get("type") == "MACD_CROSSOVER"]), # NEW
                "system_alerts": len([a for a in all_alerts if a.get("type") == "SYSTEM"])
            },
            "alerts_by_severity": {
                "HIGH": len([a for a in all_alerts if a.get("severity") == "HIGH"]),
                "MEDIUM": len([a for a in all_alerts if a.get("severity") == "MEDIUM"]),
                "LOW": len([a for a in all_alerts if a.get("severity") == "LOW"])
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@app.get("/api/backtest-recommendations")
def get_backtest_recommendations():
    """Endpoint para recomendações de backtest"""
    try:
        current_time = datetime.now()
        last_update = backtest_recommendations_cache["last_update"]
        
        # Check if we need to update recommendations
        if (last_update is None or 
            (current_time - datetime.fromisoformat(last_update)).total_seconds() >= 
            backtest_recommendations_cache["update_interval_minutes"] * 60):
            
            logger.info("🔄 Updating backtest recommendations...")
            run_backtest_periodically()
        
        recommendations = backtest_recommendations_cache["recommendations"]
        
        return {
            "timestamp": current_time.isoformat(),
            "status": "active",
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "last_backtest_update": backtest_recommendations_cache["last_update"],
            "next_update_in_minutes": backtest_recommendations_cache["update_interval_minutes"],
            "performance_summary": {
                "avg_success_rate": round(sum([r["success_rate"] for r in recommendations]) / max(len(recommendations), 1), 3),
                "avg_confidence": round(sum([r["confidence"] for r in recommendations]) / max(len(recommendations), 1), 3),
                "long_signals": len([r for r in recommendations if r.get("trade_type") == "LONG"]),
                "short_signals": len([r for r in recommendations if r.get("trade_type") == "SHORT"]),
                "high_confidence": len([r for r in recommendations if r.get("confidence", 0) > 0.8])
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting backtest recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting backtest recommendations: {str(e)}")

# ===============================================================================
# TRADING BOT API ENDPOINTS
# ===============================================================================

@app.get("/api/trading-bot/status")
def get_trading_bot_status():
    """Endpoint para status do trading bot"""
    try:
        current_time = datetime.now()
        
        # Calculate uptime (simulated)
        if trading_bot_state["last_update"]:
            last_update_dt = datetime.fromisoformat(trading_bot_state["last_update"])
            uptime_seconds = (current_time - last_update_dt).total_seconds()
        else:
            uptime_seconds = 0
            
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        
        return {
            "timestamp": current_time.isoformat(),
            "running": trading_bot_state["running"],
            "last_update": trading_bot_state["last_update"],
            "uptime": f"{uptime_hours}h {uptime_minutes}m",
            "uptime_seconds": uptime_seconds,
            "active_positions_count": len(trading_bot_state["active_positions"]),
            "active_signals_count": len(trading_bot_state["active_signals"]),
            "status_details": {
                "strategy_count": 4,  # Number of active strategies
                "last_signal": trading_bot_state["active_signals"][-1]["timestamp"] if trading_bot_state["active_signals"] else None,
                "last_trade": trading_bot_state["performance"]["last_update"],
                "auto_trading_enabled": True,
                "risk_management_active": True
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting trading bot status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trading bot status: {str(e)}")

@app.get("/api/trading-bot/positions")
def get_active_positions():
    """Endpoint para posições ativas do bot"""
    try:
        current_time = datetime.now()
        positions = trading_bot_state["active_positions"]
        
        # Update current prices and PnL for positions
        for position in positions:
            # Simula price movement
            current_price_multiplier = random.uniform(0.998, 1.002) 
            position["current_price"] = position["entry_price"] * current_price_multiplier
            
            # Calculate PnL
            if position["side"] == "long":
                position["pnl"] = (position["current_price"] - position["entry_price"]) * position["size"]
            else:  # short
                position["pnl"] = (position["entry_price"] - position["current_price"]) * position["size"]
            
            # Evita divisão por zero se entry_price for 0
            if (position["entry_price"] * position["size"]) != 0:
                position["pnl_percent"] = (position["pnl"] / (position["entry_price"] * position["size"])) * 100
            else:
                position["pnl_percent"] = 0.0
            
            position["last_update"] = current_time.isoformat()
        
        total_pnl = sum([pos["pnl"] for pos in positions])
        
        return {
            "timestamp": current_time.isoformat(),
            "total_positions": len(positions),
            "positions": positions,
            "portfolio_summary": {
                "total_pnl": total_pnl,
                "total_invested": sum([pos["entry_price"] * pos["size"] for pos in positions]),
                "long_positions": len([p for p in positions if p["side"] == "long"]),
                "short_positions": len([p for p in positions if p["side"] == "short"]),
                "profitable_positions": len([p for p in positions if p["pnl"] > 0]),
                "losing_positions": len([p for p in positions if p["pnl"] < 0])
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting active positions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active positions: {str(e)}")

@app.get("/api/trading-bot/signals")
def get_active_signals():
    """Endpoint para sinais ativos do bot"""
    try:
        current_time = datetime.now()
        signals = trading_bot_state["active_signals"]
        
        # Update signal relevance (signals expire after 1 hour)
        valid_signals = []
        for signal in signals:
            try:
                signal_time = datetime.fromisoformat(signal["timestamp"])
                hours_old = (current_time - signal_time).total_seconds() / 3600
                
                if hours_old < 1:  # Signal is still valid
                    signal["hours_old"] = round(hours_old, 2)
                    signal["expires_in_minutes"] = round((1 - hours_old) * 60, 1)
                    signal["status"] = "ACTIVE"
                    valid_signals.append(signal)
                elif hours_old < 2:  # Recently expired
                    signal["hours_old"] = round(hours_old, 2)
                    signal["expires_in_minutes"] = 0
                    signal["status"] = "EXPIRED"
                    valid_signals.append(signal)
            except ValueError:
                logger.warning(f"Timestamp inválido no sinal: {signal.get('timestamp')}. Ignorando este sinal.")
                continue

        # Update the cache with only ACTIVE signals for future processing
        trading_bot_state["active_signals"] = [s for s in valid_signals if s["status"] == "ACTIVE"]
        
        return {
            "timestamp": current_time.isoformat(),
            "total_signals": len(valid_signals),
            "active_signals": len([s for s in valid_signals if s["status"] == "ACTIVE"]),
            "expired_signals": len([s for s in valid_signals if s["status"] == "EXPIRED"]),
            "signals": valid_signals,
            "signals_summary": {
                "long_signals": len([s for s in valid_signals if s["signal_type"] == "LONG"]),
                "short_signals": len([s for s in valid_signals if s["signal_type"] == "SHORT"]),
                "high_confidence": len([s for s in valid_signals if s["confidence"] > 0.8]),
                "medium_confidence": len([s for s in valid_signals if 0.6 <= s["confidence"] <= 0.8]),
                "low_confidence": len([s for s in valid_signals if s["confidence"] < 0.6])
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting active signals: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active signals: {str(e)}")

@app.get("/api/trading-bot/performance")
def get_bot_performance():
    """Endpoint para métricas de performance do bot"""
    try:
        current_time = datetime.now()
        performance = trading_bot_state["performance"]
        
        # Calculate additional metrics
        roi_percentage = ((performance["current_balance"] - performance["start_balance"]) / performance["start_balance"]) * 100 if performance["start_balance"] > 0 else 0
        win_rate = (performance["winning_trades"] / max(performance["total_trades"], 1)) * 100
        avg_pnl_per_trade = performance["total_pnl"] / max(performance["total_trades"], 1)
        
        # Simulated additional metrics
        sharpe_ratio = random.uniform(1.0, 2.5) if performance["total_trades"] > 5 else 0
        sortino_ratio = random.uniform(1.2, 3.0) if performance["total_trades"] > 5 else 0
        calmar_ratio = random.uniform(0.8, 2.0) if performance["total_trades"] > 5 else 0
        profit_factor = random.uniform(1.1, 2.5) if performance["winning_trades"] > 0 else 0
        
        return {
            "timestamp": current_time.isoformat(),
            "current_balance": performance["current_balance"],
            "start_balance": performance["start_balance"],
            "total_pnl": performance["total_pnl"],
            "roi_percentage": round(roi_percentage, 2),
            "winning_trades": performance["winning_trades"],
            "total_trades": performance["total_trades"],
            "win_rate": round(win_rate, 2),
            "max_drawdown": performance["max_drawdown"],
            "daily_pnl": performance["daily_pnl"],
            "daily_trades": performance["daily_trades"],
            "avg_pnl_per_trade": round(avg_pnl_per_trade, 2),
            "last_update": performance["last_update"],
            "advanced_metrics": {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "sortino_ratio": round(sortino_ratio, 2),
                "calmar_ratio": round(calmar_ratio, 2),
                "volatility": round(random.uniform(5.0, 15.0), 2),
                "beta": round(random.uniform(0.7, 1.3), 2),
                "alpha": round(random.uniform(-2.0, 5.0), 2),
                "maximum_consecutive_wins": random.randint(3, 8),
                "maximum_consecutive_losses": random.randint(1, 4),
                "profit_factor": round(profit_factor, 2)
            },
            "performance_periods": {
                "daily": {
                    "pnl": performance["daily_pnl"],
                    "trades": performance["daily_trades"],
                    "return_pct": round((performance["daily_pnl"] / performance["current_balance"]) * 100, 2) if performance["current_balance"] > 0 else 0
                },
                "weekly": {
                    "pnl": round(performance["total_pnl"] * 0.3, 2),
                    "trades": int(performance["total_trades"] * 0.4),
                    "return_pct": round(random.uniform(-1.0, 3.0), 2)
                },
                "monthly": {
                    "pnl": round(performance["total_pnl"] * 0.8, 2),
                    "trades": int(performance["total_trades"] * 0.9),
                    "return_pct": round(random.uniform(-2.0, 8.0), 2)
                }
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting bot performance: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting bot performance: {str(e)}")

@app.post("/api/trading-bot/start")
def start_trading_bot():
    """Endpoint para iniciar o trading bot"""
    try:
        if trading_bot_state["running"]:
            return {
                "status": "already_running",
                "message": "Trading bot is already running",
                "timestamp": datetime.now().isoformat()
            }
        
        trading_bot_state["running"] = True
        trading_bot_state["last_update"] = datetime.now().isoformat()
        
        # Add system alert
        start_alert = {
            "type": "SYSTEM",
            "title": "🚀 Trading Bot Iniciado",
            "message": "O trading bot foi iniciado com sucesso e está monitorando o mercado",
            "severity": "HIGH",
            "timestamp": datetime.now().isoformat()
        }
        cache["alerts"].append(start_alert)
        
        logger.info("🚀 Trading bot started successfully")
        
        return {
            "status": "started",
            "message": "Trading bot started successfully",
            "timestamp": datetime.now().isoformat(),
            "bot_config": {
                "max_positions": 3,
                "risk_per_trade": 2.0,
                "strategies_active": 4,
                "auto_trading": True
            }
        }

    except Exception as e:
        logger.error(f"❌ Error starting trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting trading bot: {str(e)}")

@app.post("/api/trading-bot/stop")
def stop_trading_bot():
    """Endpoint para parar o trading bot"""
    try:
        if not trading_bot_state["running"]:
            return {
                "status": "already_stopped",
                "message": "Trading bot is already stopped",
                "timestamp": datetime.now().isoformat()
            }
        
        trading_bot_state["running"] = False
        trading_bot_state["last_update"] = datetime.now().isoformat()
        
        # Add system alert
        stop_alert = {
            "type": "SYSTEM",
            "title": "🛑 Trading Bot Parado",
            "message": "O trading bot foi parado. Posições existentes permanecem ativas para gestão manual",
            "severity": "MEDIUM",
            "timestamp": datetime.now().isoformat()
        }
        cache["alerts"].append(stop_alert)
        
        logger.info("🛑 Trading bot stopped successfully")
        
        return {
            "status": "stopped",
            "message": "Trading bot stopped successfully",
            "timestamp": datetime.now().isoformat(),
            "final_stats": {
                "active_positions": len(trading_bot_state["active_positions"]),
                "pending_signals": len(trading_bot_state["active_signals"]),
                "session_pnl": trading_bot_state["performance"]["daily_pnl"]
            }
        }

    except Exception as e:
        logger.error(f"❌ Error stopping trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping trading bot: {str(e)}")

# ===============================================================================
# SYSTEM & DEBUG ENDPOINTS
# ===============================================================================

@app.get("/api/status")
def get_system_status():
    """Endpoint para status geral do sistema"""
    try:
        current_time = datetime.now()
        
        # Calculate uptime (simulated)
        uptime_seconds = random.randint(3600, 86400)  # 1 hour to 1 day
        uptime_hours = uptime_seconds // 3600
        uptime_minutes = (uptime_seconds % 3600) // 60
        
        return {
            "timestamp": current_time.isoformat(),
            "status": "healthy",
            "version": SYSTEM_INFO["version"],
            "uptime": f"{uptime_hours}h {uptime_minutes}m",
            "uptime_seconds": uptime_seconds,
            "system_health": {
                "api_status": "operational",
                "database_status": "connected",
                "websocket_status": { # Updated to reflect both sentiment and ohlcv
                    "sentiment": "active" if sentiment_cache["websocket_connected"] else "disconnected",
                    "ohlcv": "active" if realtime_ohlcv_cache["btc"]["websocket_connected"] else "disconnected"
                },
                "fred_api_status": "simulated",
                "gate_io_api_status": "connected" if sentiment_cache["websocket_connected"] or realtime_ohlcv_cache["btc"]["websocket_connected"] else "failed",
                "trading_bot_status": "running" if trading_bot_state["running"] else "stopped"
            },
            "performance_metrics": {
                "active_connections": len(active_sentiment_websocket_connections) + len(active_ohlcv_websocket_connections), # Sum of both
                "cache_size_mb": round(len(str(cache)) / 1024 / 1024, 2),
                "requests_per_minute": random.randint(50, 200),
                "average_response_time_ms": random.randint(10, 100),
                "error_rate_percent": round(random.uniform(0.1, 2.0), 2)
            },
            "data_freshness": {
                "price_data": cache.get("timestamp"),
                "angular_analysis": cache.get("last_angular_analysis"),
                "fred_data": cache["fred_data"].get("last_fred_update"),
                "sentiment_data": sentiment_cache["btc_sentiment"].get("last_update"),
                "backtest_data": backtest_recommendations_cache.get("last_update"),
                "realtime_ohlcv_data": realtime_ohlcv_cache["btc"].get("last_update") # NEW
            },
            "feature_status": {
                "real_time_prices": True,
                "angular_analysis": len(cache["angular_data"]) > 0,
                "economic_calendar": len(cache["fred_data"]["upcoming_events"]) > 0,
                "sentiment_monitoring": sentiment_cache["websocket_connected"],
                "realtime_macd_system": realtime_ohlcv_cache["btc"]["websocket_connected"], # NEW
                "trading_bot": trading_bot_state["running"],
                "backtest_engine": len(backtest_recommendations_cache["recommendations"]) > 0,
                "alert_system": len(cache["alerts"]) > 0
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.get("/api/debug/gateio")
def debug_gateio_connection():
    """Endpoint para testar conectividade com Gate.io API"""
    try:
        current_time = datetime.now()
        
        # Test connection status
        connection_test = {
            "timestamp": current_time.isoformat(),
            "api_base_url": "https://api.gateio.ws/api/v4",
            "websocket_url": "wss://api.gateio.ws/ws/v4/",
            "test_results": {}
        }
        
        # Simulate API tests
        test_endpoints = [
            {"name": "BTC Order Book", "endpoint": "/spot/order_book?currency_pair=BTC_USDT", "status": "success" if sentiment_cache["websocket_connected"] else "failed"},
            {"name": "PAXG Order Book", "endpoint": "/spot/order_book?currency_pair=PAXG_USDT", "status": "success" if sentiment_cache["websocket_connected"] else "failed"},
            {"name": "BTC Ticker", "endpoint": "/spot/tickers?currency_pair=BTC_USDT", "status": "success" if sentiment_cache["websocket_connected"] else "failed"},
            {"name": "WebSocket Sentiment", "endpoint": "ws://sentiment", "status": "connected" if sentiment_cache["websocket_connected"] else "disconnected"}, # Updated name
            {"name": "WebSocket OHLCV", "endpoint": "ws://ohlcv", "status": "connected" if realtime_ohlcv_cache["btc"]["websocket_connected"] else "disconnected"} # NEW
        ]
        
        for test in test_endpoints:
            connection_test["test_results"][test["name"]] = {
                "endpoint": test["endpoint"],
                "status": test["status"],
                "response_time_ms": random.randint(10, 200) if test["status"] in ["success", "connected"] else None, # Updated condition
                "last_success": current_time.isoformat() if test["status"] in ["success", "connected"] else None # Updated condition
            }
        
        # Overall status
        successful_tests = len([t for t in test_endpoints if t["status"] in ["success", "connected"]])
        connection_test["overall_status"] = "healthy" if successful_tests >= 4 else "degraded" if successful_tests >= 2 else "failed" # Adjusted success threshold
        connection_test["success_rate"] = round((successful_tests / len(test_endpoints)) * 100, 1)
        
        # Current sentiment data status
        connection_test["current_data"] = {
            "btc_sentiment": {
                "buyers": sentiment_cache["btc_sentiment"]["buyers"],
                "sellers": sentiment_cache["btc_sentiment"]["sellers"],
                "last_update": sentiment_cache["btc_sentiment"]["last_update"],
                "data_age_seconds": (current_time - datetime.fromisoformat(sentiment_cache["btc_sentiment"]["last_update"])).total_seconds() if sentiment_cache["btc_sentiment"]["last_update"] else None
            },
            "paxg_sentiment": {
                "buyers": sentiment_cache["paxg_sentiment"]["buyers"],
                "sellers": sentiment_cache["paxg_sentiment"]["sellers"],
                "last_update": sentiment_cache["paxg_sentiment"]["last_update"],
                "data_age_seconds": (current_time - datetime.fromisoformat(sentiment_cache["paxg_sentiment"]["last_update"])).total_seconds() if sentiment_cache["paxg_sentiment"]["last_update"] else None
            },
            "realtime_ohlcv_btc": { # NEW
                "connected": realtime_ohlcv_cache["btc"]["websocket_connected"],
                "last_update": realtime_ohlcv_cache["btc"]["last_update"],
                "data_age_seconds": (current_time - datetime.fromisoformat(realtime_ohlcv_cache["btc"]["last_update"])).total_seconds() if realtime_ohlcv_cache["btc"]["last_update"] else None
            },
            "websocket_connections": len(active_sentiment_websocket_connections) + len(active_ohlcv_websocket_connections) # Sum of both
        }
        
        return connection_test

    except Exception as e:
        logger.error(f"❌ Error in Gate.io debug: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing Gate.io connection: {str(e)}")

# ===============================================================================
# STARTUP EVENTS & BACKGROUND TASKS
# ===============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização da aplicação - UPDATED"""
    logger.info("🚀 Starting Trading Dashboard API v6.0 with Real-time MACD")
    
    # Sistemas existentes...
    logger.info("📅 Starting FRED data scheduler...")
    start_backtest_scheduler()
    
    logger.info("🤖 Starting trading bot simulation...")
    start_trading_bot_simulation()
    
    logger.info("🎭 Starting sentiment monitoring...")
    asyncio.create_task(simulate_and_send_sentiment_data())
    
    # NOVO: Sistema MACD tempo real
    logger.info("📊 Starting real-time OHLCV and MACD system...")
    asyncio.create_task(fetch_realtime_ohlcv_gateio())
    
    # Load initial data...
    logger.info("📊 Loading initial market data...")
    try:
        initial_data = get_current_market_data()
        cache["data"] = initial_data
        cache["timestamp"] = datetime.now().isoformat()
        update_price_history(initial_data)
        update_fred_data()
        logger.info("✅ Initial data loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Error loading initial data: {e}")
    
    logger.info("🎯 All systems initialized and ready!")
    logger.info("📊 Real-time MACD crossover detection is ACTIVE!")

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de encerramento da aplicação"""
    logger.info("🛑 Shutting down Trading Dashboard API...")
    
    # Stop trading bot
    trading_bot_state["running"] = False
    
    # Close WebSocket connections
    for ws in active_sentiment_websocket_connections:
        try:
            await ws.close()
        except:
            pass
    
    # Close OHLCV WebSocket connections (NEW)
    for ws in active_ohlcv_websocket_connections:
        try:
            await ws.close()
        except:
            pass

    logger.info("✅ Shutdown completed successfully")

# ===============================================================================
# MAIN APPLICATION ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🚀 TRADING DASHBOARD API v6.0 - COMPATIBLE")
    print("=" * 80)
    print(f"📝 Description: {SYSTEM_INFO['description']}")
    print("🔧 Features:")
    for feature in SYSTEM_INFO['features']:
        print(f"   {feature}")
    print("=" * 80)
    print(f"🌐 Server starting on: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"📚 API Documentation: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/docs")
    print(f"🔍 ReDoc Documentation: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/redoc")
    print("=" * 80)
    print("🎯 Main Endpoints:")
    print("   📊 Real-time data: /api/current")
    print("   🎭 Market sentiment: /api/sentiment (+ WebSocket: /ws/sentiment)")
    print("   📈 Real-time OHLCV + MACD: /api/macd/realtime/{asset} (+ WebSocket: /ws/ohlcv) - NEW")
    print("   🤖 Trading bot: /api/trading-bot/*")
    print("   📡 Backtest signals: /api/backtest-recommendations")
    print("   📅 Economic calendar: /api/calendar")
    print("   🚨 Alerts system: /api/alerts")
    print("   ⚙️ System status: /api/status")
    print("=" * 80)
    print("⚠️  IMPORTANT NOTES:")
    print("   • This is a demonstration/educational system")
    print("   • Real Gate.io API integration for sentiment analysis")
    print("   • FRED economic data is simulated")
    print("   • Trading bot is fully simulated")
    print("   • Not financial advice - use at your own risk")
    print("=" * 80)
    
    # Run the application
    uvicorn.run(
        "radar-dash:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"],
        log_level=SERVER_CONFIG["log_level"],
        access_log=True
    )
