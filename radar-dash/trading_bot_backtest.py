import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Deque, Any
import math
from scipy import stats
import pytz # Importar pytz para timezones
import requests
import json
from dataclasses import dataclass, field
import threading
import time
import random
import asyncio
import aiohttp
import talib
import websockets
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys

# Importar o bot de trading real
# A classe do bot foi renomeada para CombinedAITradingBot e importamos RISK_CONFIG
try:
    from trading_bot_backtest import CombinedAITradingBot, API_KEY, SECRET, ENVIRONMENT, FRED_API_KEY as TRADING_BOT_FRED_API_KEY, NEWS_API_KEY as TRADING_BOT_NEWS_API_KEY, CRYPTOPANIC_API_KEY as TRADING_BOT_CRYPTOPANIC_API_KEY, TRADING_SYMBOLS, RISK_CONFIG
except ImportError as e:
    logging.error(f"Failed to import trading_bot_backtest: {e}")
    logging.error("Please ensure trading_bot_backtest.py is in the same directory or accessible via PYTHONPATH.")
    # Fallback/Dummy classes if import fails
    class CombinedAITradingBot:
        def __init__(self, environment="testnet"):
            self.environment = environment
            self.running = False
            self.performance = {
                "current_balance": 0.0, "start_balance": 0.0, "total_pnl": 0.0,
                "roi_percentage": 0.0, "winning_trades": 0, "total_trades": 0,
                "win_rate": 0.0, "max_drawdown": 0.0, "daily_pnl": 0.0,
                "daily_trades": 0, "last_update": datetime.now().isoformat(),
                "ai_accuracy": 0.0, "ai_predictions": 0, "ai_correct_predictions": 0,
                "pattern_accuracy": 0.0, "sentiment_impact": 0.0,
                "session_start_time": datetime.now().isoformat()
            }
            self.active_positions = []
            self.signals_history = []
            # Define risk_config para o mock bot
            self.risk_config = {
                'max_open_positions': 2,
                'stop_loss_atr_multiplier': 1.5,
                'position_size_usdt': 15,
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
                'ai_confidence_boost': 0.20
            }
            self.urls = {"rest": "https://api-testnet.gateapi.io", "ws": "wss://ws-testnet.gate.io/v4/ws/futures/usdt"}
            class MockAnalyzer:
                def __init__(self):
                    self.fred_events_cache = []
                    self.last_fred_update = None
                    class MockFredCalendar:
                        def __init__(self):
                            self.api_key = "MOCK_FRED_KEY"
                            self.cache = {"upcoming_events": [], "last_full_update": None}
                        def get_upcoming_releases(self, days_ahead=14):
                            return []
                        def generate_pre_event_alerts(self, hours_before=48):
                            return []
                        def get_next_critical_event(self):
                            return None
                        def get_high_impact_events_today(self):
                            return []
                    self.fred_calendar = MockFredCalendar()
                    self.sentiment_analyzer = self.MockSentimentAnalyzer()
                    self.ml_predictor = self.MockMLPredictor()
                class MockSentimentAnalyzer:
                    def __init__(self):
                        self.sentiment_cache = {}
                        self.last_update = {}
                        self.cryptopanic_api_key = "MOCK_KEY"
                        self.news_api_key = "MOCK_KEY"
                    def get_crypto_news_sentiment(self, symbol):
                        return random.uniform(0.3, 0.7)
                class MockMLPredictor:
                    def __init__(self):
                        self.model_accuracy = 0.0
                        self.training_data = []
            self.analyzer = MockAnalyzer()

        def run(self):
            self.running = True
            logging.info("Simulated trading bot loop started (Mock Bot).")
            for _ in range(3):
                time.sleep(5)
                self.performance["current_balance"] += random.uniform(-10, 10)
                self.performance["total_pnl"] += random.uniform(-10, 10)
                self.performance["total_trades"] += 1
                self.performance["winning_trades"] += random.randint(0,1)
                if self.performance["total_trades"] > 0:
                    self.performance["win_rate"] = (self.performance["winning_trades"] / self.performance["total_trades"]) * 100
                    self.performance["roi_percentage"] = (self.performance["total_pnl"] / 1000) * 100
                self.performance["last_update"] = datetime.now().isoformat()
            self.running = False
            logging.info("Simulated trading bot loop finished (Mock Bot).")

        def stop(self):
            self.running = False
            logging.info("Simulated trading bot stopped (Mock Bot).")

        def get_account_balance(self):
            return 10000.0 if self.environment == 'testnet' else 0.0

        def get_current_price(self, symbol):
            if symbol == "BTC_USDT": return 65000.0 + random.uniform(-1000, 1000)
            if symbol == "ETH_USDT": return 3500.0 + random.uniform(-100, 100)
            return 1.0

        def get_detailed_ai_status(self):
            # Usado para garantir que o frontend receba a estrutura esperada
            testnet_perf = {
                "balance": self.get_account_balance() if self.environment == 'testnet' else 0.0,
                "total_pnl": self.performance["total_pnl"] if self.environment == 'testnet' else 0.0,
                "roi_percentage": self.performance["roi_percentage"] if self.environment == 'testnet' else 0.0,
                "win_rate": self.performance["win_rate"] if self.environment == 'testnet' else 0.0,
                "total_trades": self.performance["total_trades"] if self.environment == 'testnet' else 0,
                "winning_trades": self.performance["winning_trades"] if self.environment == 'testnet' else 0,
                "daily_pnl": self.performance["daily_pnl"] if self.environment == 'testnet' else 0.0,
                "daily_trades": self.performance["daily_trades"] if self.environment == 'testnet' else 0,
            }
            live_perf = {
                "balance": self.get_account_balance() if self.environment == 'live' else 0.0,
                "total_pnl": self.performance["total_pnl"] if self.environment == 'live' else 0.0,
                "roi_percentage": self.performance["roi_percentage"] if self.environment == 'live' else 0.0,
                "win_rate": self.performance["win_rate"] if self.environment == 'live' else 0.0,
                "total_trades": self.performance["total_trades"] if self.environment == 'live' else 0,
                "winning_trades": self.performance["winning_trades"] if self.environment == 'live' else 0,
                "daily_pnl": self.performance["daily_pnl"] if self.environment == 'live' else 0.0,
                "daily_trades": self.performance["daily_trades"] if self.environment == 'live' else 0,
            }

            return {
                "status": "simulated_running" if self.running else "simulated_stopped",
                "environment": self.environment,
                "version": "2.1_simulated",
                "current_balance": self.performance["current_balance"],
                "start_balance": self.performance["start_balance"],
                "total_pnl": self.performance["total_pnl"],
                "roi_percentage": self.performance["roi_percentage"],
                "winning_trades": self.performance["winning_trades"],
                "total_trades": self.performance["total_trades"],
                "win_rate": self.performance["win_rate"],
                "daily_pnl": self.performance["daily_pnl"],
                "daily_trades": self.performance["daily_trades"],
                "testnet_performance": testnet_perf, # Incluído para compatibilidade com o frontend
                "live_performance": live_perf,       # Incluído para compatibilidade com o frontend
                "ai_accuracy": self.performance["ai_accuracy"],
                "ai_predictions": self.performance["ai_predictions"],
                "ai_correct_predictions": self.performance["ai_correct_predictions"],
                "ml_model_accuracy": self.analyzer.ml_predictor.model_accuracy,
                "training_samples": len(self.analyzer.ml_predictor.training_data),
                "last_update": self.performance["last_update"],
                "session_start_time": self.performance["session_start_time"],
                "active_positions_count": len(self.active_positions),
                "active_positions": [],
                "recent_signals_count": len(self.signals_history),
                "recent_signals": [],
                "trade_history_count": 0,
                "trade_history": [],
                "config": {}, # Mock config, adjust if needed
                "ai_system_status": {
                    "ml_available": True, "xgb_available": True, "sentiment_available": True,
                    "model_trained": True, "last_retrain": None, "pattern_database_size": 0,
                    "sentiment_cache_size": 0, "fred_calendar_active": False, "fred_events_cached": 0,
                    "fred_last_update": None, "cryptopanic_active": False
                },
                "ai_features": {}
            }


# Assign default values if not imported
# Estes são usados pelo radar-dash.py diretamente, fora da instância do bot.
# Eles devem ser carregados do .env.
API_KEY = os.getenv('GATE_TESTNET_API_KEY') or os.getenv('GATE_API_KEY') or "YOUR_SIMULATED_API_KEY"
SECRET = os.getenv('GATE_TESTNET_API_SECRET') or os.getenv('GATE_API_SECRET') or "YOUR_SIMULATED_SECRET"
ENVIRONMENT = os.getenv('GATE_ENVIRONMENT', 'testnet')
TRADING_BOT_FRED_API_KEY = os.getenv('FRED_API_KEY') or "DUMMY_KEY_FRED"
TRADING_BOT_NEWS_API_KEY = os.getenv('NEWS_API_KEY') or "DUMMY_KEY_NEWSAPI"
TRADING_BOT_CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY') or "DUMMY_KEY_CRYPTOPANIC"
TRADING_SYMBOLS = ['BTC_USDT', 'ETH_USDT']

# Definindo RISK_CONFIG de fallback no radar-dash.py caso a importação do bot falhe
if 'RISK_CONFIG' not in locals(): # Verifica se RISK_CONFIG já foi importado do bot real
    RISK_CONFIG = {
        'testnet': {
            'position_size_usdt': 15, 'max_open_positions': 2, 'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 4.0, 'max_daily_trades': 10, 'cooldown_minutes': 10,
            'trailing_stop_percent': 1.2, 'reversal_fall_pct': 1.0, 'reversal_rise_pct': 1.0,
            'reversal_volume_multiplier': 2.5, 'reversal_cooldown_minutes': 45,
            'max_correlation': 0.8, 'min_volume_ratio': 1.5, 'rsi_oversold': 25,
            'rsi_overbought': 75, 'ai_confidence_boost': 0.20
        },
        'live': {
            'position_size_usdt': 30, 'max_open_positions': 1, 'stop_loss_atr_multiplier': 1.5,
            'take_profit_atr_multiplier': 3.0, 'max_daily_trades': 6, 'cooldown_minutes': 20,
            'trailing_stop_percent': 0.8, 'reversal_fall_pct': 1.5, 'reversal_rise_pct': 1.5,
            'reversal_volume_multiplier': 3.0, 'reversal_cooldown_minutes': 90,
            'max_correlation': 0.6, 'min_volume_ratio': 2.0, 'rsi_oversold': 20,
            'rsi_overbought': 80, 'ai_confidence_boost': 0.15
        }
    }


# ===============================================================================
# 🔧 CONFIGURAÇÕES (GLOBAL VARIABLES)
# ===============================================================================

# Símbolos dos ativos
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

# Chave da API FRED (usará a do bot)
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
        "📊 Dados de mercado em tempo real (via YFinance e Gate.io)",
        "📐 Análise angular avançada",
        "🚨 Sistema de alertas inteligente",
        "📅 Calendário econômico FRED integrado (via Trading Bot)",
        "🎯 Detecção de padrões complexos",
        "🤖 Trading Bot REAL (com IA, Kelly, Multi-API)",
        "📰 Análise de Sentimento (NewsAPI, CryptoPanic)",
        "📈 MACD em tempo real e detecção de cruzamentos"
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

# Cache para as recomendações do backtest (simulado aqui)
backtest_recommendations_cache = {
    "recommendations": [],
    "last_update": None,
    "update_interval_minutes": 20
}

# Instância global do bot de trading real
real_trading_bot: Optional[CombinedAITradingBot] = None

# Lista de conexões WebSocket ativas
active_sentiment_websocket_connections: List[WebSocket] = []
active_rsi_macd_websocket_connections: List[WebSocket] = []

# ===============================================================================
# 📊 CACHE PARA DADOS TEMPO REAL (NOVO)
# ===============================================================================

# Cache para dados OHLCV em tempo real
realtime_ohlcv_cache = {
    "btc": {
        "candles": deque(maxlen=200),
        "current_candle": None,
        "macd_data": {
            "macd": [],
            "signal": [],
            "histogram": [],
            "last_crossover": None,
            "crossover_alerts": []
        },
        "rsi_data": { # NOVO: Dados RSI
            "rsi": [],
            "last_value": 0.0,
            "angle": 0.0,
            "strength": 0.0,
            "trend": "NEUTRAL"
        },
        "macd_angle_data": { # NOVO: Dados de ângulo do MACD
            "macd_angle": 0.0,
            "macd_angle_strength": 0.0,
            "signal_angle": 0.0,
            "signal_angle_strength": 0.0,
        },
        "stochrsi_data": {"k_value": 0.0, "d_value": 0.0, "k_series": [], "d_series": []}, # NOVO
        "last_update": None,
        "websocket_connected": False
    },
    "eth": {
        "candles": deque(maxlen=200),
        "current_candle": None,
        "macd_data": {
            "macd": [],
            "signal": [],
            "histogram": [],
            "last_crossover": None,
            "crossover_alerts": []
        },
        "rsi_data": { # NOVO: Dados RSI
            "rsi": [],
            "last_value": 0.0,
            "angle": 0.0,
            "strength": 0.0,
            "trend": "NEUTRAL"
        },
        "macd_angle_data": { # NOVO: Dados de ângulo do MACD
            "macd_angle": 0.0,
            "macd_angle_strength": 0.0,
            "signal_angle": 0.0,
            "signal_angle_strength": 0.0,
        },
        "stochrsi_data": {"k_value": 0.0, "d_value": 0.0, "k_series": [], "d_series": []}, # NOVO
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

def calculate_volume_colors_and_macd(hist_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula cores do volume baseado em alta/baixa e melhora dados MACD
    CORRIGIDO para funcionar com dados reais da Gate.io
    """
    try:
        if hist_data.empty or len(hist_data) < 2:
            logger.warning("DataFrame vazio ou insuficiente para calculate_volume_colors_and_macd")
            return hist_data
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in hist_data.columns]
        if missing_cols:
            logger.error(f"Colunas faltando no DataFrame: {missing_cols}")
            return hist_data
        
        hist_data = hist_data.copy()
        hist_data['price_direction'] = 'neutral'
        
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
        
        hist_data.iloc[0, hist_data.columns.get_loc('price_direction')] = 'neutral'
        
        if 'price_direction' in hist_data.columns:
            color_stats = hist_data['price_direction'].value_counts()
            logger.debug(f"Volume colors calculated: {dict(color_stats)}")
        
        return hist_data
        
    except Exception as e:
        logger.error(f"Erro ao calcular cores do volume: {e}")
        return hist_data

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
        
        required_keys = ["btc", "gold", "dxy"]
        for key in required_keys:
            if key not in latest or not isinstance(latest[key], dict):
                logger.warning(f"Missing or invalid '{key}' data in angular analysis. Skipping patterns.")
                return {"patterns": [], "confidence": 0, "total_patterns": 0}

        if (latest["dxy"]["angle"] > ANGULAR_ALERT_THRESHOLDS["perfect_divergence"]["dxy_min_angle"] and 
            latest["dxy"]["strength"] > ANGULAR_ALERT_THRESHOLDS["perfect_divergence"]["min_strength"] and
            latest["btc"]["angle"] < ANGULAR_ALERT_THRESHOLDS["perfect_divergence"]["btc_max_angle"] and 
            latest["btc"]["strength"] > ANGULAR_ALERT_THRESHOLDS["perfect_divergence"]["min_strength"]):
            patterns.append({
                "name": "PERFECT_DIVERGENCE",
                "title": "🎯 Perfect Divergence Detected",
                "description": f"DXY rising {latest['dxy']['angle']:.1f}° while BTC falling {abs(latest['btc']['angle']):.1f}°",
                "severity": "HIGH",
                "confidence": min(latest["dxy"]["strength"], latest["btc"]["strength"]),
                "type": "DIVERGENCE"
            })

        if (latest["btc"]["angle"] > ANGULAR_ALERT_THRESHOLDS["bullish_convergence"]["btc_min_angle"] and 
            latest["gold"]["angle"] > ANGULAR_ALERT_THRESHOLDS["bullish_convergence"]["gold_min_angle"] and 
            latest["dxy"]["angle"] < ANGULAR_ALERT_THRESHOLDS["bullish_convergence"]["dxy_max_angle"]):
            patterns.append({
                "name": "BULLISH_CONVERGENCE",
                "title": "🚀 Bullish Convergence Pattern",
                "description": f"BTC +{latest['btc']['angle']:.1f}°, Gold +{latest['gold']['angle']:.1f}°, DXY {latest['dxy']['angle']:.1f}°",
                "severity": "MEDIUM",
                "confidence": (latest["btc"]["strength"] + latest["gold"]["strength"]) / 2,
                "type": "CONVERGENCE"
            })

        if (latest["btc"]["angle"] < ANGULAR_ALERT_THRESHOLDS["bearish_avalanche"]["btc_max_angle"] and 
            latest["gold"]["angle"] < ANGULAR_ALERT_THRESHOLDS["bearish_avalanche"]["gold_max_angle"] and 
            latest["dxy"]["angle"] > ANGULAR_ALERT_THRESHOLDS["bearish_avalanche"]["dxy_min_angle"]):
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

        points_for_trend = 10 
        if len(cache["price_history"]) < points_for_trend:
            logger.warning(f"Insufficient historical data ({len(cache['price_history'])} points) for robust angular analysis. Skipping.")
            return

        btc_prices_hist = [point["btc_price"] for point in cache["price_history"]]
        gold_prices_hist = [point["gold_price"] for point in cache["price_history"]]
        dxy_prices_hist = [point["dxy_price"] for point in cache["price_history"]]

        btc_prices_current = btc_prices_hist + [current_data["btc"]["current_price"]]
        gold_prices_current = gold_prices_hist + [current_data["gold"]["current_price"]]
        dxy_prices_current = dxy_prices_hist + [current_data["dxy"]["current_price"]]

        angular_point = {
            "timestamp": current_time.isoformat(),
            "btc": calculate_trend_angle(btc_prices_current, 5),
            "gold": calculate_trend_angle(gold_prices_current, 5),
            "dxy": calculate_trend_angle(dxy_prices_current, 5)
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

            current_price = 0.0
            current_volume = 0
            market_open = 0.0
            day_high = 0.0
            day_low = 0.0
            previous_close = 0.0

            # Strategy 1: yfinance ticker.info
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

            # Strategy 2: yfinance ticker.history (fallback)
            if current_price == 0.0 or current_volume == 0:
                try:
                    hist_data = yf.download(symbol, period="1d", interval="1m", progress=False, threads=False, auto_adjust=True) # auto_adjust=True
                    if not hist_data.empty:
                        if current_price == 0.0 and 'Close' in hist_data.columns:
                            current_price = float(hist_data['Close'].iloc[-1])
                        if current_volume == 0 and 'Volume' in hist_data.columns:
                            current_volume = int(hist_data['Volume'].iloc[-1].item()) # .item() para evitar FutureWarning
                        if market_open == 0.0 and 'Open' in hist_data.columns:
                            market_open = float(hist_data['Open'].iloc[-1])
                        if day_high == 0.0 and 'High' in hist_data.columns:
                            day_high = float(hist_data['High'].iloc[-1])
                        if day_low == 0.0 and 'Low' in hist_data.columns:
                            day_low = float(hist_data['Low'].iloc[-1])
                        if previous_close == 0.0 and len(hist_data) > 1:
                            previous_close = float(hist_data['Close'].iloc[-2])

                except Exception as e:
                    logger.warning(f"⚠️ History method failed for {symbol}: {e}")

            # Final fallbacks for missing values
            # ATENÇÃO: Mudando para 0.0 ou valores de fallback que o frontend interpretará como "N/A"
            # para evitar a persistência de dados simulados no frontend.
            if current_price == 0.0:
                logger.warning(f"Using 0.0 as fallback price for {name} (failed to fetch real data).")
                current_price = 0.0 # Agora é 0.0, para o frontend exibir N/A
            if current_volume == 0:
                logger.warning(f"Using 0 as fallback volume for {name}.")
                current_volume = 0 # Não simular volume aleatório
            if market_open == 0.0: market_open = current_price if current_price != 0 else 0.0
            if day_high == 0.0: day_high = current_price if current_price != 0 else 0.0
            if day_low == 0.0: day_low = current_price if current_price != 0 else 0.0
            if previous_close == 0.0: previous_close = current_price if current_price != 0 else 0.0

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
                'current_price': 0.0, # Zero para frontend exibir N/A
                'current_volume': 0,
                'market_open': 0.0,
                'day_high': 0.0,
                'day_low': 0.0,
                'previous_close': 0.0
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

        if real_trading_bot and real_trading_bot.analyzer.fred_calendar:
            bot_fred_data = real_trading_bot.analyzer.fred_calendar.cache
            if real_trading_bot.environment == 'simulate_backtest' and hasattr(real_trading_bot.analyzer.fred_calendar, 'datetime_now_override'):
                real_trading_bot.analyzer.fred_calendar.datetime_now_override = datetime.now()
            
            # Garante que o cache FRED do bot esteja atualizado antes de tentar usar
            real_trading_bot.analyzer._update_fred_events_cache() 

            if bot_fred_data["last_full_update"]:
                 # Certifique-se que o e.to_dict() existe na classe EconomicEvent em trading_bot_backtest.py
                 cache["fred_data"]["upcoming_events"] = [e.to_dict() for e in bot_fred_data["upcoming_events"]]
                 cache["fred_data"]["last_fred_update"] = bot_fred_data["last_full_update"].isoformat()
                 cache["fred_data"]["pre_event_alerts"] = real_trading_bot.analyzer.fred_calendar.generate_pre_event_alerts(hours_before=48)
                 for alert in cache["fred_data"]["pre_event_alerts"]:
                     if alert not in cache["alerts"]:
                         cache["alerts"].append(alert)
                 
                 logger.debug(f"📅 FRED data synced from bot: {len(cache['fred_data']['upcoming_events'])} events")
            else:
                 logger.warning("FRED calendar in bot has not yet updated its cache.")
        else:
            logger.warning("Trading bot or its FRED calendar is not available to sync FRED data.")


        logger.debug(f"📈 History updated: {len(cache['price_history'])} price points")

    except Exception as e:
        logger.error(f"❌ Error updating price history: {e}")

# 💼 BACKTEST ENGINE SYSTEM (Simulado para dashboard)
class BacktestEngine:
    """Engine de backtest para geração de sinais de trading (simulado para o dashboard)"""

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

                trade_type = "LONG"
                if "Short" in pattern["name"] or pattern["avg_return"] < 0:
                    trade_type = "SHORT"

                entry_price = 0
                asset_key = pattern["asset"].lower()
                if asset_key == "btc":
                    entry_price = round(random.uniform(60000, 100000), 2)
                elif asset_key == "gold":
                    entry_price = round(random.uniform(2300, 2800), 2)
                elif asset_key == "dxy":
                    entry_price = round(random.uniform(100, 110), 2)
                else:
                    current_price_data = get_current_market_data()
                    entry_price = current_price_data.get(asset_key, {}).get('current_price', 1000.0)
                    if entry_price == 1.0:
                        entry_price = round(random.uniform(100, 110), 2)


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
                    "avg_pnl_per_trade": f"${simulated_total_pnl/simulated_trades:,.2f}" if simulated_trades > 0 else "$0.00",
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
        """Analisa condições atuais do mercado (simulado para backtest engine)"""
        sentiment_level = sentiment_cache["market_mood"]
        trend = "NEUTRAL"
        if sentiment_level in ["EXTREME_GREED", "GREED", "STRONGLY_BULLISH", "BULLISH"]:
            trend = "BULLISH"
        elif sentiment_level in ["EXTREME_FEAR", "FEAR", "STRONGLY_BEARISH", "BEARISH"]:
            trend = "BEARISH"

        return {
            "market_trend": trend,
            "volatility": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "volume_profile": random.choice(["INCREASING", "DECREASING", "STABLE"]),
            "sentiment": "RISK_ON" if sentiment_level in ["EXTREME_GREED", "GREED", "BULLISH", "STRONGLY_BULLISH"] else "RISK_OFF" if sentiment_level in ["EXTREME_FEAR", "FEAR", "BEARISH", "STRONGLY_BEARISH"] else "NEUTRAL",
            "macro_backdrop": random.choice(["SUPPORTIVE", "CHALLENGING", "MIXED"])
        }

    def _evaluate_pattern_conditions(self, pattern: Dict, market_conditions: Dict) -> float:
        """Avalia condições do padrão (simulado para backtest engine)"""
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
        """Determina tipo de padrão (para backtest engine)"""
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

backtest_engine_instance = BacktestEngine()

def run_backtest_periodically():
    """Executa backtest e armazena recomendações no cache (para dashboard)"""
    logger.info("Starting periodic backtest run for dashboard recommendations")
    try:
        recommendations = backtest_engine_instance.get_backtest_recommendations(top_n=5)
        backtest_recommendations_cache["recommendations"] = recommendations
        backtest_recommendations_cache["last_update"] = datetime.now().isoformat()
        logger.info(f"✅ Dashboard Backtest completed. {len(recommendations)} recommendations cached")
    except Exception as e:
        logger.error(f"❌ Error during periodic dashboard backtest run: {e}")

def start_backtest_scheduler():
    """Inicia scheduler do backtest em thread separado (para dashboard)"""
    logger.info(f"Scheduling dashboard backtest to run every {backtest_recommendations_cache['update_interval_minutes']} minutes")

    def scheduler_loop():
        run_backtest_periodically()
        while True:
            time.sleep(backtest_recommendations_cache["update_interval_minutes"] * 60)
            run_backtest_periodically()

    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Dashboard Backtest scheduler started successfully")

# 🎭 MARKET SENTIMENT SYSTEM - REAL GATE.IO INTEGRATION (PARA PAXG e dados brutos)
async def fetch_gateio_real_sentiment():
    """Busca dados reais de sentimento do Gate.io via API REST"""
    try:
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not found. Please install it: pip install aiohttp")
            return False

        btc_orderbook_url = "https://api.gateio.ws/api/v4/spot/order_book?currency_pair=BTC_USDT&limit=20"
        paxg_orderbook_url = "https://api.gateio.ws/api/v4/spot/order_book?currency_pair=PAXG_USDT&limit=20"
        btc_ticker_url = "https://api.gateio.ws/api/v4/spot/tickers?currency_pair=BTC_USDT"

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        success_count = 0

        async with aiohttp.ClientSession() as session:
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

                        logger.info(f"📊 BTC Orderbook Sentiment: {btc_buyers_pct:.1f}% buyers, {btc_sellers_pct:.1f}% sellers")

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch BTC orderbook for sentiment: {e}")

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
                        
                        base_fgi = 50 
                        
                        if change_pct > 5: base_fgi += 20
                        elif change_pct > 2: base_fgi += 10
                        elif change_pct < -5: base_fgi -= 20
                        elif change_pct < -2: base_fgi -= 10

                        buyers_influence = (sentiment_cache["btc_sentiment"]["buyers"] - 50) * 0.5 
                        base_fgi += buyers_influence

                        sentiment_cache["fear_greed_index"] = max(0, min(100, int(base_fgi)))

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
                logger.warning(f"⚠️ Failed to fetch BTC ticker for sentiment: {e}")

        return success_count >= 2

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

            # Tenta sincronizar sentimento de notícias do bot real (que usa NewsAPI/CryptoPanic)
            if real_trading_bot and real_trading_bot.analyzer.sentiment_analyzer:
                try:
                    # O símbolo deve ser uma string como "BTC_USDT"
                    bot_news_sentiment_score = real_trading_bot.analyzer.sentiment_analyzer.get_crypto_news_sentiment("BTC_USDT")
                    
                    # Usa o sentimento do bot para BTC
                    sentiment_cache["btc_sentiment"]["buyers"] = bot_news_sentiment_score * 100
                    sentiment_cache["btc_sentiment"]["sellers"] = (1 - bot_news_sentiment_score) * 100
                    sentiment_cache["btc_sentiment"]["trend"] = "BULLISH" if bot_news_sentiment_score > 0.6 else "BEARISH" if bot_news_sentiment_score < 0.4 else "NEUTRAL"
                    sentiment_cache["btc_sentiment"]["last_update"] = current_time.isoformat()
                    logger.debug(f"Synced BTC news sentiment from bot: {bot_news_sentiment_score:.3f}")

                    # Ajusta FGI baseado no sentimento de notícias do bot, dando mais peso ao FGI da Gate.io se houver
                    if real_data_successfully_fetched:
                        # Se já pegou dados da Gate.io (Orderbook, Ticker), combina com o sentimento de notícias
                        combined_fgi_influence = (sentiment_cache["fear_greed_index"] + (bot_news_sentiment_score - 0.5) * 40) / 2
                        sentiment_cache["fear_greed_index"] = max(0, min(100, int(combined_fgi_influence)))
                    else: # Se não pegou dados da Gate.io, FGI é apenas do sentimento de notícias
                        sentiment_cache["fear_greed_index"] = max(0, min(100, int(bot_news_sentiment_score * 100)))

                except Exception as e:
                    logger.warning(f"Failed to sync BTC news sentiment from bot: {e}. Using dashboard's own sentiment calculation for overall mood.")
            else:
                 logger.warning("Trading bot sentiment analyzer is not available. Using dashboard's own sentiment calculation for overall mood.")


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

            if real_data_successfully_fetched or (real_trading_bot and real_trading_bot.analyzer.sentiment_analyzer): # Se qualquer fonte de dados real funcionou
                history_point = {
                    "timestamp": current_time.isoformat(),
                    "btc_buyers": round(sentiment_cache["btc_sentiment"]["buyers"], 2),
                    "btc_sellers": round(sentiment_cache["btc_sentiment"]["sellers"], 2),
                    "paxg_buyers": round(sentiment_cache["paxg_sentiment"]["buyers"], 2),
                    "paxg_sellers": round(sentiment_cache["paxg_sentiment"]["sellers"], 2),
                    "fear_greed": sentiment_cache["fear_greed_index"],
                    "volume_estimate": sentiment_cache["btc_sentiment"]["volume_24h"],
                    "market_mood": sentiment_cache["market_mood"],
                    "data_source": "REAL_GATEIO" if real_data_successfully_fetched else "REAL_NEWS_BOT"
                }
                sentiment_cache["sentiment_history"].append(history_point)
                
                if len(sentiment_cache["sentiment_history"]) > 144:
                    sentiment_cache["sentiment_history"] = sentiment_cache["sentiment_history"][-144:]

            sentiment_cache["websocket_connected"] = real_data_successfully_fetched # Ou adicione lógica para incluir o sentimento do bot aqui

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

            disconnected_clients = []
            for connection in list(active_sentiment_websocket_connections):
                try:
                    await connection.send_json(sentiment_data_for_ws)
                    logger.debug(f"Successfully sent sentiment data to client {connection.client}")
                except Exception as e:
                    logger.warning(f"Error sending sentiment data to WebSocket client {connection.client}: {e}", exc_info=True)
                    disconnected_clients.append(connection)

            for client in disconnected_clients:
                if client in active_sentiment_websocket_connections:
                    active_sentiment_websocket_connections.remove(client)

        except Exception as e:
            logger.error(f"❌ General error in sentiment monitoring main loop: {e}")

        await asyncio.sleep(10)

async def fetch_gateio_ohlcv(symbol_pair: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Busca dados OHLCV (candlesticks) da Gate.io para FUTUROS PERPÉTUOS usando API pública.
    CORRIGIDO para processar o formato JSON real da API.
    
    symbol_pair: Ex: "BTC_USDT" (contrato futuro na Gate.io)
    interval: Ex: "1m", "5m", "1h", "1d"
    limit: Número de candles a retornar (máximo 1000)
    
    Retorna: DataFrame pandas com colunas [Open, High, Low, Close, Volume] e index datetime
    """
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
                
                processed_data = []
                for candle in raw_data:
                    try:
                        processed_candle = {
                            'Open': float(candle['o']),
                            'High': float(candle['h']),
                            'Low': float(candle['l']),
                            'Close': float(candle['c']),
                            'Volume': float(candle['v']),
                            'timestamp': pd.to_datetime(int(candle['t']), unit='s')
                        }
                        processed_data.append(processed_candle)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Skipping invalid candle for {symbol_pair}: {e}")
                        continue
                
                if not processed_data:
                    logger.error(f"❌ No valid candles processed for {symbol_pair}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(processed_data)
                
                df.set_index('timestamp', inplace=True)
                
                df = df.sort_index()
                
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
            crossover_type = "none"
            
            if macd_previous <= signal_previous and macd_current > signal_current:
                crossover_type = "bullish"
            
            elif macd_previous >= signal_previous and macd_current < signal_current:
                crossover_type = "bearish"
            
            if crossover_type != "none":
                strength = abs(macd_current - signal_current)
                strength_normalized = min(strength * 10, 1.0)

                alert = {
                    "type": "MACD_CROSSOVER",
                    "asset": asset.upper(),
                    "crossover_type": crossover_type,
                    "strength": round(strength_normalized, 3),
                    "macd_value": round(macd_current, 4),
                    "signal_value": round(signal_current, 4),
                    "divergence": round(strength, 4),
                    "timestamp": datetime.now().isoformat(),
                    "title": f"🎯 MACD {crossover_type.upper()} Crossover - {asset.upper()}",
                    "message": f"MACD line crossed {'above' if crossover_type == 'bullish' else 'below'} Signal line for {asset.upper()}",
                    "severity": "HIGH" if strength_normalized > 0.5 else "MEDIUM",
                    "trading_signal": "BUY" if crossover_type == "bullish" else "SELL",
                    "confidence": round(strength_normalized * 100, 1)
                }
                
                self.crossover_history.append(alert)
                
                cache["alerts"].append(alert)
                
                logger.info(f"🎯 MACD {crossover_type.upper()} crossover detected for {asset}: strength {strength_normalized:.3f}")
                
                return {
                    "type": crossover_type,
                    "strength": strength_normalized,
                    "alert": alert
                }
            
            return {"type": "none", "strength": 0, "alert": None}
            
        except Exception as e:
            logger.error(f"Error in MACD crossover detection: {e}")
            return {"type": "none", "strength": 0, "alert": None}
    
    def get_recent_crossovers(self, limit: int = 10) -> List[Dict]:
        """Retorna cruzamentos recentes"""
        return list(self.crossover_history)[-limit:]

macd_detector = MACDCrossoverDetector()

# ===============================================================================
# 📊 PROCESSADOR DE DADOS TEMPO REAL (NOVO)
# ===============================================================================
def calculate_trend_angle_indicator(values: List[float], time_window: int = 5) -> Dict:
    """Calcula o ângulo de tendência para uma série de valores de indicador."""
    if len(values) < time_window:
        return {"angle": 0, "strength": 0}

    try:
        recent_values = values[-time_window:]
        x = np.arange(len(recent_values))
        
        # Garante que não há valores NaN, que causariam erro na regressão
        valid_indices = ~np.isnan(recent_values)
        if not np.any(valid_indices): # Se todos forem NaN
            return {"angle": 0, "strength": 0}
        
        recent_values_clean = np.array(recent_values)[valid_indices]
        x_clean = x[valid_indices]

        if len(x_clean) < 2: # Mínimo de 2 pontos para regressão
            return {"angle": 0, "strength": 0}

        slope, _, r_value, _, _ = stats.linregress(x_clean, recent_values_clean)
        angle = math.degrees(math.atan(slope))
        strength = r_value ** 2 # R-quadrado como força

        return {
            "angle": round(angle, 2),
            "strength": round(strength, 3)
        }
    except Exception as e:
        logger.warning(f"Erro ao calcular ângulo do indicador: {e}")
        return {"angle": 0, "strength": 0}

def update_realtime_macd(asset: str, new_candle: Dict):
    """
    Atualiza MACD, RSI, STOCHRSI e suas inclinações em tempo real com novo candle
    """
    try:
        # Garante que o asset existe no cache, inicializando se necessário
        if asset not in realtime_ohlcv_cache:
            logger.warning(f"Ativo {asset} não encontrado no cache OHLCV. Inicializando com defaults.")
            realtime_ohlcv_cache[asset] = {
                "candles": deque(maxlen=200),
                "current_candle": None,
                "macd_data": {"macd": [], "signal": [], "histogram": [], "last_crossover": None, "crossover_alerts": []},
                "rsi_data": {"rsi": [], "last_value": 0.0, "angle": 0.0, "strength": 0.0, "trend": "NEUTRAL"},
                "macd_angle_data": {"macd_angle": 0.0, "macd_angle_strength": 0.0, "signal_angle": 0.0, "signal_angle_strength": 0.0},
                "stochrsi_data": {"k_value": 0.0, "d_value": 0.0, "k_series": [], "d_series": []}, # NOVO
                "last_update": None,
                "websocket_connected": False
            }


        realtime_ohlcv_cache[asset]["candles"].append(new_candle)
        realtime_ohlcv_cache[asset]["current_candle"] = new_candle
        realtime_ohlcv_cache[asset]["last_update"] = datetime.now().isoformat()

        candles = list(realtime_ohlcv_cache[asset]["candles"])
        # Garante que os preços são numéricos e remove NaNs
        close_prices_raw = np.array([float(c["close"]) for c in candles])
        # Apenas use dados não-NaN para talib.
        close_prices = close_prices_raw[~np.isnan(close_prices_raw)].astype(np.float64)
        
        # --- Cálculo do MACD ---
        macd_clean = []
        signal_clean = []
        hist_clean = []
        
        if len(close_prices) >= 34: # MACD requires at least 34 periods
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices,
                fastperiod=12,
                slowperiod=26,  
                signalperiod=9
            )
            
            macd_clean = np.nan_to_num(macd).tolist()
            signal_clean = np.nan_to_num(macd_signal).tolist()
            hist_clean = np.nan_to_num(macd_hist).tolist()
            
            realtime_ohlcv_cache[asset]["macd_data"].update({
                "macd": macd_clean,
                "signal": signal_clean,
                "histogram": hist_clean,
                "last_update": datetime.now().isoformat()
            })
            
            if len(macd_clean) >= 2 and len(signal_clean) >= 2:
                crossover = macd_detector.detect_crossover(
                    asset=asset,
                    macd_current=macd_clean[-1],
                    signal_current=signal_clean[-1],  
                    macd_previous=macd_clean[-2],
                    signal_previous=signal_clean[-2]
                )
                if crossover["type"] != "none":
                    realtime_ohlcv_cache[asset]["macd_data"]["last_crossover"] = crossover
            
            # Calcular ângulos do MACD
            if len(macd_clean) >= 5 and len(signal_clean) >= 5: # Mínimo 5 pontos para calcular ângulo
                macd_angle_info = calculate_trend_angle_indicator(macd_clean)
                signal_angle_info = calculate_trend_angle_indicator(signal_clean)
                realtime_ohlcv_cache[asset]["macd_angle_data"].update({
                    "macd_angle": macd_angle_info["angle"],
                    "macd_angle_strength": macd_angle_info["strength"],
                    "signal_angle": signal_angle_info["angle"],
                    "signal_angle_strength": signal_angle_info["strength"],
                })
            else: # Se não há dados suficientes, resetar ângulos
                realtime_ohlcv_cache[asset]["macd_angle_data"] = {"macd_angle": 0, "macd_angle_strength": 0, "signal_angle": 0, "signal_angle_strength": 0}
            
        else: # Se não há candles suficientes para MACD, resetar tudo
            logger.debug(f"Insufficient candles for MACD for {asset} ({len(close_prices)}/34)")
            realtime_ohlcv_cache[asset]["macd_data"] = {"macd": [], "signal": [], "histogram": [], "last_crossover": None}
            realtime_ohlcv_cache[asset]["macd_angle_data"] = {"macd_angle": 0, "macd_angle_strength": 0, "signal_angle": 0, "signal_angle_strength": 0}


        # --- Cálculo do RSI ---
        rsi_clean = []
        if len(close_prices) >= 14: # RSI requires at least 14 periods
            rsi_values = talib.RSI(close_prices, timeperiod=14)
            rsi_clean = np.nan_to_num(rsi_values).tolist()
            last_rsi_value = rsi_clean[-1] if rsi_clean else 0.0

            rsi_angle_info = calculate_trend_angle_indicator(rsi_clean)
            
            rsi_trend = "NEUTRAL"
            if rsi_angle_info["angle"] > 10 and rsi_angle_info["strength"] > 0.4:
                rsi_trend = "RISING"
            elif rsi_angle_info["angle"] < -10 and rsi_angle_info["strength"] > 0.4:
                rsi_trend = "FALLING"

            realtime_ohlcv_cache[asset]["rsi_data"].update({
                "rsi": rsi_clean,
                "last_value": round(last_rsi_value, 2),
                "angle": rsi_angle_info["angle"],
                "strength": rsi_angle_info["strength"],
                "trend": rsi_trend,
                "last_update": datetime.now().isoformat()
            })
        else: # Se não há candles suficientes para RSI, resetar
            logger.debug(f"Insufficient candles for RSI for {asset} ({len(close_prices)}/14)")
            realtime_ohlcv_cache[asset]["rsi_data"] = {"rsi": [], "last_value": 0.0, "angle": 0, "strength": 0, "trend": "NEUTRAL"}
            
        # --- Cálculo do STOCHRSI (NOVO) ---
        stochrsi_k_clean = []
        stochrsi_d_clean = []
        # STOCHRSI (14,3,3) precisa de 14 para RSI, 3 para %K. Total 14 + (3-1) = 16 para o primeiro %K real.
        # Talib calcula internamente. Uma regra segura é len >= 14 + fastk_period + signal_period (se houver) ou 34 para valores MACD
        if len(close_prices) >= 17: # STOCHRSI (14,3,3) precisa de no mínimo 17 candles
            fastk, fastd = talib.STOCHRSI(
                close_prices,
                timeperiod=14,
                fastk_period=3,
                fastd_period=3,
                fastd_matype=0 # SMA
            )
            stochrsi_k_clean = np.nan_to_num(fastk).tolist()
            stochrsi_d_clean = np.nan_to_num(fastd).tolist()

            realtime_ohlcv_cache[asset]["stochrsi_data"].update({
                "k_value": round(stochrsi_k_clean[-1], 2) if stochrsi_k_clean else 0.0,
                "d_value": round(stochrsi_d_clean[-1], 2) if stochrsi_d_clean else 0.0,
                "k_series": stochrsi_k_clean,
                "d_series": stochrsi_d_clean,
                "last_update": datetime.now().isoformat()
            })
        else:
            logger.debug(f"Insufficient candles for STOCHRSI for {asset} ({len(close_prices)}/17)")
            realtime_ohlcv_cache[asset]["stochrsi_data"] = {"k_value": 0, "d_value": 0, "k_series": [], "d_series": []}

        logger.debug(f"📊 Indicators updated for {asset}: MACD ({len(macd_clean or [])} pts), RSI ({len(rsi_clean or [])} pts), STOCHRSI K/D ({len(stochrsi_k_clean or [])} pts)")
        
    except Exception as e:
        logger.error(f"Error updating realtime indicators for {asset}: {e}", exc_info=True)

# ===============================================================================
# 🔌 WEBSOCKET PARA DADOS OHLCV TEMPO REAL (NOVO)
# ===============================================================================

async def fetch_realtime_ohlcv_gateio():
    """
    Conecta ao WebSocket da Gate.io para dados OHLCV em tempo real
    para BTC_USDT e ETH_USDT.
    """
    logger.info("🔌 Starting Gate.io OHLCV WebSocket connection...")

    currency_pairs_to_subscribe = [s for s in TRADING_SYMBOLS if s.endswith('_USDT')]

    while True:
        websocket = None
        try:
            ws_url = "wss://api.gateio.ws/ws/v4/"
            
            websocket = await websockets.connect(ws_url, ping_interval=30, ping_timeout=10)
            logger.info("✅ Connected to Gate.io OHLCV WebSocket")
            
            for asset_pair in currency_pairs_to_subscribe:
                asset_key_init = asset_pair.split('_')[0].lower()
                if asset_key_init in realtime_ohlcv_cache:
                    realtime_ohlcv_cache[asset_key_init]["websocket_connected"] = True
                else:
                    realtime_ohlcv_cache[asset_key_init] = {
                        "candles": deque(maxlen=200),
                        "current_candle": None,
                        "macd_data": {"macd": [], "signal": [], "histogram": [], "last_crossover": None, "crossover_alerts": []},
                        "last_update": None,
                        "websocket_connected": True
                    }

            for pair in currency_pairs_to_subscribe:
                subscribe_message = {
                    "time": int(time.time()),
                    "channel": "spot.candlesticks",
                    "event": "subscribe",
                    "payload": ["1m", pair]
                }
                await websocket.send(json.dumps(subscribe_message))
                logger.info(f"📊 Subscribed to {pair} 1m candlesticks")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if (data.get("channel") == "spot.candlesticks" and 
                        data.get("event") == "update"):
                        
                        result = data.get("result", {})
                        
                        # CORREÇÃO: Usar 'n' em vez de 'name' (formato atual da API)
                        currency_pair_full_str = result.get("n")
                        
                        # Verificar se temos uma string válida para o par de moedas
                        if not currency_pair_full_str or not isinstance(currency_pair_full_str, str):
                            logger.warning(f"Invalid currency pair format: {result}")
                            continue
                            
                        timestamp_str = result.get("t")
                        open_str = result.get("o")
                        high_str = result.get("h")
                        low_str = result.get("l")
                        close_str = result.get("c")
                        volume_str = result.get("v")

                        # Validação inicial: garantir que as chaves críticas estão presentes e são strings
                        required_fields = [timestamp_str, open_str, high_str, low_str, close_str, volume_str]
                        if not all(isinstance(s, str) and s for s in required_fields):
                            logger.warning(f"Skipping incomplete candlestick update: {result}")
                            continue
                        
                        try:
                            # Tentar converter todas as strings para seus tipos numéricos
                            timestamp = int(timestamp_str)
                            open_price = float(open_str)
                            high_price = float(high_str)
                            low_price = float(low_str)
                            close_price = float(close_str)
                            volume = float(volume_str)
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Skipping invalid candlestick values: {e} - Data: {result}")
                            continue
                        
                        # Extrair o símbolo do ativo (e.g., 'btc' de '1m_BTC_USDT')
                        parts = currency_pair_full_str.split('_')
                        if len(parts) < 2:
                            logger.warning(f"Skipping invalid currency pair format: {currency_pair_full_str}")
                            continue
                        
                        # Determinar se o formato é "intervalo_BASE_QUOTE" ou "BASE_QUOTE"
                        if parts[0].endswith('m') or parts[0].endswith('h') or parts[0].endswith('d'):
                            asset_key = parts[1].lower()
                        else:
                            asset_key = parts[0].lower()

                        # Verificar se o asset_key é um dos que estamos monitorando
                        valid_assets = [s.split('_')[0].lower() for s in TRADING_SYMBOLS]
                        if asset_key not in valid_assets:
                            logger.debug(f"Skipping unsubscribed asset: {asset_key} (from {currency_pair_full_str})")
                            continue
                        
                        new_candle = {
                            "timestamp": timestamp,
                            "open": open_price, 
                            "high": high_price, 
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                            "datetime": datetime.fromtimestamp(timestamp).isoformat()
                        }
                        
                        update_realtime_macd(asset_key, new_candle)
                        
                        broadcast_data = {
                        "type": "ohlcv_update",
                        "asset": asset_key.upper(),
                        "candle": new_candle,
                        "macd_data": realtime_ohlcv_cache[asset_key]["macd_data"],
                        "rsi_data": realtime_ohlcv_cache[asset_key]["rsi_data"], # NOVO
                        "macd_angle_data": realtime_ohlcv_cache[asset_key]["macd_angle_data"], # NOVO
                        "stochrsi_data": realtime_ohlcv_cache[asset_key]["stochrsi_data"], # NOVO
                        "timestamp": datetime.now().isoformat()
                    }
                    
                        await broadcast_ohlcv_update(broadcast_data)
                        
                        logger.debug(f"📊 New {asset_key.upper()} candle: ${new_candle['close']:,.2f}, Vol: {new_candle['volume']:,.0f}")
                        
                except Exception as e:
                    logger.warning(f"Error processing WebSocket message: {e}", exc_info=True)
                    continue
                
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Gate.io OHLCV WebSocket connection closed normally.")
        except Exception as e:
            logger.error(f"Gate.io OHLCV WebSocket error: {e}", exc_info=True)
        finally:
            for asset_pair in currency_pairs_to_subscribe:
                asset_key_final = asset_pair.split('_')[0].lower()
                if asset_key_final in realtime_ohlcv_cache:
                    realtime_ohlcv_cache[asset_key_final]["websocket_connected"] = False
            
            if websocket:
                await websocket.close()
            
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
            logger.debug(f"Successfully sent OHLCV data to client {connection.client}")
        except Exception as e:
            logger.warning(f"Error sending to OHLCV WebSocket client: {e}", exc_info=True)
            disconnected_clients.append(connection)
    
    for client in disconnected_clients:
        if client in active_ohlcv_websocket_connections:
            active_ohlcv_websocket_connections.remove(client)

# ===============================================================================
# 📊 ENDPOINTS PARA DADOS MACD TEMPO REAL (NOVO)
# ===============================================================================

@app.get("/api/macd/realtime/{asset}")
def get_realtime_macd(asset: str):
    """Endpoint para MACD em tempo real, com fallback graceful se dados não estiverem prontos."""
    try:
        asset_lower = asset.lower()
        
        # Se o ativo não está no cache ou não tem candles, retorne um estado de 'warming up'
        if asset_lower not in realtime_ohlcv_cache or not realtime_ohlcv_cache[asset_lower]["candles"]:
            logger.warning(f"⚠️ Realtime MACD for {asset}: Cache is empty or warming up. Returning default values.")
            return {
                "asset": asset.upper(),
                "macd_data": {"macd": [], "signal": [], "histogram": [], "last_crossover": None, "crossover_alerts": []}, # Estrutura vazia
                "rsi_data": {"rsi": [], "last_value": 0.0, "angle": 0.0, "strength": 0.0, "trend": "NEUTRAL"},
                "macd_angle_data": {"macd_angle": 0.0, "macd_angle_strength": 0.0, "signal_angle": 0.0, "signal_angle_strength": 0.0},
                "stochrsi_data": {"k_value": 0.0, "d_value": 0.0, "k_series": [], "d_series": []},
                "current_candle": {},
                "websocket_connected": realtime_ohlcv_cache.get(asset_lower, {}).get("websocket_connected", False),
                "last_update": None,
                "candles_count": 0,
                "recent_crossovers": [],
                "timestamp": datetime.now().isoformat(),
                "status": "warming_up",
                "message": f"Real-time data for {asset} not yet available. Please wait."
            }
        
        asset_data = realtime_ohlcv_cache[asset_lower]
        
        current_candle_formatted = {}
        if asset_data["current_candle"]:
            current_candle_formatted = {
                "open": round(asset_data["current_candle"]["open"], 2),
                "high": round(asset_data["current_candle"]["high"], 2),
                "low": round(asset_data["current_candle"]["low"], 2),
                "close": round(asset_data["current_candle"]["close"], 2),
                "volume": int(asset_data["current_candle"]["volume"]),
                "datetime": asset_data["current_candle"]["datetime"]
            }

        return {
            "asset": asset.upper(),
            "macd_data": asset_data["macd_data"],
            "rsi_data": asset_data["rsi_data"],
            "macd_angle_data": asset_data["macd_angle_data"],
            "stochrsi_data": asset_data.get("stochrsi_data", {"k_value": 0, "d_value": 0}),
            "current_candle": current_candle_formatted,
            "websocket_connected": asset_data["websocket_connected"],
            "last_update": asset_data["last_update"],
            "candles_count": len(asset_data["candles"]),
            "recent_crossovers": macd_detector.get_recent_crossovers(5),
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting realtime MACD for {asset}: {e}", exc_info=True)
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
        logger.error(f"Error getting crossovers: {e}", exc_info=True)
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
            "💰 Trade Signals (Backtest Dashboard)": {
                "/api/backtest-recommendations": "Top N backtest recommendations (Dashboard's internal backtest)"
            },
            "🤖 Trading Bot (Real)": {
                "/api/trading-bot/status": "Current status of the REAL trading bot",
                "/api/trading-bot/positions": "List of active REAL positions",
                "/api/trading-bot/signals": "Recently detected REAL trade signals",
                "/api/trading-bot/performance": "Performance metrics of the REAL bot",
                "/api/trading-bot/start": "Start the REAL trading bot",
                "/api/trading-bot/stop": "Stop the REAL trading bot",
                "/api/trading-bot/refresh-balance": "Force refresh bot balance",
                "/api/trading-bot/diagnostics": "Full diagnostics of the bot's connectivity"
            },
            "⚙️ System": {
                "/api/status": "API status and health check",
                "/api/debug/gateio": "Gate.io API connectivity test",
                "/api/debug/test-data-fetch": "Test data fetching from all sources",
                "/docs": "API documentation",
            },
            "📈 Real-time MACD": {
                "/ws/ohlcv": "Real-time OHLCV and MACD WebSocket stream for BTC/ETH",
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

        fred_info = {
            "today_events": 0,
            "today_high_impact": 0,
            "next_critical_event": None,
            "total_upcoming": 0,
            "pre_alerts_active": 0
        }
        if real_trading_bot and real_trading_bot.analyzer.fred_calendar:
            bot_fred_calendar_instance = real_trading_bot.analyzer.fred_calendar
            if real_trading_bot.environment == 'simulate_backtest' and hasattr(bot_fred_calendar_instance, 'datetime_now_override'):
                bot_fred_calendar_instance.datetime_now_override = datetime.now()
            
            # Chama para garantir que o cache seja atualizado (get_upcoming_releases já atualiza internamente)
            bot_fred_calendar_instance.get_upcoming_releases(days_ahead=14)

            upcoming_events_from_bot = bot_fred_calendar_instance.cache["upcoming_events"]
            pre_event_alerts_from_bot = bot_fred_calendar_instance.generate_pre_event_alerts(hours_before=48)
            next_critical_event_from_bot = bot_fred_calendar_instance.get_next_critical_event()
            today_events_from_bot = bot_fred_calendar_instance.get_high_impact_events_today()

            fred_info = {
                "today_events": len(today_events_from_bot),
                "today_high_impact": len([e for e in today_events_from_bot if e.importance == "HIGH"]),
                "next_critical_event": {
                    "name": next_critical_event_from_bot.name,
                    "date": next_critical_event_from_bot.date.isoformat(),
                    "impact_score": next_critical_event_from_bot.impact_score
                } if next_critical_event_from_bot else None,
                "total_upcoming": len(upcoming_events_from_bot),
                "pre_alerts_active": len(pre_event_alerts_from_bot)
            }


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

        realtime_macd_info = {
            "btc_websocket_connected": realtime_ohlcv_cache["btc"]["websocket_connected"],
            "btc_last_update": realtime_ohlcv_cache["btc"]["last_update"],
            "btc_crossovers_count": len(macd_detector.crossover_history)
        }
        if 'eth' in realtime_ohlcv_cache:
            realtime_macd_info["eth_websocket_connected"] = realtime_ohlcv_cache["eth"]["websocket_connected"]
            realtime_macd_info["eth_last_update"] = realtime_ohlcv_cache["eth"]["last_update"]
            realtime_macd_info["eth_crossovers_count"] = len([c for c in macd_detector.crossover_history if c['asset'] == 'ETH'])


        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "market_status": "Real-time data available",
            "data_quality": {
                "price_data_valid": all(current_data[asset].get("current_price", 0) > 0 for asset in SYMBOLS),
                "angular_analysis_active": len(cache["angular_data"]) > 0,
                "fred_calendar_active": fred_info["total_upcoming"] > 0,
                "sentiment_monitoring_active": sentiment_cache["websocket_connected"],
                "realtime_ohlcv_active": realtime_ohlcv_cache["btc"]["websocket_connected"] or ('eth' in realtime_ohlcv_cache and realtime_ohlcv_cache["eth"]["websocket_connected"])
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
                "macd_crossovers": len([a for a in cache["alerts"] if a.get("type", "") == "MACD_CROSSOVER"]),
                "price_history_points": len(cache["price_history"]),
                "angular_history_points": len(cache["angular_data"]),
                "last_angular_analysis": cache["last_angular_analysis"],
                "economic_calendar": fred_info,
                "market_sentiment": sentiment_info,
                "realtime_macd_status": realtime_macd_info
            },
            "system_health": {
                "cache_size": len(cache["price_history"]),
                "uptime_status": "healthy",
                "last_data_refresh": datetime.now().isoformat(),
                "api_endpoints_active": 13,
                "websocket_status": {
                    "sentiment": "connected" if sentiment_cache["websocket_connected"] else "disconnected_or_failed_api",
                    "ohlcv": "connected" if (realtime_ohlcv_cache["btc"]["websocket_connected"] or ('eth' in realtime_ohlcv_cache and realtime_ohlcv_cache["eth"]["websocket_connected"])) else "disconnected_or_failed_api"
                }
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting current data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting current data: {str(e)}")

@app.get("/api/debug/system-status")
def get_debug_system_status():
    """Endpoint de debug para verificar status completo do sistema"""
    global real_trading_bot
    
    try:
        current_time = datetime.now()
        
        debug_info = {
            "timestamp": current_time.isoformat(),
            "system_overview": {
                "api_version": SYSTEM_INFO['version'],
                "environment": ENVIRONMENT.upper() if ENVIRONMENT else "NOT_SET",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            },
            "trading_bot_status": {
                "initialized": real_trading_bot is not None,
                "running": real_trading_bot.running if real_trading_bot else False,
                "environment": real_trading_bot.environment.upper() if real_trading_bot else "N/A",
                "balance_check": None,
                "error": None
            },
            "api_credentials": {
                "api_key_configured": bool(API_KEY),
                "api_key_length": len(API_KEY) if API_KEY else 0,
                "api_key_preview": f"{API_KEY[:8]}..." if API_KEY and len(API_KEY) > 8 else "NOT_SET",
                "secret_configured": bool(SECRET),
                "secret_length": len(SECRET) if SECRET else 0,
                "environment_variable": ENVIRONMENT,
                "newsapi_key_configured": bool(TRADING_BOT_NEWS_API_KEY and TRADING_BOT_NEWS_API_KEY != "DUMMY_KEY_NEWSAPI"),
                "fred_api_key_configured": bool(TRADING_BOT_FRED_API_KEY and TRADING_BOT_FRED_API_KEY != "DUMMY_KEY_FRED"),
                "cryptopanic_api_key_configured": bool(TRADING_BOT_CRYPTOPANIC_API_KEY and TRADING_BOT_CRYPTOPANIC_API_KEY != "DUMMY_KEY_CRYPTOPANIC")
            },
            "data_sources": {
                "yfinance_status": "available",
                "gateio_sentiment_ws_status": {
                    "connected": sentiment_cache["websocket_connected"],
                    "last_update": sentiment_cache["btc_sentiment"]["last_update"],
                    "btc_buyers": sentiment_cache["btc_sentiment"]["buyers"],
                    "data_age_minutes": None
                },
                "gateio_ohlcv_ws_status": {
                    "connected": realtime_ohlcv_cache["btc"]["websocket_connected"],
                    "last_update": realtime_ohlcv_cache["btc"]["last_update"],
                    "candles_count": len(realtime_ohlcv_cache["btc"]["candles"])
                },
                "bot_internal_fred_calendar": {
                    "active": real_trading_bot and real_trading_bot.analyzer.fred_calendar.api_key is not None,
                    "last_update": real_trading_bot.analyzer.last_fred_update.isoformat() if real_trading_bot and real_trading_bot.analyzer.last_fred_update else None,
                    "events_cached": len(real_trading_bot.analyzer.fred_events_cache) if real_trading_bot else 0
                },
                "bot_internal_sentiment_analysis": {
                    "active": real_trading_bot and real_trading_bot.analyzer.sentiment_analyzer.cryptopanic_api_key is not None,
                    "last_update": real_trading_bot.analyzer.sentiment_analyzer.last_update.get('BTC_USDT_sentiment_multi_source', None) if real_trading_bot else None,
                    "cache_size": len(real_trading_bot.analyzer.sentiment_analyzer.sentiment_cache) if real_trading_bot else 0
                }
            },
            "cache_status": {
                "price_history_points": len(cache["price_history"]),
                "angular_data_points": len(cache["angular_data"]),
                "total_alerts": len(cache["alerts"]),
                "fred_events_dashboard_cache": len(cache["fred_data"]["upcoming_events"]),
                "backtest_recommendations": len(backtest_recommendations_cache["recommendations"])
            },
            "websocket_connections": {
                "sentiment_clients": len(active_sentiment_websocket_connections),
                "ohlcv_clients": len(active_ohlcv_websocket_connections),
                "rsi_macd_clients": len(active_rsi_macd_websocket_connections)
            }
        }
        
        if sentiment_cache["btc_sentiment"]["last_update"]:
            try:
                last_update = datetime.fromisoformat(sentiment_cache["btc_sentiment"]["last_update"])
                age_seconds = (current_time - last_update).total_seconds()
                debug_info["data_sources"]["gateio_sentiment_ws_status"]["data_age_minutes"] = round(age_seconds / 60, 1)
            except:
                pass
        
        if real_trading_bot:
            try:
                balance = real_trading_bot.get_account_balance()
                debug_info["trading_bot_status"]["balance_check"] = {
                    "status": "success",
                    "balance": balance,
                    "message": f"Balance: ${balance}"
                }
            except Exception as e:
                debug_info["trading_bot_status"]["balance_check"] = {
                    "status": "failed",
                    "error": str(e)
                }
                debug_info["trading_bot_status"]["error"] = str(e)
        
        issues = []
        if not real_trading_bot:
            issues.append("Trading bot not initialized")
        if not API_KEY or not SECRET:
            issues.append("Gate.io API credentials missing")
        if not sentiment_cache["websocket_connected"]:
            issues.append("Gate.io sentiment API disconnected (Dashboard's direct fetch)")
        if not realtime_ohlcv_cache["btc"]["websocket_connected"]:
            issues.append("Gate.io OHLCV WebSocket disconnected")
        if TRADING_BOT_FRED_API_KEY == "DUMMY_KEY_FRED" or (real_trading_bot and real_trading_bot.analyzer.fred_calendar.api_key == "DUMMY_KEY_FRED"):
            issues.append("FRED API Key is missing or invalid for the Trading Bot")
        if TRADING_BOT_NEWS_API_KEY == "DUMMY_KEY_NEWSAPI" or (real_trading_bot and real_trading_bot.analyzer.sentiment_analyzer.news_api_key == "DUMMY_KEY_NEWSAPI"):
            issues.append("NEWS_API_KEY is missing for the Trading Bot's sentiment analysis")
        if TRADING_BOT_CRYPTOPANIC_API_KEY == "DUMMY_KEY_CRYPTOPANIC" or (real_trading_bot and real_trading_bot.analyzer.sentiment_analyzer.cryptopanic_api_key == "DUMMY_KEY_CRYPTOPANIC"):
            issues.append("CRYPTOPANIC_API_KEY is missing for the Trading Bot's sentiment analysis")

        debug_info["overall_status"] = {
            "healthy": len(issues) == 0,
            "issues_count": len(issues),
            "issues": issues,
            "recommendations": []
        }
        
        if not API_KEY or not SECRET:
            debug_info["overall_status"]["recommendations"].append("Configure GATE_TESTNET_API_KEY and GATE_TESTNET_API_SECRET in .env file")
        
        if not real_trading_bot:
            debug_info["overall_status"]["recommendations"].append("Restart application after configuring all API credentials")
        
        if not sentiment_cache["websocket_connected"]:
            debug_info["overall_status"]["recommendations"].append("Check Gate.io API connectivity for Dashboard's sentiment fetch")
        
        return debug_info
        
    except Exception as e:
        logger.error(f"❌ Error in debug system status: {e}", exc_info=True)
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "debug_endpoint_failed"
        }


@app.get("/api/debug/test-data-fetch")
def test_data_fetch():
    """Testa busca de dados de diferentes fontes"""
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    try:
        gold_data = yf.download('GC=F', period='5d', interval='1d', progress=False, auto_adjust=True)
        test_results["tests"]["yfinance_gold"] = {
            "status": "success" if not gold_data.empty else "empty_data",
            "data_points": len(gold_data) if not gold_data.empty else 0,
            "columns": list(gold_data.columns) if not gold_data.empty else []
        }
    except Exception as e:
        test_results["tests"]["yfinance_gold"] = {
            "status": "failed",
            "error": str(e)
        }
    
    try:
        btc_data = yf.download('BTC-USD', period='5d', interval='1d', progress=False, auto_adjust=True)
        test_results["tests"]["yfinance_btc"] = {
            "status": "success" if not btc_data.empty else "empty_data",
            "data_points": len(btc_data) if not btc_data.empty else 0,
            "columns": list(btc_data.columns) if not btc_data.empty else []
        }
    except Exception as e:
        test_results["tests"]["yfinance_btc"] = {
            "status": "failed",
            "error": str(e)
        }
    
    try:
        dxy_data = yf.download('DX-Y.NYB', period='5d', interval='1d', progress=False, auto_adjust=True)
        test_results["tests"]["yfinance_dxy"] = {
            "status": "success" if not dxy_data.empty else "empty_data",
            "data_points": len(dxy_data) if not dxy_data.empty else 0,
            "columns": list(dxy_data.columns) if not dxy_data.empty else []
        }
    except Exception as e:
        test_results["tests"]["yfinance_dxy"] = {
            "status": "failed",
            "error": str(e)
        }
    
    test_results["tests"]["gateio_sentiment_dashboard_fetch"] = {
        "status": "connected" if sentiment_cache["websocket_connected"] else "disconnected",
        "btc_buyers": sentiment_cache["btc_sentiment"]["buyers"],
        "last_update": sentiment_cache["btc_sentiment"]["last_update"]
    }

    test_results["tests"]["gateio_ohlcv_dashboard_fetch"] = {
        "status": "connected" if realtime_ohlcv_cache["btc"]["websocket_connected"] else "disconnected",
        "btc_candles_count": len(realtime_ohlcv_cache["btc"]["candles"]),
        "last_update": realtime_ohlcv_cache["btc"]["last_update"]
    }
    
    global real_trading_bot
    if real_trading_bot:
        try:
            balance = real_trading_bot.get_account_balance()
            test_results["tests"]["trading_bot_main_api"] = {
                "status": "working",
                "balance": balance,
                "environment": real_trading_bot.environment
            }
        except Exception as e:
            test_results["tests"]["trading_bot_main_api"] = {
                "status": "error",
                "error": str(e)
            }
        
        try:
            fred_events_count = len(real_trading_bot.analyzer.fred_events_cache)
            test_results["tests"]["bot_internal_fred"] = {
                "status": "active" if fred_events_count > 0 else "inactive",
                "events_count": fred_events_count,
                "api_key_configured": real_trading_bot.analyzer.fred_calendar.api_key is not None
            }
        except Exception as e:
            test_results["tests"]["bot_internal_fred"] = {
                "status": "error",
                "error": str(e)
            }

        try:
            bot_sentiment_cache_size = len(real_trading_bot.analyzer.sentiment_analyzer.sentiment_cache)
            test_results["tests"]["bot_internal_sentiment_api"] = {
                "status": "active" if bot_sentiment_cache_size > 0 else "inactive",
                "cache_size": bot_sentiment_cache_size,
                "newsapi_key_configured": TRADING_BOT_NEWS_API_KEY != "DUMMY_KEY_NEWSAPI",
                "cryptopanic_key_configured": TRADING_BOT_CRYPTOPANIC_API_KEY != "DUMMY_KEY_CRYPTOPANIC"
            }
        except Exception as e:
            test_results["tests"]["bot_internal_sentiment_api"] = {
                "status": "error",
                "error": str(e)
            }

    else:
        test_results["tests"]["trading_bot_main_api"] = {
            "status": "not_initialized"
        }
        test_results["tests"]["bot_internal_fred"] = {
            "status": "not_initialized"
        }
        test_results["tests"]["bot_internal_sentiment_api"] = {
            "status": "not_initialized"
        }
    
    return test_results

@app.get("/api/precos/{period}")
async def get_financial_data_by_period(period: str):
    """Endpoint para dados históricos, incluindo MACD, RSI, STOCHRSI e Volume colorido - CORRIGIDO"""
    logger.info(f"📊 Fetching financial data for period: {period}")
    
    # Mapeamento de períodos e intervalos para YFinance e Gate.io
    # Aumentado 'period' para '5m' para buscar mais dados (1h a 7d)
    yfinance_period_map = {
        '1m': '7d',   # 1 minuto de intervalo, 7 dias de período (máximo permitido pelo YFinance para 1m)
        '5m': '7d',   # 5 minutos de intervalo, 7 dias de período (para cobrir mais de 1h)
        '15m': '60d',
        '30m': '60d',
        '1h': '730d', # 2 anos
        '4h': '730d',
        '1d': '1y',
        '5d': '5d',
        '1mo': '1mo',
        '3mo': '3mo',
        '6mo': '1y',
        '1y': '2y',
        '2y': '5y',
        '5y': '10y',
        'max': 'max',
    }
    
    yfinance_interval_map = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h', # Usar 1h para 1h
        '4h': '4h',
        '1d': '1d',
        '5d': '1d',
        '1mo': '1d',
        '3mo': '1d',
        '6mo': '1d',
        '1y': '1d',
        '2y': '1d',
        '5y': '1d',
        'max': '1d',
    }
    
    gateio_interval_map = {
        '1m': '1m',
        '5m': '5m', # Usar 5m para 5 dias
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '5d': '5m',
        '1mo': '1h',
        '3mo': '4h',
        '6mo': '1d',
        '1y': '1d',
        '2y': '1d',
        '5y': '1d',
        'max': '1d',
    }
    
    gateio_limit_map = {
        '1m': 1000,
        '5m': 1000, # Aumentado para 1000 para capturar mais de 1h
        '15m': 1000,
        '30m': 1000,
        '1h': 1000,
        '4h': 1000,
        '1d': 1000,
        '5d': 1000,
        '1mo': 720,
        '3mo': 500,
        '6mo': 400,
        '1y': 365,
        '2y': 730,
        '5y': 1000,
        'max': 1000,
    }

    try:
        data = {}
        
        all_symbols_to_fetch = list(SYMBOLS.keys()) + [s.split('_')[0].lower() for s in TRADING_SYMBOLS if s.split('_')[0].lower() not in SYMBOLS.keys()]

        for asset_name in all_symbols_to_fetch:
            symbol_yf = SYMBOLS.get(asset_name)
            symbol_gateio = f"{asset_name.upper()}_USDT" if asset_name.upper() in [s.split('_')[0].upper() for s in TRADING_SYMBOLS] else None

            prices = []
            volumes = []
            volume_colors = []
            asset_dates = []
            macd_values = []
            macd_signal_values = []
            macd_hist_values = []
            rsi_values = [] 
            stochrsi_k_values = [] 
            stochrsi_d_values = [] 
            opens = []
            highs = []
            lows = []

            try:
                hist = pd.DataFrame()
                
                # Definir start_date para yfinance para garantir dados suficientes
                end_date_yf = datetime.now(pytz.utc) # Usar timezone-aware datetime para consistência
                
                start_date_yf_base = end_date_yf - timedelta(days=7) 
                if period == '1d': start_date_yf_base = end_date_yf - timedelta(days=1)
                elif period == '5d': start_date_yf_base = end_date_yf - timedelta(days=5)
                elif period == '1mo': start_date_yf_base = end_date_yf - timedelta(days=30)
                elif period == '3mo': start_date_yf_base = end_date_yf - timedelta(days=90)
                elif period == '6mo': start_date_yf_base = end_date_yf - timedelta(days=180)
                elif period == '1y': start_date_yf_base = end_date_yf - timedelta(days=365)
                elif period == '2y': start_date_yf_base = end_date_yf - timedelta(days=730)
                elif period == '5y': start_date_yf_base = end_date_yf - timedelta(days=1825)
                
                yf_interval = yfinance_interval_map.get(period, '1d')
                yf_period_arg = yfinance_period_map.get(period, '1d') 
                
                if symbol_gateio:
                    gateio_interval = gateio_interval_map.get(period, '1m')
                    limit = gateio_limit_map.get(period, 1000) 
                    
                    logger.info(f"DEBUG: Fetching {asset_name.upper()} from Gate.io Futures (Contract: {symbol_gateio}, Interval: {gateio_interval}, Limit: {limit})")
                    hist = await fetch_gateio_ohlcv(symbol_gateio, gateio_interval, limit)
                    
                    if hist.empty:
                        logger.warning(f"⚠️ No Gate.io data received for {asset_name.upper()}, falling back to yfinance (if available)")
                        if symbol_yf:
                            try:
                                logger.info(f"DEBUG: Fallback {asset_name.upper()} from yfinance (Symbol: {symbol_yf}, Period: {yf_period_arg}, Interval: {yf_interval})")
                                if period in ['1m', '5m', '15m', '30m', '1h', '4h']: 
                                     hist = yf.download(symbol_yf, period=yf_period_arg, interval=yf_interval, progress=False, threads=False, auto_adjust=True)
                                else:
                                     hist = yf.download(symbol_yf, start=start_date_yf_base, end=end_date_yf, interval=yf_interval, progress=False, threads=False, auto_adjust=True)
                            except Exception as fallback_error:
                                logger.error(f"❌ Fallback yfinance also failed for {asset_name.upper()}: {fallback_error}")
                elif symbol_yf:
                    logger.info(f"DEBUG: Fetching {asset_name} from yfinance (Symbol: {symbol_yf}, Period: {yf_period_arg}, Interval: {yf_interval})")
                    try:
                        if period in ['1m', '5m', '15m', '30m', '1h', '4h']:
                             hist = yf.download(symbol_yf, period=yf_period_arg, interval=yf_interval, progress=False, threads=False, auto_adjust=True)
                        else:
                             hist = yf.download(symbol_yf, start=start_date_yf_base, end=end_date_yf, interval=yf_interval, progress=False, threads=False, auto_adjust=True)

                    except Exception as yf_error:
                        logger.error(f"❌ yfinance error for {symbol_yf}: {yf_error}")
                        try:
                            logger.info(f"DEBUG: Retry {asset_name} with conservative parameters")
                            hist = yf.download(symbol_yf, period='1mo', interval='1d', progress=False, threads=False, auto_adjust=True)
                        except Exception as retry_error:
                            logger.error(f"❌ Conservative retry also failed for {symbol_yf}: {retry_error}")
                    
                    if hist.empty:
                        logger.warning(f"⚠️ No yfinance data received for {symbol_yf}, or DataFrame is empty after download.")
                        continue
                else:
                    logger.warning(f"No valid symbol mapping found for {asset_name}. Skipping.")
                    continue

                required_ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if (not hist.empty and 
                    all(col in hist.columns for col in required_ohlcv_cols)):
                    
                    for col in required_ohlcv_cols:
                        hist[col] = pd.to_numeric(hist[col], errors='coerce')
                        if col in ['Open', 'High', 'Low', 'Close']:
                            hist[col] = hist[col].ffill().bfill() 
                        elif col == 'Volume':
                            hist[col] = hist[col].fillna(0) 

                    hist_cleaned = hist.dropna(subset=['Close']) 

                    if hist_cleaned.empty:
                        logger.warning(f"DEBUG: {asset_name} - Cleaned history is empty after NaN filling and dropping. Cannot calculate indicators.")
                        prices, volumes, opens, highs, lows, volume_colors, asset_dates = [],[],[],[],[],[],[]
                        macd_values, macd_signal_values, macd_hist_values, rsi_values = [],[],[],[]
                        stochrsi_k_values, stochrsi_d_values = [],[]
                        continue

                    hist_cleaned = calculate_volume_colors_and_macd(hist_cleaned)
                    
                    if hasattr(hist_cleaned.index, 'tz_localize'):
                        hist_cleaned.index = hist_cleaned.index.tz_localize(None) 
                    
                    prices = hist_cleaned['Close'].tolist()
                    volumes = hist_cleaned['Volume'].tolist()
                    opens = hist_cleaned['Open'].tolist()
                    highs = hist_cleaned['High'].tolist()
                    lows = hist_cleaned['Low'].tolist()
                    volume_colors = hist_cleaned.get('price_direction', ['neutral'] * len(prices)).tolist()
                                    
                    asset_dates = hist_cleaned.index.strftime('%Y-%m-%dT%H:%M:%S').tolist()
                                    
                    # --- MACD ---
                    if len(hist_cleaned) >= 34: 
                        try:
                            macd, macdsignal, macdhist = talib.MACD(
                                hist_cleaned['Close'].values.astype(np.float64),
                                fastperiod=12, 
                                slowperiod=26, 
                                signalperiod=9
                            )
                            macd_values = np.nan_to_num(macd).tolist()
                            macd_signal_values = np.nan_to_num(macdsignal).tolist()
                            macd_hist_values = np.nan_to_num(macdhist).tolist()
                            logger.info(f"DEBUG: {asset_name} - MACD calculated. Length: {len(macd_values)}")
                        except Exception as macd_error:
                            logger.warning(f"MACD calculation failed for {asset_name}: {macd_error}")
                    else:
                        logger.warning(f"DEBUG: {asset_name} - NOT enough data ({len(hist_cleaned)}) for MACD calculation.")

                    # --- RSI ---
                    if len(hist_cleaned) >= 14:
                        try:
                            rsi_values_calc = talib.RSI(hist_cleaned['Close'].values.astype(np.float64), timeperiod=14)
                            rsi_values = np.nan_to_num(rsi_values_calc).tolist()
                            logger.info(f"DEBUG: {asset_name} - RSI calculated. Length: {len(rsi_values)}")
                        except Exception as rsi_error_calc:
                            logger.error(f"❌ Error calculating RSI for {asset_name}: {rsi_error_calc}", exc_info=True)
                    else:
                        logger.warning(f"DEBUG: {asset_name} - Insufficient data ({len(hist_cleaned)}) for RSI calculation (min 14).")

                    # --- STOCHRSI ---
                    if len(hist_cleaned) >= 17:
                        try:
                            fastk, fastd = talib.STOCHRSI(
                                hist_cleaned['Close'].values.astype(np.float64),
                                timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0
                            )
                            stochrsi_k_values = np.nan_to_num(fastk).tolist()
                            stochrsi_d_values = np.nan_to_num(fastd).tolist()
                            logger.info(f"DEBUG: {asset_name} - STOCHRSI calculated. Length: {len(stochrsi_k_values)}")
                        except Exception as stochrsi_error_calc:
                            logger.error(f"❌ Error calculating STOCHRSI for {asset_name}: {stochrsi_error_calc}", exc_info=True)
                    else:
                        logger.warning(f"DEBUG: {asset_name} - Insufficient data ({len(hist_cleaned)}) for STOCHRSI calculation (min 17).")
                    
                    # Alinhar todas as séries ao menor comprimento para evitar erros no frontend
                    all_series = [
                        prices, volumes, opens, highs, lows, volume_colors, asset_dates,
                        macd_values, macd_signal_values, macd_hist_values,
                        rsi_values, stochrsi_k_values, stochrsi_d_values
                    ]
                    
                    non_empty_series = [s for s in all_series if s]
                    
                    min_len_data_all = min(len(s) for s in non_empty_series) if non_empty_series else 0
                    
                    if min_len_data_all > 0:
                        prices = prices[-min_len_data_all:]
                        volumes = volumes[-min_len_data_all:]
                        opens = opens[-min_len_data_all:]
                        highs = highs[-min_len_data_all:]
                        lows = lows[-min_len_data_all:]
                        volume_colors = volume_colors[-min_len_data_all:]
                        asset_dates = asset_dates[-min_len_data_all:]
                        macd_values = macd_values[-min_len_data_all:]
                        macd_signal_values = macd_signal_values[-min_len_data_all:]
                        macd_hist_values = macd_hist_values[-min_len_data_all:]
                        rsi_values = rsi_values[-min_len_data_all:]
                        stochrsi_k_values = stochrsi_k_values[-min_len_data_all:]
                        stochrsi_d_values = stochrsi_d_values[-min_len_data_all:]
                        logger.info(f"DEBUG: {asset_name} - All data aligned to {min_len_data_all} points.")
                    else:
                        logger.warning(f"DEBUG: {asset_name} - No valid aligned data points after indicator calculation.")
                        # Resetar tudo para vazio se não houver dados alinhados
                        prices, volumes, opens, highs, lows, volume_colors, asset_dates = [],[],[],[],[],[],[]
                        macd_values, macd_signal_values, macd_hist_values, rsi_values = [],[],[],[]
                        stochrsi_k_values, stochrsi_d_values = [],[]

                else:
                    logger.warning(f"DEBUG: {asset_name} - No required OHLCV columns found, or history is empty after cleaning.")

                avg_volume = sum(volumes) / len(volumes) if volumes else 0

                # Estrutura correta para o frontend
                data[asset_name] = {
                    'name': asset_name.capitalize(),
                    'symbol': symbol_yf if symbol_yf else symbol_gateio,
                    'price_data': prices,
                    'volume_data': volumes,
                    'volume_colors': volume_colors,
                    'open_data': opens,
                    'high_data': highs,
                    'low_data': lows,
                    'volume_avg_formatted': format_volume(int(avg_volume)),
                    'dates': asset_dates,
                    'macd_data': macd_values,
                    'macd_signal_data': macd_signal_values,
                    'macd_hist_data': macd_hist_values,
                    'rsi_data': rsi_values, 
                    'stochrsi_k_data': stochrsi_k_values, 
                    'stochrsi_d_data': stochrsi_d_values
                }
                logger.info(f"✅ {asset_name}: {len(prices)} points processed for /api/precos/{period}")

            except Exception as e:
                logger.error(f"❌ Error processing {asset_name} data in /api/precos/{period}: {e}", exc_info=True)
                # Fallback com estrutura correta
                data[asset_name] = {
                    'name': asset_name.capitalize(),
                    'symbol': symbol_yf if symbol_yf else symbol_gateio,
                    'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [],
                    'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'dates': [],
                    'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': [],
                    'rsi_data': [], 'stochrsi_k_data': [], 'stochrsi_d_data': []
                }

        # Encontrar datas comuns (usar a maior lista de datas disponível)
        all_asset_dates_lists = [asset_data['dates'] for asset_data in data.values() if asset_data['dates']]
        if all_asset_dates_lists:
            # Pega a lista de dates do BTC para ser o principal, se houver
            common_dates = data.get('btc', {}).get('dates', [])
            if not common_dates:
                common_dates = max(all_asset_dates_lists, key=len)
        else:
            common_dates = []

        response = {
            "period": period,
            "data_points": len(common_dates),
            "dates": common_dates,
            "assets": data,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        logger.info(f"✅ Response prepared for /api/precos/{period}: {len(common_dates)} data points for {len(data)} assets")
        return response
    except Exception as e:
        logger.error(f"❌ Critical error fetching financial data for {period}: {e}", exc_info=True)
        return {
            "error": f"Failed to fetch financial data for {period}: {str(e)}",
            "period": period,
            "data_points": 0,
            "dates": [],
            "assets": {
                'gold': {'name': 'Gold', 'symbol': 'GC=F', 'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [], 'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': [], 'rsi_data': [], 'stochrsi_k_data': [], 'stochrsi_d_data': []},
                'btc': {'name': 'Btc', 'symbol': 'BTC-USD', 'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [], 'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': [], 'rsi_data': [], 'stochrsi_k_data': [], 'stochrsi_d_data': []},
                'dxy': {'name': 'Dxy', 'symbol': 'DX-Y.NYB', 'price_data': [], 'volume_data': [], 'volume_colors': [], 'open_data': [], 'high_data': [], 'low_data': [], 'volume_avg_formatted': '0', 'macd_data': [], 'macd_signal_data': [], 'macd_hist_data': [], 'rsi_data': [], 'stochrsi_k_data': [], 'stochrsi_d_data': []}
            },
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

@app.get("/api/sentiment")
def get_market_sentiment():
    """Endpoint REST para sentimento de mercado em tempo real do Gate.io (do dashboard)"""
    try:
        current_time = datetime.now()
        is_real_data = sentiment_cache["websocket_connected"]

        btc_buyers_pct = sentiment_cache["btc_sentiment"]["buyers"]
        paxg_buyers_pct = sentiment_cache["paxg_sentiment"]["buyers"]

        estimated_btc_volume_24h_display = sentiment_cache["btc_sentiment"]["volume_24h"]

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
            "data_source": "Gate.io Real API" if is_real_data else "Gate.io API Failed",
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
                "trend": sentiment_cache["paxg_sentiment"]["trend"],
                "bid_ask_ratio": sentiment_cache["paxg_sentiment"]["bid_ask_ratio"],
                "total_volume": round(sentiment_cache["paxg_sentiment"]["total_bids"] + sentiment_cache["paxg_sentiment"]["total_asks"], 2)
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
        logger.error(f"❌ Error getting market sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting market sentiment: {str(e)}")

@app.websocket("/ws/sentiment")
async def websocket_endpoint_sentiment(websocket: WebSocket):
    """WebSocket para sentimento de mercado"""
    await websocket.accept()
    active_sentiment_websocket_connections.append(websocket)
    logger.info(f"✅ Cliente WebSocket conectado (sentiment): {websocket.client}")
    
    try:
        # Enviar dados iniciais
        initial_data = await get_market_sentiment_websocket_data()
        await websocket.send_json(initial_data)
        
        while True:
            # Heartbeat para manter conexão viva
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if message == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Enviar ping se não receber mensagem em 30s
                try:
                    await websocket.send_text("ping")
                except:
                    break
            except Exception as e:
                logger.warning(f"Erro na comunicação WebSocket sentiment: {e}")
                break
                
    except Exception as e:
        logger.error(f"❌ Erro no WebSocket sentiment: {e}")
    finally:
        # Cleanup
        if websocket in active_sentiment_websocket_connections:
            active_sentiment_websocket_connections.remove(websocket)
        logger.info(f"🔌 Cliente WebSocket desconectado (sentiment): {websocket.client}")

@app.get("/api/calendar")
def get_economic_calendar():
    """Endpoint para calendário econômico FRED completo (do bot)"""
    try:
        if real_trading_bot and real_trading_bot.analyzer.fred_calendar:
            bot_fred_calendar_instance = real_trading_bot.analyzer.fred_calendar
            if real_trading_bot.environment == 'simulate_backtest' and hasattr(bot_fred_calendar_instance, 'datetime_now_override'):
                bot_fred_calendar_instance.datetime_now_override = datetime.now()
            
            bot_fred_calendar_instance.get_upcoming_releases(days_ahead=14)
            
            upcoming_events = [e.to_dict() for e in bot_fred_calendar_instance.cache["upcoming_events"]]
            pre_event_alerts = bot_fred_calendar_instance.generate_pre_event_alerts(hours_before=48)

            today = datetime.now().date()
            today_events = [
                e for e in upcoming_events
                if "date" in e and datetime.fromisoformat(str(e["date"])).date() == today
            ]

            critical_events = sorted(
                [e for e e in upcoming_events if e.get("importance") == "HIGH"],
                key=lambda x: x.get("impact_score", 0), reverse=True
            )[:10]
            
            fred_api_status = "active" if bot_fred_calendar_instance.api_key else "inactive_no_key"

        else:
            logger.warning("FRED Calendar is not available in the trading bot. Returning empty data.")
            upcoming_events = []
            pre_event_alerts = []
            today_events = []
            critical_events = []
            fred_api_status = "inactive_no_bot"


        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "fred_api_status": fred_api_status,
            "data_freshness": real_trading_bot.analyzer.last_fred_update.isoformat() if real_trading_bot and real_trading_bot.analyzer.last_fred_update else None,
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
                    "GROWTH": len([e for e in upcoming_events if e.get("category") == "GROWTH"]),
                    "CONSUMPTION": len([e for e in upcoming_events if e.get("category") == "CONSUMPTION"]),
                    "MANUFACTURING": len([e for e in upcoming_events if e.get("category") == "MANUFACTURING"]),
                    "HOUSING": len([e for e in upcoming_events if e.get("category") == "HOUSING"]),
                    "TRADE": len([e for e in upcoming_events if e.get("category") == "TRADE"]),
                    "GENERAL": len([e for e in upcoming_events if e.get("category") == "GENERAL"])
                }
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting economic calendar: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting economic calendar: {str(e)}")

@app.get("/api/events")
def get_upcoming_events():
    """Endpoint para eventos econômicos próximos (do bot)"""
    try:
        if real_trading_bot and real_trading_bot.analyzer.fred_calendar:
            bot_fred_calendar_instance = real_trading_bot.analyzer.fred_calendar
            if real_trading_bot.environment == 'simulate_backtest' and hasattr(bot_fred_calendar_instance, 'datetime_now_override'):
                bot_fred_calendar_instance.datetime_now_override = datetime.now()

            bot_fred_calendar_instance.get_upcoming_releases(days_ahead=14)
            upcoming_events_raw = bot_fred_calendar_instance.cache["upcoming_events"]
        else:
            logger.warning("FRED Calendar not available in bot. Providing empty events list.")
            upcoming_events_raw = []

        current_time = datetime.now()
        
        next_week_events = []
        for event_obj in upcoming_events_raw:
            try:
                event = event_obj.to_dict()
                event_date = datetime.fromisoformat(event["date"])
                days_until = (event_date.date() - current_time.date()).days
                if 0 <= days_until <= 7:
                    event_copy = event.copy()
                    event_copy["days_until"] = days_until
                    event_copy["formatted_date"] = event_date.strftime("%d/%m/%Y %H:%M")
                    next_week_events.append(event_copy)
            except Exception as e:
                logger.warning(f"Error processing upcoming event: {e}. Event data: {event_obj}. Skipping.")
                continue
        
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
        logger.error(f"❌ Error getting upcoming events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting upcoming events: {str(e)}")

@app.get("/api/alerts")
def get_all_alerts():
    """Endpoint para todos os tipos de alertas"""
    try:
        all_alerts = cache["alerts"].copy()
        
        if real_trading_bot and real_trading_bot.analyzer.fred_calendar:
            fred_alerts_from_bot = real_trading_bot.analyzer.fred_calendar.generate_pre_event_alerts(hours_before=48)
            for alert in fred_alerts_from_bot:
                if alert not in all_alerts:
                    all_alerts.append(alert)
        
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
        
        all_alerts.extend(list(macd_detector.crossover_history))

        def alert_priority(alert):
            severity_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            return severity_weight.get(alert.get("severity", "LOW"), 1)
        
        all_alerts.sort(key=lambda x: (alert_priority(x), x.get("timestamp", "")), reverse=True)
        
        all_alerts = all_alerts[:50]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "total_alerts": len(all_alerts),
            "alerts": all_alerts,
            "alerts_by_type": {
                "angular_patterns": len([a for a in all_alerts if a.get("type") == "ANGULAR_PATTERN"]),
                "fred_events": len([a for a in all_alerts if a.get("type") == "PRE_EVENT_ALERT"]),
                "trading_signals": len([a for a in all_alerts if a.get("type", "").startswith("BTC_") or a.get("type", "").startswith("TRADING_") or a.get("type") == "MACD_CROSSOVER"]),
                "macd_crossovers": len([a for a in all_alerts if a.get("type") == "MACD_CROSSOVER"]),
                "system_alerts": len([a for a in all_alerts if a.get("type") == "SYSTEM"])
            },
            "alerts_by_severity": {
                "HIGH": len([a for a in all_alerts if a.get("severity") == "HIGH"]),
                "MEDIUM": len([a for a in all_alerts if a.get("severity") == "MEDIUM"]),
                "LOW": len([a for a in all_alerts if a.get("severity") == "LOW"])
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@app.get("/api/backtest-recommendations")
def get_backtest_recommendations():
    """Endpoint para recomendações de backtest (do dashboard)"""
    try:
        current_time = datetime.now()
        last_update = backtest_recommendations_cache["last_update"]
        
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
        logger.error(f"❌ Error getting backtest recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting backtest recommendations: {str(e)}")

# ===============================================================================
# TRADING BOT API ENDPOINTS
# ===============================================================================

@app.get("/api/trading-bot/status")
def get_trading_bot_status():
    """Endpoint para status do trading bot - ATUALIZADO COM AMBIENTE"""
    global real_trading_bot, current_bot_environment
    
    try:
        if real_trading_bot is None:
            logger.warning("⚠️ Bot não inicializado. Retornando status de erro e dados fallback.")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "bot_initialization_failed",
                "error": "Trading bot instance is None. Check startup logs.",
                "environment": current_bot_environment.upper(),
                "running": False,
                "bot_info": {
                    "environment": current_bot_environment.upper(),
                    "strategy": "RSI_MACD_COMBINED",
                    "running": False,
                    "symbols": ["BTC_USDT", "ETH_USDT"],
                    "error": "Bot not initialized"
                },
                "performance": {
                    "start_balance": 0, "current_balance": 0, "total_trades": 0,
                    "winning_trades": 0, "total_pnl": 0, "daily_pnl": 0,
                    "max_drawdown": 0, "win_rate": 0, "roi_percentage": 0,
                    "strategy_name": "RSI_MACD_COMBINED", 
                    "environment": current_bot_environment.upper(),
                    "last_update": datetime.now().isoformat()
                },
                "testnet_performance": {"balance": 0.0, "total_pnl": 0.0, "roi_percentage": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0, "daily_pnl": 0.0, "daily_trades": 0},
                "live_performance": {"balance": 0.0, "total_pnl": 0.0, "roi_percentage": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0, "daily_pnl": 0.0, "daily_trades": 0},
                "active_positions": [], "recent_signals": [],
                "daily_stats": {"trades_today": 0, "max_daily_trades": 0, "daily_pnl": 0, "date": datetime.now().date().isoformat()}
            }

        try:
            status_report = real_trading_bot.get_detailed_ai_status()
            
            # Adicionar informações de ambiente ao status
            status_report["environment_info"] = {
                "current": current_bot_environment.upper(),
                "actual_bot_environment": real_trading_bot.environment.upper(),
                "can_change": not real_trading_bot.running,
                "is_live": current_bot_environment == "live",
                "risk_config": RISK_CONFIG.get(current_bot_environment, {}),
                "environment_matched": current_bot_environment == real_trading_bot.environment
            }

            status_report["debug_info"] = {
                "api_connectivity": "attempting_connection",
                "last_balance_check": datetime.now().isoformat(),
                "environment_config": {
                    "dashboard_environment": current_bot_environment,
                    "bot_environment": real_trading_bot.environment,
                    "has_api_key": bool(API_KEY),
                    "has_secret": bool(SECRET),
                    "base_urls": real_trading_bot.urls
                }
            }
            
            return status_report
            
        except Exception as status_error:
            logger.error(f"❌ Erro ao obter status do bot real: {status_error}", exc_info=True)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "partial_error",
                "error": str(status_error),
                "environment": current_bot_environment.upper(),
                "bot_info": {
                    "environment": current_bot_environment.upper(),
                    "strategy": "RSI_MACD_COMBINED",
                    "running": real_trading_bot.running,
                    "symbols": TRADING_SYMBOLS
                },
                "performance": real_trading_bot.performance if real_trading_bot else {},
                # Adicionar performance de testnet/live como fallback aqui também
                "testnet_performance": real_trading_bot.testnet_performance if real_trading_bot and hasattr(real_trading_bot, 'testnet_performance') else {"balance": 0.0, "total_pnl": 0.0, "roi_percentage": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0, "daily_pnl": 0.0, "daily_trades": 0},
                "live_performance": real_trading_bot.live_performance if real_trading_bot and hasattr(real_trading_bot, 'live_performance') else {"balance": 0.0, "total_pnl": 0.0, "roi_percentage": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0, "daily_pnl": 0.0, "daily_trades": 0},
                "active_positions": [],
                "recent_signals": [],
                "environment_info": {
                    "current": current_bot_environment.upper(),
                    "can_change": not real_trading_bot.running,
                    "is_live": current_bot_environment == "live"
                },
                "debug_info": {
                    "error_occurred": True,
                    "error_message": str(status_error)
                }
            }

    except Exception as e:
        logger.error(f"❌ Erro crítico no endpoint de status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Critical error in trading bot status: {str(e)}")


@app.get("/api/trading-bot/signals")
def get_active_signals():
    """Endpoint para sinais ativos do bot - VERSÃO CORRIGIDA"""
    global real_trading_bot
    
    try:
        if real_trading_bot is None:
            logger.warning("⚠️ Bot não inicializado para sinais. Retornando dados simulados...")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "bot_not_initialized",
                "total_signals": 0,
                "active_signals": 0,
                "expired_signals": 0,
                "signals": [],
                "signals_summary": {
                    "long_signals": 0,
                    "short_signals": 0,
                    "high_confidence": 0,
                    "medium_confidence": 0,
                    "low_confidence": 0
                },
                "debug_info": {
                    "bot_initialized": False,
                    "environment": ENVIRONMENT.upper()
                }
            }

        signals = []
        try:
            signals = [signal.to_dict() for signal in real_trading_bot.signals_history]
            logger.debug(f"Fetched {len(signals)} signals from real bot.")
        except Exception as signals_error:
            logger.error(f"❌ Erro ao obter sinais do bot real: {signals_error}", exc_info=True)
            signals = []
        
        if not signals:
            logger.warning("No real signals from bot, generating simulated signals for display.")
            for symbol in TRADING_SYMBOLS:
                simulated_signal = {
                    "symbol": symbol,
                    "action": random.choice(["BUY", "SELL", "HOLD"]),
                    "final_confidence": round(random.uniform(0.6, 0.9), 3),
                    "ai_probability": round(random.uniform(0.5, 0.9), 3),
                    "signal_type": "LONG" if random.random() > 0.5 else "SHORT",
                    "description": f"Simulado: Sinal forte detectado para {symbol}",
                    "entry_price": round(real_trading_bot.get_current_price(symbol), 2),
                    "timestamp": datetime.now().isoformat(),
                    "environment": ENVIRONMENT.upper(),
                    "status": "SIMULATED"
                }
                signals.append(simulated_signal)
        
        valid_signals = [s for s in signals if s.get('action') != 'HOLD']
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_signals": len(signals),
            "active_signals": len(valid_signals),
            "expired_signals": len(signals) - len(valid_signals),
            "signals": valid_signals[:10],
            "signals_summary": {
                "long_signals": len([s for s in valid_signals if s.get("action") == "BUY"]),
                "short_signals": len([s for s in valid_signals if s.get("action") == "SELL"]),
                "high_confidence": len([s for s in valid_signals if s.get("final_confidence", 0) > 0.8]),
                "medium_confidence": len([s for s in valid_signals if 0.6 <= s.get("final_confidence", 0) <= 0.8]),
                "low_confidence": len([s for s in valid_signals if s.get("final_confidence", 0) < 0.6])
            },
            "debug_info": {
                "bot_initialized": True,
                "bot_running": real_trading_bot.running,
                "environment": ENVIRONMENT.upper(),
                "signals_source": "real_bot" if real_trading_bot and real_trading_bot.signals_history else "simulated"
            }
        }

    except Exception as e:
        logger.error(f"❌ Erro ao obter sinais ativos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting active signals: {str(e)}")


@app.get("/api/trading-bot/test-connection")
def test_bot_connection():
    """Endpoint para testar conectividade do bot"""
    global real_trading_bot
    
    try:
        connection_results = {
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT.upper(),
            "tests": {}
        }
        
        connection_results["tests"]["credentials"] = {
            "api_key_configured": bool(API_KEY and len(API_KEY) > 10),
            "secret_configured": bool(SECRET and len(SECRET) > 10),
            "environment_set": bool(ENVIRONMENT)
        }
        
        if real_trading_bot is None:
            connection_results["tests"]["bot_initialization"] = {
                "status": "failed",
                "error": "Bot is None. Check startup logs."
            }
        else:
            connection_results["tests"]["bot_initialization"] = {
                "status": "success"
            }
        
        if real_trading_bot:
            try:
                balance = real_trading_bot.get_account_balance()
                connection_results["tests"]["balance_retrieval"] = {
                    "status": "success" if balance > 0 else "zero_balance",
                    "balance": balance,
                    "message": f"Saldo obtido: ${balance}"
                }
            except Exception as e:
                connection_results["tests"]["balance_retrieval"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        if real_trading_bot:
            try:
                btc_price = real_trading_bot.get_current_price("BTC_USDT")
                connection_results["tests"]["price_retrieval"] = {
                    "status": "success" if btc_price > 0 else "failed",
                    "value": btc_price,
                    "message": f"Preço BTC: ${btc_price}"
                }
            except Exception as e:
                connection_results["tests"]["price_retrieval"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        if real_trading_bot.analyzer.fred_calendar.api_key:
            try:
                fred_events = real_trading_bot.analyzer.fred_calendar.get_upcoming_releases(days_ahead=1)
                diagnostics["connectivity_tests"]["fred_api_bot"] = {
                    "status": "success" if fred_events else "empty_or_failed",
                    "events_count": len(fred_events)
                }
            except Exception as e:
                diagnostics["connectivity_tests"]["fred_api_bot"] = {
                    "status": "failed",
                    "error": str(e)
                }
        else:
            diagnostics["connectivity_tests"]["fred_api_bot"] = {"status": "skipped", "message": "FRED_API_KEY not configured for bot."}

        if (real_trading_bot.analyzer.sentiment_analyzer.cryptopanic_api_key and real_trading_bot.analyzer.sentiment_analyzer.cryptopanic_api_key != "DUMMY_KEY_CRYPTOPANIC") or (real_trading_bot.analyzer.sentiment_analyzer.news_api_key and real_trading_bot.analyzer.sentiment_analyzer.news_api_key != "DUMMY_KEY_NEWSAPI"):
            try:
                btc_sentiment = real_trading_bot.analyzer.sentiment_analyzer.get_crypto_news_sentiment("BTC_USDT")
                diagnostics["connectivity_tests"]["sentiment_apis_bot"] = {
                    "status": "success" if btc_sentiment != 0.5 else "neutral_or_failed_fetch",
                    "btc_sentiment_score": btc_sentiment
                }
            except Exception as e:
                diagnostics["connectivity_tests"]["sentiment_apis_bot"] = {
                    "status": "failed",
                    "error": str(e)
                }
        else:
            diagnostics["connectivity_tests"]["sentiment_apis_bot"] = {"status": "skipped", "message": "NEWS_API_KEY/CRYPTOPANIC_API_KEY not configured for bot."}
    # O 'else' abaixo é para o 'if real_trading_bot:' inicial da função.
    # Ele deve estar no mesmo nível do 'if real_trading_bot:'.
    else: 
        diagnostics["recommendations"].extend([
            "Check if API_KEY and SECRET are configured in .env file",
            "Verify API key permissions on Gate.io",
            "Restart the application after fixing credentials",
            "Ensure FRED_API_KEY, NEWS_API_KEY, CRYPTOPANIC_API_KEY are configured in config.py or .env for the bot's full functionality."
        ])
    
    return diagnostics
    
@app.websocket("/ws/rsi-macd")
async def websocket_rsi_macd_endpoint(websocket: WebSocket):
    """WebSocket específico para dados RSI+MACD do card"""
    await websocket.accept()
    active_rsi_macd_websocket_connections.append(websocket)
    logger.info(f"RSI+MACD WebSocket client connected: {websocket.client}")
    
    try:
        initial_data = await get_rsi_macd_websocket_data()
        await websocket.send_json(initial_data)
        logger.info(f"Sent initial RSI+MACD data to client {websocket.client}")
        
        while True:
            try:
                # Heartbeat para manter conexão viva
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if message == "ping":
                    await websocket.send_text("pong")
                # Enviar dados atualizados periodicamente, ou se a mensagem não for 'ping'
                updated_data = await get_rsi_macd_websocket_data()
                await websocket.send_json(updated_data)
                
            except asyncio.TimeoutError:
                # Se não receber mensagem em 30s, tenta enviar dados atualizados
                updated_data = await get_rsi_macd_websocket_data()
                await websocket.send_json(updated_data)
                
            except Exception as e:
                logger.warning(f"Erro na comunicação WebSocket RSI+MACD: {e}")
                break # Sai do loop se houver erro de comunicação
                
    except WebSocketDisconnect as e:
        if websocket in active_rsi_macd_websocket_connections:
            active_rsi_macd_websocket_connections.remove(websocket)
        logger.info(f"RSI+MACD WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        if websocket in active_rsi_macd_websocket_connections:
            active_rsi_macd_websocket_connections.remove(websocket)
        logger.error(f"RSI+MACD WebSocket error: {e}", exc_info=True)

async def get_rsi_macd_websocket_data():
    """Gera dados para WebSocket RSI+MACD"""
    global real_trading_bot
    
    try:
        current_time = datetime.now()
        
        # Tenta obter dados reais do cache de indicadores em tempo real
        btc_ohlcv_cache = realtime_ohlcv_cache.get("btc", {})
        
        rsi_value = btc_ohlcv_cache.get("rsi_data", {}).get("last_value", 0)
        
        macd_data = btc_ohlcv_cache.get("macd_data", {})
        macd_value = macd_data.get("macd", [])[-1] if macd_data.get("macd") else 0
        signal_value = macd_data.get("signal", [])[-1] if macd_data.get("signal") else 0
        histogram = macd_data.get("histogram", [])[-1] if macd_data.get("histogram") else 0

        # Novos dados do STOCHRSI
        stochrsi_k_val = btc_ohlcv_cache.get("stochrsi_data", {}).get("k_value", 0)
        stochrsi_d_val = btc_ohlcv_cache.get("stochrsi_data", {}).get("d_value", 0)

        # Angulos (do cache, se existirem)
        rsi_angle = btc_ohlcv_cache.get("rsi_data", {}).get("angle", 0)
        macd_angle = btc_ohlcv_cache.get("macd_angle_data", {}).get("macd_angle", 0)
        signal_angle = btc_ohlcv_cache.get("macd_angle_data", {}).get("signal_angle", 0)
        
        # Sinal (determinado no backend, ou fallback aqui)
        signal_type = macd_data.get("trend", "HOLD") if macd_data.get("trend") else "HOLD"
        
        data_source = "real_time_indicators"
        if not btc_ohlcv_cache.get("websocket_connected"):
            data_source = "websocket_disconnected"

        # Lógica de sinal para exibição (corrigida a sintaxe Python)
        macd_hist_threshold_display = 0.0005
        rsi_middle_threshold_display = 5 

        if histogram > macd_hist_threshold_display and rsi_value < 70 and abs(rsi_value - 50) > rsi_middle_threshold_display:
            signal_type = "BUY"
        elif histogram < -macd_hist_threshold_display and rsi_value > 30 and abs(rsi_value - 50) > rsi_middle_threshold_display:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
          
        # Considerar STOCHRSI para o sinal combinado (corrigida a sintaxe Python)
        if stochrsi_k_val > stochrsi_d_val and stochrsi_k_val < 80 and stochrsi_d_val < 80 and signal_type == "BUY":
            signal_type = "STRONG_BUY"
        elif stochrsi_k_val < stochrsi_d_val and stochrsi_k_val > 20 and stochrsi_d_val > 20 and signal_type == "SELL":
            signal_type = "STRONG_SELL"


        rsi_zone = "oversold" if rsi_value < 30 else ("overbought" if rsi_value > 70 else "neutral")
        rsi_trend = btc_ohlcv_cache.get("rsi_data", {}).get("trend", "NEUTRAL") 
        rsi_confidence = 85 if rsi_zone in ["oversold", "overbought"] else 45
        macd_strength = min(abs(histogram) * 15, 90)
        combined_confidence = round((rsi_confidence + macd_strength) / 2, 0) 

        signal_color = "#FFD700"
        if "BUY" in signal_type: signal_color = "#00ff00" # Ajustado para incluir STRONG_BUY
        if "SELL" in signal_type: signal_color = "#ff0000" # Ajustado para incluir STRONG_SELL


        return {
            "type": "rsi_macd_update",
            "timestamp": current_time.isoformat(),
            "symbol": "BTC_USDT",
            "environment": ENVIRONMENT.upper(),
            "data_source": data_source,
            "rsi": {
                "value": rsi_value,
                "zone": rsi_zone,
                "trend": rsi_trend,
                "confidence": rsi_confidence,
                "color": "#ff4444" if rsi_zone == "overbought" else "#44ff44" if rsi_zone == "oversold" else "#ffaa00",
                "angle": rsi_angle
            },
            "macd": {
                "macd": macd_value,
                "signal": signal_value,
                "histogram": histogram,
                "trend": "bullish" if histogram > 0 else "bearish" if histogram < 0 else "neutral",
                "strength": round(abs(histogram), 2)
            },
            "macd_angle_data": { 
                "macd_angle": macd_angle,
                "signal_angle": signal_angle,
            },
            "stochrsi": { 
                "k_value": stochrsi_k_val,
                "d_value": stochrsi_d_val
            },
            "combined": {
                "confidence": combined_confidence,
                "signal_type": signal_type,
                "signal_color": signal_color,
                "recommendation": f"{signal_type.replace('_', ' ').title()} - Confiança: {combined_confidence}%"
            },
            "volume": {
                "confirmation": random.choice([True, False]),
                "status": random.choice(["Alto", "Médio", "Baixo"])
            },
            "additional": {
                "risk_reward": round(random.uniform(1.2, 3.5), 1),
                "last_update": current_time.strftime("%H:%M:%S"),
                "validity": "valid",
                "bot_running": real_trading_bot.running if real_trading_bot else False
            }
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar dados WebSocket RSI+MACD: {e}", exc_info=True)
        return {
            "type": "rsi_macd_error",
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTC_USDT",
            "error": str(e),
            "rsi": {"value": 0, "zone": "neutral", "trend": "flat", "angle": 0},
            "macd": {"macd": 0, "signal": 0, "histogram": 0},
            "macd_angle_data": {"macd_angle": 0, "signal_angle": 0},
            "stochrsi": {"k_value": 0, "d_value": 0},
            "combined": {"confidence": 0, "signal_type": "ERROR", "signal_color": "#FFD700"}
        }


async def broadcast_rsi_macd_updates():
    """Envia atualizações periódicas para todos os clientes RSI+MACD conectados"""
    while True:
        try:
            if active_rsi_macd_websocket_connections:
                data = await get_rsi_macd_websocket_data()
                
                disconnected_clients = []
                for connection in list(active_rsi_macd_websocket_connections):
                    try:
                        await connection.send_json(data)
                    except Exception as e:
                        logger.warning(f"Erro enviando dados RSI+MACD: {e}")
                        disconnected_clients.append(connection)
                
                for client in disconnected_clients:
                    if client in active_rsi_macd_websocket_connections:
                        active_rsi_macd_websocket_connections.remove(client)
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in RSI+MACD broadcast loop: {e}")
            await asyncio.sleep(5)

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de encerramento da aplicação"""
    global real_trading_bot

    logger.info("🛑 Shutting down Trading Dashboard API...")
    
    if real_trading_bot and real_trading_bot.running:
        real_trading_bot.stop()
        
    for ws in active_sentiment_websocket_connections:
        try:
            await ws.close()
        except:
            pass
    
    for ws in active_ohlcv_websocket_connections:
        try:
            await ws.close()
        except:
            pass

    if hasattr(app.state, 'bot_executor') and app.state.bot_executor:
        app.state.bot_executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor for bot shut down.")

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
    print("   📈 Real-time OHLCV + MACD: /api/macd/realtime/{asset} (+ WebSocket: /ws/ohlcv)")
    print("   🤖 Trading bot: /api/trading-bot/*")
    print("   📡 Backtest signals: /api/backtest-recommendations")
    print("   📅 Economic calendar: /api/calendar")
    print("   🚨 Alerts system: /api/alerts")
    print("   ⚙️ System status: /api/status")
    print("=" * 80)
    print("⚠️  IMPORTANT NOTES:")
    print("   • This is a demonstration/educational system")
    print("   • Real Gate.io API integration for sentiment analysis (PAXG and FGI) and OHLCV")
    print("   • FRED economic data is sourced from the REAL Trading Bot's FRED Calendar")
    print("   • Market sentiment for BTC/ETH is aggregated by the REAL Trading Bot (NewsAPI, CryptoPanic)")
    print("   • Trading bot is now REAL (fetches actual balance/positions and makes decisions)")
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