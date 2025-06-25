import ccxt
import pandas as pd
import numpy as np
import logging
import time
import ta  # Usando apenas a biblioteca ta para indicadores
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import asyncio
import threading
import random
from concurrent.futures import ThreadPoolExecutor
import websocket
import requests

# ML e análise (mantido do original, mas não diretamente usado na StrategyEngine fornecida)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ============================================================================
# 🔧 CONFIGURAÇÕES DO TRADING BOT AO VIVO
# ============================================================================

# Configurações da Gate.io TESTNET (mantidas)
GATEIO_TESTNET_CONFIG = {
    'apiKey': '99f9b66eb363e7c2a7fafb21d6514f8f',
    'secret': '4ac51e8c8d5e9376406a9ccb0d3bd6921d01b9e7e200c159fa937e7b243eb8a7',
    'sandbox': True,  # Modo testnet
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'createMarketBuyOrderRequiresPrice': False,
    },
    'urls': {
        'api': {
            'public': 'https://fx-api-testnet.gateio.ws',
            'private': 'https://fx-api-testnet.gateio.ws',
        }
    }
}

# Símbolos para trading ao vivo
LIVE_SYMBOLS = ['BTC', 'ETH', 'SOL', 'DOGE']
CCXT_SYMBOL_MAP = {
    'BTC': 'BTC/USDT:USDT',
    'ETH': 'ETH/USDT:USDT',
    'SOL': 'SOL/USDT:USDT',
    'DOGE': 'DOGE/USDT:USDT'
}

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']
PRIMARY_TIMEFRAME = '5m'

# Configurações de risco
RISK_CONFIG = {
    'max_position_size_percent': 10,
    'max_daily_loss_percent': 5,
    'max_open_positions': 3,
    'min_confidence_threshold': 0.75,
    'stop_loss_percent': 2,
    'take_profit_percent': 6,
}

# ============================================================================
# 📊 CLASSES DE DADOS (LiveSignal e LivePosition)
# ============================================================================

@dataclass
class LiveSignal:
    """Sinal de trading em tempo real"""
    symbol: str
    signal_type: str  # 'LONG' ou 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy_name: str
    timeframe: str
    timestamp: datetime
    market_context: Dict
    technical_indicators: Dict
    risk_reward_ratio: float
    expected_duration: str
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'market_context': self.market_context,
            'technical_indicators': self.technical_indicators,
            'risk_reward_ratio': self.risk_reward_ratio,
            'expected_duration': self.expected_duration
        }

@dataclass
class LivePosition:
    """Posição ativa no mercado"""
    symbol: str
    side: str  # 'long' ou 'short'
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    timestamp: datetime
    strategy_name: str
    confidence: float
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'confidence': self.confidence
        }

# ============================================================================
# 🧠 SISTEMA DE ESTRATÉGIAS INTELIGENTES (ATUALIZADO COM TA)
# ============================================================================

class StrategyEngine:
    """Engine de estratégias de trading com análise técnica usando biblioteca ta"""
    
    def __init__(self, df: pd.DataFrame, symbol: str, timeframe: str):
        self.df = df.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        self.signals: List[LiveSignal] = []
        self.indicators: Dict = {}
        
        # Validação e preparação dos dados
        if not all(col in self.df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', 'volume' columns.")
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        if len(self.df) >= 50:
            self.apply_indicators()
        else:
            logging.warning("Dados insuficientes para StrategyEngine")
            self.df = pd.DataFrame()

    def apply_indicators(self):
        """Calcula todos os indicadores técnicos usando ta"""
        try:
            # MACD
            macd_calc = ta.trend.MACD(self.df['close'])
            self.df['macd'] = macd_calc.macd()
            self.df['macd_signal'] = macd_calc.macd_signal()
            self.df['macd_hist'] = macd_calc.macd_diff()
            
            # Médias móveis
            self.df['sma20'] = ta.trend.sma_indicator(self.df['close'], window=20)
            self.df['sma50'] = ta.trend.sma_indicator(self.df['close'], window=50)
            
            # RSI
            self.df['rsi'] = ta.momentum.RSIIndicator(self.df['close'], window=14).rsi()
            
            # Volume
            self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
            
            # Remover NaNs resultantes
            self.df.dropna(inplace=True)
            
            # Armazenar indicadores para referência
            self.indicators = {
                'macd': self.df['macd'],
                'macd_signal': self.df['macd_signal'],
                'macd_hist': self.df['macd_hist'],
                'sma20': self.df['sma20'],
                'sma50': self.df['sma50'],
                'rsi': self.df['rsi'],
                'volume_sma': self.df['volume_sma']
            }
            
        except Exception as e:
            logging.error(f"Erro ao calcular indicadores: {e}")
            self.indicators = {}

    def _create_live_signal(self, signal_type: str, confidence: float, 
                           strategy_name: str, entry_price: float) -> Optional[LiveSignal]:
        """Cria objeto LiveSignal com gerenciamento de risco"""
        if pd.isna(entry_price):
            return None

        # Cálculo de stop loss e take profit baseado no risco
        if signal_type == 'LONG':
            stop_loss = entry_price * (1 - RISK_CONFIG['stop_loss_percent'] / 100)
            take_profit = entry_price * (1 + RISK_CONFIG['take_profit_percent'] / 100)
        else:  # SHORT
            stop_loss = entry_price * (1 + RISK_CONFIG['stop_loss_percent'] / 100)
            take_profit = entry_price * (1 - RISK_CONFIG['take_profit_percent'] / 100)

        return LiveSignal(
            symbol=self.symbol,
            signal_type=signal_type,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence=float(confidence),
            strategy_name=strategy_name,
            timeframe=self.timeframe,
            timestamp=datetime.now(),
            market_context={},  # Contexto simplificado
            technical_indicators={k: v.iloc[-1] for k, v in self.indicators.items()},
            risk_reward_ratio=3.0,
            expected_duration='N/A'
        )

    def macd_strong_crossover(self):
        """Estratégia de cruzamento MACD com confirmação"""
        if len(self.df) < 2:
            return
            
        i = -1  # Último índice
        
        # Cruzamento de alta
        if (self.df['macd'].iloc[i-1] < self.df['macd_signal'].iloc[i-1] and
            self.df['macd'].iloc[i] > self.df['macd_signal'].iloc[i] and
            self.df['macd_hist'].iloc[i] > 0):
            
            signal = self._create_live_signal(
                'LONG', 0.85, 'MACD STRONG CROSSOVER', self.df['close'].iloc[i])
            if signal:
                self.signals.append(signal)

        # Cruzamento de baixa
        if (self.df['macd'].iloc[i-1] > self.df['macd_signal'].iloc[i-1] and
            self.df['macd'].iloc[i] < self.df['macd_signal'].iloc[i] and
            self.df['macd_hist'].iloc[i] < 0):
            
            signal = self._create_live_signal(
                'SHORT', 0.85, 'MACD STRONG CROSSUNDER', self.df['close'].iloc[i])
            if signal:
                self.signals.append(signal)

    def volume_breakout(self):
        """Estratégia de rompimento com volume acima da média"""
        if len(self.df) < 2:
            return
            
        i = -1
        if (self.df['close'].iloc[i] > self.df['close'].iloc[i-1] and
            self.df['volume'].iloc[i] > self.df['volume_sma'].iloc[i] * 1.5):
            
            signal = self._create_live_signal(
                'LONG', 0.8, 'VOLUME BREAKOUT', self.df['close'].iloc[i])
            if signal:
                self.signals.append(signal)

    def candle_rejection(self):
        """Estratégia de candle de rejeição"""
        if len(self.df) < 1:
            return
            
        i = -1
        body = abs(self.df['open'].iloc[i] - self.df['close'].iloc[i])
        upper_wick = self.df['high'].iloc[i] - max(self.df['open'].iloc[i], self.df['close'].iloc[i])
        
        # Rejeição no topo (sinal de baixa)
        if (upper_wick > body * 1.5 and
            self.df['volume'].iloc[i] > self.df['volume_sma'].iloc[i] * 1.2):
            
            signal = self._create_live_signal(
                'SHORT', 0.75, 'REJECTION CANDLE', self.df['close'].iloc[i])
            if signal:
                self.signals.append(signal)

    def trend_follow(self):
        """Estratégia de seguimento de tendência"""
        if len(self.df) < 1:
            return
            
        i = -1
        if self.df['sma20'].iloc[i] > self.df['sma50'].iloc[i]:
            signal = self._create_live_signal(
                'LONG', 0.65, 'TREND FOLLOWING', self.df['close'].iloc[i])
            if signal:
                self.signals.append(signal)

    def run_all_strategies(self) -> List[LiveSignal]:
        """Executa todas as estratégias e retorna sinais válidos"""
        self.signals = []
        
        if self.df.empty:
            return []
        
        self.macd_strong_crossover()
        self.volume_breakout()
        self.candle_rejection()
        self.trend_follow()
        
        return [s for s in self.signals if s.confidence >= RISK_CONFIG['min_confidence_threshold']]

# ============================================================================
# 🤖 TRADING BOT PRINCIPAL (CLASSE LiveTradingBot)
# ============================================================================

class LiveTradingBot:
    """Bot de trading ao vivo com testnet Gate.io"""
    
    def __init__(self):
        self.exchange = None
        self.active_signals = []
        self.active_positions = []
        self.performance_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'daily_pnl': 0,
            'max_drawdown': 0,
            'start_balance': 0,
            'current_balance': 0,
            'roi_percentage': 0,
            'win_rate': 0,
            'daily_trades': 0,
            'last_update': None,
            'advanced_metrics': {
                "sharpe_ratio": 0, "sortino_ratio": 0, "profit_factor": 0,
                "maximum_consecutive_wins": 0, "volatility": 0, "beta": 0
            }
        }
        self.running = False
        self.last_analysis = {}
        
        # Configuração de logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Inicializa conexão com exchange"""
        try:
            self.exchange = ccxt.gateio(GATEIO_TESTNET_CONFIG)
            balance = await self.exchange.fetch_balance()
            self.performance_data['start_balance'] = balance['USDT']['total']
            self.performance_data['current_balance'] = balance['USDT']['total']
            self.logger.info(f"✅ Conectado ao testnet Gate.io. Saldo: {balance['USDT']['total']} USDT")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro ao conectar exchange: {e}")
            return False
    
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Busca dados OHLCV"""
        try:
            ccxt_symbol = CCXT_SYMBOL_MAP.get(symbol, f"{symbol}/USDT:USDT")
            ohlcv = await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Erro ao buscar dados {symbol}: {e}")
            return pd.DataFrame()
    
    async def analyze_markets(self) -> List[LiveSignal]:
        """Analisa todos os mercados usando a nova StrategyEngine"""
        all_signals = []
        
        try:
            for display_symbol in LIVE_SYMBOLS:
                df = await self.fetch_ohlcv_data(display_symbol, PRIMARY_TIMEFRAME)
                if df.empty or len(df) < 50:
                    self.logger.warning(f"Dados insuficientes para {display_symbol}. Pulando análise.")
                    continue
                
                # Usando a nova StrategyEngine
                strategy_engine = StrategyEngine(df, display_symbol, PRIMARY_TIMEFRAME)
                generated_signals = strategy_engine.run_all_strategies()
                all_signals.extend(generated_signals)
                
                # Armazena última análise
                self.last_analysis[f"{display_symbol}_{PRIMARY_TIMEFRAME}"] = {
                    'timestamp': datetime.now(),
                    'signals_count': len(generated_signals),
                    'price': df['close'].iloc[-1]
                }
                        
        except Exception as e:
            self.logger.error(f"Erro na análise de mercados: {e}")
        
        # Ordena por confiança e limita a 10 sinais
        all_signals.sort(key=lambda x: x.confidence, reverse=True)
        return all_signals[:10]

    async def update_positions(self):
        """Atualiza posições ativas (simulado)"""
        if self.running:
            positions_to_keep = []
            for pos in self.active_positions:
                pos.current_price = pos.entry_price * (1 + random.uniform(-0.005, 0.005))
                if pos.side == 'long':
                    pos.pnl = (pos.current_price - pos.entry_price) * pos.size
                else:  # short
                    pos.pnl = (pos.entry_price - pos.current_price) * pos.size
                pos.pnl_percent = (pos.pnl / (pos.entry_price * pos.size)) * 100 if (pos.entry_price * pos.size) != 0 else 0
                pos.timestamp = datetime.now()
                positions_to_keep.append(pos)
            self.active_positions = positions_to_keep

    async def _process_signals_and_positions(self, signals: List[LiveSignal]):
        """Processa novos sinais e gerencia posições"""
        processed_symbols = set()
        signals_to_consider_opening = []
        
        for new_signal in signals:
            opposing_positions = [
                pos for pos in self.active_positions
                if pos.symbol == new_signal.symbol and (
                    (pos.side == 'long' and new_signal.signal_type == 'SHORT') or
                    (pos.side == 'short' and new_signal.signal_type == 'LONG')
                )
            ]

            if opposing_positions:
                self.logger.info(f"🚨 Sinal oposto detectado para {new_signal.symbol}. Fechando posição.")
                for pos in opposing_positions:
                    # Simula fechamento da posição
                    self.performance_data['total_trades'] += 1
                    closed_pnl = pos.pnl
                    if closed_pnl >= 0: 
                        self.performance_data['winning_trades'] += 1
                    self.performance_data['total_pnl'] += closed_pnl
                    self.performance_data['daily_pnl'] += closed_pnl
                    self.performance_data['daily_trades'] += 1
                    self.active_positions.remove(pos)
                    processed_symbols.add(new_signal.symbol)

            if new_signal.symbol not in processed_symbols:
                signals_to_consider_opening.append(new_signal)
                processed_symbols.add(new_signal.symbol)

        for signal in signals_to_consider_opening:
            if (not any(pos.symbol == signal.symbol for pos in self.active_positions) and \
               self.running and \
               signal.confidence >= RISK_CONFIG['min_confidence_threshold'] and \
               len(self.active_positions) < RISK_CONFIG['max_open_positions']:
                await self.execute_signal(signal)

    async def execute_signal(self, signal: LiveSignal) -> bool:
        """Executa um sinal de trading (simulado)"""
        self.logger.info(f"SIMULADO: Executando sinal: {signal.signal_type} {signal.symbol}")
        
        if len(self.active_positions) < RISK_CONFIG['max_open_positions']:
            new_position = LivePosition(
                symbol=signal.symbol,
                side=signal.signal_type.lower(),
                size=random.uniform(0.001, 0.1),
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                pnl=0.0,
                pnl_percent=0.0,
                timestamp=datetime.now(),
                strategy_name=signal.strategy_name,
                confidence=signal.confidence
            )
            self.active_positions.append(new_position)
            self.logger.info(f"SIMULADO: Posição aberta: {new_position.side.upper()} {new_position.symbol}")
            return True
        return False

    async def run_trading_loop(self):
        """Loop principal de trading"""
        self.running = True
        self.logger.info("🚀 Iniciando loop de trading...")
        
        while self.running:
            try:
                await self.update_positions()
                signals = await self.analyze_markets()
                self.active_signals = signals
                await self._process_signals_and_positions(signals)
                await self.update_performance()
                self.logger.info(f"📊 Sinais: {len(self.active_signals)}, Posições: {len(self.active_positions)}")
                await asyncio.sleep(30)
            except Exception as e:
                self.logger.error(f"❌ Erro no loop: {e}")
                await asyncio.sleep(10)
    
    async def update_performance(self):
        """Atualiza métricas de performance (simulado)"""
        try:
            # Simulação de trades aleatórios para preencher métricas
            if self.running and random.random() > 0.7:
                if not self.active_positions and self.performance_data['total_trades'] % 5 == 0:
                    self.performance_data['total_trades'] += 1
                    if random.random() > 0.55:
                        self.performance_data['winning_trades'] += 1
                        pnl_change = random.uniform(20, 200)
                    else:
                        pnl_change = -random.uniform(10, 100)
                    self.performance_data['total_pnl'] += pnl_change
                    self.performance_data['daily_pnl'] += pnl_change
                    self.performance_data['daily_trades'] += 1

            # Atualiza saldo
            self.performance_data['current_balance'] = \
                self.performance_data['start_balance'] + self.performance_data['total_pnl']
            
            # Calcula ROI e taxa de acerto
            start = self.performance_data['start_balance']
            current = self.performance_data['current_balance']
            total_trades = max(self.performance_data['total_trades'], 1)
            
            self.performance_data['roi_percentage'] = ((current - start) / start) * 100
            self.performance_data['win_rate'] = \
                (self.performance_data['winning_trades'] / total_trades) * 100
            
            self.performance_data['last_update'] = datetime.now().isoformat()
            
            # Simula métricas avançadas
            self.performance_data['advanced_metrics'] = {
                "sharpe_ratio": round(random.uniform(1.0, 2.5), 2),
                "sortino_ratio": round(random.uniform(1.2, 3.0), 2),
                "profit_factor": round(random.uniform(1.1, 2.5), 2),
                "maximum_consecutive_wins": random.randint(3, 8),
                "volatility": round(random.uniform(5.0, 15.0), 2),
                "beta": round(random.uniform(0.7, 1.3), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar performance: {e}")
    
    def get_status(self) -> Dict:
        """Retorna status atual do bot"""
        return {
            'running': self.running,
            'active_signals': [signal.to_dict() for signal in self.active_signals],
            'active_positions': [pos.to_dict() for pos in self.active_positions],
            'performance': self.performance_data,
            'last_analysis': self.last_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def stop(self):
        """Para o bot"""
        self.running = False
        self.logger.info("🛑 Bot parado")

# ============================================================================
# 🌐 INSTÂNCIA GLOBAL DO BOT
# ============================================================================

trading_bot = LiveTradingBot()

# ============================================================================
# 🔧 FUNÇÕES DE INICIALIZAÇÃO
# ============================================================================

async def start_live_trading_bot():
    """Inicia o bot de trading ao vivo"""
    try:
        success = await trading_bot.initialize()
        if success:
            asyncio.create_task(trading_bot.run_trading_loop())
            return True
        return False
    except Exception as e:
        logging.error(f"Erro ao iniciar bot: {e}")
        return False

def get_live_recommendations() -> List[Dict]:
    """Retorna recomendações em tempo real"""
    try:
        return [signal.to_dict() for signal in trading_bot.active_signals]
    except Exception as e:
        logging.error(f"Erro ao obter recomendações: {e}")
        return []

def get_live_positions() -> List[Dict]:
    """Retorna posições ativas"""
    try:
        return [pos.to_dict() for pos in trading_bot.active_positions]
    except Exception as e:
        logging.error(f"Erro ao obter posições: {e}")
        return []

def get_bot_status() -> Dict:
    """Retorna status do bot"""
    try:
        return trading_bot.get_status()
    except Exception as e:
        logging.error(f"Erro ao obter status: {e}")
        return {'error': str(e)}

# ============================================================================
# 🧪 FUNÇÃO DE TESTE
# ============================================================================

async def test_trading_bot():
    """Testa o bot de trading"""
    print("🧪 Testando Trading Bot...")
    
    bot = LiveTradingBot()
    
    if await bot.initialize():
        print("✅ Bot inicializado com sucesso")
        signals = await bot.analyze_markets()
        print(f"📊 Encontrados {len(signals)} sinais")
        
        for signal in signals[:3]:
            print(f"🎯 {signal.signal_type} {signal.symbol} - Confiança: {signal.confidence:.2f}")
            
        bot.stop()
    else:
        print("❌ Falha na inicialização")

# ============================================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    asyncio.run(test_trading_bot())