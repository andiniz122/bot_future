#!/usr/bin/env python3
"""
Sistema de Estratégias Autônomo - Bot Verdadeiramente Independente
Capaz de descobrir, testar e evoluir estratégias automaticamente
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('autonomous_strategy_engine')

# =====================================================================
# CLASSES DE DADOS PARA ESTRATÉGIAS
# =====================================================================

@dataclass
class StrategyGenes:
    """DNA de uma estratégia - parâmetros que podem evoluir"""
    # Indicadores técnicos
    ema_fast: int = field(default_factory=lambda: random.randint(5, 15))
    ema_slow: int = field(default_factory=lambda: random.randint(20, 50))
    rsi_period: int = field(default_factory=lambda: random.randint(10, 25))
    rsi_oversold: float = field(default_factory=lambda: random.uniform(20, 35))
    rsi_overbought: float = field(default_factory=lambda: random.uniform(65, 80))
    
    # Bollinger Bands
    bb_period: int = field(default_factory=lambda: random.randint(15, 30))
    bb_std: float = field(default_factory=lambda: random.uniform(1.5, 2.5))
    
    # MACD
    macd_fast: int = field(default_factory=lambda: random.randint(8, 15))
    macd_slow: int = field(default_factory=lambda: random.randint(20, 30))
    macd_signal: int = field(default_factory=lambda: random.randint(5, 12))
    
    # Volume
    volume_period: int = field(default_factory=lambda: random.randint(10, 30))
    volume_threshold: float = field(default_factory=lambda: random.uniform(1.2, 2.5))
    
    # Níveis de confiança
    min_confidence: float = field(default_factory=lambda: random.uniform(0.4, 0.8))
    
    # Gestão de risco
    stop_loss: float = field(default_factory=lambda: random.uniform(0.01, 0.03))
    take_profit: float = field(default_factory=lambda: random.uniform(0.015, 0.05))
    
    # Filtros de mercado
    min_volatility: float = field(default_factory=lambda: random.uniform(0.005, 0.02))
    max_volatility: float = field(default_factory=lambda: random.uniform(0.03, 0.08))
    
    # Timing
    entry_cooldown: int = field(default_factory=lambda: random.randint(5, 20))
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutação genética dos parâmetros"""
        if random.random() < mutation_rate:
            self.ema_fast = max(5, min(15, self.ema_fast + random.randint(-2, 2)))
        if random.random() < mutation_rate:
            self.ema_slow = max(20, min(50, self.ema_slow + random.randint(-5, 5)))
        if random.random() < mutation_rate:
            self.rsi_period = max(10, min(25, self.rsi_period + random.randint(-2, 2)))
        if random.random() < mutation_rate:
            self.rsi_oversold = max(15, min(40, self.rsi_oversold + random.uniform(-3, 3)))
        if random.random() < mutation_rate:
            self.rsi_overbought = max(60, min(85, self.rsi_overbought + random.uniform(-3, 3)))
        if random.random() < mutation_rate:
            self.bb_period = max(15, min(30, self.bb_period + random.randint(-2, 2)))
        if random.random() < mutation_rate:
            self.bb_std = max(1.5, min(2.5, self.bb_std + random.uniform(-0.2, 0.2)))
        if random.random() < mutation_rate:
            self.volume_threshold = max(1.1, min(3.0, self.volume_threshold + random.uniform(-0.2, 0.2)))
        if random.random() < mutation_rate:
            self.min_confidence = max(0.3, min(0.9, self.min_confidence + random.uniform(-0.05, 0.05)))
        if random.random() < mutation_rate:
            self.stop_loss = max(0.005, min(0.05, self.stop_loss + random.uniform(-0.005, 0.005)))
        if random.random() < mutation_rate:
            self.take_profit = max(0.01, min(0.08, self.take_profit + random.uniform(-0.01, 0.01)))

@dataclass
class StrategyPerformance:
    """Métricas de performance de uma estratégia"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Métricas de robustez
    consistency_score: float = 0.0
    adaptability_score: float = 0.0
    market_correlation: float = 0.0
    
    def update_metrics(self, trades: List[Dict]):
        """Atualiza métricas baseado em lista de trades"""
        if not trades:
            return
            
        self.total_trades = len(trades)
        
        profits = [t['profit'] for t in trades if 'profit' in t]
        if profits:
            self.total_return = sum(profits)
            self.winning_trades = len([p for p in profits if p > 0])
            self.losing_trades = len([p for p in profits if p < 0])
            self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum([p for p in profits if p > 0])
            gross_loss = abs(sum([p for p in profits if p < 0]))
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe ratio simplificado
            if len(profits) > 1:
                mean_return = np.mean(profits)
                std_return = np.std(profits)
                self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Max drawdown
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            self.max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
        self.last_updated = datetime.now()

@dataclass
class AutonomousStrategy:
    """Estratégia autônoma com capacidade de evolução"""
    id: str
    name: str
    genes: StrategyGenes
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)
    creation_date: datetime = field(default_factory=datetime.now)
    last_trade_time: Optional[datetime] = None
    active: bool = True
    
    # Componentes ML
    ml_model: Optional[Any] = None
    feature_scaler: Optional[StandardScaler] = None
    
    # Histórico de trades
    trade_history: List[Dict] = field(default_factory=list)
    
    # Métricas de adaptação
    adaptation_cycles: int = 0
    last_adaptation: Optional[datetime] = None
    
    def calculate_fitness(self) -> float:
        """Calcula fitness da estratégia para seleção genética"""
        if self.performance.total_trades < 10:
            return 0.0
            
        # Combina múltiplos fatores
        return_factor = max(0, self.performance.total_return)
        win_rate_factor = self.performance.win_rate
        profit_factor = min(5.0, self.performance.profit_factor)  # Cap para evitar outliers
        drawdown_penalty = 1.0 - abs(self.performance.max_drawdown)
        sharpe_factor = max(0, self.performance.sharpe_ratio)
        
        # Penaliza estratégias com poucos trades
        trade_factor = min(1.0, self.performance.total_trades / 50.0)
        
        fitness = (
            return_factor * 0.3 +
            win_rate_factor * 0.2 +
            profit_factor * 0.15 +
            drawdown_penalty * 0.2 +
            sharpe_factor * 0.1 +
            trade_factor * 0.05
        )
        
        return max(0.0, fitness)

# =====================================================================
# GERADOR DE FEATURES DINÂMICO
# =====================================================================

class DynamicFeatureGenerator:
    """Gerador dinâmico de features para ML"""
    
    def __init__(self):
        self.feature_functions = {
            'price_momentum': self._price_momentum,
            'volume_profile': self._volume_profile,
            'volatility_regime': self._volatility_regime,
            'trend_strength': self._trend_strength,
            'support_resistance': self._support_resistance,
            'pattern_recognition': self._pattern_recognition,
            'cross_correlation': self._cross_correlation,
            'market_microstructure': self._market_microstructure,
        }
    
    def generate_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Gera features dinâmicas para um DataFrame"""
        features_df = df.copy()
        
        for feature_name, feature_func in self.feature_functions.items():
            try:
                new_features = feature_func(df)
                if isinstance(new_features, pd.DataFrame):
                    features_df = pd.concat([features_df, new_features], axis=1)
                elif isinstance(new_features, pd.Series):
                    features_df[feature_name] = new_features
            except Exception as e:
                logger.warning(f"Erro gerando feature {feature_name} para {symbol}: {e}")
                continue
                
        return features_df
    
    def _price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum de preço"""
        features = pd.DataFrame(index=df.index)
        
        # Momentum de diferentes períodos
        for period in [3, 5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            features[f'momentum_acc_{period}'] = features[f'momentum_{period}'].diff()
        
        # Rate of Change
        features['roc_5'] = talib.ROC(df['close'], timeperiod=5)
        features['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        
        return features
    
    def _volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de perfil de volume"""
        features = pd.DataFrame(index=df.index)
        
        # Volume normalizado
        features['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume-Price Trend
        features['vpt'] = talib.OBV(df['close'], df['volume'])
        
        # Volume Rate of Change
        features['volume_roc'] = df['volume'].pct_change()
        
        # Volume Spike Detection
        features['volume_spike'] = (features['volume_norm'] > 2.0).astype(int)
        
        return features
    
    def _volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de regime de volatilidade"""
        features = pd.DataFrame(index=df.index)
        
        # ATR normalizado
        features['atr_norm'] = talib.ATR(df['high'], df['low'], df['close'], 14) / df['close']
        
        # Volatilidade realizada
        returns = df['close'].pct_change()
        features['realized_vol'] = returns.rolling(20).std()
        
        # Regime de volatilidade
        vol_percentile = features['realized_vol'].rolling(100).rank(pct=True)
        features['vol_regime'] = pd.cut(vol_percentile, bins=3, labels=[0, 1, 2])
        
        return features
    
    def _trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de força de tendência"""
        features = pd.DataFrame(index=df.index)
        
        # ADX
        features['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        
        # Trend consistency
        ema_fast = talib.EMA(df['close'], 10)
        ema_slow = talib.EMA(df['close'], 20)
        features['trend_direction'] = np.where(ema_fast > ema_slow, 1, -1)
        features['trend_consistency'] = features['trend_direction'].rolling(10).sum() / 10
        
        return features
    
    def _support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de suporte e resistência"""
        features = pd.DataFrame(index=df.index)
        
        # Distância de máximas/mínimas locais
        features['high_20'] = df['high'].rolling(20).max()
        features['low_20'] = df['low'].rolling(20).min()
        features['dist_to_high'] = (features['high_20'] - df['close']) / df['close']
        features['dist_to_low'] = (df['close'] - features['low_20']) / df['close']
        
        return features
    
    def _pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de reconhecimento de padrões"""
        features = pd.DataFrame(index=df.index)
        
        # Padrões de candlestick usando talib
        patterns = [
            'CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDLMORNINGSTAR',
            'CDLEVENINGSTAR', 'CDLHARAMI', 'CDLPIERCING', 'CDLDARKCLOUD'
        ]
        
        for pattern in patterns:
            try:
                features[pattern.lower()] = getattr(talib, pattern)(
                    df['open'], df['high'], df['low'], df['close']
                )
            except:
                continue
                
        return features
    
    def _cross_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de correlação cruzada"""
        features = pd.DataFrame(index=df.index)
        
        # Correlação entre preço e volume
        features['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # Correlação entre high-low spread e volume
        features['spread_volume_corr'] = (df['high'] - df['low']).rolling(20).corr(df['volume'])
        
        return features
    
    def _market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de microestrutura de mercado"""
        features = pd.DataFrame(index=df.index)
        
        # Spread bid-ask proxy
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Imbalance proxy
        features['imbalance_proxy'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Tick direction
        features['tick_direction'] = np.sign(df['close'].diff())
        
        return features

# =====================================================================
# MOTOR DE BACKTESTING PARALELO
# =====================================================================

class ParallelBacktester:
    """Backtester paralelo para testar múltiplas estratégias"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.feature_generator = DynamicFeatureGenerator()
    
    async def backtest_strategy(self, strategy: AutonomousStrategy, 
                              data: Dict[str, pd.DataFrame]) -> StrategyPerformance:
        """Backtesta uma estratégia específica"""
        
        # Executar backtest em thread separada para não bloquear
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_backtest_sync, strategy, data)
            performance = await loop.run_in_executor(None, future.result)
        
        return performance
    
    def _run_backtest_sync(self, strategy: AutonomousStrategy, 
                          data: Dict[str, pd.DataFrame]) -> StrategyPerformance:
        """Executa backtest sincronamente"""
        
        all_trades = []
        
        for symbol, df in data.items():
            if len(df) < 100:  # Dados insuficientes
                continue
                
            # Gerar features
            features_df = self.feature_generator.generate_features(df, symbol)
            
            # Executar estratégia
            trades = self._execute_strategy_on_data(strategy, features_df, symbol)
            all_trades.extend(trades)
        
        # Calcular performance
        performance = StrategyPerformance()
        performance.update_metrics(all_trades)
        
        return performance
    
    def _execute_strategy_on_data(self, strategy: AutonomousStrategy, 
                                 df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Executa estratégia nos dados históricos"""
        
        trades = []
        position = None
        
        # Calcular indicadores
        indicators = self._calculate_indicators(df, strategy.genes)
        
        for i in range(len(df)):
            if i < 50:  # Período de warm-up
                continue
                
            current_data = {
                'price': df['close'].iloc[i],
                'volume': df['volume'].iloc[i],
                'timestamp': df.index[i],
                'indicators': {k: v.iloc[i] if not pd.isna(v.iloc[i]) else 0 
                             for k, v in indicators.items()}
            }
            
            # Verificar sinal de entrada
            if position is None:
                signal = self._generate_signal(current_data, strategy.genes)
                if signal['action'] in ['BUY', 'SELL'] and signal['confidence'] >= strategy.genes.min_confidence:
                    position = {
                        'symbol': symbol,
                        'action': signal['action'],
                        'entry_price': current_data['price'],
                        'entry_time': current_data['timestamp'],
                        'stop_loss': current_data['price'] * (1 - strategy.genes.stop_loss) if signal['action'] == 'BUY' else current_data['price'] * (1 + strategy.genes.stop_loss),
                        'take_profit': current_data['price'] * (1 + strategy.genes.take_profit) if signal['action'] == 'BUY' else current_data['price'] * (1 - strategy.genes.take_profit),
                        'confidence': signal['confidence']
                    }
            
            # Verificar saída
            elif position is not None:
                should_exit, exit_reason = self._should_exit_position(position, current_data, strategy.genes)
                
                if should_exit:
                    # Calcular lucro/prejuízo
                    if position['action'] == 'BUY':
                        profit_pct = (current_data['price'] - position['entry_price']) / position['entry_price']
                    else:
                        profit_pct = (position['entry_price'] - current_data['price']) / position['entry_price']
                    
                    trade = {
                        'symbol': symbol,
                        'action': position['action'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_data['price'],
                        'entry_time': position['entry_time'],
                        'exit_time': current_data['timestamp'],
                        'profit': profit_pct,
                        'exit_reason': exit_reason,
                        'confidence': position['confidence']
                    }
                    
                    trades.append(trade)
                    position = None
        
        return trades
    
    def _calculate_indicators(self, df: pd.DataFrame, genes: StrategyGenes) -> Dict[str, pd.Series]:
        """Calcula indicadores técnicos"""
        indicators = {}
        
        # EMAs
        indicators['ema_fast'] = talib.EMA(df['close'], genes.ema_fast)
        indicators['ema_slow'] = talib.EMA(df['close'], genes.ema_slow)
        
        # RSI
        indicators['rsi'] = talib.RSI(df['close'], genes.rsi_period)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], genes.bb_period, genes.bb_std)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'], genes.macd_fast, genes.macd_slow, genes.macd_signal)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_hist'] = macd_hist
        
        # Volume
        indicators['volume_ma'] = talib.SMA(df['volume'], genes.volume_period)
        
        # Volatilidade
        indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        
        return indicators
    
    def _generate_signal(self, data: Dict, genes: StrategyGenes) -> Dict[str, Any]:
        """Gera sinal de trading baseado nos genes da estratégia"""
        
        indicators = data['indicators']
        price = data['price']
        
        signals = []
        confidence_factors = []
        
        # Sinal EMA Crossover
        if indicators['ema_fast'] > indicators['ema_slow']:
            signals.append('BUY')
            confidence_factors.append(0.3)
        elif indicators['ema_fast'] < indicators['ema_slow']:
            signals.append('SELL')
            confidence_factors.append(0.3)
        
        # Sinal RSI
        if indicators['rsi'] < genes.rsi_oversold:
            signals.append('BUY')
            confidence_factors.append(0.25)
        elif indicators['rsi'] > genes.rsi_overbought:
            signals.append('SELL')
            confidence_factors.append(0.25)
        
        # Sinal Bollinger Bands
        if price < indicators['bb_lower']:
            signals.append('BUY')
            confidence_factors.append(0.2)
        elif price > indicators['bb_upper']:
            signals.append('SELL')
            confidence_factors.append(0.2)
        
        # Sinal MACD
        if indicators['macd'] > indicators['macd_signal']:
            signals.append('BUY')
            confidence_factors.append(0.15)
        elif indicators['macd'] < indicators['macd_signal']:
            signals.append('SELL')
            confidence_factors.append(0.15)
        
        # Sinal de Volume
        if data['volume'] > indicators['volume_ma'] * genes.volume_threshold:
            confidence_factors.append(0.1)  # Adiciona confiança, não direção
        
        # Determinar ação predominante
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        
        if buy_signals > sell_signals:
            action = 'BUY'
            confidence = sum(confidence_factors) * (buy_signals / len(signals))
        elif sell_signals > buy_signals:
            action = 'SELL'
            confidence = sum(confidence_factors) * (sell_signals / len(signals))
        else:
            action = 'HOLD'
            confidence = 0.0
        
        return {
            'action': action,
            'confidence': min(1.0, confidence),
            'signals_count': len(signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def _should_exit_position(self, position: Dict, current_data: Dict, genes: StrategyGenes) -> Tuple[bool, str]:
        """Determina se deve sair da posição"""
        
        current_price = current_data['price']
        
        # Stop Loss
        if position['action'] == 'BUY' and current_price <= position['stop_loss']:
            return True, 'stop_loss'
        elif position['action'] == 'SELL' and current_price >= position['stop_loss']:
            return True, 'stop_loss'
        
        # Take Profit
        if position['action'] == 'BUY' and current_price >= position['take_profit']:
            return True, 'take_profit'
        elif position['action'] == 'SELL' and current_price <= position['take_profit']:
            return True, 'take_profit'
        
        # Sinal contrário
        signal = self._generate_signal(current_data, genes)
        if signal['confidence'] >= genes.min_confidence:
            if (position['action'] == 'BUY' and signal['action'] == 'SELL') or \
               (position['action'] == 'SELL' and signal['action'] == 'BUY'):
                return True, 'signal_reversal'
        
        return False, 'holding'

# =====================================================================
# ALGORITMO GENÉTICO PARA EVOLUÇÃO DE ESTRATÉGIAS
# =====================================================================

class StrategyEvolver:
    """Evolui estratégias usando algoritmo genético"""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15, 
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        
    def evolve_population(self, strategies: List[AutonomousStrategy]) -> List[AutonomousStrategy]:
        """Evolui uma população de estratégias"""
        
        # Calcular fitness de cada estratégia
        fitness_scores = [(strategy, strategy.calculate_fitness()) for strategy in strategies]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar os melhores (elitismo)
        elite_size = max(2, self.population_size // 10)
        elite = [strategy for strategy, _ in fitness_scores[:elite_size]]
        
        # Gerar nova população
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Seleção por torneio
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1
            
            # Mutação
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        self.generation += 1
        logger.info(f"🧬 Geração {self.generation} evoluída. Melhor fitness: {fitness_scores[0][1]:.4f}")
        
        return new_population
    
    def _tournament_selection(self, fitness_scores: List[Tuple[AutonomousStrategy, float]], 
                            tournament_size: int = 3) -> AutonomousStrategy:
        """Seleção por torneio"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return max(tournament, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1: AutonomousStrategy, parent2: AutonomousStrategy) -> AutonomousStrategy:
        """Crossover entre duas estratégias"""
        
        # Criar nova estratégia com ID único
        child_id = f"evolved_{int(time.time())}_{random.randint(1000, 9999)}"
        child_name = f"Evolved Gen{self.generation}"
        
        # Crossover dos genes
        child_genes = StrategyGenes()
        
        # Crossover de cada parâmetro
        child_genes.ema_fast = parent1.genes.ema_fast if random.random() < 0.5 else parent2.genes.ema_fast
        child_genes.ema_slow = parent1.genes.ema_slow if random.random() < 0.5 else parent2.genes.ema_slow
        child_genes.rsi_period = parent1.genes.rsi_period if random.random() < 0.5 else parent2.genes.rsi_period
        child_genes.rsi_oversold = parent1.genes.rsi_oversold if random.random() < 0.5 else parent2.genes.rsi_oversold
        child_genes.rsi_overbought = parent1.genes.rsi_overbought if random.random() < 0.5 else parent2.genes.rsi_overbought
        child_genes.bb_period = parent1.genes.bb_period if random.random() < 0.5 else parent2.genes.bb_period
        child_genes.bb_std = parent1.genes.bb_std if random.random() < 0.5 else parent2.genes.bb_std
        child_genes.volume_threshold = parent1.genes.volume_threshold if random.random() < 0.5 else parent2.genes.volume_threshold
        child_genes.min_confidence = parent1.genes.min_confidence if random.random() < 0.5 else parent2.genes.min_confidence
        child_genes.stop_loss = parent1.genes.stop_loss if random.random() < 0.5 else parent2.genes.stop_loss
        child_genes.take_profit = parent1.genes.take_profit if random.random() < 0.5 else parent2.genes.take_profit
        
        return AutonomousStrategy(
            id=child_id,
            name=child_name,
            genes=child_genes
        )
    
    def _mutate(self, strategy: AutonomousStrategy) -> AutonomousStrategy:
        """Mutação de uma estratégia"""
        
        # Criar cópia da estratégia
        mutated_id = f"mutated_{int(time.time())}_{random.randint(1000, 9999)}"
        mutated_strategy = AutonomousStrategy(
            id=mutated_id,
            name=f"Mutated {strategy.name}",
            genes=StrategyGenes(**strategy.genes.__dict__)
        )
        
        # Aplicar mutação
        mutated_strategy.genes.mutate(self.mutation_rate)
        
        return mutated_strategy

# =====================================================================
# SISTEMA DE DESCOBERTA DE MERCADO
# =====================================================================

class MarketDiscoveryEngine:
    """Sistema de descoberta e análise de mercado"""
    
    def __init__(self, data_collector, gate_api):
        self.data_collector = data_collector
        self.gate_api = gate_api
        
        # Cache de análises
        self.market_analysis_cache = {}
        self.correlation_matrix = pd.DataFrame()
        self.market_regimes = {}
        
        # Configurações
        self.min_volume_24h = 1_000_000  # $1M
        self.max_symbols_to_analyze = 50
        
    async def discover_market_opportunities(self) -> List[Dict[str, Any]]:
        """Descobre oportunidades de mercado dinamicamente"""
        
        logger.info("🔍 Iniciando descoberta de mercado...")
        
        # 1. Descobrir novos símbolos ativos
        active_symbols = await self._discover_active_symbols()
        
        # 2. Analisar padrões de mercado
        market_patterns = await self._analyze_market_patterns(active_symbols)
        
        # 3. Detectar correlações e anomalias
        correlations = await self._detect_correlations(active_symbols)
        
        # 4. Identificar regimes de mercado
        market_regimes = await self._identify_market_regimes(active_symbols)
        
        # 5. Consolidar oportunidades
        opportunities = self._consolidate_opportunities(market_patterns, correlations, market_regimes)
        
        logger.info(f"🎯 Descobertas {len(opportunities)} oportunidades de mercado")
        
        return opportunities
    
    async def _discover_active_symbols(self) -> List[str]:
        """Descobre símbolos ativos com base em volume e volatilidade"""
        
        try:
            # Obter todos os tickers
            tickers = await self.gate_api.get_futures_tickers()
            
            # Filtrar por volume e volatilidade
            active_symbols = []
            
            for ticker in tickers:
                try:
                    symbol = ticker.get('contract', '')
                    volume_24h = float(ticker.get('vol_usdt_24h', 0))
                    price_change = abs(float(ticker.get('change_percentage', 0)))
                    
                    if (volume_24h >= self.min_volume_24h and 
                        price_change >= 0.5 and  # Pelo menos 0.5% de movimento
                        '_USDT' in symbol):
                        
                        active_symbols.append(symbol)
                        
                except Exception as e:
                    continue
            
            # Limitar número de símbolos
            active_symbols = sorted(active_symbols, 
                                  key=lambda s: float(next(t['vol_usdt_24h'] for t in tickers if t.get('contract') == s)), 
                                  reverse=True)[:self.max_symbols_to_analyze]
            
            logger.info(f"📊 Descobertos {len(active_symbols)} símbolos ativos")
            return active_symbols
            
        except Exception as e:
            logger.error(f"❌ Erro descobrindo símbolos ativos: {e}")
            return []
    
    async def _analyze_market_patterns(self, symbols: List[str]) -> Dict[str, Any]:
        """Analisa padrões de mercado"""
        
        patterns = {
            'trending_up': [],
            'trending_down': [],
            'range_bound': [],
            'breakout_candidates': [],
            'volume_spikes': [],
            'volatility_expansion': []
        }
        
        for symbol in symbols:
            try:
                # Obter dados históricos
                klines = await self.data_collector.get_klines_data(symbol)
                if klines is None or len(klines) < 50:
                    continue
                
                # Analisar tendência
                trend = self._analyze_trend(klines)
                if trend == 'up':
                    patterns['trending_up'].append(symbol)
                elif trend == 'down':
                    patterns['trending_down'].append(symbol)
                else:
                    patterns['range_bound'].append(symbol)
                
                # Detectar breakouts potenciais
                if self._is_breakout_candidate(klines):
                    patterns['breakout_candidates'].append(symbol)
                
                # Detectar spikes de volume
                if self._has_volume_spike(klines):
                    patterns['volume_spikes'].append(symbol)
                
                # Detectar expansão de volatilidade
                if self._has_volatility_expansion(klines):
                    patterns['volatility_expansion'].append(symbol)
                
            except Exception as e:
                logger.warning(f"Erro analisando padrões para {symbol}: {e}")
                continue
        
        return patterns
    
    def _analyze_trend(self, df: pd.DataFrame) -> str:
        """Analisa tendência usando múltiplos indicadores"""
        
        # EMAs
        ema_20 = talib.EMA(df['close'], 20)
        ema_50 = talib.EMA(df['close'], 50)
        
        # Últimos valores
        current_ema20 = ema_20.iloc[-1]
        current_ema50 = ema_50.iloc[-1]
        prev_ema20 = ema_20.iloc[-10]
        prev_ema50 = ema_50.iloc[-10]
        
        # Condições de tendência
        ema_bullish = current_ema20 > current_ema50 and current_ema20 > prev_ema20
        ema_bearish = current_ema20 < current_ema50 and current_ema20 < prev_ema20
        
        # Price action
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        current_price = df['close'].iloc[-1]
        
        price_position = (current_price - recent_low) / (recent_high - recent_low)
        
        if ema_bullish and price_position > 0.6:
            return 'up'
        elif ema_bearish and price_position < 0.4:
            return 'down'
        else:
            return 'sideways'
    
    def _is_breakout_candidate(self, df: pd.DataFrame) -> bool:
        """Identifica candidatos a breakout"""
        
        # Volatilidade baixa seguida de contração
        atr = talib.ATR(df['high'], df['low'], df['close'], 14)
        atr_ratio = atr.iloc[-1] / atr.iloc[-20:].mean()
        
        # Bollinger Bands squeeze
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], 20, 2)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_squeeze = bb_width.iloc[-1] < bb_width.iloc[-20:].quantile(0.2)
        
        # Volume diminuindo
        volume_declining = df['volume'].iloc[-5:].mean() < df['volume'].iloc[-20:-5].mean()
        
        return atr_ratio < 0.8 and bb_squeeze and volume_declining
    
    def _has_volume_spike(self, df: pd.DataFrame) -> bool:
        """Detecta spikes de volume"""
        
        volume_ma = df['volume'].rolling(20).mean()
        recent_volume = df['volume'].iloc[-1]
        
        return recent_volume > volume_ma.iloc[-1] * 2.0
    
    def _has_volatility_expansion(self, df: pd.DataFrame) -> bool:
        """Detecta expansão de volatilidade"""
        
        atr = talib.ATR(df['high'], df['low'], df['close'], 14)
        atr_ratio = atr.iloc[-1] / atr.iloc[-20:].mean()
        
        return atr_ratio > 1.5
    
    async def _detect_correlations(self, symbols: List[str]) -> Dict[str, Any]:
        """Detecta correlações entre símbolos"""
        
        correlations = {
            'high_correlation_pairs': [],
            'negative_correlation_pairs': [],
            'correlation_matrix': None
        }
        
        try:
            # Coletar dados de preços
            price_data = {}
            for symbol in symbols:
                klines = await self.data_collector.get_klines_data(symbol)
                if klines is not None and len(klines) >= 100:
                    price_data[symbol] = klines['close'].pct_change().dropna()
            
            if len(price_data) < 2:
                return correlations
            
            # Criar DataFrame de retornos
            returns_df = pd.DataFrame(price_data)
            returns_df = returns_df.dropna()
            
            # Calcular matriz de correlação
            corr_matrix = returns_df.corr()
            correlations['correlation_matrix'] = corr_matrix
            
            # Encontrar pares com alta correlação
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.columns[i]
                    symbol2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if correlation > 0.7:
                        correlations['high_correlation_pairs'].append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation
                        })
                    elif correlation < -0.5:
                        correlations['negative_correlation_pairs'].append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation
                        })
            
            self.correlation_matrix = corr_matrix
            
        except Exception as e:
            logger.error(f"❌ Erro detectando correlações: {e}")
        
        return correlations
    
    async def _identify_market_regimes(self, symbols: List[str]) -> Dict[str, str]:
        """Identifica regimes de mercado para cada símbolo"""
        
        regimes = {}
        
        for symbol in symbols:
            try:
                klines = await self.data_collector.get_klines_data(symbol)
                if klines is None or len(klines) < 50:
                    continue
                
                # Análise de volatilidade
                returns = klines['close'].pct_change().dropna()
                vol_current = returns.tail(20).std()
                vol_historical = returns.std()
                
                # Análise de tendência
                trend = self._analyze_trend(klines)
                
                # Classificar regime
                if vol_current > vol_historical * 1.5:
                    if trend == 'up':
                        regime = 'high_vol_bull'
                    elif trend == 'down':
                        regime = 'high_vol_bear'
                    else:
                        regime = 'high_vol_sideways'
                else:
                    if trend == 'up':
                        regime = 'low_vol_bull'
                    elif trend == 'down':
                        regime = 'low_vol_bear'
                    else:
                        regime = 'low_vol_sideways'
                
                regimes[symbol] = regime
                
            except Exception as e:
                logger.warning(f"Erro identificando regime para {symbol}: {e}")
                continue
        
        self.market_regimes = regimes
        return regimes
    
    def _consolidate_opportunities(self, patterns: Dict, correlations: Dict, regimes: Dict) -> List[Dict]:
        """Consolida oportunidades descobertas"""
        
        opportunities = []
        
        # Oportunidades de breakout
        for symbol in patterns.get('breakout_candidates', []):
            opportunities.append({
                'type': 'breakout',
                'symbol': symbol,
                'regime': regimes.get(symbol, 'unknown'),
                'confidence': 0.7
            })
        
        # Oportunidades de momentum
        for symbol in patterns.get('volume_spikes', []):
            if symbol in patterns.get('trending_up', []):
                opportunities.append({
                    'type': 'momentum_long',
                    'symbol': symbol,
                    'regime': regimes.get(symbol, 'unknown'),
                    'confidence': 0.8
                })
            elif symbol in patterns.get('trending_down', []):
                opportunities.append({
                    'type': 'momentum_short',
                    'symbol': symbol,
                    'regime': regimes.get(symbol, 'unknown'),
                    'confidence': 0.8
                })
        
        # Oportunidades de correlação
        for pair in correlations.get('high_correlation_pairs', []):
            if pair['correlation'] > 0.8:
                opportunities.append({
                    'type': 'correlation_trade',
                    'symbol1': pair['symbol1'],
                    'symbol2': pair['symbol2'],
                    'correlation': pair['correlation'],
                    'confidence': 0.6
                })
        
        # Oportunidades de reversão
        for symbol in patterns.get('volatility_expansion', []):
            if regimes.get(symbol, '').endswith('_bear'):
                opportunities.append({
                    'type': 'mean_reversion',
                    'symbol': symbol,
                    'regime': regimes.get(symbol, 'unknown'),
                    'confidence': 0.5
                })
        
        return opportunities

# =====================================================================
# SISTEMA PRINCIPAL AUTÔNOMO
# =====================================================================

class AutonomousTradingSystem:
    """Sistema de trading verdadeiramente autônomo"""
    
    def __init__(self, data_collector, portfolio_manager, gate_api):
        self.data_collector = data_collector
        self.portfolio_manager = portfolio_manager
        self.gate_api = gate_api
        
        # Componentes principais
        self.backtester = ParallelBacktester(max_workers=4)
        self.evolver = StrategyEvolver(population_size=15)
        self.market_discovery = MarketDiscoveryEngine(data_collector, gate_api)
        
        # População de estratégias
        self.strategy_population = []
        self.active_strategies = []
        
        # Configurações
        self.max_active_strategies = 5
        self.evolution_frequency = 24  # horas
        self.discovery_frequency = 6   # horas
        
        # Estado
        self.last_evolution = None
        self.last_discovery = None
        self.system_stats = {
            'total_strategies_created': 0,
            'total_evolutions': 0,
            'total_discoveries': 0,
            'best_strategy_fitness': 0.0
        }
        
        # Dados para análise
        self.historical_data = {}
        self.market_data_buffer = {}
        
    async def initialize(self):
        """Inicializa o sistema autônomo"""
        
        logger.info("🚀 Inicializando Sistema de Trading Autônomo...")
        
        # Criar população inicial
        await self._create_initial_population()
        
        # Descobrir oportunidades de mercado
        await self._initial_market_discovery()
        
        # Carregar dados históricos
        await self._load_historical_data()
        
        logger.info("✅ Sistema Autônomo inicializado com sucesso!")
    
    async def _create_initial_population(self):
        """Cria população inicial de estratégias"""
        
        logger.info("🧬 Criando população inicial de estratégias...")
        
        # Criar estratégias com diferentes características
        strategy_templates = [
            {"name": "Momentum Trader", "focus": "momentum"},
            {"name": "Mean Reverter", "focus": "mean_reversion"},
            {"name": "Breakout Hunter", "focus": "breakout"},
            {"name": "Scalper", "focus": "scalping"},
            {"name": "Trend Follower", "focus": "trend_following"},
            {"name": "Volatility Trader", "focus": "volatility"},
            {"name": "Volume Trader", "focus": "volume"},
            {"name": "Multi-Timeframe", "focus": "multi_tf"},
        ]
        
        for i, template in enumerate(strategy_templates):
            for variation in range(3):  # 3 variações de cada template
                strategy_id = f"{template['focus']}_{i}_{variation}"
                
                # Criar genes específicos para o foco
                genes = self._create_focused_genes(template['focus'])
                
                strategy = AutonomousStrategy(
                    id=strategy_id,
                    name=f"{template['name']} v{variation+1}",
                    genes=genes
                )
                
                self.strategy_population.append(strategy)
                self.system_stats['total_strategies_created'] += 1
        
        # Adicionar algumas estratégias completamente aleatórias
        for i in range(5):
            random_strategy = AutonomousStrategy(
                id=f"random_{i}",
                name=f"Random Strategy {i+1}",
                genes=StrategyGenes()  # Genes aleatórios
            )
            self.strategy_population.append(random_strategy)
            self.system_stats['total_strategies_created'] += 1
        
        logger.info(f"🧬 Criadas {len(self.strategy_population)} estratégias iniciais")
    
    def _create_focused_genes(self, focus: str) -> StrategyGenes:
        """Cria genes focados em um tipo específico de estratégia"""
        
        genes = StrategyGenes()
        
        if focus == "momentum":
            genes.ema_fast = random.randint(5, 10)
            genes.ema_slow = random.randint(15, 25)
            genes.rsi_period = random.randint(10, 14)
            genes.min_confidence = random.uniform(0.6, 0.8)
            genes.take_profit = random.uniform(0.02, 0.04)
            
        elif focus == "mean_reversion":
            genes.rsi_period = random.randint(20, 30)
            genes.rsi_oversold = random.uniform(15, 25)
            genes.rsi_overbought = random.uniform(75, 85)
            genes.bb_period = random.randint(25, 35)
            genes.min_confidence = random.uniform(0.5, 0.7)
            
        elif focus == "breakout":
            genes.bb_period = random.randint(20, 30)
            genes.bb_std = random.uniform(2.0, 2.5)
            genes.volume_threshold = random.uniform(1.5, 2.5)
            genes.min_confidence = random.uniform(0.7, 0.9)
            genes.take_profit = random.uniform(0.03, 0.06)
            
        elif focus == "scalping":
            genes.ema_fast = random.randint(3, 8)
            genes.ema_slow = random.randint(12, 18)
            genes.min_confidence = random.uniform(0.4, 0.6)
            genes.take_profit = random.uniform(0.005, 0.015)
            genes.stop_loss = random.uniform(0.005, 0.015)
            
        elif focus == "trend_following":
            genes.ema_fast = random.randint(12, 18)
            genes.ema_slow = random.randint(30, 50)
            genes.adx_period = random.randint(12, 18)
            genes.min_confidence = random.uniform(0.6, 0.8)
            genes.take_profit = random.uniform(0.025, 0.05)
            
        elif focus == "volatility":
            genes.bb_period = random.randint(15, 25)
            genes.bb_std = random.uniform(1.8, 2.2)
            genes.min_volatility = random.uniform(0.01, 0.02)
            genes.max_volatility = random.uniform(0.05, 0.1)
            
        elif focus == "volume":
            genes.volume_period = random.randint(15, 25)
            genes.volume_threshold = random.uniform(1.8, 2.8)
            genes.min_confidence = random.uniform(0.5, 0.7)
            
        return genes
    
    async def _initial_market_discovery(self):
        """Descoberta inicial de mercado"""
        
        logger.info("🔍 Realizando descoberta inicial de mercado...")
        
        opportunities = await self.market_discovery.discover_market_opportunities()
        
        logger.info(f"🎯 Descobertas {len(opportunities)} oportunidades iniciais")
        
        self.last_discovery = datetime.now()
        self.system_stats['total_discoveries'] += 1
    
    async def _load_historical_data(self):
        """Carrega dados históricos para backtesting"""
        
        logger.info("📊 Carregando dados históricos...")
        
        # Obter símbolos ativos
        active_symbols = await self.market_discovery._discover_active_symbols()
        
        # Carregar dados históricos
        for symbol in active_symbols[:20]:  # Limitar para não sobrecarregar
            try:
                klines = await self.data_collector.get_klines_data(symbol)
                if klines is not None and len(klines) >= 200:
                    self.historical_data[symbol] = klines
            except Exception as e:
                logger.warning(f"Erro carregando dados para {symbol}: {e}")
        
        logger.info(f"📊 Dados históricos carregados para {len(self.historical_data)} símbolos")
    
    async def run_autonomous_cycle(self):
        """Executa um ciclo completo do sistema autônomo"""
        
        logger.info("🔄 Executando ciclo autônomo...")
        
        # 1. Avaliar performance das estratégias ativas
        await self._evaluate_active_strategies()
        
        # 2. Descobrir novas oportunidades de mercado
        if self._should_discover_market():
            await self._discover_market_opportunities()
        
        # 3. Evoluir população de estratégias
        if self._should_evolve_strategies():
            await self._evolve_strategy_population()
        
        # 4. Selecionar estratégias ativas
        await self._select_active_strategies()
        
        # 5. Executar trades baseado nas estratégias ativas
        await self._execute_strategy_trades()
        
        # 6. Atualizar estatísticas
        self._update_system_stats()
        
        logger.info("✅ Ciclo autônomo concluído")
    
    async def _evaluate_active_strategies(self):
        """Avalia performance das estratégias ativas"""
        
        logger.info("📊 Avaliando performance das estratégias ativas...")
        
        for strategy in self.active_strategies:
            try:
                # Atualizar dados de performance
                if strategy.trade_history:
                    strategy.performance.update_metrics(strategy.trade_history)
                
                # Backtesting com dados mais recentes
                if self.historical_data:
                    recent_performance = await self.backtester.backtest_strategy(
                        strategy, self.historical_data
                    )
                    
                    # Atualizar performance
                    strategy.performance = recent_performance
                    
                logger.debug(f"Strategy {strategy.name}: Fitness {strategy.calculate_fitness():.4f}")
                
            except Exception as e:
                logger.error(f"Erro avaliando estratégia {strategy.name}: {e}")
    
    def _should_discover_market(self) -> bool:
        """Determina se deve fazer descoberta de mercado"""
        
        if self.last_discovery is None:
            return True
        
        hours_since_last = (datetime.now() - self.last_discovery).total_seconds() / 3600
        return hours_since_last >= self.discovery_frequency
    
    def _should_evolve_strategies(self) -> bool:
        """Determina se deve evoluir estratégias"""
        
        if self.last_evolution is None:
            return True
        
        hours_since_last = (datetime.now() - self.last_evolution).total_seconds() / 3600
        return hours_since_last >= self.evolution_frequency
    
    async def _discover_market_opportunities(self):
        """Descobre novas oportunidades de mercado"""
        
        logger.info("🔍 Descobrindo novas oportunidades de mercado...")
        
        opportunities = await self.market_discovery.discover_market_opportunities()
        
        # Criar estratégias específicas para oportunidades descobertas
        for opportunity in opportunities:
            if opportunity.get('confidence', 0) > 0.6:
                await self._create_opportunity_strategy(opportunity)
        
        self.last_discovery = datetime.now()
        self.system_stats['total_discoveries'] += 1
    
    async def _create_opportunity_strategy(self, opportunity: Dict):
        """Cria estratégia específica para uma oportunidade"""
        
        strategy_type = opportunity.get('type', 'generic')
        symbol = opportunity.get('symbol', 'UNKNOWN')
        
        # Criar genes específicos para a oportunidade
        genes = StrategyGenes()
        
        if strategy_type == 'breakout':
            genes.bb_period = random.randint(18, 25)
            genes.volume_threshold = random.uniform(1.8, 2.5)
            genes.min_confidence = 0.7
            
        elif strategy_type == 'momentum_long':
            genes.ema_fast = random.randint(8, 12)
            genes.ema_slow = random.randint(20, 30)
            genes.rsi_period = random.randint(12, 16)
            genes.min_confidence = 0.6
            
        # Criar estratégia
        strategy_id = f"opportunity_{strategy_type}_{symbol}_{int(time.time())}"
        strategy = AutonomousStrategy(
            id=strategy_id,
            name=f"Opportunity {strategy_type.title()} for {symbol}",
            genes=genes
        )
        
        # Adicionar à população
        self.strategy_population.append(strategy)
        self.system_stats['total_strategies_created'] += 1
        
        logger.info(f"🎯 Criada estratégia para oportunidade {strategy_type} em {symbol}")
    
    async def _evolve_strategy_population(self):
        """Evolui a população de estratégias"""
        
        logger.info("🧬 Evoluindo população de estratégias...")
        
        # Avaliar todas as estratégias
        for strategy in self.strategy_population:
            if strategy.trade_history:
                strategy.performance.update_metrics(strategy.trade_history)
        
        # Evoluir população
        self.strategy_population = self.evolver.evolve_population(self.strategy_population)
        
        self.last_evolution = datetime.now()
        self.system_stats['total_evolutions'] += 1
        
        # Atualizar melhor fitness
        fitness_scores = [s.calculate_fitness() for s in self.strategy_population]
        self.system_stats['best_strategy_fitness'] = max(fitness_scores) if fitness_scores else 0.0
        
        logger.info(f"🧬 População evoluída. Melhor fitness: {self.system_stats['best_strategy_fitness']:.4f}")
    
    async def _select_active_strategies(self):
        """Seleciona estratégias ativas para trading"""
        
        # Calcular fitness de todas as estratégias
        strategy_fitness = [(s, s.calculate_fitness()) for s in self.strategy_population]
        strategy_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar as melhores
        self.active_strategies = [s for s, _ in strategy_fitness[:self.max_active_strategies]]
        
        # Log das estratégias selecionadas
        logger.info("🎯 Estratégias ativas selecionadas:")
        for i, strategy in enumerate(self.active_strategies):
            fitness = strategy.calculate_fitness()
            logger.info(f"  {i+1}. {strategy.name} (Fitness: {fitness:.4f})")
    
    async def _execute_strategy_trades(self):
        """Executa trades baseado nas estratégias ativas"""
        
        logger.info("💼 Executando trades das estratégias ativas...")
        
        # Obter dados de mercado atuais
        current_market_data = await self._get_current_market_data()
        
        for strategy in self.active_strategies:
            try:
                # Gerar sinais para a estratégia
                signals = await self._generate_strategy_signals(strategy, current_market_data)
                
                # Executar trades baseado nos sinais
                for signal in signals:
                    if signal['confidence'] >= strategy.genes.min_confidence:
                        await self._execute_signal_trade(signal, strategy)
                        
            except Exception as e:
                logger.error(f"Erro executando trades para {strategy.name}: {e}")
    
    async def _get_current_market_data(self) -> Dict[str, pd.DataFrame]:
        """Obtém dados de mercado atuais"""
        
        current_data = {}
        
        # Obter dados dos símbolos ativos
        for symbol in list(self.historical_data.keys())[:10]:  # Limitar para performance
            try:
                klines = await self.data_collector.get_klines_data(symbol)
                if klines is not None and len(klines) >= 50:
                    current_data[symbol] = klines.tail(100)  # Últimas 100 barras
            except Exception as e:
                logger.warning(f"Erro obtendo dados para {symbol}: {e}")
        
        return current_data
    
    async def _generate_strategy_signals(self, strategy: AutonomousStrategy, 
                                       market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Gera sinais para uma estratégia específica"""
        
        signals = []
        
        for symbol, df in market_data.items():
            try:
                # Calcular indicadores
                indicators = self.backtester._calculate_indicators(df, strategy.genes)
                
                # Obter último ponto de dados
                last_data = {
                    'price': df['close'].iloc[-1],
                    'volume': df['volume'].iloc[-1],
                    'timestamp': df.index[-1],
                    'indicators': {k: v.iloc[-1] if not pd.isna(v.iloc[-1]) else 0 
                                 for k, v in indicators.items()}
                }
                
                # Gerar sinal
                signal = self.backtester._generate_signal(last_data, strategy.genes)
                
                if signal['action'] != 'HOLD':
                    signals.append({
                        'symbol': symbol,
                        'action': signal['action'],
                        'confidence': signal['confidence'],
                        'price': last_data['price'],
                        'strategy_id': strategy.id,
                        'timestamp': last_data['timestamp']
                    })
                    
            except Exception as e:
                logger.warning(f"Erro gerando sinal para {symbol}: {e}")
        
        return signals
    
    async def _execute_signal_trade(self, signal: Dict, strategy: AutonomousStrategy):
        """Executa trade baseado em sinal"""
        
        symbol = signal['symbol']
        action = signal['action']
        confidence = signal['confidence']
        
        logger.info(f"📤 Executando trade: {action} {symbol} (Confiança: {confidence:.2f})")
        
        try:
            # Verificar se pode abrir nova posição
            can_trade = await self.portfolio_manager.can_open_new_position_overall()
            
            if not can_trade:
                logger.warning(f"❌ Não pode executar trade para {symbol}: Limites de risco")
                return
            
            # Executar trade via portfolio manager
            trade_signal = {
                'action': action,
                'price': signal['price'],
                'confidence': confidence,
                'strategy_id': strategy.id
            }
            
            result = await self.portfolio_manager.execute_trade(symbol, trade_signal)
            
            if result.get('success'):
                logger.info(f"✅ Trade executado com sucesso: {symbol}")
                
                # Registrar trade na estratégia
                trade_record = {
                    'symbol': symbol,
                    'action': action,
                    'price': signal['price'],
                    'confidence': confidence,
                    'timestamp': signal['timestamp'],
                    'strategy_id': strategy.id
                }
                
                strategy.trade_history.append(trade_record)
                strategy.last_trade_time = datetime.now()
                
            else:
                logger.warning(f"❌ Falha ao executar trade: {result.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            logger.error(f"❌ Erro executando trade para {symbol}: {e}")
    
    def _update_system_stats(self):
        """Atualiza estatísticas do sistema"""
        
        # Calcular estatísticas das estratégias
        active_strategies_count = len(self.active_strategies)
        total_trades = sum(len(s.trade_history) for s in self.strategy_population)
        
        # Atualizar stats
        self.system_stats.update({
            'active_strategies_count': active_strategies_count,
            'total_strategies_in_population': len(self.strategy_population),
            'total_trades_executed': total_trades,
            'last_cycle_time': datetime.now().isoformat()
        })
        
        logger.info(f"📊 Stats: {active_strategies_count} estratégias ativas, "
                   f"{len(self.strategy_population)} na população, "
                   f"{total_trades} trades executados")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        
        return {
            'system_stats': self.system_stats,
            'active_strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'fitness': s.calculate_fitness(),
                    'total_trades': len(s.trade_history),
                    'last_trade': s.last_trade_time.isoformat() if s.last_trade_time else None
                }
                for s in self.active_strategies
            ],
            'population_size': len(self.strategy_population),
            'last_evolution': self.last_evolution.isoformat() if self.last_evolution else None,
            'last_discovery': self.last_discovery.isoformat() if self.last_discovery else None,
            'market_opportunities': len(self.market_discovery.market_analysis_cache)
        }
    
    async def save_system_state(self, filepath: str):
        """Salva estado do sistema"""
        
        state = {
            'strategy_population': [
                {
                    'id': s.id,
                    'name': s.name,
                    'genes': s.genes.__dict__,
                    'performance': s.performance.__dict__,
                    'trade_history': s.trade_history,
                    'creation_date': s.creation_date.isoformat(),
                    'active': s.active
                }
                for s in self.strategy_population
            ],
            'system_stats': self.system_stats,
            'last_evolution': self.last_evolution.isoformat() if self.last_evolution else None,
            'last_discovery': self.last_discovery.isoformat() if self.last_discovery else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"💾 Estado do sistema salvo em {filepath}")
    
    async def load_system_state(self, filepath: str):
        """Carrega estado do sistema"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Reconstruir estratégias
            self.strategy_population = []
            for s_data in state.get('strategy_population', []):
                strategy = AutonomousStrategy(
                    id=s_data['id'],
                    name=s_data['name'],
                    genes=StrategyGenes(**s_data['genes']),
                    creation_date=datetime.fromisoformat(s_data['creation_date']),
                    active=s_data['active']
                )
                
                # Reconstruir performance
                performance_data = s_data.get('performance', {})
                strategy.performance = StrategyPerformance(**performance_data)
                
                # Reconstruir histórico
                strategy.trade_history = s_data.get('trade_history', [])
                
                self.strategy_population.append(strategy)
            
            # Reconstruir stats
            self.system_stats = state.get('system_stats', {})
            
            # Reconstruir datas
            if state.get('last_evolution'):
                self.last_evolution = datetime.fromisoformat(state['last_evolution'])
            if state.get('last_discovery'):
                self.last_discovery = datetime.fromisoformat(state['last_discovery'])
            
            logger.info(f"📂 Estado do sistema carregado de {filepath}")
            logger.info(f"🔄 {len(self.strategy_population)} estratégias carregadas")
            
        except Exception as e:
            logger.error(f"❌ Erro carregando estado do sistema: {e}")