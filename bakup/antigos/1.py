#!/usr/bin/env python3
"""
Analisador de Mercado Avançado - Sistema de Análise Inteligente
Capaz de detectar padrões complexos e prever movimentos de mercado
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('advanced_market_analyzer')

# =====================================================================
# CLASSES DE DADOS PARA ANÁLISE
# =====================================================================

@dataclass
class MarketRegimeData:
    """Dados de regime de mercado"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float
    volatility_percentile: float
    trend_strength: float
    momentum_score: float
    volume_profile: str  # 'high', 'normal', 'low'
    correlation_breakdown: bool
    regime_duration_days: int
    expected_duration_days: int
    
@dataclass
class AnomalyDetection:
    """Detecção de anomalias"""
    anomaly_score: float
    anomaly_type: str  # 'price', 'volume', 'volatility', 'correlation'
    significance_level: float
    description: str
    trading_opportunity: bool
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class MarketMicrostructure:
    """Análise de microestrutura de mercado"""
    bid_ask_spread_proxy: float
    price_impact_estimate: float
    order_flow_imbalance: float
    liquidity_score: float
    manipulation_probability: float
    optimal_entry_size: float
    slippage_estimate: float

@dataclass
class CycleAnalysis:
    """Análise de ciclos de mercado"""
    dominant_cycle_period: int
    cycle_strength: float
    cycle_phase: str  # 'accumulation', 'markup', 'distribution', 'markdown'
    next_phase_probability: float
    cycle_reliability: float

# =====================================================================
# ANALISADOR DE PADRÕES AVANÇADO
# =====================================================================

class AdvancedPatternAnalyzer:
    """Analisador de padrões complexos de mercado"""
    
    def __init__(self):
        self.pattern_library = self._initialize_pattern_library()
        self.ml_models = {}
        self.scalers = {}
        
    def _initialize_pattern_library(self) -> Dict[str, Dict]:
        """Inicializa biblioteca de padrões"""
        return {
            'wyckoff_accumulation': {
                'description': 'Padrão de acumulação Wyckoff',
                'timeframe': 'daily',
                'reliability': 0.75,
                'min_bars': 50
            },
            'elliott_wave': {
                'description': 'Ondas de Elliott',
                'timeframe': 'multiple',
                'reliability': 0.65,
                'min_bars': 100
            },
            'head_shoulders': {
                'description': 'Ombro-Cabeça-Ombro',
                'timeframe': 'multiple',
                'reliability': 0.70,
                'min_bars': 30
            },
            'diamond_pattern': {
                'description': 'Padrão Diamante',
                'timeframe': 'multiple',
                'reliability': 0.68,
                'min_bars': 25
            },
            'cup_handle': {
                'description': 'Xícara com Alça',
                'timeframe': 'daily',
                'reliability': 0.72,
                'min_bars': 40
            },
            'harmonic_patterns': {
                'description': 'Padrões Harmônicos (Gartley, Butterfly, etc)',
                'timeframe': 'multiple',
                'reliability': 0.78,
                'min_bars': 20
            }
        }
    
    def detect_complex_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detecta padrões complexos nos dados"""
        
        patterns_found = []
        
        if len(df) < 50:
            return patterns_found
        
        try:
            # Detectar padrões Wyckoff
            wyckoff_pattern = self._detect_wyckoff_patterns(df)
            if wyckoff_pattern:
                patterns_found.append(wyckoff_pattern)
            
            # Detectar Head & Shoulders
            hs_pattern = self._detect_head_shoulders(df)
            if hs_pattern:
                patterns_found.append(hs_pattern)
            
            # Detectar padrões harmônicos
            harmonic_patterns = self._detect_harmonic_patterns(df)
            patterns_found.extend(harmonic_patterns)
            
            # Detectar padrões de volume
            volume_patterns = self._detect_volume_patterns(df)
            patterns_found.extend(volume_patterns)
            
            # Detectar divergências
            divergences = self._detect_divergences(df)
            patterns_found.extend(divergences)
            
        except Exception as e:
            logger.warning(f"Erro detectando padrões para {symbol}: {e}")
        
        return patterns_found
    
    def _detect_wyckoff_patterns(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta padrões de acumulação/distribuição Wyckoff"""
        
        if len(df) < 100:
            return None
        
        # Calcular volume médio e range de preços
        volume_ma = df['volume'].rolling(50).mean()
        price_range = df['high'] - df['low']
        
        # Identificar fases do ciclo Wyckoff
        recent_data = df.tail(50)
        
        # Fase 1: Parada da tendência anterior (alta volatilidade, alto volume)
        high_vol_periods = recent_data['volume'] > volume_ma.tail(50) * 1.5
        high_volatility = price_range.tail(50) > price_range.rolling(20).mean().tail(50) * 1.3
        
        # Fase 2: Acumulação/Distribuição (baixa volatilidade, volume decrescente)
        low_vol_periods = recent_data['volume'] < volume_ma.tail(50) * 0.8
        low_volatility = price_range.tail(50) < price_range.rolling(20).mean().tail(50) * 0.8
        
        # Detectar se estamos em acumulação ou distribuição
        price_trend = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        volume_trend = recent_data['volume'].tail(10).mean() - recent_data['volume'].head(10).mean()
        
        # Critérios para padrão válido
        if (high_vol_periods.sum() > 5 and low_vol_periods.sum() > 10):
            if price_trend > 0 and volume_trend < 0:
                pattern_type = 'wyckoff_accumulation'
                confidence = 0.75
            elif price_trend < 0 and volume_trend > 0:
                pattern_type = 'wyckoff_distribution'
                confidence = 0.70
            else:
                return None
            
            return {
                'pattern': pattern_type,
                'confidence': confidence,
                'timeframe': '1h',
                'description': f'Padrão {pattern_type} detectado',
                'target_move': abs(price_trend) * 1.5,
                'invalidation_level': recent_data['low'].min() if pattern_type == 'wyckoff_accumulation' else recent_data['high'].max()
            }
        
        return None
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detecta padrão Head & Shoulders"""
        
        if len(df) < 50:
            return None
        
        # Encontrar picos e vales
        highs = df['high'].rolling(5, center=True).max() == df['high']
        lows = df['low'].rolling(5, center=True).min() == df['low']
        
        # Obter últimos picos significativos
        recent_highs = df[highs].tail(5)
        recent_lows = df[lows].tail(4)
        
        if len(recent_highs) < 3 or len(recent_lows) < 2:
            return None
        
        # Verificar estrutura H&S
        peak_prices = recent_highs['high'].values
        valley_prices = recent_lows['low'].values
        
        # Critérios básicos para H&S
        if len(peak_prices) >= 3:
            left_shoulder = peak_prices[-3]
            head = peak_prices[-2]
            right_shoulder = peak_prices[-1]
            
            # Head deve ser maior que os ombros
            if head > left_shoulder and head > right_shoulder:
                # Ombros devem ser aproximadamente iguais (±5%)
                shoulder_diff = abs(left_shoulder - right_shoulder) / head
                
                if shoulder_diff < 0.05:
                    # Calcular neckline
                    if len(valley_prices) >= 2:
                        neckline = (valley_prices[-2] + valley_prices[-1]) / 2
                        
                        # Target de preço
                        head_to_neckline = head - neckline
                        target_price = neckline - head_to_neckline
                        
                        return {
                            'pattern': 'head_shoulders',
                            'confidence': 0.72,
                            'timeframe': '1h',
                            'description': 'Padrão Head & Shoulders detectado',
                            'neckline': neckline,
                            'target_price': target_price,
                            'invalidation_level': head
                        }
        
        return None
    
    def _detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta padrões harmônicos (Gartley, Butterfly, etc)"""
        
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Encontrar pivots
        pivots = self._find_pivots(df)
        
        if len(pivots) < 5:
            return patterns
        
        # Verificar ratios de Fibonacci para padrões harmônicos
        for i in range(len(pivots) - 4):
            try:
                X, A, B, C, D = pivots[i:i+5]
                
                # Calcular ratios
                XA = abs(A['price'] - X['price'])
                AB = abs(B['price'] - A['price'])
                BC = abs(C['price'] - B['price'])
                CD = abs(D['price'] - C['price'])
                
                if XA == 0 or AB == 0 or BC == 0:
                    continue
                
                AB_XA = AB / XA
                BC_AB = BC / AB
                CD_BC = CD / BC
                
                # Verificar padrão Gartley
                if (0.58 <= AB_XA <= 0.65 and 
                    0.38 <= BC_AB <= 0.90 and 
                    1.25 <= CD_BC <= 1.65):
                    
                    patterns.append({
                        'pattern': 'gartley',
                        'confidence': 0.78,
                        'timeframe': '1h',
                        'description': 'Padrão Gartley detectado',
                        'entry_level': D['price'],
                        'target_levels': [
                            C['price'] + (C['price'] - D['price']) * 0.618,
                            C['price'] + (C['price'] - D['price']) * 1.0
                        ],
                        'stop_loss': D['price'] + (D['price'] - C['price']) * 0.382
                    })
                
                # Verificar padrão Butterfly
                elif (0.78 <= AB_XA <= 0.82 and 
                      0.38 <= BC_AB <= 0.90 and 
                      1.60 <= CD_BC <= 2.20):
                    
                    patterns.append({
                        'pattern': 'butterfly',
                        'confidence': 0.75,
                        'timeframe': '1h',
                        'description': 'Padrão Butterfly detectado',
                        'entry_level': D['price'],
                        'target_levels': [
                            C['price'] + (C['price'] - D['price']) * 0.382,
                            C['price'] + (C['price'] - D['price']) * 0.618
                        ],
                        'stop_loss': D['price'] + (D['price'] - C['price']) * 0.236
                    })
                    
            except Exception as e:
                continue
        
        return patterns
    
    def _find_pivots(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Encontra pontos de pivot (máximos e mínimos locais)"""
        
        pivots = []
        
        # Encontrar máximos locais
        for i in range(window, len(df) - window):
            is_high = all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1))
            is_high = is_high and all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1))
            
            if is_high:
                pivots.append({
                    'type': 'high',
                    'price': df['high'].iloc[i],
                    'index': i,
                    'timestamp': df.index[i]
                })
        
        # Encontrar mínimos locais
        for i in range(window, len(df) - window):
            is_low = all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1))
            is_low = is_low and all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1))
            
            if is_low:
                pivots.append({
                    'type': 'low',
                    'price': df['low'].iloc[i],
                    'index': i,
                    'timestamp': df.index[i]
                })
        
        # Ordenar por índice
        pivots.sort(key=lambda x: x['index'])
        
        return pivots
    
    def _detect_volume_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta padrões de volume"""
        
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        volume_ma = df['volume'].rolling(20).mean()
        price_change = df['close'].pct_change()
        
        # Volume Climax (alto volume com pequena mudança de preço)
        recent_volume = df['volume'].tail(5)
        recent_price_change = abs(price_change.tail(5))
        
        high_volume_low_movement = (
            (recent_volume > volume_ma.tail(5) * 2.0) & 
            (recent_price_change < 0.02)
        )
        
        if high_volume_low_movement.sum() >= 2:
            patterns.append({
                'pattern': 'volume_climax',
                'confidence': 0.65,
                'timeframe': '1h',
                'description': 'Volume Climax - possível reversão',
                'implication': 'reversal_warning'
            })
        
        # Volume Dry Up (volume diminuindo durante movimento)
        volume_trend = df['volume'].tail(10).rolling(5).mean()
        volume_declining = volume_trend.diff() < 0
        
        if volume_declining.sum() >= 4:
            patterns.append({
                'pattern': 'volume_dry_up',
                'confidence': 0.60,
                'timeframe': '1h',
                'description': 'Volume secando - movimento perdendo força',
                'implication': 'trend_weakening'
            })
        
        return patterns
    
    def _detect_divergences(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta divergências entre preço e indicadores"""
        
        divergences = []
        
        if len(df) < 50:
            return divergences
        
        # Calcular RSI
        rsi = talib.RSI(df['close'], timeperiod=14)
        
        # Calcular MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        
        # Encontrar divergências RSI
        price_peaks = df['high'].rolling(10, center=True).max() == df['high']
        rsi_peaks = rsi.rolling(10, center=True).max() == rsi
        
        price_valleys = df['low'].rolling(10, center=True).min() == df['low']
        rsi_valleys = rsi.rolling(10, center=True).min() == rsi
        
        # Verificar divergência bearish (preço subindo, RSI descendo)
        recent_price_peaks = df[price_peaks].tail(3)
        recent_rsi_peaks = rsi[rsi_peaks].tail(3)
        
        if len(recent_price_peaks) >= 2 and len(recent_rsi_peaks) >= 2:
            if (recent_price_peaks['high'].iloc[-1] > recent_price_peaks['high'].iloc[-2] and
                recent_rsi_peaks.iloc[-1] < recent_rsi_peaks.iloc[-2]):
                
                divergences.append({
                    'pattern': 'bearish_divergence_rsi',
                    'confidence': 0.68,
                    'timeframe': '1h',
                    'description': 'Divergência bearish RSI detectada',
                    'implication': 'potential_reversal_down'
                })
        
        # Verificar divergência bullish (preço descendo, RSI subindo)
        recent_price_valleys = df[price_valleys].tail(3)
        recent_rsi_valleys = rsi[rsi_valleys].tail(3)
        
        if len(recent_price_valleys) >= 2 and len(recent_rsi_valleys) >= 2:
            if (recent_price_valleys['low'].iloc[-1] < recent_price_valleys['low'].iloc[-2] and
                recent_rsi_valleys.iloc[-1] > recent_rsi_valleys.iloc[-2]):
                
                divergences.append({
                    'pattern': 'bullish_divergence_rsi',
                    'confidence': 0.68,
                    'timeframe': '1h',
                    'description': 'Divergência bullish RSI detectada',
                    'implication': 'potential_reversal_up'
                })
        
        return divergences

# =====================================================================
# ANALISADOR DE REGIME DE MERCADO
# =====================================================================

class MarketRegimeAnalyzer:
    """Analisador avançado de regimes de mercado"""
    
    def __init__(self):
        self.regime_history = {}
        self.regime_models = {}
        
    def analyze_market_regime(self, df: pd.DataFrame, symbol: str) -> MarketRegimeData:
        """Analisa o regime atual de mercado"""
        
        if len(df) < 100:
            return self._default_regime()
        
        # Calcular features para análise de regime
        features = self._calculate_regime_features(df)
        
        # Detectar regime usando múltiplas abordagens
        volatility_regime = self._analyze_volatility_regime(features)
        trend_regime = self._analyze_trend_regime(features)
        volume_regime = self._analyze_volume_regime(features)
        
        # Combinar análises
        regime_type = self._combine_regime_analysis(volatility_regime, trend_regime, volume_regime)
        
        # Calcular métricas detalhadas
        confidence = self._calculate_regime_confidence(features, regime_type)
        volatility_percentile = self._calculate_volatility_percentile(features)
        trend_strength = self._calculate_trend_strength(features)
        momentum_score = self._calculate_momentum_score(features)
        
        # Detectar breakdown de correlação
        correlation_breakdown = self._detect_correlation_breakdown(df)
        
        # Estimar duração do regime
        regime_duration = self._estimate_regime_duration(features, regime_type)
        expected_duration = self._predict_regime_duration(features, regime_type)
        
        return MarketRegimeData(
            regime_type=regime_type,
            confidence=confidence,
            volatility_percentile=volatility_percentile,
            trend_strength=trend_strength,
            momentum_score=momentum_score,
            volume_profile=volume_regime,
            correlation_breakdown=correlation_breakdown,
            regime_duration_days=regime_duration,
            expected_duration_days=expected_duration
        )
    
    def _calculate_regime_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcula features para análise de regime"""
        
        features = {}
        
        # Retornos
        returns = df['close'].pct_change().dropna()
        features['returns'] = returns.values
        
        # Volatilidade realizada
        features['realized_vol'] = returns.rolling(20).std().values
        
        # Skewness e Kurtosis
        features['skewness'] = returns.rolling(20).skew().values
        features['kurtosis'] = returns.rolling(20).kurt().values
        
        # Trend features
        features['sma_20'] = talib.SMA(df['close'], 20)
        features['sma_50'] = talib.SMA(df['close'], 50)
        features['ema_12'] = talib.EMA(df['close'], 12)
        features['ema_26'] = talib.EMA(df['close'], 26)
        
        # Momentum
        features['rsi'] = talib.RSI(df['close'], 14)
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(df['close'])
        
        # Volume
        features['volume'] = df['volume'].values
        features['volume_sma'] = talib.SMA(df['volume'], 20)
        features['volume_ratio'] = df['volume'] / features['volume_sma']
        
        # Volatility
        features['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        features['bbands_upper'], features['bbands_middle'], features['bbands_lower'] = talib.BBANDS(df['close'], 20)
        features['bbands_width'] = (features['bbands_upper'] - features['bbands_lower']) / features['bbands_middle']
        
        return features
    
    def _analyze_volatility_regime(self, features: Dict) -> str:
        """Analisa regime de volatilidade"""
        
        current_vol = features['realized_vol'][-1] if len(features['realized_vol']) > 0 else 0
        historical_vol = np.nanmean(features['realized_vol'])
        vol_percentile = np.nanpercentile(features['realized_vol'], 80)
        
        if current_vol > vol_percentile:
            return 'high_volatility'
        elif current_vol < np.nanpercentile(features['realized_vol'], 20):
            return 'low_volatility'
        else:
            return 'normal_volatility'
    
    def _analyze_trend_regime(self, features: Dict) -> str:
        """Analisa regime de tendência"""
        
        # Comparar EMAs
        ema_12 = features['ema_12'][-1] if len(features['ema_12']) > 0 else 0
        ema_26 = features['ema_26'][-1] if len(features['ema_26']) > 0 else 0
        
        # Comparar SMAs
        sma_20 = features['sma_20'][-1] if len(features['sma_20']) > 0 else 0
        sma_50 = features['sma_50'][-1] if len(features['sma_50']) > 0 else 0
        
        # RSI
        rsi = features['rsi'][-1] if len(features['rsi']) > 0 else 50
        
        # Determinar regime
        if ema_12 > ema_26 and sma_20 > sma_50 and rsi > 55:
            return 'bullish_trend'
        elif ema_12 < ema_26 and sma_20 < sma_50 and rsi < 45:
            return 'bearish_trend'
        else:
            return 'sideways_trend'
    
    def _analyze_volume_regime(self, features: Dict) -> str:
        """Analisa regime de volume"""
        
        volume_ratio = features['volume_ratio'][-5:] if len(features['volume_ratio']) >= 5 else []
        
        if len(volume_ratio) == 0:
            return 'normal'
        
        avg_ratio = np.mean(volume_ratio)
        
        if avg_ratio > 1.5:
            return 'high'
        elif avg_ratio < 0.7:
            return 'low'
        else:
            return 'normal'
    
    def _combine_regime_analysis(self, vol_regime: str, trend_regime: str, volume_regime: str) -> str:
        """Combina diferentes análises de regime"""
        
        # Lógica de combinação
        if vol_regime == 'high_volatility':
            if trend_regime == 'bullish_trend':
                return 'volatile_bull'
            elif trend_regime == 'bearish_trend':
                return 'volatile_bear'
            else:
                return 'volatile_sideways'
        
        elif vol_regime == 'low_volatility':
            if trend_regime == 'bullish_trend':
                return 'stable_bull'
            elif trend_regime == 'bearish_trend':
                return 'stable_bear'
            else:
                return 'stable_sideways'
        
        else:  # normal volatility
            if trend_regime == 'bullish_trend':
                return 'bull'
            elif trend_regime == 'bearish_trend':
                return 'bear'
            else:
                return 'sideways'
    
    def _calculate_regime_confidence(self, features: Dict, regime_type: str) -> float:
        """Calcula confiança no regime detectado"""
        
        # Baseado na consistência dos indicadores
        confidence_factors = []
        
        # Consistência de tendência
        if 'ema_12' in features and 'ema_26' in features:
            ema_diff = features['ema_12'][-10:] - features['ema_26'][-10:]
            trend_consistency = np.sum(np.sign(ema_diff) == np.sign(ema_diff[-1])) / len(ema_diff)
            confidence_factors.append(trend_consistency)
        
        # Consistência de volume
        if 'volume_ratio' in features:
            volume_consistency = 1.0 - np.std(features['volume_ratio'][-10:])
            confidence_factors.append(max(0, volume_consistency))
        
        # Consistência de volatilidade
        if 'realized_vol' in features:
            vol_consistency = 1.0 - (np.std(features['realized_vol'][-10:]) / np.mean(features['realized_vol'][-10:]))
            confidence_factors.append(max(0, vol_consistency))
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_volatility_percentile(self, features: Dict) -> float:
        """Calcula percentil de volatilidade atual"""
        
        if 'realized_vol' in features and len(features['realized_vol']) > 0:
            current_vol = features['realized_vol'][-1]
            historical_vols = features['realized_vol'][:-1]
            
            if len(historical_vols) > 0:
                percentile = (np.sum(historical_vols <= current_vol) / len(historical_vols)) * 100
                return percentile
        
        return 50.0  # Default
    
    def _calculate_trend_strength(self, features: Dict) -> float:
        """Calcula força da tendência"""
        
        if 'ema_12' in features and 'ema_26' in features:
            ema_diff = abs(features['ema_12'][-1] - features['ema_26'][-1])
            price = features['ema_12'][-1]
            return (ema_diff / price) * 100 if price > 0 else 0
        
        return 0.0
    
    def _calculate_momentum_score(self, features: Dict) -> float:
        """Calcula score de momentum"""
        
        scores = []
        
        # RSI momentum
        if 'rsi' in features:
            rsi = features['rsi'][-1]
            if rsi > 60:
                scores.append(0.8)
            elif rsi < 40:
                scores.append(-0.8)
            else:
                scores.append(0.0)
        
        # MACD momentum
        if 'macd' in features and 'macd_signal' in features:
            macd_diff = features['macd'][-1] - features['macd_signal'][-1]
            scores.append(np.tanh(macd_diff * 1000))  # Normalizar
        
        return np.mean(scores) if scores else 0.0
    
    def _detect_correlation_breakdown(self, df: pd.DataFrame) -> bool:
        """Detecta breakdown de correlação"""
        
        # Simplificado - comparar correlação recente vs histórica
        if len(df) < 50:
            return False
        
        # Correlação entre high e low (proxy para estrutura normal)
        recent_corr = df[['high', 'low']].tail(20).corr().iloc[0, 1]
        historical_corr = df[['high', 'low']].corr().iloc[0, 1]
        
        # Se correlação diminuiu significativamente
        return recent_corr < historical_corr * 0.8
    
    def _estimate_regime_duration(self, features: Dict, regime_type: str) -> int:
        """Estima duração atual do regime em dias"""
        
        # Simplificado - baseado na consistência da tendência
        if 'ema_12' in features and 'ema_26' in features:
            ema_diff = features['ema_12'] - features['ema_26']
            
            # Contar quantos períodos a tendência se mantém
            current_sign = np.sign(ema_diff[-1])
            duration = 0
            
            for i in range(len(ema_diff) - 1, -1, -1):
                if np.sign(ema_diff[i]) == current_sign:
                    duration += 1
                else:
                    break
            
            # Converter para dias (assumindo dados de 1h)
            return duration // 24
        
        return 0
    
    def _predict_regime_duration(self, features: Dict, regime_type: str) -> int:
        """Prediz duração esperada do regime"""
        
        # Baseado em padrões históricos típicos
        regime_durations = {
            'bull': 30,
            'bear': 20,
            'sideways': 15,
            'volatile_bull': 10,
            'volatile_bear': 8,
            'volatile_sideways': 5,
            'stable_bull': 45,
            'stable_bear': 35,
            'stable_sideways': 25
        }
        
        return regime_durations.get(regime_type, 15)
    
    def _default_regime(self) -> MarketRegimeData:
        """Retorna regime padrão quando dados insuficientes"""
        
        return MarketRegimeData(
            regime_type='unknown',
            confidence=0.0,
            volatility_percentile=50.0,
            trend_strength=0.0,
            momentum_score=0.0,
            volume_profile='normal',
            correlation_breakdown=False,
            regime_duration_days=0,
            expected_duration_days=15
        )

# =====================================================================
# DETECTOR DE ANOMALIAS
# =====================================================================

class MarketAnomalyDetector:
    """Detector de anomalias de mercado usando ML"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trained_symbols = set()
        
    def detect_anomalies(self, df: pd.DataFrame, symbol: str) -> List[AnomalyDetection]:
        """Detecta anomalias nos dados de mercado"""
        
        anomalies = []
        
        if len(df) < 100:
            return anomalies
        
        try:
            # Preparar features
            features = self._prepare_anomaly_features(df)
            
            # Treinar modelo se necessário
            if symbol not in self.trained_symbols:
                self._train_anomaly_model(features, symbol)
            
            # Detectar anomalias
            price_anomalies = self._detect_price_anomalies(features, symbol)
            volume_anomalies = self._detect_volume_anomalies(features, symbol)
            volatility_anomalies = self._detect_volatility_anomalies(features, symbol)
            
            anomalies.extend(price_anomalies)
            anomalies.extend(volume_anomalies)
            anomalies.extend(volatility_anomalies)
            
        except Exception as e:
            logger.warning(f"Erro detectando anomalias para {symbol}: {e}")
        
        return anomalies
    
    def _prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para detecção de anomalias"""
        
        features_df = pd.DataFrame(index=df.index)
        
        # Price features
        features_df['price_change'] = df['close'].pct_change()
        features_df['price_acceleration'] = features_df['price_change'].diff()
        features_df['price_z_score'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        
        # Volume features
        features_df['volume_change'] = df['volume'].pct_change()
        features_df['volume_z_score'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
        
        # Volatility features
        features_df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features_df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        features_df['volatility'] = features_df['price_change'].rolling(20).std()
        
        # Technical indicators
        features_df['rsi'] = talib.RSI(df['close'], 14)
        features_df['macd'], features_df['macd_signal'], features_df['macd_hist'] = talib.MACD(df['close'])
        
        # Bollinger Bands position
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], 20)
        features_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        return features_df.dropna()
    
    def _train_anomaly_model(self, features: pd.DataFrame, symbol: str):
        """Treina modelo de detecção de anomalias"""
        
        try:
            # Selecionar features numéricas
            numeric_features = features.select_dtypes(include=[np.number]).columns
            X = features[numeric_features].values
            
            # Normalizar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Treinar Isolation Forest
            model = IsolationForest(
                contamination=0.1,  # 10% de anomalias esperadas
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Salvar modelo e scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.trained_symbols.add(symbol)
            
            logger.info(f"Modelo de anomalia treinado para {symbol}")
            
        except Exception as e:
            logger.error(f"Erro treinando modelo de anomalia para {symbol}: {e}")
    
    def _detect_price_anomalies(self, features: pd.DataFrame, symbol: str) -> List[AnomalyDetection]:
        """Detecta anomalias de preço"""
        
        anomalies = []
        
        if symbol not in self.models:
            return anomalies
        
        try:
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # Preparar dados recentes
            recent_data = features.tail(10)
            numeric_features = recent_data.select_dtypes(include=[np.number]).columns
            X = recent_data[numeric_features].values
            X_scaled = scaler.transform(X)
            
            # Predizer anomalias
            anomaly_scores = model.decision_function(X_scaled)
            is_anomaly = model.predict(X_scaled) == -1
            
            # Processar resultados
            for i, (is_anom, score) in enumerate(zip(is_anomaly, anomaly_scores)):
                if is_anom:
                    # Determinar tipo de anomalia
                    price_change = recent_data['price_change'].iloc[i]
                    volume_change = recent_data['volume_change'].iloc[i]
                    
                    if abs(price_change) > 0.05:  # Movimento > 5%
                        anomaly_type = 'large_price_movement'
                        significance = abs(price_change) * 10
                        description = f"Movimento de preço anômalo: {price_change:.2%}"
                        trading_opportunity = True
                        risk_level = 'high' if abs(price_change) > 0.1 else 'medium'
                    else:
                        anomaly_type = 'price_pattern_anomaly'
                        significance = abs(score)
                        description = "Padrão de preço anômalo detectado"
                        trading_opportunity = False
                        risk_level = 'low'
                    
                    anomalies.append(AnomalyDetection(
                        anomaly_score=abs(score),
                        anomaly_type=anomaly_type,
                        significance_level=significance,
                        description=description,
                        trading_opportunity=trading_opportunity,
                        risk_level=risk_level
                    ))
                    
        except Exception as e:
            logger.warning(f"Erro detectando anomalias de preço para {symbol}: {e}")
        
        return anomalies
    
    def _detect_volume_anomalies(self, features: pd.DataFrame, symbol: str) -> List[AnomalyDetection]:
        """Detecta anomalias de volume"""
        
        anomalies = []
        
        # Volume spikes
        volume_z = features['volume_z_score'].tail(5)
        
        for i, z_score in enumerate(volume_z):
            if abs(z_score) > 3:  # 3 desvios padrão
                anomalies.append(AnomalyDetection(
                    anomaly_score=abs(z_score),
                    anomaly_type='volume_spike',
                    significance_level=abs(z_score),
                    description=f"Spike de volume anômalo (Z-score: {z_score:.2f})",
                    trading_opportunity=True,
                    risk_level='medium'
                ))
        
        return anomalies
    
    def _detect_volatility_anomalies(self, features: pd.DataFrame, symbol: str) -> List[AnomalyDetection]:
        """Detecta anomalias de volatilidade"""
        
        anomalies = []
        
        # Volatility expansion
        current_vol = features['volatility'].iloc[-1]
        historical_vol = features['volatility'].mean()
        
        if current_vol > historical_vol * 2:
            anomalies.append(AnomalyDetection(
                anomaly_score=current_vol / historical_vol,
                anomaly_type='volatility_expansion',
                significance_level=current_vol / historical_vol,
                description=f"Expansão anômala de volatilidade ({current_vol/historical_vol:.1f}x)",
                trading_opportunity=True,
                risk_level='high'
            ))
        
        return anomalies

# =====================================================================
# SISTEMA PRINCIPAL DE ANÁLISE AVANÇADA
# =====================================================================

class AdvancedMarketAnalyzer:
    """Sistema principal de análise avançada de mercado"""
    
    def __init__(self):
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.anomaly_detector = MarketAnomalyDetector()
        
        # Cache de análises
        self.analysis_cache = {}
        self.cache_duration = 300  # 5 minutos
        
    async def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Análise abrangente de um símbolo"""
        
        # Verificar cache
        cache_key = f"{symbol}_{len(df)}"
        if cache_key in self.analysis_cache:
            cached_time, cached_analysis = self.analysis_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_analysis
        
        analysis = {}
        
        try:
            # Análise de padrões complexos
            patterns = self.pattern_analyzer.detect_complex_patterns(df, symbol)
            analysis['patterns'] = patterns
            
            # Análise de regime de mercado
            regime = self.regime_analyzer.analyze_market_regime(df, symbol)
            analysis['market_regime'] = regime.__dict__
            
            # Detecção de anomalias
            anomalies = self.anomaly_detector.detect_anomalies(df, symbol)
            analysis['anomalies'] = [anomaly.__dict__ for anomaly in anomalies]
            
            # Análise de microestrutura
            microstructure = self._analyze_microstructure(df)
            analysis['microstructure'] = microstructure.__dict__
            
            # Análise de ciclos
            cycles = self._analyze_cycles(df)
            analysis['cycles'] = cycles.__dict__
            
            # Score geral de oportunidade
            opportunity_score = self._calculate_opportunity_score(analysis)
            analysis['opportunity_score'] = opportunity_score
            
            # Recomendações
            recommendations = self._generate_recommendations(analysis)
            analysis['recommendations'] = recommendations
            
            # Cache da análise
            self.analysis_cache[cache_key] = (time.time(), analysis)
            
        except Exception as e:
            logger.error(f"Erro na análise abrangente de {symbol}: {e}")
            analysis = {'error': str(e)}
        
        return analysis
    
    def _analyze_microestrutura(self, df: pd.DataFrame) -> MarketMicrostructure:
        """Análise de microestrutura de mercado"""
        
        # Spread bid-ask proxy
        spread_proxy = (df['high'] - df['low']).rolling(20).mean().iloc[-1] / df['close'].iloc[-1]
        
        # Price impact estimate
        returns = df['close'].pct_change()
        volume_impact = abs(returns.corr(df['volume'].pct_change()))
        
        # Order flow imbalance proxy
        body_size = abs(df['close'] - df['open'])
        wick_size = (df['high'] - df['low']) - body_size
        imbalance = (body_size - wick_size).rolling(20).mean().iloc[-1]
        
        # Liquidity score
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        recent_volume = df['volume'].tail(10).mean()
        liquidity_score = min(1.0, recent_volume / avg_volume)
        
        # Manipulation probability (simplificado)
        price_vol_corr = abs(returns.tail(50).corr(df['volume'].pct_change().tail(50)))
        manipulation_prob = 1 - price_vol_corr if not np.isnan(price_vol_corr) else 0.5
        
        # Optimal entry size
        atr = talib.ATR(df['high'], df['low'], df['close'], 14).iloc[-1]
        optimal_size = avg_volume * 0.05  # 5% do volume médio
        
        # Slippage estimate
        slippage_estimate = spread_proxy * 2  # Estimativa conservadora
        
        return MarketMicrostructure(
            bid_ask_spread_proxy=spread_proxy,
            price_impact_estimate=volume_impact,
            order_flow_imbalance=imbalance,
            liquidity_score=liquidity_score,
            manipulation_probability=manipulation_prob,
            optimal_entry_size=optimal_size,
            slippage_estimate=slippage_estimate
        )
    
    def _analyze_cycles(self, df: pd.DataFrame) -> CycleAnalysis:
        """Análise de ciclos de mercado"""
        
        if len(df) < 100:
            return CycleAnalysis(
                dominant_cycle_period=0,
                cycle_strength=0.0,
                cycle_phase='unknown',
                next_phase_probability=0.0,
                cycle_reliability=0.0
            )
        
        # Análise de Fourier para encontrar ciclos dominantes
        prices = df['close'].values
        fft = np.fft.fft(prices)
        freqs = np.fft.fftfreq(len(prices))
        
        # Encontrar frequência dominante (excluindo DC)
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        dominant_period = int(1 / abs(freqs[dominant_freq_idx]))
        
        # Força do ciclo
        cycle_strength = np.abs(fft[dominant_freq_idx]) / np.sum(np.abs(fft))
        
        # Determinar fase do ciclo
        recent_prices = prices[-dominant_period:]
        if len(recent_prices) >= dominant_period:
            phase_position = np.argmax(recent_prices) / len(recent_prices)
            
            if phase_position < 0.25:
                cycle_phase = 'accumulation'
            elif phase_position < 0.5:
                cycle_phase = 'markup'
            elif phase_position < 0.75:
                cycle_phase = 'distribution'
            else:
                cycle_phase = 'markdown'
        else:
            cycle_phase = 'unknown'
        
        # Probabilidade da próxima fase
        next_phase_prob = cycle_strength * 0.8  # Ajustar baseado na força
        
        # Confiabilidade do ciclo
        cycle_reliability = min(1.0, cycle_strength * 2)
        
        return CycleAnalysis(
            dominant_cycle_period=dominant_period,
            cycle_strength=float(cycle_strength),
            cycle_phase=cycle_phase,
            next_phase_probability=float(next_phase_prob),
            cycle_reliability=float(cycle_reliability)
        )
    
    def _calculate_opportunity_score(self, analysis: Dict) -> float:
        """Calcula score geral de oportunidade"""
        
        score = 0.0
        factors = []
        
        # Score baseado em padrões
        patterns = analysis.get('patterns', [])
        if patterns:
            pattern_scores = [p.get('confidence', 0) for p in patterns]
            factors.append(np.mean(pattern_scores))
        
        # Score baseado em regime
        regime = analysis.get('market_regime', {})
        regime_confidence = regime.get('confidence', 0)
        factors.append(regime_confidence)
        
        # Score baseado em anomalias
        anomalies = analysis.get('anomalies', [])
        trading_anomalies = [a for a in anomalies if a.get('trading_opportunity', False)]
        if trading_anomalies:
            anomaly_scores = [a.get('significance_level', 0) for a in trading_anomalies]
            factors.append(min(1.0, np.mean(anomaly_scores) / 3))  # Normalizar
        
        # Score baseado em microestrutura
        microstructure = analysis.get('microstructure', {})
        liquidity_score = microstructure.get('liquidity_score', 0.5)
        factors.append(liquidity_score)
        
        # Score baseado em ciclos
        cycles = analysis.get('cycles', {})
        cycle_reliability = cycles.get('cycle_reliability', 0)
        factors.append(cycle_reliability)
        
        # Média ponderada
        if factors:
            score = np.mean(factors)
        
        return float(score)
    
    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Gera recomendações baseadas na análise"""
        
        recommendations = []
        
        # Recomendações baseadas em padrões
        patterns = analysis.get('patterns', [])
        for pattern in patterns:
            if pattern.get('confidence', 0) > 0.7:
                recommendations.append({
                    'type': 'pattern_trade',
                    'description': f"Considerar trade baseado em {pattern.get('pattern', 'padrão')}",
                    'confidence': pattern.get('confidence', 0),
                    'timeframe': pattern.get('timeframe', '1h'),
                    'risk_level': 'medium'
                })
        
        # Recomendações baseadas em regime
        regime = analysis.get('market_regime', {})
        regime_type = regime.get('regime_type', '')
        
        if 'bull' in regime_type:
            recommendations.append({
                'type': 'regime_bias',
                'description': 'Favorecer posições long baseado no regime bullish',
                'confidence': regime.get('confidence', 0),
                'timeframe': 'multiple',
                'risk_level': 'low'
            })
        elif 'bear' in regime_type:
            recommendations.append({
                'type': 'regime_bias',
                'description': 'Favorecer posições short baseado no regime bearish',
                'confidence': regime.get('confidence', 0),
                'timeframe': 'multiple',
                'risk_level': 'low'
            })
        
        # Recomendações baseadas em anomalias
        anomalies = analysis.get('anomalies', [])
        for anomaly in anomalies:
            if anomaly.get('trading_opportunity', False):
                recommendations.append({
                    'type': 'anomaly_trade',
                    'description': f"Oportunidade de trade: {anomaly.get('description', 'anomalia detectada')}",
                    'confidence': min(1.0, anomaly.get('significance_level', 0) / 3),
                    'timeframe': '1h',
                    'risk_level': anomaly.get('risk_level', 'medium')
                })
        
        # Recomendações de gestão de risco
        microstructure = analysis.get('microstructure', {})
        if microstructure.get('liquidity_score', 0) < 0.3:
            recommendations.append({
                'type': 'risk_management',
                'description': 'Baixa liquidez detectada - reduzir tamanho de posição',
                'confidence': 0.9,
                'timeframe': 'immediate',
                'risk_level': 'high'
            })
        
        return recommendations
    
    def get_market_summary(self, symbols_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Gera resumo geral do mercado"""
        
        summary = {
            'total_symbols_analyzed': len(symbols_analysis),
            'high_opportunity_symbols': [],
            'dominant_regime': 'mixed',
            'market_stress_level': 'normal',
            'anomalies_detected': 0,
            'patterns_detected': 0,
            'recommendations_count': 0
        }
        
        # Analisar símbolos
        opportunity_scores = []
        regimes = []
        total_anomalies = 0
        total_patterns = 0
        total_recommendations = 0
        
        for symbol, analysis in symbols_analysis.items():
            if 'error' in analysis:
                continue
                
            # Opportunity scores
            opp_score = analysis.get('opportunity_score', 0)
            opportunity_scores.append((symbol, opp_score))
            
            # Regimes
            regime = analysis.get('market_regime', {}).get('regime_type', 'unknown')
            regimes.append(regime)
            
            # Contadores
            total_anomalies += len(analysis.get('anomalies', []))
            total_patterns += len(analysis.get('patterns', []))
            total_recommendations += len(analysis.get('recommendations', []))
        
        # Top oportunidades
        opportunity_scores.sort(key=lambda x: x[1], reverse=True)
        summary['high_opportunity_symbols'] = opportunity_scores[:5]
        
        # Regime dominante
        if regimes:
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            summary['dominant_regime'] = max(regime_counts, key=regime_counts.get)
        
        # Nível de stress do mercado
        if opportunity_scores:
            avg_opportunity = np.mean([score for _, score in opportunity_scores])
            if avg_opportunity > 0.7:
                summary['market_stress_level'] = 'high_opportunity'
            elif avg_opportunity < 0.3:
                summary['market_stress_level'] = 'low_opportunity'
        
        summary['anomalies_detected'] = total_anomalies
        summary['patterns_detected'] = total_patterns
        summary['recommendations_count'] = total_recommendations
        
        return summary