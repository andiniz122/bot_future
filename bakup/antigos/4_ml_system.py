#!/usr/bin/env python3
"""
Sistema de Machine Learning Adaptativo
Aprende continuamente com os resultados e adapta as estratégias
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import json
import os
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

# Deep learning (opcional - se disponível)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('adaptive_ml_system')

# =====================================================================
# CLASSES DE DADOS PARA ML
# =====================================================================

@dataclass
class MLModelPerformance:
    """Performance de um modelo ML"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    training_samples: int = 0
    last_trained: Optional[datetime] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cross_val_scores: List[float] = field(default_factory=list)
    
    def calculate_composite_score(self) -> float:
        """Calcula score composto considerando múltiplas métricas"""
        return (self.accuracy * 0.3 + 
                self.precision * 0.25 + 
                self.recall * 0.25 + 
                self.f1_score * 0.2)

@dataclass
class AdaptiveLearningState:
    """Estado de aprendizado adaptativo"""
    total_trades_learned: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    model_adaptations: int = 0
    last_adaptation: Optional[datetime] = None
    learning_rate: float = 0.01
    confidence_threshold: float = 0.6
    adaptation_trigger_threshold: int = 10
    
    def get_success_rate(self) -> float:
        total = self.successful_predictions + self.failed_predictions
        return self.successful_predictions / total if total > 0 else 0.0

@dataclass
class FeatureImportanceAnalysis:
    """Análise de importância de features"""
    feature_rankings: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    redundant_features: List[str] = field(default_factory=list)
    new_feature_suggestions: List[str] = field(default_factory=list)
    feature_stability_scores: Dict[str, float] = field(default_factory=dict)

# =====================================================================
# SISTEMA DE FEATURE ENGINEERING DINÂMICO
# =====================================================================

class DynamicFeatureEngineer:
    """Sistema de feature engineering que evolui automaticamente"""
    
    def __init__(self):
        self.feature_catalog = self._initialize_feature_catalog()
        self.active_features = set()
        self.feature_performance_history = {}
        self.feature_combinations = []
        self.custom_features = {}
        
    def _initialize_feature_catalog(self) -> Dict[str, Dict]:
        """Inicializa catálogo de features disponíveis"""
        return {
            # Price-based features
            'price_momentum': {
                'function': self._calculate_price_momentum,
                'parameters': {'periods': [3, 5, 10, 20]},
                'category': 'momentum',
                'complexity': 'low'
            },
            'price_acceleration': {
                'function': self._calculate_price_acceleration,
                'parameters': {'periods': [5, 10, 15]},
                'category': 'momentum',
                'complexity': 'medium'
            },
            'relative_strength': {
                'function': self._calculate_relative_strength,
                'parameters': {'periods': [10, 20, 50]},
                'category': 'strength',
                'complexity': 'medium'
            },
            
            # Volume-based features
            'volume_profile': {
                'function': self._calculate_volume_profile,
                'parameters': {'periods': [10, 20, 30]},
                'category': 'volume',
                'complexity': 'high'
            },
            'volume_momentum': {
                'function': self._calculate_volume_momentum,
                'parameters': {'periods': [5, 10, 20]},
                'category': 'volume',
                'complexity': 'low'
            },
            
            # Volatility-based features
            'volatility_regime': {
                'function': self._calculate_volatility_regime,
                'parameters': {'periods': [20, 50, 100]},
                'category': 'volatility',
                'complexity': 'high'
            },
            'volatility_clustering': {
                'function': self._calculate_volatility_clustering,
                'parameters': {'periods': [10, 20]},
                'category': 'volatility',
                'complexity': 'medium'
            },
            
            # Technical indicators
            'custom_oscillator': {
                'function': self._calculate_custom_oscillator,
                'parameters': {'fast': [5, 8, 12], 'slow': [20, 26, 34]},
                'category': 'technical',
                'complexity': 'medium'
            },
            'adaptive_bands': {
                'function': self._calculate_adaptive_bands,
                'parameters': {'periods': [15, 20, 25], 'multiplier': [1.5, 2.0, 2.5]},
                'category': 'technical',
                'complexity': 'high'
            },
            
            # Market microstructure
            'order_flow_imbalance': {
                'function': self._calculate_order_flow_imbalance,
                'parameters': {'periods': [5, 10, 15]},
                'category': 'microstructure',
                'complexity': 'high'
            },
            'tick_direction_bias': {
                'function': self._calculate_tick_direction_bias,
                'parameters': {'periods': [10, 20, 30]},
                'category': 'microstructure',
                'complexity': 'medium'
            },
            
            # Cross-asset features
            'market_correlation': {
                'function': self._calculate_market_correlation,
                'parameters': {'periods': [20, 50, 100]},
                'category': 'correlation',
                'complexity': 'high'
            },
            
            # Time-based features
            'time_of_day_effect': {
                'function': self._calculate_time_of_day_effect,
                'parameters': {},
                'category': 'temporal',
                'complexity': 'low'
            },
            'day_of_week_effect': {
                'function': self._calculate_day_of_week_effect,
                'parameters': {},
                'category': 'temporal',
                'complexity': 'low'
            }
        }
    
    def generate_features(self, df: pd.DataFrame, target_performance: float = None) -> pd.DataFrame:
        """Gera features dinamicamente baseado na performance"""
        
        features_df = df.copy()
        
        # Selecionar features baseado na performance histórica
        selected_features = self._select_optimal_features(target_performance)
        
        # Gerar features selecionadas
        for feature_name in selected_features:
            try:
                if feature_name in self.feature_catalog:
                    feature_config = self.feature_catalog[feature_name]
                    feature_function = feature_config['function']
                    
                    # Gerar feature
                    new_features = feature_function(df, **feature_config.get('parameters', {}))
                    
                    if isinstance(new_features, pd.DataFrame):
                        features_df = pd.concat([features_df, new_features], axis=1)
                    elif isinstance(new_features, pd.Series):
                        features_df[feature_name] = new_features
                        
                elif feature_name in self.custom_features:
                    # Feature customizada
                    custom_function = self.custom_features[feature_name]
                    new_feature = custom_function(df)
                    features_df[feature_name] = new_feature
                    
            except Exception as e:
                logger.warning(f"Erro gerando feature {feature_name}: {e}")
                continue
        
        # Gerar combinações de features se performance não estiver boa
        if target_performance and target_performance < 0.6:
            features_df = self._generate_feature_combinations(features_df)
        
        return features_df
    
    def _select_optimal_features(self, target_performance: float = None) -> List[str]:
        """Seleciona features ótimas baseado na performance"""
        
        if not self.feature_performance_history:
            # Primeira execução - usar features básicas
            return ['price_momentum', 'volume_momentum', 'volatility_regime', 'custom_oscillator']
        
        # Ordenar features por performance
        feature_scores = []
        for feature, performance_data in self.feature_performance_history.items():
            avg_performance = np.mean(performance_data) if performance_data else 0
            feature_scores.append((feature, avg_performance))
        
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar top features
        top_features = [feature for feature, _ in feature_scores[:10]]
        
        # Adicionar features exploratórias se performance baixa
        if target_performance and target_performance < 0.5:
            unused_features = set(self.feature_catalog.keys()) - set(top_features)
            if unused_features:
                exploration_features = list(unused_features)[:3]
                top_features.extend(exploration_features)
        
        return top_features
    
    def update_feature_performance(self, feature_name: str, performance: float):
        """Atualiza performance de uma feature"""
        if feature_name not in self.feature_performance_history:
            self.feature_performance_history[feature_name] = []
        
        self.feature_performance_history[feature_name].append(performance)
        
        # Manter apenas últimas 50 performances para cada feature
        if len(self.feature_performance_history[feature_name]) > 50:
            self.feature_performance_history[feature_name] = self.feature_performance_history[feature_name][-50:]
    
    def create_custom_feature(self, name: str, function: callable):
        """Cria uma feature customizada"""
        self.custom_features[name] = function
        logger.info(f"Feature customizada '{name}' criada")
    
    # Implementações das funções de features
    def _calculate_price_momentum(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for period in periods:
            features[f'price_momentum_{period}'] = df['close'].pct_change(period)
        return features
    
    def _calculate_price_acceleration(self, df: pd.DataFrame, periods: List[int] = [5, 10]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for period in periods:
            momentum = df['close'].pct_change(period)
            features[f'price_acceleration_{period}'] = momentum.diff()
        return features
    
    def _calculate_relative_strength(self, df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for period in periods:
            high_ma = df['high'].rolling(period).max()
            low_ma = df['low'].rolling(period).min()
            features[f'relative_strength_{period}'] = (df['close'] - low_ma) / (high_ma - low_ma)
        return features
    
    def _calculate_volume_profile(self, df: pd.DataFrame, periods: List[int] = [20]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for period in periods:
            volume_ma = df['volume'].rolling(period).mean()
            features[f'volume_profile_{period}'] = df['volume'] / volume_ma
        return features
    
    def _calculate_volume_momentum(self, df: pd.DataFrame, periods: List[int] = [5, 10]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for period in periods:
            features[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
        return features
    
    def _calculate_volatility_regime(self, df: pd.DataFrame, periods: List[int] = [20, 50]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        
        for period in periods:
            realized_vol = returns.rolling(period).std()
            vol_percentile = realized_vol.rolling(100).rank(pct=True)
            features[f'volatility_regime_{period}'] = vol_percentile
        
        return features
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        
        for period in periods:
            vol = returns.rolling(period).std()
            vol_change = vol.diff()
            features[f'volatility_clustering_{period}'] = vol_change
        
        return features
    
    def _calculate_custom_oscillator(self, df: pd.DataFrame, fast: List[int] = [8, 12], slow: List[int] = [21, 26]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        for f in fast:
            for s in slow:
                if f < s:
                    ema_fast = df['close'].ewm(span=f).mean()
                    ema_slow = df['close'].ewm(span=s).mean()
                    features[f'custom_oscillator_{f}_{s}'] = (ema_fast - ema_slow) / ema_slow
        
        return features
    
    def _calculate_adaptive_bands(self, df: pd.DataFrame, periods: List[int] = [20], multiplier: List[float] = [2.0]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        for period in periods:
            for mult in multiplier:
                sma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                
                upper_band = sma + (std * mult)
                lower_band = sma - (std * mult)
                
                features[f'adaptive_bands_position_{period}_{mult}'] = (df['close'] - lower_band) / (upper_band - lower_band)
        
        return features
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame, periods: List[int] = [10]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # Proxy para order flow usando OHLC
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        for period in periods:
            imbalance = typical_price.diff().rolling(period).sum()
            features[f'order_flow_imbalance_{period}'] = imbalance
        
        return features
    
    def _calculate_tick_direction_bias(self, df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        tick_direction = np.sign(df['close'].diff())
        
        for period in periods:
            bias = tick_direction.rolling(period).mean()
            features[f'tick_direction_bias_{period}'] = bias
        
        return features
    
    def _calculate_market_correlation(self, df: pd.DataFrame, periods: List[int] = [20]) -> pd.DataFrame:
        # Placeholder - necessitaria dados de outros ativos
        features = pd.DataFrame(index=df.index)
        
        for period in periods:
            # Correlação com próprio histórico como proxy
            returns = df['close'].pct_change()
            lag_correlation = returns.rolling(period).corr(returns.shift(1))
            features[f'market_correlation_{period}'] = lag_correlation
        
        return features
    
    def _calculate_time_of_day_effect(self, df: pd.DataFrame) -> pd.Series:
        """Calcula efeito da hora do dia"""
        if hasattr(df.index, 'hour'):
            return pd.Series(df.index.hour, index=df.index, name='hour_of_day')
        else:
            return pd.Series(0, index=df.index, name='hour_of_day')
    
    def _calculate_day_of_week_effect(self, df: pd.DataFrame) -> pd.Series:
        """Calcula efeito do dia da semana"""
        if hasattr(df.index, 'dayofweek'):
            return pd.Series(df.index.dayofweek, index=df.index, name='day_of_week')
        else:
            return pd.Series(0, index=df.index, name='day_of_week')
    
    def _generate_feature_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera combinações de features existentes"""
        
        # Encontrar features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        # Criar algumas combinações úteis
        combinations_created = 0
        max_combinations = 5
        
        for i, feat1 in enumerate(numeric_features[:10]):  # Limitar para performance
            for feat2 in numeric_features[i+1:10]:
                if combinations_created >= max_combinations:
                    break
                
                try:
                    # Ratio
                    if (df[feat2] != 0).all():
                        df[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2]
                        combinations_created += 1
                        
                    # Diferença normalizada
                    if combinations_created < max_combinations:
                        df[f'{feat1}_minus_{feat2}_norm'] = (df[feat1] - df[feat2]) / (df[feat1] + df[feat2] + 1e-8)
                        combinations_created += 1
                        
                except Exception as e:
                    continue
                    
                if combinations_created >= max_combinations:
                    break
        
        return df

# =====================================================================
# SISTEMA DE MODELOS ENSEMBLE ADAPTATIVO
# =====================================================================

class AdaptiveEnsembleSystem:
    """Sistema de ensemble que se adapta automaticamente"""
    
    def __init__(self):
        self.base_models = self._initialize_base_models()
        self.meta_model = None
        self.model_weights = {}
        self.model_performances = {}
        self.ensemble_history = []
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def _initialize_base_models(self) -> Dict[str, Any]:
        """Inicializa modelos base"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Adicionar modelo de deep learning se disponível
        if TORCH_AVAILABLE:
            models['deep_neural_network'] = self._create_deep_model()
        
        return models
    
    def _create_deep_model(self):
        """Cria modelo de deep learning"""
        class DeepClassifier(nn.Module):
            def __init__(self, input_size):
                super(DeepClassifier, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)  # Binary classification
                )
            
            def forward(self, x):
                return self.network(x)
        
        return DeepClassifier
    
    def train_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, MLModelPerformance]:
        """Treina ensemble de modelos"""
        
        if len(X) < 50:
            logger.warning("Dados insuficientes para treinar ensemble")
            return {}
        
        # Preparar dados
        X_processed = self._preprocess_features(X)
        
        # Validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        performances = {}
        model_predictions = {}
        
        # Treinar cada modelo base
        for model_name, model in self.base_models.items():
            try:
                performance = self._train_single_model(
                    model, model_name, X_processed, y, tscv
                )
                performances[model_name] = performance
                
                # Guardar predições para meta-modelo
                model_predictions[model_name] = self._get_model_predictions(
                    model, X_processed, y, tscv
                )
                
            except Exception as e:
                logger.error(f"Erro treinando modelo {model_name}: {e}")
                continue
        
        # Treinar meta-modelo (stacking)
        if len(model_predictions) >= 2:
            self._train_meta_model(model_predictions, y)
        
        # Calcular pesos dos modelos
        self._calculate_model_weights(performances)
        
        self.model_performances = performances
        
        return performances
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pré-processa features"""
        
        # Remover features com muitos NaN
        X_clean = X.dropna(axis=1, thresh=len(X)*0.7)
        
        # Preencher NaN restantes
        X_clean = X_clean.fillna(X_clean.median())
        
        # Selecionar melhores features se muitas features
        if X_clean.shape[1] > 50:
            if self.feature_selector is None:
                self.feature_selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(50, X_clean.shape[1])
                )
            
            try:
                y_temp = np.random.randint(0, 2, len(X_clean))  # Dummy target para fit
                X_clean = pd.DataFrame(
                    self.feature_selector.fit_transform(X_clean, y_temp),
                    index=X_clean.index,
                    columns=X_clean.columns[self.feature_selector.get_support()]
                )
            except:
                pass  # Se falhar, usar todas as features
        
        # Escalar features
        try:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_clean),
                index=X_clean.index,
                columns=X_clean.columns
            )
            return X_scaled
        except:
            return X_clean
    
    def _train_single_model(self, model, model_name: str, X: pd.DataFrame, 
                           y: np.ndarray, cv) -> MLModelPerformance:
        """Treina um modelo individual"""
        
        # Validação cruzada
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Treinar modelo completo
        model.fit(X, y)
        
        # Predições
        y_pred = model.predict(X)
        
        # Calcular métricas
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_[0])))
        
        return MLModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_samples=len(X),
            last_trained=datetime.now(),
            feature_importance=feature_importance,
            cross_val_scores=cv_scores.tolist()
        )
    
    def _get_model_predictions(self, model, X: pd.DataFrame, y: np.ndarray, cv) -> np.ndarray:
        """Obtém predições do modelo para meta-aprendizado"""
        
        predictions = np.zeros(len(y))
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y[train_idx]
            
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_val)[:, 1]  # Probabilidade da classe positiva
            predictions[val_idx] = pred_proba
        
        return predictions
    
    def _train_meta_model(self, model_predictions: Dict[str, np.ndarray], y: np.ndarray):
        """Treina meta-modelo para stacking"""
        
        # Criar features do meta-modelo
        meta_features = np.column_stack(list(model_predictions.values()))
        
        # Treinar meta-modelo simples
        self.meta_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        try:
            self.meta_model.fit(meta_features, y)
            logger.info("Meta-modelo treinado com sucesso")
        except Exception as e:
            logger.error(f"Erro treinando meta-modelo: {e}")
            self.meta_model = None
    
    def _calculate_model_weights(self, performances: Dict[str, MLModelPerformance]):
        """Calcula pesos dos modelos baseado na performance"""
        
        # Usar score composto para determinar pesos
        total_score = 0
        model_scores = {}
        
        for model_name, performance in performances.items():
            score = performance.calculate_composite_score()
            model_scores[model_name] = score
            total_score += score
        
        # Normalizar pesos
        if total_score > 0:
            self.model_weights = {
                model: score / total_score 
                for model, score in model_scores.items()
            }
        else:
            # Pesos iguais se não há performance
            num_models = len(performances)
            self.model_weights = {
                model: 1.0 / num_models 
                for model in performances.keys()
            }
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Faz predições usando ensemble"""
        
        if not self.base_models or not self.model_weights:
            raise ValueError("Ensemble não foi treinado")
        
        # Pré-processar features
        X_processed = self._preprocess_features(X)
        
        predictions = []
        probabilities = []
        
        # Coletar predições de todos os modelos
        model_predictions = {}
        for model_name, model in self.base_models.items():
            if model_name in self.model_weights:
                try:
                    pred = model.predict(X_processed)
                    pred_proba = model.predict_proba(X_processed)[:, 1]
                    
                    predictions.append(pred)
                    probabilities.append(pred_proba)
                    model_predictions[model_name] = pred_proba
                    
                except Exception as e:
                    logger.warning(f"Erro na predição do modelo {model_name}: {e}")
                    continue
        
        if not predictions:
            raise ValueError("Nenhum modelo conseguiu fazer predições")
        
        # Ensemble com pesos
        weighted_proba = np.zeros(len(X_processed))
        total_weight = 0
        
        for i, model_name in enumerate(model_predictions.keys()):
            weight = self.model_weights.get(model_name, 0)
            weighted_proba += probabilities[i] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_proba /= total_weight
        
        # Usar meta-modelo se disponível
        if self.meta_model and len(model_predictions) >= 2:
            try:
                meta_features = np.column_stack(list(model_predictions.values()))
                meta_proba = self.meta_model.predict_proba(meta_features)[:, 1]
                
                # Combinar predições ensemble e meta-modelo
                final_proba = (weighted_proba * 0.7) + (meta_proba * 0.3)
            except Exception as e:
                logger.warning(f"Erro no meta-modelo: {e}")
                final_proba = weighted_proba
        else:
            final_proba = weighted_proba
        
        # Converter probabilidades em predições
        final_pred = (final_proba > 0.5).astype(int)
        
        return final_pred, final_proba
    
    def adapt_ensemble(self, new_X: pd.DataFrame, new_y: np.ndarray, 
                      performance_feedback: float):
        """Adapta ensemble baseado em nova performance"""
        
        # Re-treinar modelos com novos dados
        updated_performances = self.train_ensemble(new_X, new_y)
        
        # Ajustar pesos baseado no feedback
        if performance_feedback < 0.6:  # Performance baixa
            # Dar mais peso aos modelos que performaram melhor recentemente
            for model_name in self.model_weights:
                if model_name in updated_performances:
                    recent_performance = updated_performances[model_name].calculate_composite_score()
                    self.model_weights[model_name] *= (1 + recent_performance)
            
            # Normalizar pesos
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                self.model_weights = {
                    model: weight / total_weight 
                    for model, weight in self.model_weights.items()
                }
        
        logger.info(f"Ensemble adaptado. Nova performance esperada baseada em feedback: {performance_feedback:.3f}")

# =====================================================================
# SISTEMA PRINCIPAL DE ML ADAPTATIVO
# =====================================================================

class AdaptiveMLSystem:
    """Sistema principal de Machine Learning Adaptativo"""
    
    def __init__(self, save_directory: str = "ml_models"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        
        # Componentes principais
        self.feature_engineer = DynamicFeatureEngineer()
        self.ensemble_system = AdaptiveEnsembleSystem()
        
        # Estado de aprendizado
        self.learning_state = AdaptiveLearningState()
        self.symbol_models = {}  # Modelos específicos por símbolo
        self.global_model = None  # Modelo global
        
        # Histórico de aprendizado
        self.learning_history = []
        self.feedback_buffer = []
        
        # Configurações adaptativas
        self.min_samples_for_training = 100
        self.retrain_frequency = 50  # Re-treinar a cada N novos samples
        self.performance_window = 20   # Janela para calcular performance média
        
    async def learn_from_trade_result(self, symbol: str, features: pd.DataFrame, 
                                    action: str, result: float):
        """Aprende com resultado de um trade"""
        
        # Converter resultado em label binário
        label = 1 if result > 0 else 0
        
        # Armazenar para aprendizado
        learning_sample = {
            'symbol': symbol,
            'features': features,
            'action': action,
            'result': result,
            'label': label,
            'timestamp': datetime.now()
        }
        
        self.feedback_buffer.append(learning_sample)
        self.learning_state.total_trades_learned += 1
        
        if label == 1:
            self.learning_state.successful_predictions += 1
        else:
            self.learning_state.failed_predictions += 1
        
        # Trigger adaptação se necessário
        if len(self.feedback_buffer) >= self.learning_state.adaptation_trigger_threshold:
            await self._trigger_adaptation()
    
    async def _trigger_adaptation(self):
        """Dispara processo de adaptação"""
        
        logger.info("🧠 Iniciando adaptação do sistema ML...")
        
        try:
            # Processar buffer de feedback
            adaptation_data = self._process_feedback_buffer()
            
            if not adaptation_data:
                return
            
            # Adaptar features
            await self._adapt_features(adaptation_data)
            
            # Re-treinar modelos
            await self._retrain_models(adaptation_data)
            
            # Atualizar estado
            self.learning_state.model_adaptations += 1
            self.learning_state.last_adaptation = datetime.now()
            
            # Ajustar parâmetros de aprendizado
            self._adjust_learning_parameters()
            
            # Limpar buffer
            self.feedback_buffer.clear()
            
            logger.info("✅ Adaptação do sistema ML concluída")
            
        except Exception as e:
            logger.error(f"❌ Erro na adaptação ML: {e}", exc_info=True)
    
    def _process_feedback_buffer(self) -> Dict[str, Any]:
        """Processa buffer de feedback para adaptação"""
        
        if not self.feedback_buffer:
            return {}
        
        # Agrupar por símbolo
        symbol_data = {}
        
        for sample in self.feedback_buffer:
            symbol = sample['symbol']
            if symbol not in symbol_data:
                symbol_data[symbol] = {
                    'features': [],
                    'labels': [],
                    'results': []
                }
            
            symbol_data[symbol]['features'].append(sample['features'])
            symbol_data[symbol]['labels'].append(sample['label'])
            symbol_data[symbol]['results'].append(sample['result'])
        
        # Processar dados por símbolo
        processed_data = {}
        
        for symbol, data in symbol_data.items():
            if len(data['features']) < 5:  # Mínimo de samples
                continue
            
            # Concatenar features
            try:
                features_df = pd.concat(data['features'], ignore_index=True)
                labels_array = np.array(data['labels'])
                results_array = np.array(data['results'])
                
                processed_data[symbol] = {
                    'features': features_df,
                    'labels': labels_array,
                    'results': results_array,
                    'performance': np.mean(results_array)
                }
                
            except Exception as e:
                logger.warning(f"Erro processando dados de {symbol}: {e}")
                continue
        
        return processed_data
    
    async def _adapt_features(self, adaptation_data: Dict[str, Any]):
        """Adapta sistema de features baseado nos resultados"""
        
        # Calcular performance média por feature
        feature_performances = {}
        
        for symbol, data in adaptation_data.items():
            features_df = data['features']
            performance = data['performance']
            
            # Atualizar performance de cada feature
            for feature_name in features_df.columns:
                if feature_name not in feature_performances:
                    feature_performances[feature_name] = []
                feature_performances[feature_name].append(performance)
        
        # Atualizar feature engineer
        for feature_name, performances in feature_performances.items():
            avg_performance = np.mean(performances)
            self.feature_engineer.update_feature_performance(feature_name, avg_performance)
        
        # Criar novas features se performance baixa
        overall_performance = np.mean([data['performance'] for data in adaptation_data.values()])
        
        if overall_performance < 0.4:
            await self._create_new_features(adaptation_data)
    
    async def _create_new_features(self, adaptation_data: Dict[str, Any]):
        """Cria novas features baseado em análise de dados"""
        
        logger.info("🔬 Criando novas features experimentais...")
        
        # Analisar correlações para criar features combinadas
        for symbol, data in adaptation_data.items():
            features_df = data['features']
            labels = data['labels']
            
            # Calcular correlações com o target
            correlations = {}
            for column in features_df.columns:
                if features_df[column].dtype in [np.float64, np.int64]:
                    corr = features_df[column].corr(pd.Series(labels))
                    if not np.isnan(corr):
                        correlations[column] = abs(corr)
            
            # Encontrar features mais correlacionadas
            if correlations:
                top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Criar features combinadas
                for i, (feat1, _) in enumerate(top_features):
                    for feat2, _ in top_features[i+1:]:
                        # Criar nova feature combinada
                        new_feature_name = f"combined_{feat1}_{feat2}_{symbol}"
                        
                        def create_combined_feature(df, f1=feat1, f2=feat2):
                            if f1 in df.columns and f2 in df.columns:
                                return df[f1] * df[f2]  # Multiplicação simples
                            return pd.Series(0, index=df.index)
                        
                        self.feature_engineer.create_custom_feature(
                            new_feature_name, 
                            create_combined_feature
                        )
                        
                        break  # Criar apenas uma nova feature por adaptação
                    break
    
    async def _retrain_models(self, adaptation_data: Dict[str, Any]):
        """Re-treina modelos com novos dados"""
        
        # Re-treinar modelo global
        all_features = []
        all_labels = []
        
        for symbol, data in adaptation_data.items():
            all_features.append(data['features'])
            all_labels.extend(data['labels'])
        
        if all_features:
            global_features = pd.concat(all_features, ignore_index=True)
            global_labels = np.array(all_labels)
            
            if len(global_features) >= self.min_samples_for_training:
                # Calcular performance atual para feedback
                current_performance = np.mean([data['performance'] for data in adaptation_data.values()])
                
                # Adaptar ensemble
                self.ensemble_system.adapt_ensemble(
                    global_features, 
                    global_labels, 
                    current_performance
                )
        
        # Re-treinar modelos por símbolo
        for symbol, data in adaptation_data.items():
            if len(data['features']) >= 20:  # Mínimo para modelo específico
                try:
                    # Criar ensemble específico para o símbolo
                    symbol_ensemble = AdaptiveEnsembleSystem()
                    performances = symbol_ensemble.train_ensemble(
                        data['features'], 
                        data['labels']
                    )
                    
                    if performances:
                        self.symbol_models[symbol] = symbol_ensemble
                        logger.info(f"Modelo específico re-treinado para {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Erro re-treinando modelo para {symbol}: {e}")
    
    def _adjust_learning_parameters(self):
        """Ajusta parâmetros de aprendizado baseado na performance"""
        
        success_rate = self.learning_state.get_success_rate()
        
        # Ajustar threshold de confiança
        if success_rate > 0.7:
            # Alta taxa de sucesso - ser mais seletivo
            self.learning_state.confidence_threshold = min(0.8, self.learning_state.confidence_threshold + 0.05)
        elif success_rate < 0.4:
            # Baixa taxa de sucesso - ser menos seletivo
            self.learning_state.confidence_threshold = max(0.4, self.learning_state.confidence_threshold - 0.05)
        
        # Ajustar trigger de adaptação
        if success_rate < 0.5:
            # Performance baixa - adaptar mais frequentemente
            self.learning_state.adaptation_trigger_threshold = max(5, self.learning_state.adaptation_trigger_threshold - 2)
        else:
            # Performance boa - adaptar menos frequentemente
            self.learning_state.adaptation_trigger_threshold = min(20, self.learning_state.adaptation_trigger_threshold + 1)
        
        logger.info(f"Parâmetros ajustados: Confiança={self.learning_state.confidence_threshold:.2f}, "
                   f"Trigger={self.learning_state.adaptation_trigger_threshold}")
    
    async def predict_trade_outcome(self, symbol: str, features: pd.DataFrame) -> Tuple[float, float]:
        """Prediz resultado de um trade"""
        
        if features.empty:
            return 0.5, 0.0  # Neutral prediction
        
        try:
            # Gerar features otimizadas
            enhanced_features = self.feature_engineer.generate_features(
                features, 
                target_performance=self.learning_state.get_success_rate()
            )
            
            # Usar modelo específico do símbolo se disponível
            if symbol in self.symbol_models:
                try:
                    pred, proba = self.symbol_models[symbol].predict_ensemble(enhanced_features)
                    return float(proba[-1]), float(pred[-1])
                except Exception as e:
                    logger.warning(f"Erro no modelo específico de {symbol}: {e}")
            
            # Fallback para modelo global
            if hasattr(self.ensemble_system, 'base_models') and self.ensemble_system.base_models:
                try:
                    pred, proba = self.ensemble_system.predict_ensemble(enhanced_features)
                    return float(proba[-1]), float(pred[-1])
                except Exception as e:
                    logger.warning(f"Erro no modelo global: {e}")
            
        except Exception as e:
            logger.error(f"Erro na predição para {symbol}: {e}")
        
        return 0.5, 0.0  # Neutral if all fails
    
    async def save_models(self):
        """Salva todos os modelos"""
        
        try:
            # Salvar estado de aprendizado
            state_file = self.save_directory / "learning_state.json"
            with open(state_file, 'w') as f:
                json.dump({
                    'learning_state': self.learning_state.__dict__,
                    'feature_performance_history': self.feature_engineer.feature_performance_history,
                    'model_weights': self.ensemble_system.model_weights,
                }, f, indent=2, default=str)
            
            # Salvar modelos ensemble
            ensemble_file = self.save_directory / "ensemble_system.pkl"
            with open(ensemble_file, 'wb') as f:
                pickle.dump(self.ensemble_system, f)
            
            # Salvar modelos por símbolo
            for symbol, model in self.symbol_models.items():
                symbol_file = self.save_directory / f"model_{symbol}.pkl"
                with open(symbol_file, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info("💾 Modelos ML salvos com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro salvando modelos: {e}")
    
    async def load_models(self):
        """Carrega modelos salvos"""
        
        try:
            # Carregar estado de aprendizado
            state_file = self.save_directory / "learning_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    
                    # Reconstruir learning state
                    state_data = data.get('learning_state', {})
                    for key, value in state_data.items():
                        if hasattr(self.learning_state, key):
                            setattr(self.learning_state, key, value)
                    
                    # Reconstruir feature performance
                    self.feature_engineer.feature_performance_history = data.get('feature_performance_history', {})
                    
                    # Reconstruir model weights
                    self.ensemble_system.model_weights = data.get('model_weights', {})
            
            # Carregar ensemble system
            ensemble_file = self.save_directory / "ensemble_system.pkl"
            if ensemble_file.exists():
                with open(ensemble_file, 'rb') as f:
                    self.ensemble_system = pickle.load(f)
            
            # Carregar modelos por símbolo
            for model_file in self.save_directory.glob("model_*.pkl"):
                symbol = model_file.stem.replace("model_", "")
                with open(model_file, 'rb') as f:
                    self.symbol_models[symbol] = pickle.load(f)
            
            logger.info(f"📂 Modelos ML carregados: {len(self.symbol_models)} específicos")
            
        except Exception as e:
            logger.error(f"❌ Erro carregando modelos: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de aprendizado"""
        
        return {
            'learning_state': self.learning_state.__dict__,
            'total_models': len(self.symbol_models) + (1 if self.ensemble_system else 0),
            'symbol_models': list(self.symbol_models.keys()),
            'feature_count': len(self.feature_engineer.feature_catalog),
            'custom_features': len(self.feature_engineer.custom_features),
            'model_performances': {
                name: perf.__dict__ if hasattr(perf, '__dict__') else perf
                for name, perf in self.ensemble_system.model_performances.items()
            } if hasattr(self.ensemble_system, 'model_performances') else {},
            'recent_feedback_count': len(self.feedback_buffer)
        }