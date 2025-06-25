import pandas as pd
import talib
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore') # Ignora warnings do sklearn/pandas que não são críticos

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Regimes de mercado otimizados para futuros"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT_BULL = "breakout_bull"
    BREAKOUT_BEAR = "breakout_bear"
    SCALPING_FAVORABLE = "scalping_favorable"
    HIGH_LEVERAGE_RISK = "high_leverage_risk"

class SignalStrength(Enum):
    """Força do sinal de trading"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

class RiskLevel(Enum):
    """Níveis de risco adaptados para futuros"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME_LEVERAGE = 6

@dataclass
class FuturesMarketContext:
    """Contexto específico para futuros"""
    regime: MarketRegime
    volatility_percentile: float
    volume_profile: str
    time_of_day: str
    correlation_strength: float
    market_sentiment: float
    leverage_environment: str
    funding_rate_pressure: float
    liquidity_score: float
    gap_risk: float
    # Adicionado para Open Interest (se for implementado na coleta)
    open_interest_growth: float = 0.0

@dataclass
class TradingSignal:
    """Estrutura para sinal de trading aprimorada para futuros"""
    action: str  # BUY (LONG), SELL (SHORT), HOLD, CLOSE_LONG, CLOSE_SHORT
    strength: SignalStrength
    price: float
    confidence: float  # 0-100%
    ml_probability: float  # Probabilidade do ML
    indicators: Dict[str, float]
    reasons: List[str]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.MODERATE
    position_size: float = 0.1
    market_context: Optional[FuturesMarketContext] = None
    expected_return: Optional[float] = None # Retorno esperado da estratégia
    sharpe_estimate: Optional[float] = None
    max_drawdown_est: Optional[float] = None
    leverage_recommendation: float = 1.0
    funding_consideration: float = 0.0
    urgency_score: float = 0.5
    symbol: str = "UNKNOWN" # Adicionado para melhor rastreamento
    # Adicionado para a seleção de oportunidades
    opportunity_score: float = 0.0

# =======================================================================
# >>>>> CORREÇÃO: MOVIDO PerformanceMetrics PARA CIMA <<<<<
# =======================================================================
@dataclass
class PerformanceMetrics:
    """Métricas de performance para futuros"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    avg_leverage_used: float = 1.0
    funding_cost_impact: float = 0.0
# =======================================================================
# >>>>> FIM DA CORREÇÃO <<<<<
# =======================================================================

class FuturesFeatureEngineering:
    """Engenharia de features específica para futuros"""
    
    @staticmethod
    def add_futures_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features específicas para futuros"""
        
        # Garante que df é uma cópia para evitar SettingWithCopyWarning
        df = df.copy()

        # Certifica-se de que colunas essenciais existem e não são totalmente NaN para operações
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns or df[col].isnull().all():
                logger.debug(f"⚠️ FeatureEngineering: Coluna '{col}' ausente ou toda NaN. Retornando df sem features derivadas de '{col}'.")
                # Se uma coluna crítica estiver faltando ou for toda NaN, não podemos criar features que dependem dela.
                # É melhor retornar o DataFrame como está, e os métodos subsequentes que usam essas features devem lidar com NaNs.
                return df 

        # Preenche NaNs nas colunas críticas antes de cálculos onde a ordem importa (diff, shift)
        # Atenção: Isso pode introduzir viés se houver muitos NaNs. A validação inicial é crucial.
        # Ajuste: Apenas `ffill` e `bfill` devem ser usados aqui, o `fillna(0)` final será feito no process_market_data
        df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill')
        # df[required_cols] = df[required_cols].fillna(0) # Removido: O fillna(0) final será feito no process_market_data

        # Momentum e volatilidade intraday
        if len(df) > 1:
            df['intraday_return'] = (df['close'] - df['open']) / (df['open'].replace(0, np.nan) + 1e-8)
            df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)
        else: # Para DataFrames com 0 ou 1 linha
            df['intraday_return'] = np.nan
            df['overnight_gap'] = np.nan

        df['true_range_pct'] = (df['high'] - df['low']) / (df['close'].replace(0, np.nan) + 1e-8)
        
        # Features de liquidez (proxy via volume)
        if len(df) > 3: # Necessita de pelo menos 3 para pct_change(3)
            df['volume_momentum'] = df['volume'].pct_change(3)
            df['volume_acceleration'] = df['volume_momentum'].diff()
        else:
            df['volume_momentum'] = np.nan
            df['volume_acceleration'] = np.nan

        # O método .rolling() exige um número mínimo de períodos, que deve ser maior que 0
        if len(df['volume'].dropna()) >= 20:
            df['liquidity_score'] = df['volume'].rolling(20, min_periods=10).rank(pct=True) # min_periods para flexibilidade
        else:
            df['liquidity_score'] = np.nan # Se não há dados suficientes para rolling

        # Pressure metrics para long/short
        # Adiciona uma pequena constante (1e-8) ao denominador para evitar divisão por zero
        # Lida com casos onde high - low pode ser zero, adicionando +1e-8
        df['long_pressure'] = np.where(df['close'] > df['open'], 
                                     df['volume'] * (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8),
                                     0)
        df['short_pressure'] = np.where(df['close'] < df['open'], 
                                      df['volume'] * (df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8),
                                      0)
        
        # Breakout features (críticos para futuros)
        if len(df['close'].dropna()) >= 20: # Necessita de pelo menos 20 para rolling max/min
            df['breakout_strength'] = np.where(
                df['close'] > df['high'].rolling(20, min_periods=10).max().shift(1),
                (df['close'] - df['high'].rolling(20, min_periods=10).max().shift(1)) / (df['close'].replace(0, np.nan) + 1e-8),
                np.where(
                    df['close'] < df['low'].rolling(20, min_periods=10).min().shift(1),
                    (df['low'].rolling(20, min_periods=10).min().shift(1) - df['close']) / (df['close'].replace(0, np.nan) + 1e-8),
                    0
                )
            )
        else:
            df['breakout_strength'] = np.nan

        # Volatility regimes específicos para futuros
        returns = df['close'].pct_change()
        if len(returns.dropna()) >= 12: # Necessita de pelo menos 12 para rolling std
            df['realized_vol_5m'] = returns.rolling(12, min_periods=6).std() * np.sqrt(288)  # 5min periods in day
        else:
            df['realized_vol_5m'] = np.nan

        if len(df['realized_vol_5m'].dropna()) >= 100: # Necessita de 100 para quantile
            df['vol_regime_intraday'] = np.where(
                df['realized_vol_5m'] > df['realized_vol_5m'].rolling(100, min_periods=50).quantile(0.8), 'HIGH',
                np.where(df['realized_vol_5m'] < df['realized_vol_5m'].rolling(100, min_periods=50).quantile(0.3), 'LOW', 'NORMAL')
            )
        else:
            df['vol_regime_intraday'] = 'NORMAL' # Default se não há dados suficientes

        # Mean reversion signals
        if len(df['close'].dropna()) >= 20: # Necessita de 20 para rolling mean/std
            mean_20 = df['close'].rolling(20, min_periods=10).mean()
            std_20 = df['close'].rolling(20, min_periods=10).std()
            df['mean_reversion_signal'] = (df['close'] - mean_20) / (std_20 + 1e-8)
        else:
            df['mean_reversion_signal'] = np.nan
        
        # Trend acceleration
        if len(df['close'].dropna()) >= 20: # Necessita de 20 para sma_slow
            sma_fast = df['close'].rolling(5, min_periods=3).mean()
            sma_slow = df['close'].rolling(20, min_periods=10).mean()
            df['trend_acceleration'] = (sma_fast - sma_slow).diff()
        else:
            df['trend_acceleration'] = np.nan
        
        # Adicionar funding rate se disponível (específico Gate.io)
        # O data_collector DEVE fornecer esta coluna
        if 'funding_rate' not in df.columns:
            df['funding_rate'] = np.nan  # Placeholder se não disponível da coleta
        
        # Calcular pressão do funding rate
        if len(df['funding_rate'].dropna()) >= 8:
            df['funding_pressure'] = df['funding_rate'].rolling(8, min_periods=1).mean() * 100
        else:
            df['funding_pressure'] = np.nan
        
        # Preenche NaNs introduzidos pelos cálculos (rolling, diff, etc.) com 0 ou ffill/bfill
        # Removido: O fillna final será feito no process_market_data

        return df
    
    @staticmethod
    def add_futures_timing_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de timing específicas para futuros 24/7"""
        df = df.copy()

        if df.empty:
            logger.debug("DataFrame vazio em add_futures_timing_features.")
            return df
        
        # Garante que o índice é DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            # Tenta converter o índice se não for DatetimeIndex
            try:
                # Corrigido para lidar com index.name
                if df.index.name is None:
                    df.index.name = 'timestamp' # Dá um nome temporário se não tiver
                
                df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
                df.dropna(subset=[df.index.name], inplace=True) # Remove linhas com índices NaT
                if df.empty:
                    logger.warning("DataFrame vazio após tentar converter índice para datetime.")
                    return df
            except Exception as e:
                logger.warning(f"Não foi possível converter índice para datetime para features de timing: {e}. Pulando timing features.")
                df['market_session'] = 'UNKNOWN'
                df['is_weekend_effect'] = 0
                for session in ['US_MAIN', 'ASIA', 'EUROPE', 'GLOBAL']:
                    df[f'vol_vs_{session.lower()}'] = 1.0
                return df


        # Ajustar sessões para horários UTC (Gate.io opera 24/7)
        hour = df.index.hour
        conditions = [
            (hour >= 0) & (hour < 8),   # Asia: 00-07h UTC
            (hour >= 8) & (hour < 16),  # Europe: 08-15h UTC
            (hour >= 16) & (hour < 24)  # US: 16-23h UTC
        ]
        choices = ['ASIA', 'EUROPE', 'US']
        df['market_session'] = np.select(conditions, choices, default='GLOBAL') # Default GLOBAL para o restante

        # Volatilidade por sessão
        # Atenção: df['close'].pct_change().std() pode ser NaN para DF pequeno
        # Certificar que global_vol não é NaN ou 0 antes de usar como divisor
        if 'close' in df.columns and len(df['close'].dropna()) > 1:
            global_vol = df['close'].pct_change().std()
            if pd.isna(global_vol) or global_vol == 0:
                global_vol = 1e-8 # Evita divisão por zero
        else:
            global_vol = 1e-8 # Se não há dados suficientes para calcular vol global

        for session in ['ASIA', 'EUROPE', 'US', 'GLOBAL']:
            session_mask = df['market_session'] == session
            if session_mask.sum() > 10 and 'close' in df.columns: # Precisa de pelo menos 10 pontos para calcular o std
                session_returns = df.loc[session_mask, 'close'].pct_change().dropna()
                session_vol = session_returns.std()
                
                if pd.isna(session_vol) or session_vol == 0:
                    df[f'vol_vs_{session.lower()}'] = 1.0 # Volatilidade neutra
                else:
                    # Cálculo de vol_vs_session com rolling para suavizar e min_periods
                    # A divisão por global_vol não é ideal aqui se a intenção é comparar com a vol da sessão atual.
                    # Correção: A intenção é "volatilidade da barra atual vs volatilidade típica daquela sessão".
                    # Mantenho o cálculo original que usa a volatilidade da sessão inteira como base para comparação.
                    df[f'vol_vs_{session.lower()}'] = df['close'].pct_change().rolling(20, min_periods=5).std() / (session_vol + 1e-8)
                    df[f'vol_vs_{session.lower()}'].fillna(1.0, inplace=True) # Preenche NaNs resultantes do rolling
            else:
                df[f'vol_vs_{session.lower()}'] = 1.0 # Volatilidade neutra se não houver dados suficientes

        # Weekend/holiday effects
        df['is_weekend_effect'] = ((df.index.dayofweek == 4) | (df.index.dayofweek == 6)).astype(int)
        
        return df.copy()

class FuturesMLPredictor:
    """Preditor ML otimizado para futuros"""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        # Usar parâmetros mais agressivos para futuros
        self.model = RandomForestClassifier(
            n_estimators=150,  # Mais árvores
            random_state=42, 
            max_depth=12,  # Mais profundidade
            min_samples_split=3,  # Mais sensível
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.last_accuracy = 0.0 
    
    def prepare_futures_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features específicas para futuros, garantindo tipos numéricos
        e tratando NaNs e Infinitos de forma robusta.
        """
        
        # PASSO 0: Verificação inicial do DataFrame de entrada
        if df.empty:
            logger.debug("DataFrame de entrada para prepare_futures_features está vazio.")
            return pd.DataFrame()
        
        # Cria uma cópia para evitar SettingWithCopyWarning e trabalha com ela.
        df_cleaned = df.copy()
        
        # Aplica conversão para numérico em TODAS as colunas que podem ser features.
        # errors='coerce' transformará o que não for número em NaN.
        for col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype(np.float64)

        # Preenche quaisquer NaNs remanescentes com 0.0 (ou outro valor padrão adequado para features).
        df_cleaned = df_cleaned.fillna(0.0) 
        
        # Substitui infinitos (+/-inf) por 0.0.
        df_cleaned = df_cleaned.replace([np.inf, -np.inf], 0.0)

        # Inicia o DataFrame para features ML, garantindo o mesmo índice do df_cleaned
        ml_features = pd.DataFrame(index=df_cleaned.index)
        
        # --- Lógica de criação das features (como no seu código existente) ---
        # Certifique-se de que cada `if` verifica a existência da coluna no `df_cleaned`
        # e que as operações usam `df_cleaned[...]` para acessar as Series do Pandas.

        # Features básicas de momentum (mais sensitivas para futuros)
        if 'rsi' in df_cleaned.columns:
            ml_features['rsi'] = df_cleaned['rsi']
            ml_features['rsi_velocity'] = df_cleaned['rsi'].diff().fillna(0)
            ml_features['rsi_divergence'] = ((df_cleaned['close'].diff(5).fillna(0) > 0) != (df_cleaned['rsi'].diff(5).fillna(0) > 0)).astype(int)
        
        # MACD com mais sensibilidade
        if 'macd_hist' in df_cleaned.columns:
            ml_features['macd_hist'] = df_cleaned['macd_hist']
            ml_features['macd_momentum'] = df_cleaned['macd_hist'].diff().fillna(0)
            ml_features['macd_acceleration'] = df_cleaned['macd_hist'].diff(2).fillna(0)
        
        # Features de trend específicas para futuros
        if 'ema_fast' in df_cleaned.columns and 'ema_slow' in df_cleaned.columns and 'close' in df_cleaned.columns:
            # Adicionado 1e-8 para evitar divisão por zero no denominador
            ml_features['ema_spread'] = (df_cleaned['ema_fast'] - df_cleaned['ema_slow']) / (df_cleaned['close'] + 1e-8)
            ml_features['ema_slope'] = df_cleaned['ema_fast'].diff(3) / (df_cleaned['close'] + 1e-8)
            ml_features['price_vs_ema'] = (df_cleaned['close'] - df_cleaned['ema_fast']) / (df_cleaned['ema_fast'] + 1e-8)
        
        # Volume features (críticas para futuros)
        if 'volume_ratio' in df_cleaned.columns and 'volume' in df_cleaned.columns:
            ml_features['volume_ratio'] = df_cleaned['volume_ratio']
            if len(df_cleaned['volume'].dropna()) >= 20: # Requisito para rolling mean
                ml_features['volume_trend'] = df_cleaned['volume'].rolling(5, min_periods=1).mean() / (df_cleaned['volume'].rolling(20, min_periods=1).mean() + 1e-8)
            else:
                ml_features['volume_trend'] = 0.0 # Se não há dados suficientes, define como 0.0
        # Corrigido: 'volume' também deve ser verificado para volume_trend
        if 'volume' in df_cleaned.columns:
             if len(df_cleaned['volume'].dropna()) >= 20:
                ml_features['volume_trend'] = df_cleaned['volume'].rolling(5, min_periods=1).mean() / (df_cleaned['volume'].rolling(20, min_periods=1).mean() + 1e-8)
             else:
                ml_features['volume_trend'] = 0.0


        if 'long_pressure' in df_cleaned.columns and 'short_pressure' in df_cleaned.columns:
            ml_features['pressure_ratio'] = df_cleaned['long_pressure'] / (df_cleaned['short_pressure'] + 1e-8)
            ml_features['net_pressure'] = df_cleaned['long_pressure'] - df_cleaned['short_pressure']
        
        # Volatility features específicas
        if 'realized_vol_5m' in df_cleaned.columns:
            if len(df_cleaned['realized_vol_5m'].dropna()) >= 50:
                ml_features['vol_regime'] = (df_cleaned['realized_vol_5m'] > df_cleaned['realized_vol_5m'].rolling(50, min_periods=1).median()).astype(int)
                ml_features['vol_percentile'] = df_cleaned['realized_vol_5m'].rolling(100, min_periods=1).rank(pct=True)
            else:
                ml_features['vol_regime'] = 0
                ml_features['vol_percentile'] = 0.5 # Valor neutro se não há dados suficientes
        
        # Breakout features
        if 'breakout_strength' in df_cleaned.columns:
            ml_features['breakout_strength'] = df_cleaned['breakout_strength']
            ml_features['breakout_momentum'] = df_cleaned['breakout_strength'].rolling(3, min_periods=1).sum().fillna(0)
        
        # Mean reversion features
        if 'mean_reversion_signal' in df_cleaned.columns:
            ml_features['mean_reversion'] = df_cleaned['mean_reversion_signal']
            ml_features['mean_reversion_extreme'] = (abs(df_cleaned['mean_reversion_signal']) > 2).astype(int)
        
        # Price action features para futuros
        if 'close' in df_cleaned.columns and 'open' in df_cleaned.columns:
            ml_features['body_size'] = abs(df_cleaned['close'] - df_cleaned['open']) / (df_cleaned['close'] + 1e-8)
            ml_features['intraday_return'] = (df_cleaned['close'] - df_cleaned['open']) / (df_cleaned['open'] + 1e-8)
            if len(df_cleaned) > 1 and df_cleaned['close'].shift(1).iloc[-1] != 0: # Corrigido para evitar erro de divisão por zero
                ml_features['gap_size'] = (df_cleaned['open'] - df_cleaned['close'].shift(1)) / (df_cleaned['close'].shift(1) + 1e-8)
            else:
                ml_features['gap_size'] = 0.0
        
        # Features de Bollinger Bands
        if 'bb_position' in df_cleaned.columns:
            ml_features['bb_position'] = df_cleaned['bb_position']
            ml_features['bb_extreme'] = ((df_cleaned['bb_position'] < 10) | (df_cleaned['bb_position'] > 90)).astype(int)
            
        if 'bb_width' in df_cleaned.columns:
            if len(df_cleaned['bb_width'].dropna()) >= 20:
                ml_features['bb_squeeze'] = (df_cleaned['bb_width'] < df_cleaned['bb_width'].rolling(20, min_periods=10).quantile(0.2)).astype(int)
            else:
                ml_features['bb_squeeze'] = 0 # Define como 0 se não há dados suficientes
        
        # Features de ADX
        if 'adx' in df_cleaned.columns:
            ml_features['adx_strength'] = df_cleaned['adx'] / 50
            ml_features['adx_rising'] = (df_cleaned['adx'].diff().fillna(0) > 0).astype(int)
        
        # Timing features
        if 'market_session' in df_cleaned.columns:
            if not df_cleaned['market_session'].isnull().all():
                for session in ['US_MAIN', 'ASIA', 'EUROPE', 'GLOBAL']: # Ajuste para as novas sessões
                    ml_features[f'session_{session.lower()}'] = (df_cleaned['market_session'] == session).astype(int)
            else: # Se a coluna market_session é toda NaN, preenche com 0
                for session in ['US_MAIN', 'ASIA', 'EUROPE', 'GLOBAL']:
                    ml_features[f'session_{session.lower()}'] = 0

        # Funding rate features
        if 'funding_pressure' in df_cleaned.columns and 'funding_rate' in df_cleaned.columns: # Corrigido: verificar funding_rate também
            ml_features['funding_pressure'] = df_cleaned['funding_pressure']
            ml_features['funding_negative'] = (df_cleaned['funding_rate'] < 0).astype(int)
        else: # Se funding_rate ou funding_pressure não existem, preenche com 0
            ml_features['funding_pressure'] = 0.0
            ml_features['funding_negative'] = 0
        
        # PASSO FINAL: Garantir que todas as features ML são numéricas e não contêm NaNs/Infinitos
        for col in ml_features.columns:
            ml_features[col] = pd.to_numeric(ml_features[col], errors='coerce').astype(np.float64)
        
        ml_features = ml_features.fillna(0.0) 
        ml_features = ml_features.replace([np.inf, -np.inf], 0.0)
        
        return ml_features
    
    def create_futures_targets(self, df: pd.DataFrame, forward_periods: int = 3) -> np.ndarray:
        """Cria targets otimizados para futuros (mais agressivos)"""
        # Garante que df tem dados suficientes para shift e que o denominador não é zero
        if len(df) <= forward_periods or (df['close'].iloc[:-forward_periods] == 0).any() or 'close' not in df.columns: # Corrigido: verificar 'close'
            return np.full(len(df), np.nan)
            
        future_returns = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Thresholds menores para futuros (devido à maior frequência de sinais)
        buy_threshold = 0.003
        sell_threshold = -0.003
        
        targets = np.where(future_returns > buy_threshold, 1,
                          np.where(future_returns < sell_threshold, -1, 0))
        
        return targets
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_SYMBOL_TRAIN') # Pega o nome do símbolo
        features = self.prepare_futures_features(df)
        targets = self.create_futures_targets(df)

        # Remove NaN e Inf/non-finite de features e targets
        valid_idx = np.isfinite(features.values).all(axis=1) & np.isfinite(targets)
        features = features[valid_idx]
        targets = targets[valid_idx]

        if len(features) < self.lookback_periods:
            logger.warning(f"{symbol_name}: Dados insuficientes para treinar ML de futuros. Necessário {self.lookback_periods}, encontrado {len(features)}.")
            self.is_trained = False
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n_samples": 0}

        # --- NOVO: Verificação de classes para estratificação ---
        unique_targets, counts = np.unique(targets, return_counts=True)
        # Corrigido: garantir que há pelo menos 2 amostras por classe para estratificação
        if len(unique_targets) < 2 or any(c < 2 for c in counts):
            logger.warning(f"{symbol_name}: Classes insuficientes ou muito poucas amostras para estratificação no ML. Usando split simples.")
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.3, random_state=42, stratify=targets
            )

        # Escalar features
        try:
            # Corrigido: se X_train ou X_test forem vazios, scaler vai falhar
            if X_train.empty:
                logger.warning(f"{symbol_name}: X_train vazio, impossível escalar para treinamento ML.")
                self.is_trained = False
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n_samples": 0}

            X_train_scaled = self.scaler.fit_transform(X_train.astype(np.float64))
            X_test_scaled = self.scaler.transform(X_test.astype(np.float64))
        except Exception as e:
            logger.error(f"{symbol_name}: Erro ao escalar features para treinamento ML: {e}. Provavelmente dados zero/constantes.")
            self.is_trained = False
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n_samples": 0}
        
        # Verificar se os dados escalados contêm NaN ou Inf
        if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_test_scaled).all():
            logger.error(f"{symbol_name}: Dados escalados para treinamento ML contêm NaN ou Inf. Abortando treinamento.")
            self.is_trained = False
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n_samples": 0}

        # Treinar modelo
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Avaliar
        y_pred = self.model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        self.last_accuracy = accuracy
        
        # Feature importance
        self.feature_importance = dict(zip(features.columns, self.model.feature_importances_))
        
        # Estatísticas por classe
        class_stats = {}
        for cls in unique_targets:
            cls_mask = targets == cls
            class_stats[f'class_{cls}_count'] = cls_mask.sum()
        
        logger.info(f"🤖 {symbol_name}: ML FUTUROS treinado - Accuracy: {accuracy:.3f}")
        logger.info(f"📊 {symbol_name}: Distribuição: {class_stats}")
        
        return {
            "accuracy": accuracy,
            "n_samples": len(X_train),
            "features": len(features.columns),
            "class_distribution": class_stats
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Prediz sinal usando ML para futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_SYMBOL_PREDICT') # Pega o nome do símbolo
        if not self.is_trained:
            logger.warning(f"{symbol_name}: ML não treinado para predição. Retornando neutro.")
            return 0, 0.4
        
        features = self.prepare_futures_features(df)
        
        # Validação final do features para predição
        if features.empty or len(features) == 0:
            logger.warning(f"{symbol_name}: Features para predição ML são vazias após preparação. Retornando neutro.")
            return 0, 0.4

        # Usar apenas o último registro e garantir que não há NaNs/Infinitos
        latest_features = features.iloc[-1:].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        try:
            # Garante que todas as colunas de X_train_scaled são as mesmas e na mesma ordem
            # Este é um ponto comum de erro se as features do prepare_futures_features mudam entre treino e predição.
            # É bom que prepare_futures_features já lide com NaNs e Inf, mas uma checagem final.
            if self.scaler.feature_names_in_ is not None:
                # Corrigido: se a feature mais recente não tem uma coluna que o scaler espera, preencher com 0
                reordered_features = pd.DataFrame(0.0, index=latest_features.index, columns=self.scaler.feature_names_in_)
                for col in latest_features.columns:
                    if col in reordered_features.columns:
                        reordered_features[col] = latest_features[col]
                latest_features = reordered_features
            else: # Se o scaler ainda não foi treinado e não tem feature_names_in_
                logger.warning(f"{symbol_name}: Scaler não tem feature_names_in_. Isso pode ocorrer se o ML não foi treinado adequadamente.")
                return 0, 0.4 # Retorna neutro se o scaler não está pronto

            scaled_features = self.scaler.transform(latest_features.astype(np.float64))
            
            if not np.isfinite(scaled_features).all():
                logger.warning(f"{symbol_name}: Dados escalados para predição ML contêm NaN ou Inf. Retornando neutro.")
                return 0, 0.4

            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            confidence = np.max(probabilities)
            
            # Boost de confiança para futuros (mercado mais líquido)
            confidence = min(0.95, confidence * 1.2)
            
            return int(prediction), float(confidence)
        except Exception as e:
            logger.warning(f"{symbol_name}: Erro na predição ML de futuros: {e}", exc_info=True)
            return 0, 0.4

class FuturesRiskManager:
    """Gerenciador de risco específico para futuros"""
    
    def __init__(self, config: Dict):
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.15)  # Maior para futuros
        self.max_single_position = config.get('max_single_position', 0.08)  # 8% por posição
        self.max_leverage = config.get('max_leverage', 3.0)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.var_confidence = config.get('var_confidence', 0.05)
        self.funding_rate_threshold = config.get('funding_rate_threshold', 0.01)  # 1% funding
        
        # [NOVA LÓGICA] Alavancagem máxima puxada da configuração (padrão 5.0 para Gate.io)
        self.max_leverage = config.get('max_leverage', 5.0)
        
    def calculate_optimal_leverage(self, 
                                 signal_confidence: float,
                                 volatility_percentile: float,
                                 market_regime: MarketRegime) -> float:
        """Calcula alavancagem ótima baseada no contexto"""
        
        base_leverage = 1.0
        
        # Ajustar por confiança do sinal
        confidence_multiplier = min(2.5, signal_confidence / 40)  # Max 2.5x a 100% confidence
        
        # Ajustar por volatilidade (menos alavancagem em alta volatilidade)
        # [NOVA LÓGICA] Reduzir alavancagem em alta volatilidade
        vol_multiplier = max(0.25, 1.5 - volatility_percentile) # 0.25 é o mínimo
        
        # Ajustar por regime de mercado
        regime_multipliers = {
            MarketRegime.TRENDING_BULL: 1.3,
            MarketRegime.TRENDING_BEAR: 1.3,
            MarketRegime.BREAKOUT_BULL: 1.5,
            MarketRegime.BREAKOUT_BEAR: 1.5,
            MarketRegime.RANGING_HIGH_VOL: 0.7, # Reduz alavancagem em alta volatilidade lateral
            MarketRegime.RANGING_LOW_VOL: 1.0,
            MarketRegime.SCALPING_FAVORABLE: 2.0, # Aumenta para scalping
            MarketRegime.HIGH_LEVERAGE_RISK: 0.5 # Reduz drasticamente em risco de alavancagem alta
        }
        
        regime_multiplier = regime_multipliers.get(market_regime, 1.0)
        
        # Calcular alavancagem final
        optimal_leverage = base_leverage * confidence_multiplier * vol_multiplier * regime_multiplier
        optimal_leverage = max(1.0, min(self.max_leverage, optimal_leverage))
        
        return optimal_leverage
    
    def calculate_position_size_futures(self, 
                                      signal: 'TradingSignal',
                                      portfolio_value: float,
                                      current_leverage: float = 1.0) -> float:
        """Calcula tamanho da posição específico para futuros"""
        
        # Base size menor para futuros devido à alavancagem
        base_size = self.max_single_position * 0.8  # 80% do máximo
        
        # Ajustar por confiança
        confidence_adj = signal.confidence / 100
        
        # Ajustar por força do sinal
        strength_multipliers = {
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.7,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.EXTREME: 1.5
        }
        
        strength_adj = strength_multipliers[signal.strength]
        
        # Ajustar por nível de risco
        risk_multipliers = {
            RiskLevel.VERY_LOW: 1.5,
            RiskLevel.LOW: 1.2,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 0.7,
            RiskLevel.VERY_HIGH: 0.4,
            RiskLevel.EXTREME_LEVERAGE: 0.2
        }
        risk_adj = risk_multipliers[signal.risk_level]
        
        # Calcular tamanho final
        final_size = base_size * confidence_adj * strength_adj * risk_adj
        
        # [NOVA LÓGICA] Ajustar posição para alta volatilidade
        if signal.market_context and signal.market_context.volatility_percentile > 0.8:
            final_size *= 0.5 # Reduz o tamanho da posição pela metade em alta volatilidade
            logger.debug(f"Reduzindo tamanho da posição devido a alta volatilidade ({signal.market_context.volatility_percentile:.2f}).")
        
        # Aplicar limites
        final_size = max(0.01, min(self.max_single_position, final_size))
        
        return final_size
    
    def assess_futures_market_risk(self, df: pd.DataFrame) -> RiskLevel:
        """
        Avalia risco específico para futuros.
        Corrigido para lidar com símbolos 'UNKNOWN' e DataFrames vazios/curtos.
        """
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_RISK_ASSESS') # Pega o nome do símbolo

        if symbol_name == 'UNKNOWN' or df.empty or len(df) < 20: # Corrigido: Tratar UNKNOWN
            logger.warning(f"🚨 {symbol_name}: Dados insuficientes ou símbolo UNKNOWN para avaliação de risco ({len(df)} barras). Retornando HIGH_RISK.")
            return RiskLevel.HIGH
        
        risk_score = 0
        
        # Volatilidade intraday
        if 'realized_vol_5m' in df.columns and not df['realized_vol_5m'].isnull().all() and len(df['realized_vol_5m'].dropna()) >= 100:
            vol_percentile = df['realized_vol_5m'].rolling(100, min_periods=50).rank(pct=True).iloc[-1]
            if not pd.isna(vol_percentile):
                if vol_percentile > 0.9: risk_score += 3
                elif vol_percentile > 0.8: risk_score += 2
                elif vol_percentile > 0.7: risk_score += 1
        
        # Volume anômalo (risco de manipulação)
        if 'volume_ratio' in df.columns and not df['volume_ratio'].isnull().all():
            if not pd.isna(df['volume_ratio'].iloc[-1]):
                vol_ratio = df['volume_ratio'].iloc[-1]
                if vol_ratio > 5: risk_score += 2
                elif vol_ratio > 3: risk_score += 1
        
        # Gaps significativos (risco de liquidez)
        if 'overnight_gap' in df.columns and not df['overnight_gap'].isnull().all():
            # Tratar NaNs nos gaps
            recent_gaps = abs(df['overnight_gap'].tail(5).dropna()).max()
            if not pd.isna(recent_gaps):
                if recent_gaps > 0.03: risk_score += 2  # Gap > 3%
                elif recent_gaps > 0.02: risk_score += 1  # Gap > 2%
        
        # Breakouts extremos (risco de reversão)
        if 'breakout_strength' in df.columns and not df['breakout_strength'].isnull().all():
            if not pd.isna(df['breakout_strength'].iloc[-1]):
                breakout = abs(df['breakout_strength'].iloc[-1])
                if breakout > 0.05: risk_score += 2
                elif breakout > 0.03: risk_score += 1
        
        # Drawdown atual
        if 'close' in df.columns and not df['close'].isnull().all():
            if len(df['close'].dropna()) > 1:
                peak = df['close'].expanding(min_periods=1).max()
                if not peak.empty and not pd.isna(peak.iloc[-1]) and not pd.isna(df['close'].iloc[-1]) and peak.iloc[-1] > 0: # Corrigido: verificar se peak.iloc[-1] > 0
                    current_dd = abs((df['close'].iloc[-1] - peak.iloc[-1]) / (peak.iloc[-1] + 1e-8))
                    if current_dd > 0.15: risk_score += 2
                    elif current_dd > 0.10: risk_score += 1
        
        # [NOVA LÓGICA] Detecção de funding rate negativo como risco extremo
        if 'funding_rate' in df.columns and not df['funding_rate'].isnull().iloc[-1]:
            if df['funding_rate'].iloc[-1] < -0.0005: # -0.05%
                risk_score += 5 # Penalidade severa, pode levar a EXTREME_LEVERAGE
                logger.warning(f"🚨 Risco: Funding rate negativo extremo ({df['funding_rate'].iloc[-1]:.5f}) para {symbol_name}.")

        # [NOVA LÓGICA] Identificar squeezes de liquidez / baixa liquidez como risco
        if 'liquidity_score' in df.columns and not df['liquidity_score'].isnull().iloc[-1]:
            if df['liquidity_score'].iloc[-1] < 0.2:
                risk_score += 4 # Penalidade alta para baixa liquidez
                logger.warning(f"🚨 Risco: Baixa liquidez detectada ({df['liquidity_score'].iloc[-1]:.2f}) para {symbol_name}.")

        # Classificar risco
        if risk_score >= 8:
            return RiskLevel.EXTREME_LEVERAGE
        elif risk_score >= 6:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

class FuturesAdvancedStrategy:
    """Estratégia avançada otimizada para futuros perpétuos"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Required bars deve ser o máximo de todos os períodos de indicadores e ML
        # O ML precisa de 50 (lookback_periods), então 50 é o mínimo aqui.
        # Os indicadores também precisam de ~21 (ema_slow), 18 (bb), 12 (rsi/adx).
        self.required_bars = max(config.get('ema_slow', 21), 
                                 config.get('bb_period', 18), 
                                 config.get('rsi_period', 12), 
                                 config.get('adx_period', 12),
                                 50) # 50 é o mínimo para ML
        
        # Parâmetros otimizados para futuros
        self.ema_fast = config.get('ema_fast', 8)
        self.ema_slow = config.get('ema_slow', 21)
        self.rsi_period = config.get('rsi_period', 12)
        self.bb_period = config.get('bb_period', 18)
        self.bb_std = config.get('bb_std', 2.0)
        self.adx_period = config.get('adx_period', 12)
        
        # Parâmetros específicos para futuros
        self.breakout_lookback = config.get('breakout_lookback', 15)
        self.scalping_threshold = config.get('scalping_threshold', 0.002)
        self.momentum_periods = config.get('momentum_periods', 5)
        self.volatility_adjustment = config.get('volatility_adjustment', True)
        
        # Componentes
        self.ml_predictor = FuturesMLPredictor(lookback_periods=self.required_bars) # ML agora usa required_bars
        self.risk_manager = FuturesRiskManager(config)
        self.feature_engineering = FuturesFeatureEngineering()
        
        # Estado
        self.last_ml_train = 0
        self.performance_metrics = PerformanceMetrics()
        self.ml_retrain_interval = config.get('ml_retrain_interval', 500)
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores otimizados para futuros.
        Corrigido para lidar com DataFrames vazios/curtos e NaNs.
        """
        df = df.copy()
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_INDICATORS') # Pega o nome do nome do símbolo

        # 1. Garante que as colunas essenciais existem e são numéricas
        required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlcv_cols:
            if col not in df.columns:
                logger.warning(f"⚠️ {symbol_name}: Coluna '{col}' ausente. Criando com NaNs.")
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
        # 2. Preenche NaNs nas colunas essenciais (ffill/bfill para manter a tendência, depois 0)
        # Isso é CRÍTICO antes de calcular indicadores.
        df[required_ohlcv_cols] = df[required_ohlcv_cols].fillna(method='ffill').fillna(method='bfill')

        # 3. Verifica se há dados suficientes após a limpeza para cálculos iniciais
        if len(df) < 2: # Mínimo para pct_change ou diff simples
            logger.warning(f"⚠️ {symbol_name}: Dados insuficientes ({len(df)} barras) após limpeza para cálculo de indicadores básicos. Retornando DF sem indicadores calculados.")
            return df # Retorna o DataFrame, mesmo que incompleto, para permitir outros processamentos
        
        # Agora sim, extrair os valores como numpy arrays para as funções TALib
        # E garantir que são float64, pois algumas funções TALib são sensíveis
        open_np = df['open'].values.astype(np.float64)
        high_np = df['high'].values.astype(np.float64)
        low_np = df['low'].values.astype(np.float64)
        close_np = df['close'].values.astype(np.float64)
        volume_np = df['volume'].values.astype(np.float64)

        # === MÉDIAS MÓVEIS OTIMIZADAS PARA FUTUROS ===
        # Adicionado verificação de dados suficientes para cada cálculo do TA-Lib
        if len(close_np) >= self.ema_fast:
            df['ema_fast'] = talib.EMA(close_np, timeperiod=self.ema_fast)
        else: df['ema_fast'] = np.nan

        if len(close_np) >= self.ema_slow:
            df['ema_slow'] = talib.EMA(close_np, timeperiod=self.ema_slow)
        else: df['ema_slow'] = np.nan

        if len(close_np) >= 5:
            df['ema_very_fast'] = talib.EMA(close_np, timeperiod=5)
        else: df['ema_very_fast'] = np.nan

        if len(close_np) >= 20:
            df['sma_20'] = talib.SMA(close_np, timeperiod=20)
        else: df['sma_20'] = np.nan

        if len(close_np) >= 15:
            df['tema'] = talib.TEMA(close_np, timeperiod=15)
        else: df['tema'] = np.nan
        
        # === OSCILADORES SENSÍVEIS ===
        if len(close_np) >= self.rsi_period:
            df['rsi'] = talib.RSI(close_np, timeperiod=self.rsi_period)
        else: df['rsi'] = np.nan

        if len(close_np) >= 7:
            df['rsi_fast'] = talib.RSI(close_np, timeperiod=7)
        else: df['rsi_fast'] = np.nan

        if len(close_np) >= 10:
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_np, low_np, close_np, 
                                                  fastk_period=10, 
                                                  slowk_period=3, 
                                                  slowd_period=3)
        else:
            df['stoch_k'], df['stoch_d'] = np.nan, np.nan

        if len(close_np) >= 10:
            df['williams_r'] = talib.WILLR(high_np, low_np, close_np, timeperiod=10)
        else: df['williams_r'] = np.nan

        if len(close_np) >= 15:
            df['cci'] = talib.CCI(high_np, low_np, close_np, timeperiod=15)
        else: df['cci'] = np.nan
        
        # === MACD OTIMIZADO ===
        if len(close_np) >= max(8, 21): # fastperiod e slowperiod
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close_np, 
                                                                   fastperiod=8,
                                                                   slowperiod=21,
                                                                   signalperiod=7)
            df['macd_slope'] = df['macd'].diff(2)
        else:
            df['macd'], df['macd_signal'], df['macd_hist'], df['macd_slope'] = np.nan, np.nan, np.nan, np.nan
        
        # === BOLLINGER BANDS ===
        if len(close_np) >= self.bb_period:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close_np, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
            )
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8) * 100
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8) * 100
            
            if len(df['bb_width'].dropna()) >= 15:
                df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(15, min_periods=5).median() * 0.8 
            else:
                df['bb_squeeze'] = np.nan
        else:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = np.nan, np.nan, np.nan
            df['bb_width'], df['bb_position'], df['bb_squeeze'] = np.nan, np.nan, np.nan


        # === TREND INDICATORS ===
        if len(high_np) >= self.adx_period:
            df['adx'] = talib.ADX(high_np, low_np, close_np, timeperiod=self.adx_period)
            df['plus_di'] = talib.PLUS_DI(high_np, low_np, close_np, timeperiod=self.adx_period)
            df['minus_di'] = talib.MINUS_DI(high_np, low_np, close_np, timeperiod=self.adx_period)
        else:
            df['adx'], df['plus_di'], df['minus_di'] = np.nan, np.nan, np.nan

        if len(high_np) >= 15:
            df['aroon_up'], df['aroon_down'] = talib.AROON(high_np, low_np, timeperiod=15)
        else: df['aroon_up'], df['aroon_down'] = np.nan, np.nan
        
        # === VOLUME INDICATORS PARA FUTUROS ===
        if len(close_np) >= 1 and len(volume_np) >= 1: # OBV e AD precisam de pelo menos 1
            df['obv'] = talib.OBV(close_np, volume_np)
            df['ad'] = talib.AD(high_np, low_np, close_np, volume_np)
        else: df['obv'], df['ad'] = np.nan, np.nan

        if len(volume_np) >= 15:
            df['volume_sma'] = talib.SMA(volume_np, timeperiod=15)
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        else: df['volume_sma'], df['volume_ratio'] = np.nan, np.nan
        
        # VWAP simplificado
        if len(volume_np) >= 20 and len(close_np) >= 20: # Requere volume para rolling sum
            df['vwap'] = (df['close'] * df['volume']).rolling(20, min_periods=10).sum() / (df['volume'].rolling(20, min_periods=10).sum() + 1e-8)
            df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8) * 100
        else:
            df['vwap'], df['vwap_deviation'] = np.nan, np.nan

        # === VOLATILIDADE ===
        if len(high_np) >= 12:
            df['atr'] = talib.ATR(high_np, low_np, close_np, timeperiod=12)
            df['natr'] = talib.NATR(high_np, low_np, close_np, timeperiod=12)
        else: df['atr'], df['natr'] = np.nan, np.nan
        
        # === CANDLESTICK PATTERNS RELEVANTES PARA FUTUROS ===
        # Certifique-se que open_np está definido e tem dados
        if len(open_np) > 0 and len(high_np) > 0 and len(low_np) > 0 and len(close_np) > 0:
            df['doji'] = talib.CDLDOJI(open_np, high_np, low_np, close_np)
            df['hammer'] = talib.CDLHAMMER(open_np, high_np, low_np, close_np)
            df['engulfing'] = talib.CDLENGULFING(open_np, high_np, low_np, close_np)
            df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_np, high_np, low_np, close_np)
            df['bullish_marubozu'] = talib.CDLMARUBOZU(open_np, high_np, low_np, close_np)
            df['bearish_marubozu'] = talib.CDLMARUBOZU(open_np, high_np, low_np, close_np) # Marubozu pode ser de baixa também
            df['doji_star'] = talib.CDLDOJISTAR(open_np, high_np, low_np, close_np)
            df['harami'] = talib.CDLHARAMI(open_np, high_np, low_np, close_np)
            df['morning_star'] = talib.CDLMORNINGSTAR(open_np, high_np, low_np, close_np)
            df['evening_star'] = talib.CDLEVENINGSTAR(open_np, high_np, low_np, close_np)
        else:
            for pattern_col in ['doji', 'hammer', 'engulfing', 'shooting_star', 'bullish_marubozu', 'bearish_marubozu', 'doji_star', 'harami', 'morning_star', 'evening_star']:
                df[pattern_col] = np.nan
        
        # === FEATURES ESPECÍFICAS PARA FUTUROS ===
        df = self.feature_engineering.add_futures_specific_features(df)
        df = self.feature_engineering.add_futures_timing_features(df)
        
        # === REGIME DETECTION PARA FUTUROS ===
        df['market_regime'] = self.detect_futures_market_regime(df)
        
        # Preencher NaNs remanescentes (após todos os cálculos)
        # Use um ffill/bfill antes do 0.0 para preencher NaNs que são realmente "faltantes"
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0) # Substitui infinitos

        logger.debug(f"✅ {symbol_name}: Indicadores calculados. DataFrame shape: {df.shape}")
        logger.debug(f"✅ {symbol_name}: NaNs restantes após cálculo: {df.isnull().sum().sum()}")
        #logger.debug(f"✅ {symbol_name}: Últimas 3 linhas:\n{df.tail(3)}") # Removido para evitar logs muito grandes

        return df
    
    def detect_futures_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta regime específico para futuros.
        Corrigido para lidar com DataFrames vazios/curtos e NaNs.
        """
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_REGIME')
        
        if df.empty or len(df) < 30: # Mínimo para detecção de regime
            logger.warning(f"⚠️ Regime Detection para {symbol_name}: Dados insuficientes ({len(df)} < 30). Retornando regime padrão.")
            return pd.Series(MarketRegime.RANGING_LOW_VOL.value, index=df.index, dtype=object)
        
        # Filtrar as colunas necessárias e remover NaNs antes do loop
        required_regime_cols = ['close', 'volume_ratio', 'adx', 'realized_vol_5m', 'breakout_strength', 'funding_rate', 'liquidity_score']
        df_filtered = df[required_regime_cols].dropna()

        if df_filtered.empty or len(df_filtered) < 30: # Re-verifica após dropna
            logger.warning(f"⚠️ Regime Detection para {symbol_name}: Dados insuficientes após remoção de NaNs para cálculo de regime. Retornando regime padrão.")
            return pd.Series(MarketRegime.RANGING_LOW_VOL.value, index=df.index, dtype=object)

        regimes = []
        
        for i in range(len(df)): # Iterar sobre o DF original para manter o índice
            # Usar iloc para acessar os dados, mas com fallback para NaN
            current_close = df.iloc[i]['close'] if 'close' in df.columns and not pd.isna(df.iloc[i]['close']) else np.nan
            
            # Garante que a janela tem dados válidos
            if i < self.breakout_lookback or pd.isna(current_close): # Aumentado para 20
                regimes.append(MarketRegime.RANGING_LOW_VOL.value)
                continue
            
            window = df.iloc[max(0, i-self.breakout_lookback):i+1] # Usar breakout_lookback do config
            
            # Métricas. Tratar NaNs para evitar erros.
            prev_close = window['close'].iloc[0] if not window['close'].empty and not pd.isna(window['close'].iloc[0]) else np.nan
            
            price_change = (current_close - prev_close) / (prev_close + 1e-8) if not pd.isna(current_close) and not pd.isna(prev_close) and prev_close != 0 else 0.0
            
            # Volatilidade da janela
            window_returns = window['close'].pct_change().dropna()
            volatility = window_returns.std() if len(window_returns) > 1 else 0.0

            adx = window['adx'].iloc[-1] if 'adx' in window.columns and not window['adx'].isnull().iloc[-1] else 20
            volume_surge = window['volume_ratio'].iloc[-1] if 'volume_ratio' in window.columns and not window['volume_ratio'].isnull().iloc[-1] else 1
            
            # Detectar breakout
            breakout = 0.0
            if 'breakout_strength' in window.columns and not window['breakout_strength'].isnull().iloc[-1]:
                breakout = abs(window['breakout_strength'].iloc[-1])
                if breakout > 0.02:  # Breakout significativo
                    regime = MarketRegime.BREAKOUT_BULL.value if price_change > 0 else MarketRegime.BREAKOUT_BEAR.value
                    regimes.append(regime)
                    continue
            
            # Detectar condições de scalping
            if 'realized_vol_5m' in window.columns and not window['realized_vol_5m'].isnull().iloc[-1]:
                if volatility > 0.01 and volume_surge > 1.5 and adx < 25:
                    regimes.append(MarketRegime.SCALPING_FAVORABLE.value)
                    continue
            
            # Detectar risco de alta alavancagem
            if 'realized_vol_5m' in window.columns and len(window['realized_vol_5m'].dropna()) >= 100:
                rolling_std_vol_series = window['realized_vol_5m'].rolling(100, min_periods=50).std()
                if not rolling_std_vol_series.empty and not pd.isna(rolling_std_vol_series.iloc[-1]) and rolling_std_vol_series.iloc[-1] > 0:
                    rolling_std_vol = rolling_std_vol_series.iloc[-1]
                    if volatility > rolling_std_vol * 2:
                        regimes.append(MarketRegime.HIGH_LEVERAGE_RISK.value)
                        continue

            # [NOVA LÓGICA] Adicionar detecção de funding rate negativo
            if 'funding_rate' in window.columns and not window['funding_rate'].isnull().iloc[-1]:
                if window['funding_rate'].iloc[-1] < -0.0005: # -0.05%
                    regimes.append(MarketRegime.HIGH_LEVERAGE_RISK.value) # Sinal de alto risco
                    continue
            
            # [NOVA LÓGICA] Identificar squeezes de liquidez / baixa liquidez como risco
            if 'liquidity_score' in window.columns and not window['liquidity_score'].isnull().iloc[-1]:
                if window['liquidity_score'].iloc[-1] < 0.2: # Score de liquidez baixo
                    regimes.append(MarketRegime.HIGH_LEVERAGE_RISK.value) # Sinal de alto risco
                    continue
            
            # Regimes tradicionais
            if adx > 25 and abs(price_change) > volatility * 2:
                if price_change > 0:
                    regime = MarketRegime.TRENDING_BULL.value
                else:
                    regime = MarketRegime.TRENDING_BEAR.value
            elif 'realized_vol_5m' in window.columns and len(window['realized_vol_5m'].dropna()) >= 30:
                rolling_std_vol_series = window['realized_vol_5m'].rolling(30, min_periods=10).std()
                if not rolling_std_vol_series.empty and not pd.isna(rolling_std_vol_series.iloc[-1]) and rolling_std_vol_series.iloc[-1] > 0:
                    rolling_std_vol = rolling_std_vol_series.iloc[-1]
                    if volatility > rolling_std_vol * 1.5:
                        regime = MarketRegime.RANGING_HIGH_VOL.value
                    else:
                        regime = MarketRegime.RANGING_LOW_VOL.value
                else:
                    regime = MarketRegime.RANGING_LOW_VOL.value # Fallback se o rolling std for inválido
            else:
                regime = MarketRegime.RANGING_LOW_VOL.value
            
            regimes.append(regime)
        
        # Retorna uma Series com o mesmo índice do DataFrame original
        return pd.Series(regimes, index=df.index, dtype=object)

    def analyze_trend_futures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Análise de tendência otimizada para futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_TREND')
        # Corrigido: Len do df deve ser suficiente para as EMAs
        if df.empty or len(df) < self.ema_slow:
            logger.debug(f"Análise de tendência para {symbol_name}: Dados insuficientes.")
            return {'score': 0.0, 'strength': 0.0, 'short_momentum': 0.0, 'medium_momentum': 0.0, 'ema_alignment': 0.0, 'aroon_trend': 0.0}

        latest = df.iloc[-1]
        
        # Múltiplas EMAs (mais peso para EMAs rápidas)
        ema_alignment = 0
        if 'ema_very_fast' in df.columns and 'ema_fast' in df.columns and 'ema_slow' in df.columns and 'close' in df.columns:
            if not pd.isna(latest['ema_very_fast']) and not pd.isna(latest['ema_fast']) and not pd.isna(latest['ema_slow']) and not pd.isna(latest['close']):
                if latest['close'] > latest['ema_very_fast'] > latest['ema_fast'] > latest['ema_slow']:
                    ema_alignment = 1
                elif latest['close'] < latest['ema_very_fast'] < latest['ema_fast'] < latest['ema_slow']:
                    ema_alignment = -1
                elif latest['close'] > latest['ema_fast']:
                    ema_alignment = 0.5
                elif latest['close'] < latest['ema_fast']:
                    ema_alignment = -0.5
        
        # ADX com threshold menor para futuros (usando adx_period do config)
        adx_strength = min(latest['adx'] / 40, 1.0) if 'adx' in df.columns and not pd.isna(latest['adx']) else 0
        
        # Momentum de curto prazo (mais importante em futuros)
        short_momentum = 0.0
        if len(df) >= 5 and 'close' in df.columns and not df['close'].isnull().iloc[-5] and not pd.isna(latest['close']): # Corrigido: verificar NaNs
            if df['close'].iloc[-5] != 0:
                short_momentum = (latest['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        medium_momentum = 0.0
        if len(df) >= 15 and 'close' in df.columns and not df['close'].isnull().iloc[-15] and not pd.isna(latest['close']): # Corrigido: verificar NaNs
            if df['close'].iloc[-15] != 0:
                medium_momentum = (latest['close'] - df['close'].iloc[-15]) / df['close'].iloc[-15]
        
        # Aroon para confirmar tendência
        aroon_trend = 0
        if 'aroon_up' in df.columns and 'aroon_down' in df.columns:
            if not pd.isna(latest['aroon_up']) and not pd.isna(latest['aroon_down']):
                aroon_diff = latest['aroon_up'] - latest['aroon_down']
                aroon_trend = np.tanh(aroon_diff / 50)
        
        # Score combinado (mais peso para momentum curto)
        trend_score = (
            ema_alignment * 0.3 + 
            np.tanh(short_momentum * 20) * 0.4 + # Ajustado peso
            np.tanh(medium_momentum * 10) * 0.2 + # Ajustado peso
            adx_strength * aroon_trend * 0.1
        )
        
        return {
            'score': trend_score,
            'strength': adx_strength,
            'short_momentum': short_momentum,
            'medium_momentum': medium_momentum,
            'ema_alignment': ema_alignment,
            'aroon_trend': aroon_trend
        }
    
    def analyze_momentum_futures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Análise de momentum otimizada para futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_MOMENTUM')
        if df.empty:
            logger.debug(f"Análise de momentum para {symbol_name}: DataFrame vazio.")
            return {'score': 0.0, 'rsi_signal': 0.0, 'macd_signal': 0.0, 'stoch_signal': 0.0, 'williams_signal': 0.0}

        latest = df.iloc[-1]
        
        # RSI duplo (normal + rápido)
        rsi_signal = 0
        if 'rsi_fast' in df.columns and 'rsi' in df.columns:
            if not pd.isna(latest['rsi_fast']) and not pd.isna(latest['rsi']):
                rsi_fast = latest['rsi_fast']
                rsi_normal = latest['rsi']
                
                if rsi_fast < 25 or rsi_normal < 30: # Sobrevendido forte
                    rsi_signal = 1
                elif rsi_fast > 75 or rsi_normal > 70: # Sobrecomprado forte
                    rsi_signal = -1
                else:
                    rsi_signal = (50 - (rsi_fast + rsi_normal) / 2) / 25 # Normalizado entre -1 e 1
        
        # MACD mais sensível
        macd_signal = 0
        if 'macd_hist' in df.columns and not pd.isna(latest['macd_hist']):
            if len(df) >= 2 and not pd.isna(df['macd_hist'].iloc[-2]):
                # Cruzamento de linha zero do MACD Histograma
                if latest['macd_hist'] > 0 and df['macd_hist'].iloc[-2] <= 0:
                    macd_signal = 1.5 # Sinal forte de compra
                elif latest['macd_hist'] < 0 and df['macd_hist'].iloc[-2] >= 0:
                    macd_signal = -1.5 # Sinal forte de venda
                else:
                    macd_signal = np.tanh(latest['macd_hist'] * 50) # Tanh para normalizar score
            else: # Se não houver dados anteriores para diff
                macd_signal = np.tanh(latest['macd_hist'] * 50)
        
        # Stochastic
        stoch_signal = 0
        if 'stoch_k' in df.columns and not pd.isna(latest['stoch_k']):
            if latest['stoch_k'] < 15: # Sobrevendido
                stoch_signal = 1
            elif latest['stoch_k'] > 85: # Sobrecomprado
                stoch_signal = -1
            else:
                stoch_signal = (50 - latest['stoch_k']) / 35 # Normalizado
        
        # Williams %R
        williams_signal = 0
        if 'williams_r' in df.columns and not pd.isna(latest['williams_r']):
            if latest['williams_r'] < -85: # Sobrevendido
                williams_signal = 1
            elif latest['williams_r'] > -15: # Sobrecomprado
                williams_signal = -1
            else:
                williams_signal = (latest['williams_r'] + 50) / 35 # Normalizado
        
        # Score combinado com mais peso para MACD em futuros
        momentum_score = (
            rsi_signal * 0.25 + 
            macd_signal * 0.45 +
            stoch_signal * 0.20 + 
            williams_signal * 0.10
        )
        
        return {
            'score': momentum_score,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'stoch_signal': stoch_signal,
            'williams_signal': williams_signal
        }
    
    def analyze_volume_futures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Análise de volume específica para futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_VOLUME')
        if df.empty:
            logger.debug(f"Análise de volume para {symbol_name}: DataFrame vazio.")
            return {'score': 0.0, 'volume_strength': 0.0, 'obv_trend': 0.0, 'pressure_score': 0.0, 'vwap_signal': 0.0, 'liquidity_score': 0.0}

        latest = df.iloc[-1]
        
        # Volume ratio mais sensível
        vol_strength = 0
        if 'volume_ratio' in df.columns and not pd.isna(latest['volume_ratio']):
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 2.5: # Volume muito acima da média
                vol_strength = 1.0
            elif vol_ratio > 1.5:
                vol_strength = 0.7
            elif vol_ratio > 1.2:
                vol_strength = 0.4
            elif vol_ratio < 0.8: # Volume abaixo da média
                vol_strength = -0.3
            else:
                vol_strength = (vol_ratio - 1) / 2 # Normalizado
        
        # OBV trend
        obv_trend = 0
        if 'obv' in df.columns and len(df['obv'].dropna()) >= 10:
            obv_ma = df['obv'].rolling(10, min_periods=5).mean()
            if not obv_ma.empty and not pd.isna(latest['obv']) and not pd.isna(obv_ma.iloc[-1]) and df['obv'].std() > 0:
                obv_trend = np.tanh((latest['obv'] - obv_ma.iloc[-1]) / (df['obv'].std())) # Normalizado
            else:
                obv_trend = 0.0
        
        # Pressure específico para futuros
        pressure_score = 0
        if 'long_pressure' in df.columns and 'short_pressure' in df.columns:
            if not pd.isna(latest['long_pressure']) and not pd.isna(latest['short_pressure']):
                lp = latest['long_pressure']
                sp = latest['short_pressure']
                if lp + sp > 0: # Evita divisão por zero
                    pressure_score = (lp - sp) / (lp + sp) # Normalizado entre -1 e 1
                    if vol_strength > 0.5: # Confirmação por volume
                        pressure_score *= 1.5
        
        # VWAP deviation
        vwap_signal = 0
        if 'vwap_deviation' in df.columns and not pd.isna(latest['vwap_deviation']):
            vwap_dev = latest['vwap_deviation']
            vwap_signal = np.tanh(vwap_dev / 2) # Normalizado
        
        # Liquidity score
        liquidity_score = latest.get('liquidity_score', 0.5) # Default 0.5 se não calculado
        
        volume_score = (
            vol_strength * 0.3 + 
            obv_trend * 0.25 + 
            pressure_score * 0.25 + 
            vwap_signal * 0.15 +
            (liquidity_score - 0.5) * 0.05 # Pequeno ajuste baseado na liquidez geral
        )
        
        return {
            'score': volume_score,
            'volume_strength': vol_strength,
            'obv_trend': obv_trend,
            'pressure_score': pressure_score,
            'vwap_signal': vwap_signal,
            'liquidity_score': liquidity_score
        }
    
    def analyze_volatility_futures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Análise de volatilidade específica para futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_VOLATILITY')
        if df.empty:
            logger.debug(f"Análise de volatilidade para {symbol_name}: DataFrame vazio.")
            return {'score': 0.0, 'bb_signal': 0.0, 'atr_signal': 0.0, 'volatility_percentile': 0.5, 'squeeze': 0.0, 'tr_signal': 0.0}

        latest = df.iloc[-1]
        
        # Bollinger Bands com thresholds mais agressivos
        bb_signal = 0
        if 'bb_position' in df.columns and not pd.isna(latest['bb_position']):
            bb_pos = latest['bb_position']
            if bb_pos < 5: # Muito abaixo da banda inferior (sobrevendido)
                bb_signal = 1.2
            elif bb_pos < 15:
                bb_signal = 0.8
            elif bb_pos > 95: # Muito acima da banda superior (sobrecomprado)
                bb_signal = -1.2
            elif bb_pos > 85:
                bb_signal = -0.8
            else:
                bb_signal = (50 - bb_pos) / 30 # Normalizado
        
        # ATR para stop loss adaptativo
        atr_signal = 0
        if 'atr' in df.columns and len(df['atr'].dropna()) >= 15:
            atr_ma = df['atr'].rolling(15, min_periods=5).mean()
            if not atr_ma.empty and not pd.isna(latest['atr']) and not pd.isna(atr_ma.iloc[-1]) and atr_ma.iloc[-1] > 0:
                atr_ratio = latest['atr'] / atr_ma.iloc[-1]
                atr_signal = np.tanh(-(atr_ratio - 1)) # Sinal negativo para ATR alto (maior volatilidade)
        
        # Realized volatility regime
        vol_regime_signal = 0
        if 'realized_vol_5m' in df.columns and len(df['realized_vol_5m'].dropna()) >= 50:
            vol_percentile_series = df['realized_vol_5m'].rolling(50, min_periods=25).rank(pct=True)
            if not vol_percentile_series.empty and not pd.isna(vol_percentile_series.iloc[-1]):
                vol_percentile = vol_percentile_series.iloc[-1]
                if vol_percentile < 0.3: # Baixa volatilidade (favorável para alguns setups)
                    vol_regime_signal = 0.8
                elif vol_percentile > 0.8: # Alta volatilidade (desfavorável para outros)
                    vol_regime_signal = -0.5
                else:
                    vol_regime_signal = 0.5 - vol_percentile # Normalizado
            else:
                vol_regime_signal = 0.5 # Default se NaN
        else:
            vol_regime_signal = 0.5 # Default se dados insuficientes
        
        # BB Squeeze
        squeeze_signal = 0
        if 'bb_squeeze' in df.columns and not pd.isna(latest['bb_squeeze']):
            squeeze_signal = 1.0 if latest['bb_squeeze'] else 0
        
        # True range percentage
        tr_signal = 0
        if 'true_range_pct' in df.columns and len(df['true_range_pct'].dropna()) >= 15:
            tr_pct = latest['true_range_pct']
            tr_ma_series = df['true_range_pct'].rolling(15, min_periods=5).mean()
            if not tr_ma_series.empty and not pd.isna(tr_pct) and not pd.isna(tr_ma_series.iloc[-1]) and tr_ma_series.iloc[-1] > 0:
                tr_signal = np.tanh(-(tr_pct / tr_ma_series.iloc[-1] - 1)) # Sinal negativo para True Range alto
            else:
                tr_signal = 0.0 # Default se NaN
        else:
            tr_signal = 0.0 # Default se dados insuficientes
        
        volatility_score = (
            bb_signal * 0.35 + 
            atr_signal * 0.25 + 
            vol_regime_signal * 0.20 + 
            squeeze_signal * 0.15 + 
            tr_signal * 0.05
        )
        
        return {
            'score': volatility_score,
            'bb_signal': bb_signal,
            'atr_signal': atr_signal,
            'volatility_percentile': vol_regime_signal + 0.5, # Retorna um valor entre 0 e 1 aproximadamente
            'squeeze': squeeze_signal,
            'tr_signal': tr_signal
        }
    
    def analyze_breakouts_futures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Análise específica de breakouts para futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_BREAKOUTS')
        if df.empty:
            logger.debug(f"Análise de breakouts para {symbol_name}: DataFrame vazio.")
            return {'score': 0.0, 'breakout_strength': 0.0, 'range_breakout': 0.0, 'volume_confirmation': 0.0}

        latest = df.iloc[-1]
        
        breakout_score = 0
        breakout_strength = 0
        
        # Breakout strength
        if 'breakout_strength' in df.columns and not pd.isna(latest['breakout_strength']):
            breakout_strength = latest['breakout_strength']
            if abs(breakout_strength) > 0.02: # Breakout de 2%
                breakout_score = np.tanh(breakout_strength * 50) # Normalizado
        
        # Range breakout
        range_breakout = 0
        if len(df) >= 20 and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            if not pd.isna(recent_high) and not pd.isna(recent_low) and latest['close'] > 0:
                range_size = (recent_high - recent_low) / latest['close'] # Normalizado pelo preço
                
                if latest['close'] > recent_high and range_size > 1e-8: # Preço acima do high
                    range_breakout = min(1.0, (latest['close'] - recent_high) / (range_size * latest['close'] + 1e-8)) # Normalizado
                elif latest['close'] < recent_low and range_size > 1e-8: # Preço abaixo do low
                    range_breakout = max(-1.0, (latest['close'] - recent_low) / (range_size * latest['close'] + 1e-8)) # Normalizado
        
        # Volume confirmation
        volume_confirmation = 0
        if 'volume_ratio' in df.columns and not pd.isna(latest['volume_ratio']):
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 1.5: # Volume 50% acima da média
                volume_confirmation = min(1.0, (vol_ratio - 1) / 2) # Normalizado
        
        # Combine signals
        final_breakout_score = (
            breakout_score * 0.4 + 
            range_breakout * 0.4 + 
            volume_confirmation * 0.2
        )
        
        return {
            'score': final_breakout_score,
            'breakout_strength': breakout_strength,
            'range_breakout': range_breakout,
            'volume_confirmation': volume_confirmation
        }
    
    def analyze_scalping_futures(self, df: pd.DataFrame) -> Dict[str, float]:
        """Análise específica para scalping em futuros"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_SCALPING')
        if df.empty:
            logger.debug(f"Análise de scalping para {symbol_name}: DataFrame vazio.")
            return {'score': 0.0, 'volatility_favorable': False, 'spread_favorable': False, 'volume_favorable': False, 'mean_reversion_opportunity': False}

        latest = df.iloc[-1]
        
        # Condições ideais para scalping
        scalping_score = 0
        
        # Volatilidade moderada
        volatility_favorable = False
        if 'realized_vol_5m' in df.columns and len(df['realized_vol_5m'].dropna()) >= 50:
            vol_percentile_series = df['realized_vol_5m'].rolling(50, min_periods=25).rank(pct=True)
            if not vol_percentile_series.empty and not pd.isna(vol_percentile_series.iloc[-1]):
                vol_percentile = vol_percentile_series.iloc[-1]
                if 0.3 <= vol_percentile <= 0.7: # Volatilidade entre 30 e 70 percentil
                    volatility_favorable = True
                    scalping_score += 0.3 * 0.8
                elif (vol_percentile < 0.2 or vol_percentile > 0.8): # Volatilidade muito baixa ou muito alta
                    scalping_score += 0.3 * (-0.5)
        
        # Spread implícito baixo
        spread_favorable = False
        if 'true_range_pct' in df.columns and len(df['true_range_pct'].dropna()) >= 15:
            tr_pct = latest['true_range_pct']
            tr_ma_series = df['true_range_pct'].rolling(15, min_periods=5).mean()
            if not tr_ma_series.empty and not pd.isna(tr_pct) and not pd.isna(tr_ma_series.iloc[-1]) and tr_ma_series.iloc[-1] > 0:
                if tr_pct / tr_ma_series.iloc[-1] < 1.2: # True Range não muito acima da média
                    spread_favorable = True
                    scalping_score += 0.25 * 0.6
        
        # Volume adequado
        volume_favorable = False
        if 'liquidity_score' in df.columns and not pd.isna(latest['liquidity_score']):
            liq_score = latest['liquidity_score']
            if liq_score > 0.6: # Acima da liquidez média
                volume_favorable = True
                scalping_score += 0.25 * 0.7
        
        # Mean reversion potential
        mean_reversion_opportunity = False
        if 'mean_reversion_signal' in df.columns and not pd.isna(latest['mean_reversion_signal']):
            mr_signal = latest['mean_reversion_signal']
            if abs(mr_signal) > 1.5: # Preço desviou muito da média
                mean_reversion_opportunity = True
                scalping_score += 0.2 * (np.tanh(-abs(mr_signal)) + 0.5) # Quanto mais extremo, maior o potencial de reversão
        
        return {
            'score': scalping_score,
            'volatility_favorable': volatility_favorable,
            'spread_favorable': spread_favorable,
            'volume_favorable': volume_favorable,
            'mean_reversion_opportunity': mean_reversion_opportunity
        }
    
    def generate_futures_signal(self, 
                              df: pd.DataFrame, 
                              current_price: float,
                              portfolio_value: float = 100000) -> TradingSignal:
        """
        Gera sinal otimizado para futuros.
        Corrigido para lidar com DataFrames vazios/curtos e símbolos UNKNOWN.
        """
        
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_SIGNAL_GEN') # Pega o nome do símbolo
        
        # Validações iniciais
        if symbol_name == 'UNKNOWN' or current_price <= 0 or not np.isfinite(current_price) or df.empty or len(df) < self.required_bars:
            logger.warning(f"{symbol_name}: Dados insuficientes ou símbolo UNKNOWN. Retornando HOLD.")
            return TradingSignal("HOLD", SignalStrength.WEAK, current_price if current_price > 0 else 0, 0, 0, {}, ["Dados insuficientes/inválidos ou símbolo UNKNOWN"], symbol=symbol_name)
        
        # Treinar ML mais frequentemente para futuros
        if (len(df) - self.last_ml_train) > self.ml_retrain_interval or not self.ml_predictor.is_trained:
            logger.info(f"🤖 {symbol_name}: Treinando ML para futuros (dados: {len(df)} barras)")
            try:
                training_result = self.ml_predictor.train(df) # Passa o df já processado
                self.last_ml_train = len(df)
                logger.info(f"📊 {symbol_name}: ML futuros treinado: accuracy={training_result.get('accuracy', 0):.3f}")
            except Exception as e:
                logger.warning(f"⚠️ {symbol_name}: Erro ao treinar ML para futuros: {e}")
        
        try:
            # Análises principais
            trend_analysis = self.analyze_trend_futures(df)
            momentum_analysis = self.analyze_momentum_futures(df)
            volume_analysis = self.analyze_volume_futures(df)
            volatility_analysis = self.analyze_volatility_futures(df)
            breakout_analysis = self.analyze_breakouts_futures(df)
            scalping_analysis = self.analyze_scalping_futures(df)
            
            logger.debug(f"🔍 {symbol_name} análises: trend={trend_analysis['score']:.3f}, "
                        f"momentum={momentum_analysis['score']:.3f}, volume={volume_analysis['score']:.3f}, "
                        f"volatility={volatility_analysis['score']:.3f}, breakout={breakout_analysis['score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Erro nas análises para {symbol_name}: {e}", exc_info=True)
            return TradingSignal("HOLD", SignalStrength.WEAK, current_price, 0, 0, {}, 
                               [f"Erro nas análises: {str(e)[:50]}"], symbol=symbol_name)
        
        # Predição ML
        try:
            ml_signal, ml_confidence = self.ml_predictor.predict(df) # Passa o df já processado
            logger.debug(f"🤖 {symbol_name} ML: signal={ml_signal}, confidence={ml_confidence:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ {symbol_name}: Erro na predição ML para futuros: {e}")
            ml_signal, ml_confidence = 0, 0.4
        
        # Contexto do mercado
        try:
            market_context = self.analyze_futures_market_context(df)
        except Exception as e:
            logger.error(f"❌ Erro no contexto para {symbol_name}: {e}")
            market_context = self.create_default_futures_context()
        
        # Scoring integrado com pesos otimizados para futuros
        traditional_score = (
            trend_analysis['score'] * 0.25 +
            momentum_analysis['score'] * 0.30 +
            volume_analysis['score'] * 0.15 +
            volatility_analysis['score'] * 0.10 +
            breakout_analysis['score'] * 0.15 +
            scalping_analysis['score'] * 0.05
        )
        
        # Combinação ML + Tradicional
        ml_weight = min(ml_confidence * 0.4, 0.3) if self.ml_predictor.is_trained else 0.1
        traditional_weight = 1 - ml_weight
        
        final_score = traditional_score * traditional_weight + ml_signal * ml_weight
        
        # Ajustes por regime
        regime_multiplier = self.get_futures_regime_multiplier(market_context.regime, final_score)
        final_score *= regime_multiplier
        
        # Determinar ação e força
        action, strength = self.score_to_futures_action(final_score)
        
        # Calcular confiança para futuros
        confidence = self.calculate_futures_confidence(
            traditional_score, ml_confidence, market_context, volatility_analysis, breakout_analysis
        )
        
        # Risk management específico para futuros
        risk_level = self.risk_manager.assess_futures_market_risk(df) # Passa o df já processado
        
        # Alavancagem recomendada
        optimal_leverage = self.risk_manager.calculate_optimal_leverage(
            confidence, volatility_analysis['volatility_percentile'], market_context.regime
        )
        
        # Position size
        temp_signal = TradingSignal(action, strength, current_price, confidence, ml_confidence, {}, [], symbol=symbol_name) # Passa symbol_name
        position_size = self.risk_manager.calculate_position_size_futures(
            temp_signal, portfolio_value, optimal_leverage
        )
        
        # Stop Loss e Take Profit para futuros
        stop_loss, take_profit = self.calculate_futures_stops(
            df, current_price, action, volatility_analysis, optimal_leverage
        )
        
        # Compilar indicadores
        indicators = {
            'traditional_score': traditional_score,
            'ml_signal': ml_signal,
            'final_score': final_score,
            'trend_score': trend_analysis['score'],
            'momentum_score': momentum_analysis['score'],
            'volume_score': volume_analysis['score'],
            'volatility_score': volatility_analysis['score'],
            'breakout_score': breakout_analysis['score'],
            'scalping_score': scalping_analysis['score'],
            'optimal_leverage': optimal_leverage,
            'risk_level': risk_level.value,
            'regime': market_context.regime.value,
            'urgency_score': min(1.0, abs(final_score) + breakout_analysis['score']),
            'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns and not df['rsi'].isnull().iloc[-1] else 50,
            'macd_hist': df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns and not df['macd_hist'].isnull().iloc[-1] else 0,
            'adx': df['adx'].iloc[-1] if 'adx' in df.columns and not df['adx'].isnull().iloc[-1] else 20,
            'volume_ratio': volume_analysis['volume_strength'],
            'bb_position': df['bb_position'].iloc[-1] if 'bb_position' in df.columns and not df['bb_position'].isnull().iloc[-1] else 50
        }
        
        # Compilar razões
        reasons = self.compile_futures_reasons(
            trend_analysis, momentum_analysis, volume_analysis, 
            volatility_analysis, breakout_analysis, market_context
        )
        
        # Log do sinal
        # Ajustado para logar apenas se o símbolo não for UNKNOWN
        if symbol_name != "UNKNOWN" and confidence > 25:
            logger.info(f"🎯 FUTUROS {symbol_name}: {action} {strength.name} conf={confidence:.1f}% "
                       f"lev={optimal_leverage:.1f}x score={final_score:.3f} regime={market_context.regime.name}")
        elif symbol_name == "UNKNOWN":
             logger.warning(f"🚨 UNKNOWN SÍMBOLO GERADO SINAL: {action} {strength.name} conf={confidence:.1f}% lev={optimal_leverage:.1f}x score={final_score:.3f} regime={market_context.regime.name}")


        return TradingSignal(
            action=action,
            strength=strength,
            price=current_price,
            confidence=confidence,
            ml_probability=ml_confidence * 100,
            indicators=indicators,
            reasons=reasons,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_level=risk_level,
            position_size=position_size,
            market_context=market_context,
            leverage_recommendation=optimal_leverage,
            funding_consideration=0.0, # Manter 0.0 se não obtido da API
            urgency_score=indicators['urgency_score'],
            symbol=symbol_name # Garante que o símbolo está no sinal
        )
    
    def analyze_futures_market_context(self, df: pd.DataFrame) -> FuturesMarketContext:
        """
        Analisa contexto específico para futuros.
        Corrigido para lidar com DataFrames vazios/curtos e NaNs.
        """
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_CONTEXT')
        if df.empty or len(df) < 50: # Mínimo de 50 barras para contexto significativo
            logger.warning(f"Contexto de mercado para {symbol_name}: DataFrame vazio ou curto ({len(df)} barras). Retornando contexto padrão.")
            return self.create_default_futures_context()
            
        latest = df.iloc[-1]
        
        # Market Regime
        regime_str = latest.get('market_regime')
        if not pd.isna(regime_str):
            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.RANGING_LOW_VOL
        else:
            regime = MarketRegime.RANGING_LOW_VOL # Default se NaN
        
        # Volatility percentile
        volatility_percentile = 0.5
        if 'realized_vol_5m' in df.columns and not df['realized_vol_5m'].isnull().all() and len(df['realized_vol_5m'].dropna()) >= 50:
            percentile_series = df['realized_vol_5m'].rolling(50, min_periods=25).rank(pct=True)
            if not percentile_series.empty and not pd.isna(percentile_series.iloc[-1]):
                volatility_percentile = percentile_series.iloc[-1]
        
        # Volume profile
        vol_ratio = latest.get('volume_ratio', 1.0)
        if not pd.isna(vol_ratio):
            if vol_ratio > 2.0:
                volume_profile = "VERY_HIGH"
            elif vol_ratio > 1.5:
                volume_profile = "HIGH"
            elif vol_ratio < 0.7:
                volume_profile = "LOW"
            else:
                volume_profile = "NORMAL"
        else:
            volume_profile = "NORMAL" # Default se NaN
        
        # Leverage environment
        if volatility_percentile > 0.8:
            leverage_env = "HIGH_RISK"
        elif volatility_percentile > 0.6:
            leverage_env = "MEDIUM"
        else:
            leverage_env = "LOW_RISK"
        
        # Liquidity score
        liquidity_score = latest.get('liquidity_score', 0.5)
        if pd.isna(liquidity_score): liquidity_score = 0.5
        
        # Gap risk (aproximação)
        gap_risk = abs(latest.get('overnight_gap', 0)) * 10
        if pd.isna(gap_risk): gap_risk = 0.3
        gap_risk = min(1.0, gap_risk)
        
        # Market sentiment
        # Corrigido: Verificar se os valores existem e são numéricos antes de usar
        rsi_val = latest.get('rsi')
        macd_hist_val = latest.get('macd_hist')
        bb_pos_val = latest.get('bb_position')

        sentiment_indicators = []
        if rsi_val is not None and not pd.isna(rsi_val):
            sentiment_indicators.append((rsi_val / 100 - 0.5) * 2)
        if macd_hist_val is not None and not pd.isna(macd_hist_val):
            sentiment_indicators.append(np.tanh(macd_hist_val * 100))
        if bb_pos_val is not None and not pd.isna(bb_pos_val):
            sentiment_indicators.append((bb_pos_val / 100) * 2 - 1)

        market_sentiment = np.mean(sentiment_indicators) if sentiment_indicators else 0.0

        return FuturesMarketContext(
            regime=regime,
            volatility_percentile=volatility_percentile,
            volume_profile=volume_profile,
            time_of_day="market_hours",
            correlation_strength=0.5,
            market_sentiment=market_sentiment,
            leverage_environment=leverage_env,
            funding_rate_pressure=0.0,
            liquidity_score=liquidity_score,
            gap_risk=gap_risk
        )
    
    def create_default_futures_context(self) -> FuturesMarketContext:
        """Cria contexto padrão para futuros"""
        return FuturesMarketContext(
            regime=MarketRegime.RANGING_LOW_VOL,
            volatility_percentile=0.5,
            volume_profile="NORMAL",
            time_of_day="market_hours",
            correlation_strength=0.5,
            market_sentiment=0.0,
            leverage_environment="MEDIUM",
            funding_rate_pressure=0.0,
            liquidity_score=0.5,
            gap_risk=0.3
        )
    
    def get_futures_regime_multiplier(self, regime: MarketRegime, base_score: float) -> float:
        """Multiplicador específico para futuros por regime"""
        multipliers = {
            MarketRegime.TRENDING_BULL: 1.3 if base_score > 0 else 0.7,
            MarketRegime.TRENDING_BEAR: 1.3 if base_score < 0 else 0.7,
            MarketRegime.BREAKOUT_BULL: 1.5 if base_score > 0 else 0.5,
            MarketRegime.BREAKOUT_BEAR: 1.5 if base_score < 0 else 0.5,
            MarketRegime.RANGING_HIGH_VOL: 0.8,
            MarketRegime.RANGING_LOW_VOL: 1.0,
            MarketRegime.SCALPING_FAVORABLE: 1.2,
            MarketRegime.HIGH_LEVERAGE_RISK: 0.4,
        }
        
        return multipliers.get(regime, 1.0)
    
    def score_to_futures_action(self, score: float) -> Tuple[str, SignalStrength]:
        """Converte score em ação específica para futuros (thresholds menores)"""
        abs_score = abs(score)
        
        if abs_score < 0.08:
            return "HOLD", SignalStrength.WEAK
        elif abs_score < 0.15:
            action = "BUY" if score > 0 else "SELL"
            return action, SignalStrength.WEAK
        elif abs_score < 0.25:
            action = "BUY" if score > 0 else "SELL"
            return action, SignalStrength.MODERATE
        elif abs_score < 0.4:
            action = "BUY" if score > 0 else "SELL"
            return action, SignalStrength.STRONG
        elif abs_score < 0.6:
            action = "BUY" if score > 0 else "SELL"
            return action, SignalStrength.VERY_STRONG
        else:
            action = "BUY" if score > 0 else "SELL"
            return action, SignalStrength.EXTREME
    
    def calculate_futures_confidence(self, 
                                   traditional_score: float,
                                   ml_confidence: float,
                                   market_context: FuturesMarketContext,
                                   volatility_analysis: Dict[str, float],
                                   breakout_analysis: Dict[str, float]) -> float:
        """Calcula confiança específica para futuros"""
        
        base_confidence = (ml_confidence + abs(traditional_score) * 2.0) / 2
        
        regime_boost = 0
        if market_context.regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            regime_boost = 0.25
        elif market_context.regime in [MarketRegime.BREAKOUT_BULL, MarketRegime.BREAKOUT_BEAR]:
            regime_boost = 0.30
        elif market_context.regime == MarketRegime.SCALPING_FAVORABLE:
            regime_boost = 0.20
        
        breakout_boost = 0
        if abs(breakout_analysis.get('score', 0)) > 0.3:
            breakout_boost = 0.15
        elif abs(breakout_analysis.get('score', 0)) > 0.2:
            breakout_boost = 0.10
        
        liquidity_boost = (market_context.liquidity_score - 0.5) * 0.1
        
        vol_penalty = 0
        if 'volatility_percentile' in volatility_analysis and not pd.isna(volatility_analysis['volatility_percentile']):
            vol_penalty = max(0, (volatility_analysis['volatility_percentile'] - 0.7)) * 0.05
        
        indicator_boost = 0
        if abs(traditional_score) > 0.4:
            indicator_boost = 0.20
        elif abs(traditional_score) > 0.25:
            indicator_boost = 0.15
        elif abs(traditional_score) > 0.15:
            indicator_boost = 0.10
        
        final_confidence = (base_confidence + regime_boost + breakout_boost + 
                          liquidity_boost + indicator_boost - vol_penalty)
        
        final_confidence = max(0.20, min(0.95, final_confidence))
        
        return final_confidence * 100
    
    def calculate_futures_stops(self, 
                              df: pd.DataFrame,
                              current_price: float,
                              action: str,
                              volatility_analysis: Dict[str, float],
                              leverage: float) -> Tuple[Optional[float], Optional[float]]:
        """Calcula stop loss e take profit específicos para futuros"""
        
        if action == "HOLD":
            return None, None
        
        try:
            # Corrigido: Usar df['atr'].iloc[-1] com verificação de NaN. Se não disponível, fallback para % do preço.
            atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].isnull().iloc[-1] and df['atr'].iloc[-1] > 0 else current_price * 0.02
            
            vol_multiplier = 1.0
            if 'volatility_percentile' in volatility_analysis and not pd.isna(volatility_analysis['volatility_percentile']):
                # Volatilidade percentil está entre 0 e 1, então (1 + percentile) vai de 1 a 2.
                # Multiplicador entre 0.8 e 2.0 (se percentile entre 0 e 1, 1 + (0-0.2)*0.5 -> 0.9 até 1 + (1-0.2)*0.5 -> 1.4)
                # O objetivo é reduzir SL/TP em alta volatilidade e aumentar em baixa volatilidade
                # Aumentando vol_multiplier para volatilidade mais baixa (percentil baixo)
                # Reduzindo vol_multiplier para volatilidade mais alta (percentil alto)
                vol_multiplier = min(2.0, max(0.8, 1 + (0.5 - volatility_analysis['volatility_percentile']) * 1.5 )) # Ajustado o impacto
            
            # Ajuste pela alavancagem: quanto maior a alavancagem, mais apertado o SL/TP
            leverage_adj = 1.0 / (leverage if leverage > 0 else 1.0) # Inverso da alavancagem
            # Quanto maior a alavancagem, menor o sl_multiplier, mais apertado o SL
            
            sl_multiplier = 1.0 * vol_multiplier * leverage_adj
            tp_multiplier = 2.0 * vol_multiplier * leverage_adj # TP também ajustado pela alavancagem

            # Garantir que o sl_multiplier e tp_multiplier não são muito pequenos ou grandes
            sl_multiplier = max(0.2, min(2.0, sl_multiplier))
            tp_multiplier = max(0.4, min(4.0, tp_multiplier)) # TP deve ser maior que SL

            if action == "BUY":
                stop_loss = current_price - (atr * sl_multiplier)
                take_profit = current_price + (atr * tp_multiplier)
            else: # SELL
                stop_loss = current_price + (atr * sl_multiplier)
                take_profit = current_price - (atr * tp_multiplier)
            
            # Garante que stops são válidos e não viram NaN ou Inf
            stop_loss = float(stop_loss) if np.isfinite(stop_loss) else None
            take_profit = float(take_profit) if np.isfinite(take_profit) else None

            # Garante que SL não é pior que TP (para compra) ou melhor que TP (para venda)
            if action == "BUY" and stop_loss is not None and take_profit is not None:
                if stop_loss >= current_price: stop_loss = current_price * 0.99 # Prevent SL above entry
                if take_profit <= current_price: take_profit = current_price * 1.01 # Prevent TP below entry
            elif action == "SELL" and stop_loss is not None and take_profit is not None:
                if stop_loss <= current_price: stop_loss = current_price * 1.01 # Prevent SL below entry
                if take_profit >= current_price: take_profit = current_price * 0.99 # Prevent TP above entry

            return stop_loss, take_profit
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular stops para futuros para o símbolo: {getattr(df, 'symbol', 'UNKNOWN_STOPS')}: {e}. Usando fallback simples.")
            sl_pct = 0.01 / leverage if leverage > 0 else 0.01
            tp_pct = 0.02
            
            if action == "BUY":
                return current_price * (1 - sl_pct), current_price * (1 + tp_pct)
            else:
                return current_price * (1 + sl_pct), current_price * (1 - tp_pct)
    
    def compile_futures_reasons(self, 
                              trend_analysis: Dict,
                              momentum_analysis: Dict,
                              volume_analysis: Dict,
                              volatility_analysis: Dict,
                              breakout_analysis: Dict,
                              market_context: FuturesMarketContext) -> List[str]:
        """Compila razões específicas para futuros"""
        
        reasons = []
        
        if abs(trend_analysis.get('score', 0)) > 0.2:
            trend_dir = 'bullish' if trend_analysis['score'] > 0 else 'bearish'
            reasons.append(f"Trend {trend_dir} forte")
        
        if abs(momentum_analysis.get('score', 0)) > 0.2:
            mom_dir = 'positivo' if momentum_analysis['score'] > 0 else 'negativo'
            reasons.append(f"Momentum {mom_dir}")
        
        if abs(breakout_analysis.get('score', 0)) > 0.2:
            reasons.append("Breakout detectado")
        
        if volume_analysis.get('volume_strength', 0) > 0.5:
            reasons.append("Volume confirmativo")
        elif volume_analysis.get('pressure_score', 0) != 0 and abs(volume_analysis.get('pressure_score', 0)) > 0.3:
            pressure_type = 'compradora' if volume_analysis['pressure_score'] > 0 else 'vendedora'
            reasons.append(f"Pressão {pressure_type}")
        
        if volatility_analysis.get('squeeze', False):
            reasons.append("BB Squeeze - potencial explosão")
        elif abs(volatility_analysis.get('bb_signal', 0)) > 0.5:
            bb_zone = 'sobrevendido' if volatility_analysis['bb_signal'] > 0 else 'sobrecomprado'
            reasons.append(f"Zona {bb_zone} (BB)")
        
        if market_context.regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            reasons.append(f"Regime: {market_context.regime.value}")
        elif market_context.regime in [MarketRegime.BREAKOUT_BULL, MarketRegime.BREAKOUT_BEAR]:
            reasons.append("Regime de breakout")
        elif market_context.regime == MarketRegime.SCALPING_FAVORABLE:
            reasons.append("Condições para scalping")
        
        if hasattr(self, 'ml_predictor') and self.ml_predictor.is_trained and self.ml_predictor.last_accuracy > 0.6:
            reasons.append(f"ML confirma (acc: {self.ml_predictor.last_accuracy:.1%})")
        
        if not reasons:
            reasons = ["Análise técnica combinada"]
        
        return reasons[:4]

    def process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados de mercado para futuros.
        Esta é a função mais importante que prepara o DataFrame para todas as análises.
        Corrigida para garantir robustez contra NaNs e DataFrames vazios/curtos.
        """
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_PROCESS_DATA')
        
        if df.empty:
            logger.error(f"Erro: DataFrame de entrada para process_market_data ({symbol_name}) está vazio.")
            return pd.DataFrame()

        # Garante que as colunas críticas (open, high, low, close, volume) não têm NaNs
        # e são numéricas.
        required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Converte para numérico e preenche NaNs
        for col in required_ohlcv_cols:
            if col not in df.columns:
                df[col] = np.nan # Cria a coluna se não existir
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
        # Remover linhas onde as colunas essenciais são NaN (ou todas NaN)
        initial_rows = len(df)
        df.dropna(subset=required_ohlcv_cols, inplace=True)
        rows_after_drop = len(df)
        if initial_rows != rows_after_drop:
            logger.warning(f"⚠️ {symbol_name}: Removidas {initial_rows - rows_after_drop} linhas com NaNs em colunas críticas para process_market_data.")

        # Verifique se ainda há dados suficientes após remover NaNs
        # O self.required_bars é o mínimo para ML e indicadores mais longos (ex: EMA_slow=21, BB_period=18, ADX=12, ML=50)
        # O ideal é que o df tenha mais que self.required_bars (ex: 200 barras) para ter dados para treinamento ML.
        if len(df) < self.required_bars:
            logger.error(f"❌ {symbol_name}: Dados insuficientes ({len(df)} barras) após limpeza para cálculo de indicadores (mínimo {self.required_bars}). Retornando DataFrame vazio.")
            return pd.DataFrame() # Retorna um DataFrame vazio se não houver dados suficientes

        # Agora sim, extrair os valores como numpy arrays para as funções TALib
        # E garantir que são float64, pois algumas funções TALib são sensíveis
        open_np = df['open'].values.astype(np.float64)
        high_np = df['high'].values.astype(np.float64)
        low_np = df['low'].values.astype(np.float64)
        close_np = df['close'].values.astype(np.float64)
        volume_np = df['volume'].values.astype(np.float64)

        # === MÉDIAS MÓVEIS OTIMIZADAS PARA FUTUROS ===
        # Assegura que o array tem tamanho suficiente para o timeperiod
        if len(close_np) >= self.ema_fast: df['ema_fast'] = talib.EMA(close_np, timeperiod=self.ema_fast)
        else: df['ema_fast'] = np.nan
        if len(close_np) >= self.ema_slow: df['ema_slow'] = talib.EMA(close_np, timeperiod=self.ema_slow)
        else: df['ema_slow'] = np.nan
        if len(close_np) >= 5: df['ema_very_fast'] = talib.EMA(close_np, timeperiod=5)
        else: df['ema_very_fast'] = np.nan
        if len(close_np) >= 20: df['sma_20'] = talib.SMA(close_np, timeperiod=20)
        else: df['sma_20'] = np.nan
        if len(close_np) >= 15: df['tema'] = talib.TEMA(close_np, timeperiod=15)
        else: df['tema'] = np.nan
        
        # === OSCILADORES SENSÍVEIS ===
        if len(close_np) >= self.rsi_period: df['rsi'] = talib.RSI(close_np, timeperiod=self.rsi_period)
        else: df['rsi'] = np.nan
        if len(close_np) >= 7: df['rsi_fast'] = talib.RSI(close_np, timeperiod=7)
        else: df['rsi_fast'] = np.nan
        if len(close_np) >= 10: 
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_np, low_np, close_np, fastk_period=10, slowk_period=3, slowd_period=3)
        else: df['stoch_k'], df['stoch_d'] = np.nan, np.nan
        if len(close_np) >= 10: df['williams_r'] = talib.WILLR(high_np, low_np, close_np, timeperiod=10)
        else: df['williams_r'] = np.nan
        if len(close_np) >= 15: df['cci'] = talib.CCI(high_np, low_np, close_np, timeperiod=15)
        else: df['cci'] = np.nan
        
        # === MACD OTIMIZADO ===
        if len(close_np) >= max(8, 21): 
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close_np, fastperiod=8, slowperiod=21, signalperiod=7)
            df['macd_slope'] = df['macd'].diff(2)
        else: df['macd'], df['macd_signal'], df['macd_hist'], df['macd_slope'] = np.nan, np.nan, np.nan, np.nan
        
        # === BOLLINGER BANDS ===
        if len(close_np) >= self.bb_period:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_np, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8) * 100
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8) * 100
            if len(df['bb_width'].dropna()) >= 15: # Ajustado min_periods para rolling
                df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(15, min_periods=5).median() * 0.8 
            else: df['bb_squeeze'] = np.nan
        else: 
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = np.nan, np.nan, np.nan
            df['bb_width'], df['bb_position'], df['bb_squeeze'] = np.nan, np.nan, np.nan

        # === TREND INDICATORS ===
        if len(high_np) >= self.adx_period:
            df['adx'] = talib.ADX(high_np, low_np, close_np, timeperiod=self.adx_period)
            df['plus_di'] = talib.PLUS_DI(high_np, low_np, close_np, timeperiod=self.adx_period)
            df['minus_di'] = talib.MINUS_DI(high_np, low_np, close_np, timeperiod=self.adx_period)
        else: df['adx'], df['plus_di'], df['minus_di'] = np.nan, np.nan, np.nan
        if len(high_np) >= 15: df['aroon_up'], df['aroon_down'] = talib.AROON(high_np, low_np, timeperiod=15)
        else: df['aroon_up'], df['aroon_down'] = np.nan, np.nan
        
        # === VOLUME INDICATORS PARA FUTUROS ===
        if len(close_np) >= 1 and len(volume_np) >= 1: 
            df['obv'] = talib.OBV(close_np, volume_np)
            df['ad'] = talib.AD(high_np, low_np, close_np, volume_np)
        else: df['obv'], df['ad'] = np.nan, np.nan
        if len(volume_np) >= 15:
            df['volume_sma'] = talib.SMA(volume_np, timeperiod=15)
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        else: df['volume_sma'], df['volume_ratio'] = np.nan, np.nan
        
        # VWAP simplificado
        if len(volume_np) >= 20 and len(close_np) >=20: 
            df['vwap'] = (df['close'] * df['volume']).rolling(20, min_periods=10).sum() / (df['volume'].rolling(20, min_periods=10).sum() + 1e-8)
            df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8) * 100
        else: df['vwap'], df['vwap_deviation'] = np.nan, np.nan

        # === VOLATILIDADE ===
        if len(high_np) >= 12:
            df['atr'] = talib.ATR(high_np, low_np, close_np, timeperiod=12)
            df['natr'] = talib.NATR(high_np, low_np, close_np, timeperiod=12)
        else: df['atr'], df['natr'] = np.nan, np.nan
        
        # === CANDLESTICK PATTERNS RELEVANTES PARA FUTUROS ===
        # Certifique-se que open_np está definido e tem dados
        if len(open_np) > 0 and len(high_np) > 0 and len(low_np) > 0 and len(close_np) > 0:
            df['doji'] = talib.CDLDOJI(open_np, high_np, low_np, close_np)
            df['hammer'] = talib.CDLHAMMER(open_np, high_np, low_np, close_np)
            df['engulfing'] = talib.CDLENGULFING(open_np, high_np, low_np, close_np)
            df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_np, high_np, low_np, close_np)
            df['bullish_marubozu'] = talib.CDLMARUBOZU(open_np, high_np, low_np, close_np)
            df['bearish_marubozu'] = talib.CDLMARUBOZU(open_np, high_np, low_np, close_np) 
            df['doji_star'] = talib.CDLDOJISTAR(open_np, high_np, low_np, close_np)
            df['harami'] = talib.CDLHARAMI(open_np, high_np, low_np, close_np)
            df['morning_star'] = talib.CDLMORNINGSTAR(open_np, high_np, low_np, close_np)
            df['evening_star'] = talib.CDLEVENINGSTAR(open_np, high_np, low_np, close_np)
        else:
            for pattern_col in ['doji', 'hammer', 'engulfing', 'shooting_star', 'bullish_marubozu', 'bearish_marubozu', 'doji_star', 'harami', 'morning_star', 'evening_star']:
                df[pattern_col] = np.nan

        # === FEATURES ESPECÍFICAS PARA FUTUROS ===
        df = self.feature_engineering.add_futures_specific_features(df)
        df = self.feature_engineering.add_futures_timing_features(df)
        
        # === REGIME DETECTION PARA FUTUROS ===
        df['market_regime'] = self.detect_futures_market_regime(df)
        
        # Preencher NaNs remanescentes (após todos os cálculos)
        # Use um ffill/bfill antes do 0.0 para preencher NaNs que são realmente "faltantes"
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0) # Substitui infinitos

        logger.debug(f"✅ {symbol_name}: Indicadores calculados. DataFrame shape: {df.shape}")
        logger.debug(f"✅ {symbol_name}: NaNs restantes após cálculo: {df.isnull().sum().sum()}")
        
        return df

# Classe de compatibilidade
class AdvancedSignalEngine(FuturesAdvancedStrategy):
    """Classe de compatibilidade que funciona tanto para spot quanto futuros"""
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {
                'ema_fast': 8,
                'ema_slow': 21,
                'rsi_period': 12,
                'bb_period': 18,
                'bb_std': 2,
                'adx_period': 12,
                'max_portfolio_risk': 0.15,
                'max_single_position': 0.08,
                'max_leverage': 3.0,
                'volatility_lookback': 50,
                'ml_retrain_interval': 500,
                'breakout_lookback': 15,
                'scalping_threshold': 0.002
            }
        super().__init__(config)
    
    def analyze(self, all_klines_data: Dict[str, pd.DataFrame], 
                all_current_prices: Dict[str, float]) -> Dict[str, TradingSignal]:
        """
        Análise em lote otimizada para futuros.
        Corrigido para lidar com símbolos 'UNKNOWN' antes do processamento.
        """
        signals = {}
        successful_analyses = 0
        failed_analyses = 0
        
        # Estatísticas
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        high_confidence_signals = 0
        
        logger.info(f"🚀 Iniciando análise FUTUROS para {len(all_klines_data)} símbolos")
        
        for symbol in all_klines_data.keys():
            current_price = None 
            
            # Corrigido: Ignorar símbolos UNKNOWN ou vazios/inválidos no início
            if not symbol or symbol == "UNKNOWN" or not isinstance(symbol, str) or symbol.strip() == "":
                logger.warning(f"⚠️ Símbolo inválido ou UNKNOWN '{symbol}' detectado na entrada. Pulando análise.")
                failed_analyses += 1
                continue

            try:
                df = all_klines_data[symbol]
                
                # Anexar o nome do símbolo ao DataFrame para uso nos logs internos
                df.symbol = symbol 
                
                current_price = all_current_prices.get(symbol)
                
                if df is None or df.empty:
                    logger.warning(f"Dados vazios para {symbol}. Pulando análise.")
                    failed_analyses += 1
                    continue
                
                # Fallback para current_price se não veio do all_current_prices ou é inválido
                if current_price is None or current_price <= 0 or not np.isfinite(current_price):
                    if not df.empty and 'close' in df.columns and not df['close'].isnull().iloc[-1]:
                        current_price = df['close'].iloc[-1]
                    else:
                        logger.warning(f"Preço atual ou dados de fechamento inválidos para {symbol}. Pulando análise.")
                        failed_analyses += 1
                        continue
                
                # Verificar dados mínimos ANTES de processar
                min_bars_needed = max(self.required_bars, 100) # Considera 100 para ML
                # Corrigido: Verificação de NaNs em colunas críticas
                if len(df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])) < min_bars_needed:
                    signals[symbol] = TradingSignal(
                        action="HOLD",
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=15,
                        ml_probability=40.0,
                        indicators={},
                        reasons=[f"Dados insuficientes ({len(df.dropna(subset=['open', 'high', 'low', 'close', 'volume']))} < {min_bars_needed} barras)"],
                        risk_level=RiskLevel.HIGH,
                        symbol=symbol
                    )
                    logger.warning(f"Dados insuficientes para {symbol}. Necessário {min_bars_needed} barras.")
                    failed_analyses += 1
                    continue
                
                # Processar dados (calcular indicadores e features)
                # Passa uma cópia para evitar SettingWithCopyWarning
                df_processed = self.process_market_data(df.copy())
                
                # Se o processamento resultou em DataFrame vazio ou com muitos NaNs, pular
                if df_processed.empty or df_processed['close'].isnull().all() or len(df_processed.dropna()) < min_bars_needed:
                    logger.error(f"Erro: Dados processados para {symbol} estão vazios ou com muitos NaNs após cálculo de indicadores. Pulando análise.")
                    failed_analyses += 1
                    signals[symbol] = TradingSignal(
                        action="HOLD",
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=0,
                        ml_probability=40.0,
                        indicators={},
                        reasons=[f"Processamento de dados falhou para {symbol}"],
                        risk_level=RiskLevel.VERY_HIGH,
                        symbol=symbol
                    )
                    continue

                # Gerar sinal específico para futuros
                signal = self.generate_futures_signal(df_processed, current_price, portfolio_value=100000) # Adicione portfolio_value se ele for usado no generate_futures_signal
                signals[symbol] = signal
                successful_analyses += 1
                
                # Estatísticas
                if signal.action == "BUY":
                    buy_signals += 1
                elif signal.action == "SELL":
                    sell_signals += 1
                else:
                    hold_signals += 1
                
                if signal.confidence >= 30:
                    high_confidence_signals += 1
                
                # Log para sinais interessantes
                # Corrigido: não logar HOLDs se for UNKNOWN
                if signal.symbol != "UNKNOWN" and (signal.confidence > 35 or signal.action != "HOLD"):
                    logger.debug(f"🎯 {symbol}: {signal.action} {signal.strength.name} "
                               f"conf={signal.confidence:.1f}% lev={signal.leverage_recommendation:.1f}x")
                
            except Exception as e:
                logger.error(f"Erro analisando {symbol}: {e}", exc_info=True)
                failed_analyses += 1
                signals[symbol] = TradingSignal(
                    action="HOLD",
                    strength=SignalStrength.WEAK,
                    price=current_price if current_price is not None else 0,
                    confidence=0,
                    ml_probability=40.0,
                    indicators={},
                    reasons=[f"Erro crítico: {str(e)[:50]}"],
                    risk_level=RiskLevel.VERY_HIGH,
                    symbol=symbol # Garante que o símbolo está no sinal de erro
                )
        
        # Log final
        logger.info(f"📊 FUTUROS - Análise concluída: {successful_analyses} sucessos, {failed_analyses} falhas")
        logger.info(f"📈 Sinais: {buy_signals} LONG, {sell_signals} SHORT, {hold_signals} HOLD")
        logger.info(f"🎯 Alta confiança (≥30%): {high_confidence_signals}")
        
        if high_confidence_signals == 0:
            logger.warning("⚠️ Nenhum sinal de alta confiança para futuros - verificar condições de mercado")
        
        return signals
    
    def analyze_signal(self, df: pd.DataFrame, current_price: float = None) -> TradingSignal:
        """Método de compatibilidade para análise única"""
        symbol_name = getattr(df, 'symbol', 'UNKNOWN_SIGNAL_SINGLE') # Pega o nome do símbolo

        if current_price is None:
            if not df.empty and 'close' in df.columns and not df['close'].isnull().iloc[-1]:
                current_price = df['close'].iloc[-1]
            else:
                logger.warning(f"Não foi possível obter current_price para análise única ({symbol_name}), usando 0.")
                current_price = 0
        
        # Corrigido: verificar se o símbolo é UNKNOWN ou se o DataFrame é inválido
        if symbol_name == 'UNKNOWN' or df.empty or len(df) < self.required_bars:
            logger.warning(f"Dados insuficientes ou símbolo UNKNOWN para análise única ({symbol_name}). Retornando HOLD.")
            return TradingSignal(
                action="HOLD",
                strength=SignalStrength.WEAK,
                price=current_price,
                confidence=0,
                ml_probability=0,
                indicators={},
                reasons=["Dados insuficientes/inválidos ou símbolo UNKNOWN para análise única"],
                symbol=symbol_name
            )

        # Processar dados (garante que todos os indicadores são calculados)
        df_processed = self.process_market_data(df.copy())

        if df_processed.empty or df_processed['close'].isnull().all() or len(df_processed.dropna()) < self.required_bars:
            logger.error(f"DataFrame processado para análise única ({symbol_name}) está vazio ou contém apenas NaNs.")
            return TradingSignal(
                action="HOLD",
                strength=SignalStrength.WEAK,
                price=current_price,
                confidence=0,
                ml_probability=0,
                indicators={},
                reasons=["Dados insuficientes/inválidos para análise única"],
                symbol=symbol_name
            )
        
        return self.generate_futures_signal(df_processed, current_price)

# Factory function
def create_futures_strategy(config: Dict = None) -> FuturesAdvancedStrategy:
    """Cria estratégia otimizada para futuros"""
    if config is None:
        config = {
            'ema_fast': 8,
            'ema_slow': 21,
            'rsi_period': 12,
            'bb_period': 18,
            'bb_std': 2.0,
            'adx_period': 12,
            'max_portfolio_risk': 0.15,
            'max_single_position': 0.08,
            'max_leverage': 3.0,
            'volatility_lookback': 50,
            'ml_retrain_interval': 500,
            'breakout_lookback': 15,
            'scalping_threshold': 0.002
        }
    
    return FuturesAdvancedStrategy(config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    strategy = create_futures_strategy()
    logger.info("🚀 Estratégia FUTUROS criada com sucesso!")
    logger.info(f"⚡ Parâmetros: EMA({strategy.ema_fast},{strategy.ema_slow}), "
                f"RSI({strategy.rsi_period}), BB({strategy.bb_period})")
    logger.info(f"🛡️ Risk: {strategy.config['max_portfolio_risk']*100:.1f}% portfolio, "
                f"{strategy.config['max_single_position']*100:.1f}% por posição")
    logger.info(f"📊 Alavancagem máxima: {strategy.config['max_leverage']:.1f}x")