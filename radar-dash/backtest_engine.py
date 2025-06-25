import ccxt
import pandas as pd
import numpy as np
import logging
import time
import talib
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from itertools import product

# Novas dependências para ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os # Para verificar a existência do arquivo do modelo

# --- Configuração de logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Configurações da Exchange e Símbolos ---
# ATENÇÃO: Substitua 'SUA_CHAVE_API' e 'SEU_SECRET_API' pelas suas credenciais reais da Gate.io.
# Para backtest, as chaves não são estritamente necessárias se você tiver os dados históricos localmente,
# mas são usadas para buscar dados via CCXT.
GATEIO_API_KEY = 'SUA_CHAVE_API'
GATEIO_SECRET = 'SEU_SECRET_API'

FUTURES_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
TIMEFRAME = '1h' # Timeframe para análise (velas de 1 hora)

# --- Parâmetros Padrão da Estratégia ---
# Estes são os valores padrão que serão usados se a otimização não for executada ou se
# nenhum parâmetro for explicitamente passado para o BacktestingBot.
PIVOT_LOOKBACK = 8
SMA_PERIOD = 20 # Período da SMA para tendência
RSI_PERIOD = 14 # Período para RSI

# --- Configurações de Backtest e Gestão de Risco ---
INITIAL_CAPITAL = 10000 # Capital inicial simulado em USDT
TRADE_SIZE_PERCENT = 0.05 # 5% do capital em cada trade (para cálculo da quantidade)
TAKER_FEE = 0.0005 # Taxa de taker da Gate.io (0.05%)
SLIPPAGE_PCT = 0.0002 # 0.02% de slippage simulado na entrada/saída do preço
RISK_REWARD_RATIO = 3.0 # Relação Risco/Recompensa para Take Profit (ex: 1:3)

# --- Configurações para Otimização de Parâmetros ---
# Define os ranges de valores que o otimizador irá testar para cada parâmetro.
OPTIMIZATION_PARAMS = {
    'SMA_PERIOD': [10, 20, 30, 40, 50],
    'RSI_PERIOD': [7, 14, 21],
    'PIVOT_LOOKBACK': [5, 8, 10, 15],
    'RISK_REWARD_RATIO': [1.5, 2.0, 2.5, 3.0, 3.5]
}
# Métrica para determinar qual combinação de parâmetros é a "melhor"
# 'return_percentage', 'sharpe_ratio', 'total_trades', 'win_rate'
OPTIMIZATION_METRIC = 'sharpe_ratio'

# --- Inicialização da Exchange (para buscar dados históricos) ---
# Esta instância será usada para buscar dados da Gate.io.
exchange = ccxt.gateio({
    'apiKey': GATEIO_API_KEY,
    'secret': GATEIO_SECRET,
    'enableRateLimit': True, # Garante que o bot respeite os limites de requisição da API
    'options': {
        'defaultType': 'future', # Configura para operar com futuros
    },
})

# --- Funções Auxiliares ---

def get_historical_ohlcv(symbol: str, timeframe: str, since: int, limit: int = 1000) -> pd.DataFrame:
    """
    Busca dados OHLCV históricos de forma paginada para cobrir um longo período.
    Retorna um DataFrame pandas.
    """
    all_ohlcv = []
    current_since = since
    end_of_data = False

    while not end_of_data:
        try:
            # Busca um chunk de dados
            ohlcv_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            
            if not ohlcv_chunk:
                end_of_data = True # Não há mais dados
                break
            
            all_ohlcv.extend(ohlcv_chunk)
            
            # Atualiza 'since' para a próxima busca, pegando a timestamp do último candle + 1ms
            current_since = ohlcv_chunk[-1][0] + 1 
            
            # Se o chunk retornado for menor que o limite, significa que chegamos ao fim dos dados disponíveis
            if len(ohlcv_chunk) < limit:
                end_of_data = True

            # Respeita o rate limit da exchange para evitar bloqueios
            time.sleep(exchange.rateLimit / 1000) 

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit excedido ao buscar dados para {symbol}. Esperando 5 segundos. Erro: {e}")
            time.sleep(5) # Espera um pouco mais em caso de rate limit
        except Exception as e:
            logger.error(f"Erro ao buscar dados para {symbol}: {e}. Interrompendo busca.")
            end_of_data = True # Interrompe em caso de outros erros

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Converte colunas numéricas para float para garantir cálculos corretos
    return df.astype(float).drop_duplicates().sort_index()

def detect_pivot_points(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Detecta pontos de pivô (High/Low) usando janelas deslizantes (rolling windows).
    É mais eficiente para grandes DataFrames.
    """
    # Calcula o máximo/mínimo na janela para centralizar a detecção
    df['pivot_high_cand'] = df['high'].rolling(window=lookback * 2 + 1, center=True, min_periods=lookback+1).max()
    df['pivot_low_cand'] = df['low'].rolling(window=lookback * 2 + 1, center=True, min_periods=lookback+1).min()
    
    # Um ponto é pivô se o 'high' (ou 'low') da barra atual for igual ao máximo (ou mínimo) da janela
    df['is_pivot_high'] = (df['high'] == df['pivot_high_cand'])
    df['is_pivot_low'] = (df['low'] == df['pivot_low_cand'])
    
    # Limpa colunas temporárias
    df = df.drop(columns=['pivot_high_cand', 'pivot_low_cand'])
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcula o Average True Range (ATR) usando a biblioteca talib.
    O ATR é uma medida da volatilidade do mercado.
    """
    # talib.ATR já faz o cálculo completo do True Range e da média
    return talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)

def line_equation(idx1, price1, idx2, price2, current_idx):
    """Calcula o valor da linha em um dado índice (auxiliar para triângulos)."""
    slope = (price2 - price1) / (idx2 - idx1) if idx2 != idx1 else 0
    return price1 + slope * (current_idx - idx1)


# --- Classes de Dados ---

@dataclass
class TradeRecommendation:
    """Representa uma recomendação de entrada de trade."""
    symbol: str
    position_type: str # 'LONG' (compra)
    entry_price: float
    confidence: float # Nível de confiança na recomendação
    pattern: str # Padrão técnico que gerou a recomendação
    timestamp: datetime # Timestamp da barra que gerou o sinal

    def to_dict(self):
        """Converte a recomendação para um dicionário serializável."""
        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "entry_price": self.entry_price,
            "confidence": self.confidence,
            "pattern": self.pattern,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class BacktestTrade:
    """Representa um trade individual simulado durante o backtest."""
    symbol: str
    entry_time: datetime
    entry_price: float
    position_type: str
    quantity: float
    # Campos para gestão de risco
    stop_loss: float
    take_profit: float
    # Campos para resultado do trade
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None # Profit and Loss (Lucro e Perda)
    status: str = "OPEN" # "OPEN" ou "CLOSED"
    # Adicionado para o relatório de performance por padrão
    pattern: Optional[str] = None 

    def to_dict(self):
        """Converte o trade para um dicionário serializável."""
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "position_type": self.position_type,
            "quantity": round(self.quantity, 6),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": round(self.exit_price, 4) if self.exit_price else None,
            "pnl": round(self.pnl, 4) if self.pnl is not None else None,
            "status": self.status,
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "pattern": self.pattern
        }

# --- CLASSE: PatternConfidencePredictor (Modelo ML) ---
class PatternConfidencePredictor:
    def __init__(self, data_ref: Dict[str, pd.DataFrame]):
        """
        Inicializa o preditor de confiança de padrões.
        Args:
            data_ref: Uma referência para `self.all_data` do BacktestingBot,
                      necessário para buscar dados de velas para features.
        """
        self.model = None
        # Nomes das features que o modelo usará para prever a confiança
        self.features_names = [
            'rsi', 'atr', 'volume_ratio', 'dist_from_sma',
            'prev_trend_strength', 'market_volatility',
            'confidence_heuristic',
            # One-hot encoding para tipos de padrões (precisa ser atualizado se novos padrões forem adicionados)
            'pattern_type_pivot_low_confirmed', 
            'pattern_type_rompimento_resistencia_vol',
            'pattern_type_triangle_breakout_asc', 
            'pattern_type_rsi_bullish_divergence' 
        ]
        self.target = 'pattern_success'
        self.model_filename = 'pattern_confidence_model.pkl'
        self.all_data_ref = data_ref # Referência aos dados históricos do bot

        # Tentar carregar modelo existente
        if os.path.exists(self.model_filename):
            try:
                self.model = joblib.load(self.model_filename)
                logger.info("Modelo de confiança de padrão carregado com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo de confiança de padrão: {e}. Será treinado novamente.")
                self.model = None

    def _calculate_features_for_bar(self, df_bar: pd.Series, df_full_symbol: pd.DataFrame) -> Dict:
        """
        Calcula as features a partir de uma única barra de dados (pd.Series) e do DataFrame completo do símbolo.
        Args:
            df_bar: pd.Series contendo os dados de uma única barra (linha do DataFrame).
            df_full_symbol: pd.DataFrame completo do símbolo para cálculos que precisam de histórico (ex: rolling mean).
        Returns:
            Dict: Dicionário das features calculadas.
        """
        current_time = df_bar.name 
        
        rolling_window_vol = 20
        # Garante que temos dados suficientes para cálculos de shift e rolling
        df_slice_for_features = df_full_symbol.loc[:current_time].iloc[-max(rolling_window_vol + 5, 50):]
        
        # Preenche com 0s ou np.nan se não houver dados suficientes
        if len(df_slice_for_features) < max(rolling_window_vol, 5):
            features = {f: np.nan for f in self.features_names if not f.startswith('pattern_type')}
            features.update({
                'pattern_type_pivot_low_confirmed': 0, 
                'pattern_type_rompimento_resistencia_vol': 0,
                'pattern_type_triangle_breakout_asc': 0,
                'pattern_type_rsi_bullish_divergence': 0
            })
            return features

        # Cálculos mais seguros para features
        avg_vol_slice = df_slice_for_features['volume'].rolling(rolling_window_vol, min_periods=1).mean()
        vol_ratio = df_bar['volume'] / avg_vol_slice.iloc[-1] if not avg_vol_slice.empty and avg_vol_slice.iloc[-1] != 0 else np.nan
        
        prev_close_5 = df_slice_for_features['close'].shift(5).iloc[-1] if len(df_slice_for_features) >= 6 else np.nan
        prev_trend_s = (df_bar['close'] - prev_close_5) / prev_close_5 if prev_close_5 != 0 else np.nan

        features = {
            'rsi': df_bar['RSI'],
            'atr': df_bar['ATR'],
            'volume_ratio': vol_ratio,
            'dist_from_sma': (df_bar['close'] - df_bar['SMA']) / df_bar['SMA'] if df_bar['SMA'] != 0 else np.nan,
            'prev_trend_strength': prev_trend_s,
            'market_volatility': df_bar['ATR'] / df_bar['close'] if df_bar['close'] != 0 else np.nan
        }
        
        # Converte NaNs e Infs para 0s para o modelo (pode ajustar isso)
        for k, v in features.items():
            if pd.isna(v) or np.isinf(v):
                features[k] = 0.0
        
        return features

    def create_training_data(self, historical_trades: List[Dict]) -> pd.DataFrame:
        """
        Cria o dataset de treino para o modelo de confiança a partir de trades históricos.
        Args:
            historical_trades: Lista de dicionários de trades (results['trades']).
        Returns:
            pd.DataFrame: DataFrame com features e target para treinamento.
        """
        logger.info("Criando dados de treino para o modelo de confiança de padrões...")
        data = []
        for trade_dict in historical_trades:
            entry_time = pd.to_datetime(trade_dict['entry_time'])
            symbol = trade_dict['symbol']
            
            # Pega o DataFrame completo do símbolo e a barra de entrada
            if symbol not in self.all_data_ref or entry_time not in self.all_data_ref[symbol].index:
                logger.debug(f"Dados históricos para {symbol} em {entry_time} não encontrados para treino do ML. Pulando trade.")
                continue
            
            df_full_symbol = self.all_data_ref[symbol]
            df_bar_at_entry = df_full_symbol.loc[entry_time]
            
            # Calcula as features para esta barra de entrada
            features = self._calculate_features_for_bar(df_bar_at_entry, df_full_symbol)
            
            # Adiciona a confiança heurística que foi usada no momento do sinal
            # Isso pode ser ajustado se você tiver salvo a confiança heurística original.
            features['confidence_heuristic'] = trade_dict.get('confidence', 0.5) # Usar a confiança heurística real do trade
            
            # One-hot encoding para o tipo de padrão (separando múltiplos padrões)
            patterns_in_trade = trade_dict.get('pattern', '').split('+')
            features['pattern_type_pivot_low_confirmed'] = 1 if 'PIVÔ_DE_BAIXA_CONFIRMADO' in patterns_in_trade else 0
            features['pattern_type_rompimento_resistencia_vol'] = 1 if 'ROMPIMENTO_RESISTENCIA_VOL' in patterns_in_trade else 0
            features['pattern_type_triangle_breakout_asc'] = 1 if 'TRIANGULO_BREAKOUT_ASC' in patterns_in_trade else 0
            features['pattern_type_rsi_bullish_divergence'] = 1 if 'RSI_BULLISH_DIVERGENCE' in patterns_in_trade else 0
            
            # Target: trade bem-sucedido? (PnL > 0)
            success = 1 if trade_dict['pnl'] > 0 else 0
            
            data.append({**features, self.target: success})
        
        training_df = pd.DataFrame(data)
        training_df = training_df.dropna() # Remover linhas com NaNs que não puderam ser calculadas
        
        if training_df.empty:
            logger.warning("DataFrame de treinamento está vazio após criação. Treinamento do ML abortado.")
        else:
            logger.info(f"DataFrame de treinamento criado com {len(training_df)} amostras.")
        return training_df

    def train_model(self, training_data: pd.DataFrame):
        """
        Treina o modelo de previsão de confiança (RandomForestClassifier).
        Args:
            training_data: DataFrame de treino com features e target.
        """
        if training_data.empty or len(training_data) < 20: # Precisa de um mínimo de amostras para split e treino
            logger.warning(f"Poucas amostras ({len(training_data)}) para treinar o modelo de confiança. Mínimo recomendado: 20. Treinamento abortado.")
            return

        # Garante que todas as features esperadas estão no DataFrame e na ordem correta
        for f in self.features_names:
            if f not in training_data.columns:
                training_data[f] = 0 # Adiciona features ausentes com valor zero

        X = training_data[self.features_names]
        y = training_data[self.target]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y # stratify ajuda a manter a proporção de classes
            )
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # balanced para lidar com classes desbalanceadas
            self.model.fit(X_train, y_train)
            
            # Avaliar desempenho do modelo
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Modelo de confiança treinado com precisão de {accuracy:.2%}")
            
            # Salvar o modelo treinado para reuso
            joblib.dump(self.model, self.model_filename)
            logger.info(f"Modelo de confiança salvo em {self.model_filename}")
        except ValueError as ve:
            logger.error(f"Erro ao treinar modelo (provavelmente classes desbalanceadas no split ou poucas amostras): {ve}. Tente mais dados.")
        except Exception as e:
            logger.error(f"Erro inesperado durante o treinamento do modelo de confiança: {e}")

    def predict_confidence(self, current_data_series: pd.Series, df_full_symbol: pd.DataFrame, heuristic_confidence: float, pattern_name: str) -> float:
        """
        Prevê a probabilidade de sucesso (confiança) para um padrão no momento atual.
        Args:
            current_data_series: pd.Series da barra atual.
            df_full_symbol: DataFrame completo do símbolo para cálculos de features.
            heuristic_confidence: A confiança heurística original do sinal.
            pattern_name: O nome do padrão detectado.
        Returns:
            float: Probabilidade de sucesso prevista pelo modelo (entre 0 e 1).
        """
        if self.model is None:
            logger.warning("Modelo de confiança não está treinado/carregado. Retornando confiança heurística.")
            return heuristic_confidence # Retorna a confiança heurística como fallback
        
        try:
            # Preparar features do momento atual
            features = self._calculate_features_for_bar(current_data_series, df_full_symbol)
            features['confidence_heuristic'] = heuristic_confidence
            
            # One-hot encoding para o tipo de padrão (para previsão)
            features['pattern_type_pivot_low_confirmed'] = 1 if 'PIVÔ_DE_BAIXA_CONFIRMADO' in pattern_name else 0
            features['pattern_type_rompimento_resistencia_vol'] = 1 if 'ROMPIMENTO_RESISTENCIA_VOL' in pattern_name else 0
            features['pattern_type_triangle_breakout_asc'] = 1 if 'TRIANGULO_BREAKOUT_ASC' in pattern_name else 0
            features['pattern_type_rsi_bullish_divergence'] = 1 if 'RSI_BULLISH_DIVERGENCE' in pattern_name else 0

            # Garante que as features estejam na ordem correta esperada pelo modelo
            X_predict = pd.DataFrame([features])[self.features_names]
            
            # Fazer previsão de probabilidade da classe 1 (sucesso)
            proba = self.model.predict_proba(X_predict)[0][1]
            return proba
        except Exception as e:
            logger.error(f"Erro ao prever confiança com ML: {e}. Retornando confiança heurística. Features: {features}")
            return heuristic_confidence

# --- CLASSE: PatternEnsemble (Sistema de Ensemble de Padrões) ---
class PatternEnsemble:
    def __init__(self, pivot_lookback: int, rsi_period: int, sma_period: int):
        # Mapeia nomes de padrões para seus detectores correspondentes
        self.pattern_detectors = {
            'pivot_low_confirmed': self._detect_pivot_low_confirmed,
            'rompimento_resistencia_vol': self._detect_rompimento_resistencia_vol,
            'triangle_breakout_asc': self._detect_triangle_breakout_asc,
            'rsi_bullish_divergence': self._detect_rsi_bullish_divergence
        }
        # Pesos heurísticos iniciais para cada padrão (podem ser ajustados pelo aprendizado contínuo)
        self.pattern_weights = {
            'pivot_low_confirmed': 0.25,
            'rompimento_resistencia_vol': 0.30,
            'triangle_breakout_asc': 0.25,
            'rsi_bullish_divergence': 0.20
        }
        self.pivot_lookback = pivot_lookback
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        
        # Parâmetros de triângulo (mantidos do código original, podem ser otimizados)
        # Note: Estes deveriam ser parâmetros da estratégia global ou otimizáveis
        self.triangle_deviation_pct = 0.008
        self.triangle_breakout_factor = 1.0015
        #self.triangle_min_points = 4 # Não está sendo usado diretamente na lógica de detecção

    def detect_all_patterns(self, df_slice: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Executa todos os detectores de padrão na fatia de dados atual.
        Args:
            df_slice: Fatia do DataFrame contendo os dados necessários para detecção.
            current_time: O timestamp da barra atual.
        Returns:
            Dict: Um dicionário de padrões detectados, onde a chave é o nome do padrão
                  e o valor é um dicionário com 'confidence' e 'entry_price'.
        """
        results = {}
        
        # Assegura que há dados suficientes para os cálculos
        if len(df_slice) < max(self.pivot_lookback * 2 + 5, self.rsi_period, self.sma_period, 10): # Mínimo para garantir prev bar
            return results

        current = df_slice.iloc[-1]
        
        # Filtros de tendência e momentum aplicados a TODOS os padrões
        if current['close'] < current['SMA']: # Preço abaixo da SMA (tendência de baixa)
            return results
        # RSI < 40 pode ser um sinal de sobrevendido (bom para reversão bullish)
        # RSI > 70 pode ser um sinal de sobrecomprado (ruim para novas entradas long)
        # O filtro `current['RSI'] < 40 or current['RSI'] > 70` é muito restritivo.
        # Uma abordagem melhor seria permitir RSI < 40 para reversões e RSI > 70 para breakouts fortes
        # mas ponderar a confiança. Por simplicidade, mantive o filtro original, mas é um ponto de otimização.
        if current['RSI'] < 30 or current['RSI'] > 70: # Ajustei para 30 para permitir RSI mais baixo para divergências
            return results

        for name, detector in self.pattern_detectors.items():
            try:
                # Passa apenas os dados que o detector precisa
                pattern_result = detector(df_slice, current_time)
                if pattern_result:
                    results[name] = pattern_result
            except Exception as e:
                logger.error(f"Erro no detector de padrão '{name}': {e}")
        
        return results
    
    def _detect_pivot_low_confirmed(self, df_slice: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """Detecta um pivô de baixa (fundo) confirmado por uma vela de alta."""
        current = df_slice.iloc[-1]
        prev = df_slice.iloc[-2] if len(df_slice) >= 2 else None

        if prev is None: return None

        if current['is_pivot_low'] and (current['close'] > prev['close']):
            # Confiança baseada no RSI e na força da recuperação
            confidence = min(0.95, 0.7 + (40 - abs(current['RSI'] - 40))/40 * 0.25)
            return {'confidence': confidence, 'entry_price': current['close']}
        return None

    def _detect_rompimento_resistencia_vol(self, df_slice: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """Detecta um rompimento de resistência com volume crescente."""
        current = df_slice.iloc[-1]
        prev = df_slice.iloc[-2] if len(df_slice) >= 2 else None

        if prev is None: return None

        avg_volume_20_bars = df_slice['volume'].rolling(window=20).mean().iloc[-1] if not df_slice['volume'].rolling(window=20).mean().empty else 0
        
        if current['close'] > prev['high'] and current['volume'] > avg_volume_20_bars * 1.2:
            confidence = min(0.95, 0.75 + ((current['volume'] / avg_volume_20_bars) - 1.2) * 0.1 + (current['RSI'] - 50)/50 * 0.1)
            return {'confidence': confidence, 'entry_price': current['close']}
        return None

    def _detect_triangle_breakout_asc(self, df_slice: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        Detecta formações de triângulo ascendente e verifica rompimento.
        Esta é uma versão simplificada, a detecção robusta de triângulos é complexa.
        """
        recent_bars = min(100, len(df_slice))
        recent_df = df_slice.iloc[-recent_bars:].copy()
        
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        max_highs = []
        min_lows = []
        
        for i in range(1, len(recent_df)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                max_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                min_lows.append((i, lows[i]))
        
        if len(max_highs) < 2 or len(min_lows) < 2:
            return None
        
        try:
            resistances = sorted(max_highs, key=lambda x: x[1], reverse=True)[:2]
            resistances.sort(key=lambda x: x[0])
            
            supports = sorted(min_lows, key=lambda x: x[1])[:2]
            supports.sort(key=lambda x: x[0])
            
            r_slope = (resistances[1][1] - resistances[0][1]) / (resistances[1][0] - resistances[0][0])
            s_slope = (supports[1][1] - supports[0][1]) / (supports[1][0] - supports[0][0])
            
            # Condição para Triângulo Ascendente: Resistência plana/ligeiramente descendente, Suporte ascendente
            if r_slope > -0.0001 and s_slope > 0.0001: # R_slope quase zero ou negativa (plana/descendente), S_slope positiva
                current_idx = len(recent_df) - 1
                r_val = line_equation(resistances[0][0], resistances[0][1], resistances[1][0], resistances[1][1], current_idx)
                
                if recent_df['close'].iloc[-1] > r_val * self.triangle_breakout_factor:
                    # Confiança baseada na convergência e tipo de triângulo
                    confidence = 0.7 + min(0.3, abs(s_slope) * 100) # Exemplo de ajuste de confiança
                    return {'confidence': confidence, 'entry_price': recent_df['close'].iloc[-1]}
        except Exception: # Captura erros na lógica de triângulo
            return None
        return None

    def _detect_rsi_bullish_divergence(self, df_slice: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        Detecta divergência de RSI bullish (preço faz mínimas mais baixas, RSI faz mínimas mais altas).
        Esta é uma detecção simplificada e requer mais refinamento em um ambiente real.
        """
        if len(df_slice) < self.rsi_period + 10: return None # Precisa de histórico para pivôs e RSI

        # Identifica alguns fundos no preço e no RSI
        lows_price = df_slice['low']
        lows_rsi = df_slice['RSI']

        # Tentativa de identificar os dois últimos fundos de preço e RSI
        # Usar nsmallest para encontrar os N menores valores e seus índices
        recent_low_price_series = lows_price.tail(self.rsi_period * 2).nsmallest(2) # Busca nos ultimos 2*RSI períodos
        recent_low_rsi_series = lows_rsi.tail(self.rsi_period * 2).nsmallest(2)
        
        if len(recent_low_price_series) < 2 or len(recent_low_rsi_series) < 2: return None

        # Garante que os fundos mais recentes são realmente mais recentes
        price_low1_time, price_low1_val = recent_low_price_series.index[0], recent_low_price_series.iloc[0]
        price_low2_time, price_low2_val = recent_low_price_series.index[1], recent_low_price_series.iloc[1]
        
        rsi_low1_time, rsi_low1_val = recent_low_rsi_series.index[0], recent_low_rsi_series.iloc[0]
        rsi_low2_time, rsi_low2_val = recent_low_rsi_series.index[1], recent_low_rsi_series.iloc[1]

        # Garantir que low1 é o mais antigo e low2 o mais recente para ambos
        if price_low1_time > price_low2_time: price_low1_time, price_low2_time = price_low2_time, price_low1_time; price_low1_val, price_low2_val = price_low2_val, price_low1_val
        if rsi_low1_time > rsi_low2_time: rsi_low1_time, rsi_low2_time = rsi_low2_time, rsi_low1_time; rsi_low1_val, rsi_low2_val = rsi_low2_val, rsi_low1_val

        # Condição de divergência bullish: Preço faz mínima mais baixa, RSI faz mínima mais alta
        if price_low2_val < price_low1_val and rsi_low2_val > rsi_low1_val:
            # Além disso, o RSI deve estar em território de sobrevenda antes de subir (RSI < 30)
            if rsi_low1_val < 30 and rsi_low2_val < 50: # RSI ainda não sobrecomprado
                confidence = 0.6 + (30 - rsi_low1_val) / 30 * 0.2 # Mais divergência no fundo, mais confiança
                return {'confidence': confidence, 'entry_price': df_slice['close'].iloc[-1]}
        return None

    def combine_signals(self, signals: Dict) -> Optional[Dict]:
        """
        Combina os sinais detectados de múltiplos padrões para uma recomendação final.
        Args:
            signals: Dicionário de padrões detectados e suas confianças/preços.
        Returns:
            Optional[Dict]: Um dicionário com confiança combinada, preço de entrada e padrões,
                            ou None se a confiança combinada não for suficiente.
        """
        if not signals:
            return None
        
        total_weighted_confidence = 0
        total_weight = 0
        sum_weighted_entry_price = 0
        
        for pattern_name, data in signals.items():
            weight = self.pattern_weights.get(pattern_name, 0.1) # Peso padrão se não definido
            
            total_weighted_confidence += data['confidence'] * weight
            sum_weighted_entry_price += data['entry_price'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return None

        # Confiança combinada é a média ponderada das confianças individuais
        combined_confidence = total_weighted_confidence / total_weight
        # Preço de entrada combinado é a média ponderada dos preços de entrada individuais
        combined_entry_price = sum_weighted_entry_price / total_weight
        
        # Filtro final: Só retorna um sinal se a confiança combinada exceder um limite.
        # Este limite pode ser otimizado ou ser parte de uma regra de gestão de risco.
        if combined_confidence > 0.60: # Limite de confiança para considerar o sinal
            return {
                'confidence': combined_confidence,
                'entry_price': combined_entry_price,
                'patterns': "+".join(signals.keys()) # Concatena os nomes dos padrões
            }
        return None

# --- Classe Principal do Backtesting Bot ---
class BacktestingBot:
    def __init__(self, start_date: str, end_date: str, params: Optional[Dict] = None):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbols = FUTURES_SYMBOLS
        
        # Define os parâmetros da estratégia para esta instância do bot
        self.sma_period = params.get('SMA_PERIOD', SMA_PERIOD) if params else SMA_PERIOD
        self.rsi_period = params.get('RSI_PERIOD', RSI_PERIOD) if params else RSI_PERIOD
        self.pivot_lookback = params.get('PIVOT_LOOKBACK', PIVOT_LOOKBACK) if params else PIVOT_LOOKBACK
        self.risk_reward_ratio = params.get('RISK_REWARD_RATIO', RISK_REWARD_RATIO) if params else RISK_REWARD_RATIO

        self.min_bars = max(self.sma_period, self.rsi_period, self.pivot_lookback * 2 + 5, 100) # Aumentado para 100 para triângulos e divergências
        
        self.all_data = {} # Armazena os DataFrames OHLCV pré-processados por símbolo
        self.trades: List[BacktestTrade] = []
        self.current_capital = INITIAL_CAPITAL
        self.peak_capital = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.equity_curve = []
        self.trade_signals: List[Dict] = []

        # Inicializa o ensemble de padrões com os parâmetros atuais
        self.pattern_ensemble = PatternEnsemble(self.pivot_lookback, self.rsi_period, self.sma_period)
        # Inicializa o preditor de confiança do ML, passando a referência dos dados históricos
        self.confidence_predictor = PatternConfidencePredictor(self.all_data) 
        
        self._ml_model_trained_in_this_run = False # Flag para controlar se o modelo ML foi treinado nesta execução

        logger.info(f"📈 Bot de Backtest Inicializado ({start_date} a {end_date}) com Parâmetros: SMA={self.sma_period}, RSI={self.rsi_period}, Pivô={self.pivot_lookback}, R/R={self.risk_reward_ratio}")

    def load_historical_data(self):
        """Carrega dados históricos para todos os símbolos e pré-calcula os indicadores."""
        logger.info("Iniciando carregamento e pré-processamento de dados históricos...")
        since_ms = int(self.start_date.timestamp() * 1000)
        
        for symbol in self.symbols:
            logger.info(f"Buscando dados para {symbol}...")
            df = get_historical_ohlcv(symbol, TIMEFRAME, since_ms)
            
            if df.empty:
                logger.warning(f"Nenhum dado encontrado para {symbol}. Pulando.")
                continue

            df['SMA'] = talib.SMA(df['close'], timeperiod=self.sma_period)
            df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            df['ATR'] = calculate_atr(df, period=14)
            df = detect_pivot_points(df, self.pivot_lookback)
            
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)].dropna() 

            if len(df) < self.min_bars:
                logger.warning(f"Poucos dados para {symbol} no período especificado após cálculo de indicadores ({len(df)} barras). Mínimo necessário: {self.min_bars}. Pulando.")
                continue

            self.all_data[symbol] = df
            logger.info(f"Dados carregados e processados para {symbol}: {len(df)} barras no período de backtest.")
        
        # Após carregar os dados, atualiza a referência no preditor de confiança
        self.confidence_predictor.all_data_ref = self.all_data 

    def detect_entry_signal(self, symbol: str, current_time: datetime) -> Optional[TradeRecommendation]:
        """
        Detecta sinais de entrada combinados de múltiplos padrões, utilizando o modelo ML para refinar a confiança.
        """
        df_full = self.all_data[symbol]
        
        # Pega uma fatia do DataFrame que inclui a barra atual e dados históricos suficientes
        df_slice = df_full.loc[:current_time].iloc[-self.min_bars:].copy()
        
        if len(df_slice) < self.min_bars or df_slice.index[-1] != current_time:
            return None

        current_bar_data = df_slice.iloc[-1]
        
        # Executa todos os detectores de padrão e combina seus sinais
        detected_patterns_results = self.pattern_ensemble.detect_all_patterns(df_slice, current_time)
        combined_signal = self.pattern_ensemble.combine_signals(detected_patterns_results)
        
        if combined_signal:
            heuristic_confidence = combined_signal['confidence']
            patterns_detected_str = combined_signal['patterns']

            # Se o modelo ML foi treinado (ou carregado), use-o para refinar a confiança
            ml_confidence = heuristic_confidence # Fallback inicial
            # Só tenta prever se o modelo foi carregado/treinado OU se o arquivo existe (para carregar dinamicamente)
            if self.confidence_predictor.model is not None or os.path.exists(self.confidence_predictor.model_filename):
                ml_confidence = self.confidence_predictor.predict_confidence(current_bar_data, df_full, heuristic_confidence, patterns_detected_str)
            
            # Combina a confiança heurística com a confiança do ML
            # Estes pesos (0.6 e 0.4) podem ser ajustados ou até mesmo otimizados no futuro
            final_confidence = (0.6 * heuristic_confidence + 0.4 * ml_confidence)
            final_confidence = max(0.0, min(1.0, final_confidence)) # Garante que está entre 0 e 1

            # Filtragem adicional: se o ML der uma confiança muito baixa, podemos reduzir a confiança final
            if ml_confidence < 0.3: # Se o modelo ML prevê uma chance muito baixa de sucesso (ex: < 30%)
                final_confidence *= 0.5 # Reduz a confiança drasticamente
                logger.debug(f"ML Confidence muito baixa ({ml_confidence:.2f}) para {symbol} ({patterns_detected_str}). Confiança final reduzida.")

            # Retorna a recomendação final se a confiança combinada for suficiente
            if final_confidence > 0.65: # Limite final para gerar um sinal de trade
                return TradeRecommendation(
                    symbol=symbol,
                    position_type="LONG",
                    entry_price=combined_signal['entry_price'],
                    confidence=final_confidence,
                    pattern=patterns_detected_str,
                    timestamp=current_time
                )
        return None
    
    def execute_trade(self, symbol: str, rec: TradeRecommendation) -> Optional[BacktestTrade]:
        """Executa um trade simulado com base na recomendação, aplicando slippage e gestão de risco (SL/TP)."""
        df = self.all_data[symbol]
        current_bar_data_for_signal = df.loc[rec.timestamp]
        
        if pd.isna(current_bar_data_for_signal['ATR']) or current_bar_data_for_signal['ATR'] == 0:
            logger.warning(f"ATR inválido para {symbol} em {rec.timestamp}. Não é possível calcular SL/TP para este trade. Sinal ignorado.")
            return None

        entry_price = rec.entry_price * (1 + SLIPPAGE_PCT) 
        atr_distance = current_bar_data_for_signal['ATR'] * 1.5 
        stop_loss = entry_price - atr_distance
        
        if stop_loss <= 0: # Garante que o stop loss não é negativo ou zero
            stop_loss = entry_price * 0.95 # Usa um stop fixo de 5% se o ATR for muito pequeno
            logger.warning(f"ATR muito pequeno/zero para {symbol} em {rec.timestamp}. Usando SL de 5% fixo.")

        take_profit = entry_price + (atr_distance * self.risk_reward_ratio) 
        
        trade_amount_capital_usd = self.current_capital * TRADE_SIZE_PERCENT
        
        if entry_price == 0:
            logger.error(f"Preço de entrada zero para {symbol} em {rec.timestamp}. Não é possível abrir trade.")
            return None

        quantity = trade_amount_capital_usd / entry_price
        
        # Verifica se o capital disponível é suficiente para cobrir o custo nocional da posição
        # (mesmo que futuros usem alavancagem, simulamos que parte do capital é "travada")
        if trade_amount_capital_usd > self.current_capital:
             logger.warning(f"Capital insuficiente para abrir trade em {symbol}. Capital atual: ${self.current_capital:,.2f}, Necessário para trade: ${trade_amount_capital_usd:,.2f}. Sinal ignorado.")
             return None

        cost_of_fee_entry = quantity * entry_price * TAKER_FEE
        self.current_capital -= cost_of_fee_entry
        
        trade = BacktestTrade(
            symbol=symbol,
            entry_time=rec.timestamp,
            entry_price=entry_price,
            position_type="LONG",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern=rec.pattern
        )
        
        logger.info(f"Trade ABERTO | {symbol} | LONG | Entrada: ${entry_price:.4f} | Qtd: {quantity:.6f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f} | Capital: ${self.current_capital:,.2f} | Padrão(ões): {rec.pattern}")
        return trade

    def check_exit_conditions(self, trade: BacktestTrade, current_data: pd.Series) -> Optional[float]:
        """Verifica as condições de saída para um trade aberto."""
        current_low = current_data['low']
        current_high = current_data['high']
        current_close = current_data['close']
        
        # 1. Stop-loss atingido
        if current_low <= trade.stop_loss:
            return trade.stop_loss 
        
        # 2. Take-profit atingido
        elif current_high >= trade.take_profit:
            return trade.take_profit 
            
        # 3. Saída por tempo: Se o trade está aberto por mais de X barras (ex: 5 barras)
        bars_in_trade = (current_data.name - trade.entry_time).total_seconds() / pd.Timedelta(TIMEFRAME).total_seconds()
        if bars_in_trade >= 5:
            return current_close 
                
        return None

    def run_backtest(self) -> Dict:
        """
        Executa a simulação do backtest completo.
        Este método executa em duas "fases" para garantir que o modelo ML tenha trades para aprender:
        1. Uma fase inicial de coleta de trades (sem o ML impactando a confiança).
        2. Treinamento do modelo ML com os trades coletados.
        3. A fase real do backtest, usando o modelo ML treinado para refinar a confiança dos sinais.
        """
        self.load_historical_data() # Carrega e pré-processa todos os dados
        
        if not self.all_data:
            logger.error("Nenhum dado disponível para backtest após carregamento e processamento. Abortando backtest.")
            return {"status": "error", "message": "Nenhum dado disponível para backtest."}
            
        all_timestamps = sorted(list(set(ts for df in self.all_data.values() for ts in df.index)))
        all_timestamps = [ts for ts in all_timestamps if self.start_date <= ts <= self.end_date]
        
        if not all_timestamps:
            logger.warning("Nenhum timestamp válido encontrado no período de backtest especificado. Abortando.")
            return {"status": "error", "message": "Nenhum timestamp válido no período."}

        # --- Fase 1: Coleta de Trades para Treinamento do Modelo ML ---
        # Nesta fase, o bot opera sem o modelo ML para a confiança.
        # Os trades gerados aqui serão a base de dados para o ML aprender.
        trades_for_ml_training: List[BacktestTrade] = []
        open_trades_ml_run: Dict[str, BacktestTrade] = {}
        
        logger.info("Executando fase de coleta de dados para treino do ML (sem ML na confiança)...")

        for i, timestamp in enumerate(all_timestamps):
            # Verifica saídas para trades abertos na fase ML
            trades_to_close_symbols_ml = []
            for symbol, trade in list(open_trades_ml_run.items()):
                if symbol in self.all_data and timestamp in self.all_data[symbol].index:
                    current_bar_data_for_exit = self.all_data[symbol].loc[timestamp]
                    exit_price = self.check_exit_conditions(trade, current_bar_data_for_exit)
                    if exit_price is not None:
                        pnl = trade.quantity * (exit_price - trade.entry_price) - (trade.quantity * exit_price * TAKER_FEE)
                        trade.exit_price = exit_price
                        trade.exit_time = timestamp
                        trade.pnl = pnl
                        trade.status = "CLOSED"
                        trades_for_ml_training.append(trade) # Adiciona à lista de trades para o ML
                        trades_to_close_symbols_ml.append(symbol)
            for symbol_to_remove in trades_to_close_symbols_ml:
                del open_trades_ml_run[symbol_to_remove]

            # Verifica entradas na fase ML (sem usar o ML para confiança)
            for symbol in self.symbols:
                if symbol in open_trades_ml_run: continue # Já tem trade aberto
                if symbol in self.all_data and timestamp in self.all_data[symbol].index:
                    df_slice_for_ml_signal = self.all_data[symbol].loc[:timestamp].iloc[-self.min_bars:].copy()
                    if len(df_slice_for_ml_signal) < self.min_bars: continue

                    # Usa o ensemble para detectar padrões, mas sem o filtro de confiança do ML
                    temp_ensemble = PatternEnsemble(self.pivot_lookback, self.rsi_period, self.sma_period)
                    detected_results_ml = temp_ensemble.detect_all_patterns(df_slice_for_ml_signal, timestamp)
                    basic_combined_signal = temp_ensemble.combine_signals(detected_results_ml)

                    if basic_combined_signal and basic_combined_signal['confidence'] > 0.65: # Confiança heurística mínima para coletar
                        # Cria uma recomendação básica, importante para o ML ter um 'confidence_heuristic'
                        rec_for_ml = TradeRecommendation(
                            symbol=symbol, position_type="LONG", entry_price=basic_combined_signal['entry_price'],
                            confidence=basic_combined_signal['confidence'], pattern=basic_combined_signal['patterns'],
                            timestamp=timestamp
                        )
                        # Executa um trade "leve" para coletar dados, com SL/TP fixo para ter PnL
                        trade_amount_temp = INITIAL_CAPITAL * TRADE_SIZE_PERCENT # Usa o capital inicial para cálculo
                        quantity_temp = trade_amount_temp / rec_for_ml.entry_price
                        
                        trade_for_ml = BacktestTrade(
                            symbol=symbol, entry_time=rec_for_ml.timestamp, entry_price=rec_for_ml.entry_price * (1 + SLIPPAGE_PCT),
                            position_type="LONG", quantity=quantity_temp, stop_loss=rec_for_ml.entry_price * 0.985, # SL/TP básico para coleta
                            take_profit=rec_for_ml.entry_price * 1.015, pattern=rec_for_ml.pattern
                        )
                        open_trades_ml_run[symbol] = trade_for_ml
        
        # Fechar trades remanescentes da fase ML no final do período
        for trade in open_trades_ml_run.values():
            if trade.status == "OPEN":
                if trade.symbol in self.all_data and not self.all_data[trade.symbol].empty:
                    last_data = self.all_data[trade.symbol].loc[trade.entry_time:].iloc[-1] # Pega a última barra após a entrada
                    trade.exit_price = last_data['close']
                    trade.exit_time = last_data.name
                    trade.pnl = trade.quantity * (trade.exit_price - trade.entry_price) - (trade.quantity * trade.exit_price * TAKER_FEE)
                    trade.status = "CLOSED"
                    trades_for_ml_training.append(trade)
                else:
                    logger.warning(f"Trade para {trade.symbol} ficou aberto e não pôde ser fechado na fase ML (dados ausentes).")

        # --- Treinamento do Modelo ML após a Fase de Coleta ---
        if not self._ml_model_trained_in_this_run:
            logger.info(f"Coleta de {len(trades_for_ml_training)} trades para treinamento ML concluída. Treinando modelo de confiança...")
            training_data_df = self.confidence_predictor.create_training_data([t.to_dict() for t in trades_for_ml_training])
            if not training_data_df.empty:
                self.confidence_predictor.train_model(training_data_df)
                self._ml_model_trained_in_this_run = True # Marca que o modelo foi treinado nesta execução
            else:
                logger.warning("Não foi possível treinar o modelo de ML devido à falta de dados de treino válidos. O backtest real operará sem ML na confiança.")

        # --- Fase 2: Rodar o Backtest Real com o Modelo ML Treinado e Ensemble ---
        logger.info("Executando segunda fase do backtest (real) com o modelo ML de confiança e ensemble de padrões.")
        
        # Reinicializa o capital e os trades para o backtest real
        self.current_capital = INITIAL_CAPITAL
        self.peak_capital = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.equity_curve = []
        self.trades = [] # Limpa a lista de trades para a fase real
        open_trades_real_run = {} # Dicionário para trades abertos na fase real

        for i, timestamp in enumerate(all_timestamps):
            # Atualizar curva de equity e drawdown
            self.equity_curve.append({"timestamp": timestamp.isoformat(), "equity": self.current_capital})
            self.peak_capital = max(self.peak_capital, self.current_capital)
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Verificar saídas para trades abertos na fase real
            trades_to_close_symbols_real = []
            for symbol, trade in list(open_trades_real_run.items()):
                if symbol in self.all_data and timestamp in self.all_data[symbol].index:
                    current_bar_data_for_exit = self.all_data[symbol].loc[timestamp]
                    exit_price = self.check_exit_conditions(trade, current_bar_data_for_exit)
                    
                    if exit_price is not None:
                        pnl = trade.quantity * (exit_price - trade.entry_price)
                        pnl -= (trade.quantity * exit_price * TAKER_FEE) 
                        self.current_capital += pnl
                        trade.exit_price = exit_price
                        trade.exit_time = timestamp
                        trade.pnl = pnl
                        trade.status = "CLOSED"
                        self.trades.append(trade) # Adiciona à lista final de trades
                        trades_to_close_symbols_real.append(symbol)
                        exit_reason = "SL" if exit_price == trade.stop_loss else ("TP" if exit_price == trade.take_profit else "Tempo/Fechamento")
                        logger.debug(f"Trade FECHADO | {symbol} | PnL: ${pnl:,.2f} | Capital: ${self.current_capital:,.2f} | Motivo: {exit_reason}")
            
            for symbol_to_remove in trades_to_close_symbols_real:
                del open_trades_real_run[symbol_to_remove]
                
            # Verificar novas entradas (AGORA USANDO O ML E ENSEMBLE)
            for symbol in self.symbols:
                if symbol in open_trades_real_run: continue # Já tem trade aberto
                if symbol in self.all_data and timestamp in self.all_data[symbol].index:
                    signal = self.detect_entry_signal(symbol, timestamp) # Esta chamada usa o ML e Ensemble
                    
                    if signal and signal.confidence > 0.65:
                        self.trade_signals.append(signal.to_dict()) 
                        required_capital_for_trade = self.current_capital * TRADE_SIZE_PERCENT
                        # Proteção extra: não alocar mais de 80% do capital para o trade (ajuste se necessário)
                        if required_capital_for_trade < (self.current_capital * 0.8): 
                            trade = self.execute_trade(symbol, signal)
                            if trade: 
                                open_trades_real_run[symbol] = trade
                        else:
                            logger.debug(f"Capital (${self.current_capital:,.2f}) insuficiente ou trade muito grande para abrir nova posição em {symbol}. Sinal ignorado.")

        # Fechar trades remanescentes no final da fase real (se houver)
        for trade in open_trades_real_run.values():
            if trade.status == "OPEN":
                if trade.symbol in self.all_data and not self.all_data[trade.symbol].empty:
                    last_data_for_symbol = self.all_data[trade.symbol].loc[trade.entry_time:].iloc[-1]
                    trade.exit_price = last_data_for_symbol['close'] 
                    trade.exit_time = last_data_for_symbol.name
                    pnl = trade.quantity * (trade.exit_price - trade.entry_price) - (trade.quantity * trade.exit_price * TAKER_FEE)
                    trade.pnl = pnl
                    trade.status = "CLOSED"
                    self.current_capital += pnl 
                    self.trades.append(trade)
                    logger.debug(f"Trade FECHADO (FIM BT) | {trade.symbol} | PnL: ${pnl:,.2f} | Capital Final: ${self.current_capital:,.2f}")
                else:
                    logger.warning(f"Trade para {trade.symbol} ficou aberto e não pôde ser fechado na fase real (dados ausentes para o final do período).")
        
        # Gera e retorna o relatório final de desempenho para a fase real do backtest
        return self.generate_performance_report()

    def generate_performance_report(self) -> Dict:
        """Calcula e retorna um dicionário com métricas detalhadas de desempenho do backtest."""
        if not self.trades:
            return {
                "status": "error", # Indica que o backtest foi "executado", mas sem trades (pode acontecer na otimização)
                "message": "Nenhum trade executado no período ou com os parâmetros especificados.",
                "initial_capital": INITIAL_CAPITAL, "final_capital": self.current_capital,
                "total_return": 0.0, "return_percentage": 0.0, "total_trades": 0,
                "winning_trades": 0, "losing_trades": 0, "win_rate": 0.0,
                "max_drawdown": self.max_drawdown * 100, "sharpe_ratio": 0.0,
                "best_trade": None, "worst_trade": None, "pattern_performance": {},
                "equity_curve": self.equity_curve, "trades": [],
                "optimized_params": getattr(self, 'optimized_params', {}), # Se otimização, adiciona os params aqui
                "status": "success" # Marca como sucesso se o processo ocorreu, mesmo sem trades
            }
        
        total_return = self.current_capital - INITIAL_CAPITAL
        return_pct = (total_return / INITIAL_CAPITAL) * 100
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = (len(winning_trades) / len(self.trades) * 100) if self.trades else 0.0
        
        sharpe_ratio = 0.0
        if len(self.trades) > 1:
            # Calcula o PnL percentual para cada trade em relação ao capital que foi usado no trade (aproximado)
            trade_returns_pct = [(t.pnl / (t.quantity * t.entry_price)) for t in self.trades if (t.quantity * t.entry_price) > 0]
            if trade_returns_pct:
                avg_trade_return = np.mean(trade_returns_pct)
                std_trade_return = np.std(trade_returns_pct)
                sharpe_ratio = avg_trade_return / std_trade_return if std_trade_return > 0 else float('inf')
            
        best_trade = max(self.trades, key=lambda t: t.pnl, default=None)
        worst_trade = min(self.trades, key=lambda t: t.pnl, default=None)
        
        pattern_performance = {}
        for trade in self.trades:
            # Padrões múltiplos são separados por '+'
            patterns_in_trade = trade.pattern.split('+') if trade.pattern else ["Desconhecido"]
            for pattern_name in patterns_in_trade:
                pattern_performance.setdefault(pattern_name, {"count": 0, "total_pnl": 0.0, "avg_pnl": 0.0})
                pattern_performance[pattern_name]["count"] += 1
                pattern_performance[pattern_name]["total_pnl"] += trade.pnl
        
        for pattern_name, data in pattern_performance.items():
            if data["count"] > 0:
                data["avg_pnl"] = data["total_pnl"] / data["count"]
            else:
                data["avg_pnl"] = 0.0
            
        return {
            "initial_capital": INITIAL_CAPITAL, "final_capital": self.current_capital,
            "total_return": total_return, "return_percentage": return_pct,
            "total_trades": len(self.trades), "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades), "win_rate": win_rate,
            "max_drawdown": self.max_drawdown * 100, "sharpe_ratio": sharpe_ratio,
            "best_trade": best_trade.to_dict() if best_trade else None,
            "worst_trade": worst_trade.to_dict() if worst_trade else None,
            "pattern_performance": pattern_performance,
            "equity_curve": self.equity_curve,
            "trades": [t.to_dict() for t in self.trades],
            "status": "success" # Indica que o backtest foi concluído com sucesso
        }

# --- Função de Otimização ---

def optimize_strategy(start_date: str, end_date: str) -> Dict:
    """Realiza a otimização de parâmetros para a estratégia de backtest."""
    best_results: Optional[Dict] = None
    best_params: Dict = {}
    best_metric_value = -float('inf')

    param_items = OPTIMIZATION_PARAMS.items()
    param_names = [name for name, _ in param_items]
    param_values_lists = [values for _, values in param_items]

    all_combinations = list(product(*param_values_lists))
    total_combinations = len(all_combinations)
    logger.info(f"Iniciando otimização com {total_combinations} combinações...")
    
    for i, combo in enumerate(all_combinations):
        current_params = dict(zip(param_names, combo))
        logger.info(f"Testando combinação {i+1}/{total_combinations}: {current_params}")
        
        bot = BacktestingBot(start_date, end_date, params=current_params)
        # run_backtest agora gerencia o treinamento ML internamente
        results = bot.run_backtest() 
        
        if results and results.get('status') == 'success' and results['total_trades'] > 0:
            current_metric_value = results[OPTIMIZATION_METRIC]
            
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_results = results
                best_params = current_params
                logger.info(f"Nova melhor combinação encontrada: {best_params} com {OPTIMIZATION_METRIC}={best_metric_value:.4f}")
        else:
            logger.warning(f"Backtest falhou ou não gerou trades para os parâmetros: {current_params}. Ignorando esta combinação.")

    if best_results:
        best_results['optimized_params'] = best_params
        logger.info(f"\n--- Otimização Concluída ---")
        logger.info(f"Melhores Parâmetros: {best_params}")
        logger.info(f"Melhor {OPTIMIZATION_METRIC}: {best_metric_value:.4f}")
    else:
        logger.error("Nenhuma combinação de parâmetros resultou em um backtest bem-sucedido ou gerou trades. Verifique a estratégia, dados e ranges de parâmetros.")
        return {"status": "error", "message": "Otimização falhou. Nenhuma combinação válida encontrada."}
        
    return best_results

# --- Execução Principal do Script ---
if __name__ == "__main__":
    start_date_analysis = "2023-01-01" 
    # Use a data de hoje para o end_date, assim o backtest é o mais recente possível.
    # subtrai 1 dia para garantir que os dados de hoje já estejam fechados (se for timeframe de 1 dia)
    # ou para evitar dados incompletos.
    end_date_analysis = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # --- ESCOLHA O MODO DE EXECUÇÃO ---
    # Descomente UMA das opções abaixo:

    # MODO 1: Rodar um Backtest Simples com Parâmetros Padrão e ML de confiança.
    # O modelo ML será treinado na primeira fase deste backtest se não existir um salvo.
    logger.info(f"Iniciando backtest de {start_date_analysis} a {end_date_analysis} com parâmetros padrão e ML de confiança.")
    bot_simple_run = BacktestingBot(start_date_analysis, end_date_analysis)
    results_to_save = bot_simple_run.run_backtest()


    # MODO 2: Rodar a Otimização de Parâmetros.
    # Cada combinação de parâmetros terá seu backtest executado, e o ML será treinado/usado internamente em cada um.
    # Isso pode demorar bastante, dependendo do número de combinações e do período!
    # logger.info(f"Iniciando Otimização de {start_date_analysis} a {end_date_analysis}.")
    # results_to_save = optimize_strategy(start_date_analysis, end_date_analysis)


    # --- Salvar e Exibir Resultados Finais ---
    if results_to_save:
        try:
            # Salva os resultados em um arquivo JSON
            with open("backtest_results.json", "w") as f:
                json.dump(results_to_save, f, indent=4, default=str)
            logger.info("Resultados do backtest/otimização salvos em backtest_results.json")
        except Exception as e:
            logger.error(f"Erro ao salvar resultados em JSON: {e}")
            
        # Imprime um resumo executivo no console
        if results_to_save.get('status') == 'success':
            print("\n=== RESUMO FINAL ===")
            if 'optimized_params' in results_to_save:
                print(f"Parâmetros Otimizados: {results_to_save['optimized_params']}")
            print(f"Capital Inicial: ${results_to_save['initial_capital']:,.2f}")
            print(f"Capital Final:   ${results_to_save['final_capital']:,.2f}")
            print(f"Retorno:         ${results_to_save['total_return']:,.2f} ({results_to_save['return_percentage']:.2f}%)")
            print(f"Trades:          {results_to_save['total_trades']} (Win Rate: {results_to_save['win_rate']:.2f}%)")
            print(f"Drawdown Máx:    {results_to_save['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio:    {results_to_save['sharpe_ratio']:.4f}")
            if results_to_save['best_trade']:
                print(f"Melhor Trade PnL: ${results_to_save['best_trade']['pnl']:,.2f} ({results_to_save['best_trade']['symbol']})")
            if results_to_save['worst_trade']:
                print(f"Pior Trade PnL:   ${results_to_save['worst_trade']['pnl']:,.2f} ({results_to_save['worst_trade']['symbol']})")
            
            print("\n--- Desempenho por Padrão ---")
            for pattern_name, data in results_to_save['pattern_performance'].items():
                print(f"Padrão: {pattern_name} | Trades: {data['count']} | PnL Total: ${data['total_pnl']:,.2f} | PnL Médio: ${data['avg_pnl']:,.2f}")

        else:
            print(f"\nErro no Backtest/Otimização: {results_to_save.get('message', 'Erro desconhecido.')}")
    else:
        logger.error("Nenhum resultado válido gerado para salvar ou exibir.")