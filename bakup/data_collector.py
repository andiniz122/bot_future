import pandas as pd
import numpy as np
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from gate_api import GateAPI
import config as Config
import requests
import yfinance as yf
from textblob import TextBlob
import talib
import os

logger = logging.getLogger('data_collector_gate')

@dataclass
class BitcoinMarketContext:
    """Contexto específico do Bitcoin e seu impacto no mercado"""
    btc_price: float
    btc_trend: str  # "bullish", "bearish", "sideways"
    btc_sentiment: float  # -1.0 a 1.0
    btc_volatility_impact: float  # 0.0 a 1.0
    correlation_bias: float  # -1.0 a 1.0 para altcoins
    overall_market_fear_greed: Optional[float] = None
    altcoin_correlations: Dict[str, float] = field(default_factory=dict)  # Símbolo -> Correlação
    significant_events: List[str] = field(default_factory=list)  # Eventos macro importantes

class GateFuturesDataCollector:
    """Data Collector otimizado para Futuros Perpétuos Gate.io com contexto Bitcoin"""

    def __init__(self, gate_api: GateAPI):
        self.gate_api = gate_api
        self.klines_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.active_symbols: List[str] = []
        
        # Dados específicos para futuros
        self.funding_rates: Dict[str, float] = {}
        self.open_interest: Dict[str, float] = {}
        self.mark_prices: Dict[str, float] = {}
        self.index_prices: Dict[str, float] = {}
        
        # Controle de atualização
        self.last_klines_update: Dict[str, float] = {}
        self.last_price_update: float = 0
        self.last_funding_update: float = 0
        self.failed_symbols: Dict[str, int] = {}
        self.symbol_stats: Dict[str, Dict] = {}
        
        # Configurações
        self.max_failures_before_removal = 3
        self.cache_duration_seconds = 20
        self.min_update_interval = 5
        self.websocket_enabled = getattr(Config, 'WEBSOCKET_ENABLED', True)
        self.max_concurrent_requests = 3 
        self.request_delay = 0.2 
        self.primary_timeframe = '1m' # Alterado para '1m' como padrão, mas com fallbacks
        self.fallback_timeframes = ['5m', '15m', '1h', '4h'] # Timeframes de fallback
        
        # Contexto Bitcoin
        self.news_api_key = os.getenv('NEWS_API_KEY', getattr(Config, 'NEWS_API_KEY', None)) 
        self.btc_market_context: Optional[BitcoinMarketContext] = None
        self.last_btc_context_update: float = 0
        self.btc_context_update_interval_minutes = getattr(Config, 'BTC_CONTEXT_UPDATE_INTERVAL_MINUTES', 30) 
        self.btc_data_df: Optional[pd.DataFrame] = None
        self.btc_current_price: float = 0.0
        self.news_last_call: float = 0
        self.btc_symbol = "BTC_USDT"

        logger.info(f"🚀 GateFuturesDataCollector inicializado - WebSocket: {self.websocket_enabled}")
        logger.info(f"📰 Coleta de notícias e BTC habilitada. Intervalo: {self.btc_context_update_interval_minutes} min.")

    # ======================================================================
    # MÉTODOS PARA COLETA DE DADOS PRINCIPAL (REINSERIDOS E CORRIGIDOS)
    # ======================================================================

    async def get_daily_volume_history(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Enhanced volume history retrieval with better error handling.
        Encapsula a chamada ao método correspondente na GateAPI para obter histórico de volume diário.
        """
        logger.debug(f"DataCollector: Solicitando histórico de volume diário para {symbol} do GateAPI.")
        try:
            df = await self.gate_api.get_daily_volume_history(symbol, lookback_days)
            if df is None or df.empty:
                logger.warning(f"Volume history empty for {symbol}")
                return None
            
            # Validate data quality
            if 'volume' not in df.columns or df['volume'].isnull().all():
                logger.warning(f"Invalid volume data for {symbol}")
                return None
                
            return df
        except Exception as e:
            logger.error(f"Erro coletando volume para {symbol}: {str(e)}", exc_info=True)
            self._handle_symbol_failure(symbol, f"volume_history_error: {str(e)}")
            return None

    async def fetch_klines_parallel(self, symbols: List[str], initial_interval: str = '1m', initial_limit: int = 200) -> Dict[str, pd.DataFrame]:
        """
        Busca dados de klines para múltiplos símbolos em paralelo usando a GateAPI.
        Processa os dados brutos da API em DataFrames do pandas.
        Implementa re-tentativas com backoff exponencial para erros de Rate Limit e dados inválidos.
        Adiciona lógica para tentar timeframes de fallback se o primário falhar consistentemente.
        """
        logger.info(f"DataCollector: Buscando klines para {len(symbols)} símbolos em paralelo (intervalo inicial: {initial_interval}, limite: {initial_limit})...")
        
        symbols_to_fetch_current_round = list(symbols)
        collected_klines: Dict[str, pd.DataFrame] = {}
        
        all_intervals_to_try = [initial_interval] + self.fallback_timeframes
        
        for current_interval in all_intervals_to_try:
            if not symbols_to_fetch_current_round:
                break # Todos os símbolos já foram coletados com sucesso

            logger.info(f"DataCollector: Tentando coletar klines com intervalo: {current_interval} para {len(symbols_to_fetch_current_round)} símbolos.")
            
            retry_attempts = 0
            max_retries_per_interval = 5 
            
            # Loop de re-tentativa para o *intervalo atual*
            while symbols_to_fetch_current_round and retry_attempts < max_retries_per_interval:
                tasks = []
                # Captura os símbolos que serão tentados nesta rodada específica
                symbols_in_this_attempt = list(symbols_to_fetch_current_round) 
                
                for symbol in symbols_in_this_attempt:
                    tasks.append(self._fetch_single_klines(symbol, current_interval, initial_limit))
                
                if not tasks:
                    break # Não há mais símbolos para tentar neste intervalo

                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Reseta para o próximo loop de tentativa para este mesmo intervalo
                symbols_to_fetch_current_round = [] 
                
                for i, symbol in enumerate(symbols_in_this_attempt):
                    result = results[i]
                    
                    if isinstance(result, list): # Sucesso, a API retornou uma lista de dados brutos
                        df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                        df = df.set_index('timestamp')
                        
                        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Aprimorar verificação de validade do DataFrame
                        if not df.empty and not df['close'].isnull().all() and len(df) >= (initial_limit * 0.5): # Pelo menos 50% dos dados esperados
                            collected_klines[symbol] = df
                            self.last_klines_update[symbol] = time.time()
                            logger.debug(f"DataCollector: Klines para {symbol} coletados e processados ({len(df)} linhas) com intervalo {current_interval}.")
                            self.failed_symbols.pop(symbol, None) # Sucesso: resetar falhas
                        else:
                            # DataFrame vazio, com NaN na coluna 'close' ou com poucos dados
                            reason_msg = "klines_empty_or_invalid_after_processing"
                            if df.empty: reason_msg += "_empty_df"
                            elif df['close'].isnull().all(): reason_msg += "_all_close_nan"
                            elif len(df) < (initial_limit * 0.5): reason_msg += f"_few_data_{len(df)}"

                            logger.warning(f"DataCollector: Klines para {symbol} vazios ou inválidos após processamento com intervalo {current_interval}. Motivo: `{reason_msg}`.")
                            # Adiciona para re-tentativa neste intervalo, se não atingiu o limite de falhas
                            if self.failed_symbols.get(symbol, 0) < self.max_failures_before_removal:
                                symbols_to_fetch_current_round.append(symbol)
                                self._handle_symbol_failure(symbol, reason_msg)
                            else:
                                logger.error(f"Símbolo {symbol} atingiu o limite de falhas para dados vazios/inválidos neste intervalo ({current_interval}), não será re-tentado neste intervalo.")
                                # Marcar este símbolo como "falha persistente para este intervalo" para que não seja mais tentado neste intervalo
                                # Mas ainda pode ser tentado nos próximos timeframes de fallback.
                                
                    elif isinstance(result, dict) and result.get('code') == 'TOO_MANY_REQUESTS':
                        logger.warning(f"DataCollector: Rate Limit Atingido ao buscar {symbol} com intervalo {current_interval}. Adicionando para re-tentativa.")
                        symbols_to_fetch_current_round.append(symbol) 
                        # Não incrementa `failed_symbols` aqui, pois é um problema de rate limit temporário
                    else:
                        error_message = str(result)
                        logger.error(f"DataCollector: Falha ao coletar klines para {symbol} com intervalo {current_interval}: {error_message}")
                        if self.failed_symbols.get(symbol, 0) < self.max_failures_before_removal:
                            symbols_to_fetch_current_round.append(symbol)
                            self._handle_symbol_failure(symbol, error_message)
                        else:
                            logger.error(f"Símbolo {symbol} atingiu o limite de falhas para erro desconhecido neste intervalo, não será re-tentado neste intervalo.")
            
                if symbols_to_fetch_current_round: # Se ainda houver símbolos para buscar nesta tentativa/intervalo
                    retry_attempts += 1
                    if retry_attempts < max_retries_per_interval:
                        wait_time = 2 ** retry_attempts # Atraso exponencial
                        logger.info(f"Aguardando {wait_time}s antes de re-tentar a coleta de klines para {len(symbols_to_fetch_current_round)} símbolos com intervalo {current_interval}.")
                        await asyncio.sleep(wait_time)
                else:
                    break # Todos os símbolos foram coletados com sucesso para este intervalo ou atingiram o limite de retries para este intervalo

            # Após todas as retries para um `current_interval`, reavalia quais símbolos ainda faltam globalmente
            # e os que falharam *persistente*mente neste intervalo para serem tentados no próximo fallback.
            symbols_to_fetch_current_round = [s for s in symbols if s not in collected_klines or collected_klines[s].empty or collected_klines[s]['close'].isnull().all()]
            
            # Se a lista ainda não está vazia, significa que alguns símbolos não foram coletados nem mesmo com retries neste intervalo.
            # Estes serão passados para o próximo intervalo de fallback.

        # No final de todas as tentativas e intervalos, se ainda houver símbolos sem dados, logar e remover
        final_failed_symbols_total = [s for s in symbols if s not in collected_klines or collected_klines[s].empty or collected_klines[s]['close'].isnull().all()]
        if final_failed_symbols_total:
            for s in final_failed_symbols_total:
                # Chama _handle_symbol_failure para forçar a remoção se ele ainda não foi removido
                self._handle_symbol_failure(s, "failed_all_intervals_and_retries_final")
            logger.error(f"Excedido o número máximo de re-tentativas em todos os intervalos para {len(final_failed_symbols_total)} símbolos. Eles serão pulados neste ciclo e potencialmente removidos: {final_failed_symbols_total}")

        self.klines_data.update(collected_klines) 
        return collected_klines

    async def _fetch_single_klines(self, symbol: str, interval: str, limit: int) -> list:
        """
        Método auxiliar para buscar klines de um único símbolo usando a GateAPI.
        """
        return await self.gate_api.get_klines(symbol, interval, limit)

    async def update_current_prices_batch(self, symbols: List[str]):
        """
        Atualiza os preços atuais para uma lista de símbolos em lote usando a GateAPI.
        """
        logger.info(f"DataCollector: Atualizando preços atuais para {len(symbols)} símbolos em lote...")
        prices = await self.gate_api.get_current_prices_bulk(symbols)
        self.current_prices.update(prices)
        self.last_price_update = time.time()
        logger.debug(f"DataCollector: Preços atuais atualizados para {len(prices)} símbolos.")

    # ======================================================================
    # FUNÇÕES DE ACESSO A DADOS
    # ======================================================================

    def get_klines_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Retorna os dados de klines para um símbolo específico."""
        return self.klines_data.get(symbol)

    def get_all_klines_data(self) -> Dict[str, pd.DataFrame]:
        """Retorna todos os dados de klines coletados."""
        return self.klines_data

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Retorna o preço atual de um símbolo."""
        return self.current_prices.get(symbol)

    def get_all_current_prices(self) -> Dict[str, float]:
        """Retorna todos os preços atuais coletados."""
        return self.current_prices

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Retorna a taxa de financiamento para um símbolo."""
        return self.funding_rates.get(symbol)

    def get_open_interest(self, symbol: str) -> Optional[float]:
        """Retorna o interesse em aberto para um símbolo."""
        return self.open_interest.get(symbol)

    # ======================================================================
    # MÉTODOS DE CONTEXTO DE MERCADO E ANÁLISE BTC
    # ======================================================================

    async def update_bitcoin_market_context(self):
        """Atualiza o contexto de mercado do Bitcoin com dados em tempo real"""
        # Verifique se NEWS_API_KEY está no ambiente ou em Config
        if not self.news_api_key:
            logger.error("Chave NEWS_API_KEY não configurada. Contexto Bitcoin desabilitado.")
            return
            
        current_time = time.time()
        # Impedir atualizações muito frequentes
        if current_time - self.last_btc_context_update < self.btc_context_update_interval_minutes * 60:
            return
            
        logger.info("🔄 Atualizando contexto de mercado do Bitcoin...")
        
        try:
            # Coleta de dados do Bitcoin
            btc_data = await self._fetch_btc_market_data()
            if btc_data is None:
                return
                
            # Coleta e análise de notícias
            news_sentiment = await self._fetch_btc_news_sentiment()
            
            # Análise técnica
            trend, volatility = self._analyze_btc_technicals(btc_data)
            
            # Eventos macroeconômicos
            macro_events = await self._fetch_macro_events()
            
            # Atualizar contexto
            self.btc_market_context = BitcoinMarketContext(
                btc_price=self.btc_current_price,
                btc_trend=trend,
                btc_sentiment=news_sentiment,
                btc_volatility_impact=volatility,
                correlation_bias=0.8,  # Valor inicial será atualizado
                significant_events=macro_events
            )
            
            # Calcular correlações em tempo real
            await self.calculate_real_time_correlations()
            
            self.last_btc_context_update = current_time
            logger.info(f"✅ Contexto Bitcoin atualizado: Tendência={trend}, "
                      f"Sentimento={news_sentiment:.2f}, Volatilidade={volatility:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Erro atualizando contexto Bitcoin: {e}", exc_info=True)

    async def _fetch_btc_market_data(self) -> Optional[pd.DataFrame]:
        """Busca dados de mercado do Bitcoin"""
        try:
            # Buscar dados do BTC-USD do Yahoo Finance
            btc = yf.Ticker("BTC-USD")
            # Ajuste o período e intervalo conforme a necessidade de dados para SMA 200 (ex: 5m para 200 barras, precisa de ~17h)
            # Para ter dados suficientes para SMA 200 de 5m candles, 2 dias (2d) pode não ser suficiente se o histórico não for completo.
            # Um período maior pode ser necessário, mas cuidado com limites de yfinance.
            hist = btc.history(period="5d", interval="5m") # Aumentado para 5d para ter mais dados para TA
            
            if hist.empty:
                logger.error("Dados do BTC do Yahoo Finance vazios")
                return None
                
            # Processar dados
            hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            hist.columns = ['open', 'high', 'low', 'close', 'volume']
            self.btc_current_price = hist['close'].iloc[-1]
            self.btc_data_df = hist
            
            return hist
        except Exception as e:
            logger.error(f"Erro buscando dados BTC: {e}", exc_info=True) # Adicionado exc_info
            return None

    async def _fetch_btc_news_sentiment(self) -> float:
        """Busca notícias e calcula sentimento sobre Bitcoin"""
        # Gerenciar rate limiting (1 req/segundo para NewsAPI, se for o plano gratuito)
        current_time = time.time()
        if current_time - self.news_last_call < 1.0:
            await asyncio.sleep(1.0 - (current_time - self.news_last_call))
        self.news_last_call = time.time()
        
        try:
            # Buscar notícias recentes sobre Bitcoin
            # Notícias de alta qualidade e com foco em mercado de cripto seria melhor
            news_url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={self.news_api_key}&pageSize=30"
            response = requests.get(news_url)
            
            if response.status_code != 200:
                logger.error(f"Erro NewsAPI: {response.status_code} - {response.text}")
                return 0.0
                
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            if not articles:
                return 0.0
                
            # Analisar sentimento dos títulos
            polarities = []
            for article in articles:
                title = article.get('title', '')
                if title:
                    analysis = TextBlob(title)
                    polarities.append(analysis.sentiment.polarity)
            
            # Calcular média e ajustar para neutro se próximo de zero
            sentiment = sum(polarities) / len(polarities) if polarities else 0.0
            return sentiment if abs(sentiment) >= 0.05 else 0.0 # Define um limiar para considerar o sentimento significativo
            
        except Exception as e:
            logger.error(f"Erro análise sentimento: {e}", exc_info=True) # Adicionado exc_info
            return 0.0

    def _analyze_btc_technicals(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Realiza análise técnica nos dados do Bitcoin"""
        try:
            closes = data['close'].values
            
            # Calcular indicadores
            # Garantir que há dados suficientes para os períodos dos indicadores
            if len(closes) < 200: # Para SMA 200, precisa de pelo menos 200 pontos
                logger.warning(f"Dados insuficientes para análise técnica completa do BTC (apenas {len(closes)} pontos). Usando fallback para tendência e volatilidade.")
                trend = "sideways"
                volatility = data['close'].pct_change().dropna().std() if len(data['close'].pct_change().dropna()) > 0 else 0.05
                return trend, volatility

            sma_50 = talib.SMA(closes, timeperiod=50)
            sma_200 = talib.SMA(closes, timeperiod=200)
            rsi = talib.RSI(closes, timeperiod=14)
            atr = talib.ATR(data['high'], data['low'], closes, timeperiod=14)
            
            # Certificar que os valores mais recentes dos indicadores não são NaN
            # Talib retorna NaNs para os primeiros N períodos, então sempre pegue o último valor válido.
            if np.isnan(sma_50[-1]) or np.isnan(sma_200[-1]) or np.isnan(rsi[-1]) or np.isnan(atr[-1]):
                logger.warning("Valores de indicadores BTC NaN para o último período. Usando fallback.")
                return "sideways", 0.05

            # Determinar tendência
            if closes[-1] > sma_50[-1] > sma_200[-1]:
                trend = "bullish"
            elif closes[-1] < sma_50[-1] < sma_200[-1]:
                trend = "bearish"
            else:
                trend = "sideways"
                
            # Calcular volatilidade (ATR normalizado)
            volatility = atr[-1] / closes[-1]
            
            # Alertas RSI extremo
            if rsi[-1] > 70:
                logger.warning(f"⚠️ ALERTA BTC: RSI sobrecomprado ({rsi[-1]:.2f})")
            elif rsi[-1] < 30:
                logger.warning(f"⚠️ ALERTA BTC: RSI sobrevendido ({rsi[-1]:.2f})")
                
            return trend, volatility
            
        except Exception as e:
            logger.error(f"Erro análise técnica BTC: {e}", exc_info=True) # Adicionado exc_info
            return "sideways", 0.05

    async def _fetch_macro_events(self) -> List[str]:
        """Busca eventos macroeconômicos relevantes"""
        try:
            # Gerenciar rate limiting
            current_time = time.time()
            if current_time - self.news_last_call < 1.0:
                await asyncio.sleep(1.0 - (current_time - self.news_last_call))
            self.news_last_call = time.time()
            
            # Buscar notícias macro
            news_url = f"https://newsapi.org/v2/everything?q=interest+rates+OR+inflation+OR+Fed+OR+ECB&apiKey={self.news_api_key}&pageSize=10"
            response = requests.get(news_url)
            
            if response.status_code != 200:
                logger.error(f"Erro NewsAPI (macro): {response.status_code} - {response.text}")
                return []
                
            news_data = response.json()
            return [article['title'] for article in news_data.get('articles', [])][:5] # Limita a 5 títulos
            
        except Exception as e:
            logger.error(f"Erro buscando eventos macro: {e}", exc_info=True) # Adicionado exc_info
            return []

    async def calculate_real_time_correlations(self):
        """Calcula correlações em tempo real entre BTC e altcoins"""
        if self.btc_data_df is None or self.btc_data_df.empty or not self.klines_data:
            logger.debug("Correlação: Dados BTC ou klines vazios. Pulando cálculo de correlação.")
            return
            
        try:
            btc_closes_raw = self.btc_data_df['close']
            
            for symbol, klines in self.klines_data.items():
                if symbol == self.btc_symbol:
                    continue
                    
                # Sincronizar timestamps
                if klines.empty or btc_closes_raw.empty: 
                    logger.debug(f"Correlação: Dados vazios para BTC ou {symbol}")
                    continue

                # Garantir que os índices são datetimes UTC
                if btc_closes_raw.index.tz is None:
                    btc_closes_raw.index = btc_closes_raw.index.tz_localize(timezone.utc)
                if klines.index.tz is None:
                    klines.index = klines.index.tz_localize(timezone.utc)

                # Resample para 5 minutos, tomando o último preço disponível em cada janela
                resampled_btc = btc_closes_raw.resample('5min').last().dropna()
                resampled_alt = klines['close'].resample('5min').last().dropna()

                # Merge baseado nos índices de tempo
                merged = pd.merge(
                    resampled_btc, 
                    resampled_alt, 
                    left_index=True, 
                    right_index=True,
                    suffixes=('_btc', '_alt')
                )
                
                # Se após o merge e dropna, tivermos menos de 20 pontos, a correlação pode não ser significativa
                if len(merged) < 20: 
                    logger.debug(f"Correlação para {symbol}: Dados insuficientes após resample e merge ({len(merged)} pontos).")
                    continue
                    
                # Calcular correlação de 30 períodos
                # Usar min(30, len(merged)) para garantir que a janela não exceda os dados disponíveis
                correlation = merged['close_btc'].rolling(min(30, len(merged))).corr(merged['close_alt']).iloc[-1]
                
                # Atualizar contexto
                if self.btc_market_context:
                    self.btc_market_context.altcoin_correlations[symbol] = correlation
                    logger.debug(f"Correlação {symbol} com BTC: {correlation:.4f}")
                    
            if self.btc_market_context: 
                logger.info(f"✅ Correlações atualizadas para {len(self.btc_market_context.altcoin_correlations)} pares.")
            
        except Exception as e:
            logger.error(f"Erro calculando correlações: {e}", exc_info=True)

    def select_pairs_based_on_btc_trend(self) -> List[str]:
        """Seleciona pares para negociação com base na tendência do BTC"""
        if not self.btc_market_context:
            logger.warning("Contexto BTC não disponível. Retornando MANUAL_SYMBOLS.")
            return getattr(Config, 'MANUAL_SYMBOLS', ['BTC_USDT', 'ETH_USDT'])
            
        trend = self.btc_market_context.btc_trend
        correlations = self.btc_market_context.altcoin_correlations
        
        # Filtrar símbolos com dados válidos e que estão nos `klines_data`
        valid_symbols = [
            s for s in self.active_symbols 
            if s in self.klines_data and not self.klines_data[s].empty and s in self.current_prices and self.current_prices[s] > 0
        ]
        
        if not valid_symbols:
            logger.warning("Nenhum símbolo ativo com dados válidos para seleção de pares.")
            return []
            
        # Inicializar symbol_stats se não houver dados para evitar KeyError
        for s in valid_symbols:
            if s not in self.symbol_stats:
                self.symbol_stats[s] = {'avg_quote_volume': 0.0, 'price_volatility': 0.0}

        # Ordenar por correlação (maior correlação absoluta primeiro)
        sorted_symbols_by_corr = sorted(
            valid_symbols,
            key=lambda s: abs(correlations.get(s, 0)),
            reverse=True
        )
        
        selected_pairs = []
        
        # Seleção baseada em tendência
        if trend == "bullish":
            logger.info("Tendência BTC: BULLISH. Buscando alts correlacionadas positivamente e com bom volume.")
            selected_pairs = [
                s for s in sorted_symbols_by_corr
                if correlations.get(s, 0) > 0.6 and 
                self.symbol_stats.get(s, {}).get('avg_quote_volume', 0) > 500000 and 
                self.symbol_stats.get(s, {}).get('price_volatility', 0) < 0.1 
            ]
            selected_pairs = selected_pairs[:getattr(Config, 'MAX_ACTIVE_SYMBOLS', 10)]
            
        elif trend == "bearish":
            logger.info("Tendência BTC: BEARISH. Buscando alts com baixa/negativa correlação e baixa volatilidade.")
            selected_pairs = [
                s for s in sorted_symbols_by_corr
                if correlations.get(s, 1) < 0.2 and 
                self.symbol_stats.get(s, {}).get('price_volatility', 0) < 0.05 and 
                self.symbol_stats.get(s, {}).get('avg_quote_volume', 0) > 100000 
            ]
            selected_pairs = selected_pairs[:getattr(Config, 'MAX_ACTIVE_SYMBOLS', 5)]
            
        else:  # sideways
            logger.info("Tendência BTC: SIDEWAYS. Buscando alts com correlação moderada/baixa e volatilidade gerenciável.")
            selected_pairs = [
                s for s in sorted_symbols_by_corr
                if abs(correlations.get(s, 0)) < 0.5 and 
                self.symbol_stats.get(s, {}).get('price_volatility', 0) < 0.08 and 
                self.symbol_stats.get(s, {}).get('avg_quote_volume', 0) > 200000 
            ]
            selected_pairs = selected_pairs[:getattr(Config, 'MAX_ACTIVE_SYMBOLS', 7)]

        if not selected_pairs:
            logger.warning(f"Nenhum par selecionado com base na tendência '{trend}' do BTC. Recorrendo a símbolos de alto volume.")
            high_volume_symbols = sorted(
                valid_symbols,
                key=lambda s: self.symbol_stats.get(s, {}).get('avg_quote_volume', 0),
                reverse=True
            )
            selected_pairs = high_volume_symbols[:getattr(Config, 'MAX_ACTIVE_SYMBOLS', 5)]


        logger.info(f"Pares selecionados para negociação baseados na tendência BTC ({trend}): {selected_pairs}")
        return selected_pairs

    def check_btc_breakout_alert(self, volatility_threshold=0.05):
        """Verifica condições de breakout no BTC"""
        if not self.btc_market_context:
            return
            
        volatility = self.btc_market_context.btc_volatility_impact
        if volatility > volatility_threshold:
            logger.warning(f"🚨 ALERTA: Alta volatilidade BTC ({volatility:.2%})")
            # Adicionar lógica de notificação aqui (email, Telegram, etc.)

    # ======================================================================
    # FUNÇÕES PARA DETECTAR DUAS VELAS VERDES CONSECUTIVAS (ADICIONADAS)
    # ======================================================================
    
    def has_two_green_candles(self, symbol: str) -> bool:
        """
        Verifica se o par tem duas velas verdes consecutivas
        no timeframe primário (fechamento > abertura)
        """
        if symbol not in self.klines_data or self.klines_data[symbol].empty:
            return False
            
        df = self.klines_data[symbol]
        if len(df) < 2:
            return False
            
        # Obter as duas últimas velas
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Verificar se ambas são velas verdes
        return (
            last_candle['close'] > last_candle['open'] and 
            prev_candle['close'] > prev_candle['open']
        )

    def get_symbols_with_two_green_candles(self) -> List[str]:
        """Retorna lista de símbolos com duas velas verdes consecutivas"""
        green_symbols = []
        for symbol in self.active_symbols: 
            if self.has_two_green_candles(symbol):
                green_symbols.append(symbol)
        return green_symbols

    def check_green_candles_alert(self):
        """Dispara alerta quando detecta padrão em múltiplos pares"""
        green_symbols = self.get_symbols_with_two_green_candles()
        if green_symbols:
            logger.info(f"🚀 Padrão 2 velas verdes detectado em: {', '.join(green_symbols)}")
            
    # ======================================================================
    # FIM DAS NOVAS FUNÇÕES
    # ======================================================================

    # Métodos de tratamento de falhas e estatísticas (se já existirem ou forem adicionados)
    def _handle_symbol_failure(self, symbol: str, reason: str):
        """Incrementa contador de falhas para um símbolo e remove se exceder o limite."""
        # Não conta falhas por Rate Limit para remoção permanente
        if "TOO_MANY_REQUESTS" in reason:
            # Não incrementa self.failed_symbols[symbol] para Rate Limit
            logger.debug(f"Não contando falha de rate limit para {symbol}: {reason}")
            return

        self.failed_symbols[symbol] = self.failed_symbols.get(symbol, 0) + 1
        logger.warning(f"Falha na coleta de dados para {symbol}: {reason}. Falhas consecutivas: {self.failed_symbols[symbol]}")
        
        # Remover símbolo da lista ativa se atingir o limite de falhas
        if self.failed_symbols[symbol] >= self.max_failures_before_removal:
            logger.error(f"Símbolo {symbol} removido da lista ativa devido a múltiplas falhas. Razão final: {reason}")
            if symbol in self.active_symbols:
                self.active_symbols.remove(symbol)
            if symbol in self.klines_data:
                del self.klines_data[symbol]
            if symbol in self.current_prices:
                del self.current_prices[symbol]
            if symbol in self.symbol_stats:
                del self.symbol_stats[symbol]
            # Limpar falhas para o símbolo
            del self.failed_symbols[symbol]
        else:
            logger.debug(f"Símbolo {symbol} ainda elegível para re-tentativa (falhas: {self.failed_symbols[symbol]}/{self.max_failures_before_removal}).")

    def _validate_volume_data(self, symbol: str, volume_data: pd.DataFrame) -> bool:
        """Validate volume data meets minimum requirements"""
        if volume_data is None:
            return False
        if volume_data.empty:
            return False
        if 'volume' not in volume_data.columns:
            return False
        if volume_data['volume'].isnull().all():
            return False
        if volume_data['volume'].sum() <= 0:
            return False
        return True

    def _update_symbol_stats(self, symbol: str, df: pd.DataFrame, current_price: float):
        """Calcula e armazena estatísticas básicas por símbolo."""
        if symbol not in self.symbol_stats: 
            self.symbol_stats[symbol] = {}

        self.symbol_stats[symbol]['current_price'] = current_price

        if not self._validate_volume_data(symbol, df):
            logger.warning(f"Dados de volume inválidos para {symbol}. Definindo defaults para volume e volatilidade.")
            # Set safe defaults
            self.symbol_stats[symbol]['avg_quote_volume'] = 0.0
            self.symbol_stats[symbol]['price_volatility'] = 0.05  # Default volatility
        else:
            closes = df['close'].astype(float)
            volumes = df['volume'].astype(float)
            amounts = df['amount'].astype(float) 

            if len(closes) > 1:
                price_volatility = closes.pct_change().std()
                self.symbol_stats[symbol]['price_volatility'] = price_volatility
            else:
                self.symbol_stats[symbol]['price_volatility'] = 0.0

            if not amounts.empty:
                avg_quote_volume = amounts.mean()
                self.symbol_stats[symbol]['avg_quote_volume'] = avg_quote_volume
            else:
                self.symbol_stats[symbol]['avg_quote_volume'] = 0.0

        current_funding = self.funding_rates.get(symbol, 0.0)
        self.symbol_stats[symbol]['current_funding_rate'] = current_funding

        self.symbol_stats[symbol]['last_stats_update'] = time.time()


    def _calculate_data_quality_score(self, symbol: str) -> float:
        """Calcula uma pontuação de qualidade para os dados de um símbolo."""
        df = self.klines_data.get(symbol)
        if df is None or df.empty:
            return 0.0
        
        completeness_score = len(df) / 200.0 
        freshness_score = 1.0 - min(1.0, (time.time() - self.last_klines_update.get(symbol, 0)) / (self.cache_duration_seconds * 2))
        
        valid_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        column_completeness = sum(1 for col in valid_cols if col in df.columns and not df[col].isnull().all()) / len(valid_cols)
        
        score = (completeness_score * 0.4) + (freshness_score * 0.4) + (column_completeness * 0.2)
        return max(0.0, min(1.0, score)) 

    def get_gate_data_quality_report(self) -> Dict[str, Dict]:
        """Gera um relatório de qualidade dos dados para todos os símbolos ativos."""
        report = {}
        for symbol in self.active_symbols:
            quality_score = self._calculate_data_quality_score(symbol)
            report[symbol] = {
                'quality_score': quality_score,
                'klines_count': len(self.klines_data.get(symbol, [])) if symbol in self.klines_data else 0,
                'last_klines_update_ago_sec': time.time() - self.last_klines_update.get(symbol, 0),
                'current_price': self.current_prices.get(symbol),
                'failed_attempts': self.failed_symbols.get(symbol, 0),
                'is_in_klines_data': symbol in self.klines_data,
                'is_in_current_prices': symbol in self.current_prices,
            }
        return report

class DataCollector(GateFuturesDataCollector):
    """Classe de compatibilidade que mantém interface original"""
    pass

__all__ = ['DataCollector', 'GateFuturesDataCollector', 'BitcoinMarketContext']