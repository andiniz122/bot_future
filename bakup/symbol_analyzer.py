# symbol_analyzer.py
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Importar Config para acessar configurações gerais
import config as Config # Usará o arquivo config.py unificado

logger = logging.getLogger('symbol_analyzer')

@dataclass
class SymbolCharacteristics:
    """Armazena as características analisadas de um único símbolo."""
    symbol: str
    last_analysis_time: float = field(default_factory=time.time)
    
    # Métricas de Volatilidade
    volatility_regime: str = "NORMAL" # LOW, NORMAL, HIGH, EXTREME
    avg_daily_volatility: float = 0.0
    
    # Métricas de Liquidez
    liquidity_score: float = 0.0 # 0 a 1, onde 1 é muito líquido
    avg_daily_volume_usdt: float = 0.0

    # Métricas de Tendência
    trend_strength: float = 0.0 # -1 (forte baixa) a 1 (forte alta)
    trend_regime: str = "NEUTRAL" # UP, DOWN, NEUTRAL, RANGING
    
    # Qualidade dos Dados
    data_quality_score: float = 0.0 # 0 a 1
    
    # Confiança na Análise do Símbolo
    confidence_in_analysis: float = 0.0 # 0 a 1, quão confiável é este perfil
    
    # Efetividade de Indicadores (para backtest interno)
    rsi_effectiveness: float = 0.5 # 0 a 1
    ema_effectiveness: float = 0.5
    bb_effectiveness: float = 0.5

    # Thresholds adaptativos calculados
    confidence_threshold: float = 40.0
    min_price_change: float = 0.5
    max_position_hold_time: int = 24 # horas
    
    # Performance histórica
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Horários ótimos
    best_trading_hours: List[int] = field(default_factory=list)
    worst_trading_hours: List[int] = field(default_factory=list)
    
    # Metadados
    last_analysis: datetime = field(default_factory=datetime.now)
    analysis_count: int = 0
    # confidence_in_analysis: float = 0.0 # REMOVIDO: Já está acima
    symbol_type: str = "crypto" # Adicionado para flexibilidade futura (ex: 'crypto', 'forex')
    
class SymbolAnalyzer:
    """Analisa inteligente que calcula características específicas por símbolo"""
    
    def __init__(self):
        self.symbol_profiles: Dict[str, SymbolCharacteristics] = {}
        self.min_bars_for_analysis = 100
        self.lookback_days = 30 # Usado para determinar janela de dados
        
        logger.info("🔬 SymbolAnalyzer inicializado.")

    def get_symbol_profile(self, symbol: str) -> Optional[SymbolCharacteristics]:
        """Retorna o perfil de características de um símbolo do cache."""
        return self.symbol_profiles.get(symbol)

    def get_adaptive_thresholds_for_symbol(self, symbol: str) -> Dict[str, float]:
        """
        Calcula e retorna os thresholds adaptativos (confiança, mudança de preço)
        para um símbolo específico.
        """
        profile = self.symbol_profiles.get(symbol)
        
        # Valores base do config.py
        base_confidence = Config.MIN_CONFIDENCE_TO_TRADE
        base_price_change = Config.MIN_PRICE_CHANGE_THRESHOLD
        
        # Ajustes iniciais para evitar erros se profile for None
        conf_adj = 0.0
        price_change_adj = 0.0

        if profile:
            # Ajustar confiança com base no nível de risco e qualidade da análise do símbolo
            if profile.volatility_regime == "EXTREME":
                conf_adj += 15.0 # Exige mais confiança em alta volatilidade
                price_change_adj += 0.5 # Exige movimento maior
            elif profile.volatility_regime == "HIGH":
                conf_adj += 10.0
            elif profile.volatility_regime == "LOW":
                conf_adj -= 5.0 # Pode ser menos rigoroso em baixa volatilidade
            
            if profile.data_quality_score < 0.7:
                conf_adj += 10.0 # Exige mais confiança se a qualidade dos dados for baixa
            
            # Aumentar confiança se a própria análise do símbolo tiver alta confiança
            # Garante que profile.confidence_in_analysis está entre 0 e 1 antes de usar
            conf_adj -= (max(0.0, min(1.0, profile.confidence_in_analysis)) - 0.5) * 20 
            
        final_confidence_threshold = max(Config.MIN_CONFIDENCE_TO_TRADE * 0.7, min(95.0, base_confidence + conf_adj))
        final_price_change_threshold = max(Config.MIN_PRICE_CHANGE_THRESHOLD * 0.5, base_price_change + price_change_adj)
        
        return {
            'confidence_threshold': final_confidence_threshold,
            'min_price_change': final_price_change_threshold,
            'max_hold_hours': profile.max_position_hold_time if profile else 24.0 # Retorna max_hold_hours também
        }

    async def analyze_symbol(self, symbol: str, klines_df: pd.DataFrame, current_price: float) -> SymbolCharacteristics:
        """
        Analisa um símbolo específico e retorna suas características.
        Este método fará toda a magia de análise de dados para um único ativo.
        """
        if klines_df.empty or len(klines_df) < self.min_bars_for_analysis:
            logger.warning(f"⚠️ {symbol}: Dados insuficientes ({len(klines_df)} barras) para análise individual.")
            return self._create_default_profile(symbol) # Retorna um perfil padrão se dados insuficientes
        
        profile = SymbolCharacteristics(symbol=symbol)
        profile.last_analysis_time = time.time() # Atualiza o tempo da análise
        profile.last_analysis = datetime.now() # Data/hora da análise
        profile.analysis_count = self.symbol_profiles.get(symbol, SymbolCharacteristics(symbol)).analysis_count + 1 # Incrementa
        self.symbol_profiles[symbol] = profile # Coloca no cache já, para que get_symbol_profile possa acessá-lo mesmo durante a análise

        try:
            # 1. Qualidade dos Dados
            profile.data_quality_score = self._calculate_data_quality_score(klines_df)
            if profile.data_quality_score < 0.5:
                logger.warning(f"⚠️ {symbol}: Baixa qualidade de dados ({profile.data_quality_score:.2f}).")
                profile.risk_level = "HIGH" # Aumenta o risco se a qualidade é baixa
                profile.confidence_in_analysis = 0.3 # Reduz a confiança na análise
                # CONTINUAR ANÁLISE, mas com essas bandeiras
            
            # 2. Volatilidade
            returns = klines_df['close'].pct_change().dropna()
            if len(returns) > 1:
                profile.avg_daily_volatility = returns.std() * np.sqrt(288) # Anualizada (288 = 24h * 60min / 5min)
                
                # Para percentis, precisa de dados suficientes para rolling
                if len(returns) >= 100: # Mínimo para percentis
                    volatility_rolling = returns.rolling(window=24, min_periods=10).std() * np.sqrt(288)
                    volatility_rolling = volatility_rolling.dropna()
                    
                    if len(volatility_rolling) >= 5: # Pelo menos 5 valores para calcular percentis
                        profile.volatility_percentiles = {
                            25: float(np.percentile(volatility_rolling, 25)),
                            50: float(np.percentile(volatility_rolling, 50)),
                            75: float(np.percentile(volatility_rolling, 75)),
                            90: float(np.percentile(volatility_rolling, 90)),
                            95: float(np.percentile(volatility_rolling, 95))
                        }
                    else:
                        profile.volatility_percentiles = {k: profile.avg_daily_volatility for k in [25,50,75,90,95]} # Default se poucos dados
                else:
                     profile.volatility_percentiles = {k: profile.avg_daily_volatility for k in [25,50,75,90,95]} # Default se poucos dados

                # Classificar regime de volatilidade
                current_vol = returns.tail(24).std() * np.sqrt(288) if len(returns.tail(24)) > 1 else profile.avg_daily_volatility
                
                if current_vol < profile.volatility_percentiles.get(25, 0.02):
                    profile.volatility_regime = "LOW"
                elif current_vol < profile.volatility_percentiles.get(75, 0.05):
                    profile.volatility_regime = "NORMAL"
                elif current_vol < profile.volatility_percentiles.get(95, 0.08):
                    profile.volatility_regime = "HIGH"
                else:
                    profile.volatility_regime = "EXTREME"
            else: # Dados insuficientes para retornos
                profile.avg_daily_volatility = 0.0
                profile.volatility_regime = "UNKNOWN" # Ou um regime padrão mais cauteloso
            
            # 3. Liquidez
            if 'quote_asset_volume' in klines_df.columns and not klines_df['quote_asset_volume'].isnull().all():
                avg_volume_usdt = klines_df['quote_asset_volume'].tail(Config.KLINES_UPDATE_INTERVAL_SECONDS * 24 / 60).mean() # Média de 24h
                if pd.isna(avg_volume_usdt): avg_volume_usdt = 0.0

                profile.avg_daily_volume_usdt = avg_volume_usdt

                if avg_volume_usdt >= Config.MIN_VOLUME_24H:
                    profile.liquidity_score = 1.0
                elif avg_volume_usdt >= Config.MIN_VOLUME_24H * 0.5:
                    profile.liquidity_score = 0.7
                elif avg_volume_usdt >= Config.MIN_VOLUME_24H * 0.1:
                    profile.liquidity_score = 0.4
                else:
                    profile.liquidity_score = 0.1
            else:
                profile.liquidity_score = 0.0 # Sem dados de volume ou todos NaN

            # 4. Tendência (simplificado com EMA)
            ema_fast_period = Config.STRATEGY_CONFIG.get('ema_fast', 8)
            ema_slow_period = Config.STRATEGY_CONFIG.get('ema_slow', 21)
            
            # Garante que há dados suficientes para EMA
            if len(klines_df['close'].dropna()) >= max(ema_fast_period, ema_slow_period):
                ema_fast = klines_df['close'].ewm(span=ema_fast_period).mean()
                ema_slow = klines_df['close'].ewm(span=ema_slow_period).mean()
                
                if not ema_fast.empty and not ema_slow.empty and len(ema_fast) >= 2 and len(ema_slow) >= 2:
                    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]: # Cruzamento de alta
                        profile.trend_regime = "UP_CROSS"
                        profile.trend_strength = 0.8
                    elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]: # Cruzamento de baixa
                        profile.trend_regime = "DOWN_CROSS"
                        profile.trend_strength = -0.8
                    elif ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                        profile.trend_regime = "UP"
                        profile.trend_strength = 0.5
                    elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                        profile.trend_regime = "DOWN"
                        profile.trend_strength = -0.5
                    else:
                        profile.trend_regime = "RANGING"
                        profile.trend_strength = 0.0
                else: # Não há dados suficientes para EMA
                    profile.trend_regime = "NEUTRAL"
                    profile.trend_strength = 0.0
            else: # Dados insuficientes
                profile.trend_regime = "NEUTRAL"
                profile.trend_strength = 0.0


            # 5. Efetividade de Indicadores (Aprimorada - ainda heurística, mas mais inteligente)
            # A ideia é que essas avaliações venham de backtesting ou de um sistema de avaliação de sinais.
            # Por enquanto, vamos fazer uma heurística mais avançada.
            
            # Aprimorar o cálculo de efetividade de indicadores
            # Isso exigiria simular trades passados com cada indicador e ver o win-rate
            # Para manter simples, vamos basear na consistência do comportamento e volatilidade
            
            # Efetividade RSI: (Se o RSI geralmente reverte em zonas extremas para este símbolo)
            # (requires more complex logic, for now, use heuristic)
            if profile.volatility_regime == "RANGING": # RSI tende a ser mais eficaz em ranging markets
                profile.rsi_effectiveness = 0.8
            elif profile.volatility_regime in ["TRENDING", "EXTREME"]:
                profile.rsi_effectiveness = 0.4
            else:
                profile.rsi_effectiveness = 0.6

            # Efetividade EMA: (Se o preço geralmente segue EMAs em tendências)
            if profile.trend_regime in ["UP", "DOWN", "UP_CROSS", "DOWN_CROSS"]:
                profile.ema_effectiveness = 0.8
            else:
                profile.ema_effectiveness = 0.5

            # Efetividade BB: (Se o preço respeita as bandas na maioria das vezes)
            if profile.volatility_regime in ["NORMAL", "MEDIUM"]: # BBs eficazes em volatilidade moderada
                profile.bb_effectiveness = 0.7
            else:
                profile.bb_effectiveness = 0.4
            
            # Garante que scores de efetividade estão entre 0 e 1
            profile.rsi_effectiveness = max(0.0, min(1.0, profile.rsi_effectiveness))
            profile.ema_effectiveness = max(0.0, min(1.0, profile.ema_effectiveness))
            profile.bb_effectiveness = max(0.0, min(1.0, profile.bb_effectiveness))

            # 6. Confiança na Análise do Símbolo
            profile.confidence_in_analysis = np.mean([
                profile.data_quality_score,
                profile.liquidity_score,
                profile.rsi_effectiveness,
                profile.ema_effectiveness,
                profile.bb_effectiveness
            ])
            profile.confidence_in_analysis = max(0.0, min(1.0, profile.confidence_in_analysis))

            # 7. Nível de Risco do Símbolo
            profile.risk_level = self._calculate_symbol_risk(profile)

            # 8. Max Hold Hours (ex: symbols em tendência forte podem ter hold mais longo)
            if profile.trend_regime in ["UP", "DOWN", "UP_CROSS", "DOWN_CROSS"] and abs(profile.trend_strength) > 0.5:
                profile.max_position_hold_time = int(min(48, max(24, profile.trend_persistence * 2))) # Tendência >24h
            elif profile.volatility_regime == "EXTREME":
                profile.max_position_hold_time = 6 # Segurar por MUITO menos tempo em volatilidade extrema
            elif profile.volatility_regime == "HIGH":
                profile.max_position_hold_time = 12 # Segurar por menos tempo em alta volatilidade
            else:
                profile.max_position_hold_time = 24 # Padrão
            
            # Garantir min hold time
            profile.max_position_hold_time = max(1, profile.max_position_hold_time)

            self.symbol_profiles[symbol] = profile # Armazena o perfil no cache
            logger.debug(f"🔬 {symbol} perfil: Vol={profile.volatility_regime}, Liq={profile.liquidity_score:.2f}, Trend={profile.trend_regime}, ConfAnálise={profile.confidence_in_analysis:.2f}, Risco={profile.risk_level}, Hold={profile.max_position_hold_time}h")

            return profile

        except Exception as e:
            logger.error(f"❌ Erro na análise individual do símbolo {symbol}: {e}", exc_info=True)
            # Retorna um perfil padrão de alto risco em caso de erro
            return self._create_default_profile(symbol, confidence_in_analysis=0.1, risk_level="EXTREME")


    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calcula um score de qualidade do DataFrame de klines."""
        if df.empty or 'close' not in df.columns or df['close'].isnull().all():
            return 0.0
        
        # Verifica NaNs nas colunas essenciais
        nan_cols = ['open', 'high', 'low', 'close', 'volume'] # Removido 'quote_asset_volume' pois pode não existir no início
        nan_ratio = df[nan_cols].isnull().sum().sum() / (len(df) * len(nan_cols)) if len(nan_cols) > 0 else 0
        
        # Verifica gaps nos timestamps
        gaps_ratio = 0.0
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            time_diffs = df.index.to_series().diff().dropna().dt.total_seconds()
            expected_interval_seconds = Config.KLINES_UPDATE_INTERVAL_SECONDS
            
            if expected_interval_seconds > 0:
                large_gaps = (time_diffs > expected_interval_seconds * 1.5).sum() 
                gaps_ratio = large_gaps / len(time_diffs)
        
        # Idade dos dados
        age_score = 1.0
        if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
            latest_time = df.index.max()
            if latest_time.tz is None:
                latest_time = latest_time.tz_localize(timezone.utc)
            
            age_seconds = (datetime.now(timezone.utc) - latest_time).total_seconds()
            if age_seconds > Config.KLINES_UPDATE_INTERVAL_SECONDS * 5: 
                age_score = max(0.0, 1.0 - (age_seconds / (Config.KLINES_UPDATE_INTERVAL_SECONDS * 10))) 

        score = 1.0 - (nan_ratio * 0.5) - (gaps_ratio * 0.3) + age_score 
        return max(0.0, min(1.0, score))

    def _calculate_symbol_risk(self, profile: SymbolCharacteristics) -> str:
        """Calcula o nível de risco de um símbolo específico."""
        risk_score = 0
        
        if profile.volatility_regime == "EXTREME":
            risk_score += 3
        elif profile.volatility_regime == "HIGH":
            risk_score += 2
        
        if profile.liquidity_score < 0.3:
            risk_score += 3
        elif profile.liquidity_score < 0.6:
            risk_score += 1
            
        if profile.data_quality_score < 0.6:
            risk_score += 2
        
        # Ajustar por tendência (operar contra tendência pode aumentar risco)
        # Mais complexo, para o futuro
        # if (profile.trend_regime in ["DOWN_CROSS", "UP_CROSS"] and abs(profile.trend_strength) > 0.5):
        #     risk_score += 1 

        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def should_trade_symbol_now(self, symbol: str) -> Tuple[bool, str]:
        """
        Verifica se as condições atuais do símbolo permitem operar.
        Considera cooldowsn ou banimentos temporários.
        """
        profile = self.symbol_profiles.get(symbol)
        if not profile:
            return False, "Sem perfil de análise."
        
        if profile.confidence_in_analysis < 0.3:
            return False, f"Baixa confiança na análise ({profile.confidence_in_analysis:.2f})"
            
        # Exemplo: não operar se a liquidez é muito baixa e está em modo ULTRA_SAFE
        if profile.liquidity_score < 0.2 and Config.ULTRA_SAFE_MODE:
            return False, "Liquidez muito baixa no modo ULTRA_SAFE."
            
        # Exemplo: não operar se o risco do símbolo é EXTREMO
        if profile.risk_level == "EXTREME":
            return False, "Risco do símbolo EXTREMO."
        
        # Cooldown após falhas de trade específicas para o símbolo (implementar no main.py)
        # self.symbol_trade_cooldowns = {symbol: timestamp_cooldown}
        
        # Verificar horários ótimos de trade (se definidos)
        current_hour = datetime.now().hour
        if profile.best_trading_hours and current_hour not in profile.best_trading_hours:
            logger.debug(f"⏭️ {symbol}: Fora dos melhores horários de trade ({current_hour}h).")
            return False, f"Fora dos melhores horários de trade ({current_hour}h)"
        
        if profile.worst_trading_hours and current_hour in profile.worst_trading_hours:
            logger.warning(f"⚠️ {symbol}: Horário desfavorável ({current_hour}h).")
            return False, f"Horário desfavorável ({current_hour}h)"
            
        return True, "Condições favoráveis"
        
    def _create_default_profile(self, symbol: str, confidence_in_analysis: float = 0.2, risk_level: str = "MEDIUM") -> SymbolCharacteristics:
        """Cria um perfil padrão/fallback para símbolos com dados insuficientes ou erros."""
        # Puxa alguns padrões do Config
        return SymbolCharacteristics(
            symbol=symbol,
            avg_volatility=Config.STRATEGY_CONFIG['volatility_lookback'] / 1000.0, # Ex: 0.05
            volatility_regime="NORMAL",
            liquidity_score=0.5,
            confidence_threshold=Config.MIN_CONFIDENCE_TO_TRADE,
            min_price_change=Config.MIN_PRICE_CHANGE_THRESHOLD,
            max_position_hold_time=24,
            win_rate=0.5,
            avg_pnl_pct=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            best_trading_hours=list(range(Config.TRADING_HOURS['start_hour'], Config.TRADING_HOURS['end_hour'])), # Todas as horas de trade
            worst_trading_hours=[],
            confidence_in_analysis=confidence_in_analysis,
            risk_level=risk_level
        )

    def _log_symbol_analysis(self, profile: SymbolCharacteristics):
        """Log detalhado da análise do símbolo"""
        logger.info(f"📊 ANÁLISE INDIVIDUAL {profile.symbol}:")
        logger.info(f"   🎯 Threshold confiança: {profile.confidence_threshold:.1f}%")
        logger.info(f"   📈 Volatilidade: {profile.avg_volatility:.3f} ({profile.volatility_regime})")
        logger.info(f"   💧 Liquidez: {profile.liquidity_score:.2f} (Avg Vol USD: {profile.avg_daily_volume_usdt:,.0f})")
        logger.info(f"   🔄 Efetividade indicadores: RSI={profile.rsi_effectiveness:.2f}, EMA={profile.ema_effectiveness:.2f}, BB={profile.bb_effectiveness:.2f}")
        logger.info(f"   ⏰ Melhores horários: {profile.best_trading_hours[:3]}... | Piores: {profile.worst_trading_hours[:3]}...")
        logger.info(f"   🤝 Confiança na análise: {profile.confidence_in_analysis:.2f} | Risco: {profile.risk_level}")
        logger.info(f"   ⏱️ Max Hold: {profile.max_position_hold_time}h | Min Price Change: {profile.min_price_change:.2f}%")