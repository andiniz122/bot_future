# market_selector.py
import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Importações de módulos existentes
from data_collector import GateFuturesDataCollector
from estrategia import AdvancedSignalEngine, TradingSignal, MarketRegime, RiskLevel, SignalStrength
import config_gate as Config
import config_ultra_safe as UltraSafeConfig

logger = logging.getLogger('market_selector')

class MarketSelector:
    def __init__(self, data_collector: GateFuturesDataCollector, signal_engine: AdvancedSignalEngine):
        self.data_collector = data_collector
        self.signal_engine = signal_engine
        
        # Parâmetros de seleção (podem vir de config_ultra_safe.py ou ser ajustados)
        self.max_opportunities_to_select = UltraSafeConfig.MAX_OPEN_POSITIONS # Pegar do config_ultra_safe.py
        self.min_overall_score = 0.5 # Score mínimo para ser considerado uma oportunidade
        self.min_confidence_for_selection = UltraSafeConfig.MIN_CONFIDENCE_REQUIRED # Da config ultra segura
        self.min_expected_return = 0.005 # 0.5% de retorno esperado para ser "alto ganho"
        self.max_acceptable_risk_level = RiskLevel.MODERATE # Não aceitar oportunidades de risco HIGH ou VERY_HIGH

    async def select_top_opportunities(self) -> List[Tuple[str, TradingSignal]]:
        """
        Realiza uma varredura profunda do mercado, avalia oportunidades e seleciona
        as N melhores com base em critérios de alto ganho e baixo risco.
        """
        logger.info(f"🔍 Iniciando seleção das TOP {self.max_opportunities_to_select} oportunidades de FUTUROS na Gate.io...")
        
        all_klines_data = self.data_collector.get_all_klines_data()
        all_current_prices = self.data_collector.get_all_current_prices()

        if not all_klines_data or not all_current_prices:
            logger.warning("⚠️ Dados de mercado insuficientes para seleção de oportunidades.")
            return []

        # 1. Gerar sinais para todos os símbolos ativos
        logger.info(f"🧠 Gerando sinais para {len(all_klines_data)} símbolos...")
        opportunities_signals: Dict[str, TradingSignal] = self.signal_engine.analyze(
            all_klines_data, all_current_prices
        )

        potential_opportunities: List[Tuple[str, TradingSignal]] = []
        for symbol, signal in opportunities_signals.items():
            if signal.action != "HOLD":
                potential_opportunities.append((symbol, signal))
        
        if not potential_opportunities:
            logger.info("📊 Nenhuma oportunidade de BUY/SELL gerada pelo Signal Engine.")
            return []

        logger.info(f"🎯 {len(potential_opportunities)} oportunidades potenciais identificadas para análise aprofundada.")

        # 2. Avaliar cada oportunidade com um score abrangente
        scored_opportunities = []
        for symbol, signal in potential_opportunities:
            score, reasons_for_score = self._evaluate_opportunity_score(symbol, signal)
            
            # Log detalhado do score
            logger.debug(f"SCORE {symbol}: {signal.action} Confiança: {signal.confidence:.1f}%, Score Total: {score:.2f}, Razões: {', '.join(reasons_for_score)}")
            
            scored_opportunities.append((symbol, signal, score, reasons_for_score))

        # 3. Filtrar oportunidades por score mínimo e critérios rigorosos
        filtered_opportunities = self._apply_strict_selection_filters(scored_opportunities)
        
        if not filtered_opportunities:
            logger.warning("🔍 Nenhuma oportunidade passou pelos filtros de seleção rigorosos.")
            return []

        # 4. Ordenar por score (do maior para o menor)
        filtered_opportunities.sort(key=lambda x: x[2], reverse=True)

        # 5. Selecionar as N melhores oportunidades
        top_opportunities = filtered_opportunities[:self.max_opportunities_to_select]
        
        logger.info(f"🏆 Selecionadas {len(top_opportunities)} TOP OPORTUNIDADES:")
        for symbol, signal, score, reasons in top_opportunities:
            logger.info(f"   - {symbol}: {signal.action} | Conf: {signal.confidence:.1f}% | Score: {score:.2f} | Risco: {signal.risk_level.name} | SL: {signal.stop_loss:.4f} | TP: {signal.take_profit:.4f} | Razões: {', '.join(reasons)}")

        # Retornar apenas (symbol, TradingSignal)
        return [(op[0], op[1]) for op in top_opportunities]

    def _evaluate_opportunity_score(self, symbol: str, signal: TradingSignal) -> Tuple[float, List[str]]:
        """
        Calcula um score abrangente para a oportunidade, combinando confiança, risco,
        potencial de ganho, e fatores de mercado.
        Score de 0 a 1.
        """
        score = 0.0
        reasons = []

        # 1. Confiança do Sinal (peso alto)
        score += (signal.confidence / 100.0) * 0.40 # 40% do score vem da confiança
        if signal.confidence >= 70: reasons.append("Alta confiança do sinal")
        elif signal.confidence >= 50: reasons.append("Boa confiança do sinal")

        # 2. Força do Sinal (peso médio)
        strength_multipliers = {
            SignalStrength.WEAK: 0.2,
            SignalStrength.MODERATE: 0.4,
            SignalStrength.STRONG: 0.6,
            SignalStrength.VERY_STRONG: 0.8,
            SignalStrength.EXTREME: 1.0
        }
        score += strength_multipliers.get(signal.strength, 0.0) * 0.20 # 20% da força
        if signal.strength.value >= SignalStrength.STRONG.value: reasons.append("Sinal forte")

        # 3. Potencial de Ganho (expected_return / relação R:R) (peso médio)
        # Se signal.take_profit e signal.stop_loss existem, usar a relação R:R
        if signal.take_profit is not None and signal.stop_loss is not None and signal.price is not None:
            if signal.action == "BUY":
                potential_gain_abs = (signal.take_profit - signal.price) / signal.price
                potential_loss_abs = (signal.price - signal.stop_loss) / signal.price
            else: # SELL
                potential_gain_abs = (signal.price - signal.take_profit) / signal.price
                potential_loss_abs = (signal.stop_loss - signal.price) / signal.price
            
            # Adicionar uma pequena constante para evitar divisão por zero se potential_loss_abs for 0
            if potential_loss_abs > 0.00001: # Evitar R:R infinito ou muito grande para trades de alto risco
                risk_reward_ratio = potential_gain_abs / potential_loss_abs
                if risk_reward_ratio > 3.0: score += 0.15 # R:R > 3
                elif risk_reward_ratio > 2.0: score += 0.10 # R:R > 2
                elif risk_reward_ratio > 1.5: score += 0.05 # R:R > 1.5
                if risk_reward_ratio > 2.0: reasons.append(f"R:R favorável ({risk_reward_ratio:.1f}:1)")
            
            # Pelo menos um retorno esperado mínimo
            if potential_gain_abs >= self.min_expected_return: reasons.append("Potencial de ganho alto")
            
        # 4. Ajuste por Nível de Risco (peso negativo para risco alto)
        risk_penalties = {
            RiskLevel.VERY_LOW: 0.0,
            RiskLevel.LOW: 0.0,
            RiskLevel.MODERATE: -0.05,
            RiskLevel.HIGH: -0.15,
            RiskLevel.VERY_HIGH: -0.30,
            RiskLevel.EXTREME_LEVERAGE: -0.50 # Penalidade severa
        }
        score += risk_penalties.get(signal.risk_level, 0.0) * 0.15 # 15% do score ajustado pelo risco
        if signal.risk_level.value >= RiskLevel.HIGH.value: reasons.append(f"Alto risco ({signal.risk_level.name})")

        # 5. Contexto de Mercado (regime, liquidez, funding rate)
        if signal.market_context:
            # Multiplicador para regimes favoráveis
            regime_multipliers = {
                MarketRegime.TRENDING_BULL: 1.1 if signal.action == "BUY" else 0.9,
                MarketRegime.TRENDING_BEAR: 1.1 if signal.action == "SELL" else 0.9,
                MarketRegime.BREAKOUT_BULL: 1.2 if signal.action == "BUY" else 0.8,
                MarketRegime.BREAKOUT_BEAR: 1.2 if signal.action == "SELL" else 0.8,
                MarketRegime.SCALPING_FAVORABLE: 1.05,
                MarketRegime.RANGING_HIGH_VOL: 0.9, # Penaliza um pouco
                MarketRegime.RANGING_LOW_VOL: 1.0,
                MarketRegime.HIGH_LEVERAGE_RISK: 0.5 # Penalidade severa
            }
            score *= regime_multipliers.get(signal.market_context.regime, 1.0)
            if signal.market_context.regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]: reasons.append(f"Regime {signal.market_context.regime.name}")

            # Liquidez (quanto maior o score de liquidez, melhor)
            score += (signal.market_context.liquidity_score - 0.5) * 0.10 # 10% do score pela liquidez
            if signal.market_context.liquidity_score > 0.7: reasons.append("Boa liquidez")

            # Funding rate (se for custo, penaliza; se for ganho, bonifica)
            # Assumindo que funding_consideration é PnL da funding_rate em % da posição
            score += signal.funding_consideration * 5.0 # Converte p/ impacto maior, ajuste conforme necessário
            if signal.funding_consideration < -0.0001: reasons.append("Custo de funding presente")
            elif signal.funding_consideration > 0.0001: reasons.append("Ganho de funding presente")


        # Garantir que o score final esteja entre 0 e 1 (ou um range definido)
        final_score = max(0.0, min(1.0, score))
        
        return final_score, reasons

    def _apply_strict_selection_filters(self, scored_opportunities: List[Tuple[str, TradingSignal, float, List[str]]]) -> List[Tuple[str, TradingSignal, float, List[str]]]:
        """Aplica filtros rigorosos para selecionar as oportunidades de alto ganho."""
        
        filtered = []
        filter_stats = {
            'low_score': 0,
            'low_confidence': 0,
            'high_risk': 0,
            'low_expected_return': 0,
            'forbidden_symbol': 0,
            'max_symbols_selected': 0 # Usado apenas para logs, o corte final é no final
        }

        # Obter os símbolos proibidos do config_ultra_safe.py
        forbidden_symbols = UltraSafeConfig.FORBIDDEN_SYMBOLS
        
        for symbol, signal, score, reasons in scored_opportunities:
            # Filtro 1: Score Geral Mínimo
            if score < self.min_overall_score:
                filter_stats['low_score'] += 1
                logger.debug(f"❌ Rejeitado {symbol} (Score Baixo: {score:.2f} < {self.min_overall_score})")
                continue
            
            # Filtro 2: Confiança do Sinal Mínima (diretamente da estratégia)
            if signal.confidence < self.min_confidence_for_selection:
                filter_stats['low_confidence'] += 1
                logger.debug(f"❌ Rejeitado {symbol} (Confiança Baixa: {signal.confidence:.1f}% < {self.min_confidence_for_selection}%)")
                continue

            # Filtro 3: Nível de Risco Aceitável
            if signal.risk_level.value > self.max_acceptable_risk_level.value:
                filter_stats['high_risk'] += 1
                logger.debug(f"❌ Rejeitado {symbol} (Risco Muito Alto: {signal.risk_level.name})")
                continue
            
            # Filtro 4: Potencial de Ganho Mínimo (baseado no TP vs Preço)
            if signal.take_profit is None or signal.price is None:
                expected_gain_pct = 0.0
            else:
                if signal.action == "BUY":
                    expected_gain_pct = (signal.take_profit - signal.price) / signal.price
                else: # SELL
                    expected_gain_pct = (signal.price - signal.take_profit) / signal.price
            
            if expected_gain_pct < self.min_expected_return:
                filter_stats['low_expected_return'] += 1
                logger.debug(f"❌ Rejeitado {symbol} (Retorno Esperado Baixo: {expected_gain_pct:.2f}% < {self.min_expected_return:.2f}%)")
                continue

            # Filtro 5: Símbolos Proibidos (da UltraSafeConfig)
            if symbol in forbidden_symbols:
                filter_stats['forbidden_symbol'] += 1
                logger.warning(f"🚫 Rejeitado {symbol} (Símbolo na lista de PROIBIDOS)")
                continue

            filtered.append((symbol, signal, score, reasons))
        
        logger.info(f"📊 Resumo dos Filtros de Seleção:")
        logger.info(f"   Total Analisado: {len(scored_opportunities)}")
        logger.info(f"   Aprovados para Seleção: {len(filtered)}")
        logger.info(f"   Rejeições: Baixo Score: {filter_stats['low_score']}, Baixa Confiança: {filter_stats['low_confidence']}, Alto Risco: {filter_stats['high_risk']}, Baixo Retorno Esperado: {filter_stats['low_expected_return']}, Símbolo Proibido: {filter_stats['forbidden_symbol']}")
        
        return filtered