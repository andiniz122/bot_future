#!/usr/bin/env python3
"""
#Bot Inteligente Gate.io - Sistema de Análise Individual por Símbolo
VERSÃO CORRIGIDA - Gestão Ativa de Posições com SL/TP MELHORADO
CORREÇÃO: Problema de símbolos UNKNOWN resolvido + USANDO CONFIG.PY
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
import signal
import sys
import os
from dataclasses import dataclass

# Importações Gate.io
import config as Config
from gate_api import GateAPI 
from data_collector import GateFuturesDataCollector 
from estrategia import AdvancedSignalEngine, TradingSignal 
from portfolio_manager import GateFuturesPortfolioManager 
from notification_manager import TelegramNotifier
from telegram_bot_manager import TelegramBotManager

# Importar nosso novo analisador individual
from symbol_analyzer import SymbolAnalyzer, SymbolCharacteristics

# NOVO: Import para otimização de portfólio
from portfolio_optimizer import PortfolioOptimizer, PortfolioAction, ActionType

# Setup de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('intelligent_bot_gate')

# =====================================================================
# SISTEMA SL/TP MELHORADO INTEGRADO
# =====================================================================

@dataclass
class SLTPConfig:
    """Configuração de Stop Loss e Take Profit MELHORADA"""
    stop_loss_pct: float = -2.0      # -2% SL padrão
    take_profit_pct: float = 2.5     # +2.5% TP padrão
    emergency_sl_pct: float = -3.5   # -3.5% SL emergencial
    emergency_tp_pct: float = 4.0    # +4% TP emergencial
    trailing_stop_enabled: bool = True
    trailing_stop_distance: float = 1.0  # 1% trailing distance
    max_retries: int = 3             # Tentativas de fechamento
    quick_profit_pct: float = 1.0    # Lucro rápido 1%
    breakeven_enabled: bool = True   # Move SL para breakeven

class ImprovedSLTPManager:
    """Gerenciador MELHORADO de Stop Loss e Take Profit"""
    
    def __init__(self, portfolio_manager, gate_api, telegram_notifier):
        self.portfolio_manager = portfolio_manager
        self.gate_api = gate_api
        self.telegram_notifier = telegram_notifier
        self.config = SLTPConfig()
        
        # Cache de trailing stops
        self.trailing_stops: Dict[str, float] = {}
        
        # Cache de breakeven activado
        self.breakeven_activated: Dict[str, bool] = {}
        
        # Histórico de fechamentos
        self.closure_history: List[Dict] = []
        
        # Estatísticas
        self.stats = {
            'total_closures': 0,
            'stop_losses': 0,
            'take_profits': 0,
            'trailing_stops': 0,
            'emergency_closes': 0,
            'total_pnl_usd': 0.0,
            'profitable_closes': 0,
        }
        
        self.logger = logging.getLogger('SLTPManager')

    async def monitor_all_positions_with_sltp(self):
        """
        Monitoramento MELHORADO com SL/TP mais eficiente e robusto
        """
        try:
            await self.portfolio_manager.update_account_info()

            positions = self.portfolio_manager.open_positions.values()
            if not positions:
                self.logger.debug("💼 Nenhuma posição para monitorar SL/TP")
                return
            
            self.logger.info(f"🎯 Monitorando SL/TP MELHORADO de {len(positions)} posições")
            
            # Processar todas as posições em paralelo para eficiência
            tasks = []
            for position in positions:
                symbol = position.get('contract')
                if symbol and symbol != "UNKNOWN" and "_USDT" in symbol and float(position.get('size', 0)) > 0.000001: 
                    task = self.check_sltp_for_position(symbol, position)
                    tasks.append(task)
                else:
                    if symbol:
                        self.logger.warning(f"🚨 Posição inválida ou de size zero detectada no SLTP: {symbol} (Size: {position.get('size', 0)})")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log dos resultados
                successful_checks = sum(1 for r in results if r is True)
                self.logger.info(f"📊 SL/TP: {successful_checks}/{len(tasks)} posições verificadas")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no monitoramento SL/TP MELHORADO: {e}")

    async def check_sltp_for_position(self, symbol: str, position: Dict) -> bool:
        """
        Verificação SL/TP MELHORADA para uma posição específica
        """
        try:
            pnl_data = await self.portfolio_manager.get_position_pnl(symbol)
            if not pnl_data:
                self.logger.warning(f"⚠️ {symbol}: Não foi possível obter PnL do PortfolioManager")
                return False
            
            pnl_pct = pnl_data.get('pnl_percentage', 0)
            pnl_usd = pnl_data.get('unrealized_pnl', 0)
            size = float(position.get('size', 0))
            
            self.logger.debug(f"📊 {symbol}: PnL {pnl_pct:+.2f}% ({pnl_usd:+.2f} USDT), Size: {size}")
            
            # 1. Verificar breakeven primeiro (para posições lucrativas)
            if self.config.breakeven_enabled:
                breakeven_triggered = await self.check_breakeven_move(symbol, pnl_pct)
                if breakeven_triggered:
                    self.logger.info(f"🎯 {symbol}: Breakeven ativado")
            
            # 2. Verificar trailing stop
            if self.config.trailing_stop_enabled:
                trailing_triggered = await self.check_trailing_stop(symbol, pnl_pct, pnl_usd)
                if trailing_triggered:
                    return True  # Posição foi fechada
            
            # 3. Verificar SL/TP padrão e emergencial
            close_triggered = await self.check_standard_sltp(symbol, pnl_pct, pnl_usd, position)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro verificando SL/TP {symbol}: {e}")
            return False

    async def check_breakeven_move(self, symbol: str, pnl_pct: float) -> bool:
        """
        Sistema de Breakeven - Move SL para entrada quando em lucro
        """
        try:
            if symbol in self.breakeven_activated or pnl_pct <= self.config.quick_profit_pct:
                return False
            
            self.logger.info(f"🎯 Ativando breakeven para {symbol} (PnL: {pnl_pct:.2f}%)")
            self.breakeven_activated[symbol] = True
            
            await self.notify_breakeven_activation(symbol, pnl_pct)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro no breakeven {symbol}: {e}")
            return False

    async def check_trailing_stop(self, symbol: str, pnl_pct: float, pnl_usd: float) -> bool:
        """
        Sistema de Trailing Stop MELHORADO
        """
        try:
            if pnl_pct <= 0:
                if symbol in self.trailing_stops:
                    del self.trailing_stops[symbol]
                    self.logger.debug(f"🔄 {symbol}: Trailing stop removido (sem lucro)")
                return False
            
            if symbol not in self.trailing_stops:
                self.trailing_stops[symbol] = pnl_pct
                self.logger.info(f"🎯 Trailing stop iniciado: {symbol} @ {pnl_pct:.2f}%")
                return False
            
            best_pnl = self.trailing_stops[symbol]
            
            if pnl_pct > best_pnl:
                self.trailing_stops[symbol] = pnl_pct
                self.logger.info(f"📈 {symbol}: Novo máximo trailing {pnl_pct:.2f}% (anterior: {best_pnl:.2f}%)")
                return False
            
            drop_from_peak = best_pnl - pnl_pct
            
            if drop_from_peak >= self.config.trailing_stop_distance:
                self.logger.info(f"🎯 TRAILING STOP ATIVADO: {symbol}")
                self.logger.info(f"   📊 Máximo: {best_pnl:.2f}% | Atual: {pnl_pct:.2f}% | Queda: {drop_from_peak:.2f}%")
                
                success = await self.execute_sltp_close(
                    symbol, 
                    f"Trailing Stop (queda de {drop_from_peak:.2f}% do máximo {best_pnl:.2f}%)", 
                    pnl_pct, 
                    pnl_usd,
                    close_type="TRAILING_STOP"
                )
                
                if success:
                    if symbol in self.trailing_stops:
                        del self.trailing_stops[symbol]
                    if symbol in self.breakeven_activated:
                        del self.breakeven_activated[symbol]
                    
                    self.stats['trailing_stops'] += 1
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Erro no trailing stop {symbol}: {e}")
            return False

    async def check_standard_sltp(self, symbol: str, pnl_pct: float, pnl_usd: float, position: Dict) -> bool:
        """
        Verificação de SL/TP padrão MELHORADA
        """
        try:
            should_close = False
            close_reason = ""
            close_type = ""
            
            if pnl_pct <= self.config.emergency_sl_pct:
                should_close = True
                close_reason = f"🚨 STOP LOSS EMERGENCIAL ({pnl_pct:.2f}% ≤ {self.config.emergency_sl_pct:.1f}%)"
                close_type = "EMERGENCY_SL"
                
            elif pnl_pct >= self.config.emergency_tp_pct:
                should_close = True
                close_reason = f"🎉 TAKE PROFIT EMERGENCIAL ({pnl_pct:.2f}% ≥ {self.config.emergency_tp_pct:.1f}%)"
                close_type = "EMERGENCY_TP"
                
            elif pnl_pct <= self.get_adjusted_sl_threshold(symbol):
                threshold = self.get_adjusted_sl_threshold(symbol)
                should_close = True
                close_reason = f"Stop Loss ativado ({pnl_pct:.2f}% ≤ {threshold:.1f}%)"
                close_type = "STOP_LOSS"
                
            elif pnl_pct >= self.config.take_profit_pct:
                should_close = True
                close_reason = f"Take Profit ativado ({pnl_pct:.2f}% ≥ {self.config.take_profit_pct:.1f}%)"
                close_type = "TAKE_PROFIT"
            
            if should_close:
                success = await self.execute_sltp_close(symbol, close_reason, pnl_pct, pnl_usd, close_type)
                
                if success:
                    if close_type in ["STOP_LOSS", "EMERGENCY_SL"]:
                        self.stats['stop_losses'] += 1
                    elif close_type in ["TAKE_PROFIT", "EMERGENCY_TP"]:
                        self.stats['take_profits'] += 1
                    
                    if close_type.startswith("EMERGENCY"):
                        self.stats['emergency_closes'] += 1
                    
                    if symbol in self.trailing_stops:
                        del self.trailing_stops[symbol]
                    if symbol in self.breakeven_activated:
                        del self.breakeven_activated[symbol]
                
                return success
                
            return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro no SL/TP padrão {symbol}: {e}")
            return False

    def get_adjusted_sl_threshold(self, symbol: str) -> float:
        """Obter threshold de SL ajustado (breakeven se ativado)"""
        if symbol in self.breakeven_activated:
            return -0.1  # Breakeven com margem de 0.1%
        return self.config.stop_loss_pct

    async def execute_sltp_close(self, symbol: str, reason: str, pnl_pct: float, 
                                pnl_usd: float, close_type: str) -> bool:
        """
        Execução MELHORADA de fechamento SL/TP com retry e validação
        """
        try:
            self.logger.info(f"🔄 Executando fechamento SL/TP: {symbol}")
            self.logger.info(f"   📋 Motivo: {reason}")
            
            positions = await self.portfolio_manager.get_open_positions_ws()
            position_exists = any(p.get('contract') == symbol for p in positions)
            
            if not position_exists:
                self.logger.warning(f"⚠️ {symbol}: Posição não existe mais")
                return False
            
            for attempt in range(self.config.max_retries):
                try:
                    self.logger.info(f"   🔄 Tentativa {attempt + 1}/{self.config.max_retries}")
                    
                    close_result = await self.portfolio_manager.close_single_position(
                        symbol, 
                        reason=f"{close_type}_{reason}"
                    )
                    
                    if close_result and close_result.get('success'):
                        self.record_closure(symbol, close_type, pnl_pct, pnl_usd, reason)
                        
                        await self.notify_sltp_execution(symbol, close_type, pnl_pct, pnl_usd, reason)
                        
                        self.stats['total_closures'] += 1
                        self.stats['total_pnl_usd'] += pnl_usd
                        
                        if pnl_usd > 0:
                            self.stats['profitable_closes'] += 1
                        
                        self.logger.info(f"✅ {symbol}: Fechamento SL/TP executado com sucesso")
                        return True
                        
                    else:
                        error_msg = close_result.get('error', 'Erro desconhecido') if close_result else 'Sem resposta'
                        self.logger.warning(f"⚠️ Tentativa {attempt+1} falhou: {error_msg}")
                        
                        if attempt < self.config.max_retries - 1:
                            wait_time = (attempt + 1) * 0.5
                            await asyncio.sleep(wait_time)
                        
                except Exception as e:
                    self.logger.error(f"❌ Tentativa {attempt+1} com erro: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 0.5)
            
            self.logger.error(f"❌ {symbol}: FALHA CRÍTICA - Todas as tentativas de fechamento falharam")
            
            await self.notify_critical_failure(symbol, reason, pnl_pct, pnl_usd)
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Erro crítico executando fechamento SL/TP {symbol}: {e}")
            return False

    def record_closure(self, symbol: str, close_type: str, pnl_pct: float, 
                      pnl_usd: float, reason: str):
        """Registrar fechamento no histórico"""
        try:
            closure_record = {
                'timestamp': time.time(),
                'symbol': symbol,
                'type': close_type,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'reason': reason
            }
            
            self.closure_history.append(closure_record)
            
            if len(self.closure_history) > 100:
                self.closure_history = self.closure_history[-100:]
                
            self.logger.info(f"📝 Fechamento registrado: {symbol} ({close_type}) PnL: {pnl_pct:+.2f}%")
            
        except Exception as e:
            self.logger.error(f"❌ Erro registrando fechamento: {e}")

    async def notify_breakeven_activation(self, symbol: str, pnl_pct: float):
        """Notificar ativação do breakeven"""
        try:
            message = (
                f"🎯 <b>BREAKEVEN ATIVADO</b> 🎯\n\n"
                f"📊 Ativo: <code>{symbol}</code>\n"
                f"📈 PnL Atual: <b>+{pnl_pct:.2f}%</b>\n"
                f"🛡️ Stop Loss movido para breakeven\n"
                f"💡 Posição agora protegida!\n\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            )
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            self.logger.error(f"❌ Erro notificando breakeven: {e}")

    async def notify_sltp_execution(self, symbol: str, close_type: str, pnl_pct: float, 
                                   pnl_usd: float, reason: str):
        """Notificação MELHORADA de execução SL/TP"""
        try:
            icons = {
                'STOP_LOSS': '🛑',
                'TAKE_PROFIT': '🎯', 
                'TRAILING_STOP': '📈',
                'EMERGENCY_SL': '🚨',
                'EMERGENCY_TP': '🎉'
            }
            
            icon = icons.get(close_type, '🔄')
            pnl_sign = "+" if pnl_usd >= 0 else ""
            color = "🟢" if pnl_usd >= 0 else "🔴"
            
            win_rate = (self.stats['profitable_closes'] / max(self.stats['total_closures'], 1)) * 100
            
            message = (
                f"{icon} <b>SL/TP EXECUTADO</b> {color}\n\n"
                f"📊 Ativo: <code>{symbol}</code>\n"
                f"🏷️ Tipo: <b>{close_type.replace('_', ' ')}</b>\n"
                f"📈 Resultado: <b>{pnl_sign}{pnl_pct:.2f}%</b> (<b>{pnl_sign}{pnl_usd:.2f} USDT</b>)\n"
                f"📝 Detalhes: <i>{reason}</i>\n\n"
                f"📊 <b>Estatísticas Sessão:</b>\n"
                f"• Total fechamentos: <b>{self.stats['total_closures']}</b>\n"
                f"• Taxa de sucesso: <b>{win_rate:.1f}%</b>\n"
                f"• PnL acumulado: <b>{self.stats['total_pnl_usd']:+.2f} USDT</b>\n\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            )
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            self.logger.error(f"❌ Erro enviando notificação SL/TP: {e}", exc_info=True)

    async def notify_critical_failure(self, symbol: str, reason: str, pnl_pct: float, pnl_usd: float):
        """Notificar falha crítica no fechamento"""
        try:
            message = (
                f"🚨 <b>FALHA CRÍTICA SL/TP</b> 🚨\n\n"
                f"📊 Ativo: <code>{symbol}</code>\n"
                f"📈 PnL: <b>{pnl_pct:+.2f}%</b> ({pnl_usd:+.2f} USDT)\n"
                f"❌ Erro: <i>{reason}</i>\n\n"
                f"⚠️ <b>INTERVENÇÃO MANUAL NECESSÁRIA!</b>\n"
                f"🔧 Verificar posição manualmente na exchange\n\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            )
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            self.logger.error(f"❌ Erro notificando falha crítica: {e}", exc_info=True)

    async def get_sltp_statistics(self) -> Dict:
        """Estatísticas completas do sistema SL/TP"""
        try:
            if not self.closure_history:
                return {
                    'total_closures': 0,
                    'active_trailing_stops': len(self.trailing_stops),
                    'active_breakeven': len(self.breakeven_activated)
                }
            
            recent_closures = [c for c in self.closure_history if time.time() - c['timestamp'] < 86400]
            
            stats = {
                'total_closures': self.stats['total_closures'],
                'stop_losses': self.stats['stop_losses'],
                'take_profits': self.stats['take_profits'],
                'trailing_stops': self.stats['trailing_stops'],
                'emergency_closes': self.stats['emergency_closes'],
                'total_pnl_usd': self.stats['total_pnl_usd'],
                'profitable_closes': self.stats['profitable_closes'],
                'win_rate_pct': (self.stats['profitable_closes'] / max(self.stats['total_closures'], 1)) * 100,
                'recent_24h_closures': len(recent_closures),
                'recent_24h_pnl': sum(c['pnl_usd'] for c in recent_closures),
                'active_trailing_stops': len(self.trailing_stops),
                'active_breakeven': len(self.breakeven_activated),
                'trailing_symbols': list(self.trailing_stops.keys()),
                'breakeven_symbols': list(self.breakeven_activated.keys()),
                'avg_pnl_pct': sum(c['pnl_pct'] for c in self.closure_history) / len(self.closure_history) if self.closure_history else 0,
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando estatísticas: {e}", exc_info=True)
            return {'error': str(e)}

# =====================================================================
# CLASSES ORIGINAIS MANTIDAS
# =====================================================================

@dataclass
class MarketConditions:
    """Classe para armazenar condições gerais de mercado"""
    overall_volatility: float = 0.0
    market_sentiment: str = "NEUTRAL"
    risk_level: str = "MEDIUM"
    data_quality: float = 0.0
    timestamp: float = 0.0
    active_symbols_count: int = 0
    avg_liquidity_score: float = 0.0

class AdaptiveMarketMode:
    """Modos adaptativos do bot"""
    DISCOVERY = "discovery"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EMERGENCY = "emergency"

class SymbolMappingManager:
    """Gerenciador de mapeamento de símbolos para evitar 'UNKNOWN'"""
    
    def __init__(self):
        self.symbol_registry: Dict[str, Dict] = {}
        self.active_symbols: List[str] = []
        self.symbol_map: Dict[str, str] = {}
        
    def register_active_symbols(self, symbols: List[str]):
        """Registra símbolos ativos descobertos"""
        valid_symbols = [s for s in symbols if s and s != "UNKNOWN" and "_USDT" in s]
        self.active_symbols = valid_symbols.copy()
        logger.info(f"📋 Símbolos válidos registrados: {valid_symbols}")
        
        for symbol in valid_symbols:
            if symbol not in self.symbol_registry:
                self.symbol_registry[symbol] = {
                    'status': 'active',
                    'registered_at': time.time(),
                    'last_data_update': 0,
                    'data_quality': 'unknown'
                }
    
    def map_symbol_correctly(self, potential_unknown: str, context_symbols: List[str]) -> str:
        """Mapeia símbolos UNKNOWN para símbolos reais baseado no contexto"""
        if potential_unknown and potential_unknown != "UNKNOWN":
            return potential_unknown
        
        if context_symbols:
            for symbol in context_symbols:
                if symbol and symbol != "UNKNOWN" and "_USDT" in symbol:
                    logger.info(f"🔄 Mapeando UNKNOWN -> {symbol}")
                    return symbol
        
        return "BTC_USDT"
    
    def validate_symbol_in_data(self, symbol: str, klines_data: Dict, price_data: Dict) -> bool:
        """Valida se símbolo existe nos dados coletados"""
        
        if not symbol or symbol == "UNKNOWN" or len(symbol) < 3:
            logger.warning(f"❌ {symbol}: Símbolo inválido")
            return False
        
        has_klines = symbol in klines_data and klines_data[symbol] is not None
        has_price = symbol in price_data and price_data[symbol] is not None
        
        if not has_klines:
            logger.warning(f"❌ {symbol}: Sem dados de klines")
            return False
            
        if not has_price:
            logger.warning(f"❌ {symbol}: Sem dados de preço")
            return False
        
        klines = klines_data[symbol]
        if isinstance(klines, pd.DataFrame) and len(klines) < 20:
            logger.warning(f"❌ {symbol}: Poucos dados de klines ({len(klines)})")
            return False
        elif not isinstance(klines, pd.DataFrame):
            logger.warning(f"❌ {symbol}: Dados de klines inválidos")
            return False
            
        logger.debug(f"✅ {symbol}: Dados válidos (klines: {len(klines)}, price: {price_data[symbol]})")
        return True
    
    def clean_unknown_symbols(self, klines_data: Dict, price_data: Dict) -> Tuple[Dict, Dict]:
        """Remove símbolos UNKNOWN dos dados e corrige mapeamento"""
        
        if "UNKNOWN" in klines_data:
            klines_data.pop("UNKNOWN")
            logger.info("🧹 Removido UNKNOWN dos klines")
            
        if "UNKNOWN" in price_data:
            price_data.pop("UNKNOWN")
            logger.info("🧹 Removido UNKNOWN dos preços")
        
        valid_klines = {}
        valid_prices = {}
        
        for symbol in self.active_symbols:
            if self.validate_symbol_in_data(symbol, klines_data, price_data):
                valid_klines[symbol] = klines_data[symbol]
                valid_prices[symbol] = price_data[symbol]
        
        logger.info(f"🔍 Símbolos após limpeza: {list(valid_klines.keys())}")
        return valid_klines, valid_prices

# =====================================================================
# BOT PRINCIPAL COM SL/TP MELHORADO E CORREÇÃO DE SÍMBOLOS
# =====================================================================

class IntelligentBotGateIndividual:
    """Bot Inteligente com Análise Individual por Símbolo + SL/TP MELHORADO + CORREÇÃO SÍMBOLOS"""
    
    def __init__(self):
        # APIs e Managers
        self.gate_api = GateAPI()
        self.data_collector = GateFuturesDataCollector(self.gate_api)
        self.portfolio_manager = GateFuturesPortfolioManager(self.gate_api)
        self.telegram_notifier = TelegramNotifier()
        self.telegram_bot_manager = TelegramBotManager(self)
        
        # Analisador individual de símbolos
        self.symbol_analyzer = SymbolAnalyzer()
        
        # Gerenciador de mapeamento de símbolos
        self.symbol_mapping_manager = SymbolMappingManager()
        
        # Inicializar signal_engine
        self.signal_engine = AdvancedSignalEngine(Config.STRATEGY_CONFIG.copy())

        # Sistema de Otimização de Portfolio
        self.portfolio_optimizer = PortfolioOptimizer(
            self.portfolio_manager, 
            self.symbol_analyzer, 
            self.gate_api
        )
        self.portfolio_optimizer.data_collector = self.data_collector
        self.portfolio_optimizer.signal_engine = self.signal_engine 
        
        # *** NOVO: Sistema SL/TP MELHORADO ***
        self.sltp_manager = ImprovedSLTPManager(
            self.portfolio_manager,
            self.gate_api,
            self.telegram_notifier
        )
        
        # Configurar SL/TP personalizado (pode ser ajustado via config)
        self.sltp_manager.config.stop_loss_pct = getattr(Config, 'STOP_LOSS_PCT', -2.0)
        self.sltp_manager.config.take_profit_pct = getattr(Config, 'TAKE_PROFIT_PCT', 2.5)
        self.sltp_manager.config.trailing_stop_enabled = getattr(Config, 'TRAILING_STOP_ENABLED', True)
        self.sltp_manager.config.breakeven_enabled = getattr(Config, 'BREAKEVEN_ENABLED', True)
        self.sltp_manager.config.emergency_sl_pct = getattr(Config, 'EMERGENCY_SL_PCT', -3.5)
        self.sltp_manager.config.emergency_tp_pct = getattr(Config, 'EMERGENCY_TP_PCT', 4.0)
        self.sltp_manager.config.trailing_stop_distance = getattr(Config, 'TRAILING_STOP_DISTANCE', 1.0)
        self.sltp_manager.config.max_retries = getattr(Config, 'MAX_CLOSE_RETRIES', 3)
        self.sltp_manager.config.quick_profit_pct = getattr(Config, 'QUICK_PROFIT_FOR_BREAKEVEN', 1.0)
        
        # Configurações padrão de gestão de risco (garantir compatibilidade com config.py)
        if not hasattr(Config, 'MAX_TOTAL_RISK_PERCENT'):
            Config.MAX_TOTAL_RISK_PERCENT = 30.0  # Valor padrão do config.py
            logger.info("📊 Configuração padrão aplicada: MAX_TOTAL_RISK_PERCENT = 30.0%")
        
        if not hasattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT'):
            Config.MAX_PORTFOLIO_RISK_PERCENT = 10.0  # Fallback secundário
            logger.info("📊 Configuração padrão aplicada: MAX_PORTFOLIO_RISK_PERCENT = 10.0%")
        
        if not hasattr(Config, 'MAX_CONCURRENT_POSITIONS'):
            Config.MAX_CONCURRENT_POSITIONS = 5
            logger.info("📊 Configuração padrão aplicada: MAX_CONCURRENT_POSITIONS = 5")
        
        if not hasattr(Config, 'MAX_OPEN_POSITIONS'):
            Config.MAX_OPEN_POSITIONS = 5
            logger.info("📊 Configuração padrão aplicada: MAX_OPEN_POSITIONS = 5")
        
        if not hasattr(Config, 'MIN_BALANCE_FOR_NEW_TRADE'):
            Config.MIN_BALANCE_FOR_NEW_TRADE = 20.0
            logger.info("📊 Configuração padrão aplicada: MIN_BALANCE_FOR_NEW_TRADE = 20.0 USDT")
        
        if not hasattr(Config, 'MIN_CONFIDENCE_TO_TRADE'):
            Config.MIN_CONFIDENCE_TO_TRADE = 40.0
            logger.info("📊 Configuração padrão aplicada: MIN_CONFIDENCE_TO_TRADE = 40.0%")
        
        # Estado do bot
        self.is_running = False
        self.cycle_count = 0
        self.consecutive_failures = 0
        
        # Condições gerais de mercado
        self.current_market_conditions = MarketConditions()
        
        # Modo adaptativo
        self.current_mode = AdaptiveMarketMode.DISCOVERY
        self.last_mode_change = time.time()
        
        # Cache de análises por símbolo
        self.symbol_last_analysis: Dict[str, float] = {}
        self.analysis_interval = 3600
        
        # Estatísticas MELHORADAS
        self.stats = {
            'total_cycles': 0, 'successful_cycles': 0, 'failed_cycles': 0,
            'opportunities_found': 0, 'orders_executed': 0, 'orders_successful': 0,
            'symbols_analyzed': 0, 'individual_analyses': 0,
            'start_time': None, 'total_pnl_usd': 0.0,
            'closed_positions_count': 0, 'daily_pnl_usd': 0.0,
            'last_daily_report_date': None,
            'symbols_performance': {},
            'emergency_closes': 0, 'active_management_actions': 0,
            'symbols_discovered': 0,
            # Estatísticas SL/TP
            'sltp_total_closures': 0,
            'sltp_profitable_closes': 0,
            'sltp_total_pnl': 0.0
        }
        
        logger.info(f"🧠 Bot com SL/TP MELHORADO inicializado - Modo: {self.current_mode.upper()}")
        logger.info(f"📊 DEBUG - CONFIGURAÇÕES CARREGADAS DO CONFIG.PY:")
        logger.info(f"   • Config.MAX_TOTAL_RISK_PERCENT: {getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 'NÃO EXISTE')}")
        logger.info(f"   • Config.MAX_PORTFOLIO_RISK_PERCENT: {getattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT', 'NÃO EXISTE')}")
        logger.info(f"   • Config.MAX_CONCURRENT_POSITIONS: {getattr(Config, 'MAX_CONCURRENT_POSITIONS', 'NÃO EXISTE')}")
        logger.info(f"   • Config.MAX_OPEN_POSITIONS: {getattr(Config, 'MAX_OPEN_POSITIONS', 'NÃO EXISTE')}")
        logger.info(f"   • Config.ULTRA_SAFE_MODE: {getattr(Config, 'ULTRA_SAFE_MODE', 'NÃO EXISTE')}")
        
        logger.info(f"📊 CONFIGURAÇÕES DE RISCO CALCULADAS:")
        
        # Mostrar qual configuração está sendo usada para risco
        max_total_risk = getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 30.0)
        max_portfolio_risk = getattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT', 10.0)
        
        if hasattr(Config, 'MAX_TOTAL_RISK_PERCENT'):
            logger.info(f"   • Risco máximo do portfólio: {max_total_risk:.1f}% (usando MAX_TOTAL_RISK_PERCENT)")
        elif hasattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT'):
            logger.info(f"   • Risco máximo do portfólio: {max_portfolio_risk:.1f}% (usando MAX_PORTFOLIO_RISK_PERCENT)")
        else:
            logger.info(f"   • Risco máximo do portfólio: {max_total_risk:.1f}% (usando padrão)")
        
        max_positions = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 5)
        if hasattr(Config, 'MAX_CONCURRENT_POSITIONS'):
            logger.info(f"   • Posições máximas: {max_positions} (usando MAX_CONCURRENT_POSITIONS)")
        elif hasattr(Config, 'MAX_OPEN_POSITIONS'):
            max_positions = getattr(Config, 'MAX_OPEN_POSITIONS', 5)
            logger.info(f"   • Posições máximas: {max_positions} (usando MAX_OPEN_POSITIONS)")
        else:
            logger.info(f"   • Posições máximas: {max_positions} (usando padrão)")
            
        min_balance = getattr(Config, 'MIN_BALANCE_FOR_NEW_TRADE', 20.0)
        logger.info(f"   • Saldo mínimo para trade: {min_balance:.1f} USDT")
        
        min_confidence = getattr(Config, 'MIN_CONFIDENCE_TO_TRADE', 40.0)
        logger.info(f"   • Confiança mínima: {min_confidence:.1f}%")
        
        logger.info(f"📋 MODO ULTRA_SAFE: {'ATIVO' if getattr(Config, 'ULTRA_SAFE_MODE', False) else 'DESATIVADO'}")

    # =====================================================================
    # MÉTODOS PRINCIPAIS COM SL/TP INTEGRADO E CORREÇÃO DE SÍMBOLOS
    # =====================================================================
    
    async def run_trading_cycle_individual(self):
        """Ciclo principal com SL/TP MELHORADO integrado e correção de símbolos"""
        cycle_start = time.time()
        self.cycle_count += 1
        self.stats['total_cycles'] += 1
        
        logger.info(f"\n🔄 === CICLO COM SL/TP MELHORADO {self.cycle_count} === (USANDO CONFIG.PY)")
        
        try:
            # Debug de saldo ocasional
            if self.cycle_count % 5 == 1:
                await self.debug_balance_issue()
            
            # 1. Descoberta de símbolos
            symbols = await self.adaptive_symbol_discovery_individual()
            if not symbols:
                logger.warning("⚠️ Nenhum símbolo descoberto - executando monitoramento")
                return await self.run_monitoring_cycle()
            
            # 2. Registrar símbolos
            self.symbol_mapping_manager.register_active_symbols(symbols)
            logger.info(f"🎯 DEBUG SÍMBOLOS: Símbolos registrados no mapping manager: {self.symbol_mapping_manager.active_symbols}")
            
            # 3. Atualizar dados
            try:
                logger.info(f"📡 Coletando dados para {len(symbols)} símbolos...")
                klines_data = await self.data_collector.fetch_klines_parallel(
                    symbols, 
                    limit=getattr(Config, 'KLINES_LIMIT', 200)
                )
                
                if not klines_data:
                    logger.error("❌ Nenhum dado de klines retornado")
                    return False
                
                valid_klines = {}
                for symbol in symbols:
                    if symbol in klines_data and klines_data[symbol] is not None:
                        valid_klines[symbol] = klines_data[symbol]
                
                logger.info(f"📊 Dados válidos coletados: {list(valid_klines.keys())}")
                self.data_collector.klines_data = valid_klines
                
                await self.data_collector.update_current_prices_batch(symbols)
                
            except Exception as e:
                logger.error(f"❌ Erro atualizando dados: {e}", exc_info=True)
                return False
            
            # 4. **CORREÇÃO PRINCIPAL**: Validar e limpar dados
            all_klines = self.data_collector.get_all_klines_data()
            all_prices = self.data_collector.get_all_current_prices()
            
            logger.info(f"🔍 ANTES DA LIMPEZA - Klines: {list(all_klines.keys())}")
            logger.info(f"🔍 ANTES DA LIMPEZA - Preços: {list(all_prices.keys())}")
            
            cleaned_klines, cleaned_prices = self.symbol_mapping_manager.clean_unknown_symbols(all_klines, all_prices)
            
            logger.info(f"🔍 APÓS LIMPEZA - Klines: {list(cleaned_klines.keys())}")
            logger.info(f"🔍 APÓS LIMPEZA - Preços: {list(cleaned_prices.keys())}")
            
            if cleaned_klines != all_klines or cleaned_prices != all_prices:
                logger.info("🧹 Dados limpos - símbolos inválidos removidos")
                self.data_collector.klines_data = cleaned_klines
                self.data_collector.current_prices = cleaned_prices
            
            if not cleaned_klines or not cleaned_prices:
                logger.warning("⚠️ Sem dados válidos após limpeza")
                return False
            
            logger.info(f"📊 Dados finais válidos: {len(cleaned_klines)} símbolos")
            
            # 5. Atualizar condições e portfolio
            await self.update_market_conditions_simple()
            self.portfolio_manager.current_prices = cleaned_prices
            await self.portfolio_manager.update_account_info()
            
            # 6. *** GESTÃO ATIVA COM SL/TP MELHORADO ***
            logger.info("🎯 Executando monitoramento SL/TP MELHORADO...")
            await self.sltp_manager.monitor_all_positions_with_sltp()
            
            # 7. **CORREÇÃO PRINCIPAL**: Análise de sinais com símbolos corretos
            logger.info("🧠 Iniciando análise de sinais CORRIGIDA...")
            logger.info(f"🎯 PASSANDO PARA SIGNAL ENGINE - Klines: {list(cleaned_klines.keys())}")
            logger.info(f"🎯 PASSANDO PARA SIGNAL ENGINE - Preços: {list(cleaned_prices.keys())}")
            
            # **ESTA É A CORREÇÃO PRINCIPAL**: Passar os dados limpos diretamente
            signals = self.signal_engine.analyze(cleaned_klines, cleaned_prices)
            
            logger.info(f"🎯 SINAIS RETORNADOS PELO ENGINE: {list(signals.keys())}")
            
            opportunities = []
            for symbol, sig in signals.items():
                logger.info(f"🔍 ANALISANDO SINAL: Symbol={symbol}, Action={sig.action if sig else 'None'}")
                if symbol != "UNKNOWN" and symbol in cleaned_klines and sig and sig.action != "HOLD":
                    opportunities.append((symbol, sig))
                    logger.info(f"✅ OPORTUNIDADE VÁLIDA: {symbol} -> {sig.action}")
                elif symbol == "UNKNOWN":
                    logger.warning(f"❌ SÍMBOLO UNKNOWN DETECTADO NO SINAL!")
            
            logger.info(f"📊 Sinais válidos encontrados: {len(opportunities)}")
            
            # 8. Aplicar filtros
            if opportunities:
                filtered_opportunities = await self._simplified_intelligent_symbol_filtering(opportunities) 
            else:
                filtered_opportunities = []
            
            logger.info(f"📊 Oportunidades após filtros: {len(filtered_opportunities)}")
            
            # 9. Otimização de portfolio
            optimization_actions = await self.optimize_portfolio_intelligently(filtered_opportunities)
            
            # 10. Executar ações
            executed_optimizations = await self.execute_portfolio_optimizations(optimization_actions)
            
            # 11. **CORREÇÃO**: Aplicar fix do portfolio_manager antes de executar trades
            await self.fix_portfolio_manager_config()
            
            # 12. Executar trades restantes
            remaining_opportunities = await self.get_remaining_opportunities(filtered_opportunities, optimization_actions)
            executed_regular = 0
            
            if remaining_opportunities:
                can_trade = await self.can_open_new_position_intelligent(
                    await self.calculate_simplified_portfolio_risk(),
                    getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 30.0)
                )
                if can_trade:
                    executed_regular = await self.execute_intelligent_trades_individual(remaining_opportunities)
                else:
                    logger.warning("⚠️ Verificação final: Não pode abrir novas posições baseado no risco calculado")
            
            # 13. Atualizar estatísticas
            total_executed = executed_optimizations + executed_regular
            self.stats['opportunities_found'] += len(opportunities)
            self.stats['orders_executed'] += total_executed
            self.stats['successful_cycles'] += 1
            
            # Sincronizar stats SL/TP
            sltp_stats = await self.sltp_manager.get_sltp_statistics()
            self.stats['sltp_total_closures'] = sltp_stats.get('total_closures', 0)
            self.stats['sltp_profitable_closes'] = sltp_stats.get('profitable_closes', 0)
            self.stats['sltp_total_pnl'] = sltp_stats.get('total_pnl_usd', 0.0)
            
            # **DEBUG COMPLETO**: Verificar todas as configurações
            if self.cycle_count == 1:  # Só no primeiro ciclo
                await self.debug_all_configurations()
            
            cycle_time = time.time() - cycle_start
            logger.info(f"✅ Ciclo com SL/TP MELHORADO concluído em {cycle_time:.1f}s")
            logger.info(f"   📊 Símbolos válidos: {len(cleaned_klines)}")
            logger.info(f"   🎯 Oportunidades: {len(opportunities)}")
            logger.info(f"   ⚡ Otimizações: {executed_optimizations}")
            logger.info(f"   💼 Novos trades: {executed_regular}")
            logger.info(f"   🎯 SL/TP ativos: {sltp_stats.get('active_trailing_stops', 0)} trailing, {sltp_stats.get('active_breakeven', 0)} breakeven")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no ciclo com SL/TP: {e}", exc_info=True)
            self.consecutive_failures += 1
            if self.consecutive_failures >= getattr(Config, 'MAX_CONSECUTIVE_FAILURES', 3):
                logger.critical(f"💥 {self.consecutive_failures} falhas consecutivas! Ativando shutdown de emergência.")
                self.is_running = False
            return False

    async def run_monitoring_cycle(self):
        """Ciclo de monitoramento com SL/TP MELHORADO"""
        logger.info("👁️ Ciclo de monitoramento com SL/TP")
        
        try:
            await self.sltp_manager.monitor_all_positions_with_sltp()
            await self.update_market_conditions_simple()
            return True
        except Exception as e:
            logger.error(f"❌ Erro no monitoramento: {e}", exc_info=True)
            return False

    async def log_individual_status(self):
        """Log de status com estatísticas SL/TP"""
        try:
            balance = await self.gate_api.get_futures_balance()
            positions = await self.portfolio_manager.get_open_positions_ws()
            sltp_stats = await self.sltp_manager.get_sltp_statistics()
            
            logger.info(f"🧠 STATUS BOT COM SL/TP MELHORADO (USANDO CONFIG.PY):")
            logger.info(f"   • Ciclo: {self.cycle_count} | Sucessos: {self.stats['successful_cycles']} | Falhas Consecutivas: {self.consecutive_failures}")
            logger.info(f"   • Saldo: {balance.get('equity', 0):.2f} USDT | Posições: {len(positions)}")
            logger.info(f"   • Símbolos descobertos: {self.stats['symbols_discovered']}")
            logger.info(f"   • Ordens executadas: {self.stats['orders_executed']} ({self.stats['orders_successful']} sucessos)")
            logger.info(f"   • SL/TP fechamentos: {sltp_stats.get('total_closures', 0)} | Taxa sucesso: {sltp_stats.get('win_rate_pct', 0):.1f}%")
            logger.info(f"   • SL/TP ativos: {sltp_stats.get('active_trailing_stops', 0)} trailing | {sltp_stats.get('active_breakeven', 0)} breakeven")
            logger.info(f"   • PnL SL/TP: {sltp_stats.get('total_pnl_usd', 0):+.2f} USDT")
            
        except Exception as e:
            logger.error(f"❌ Erro no log de status: {e}", exc_info=True)

    # =====================================================================
    # MÉTODOS AUXILIARES (mantidos do código original)
    # =====================================================================
    
    async def debug_balance_issue(self):
        """Debug específico do problema de saldo"""
        logger.info("🔍 === DEBUG ESPECÍFICO DO SALDO ===")
        
        try:
            balance = await self.gate_api.get_futures_balance()
            logger.info(f"📊 Saldo completo: {balance}")
            logger.info(f"📊 Tipo: {type(balance)}")
            
            if isinstance(balance, dict): 
                for key, value in balance.items(): 
                    try:
                        float_value = float(value) if value else 0.0 
                        logger.info(f"   {key}: {value} -> {float_value:.6f}")
                    except:
                        logger.info(f"   {key}: {value} (não conversível)")
                        
        except Exception as e:
            logger.error(f"❌ Erro no debug de saldo: {e}", exc_info=True)

    async def _notify_successful_trade_execution(self, symbol: str, signal: TradingSignal, 
                                                contracts: float, usdt_size: float):
        """Notifica execução bem-sucedida"""
        try:
            action_icon = "📈" if signal.action == "BUY" else "📉"
            action_text = "COMPRA" if signal.action == "BUY" else "VENDA"
            
            message = (
                f"{action_icon} <b>TRADE EXECUTADO!</b> {action_icon}\n\n"
                f"🎯 Ativo: <code>{symbol}</code>\n"
                f"📊 Ação: <b>{action_text}</b>\n"
                f"💰 Valor: <b>{usdt_size:.2f} USDT</b>\n"
                f"📈 Quantidade: <code>{contracts:.6f}</code>\n"
                f"🎯 Confiança: <b>{signal.confidence:.1f}%</b>\n\n"
                f"🎯 <b>SL/TP MELHORADO ATIVO!</b>\n"
                f"🛡️ Stop Loss: <b>{self.sltp_manager.config.stop_loss_pct:+.1f}%</b>\n"
                f"🏆 Take Profit: <b>{self.sltp_manager.config.take_profit_pct:+.1f}%</b>\n"
                f"📈 Trailing Stop: <b>{'Ativado' if self.sltp_manager.config.trailing_stop_enabled else 'Desativado'}</b>\n\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            )
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"❌ Erro notificando sucesso: {e}", exc_info=True)

    async def adaptive_symbol_discovery_individual(self) -> List[str]:
        """Descoberta de símbolos MELHORADA com filtro de volume 24h e detecção de crescimento rápido."""
        logger.info(f"🔍 Descoberta de símbolos com filtro de volume 24h e detecção de volume spike")
        
        # Carregar configurações
        min_volume_usdt_for_selection = getattr(Config, 'MIN_VOLUME_USDT_FOR_SELECTION', 10_000_000)
        enable_volume_spike_detection = getattr(Config, 'ENABLE_VOLUME_SPIKE_DETECTION', True)
        volume_spike_multiplier = getattr(Config, 'VOLUME_SPIKE_MULTIPLIER', 2.0)
        volume_spike_lookback_days = getattr(Config, 'VOLUME_SPIKE_LOOKBACK_DAYS', 7)
        min_volume_for_spike_candidate = getattr(Config, 'MIN_VOLUME_FOR_SPIKE_CANDIDATE', 1_000_000)
        manual_fallback_symbols = getattr(Config, 'MANUAL_SYMBOLS', ['BTC_USDT', 'ETH_USDT'])
        min_active_symbols = getattr(Config, 'MIN_ACTIVE_SYMBOLS', 3)
        max_active_symbols = getattr(Config, 'MAX_ACTIVE_SYMBOLS', 10)

        logger.info(f"Filtro de volume mínimo: {min_volume_usdt_for_selection:,.0f} USDT")
        if enable_volume_spike_detection:
            logger.info(f"Detecção de volume spike ATIVA (Multiplicador: {volume_spike_multiplier}x sobre {volume_spike_lookback_days} dias)")
            logger.info(f"Volume mínimo para candidato a spike: {min_volume_for_spike_candidate:,.0f} USDT")

        try:
            logger.info("📡 Verificando instrumentos disponíveis e tickers de 24h...")
            
            instruments_info = await self.gate_api.get_instruments_info()
            available_symbols_map = {
                inst['name']: inst for inst in instruments_info 
                if isinstance(inst, dict) and 'name' in inst and "_USDT" in inst['name']
            }
            
            all_tickers = await self.gate_api.get_futures_tickers()
            tickers_map = {
                ticker['contract']: ticker for ticker in all_tickers 
                if isinstance(ticker, dict) and 'contract' in ticker and 'vol_usdt_24h' in ticker
            }

            candidate_symbols: List[Tuple[str, float]] = []

            # Filtrar por volume mínimo e coletar candidatos
            for symbol_name, inst_data in available_symbols_map.items():
                if symbol_name in tickers_map:
                    try:
                        current_volume_24h = float(tickers_map[symbol_name]['vol_usdt_24h'])
                        
                        # Primeiro filtro: volume mínimo absoluto
                        if current_volume_24h >= min_volume_usdt_for_selection:
                            candidate_symbols.append((symbol_name, current_volume_24h))
                            logger.debug(f"Candidato (volume normal): {symbol_name} - {current_volume_24h:,.0f} USDT")
                        elif enable_volume_spike_detection and current_volume_24h >= min_volume_for_spike_candidate:
                            candidate_symbols.append((symbol_name, current_volume_24h))
                            logger.debug(f"Candidato (spike potencial): {symbol_name} - {current_volume_24h:,.0f} USDT")

                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ {symbol_name}: Não foi possível obter ou converter volume 24h. Pulando.", exc_info=True)
            
            if not candidate_symbols:
                logger.warning("⚠️ Nenhuns símbolos candidatos encontrados com os filtros iniciais. Usando fallback.")
                self.data_collector.active_symbols = manual_fallback_symbols
                return manual_fallback_symbols

            # Priorizar símbolos com volume spike
            eligible_symbols_final = []
            spike_detected_symbols = []

            if enable_volume_spike_detection:
                logger.info(f"Analisando {len(candidate_symbols)} símbolos para detecção de volume spike...")
                
                historical_volume_tasks = []
                for symbol, _ in candidate_symbols:
                    historical_volume_tasks.append(self.data_collector.get_daily_volume_history(symbol, volume_spike_lookback_days))
                
                historical_volumes_results = await asyncio.gather(*historical_volume_tasks, return_exceptions=True)

                for i, (symbol, current_volume) in enumerate(candidate_symbols):
                    historical_volumes_df = historical_volumes_results[i]
                    if isinstance(historical_volumes_df, pd.DataFrame) and not historical_volumes_df.empty:
                        avg_historical_volume = historical_volumes_df['quote_asset_volume'].mean() 

                        if avg_historical_volume > 0:
                            if current_volume >= (avg_historical_volume * volume_spike_multiplier):
                                spike_detected_symbols.append(symbol)
                                logger.info(f"🔥 VOLUME SPIKE DETECTADO para {symbol}: Atual {current_volume:,.0f} USDT (Média Histórica {avg_historical_volume:,.0f} USDT, Multiplicador {volume_spike_multiplier}x)")
                                if symbol not in eligible_symbols_final:
                                    eligible_symbols_final.append(symbol)
                            else:
                                logger.debug(f"Volume de {symbol} não é spike: Atual {current_volume:,.0f} vs Média {avg_historical_volume:,.0f}")
                        else:
                            logger.warning(f"⚠️ {symbol}: Média de volume histórico é zero. Não foi possível verificar spike.", exc_info=True)
                    else:
                        logger.warning(f"⚠️ {symbol}: Falha ao obter histórico de volume ou dados insuficientes.", exc_info=True)
            
            # Adicionar símbolos de prioridade se ainda não estiverem na lista de spikes e atendem ao volume mínimo
            for symbol in Config.MANUAL_SYMBOLS:
                if symbol not in eligible_symbols_final and symbol in tickers_map:
                    try:
                        current_volume = float(tickers_map[symbol]['vol_usdt_24h'])
                        if current_volume >= min_volume_usdt_for_selection:
                            if symbol not in eligible_symbols_final:
                                eligible_symbols_final.append(symbol)
                                logger.debug(f"Prioridade {symbol} adicionado (volume normal alto).")
                    except (ValueError, TypeError):
                        pass
            
            # Adicionar outros símbolos de alto volume que não são spikes nem prioridade, até o limite máximo
            all_sorted_by_volume = sorted(
                [(s, v) for s, v in tickers_map.items() if s in available_symbols_map and float(v['vol_usdt_24h']) >= min_volume_usdt_for_selection],
                key=lambda x: float(x[1]['vol_usdt_24h']), reverse=True
            )

            for symbol_data, _ in all_sorted_by_volume:
                symbol = symbol_data
                if len(eligible_symbols_final) >= max_active_symbols:
                    break
                if symbol not in eligible_symbols_final:
                    eligible_symbols_final.append(symbol)
                    logger.debug(f"Adicionado {symbol} por alto volume.")
            
            if eligible_symbols_final:
                self.data_collector.active_symbols = eligible_symbols_final
                self.stats['symbols_discovered'] = len(eligible_symbols_final)
                logger.info(f"🎯 Símbolos selecionados ({len(eligible_symbols_final)}): {eligible_symbols_final}")
                return eligible_symbols_final
            else:
                logger.error(f"❌ Nenhum símbolo válido encontrado com os critérios! Usando MANUAL_SYMBOLS como fallback.")
                self.data_collector.active_symbols = manual_fallback_symbols
                return manual_fallback_symbols
                
        except Exception as e:
            logger.error(f"❌ Erro na descoberta de símbolos com volume ou detecção de spike. Usando MANUAL_SYMBOLS como fallback: {e}", exc_info=True)
            self.data_collector.active_symbols = manual_fallback_symbols
            return manual_fallback_symbols

    async def analyze_symbols_individually(self, symbols: List[str]):
        """Análise individual dos símbolos - VERSÃO SIMPLIFICADA"""
        logger.info(f"🔬 Análise individual simplificada de {len(symbols)} símbolos")
        
        analyzed_count = 0
        
        for symbol in symbols:
            if not symbol or symbol == "UNKNOWN":
                continue
                
            try:
                klines_data = self.data_collector.get_klines_data(symbol)
                current_price = self.data_collector.get_current_price(symbol)
                
                if klines_data is not None and current_price and current_price > 0:
                    # Análise simplificada sem depender do SymbolAnalyzer
                    characteristics = await self.create_simple_symbol_characteristics(symbol, klines_data, current_price)
                    if characteristics:
                        # Armazenar no symbol_analyzer para compatibilidade
                        if not hasattr(self.symbol_analyzer, 'symbol_profiles'):
                            self.symbol_analyzer.symbol_profiles = {}
                        self.symbol_analyzer.symbol_profiles[symbol] = characteristics
                        analyzed_count += 1
                        self.stats['individual_analyses'] += 1
                        logger.debug(f"✅ {symbol}: Análise simplificada concluída")
                else:
                    logger.warning(f"⚠️ {symbol}: Dados insuficientes")
                    
            except Exception as e:
                logger.error(f"❌ Erro analisando {symbol}: {e}")
        
        logger.info(f"📊 Análise simplificada concluída: {analyzed_count} perfis criados")

    async def create_simple_symbol_characteristics(self, symbol: str, klines_data, current_price: float) -> Dict:
        """Cria características básicas do símbolo sem depender do SymbolAnalyzer"""
        try:
            if not isinstance(klines_data, pd.DataFrame) or len(klines_data) < 20:
                return None
            
            # Calcular estatísticas básicas
            close_prices = klines_data['close'].astype(float)
            
            # Volatilidade (desvio padrão dos últimos 24 períodos)
            returns = close_prices.pct_change().dropna()
            volatility = returns.tail(24).std() if len(returns) >= 24 else returns.std()
            
            # Tendência (slope dos últimos 20 períodos)
            recent_prices = close_prices.tail(20)
            if len(recent_prices) >= 2:
                x = range(len(recent_prices))
                slope = np.polyfit(x, recent_prices, 1)[0]
                trend = "UPTREND" if slope > 0 else "DOWNTREND" if slope < 0 else "SIDEWAYS"
            else:
                trend = "UNKNOWN"
            
            # Volume médio (se disponível)
            avg_volume = 0.0
            if 'volume' in klines_data.columns:
                avg_volume = float(klines_data['volume'].tail(24).mean())
            
            # Criar objeto de características simplificado
            characteristics = {
                'symbol': symbol,
                'volatility': float(volatility) if not pd.isna(volatility) else 0.0,
                'trend': trend,
                'avg_volume_24h': avg_volume,
                'current_price': current_price,
                'data_quality': min(len(klines_data) / 100, 1.0),  # Qualidade baseada na qtd de dados
                'last_updated': time.time(),
                'analysis_type': 'simplified'
            }
            
            logger.debug(f"📊 {symbol}: Vol={volatility:.6f}, Trend={trend}, Price={current_price:.4f}")
            return characteristics
            
        except Exception as e:
            logger.error(f"❌ Erro criando características para {symbol}: {e}")
            return None

    async def _simplified_intelligent_symbol_filtering(self, opportunities: List[Tuple[str, TradingSignal]]) -> List[Tuple[str, TradingSignal]]:
        """Filtragem inteligente de oportunidades com gestão de risco MELHORADA"""
        
        if not opportunities:
            return []
        
        logger.info(f"🚀 Filtragem de {len(opportunities)} oportunidades")
        filtered = []
        
        # Verificar risco geral da carteira primeiro
        try:
            # Usar uma implementação simplificada de cálculo de risco
            current_risk = await self.calculate_simplified_portfolio_risk()
            
            # **CORREÇÃO**: Usar configurações do config.py em vez de valores hardcoded
            max_risk_limit = getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 30.0)  # Valor do config.py (30%)
            # Fallback secundário para MAX_PORTFOLIO_RISK_PERCENT se MAX_TOTAL_RISK_PERCENT não existir
            if max_risk_limit == 30.0 and hasattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT'):
                max_risk_limit = getattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT', 10.0)
            
            logger.info(f"📊 GESTÃO DE RISCO: Risco atual {current_risk:.2f}% | Limite config.py: {max_risk_limit:.1f}%")
            
            # Se risco está muito alto, só permitir trades de alta confiança
            high_risk_threshold = max_risk_limit * 0.8  # 80% do limite
            if current_risk >= high_risk_threshold:
                logger.warning(f"⚠️ RISCO ELEVADO ({current_risk:.2f}%) - Só aceitando trades de ALTA confiança (≥80%)")
                high_risk_mode = True
                min_confidence_high_risk = 80.0
            else:
                high_risk_mode = False
                min_confidence_high_risk = 0.0
                
        except Exception as e:
            logger.error(f"❌ Erro calculando risco da carteira: {e}")
            high_risk_mode = False
            current_risk = 0.0
            min_confidence_high_risk = 0.0
        
        for symbol, signal in opportunities:
            logger.info(f"\n🎯 Analisando: {symbol}")
            logger.info(f"   📊 Sinal: {signal.action} | Confiança: {signal.confidence:.1f}%")
            logger.debug(f"   🔍 Risco atual: {current_risk:.2f}% | Limite: {max_risk_limit:.1f}%")
            logger.debug(f"   🔍 Modo alto risco: {high_risk_mode} | Min conf. alto risco: {min_confidence_high_risk:.1f}%")
            
            # Validações básicas
            if not symbol or symbol == "UNKNOWN" or symbol.strip() == "" or "_USDT" not in symbol:
                logger.warning(f"   ❌ Símbolo inválido: '{symbol}'")
                continue
            
            valid_symbols = self.symbol_mapping_manager.active_symbols
            if not valid_symbols or symbol not in valid_symbols:
                logger.warning(f"   ❌ Símbolo não está na lista de ativos")
                continue
            
            # Threshold de confiança ADAPTATIVO baseado no risco
            base_min_confidence = getattr(Config, 'MIN_CONFIDENCE_TO_TRADE', 40.0)
            
            if high_risk_mode:
                min_confidence = max(base_min_confidence, min_confidence_high_risk)
                logger.info(f"   ⚠️ MODO RISCO ALTO: Confiança mínima ajustada para {min_confidence:.1f}%")
            else:
                min_confidence = base_min_confidence
            
            if signal.confidence < min_confidence:
                logger.warning(f"   ❌ Confiança insuficiente ({signal.confidence:.1f}% < {min_confidence:.1f}%)")
                continue
            
            # Verificar preço
            current_price = self.data_collector.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logger.warning(f"   ❌ Sem preço válido")
                continue
            
            # Verificar saldo
            try:
                balance = await self.gate_api.get_futures_balance()
                available_usdt = balance.get('available', 0.0)
                if available_usdt <= 0:
                    available_usdt = balance.get('free', 0.0)

                min_balance_usdt_for_trade = getattr(Config, 'TARGET_USDT_FOR_NEW_TRADE', 10.0)

                if available_usdt < min_balance_usdt_for_trade:
                    logger.warning(f"   💰 Saldo insuficiente ({available_usdt:.2f} USDT) para um novo trade (Mínimo: {min_balance_usdt_for_trade:.2f})")
                    continue
                
            except Exception as e:
                logger.warning(f"   ⚠️ Erro verificando saldo: {e}", exc_info=True)
            
            # Verificar limite de posições com GESTÃO INTELIGENTE
            try:
                # Usar verificação personalizada em vez do portfolio manager
                can_open = await self.can_open_new_position_intelligent(current_risk, max_risk_limit)
                
                if not can_open:
                    # Se não pode abrir posição, verificar se é por risco muito alto
                    # Para sinais de MUITO alta confiança (≥90%), permitir 1 exceção
                    if signal.confidence >= 90.0 and current_risk <= max_risk_limit * 1.2:  # 20% acima do limite
                        logger.info(f"   🎯 EXCEÇÃO ALTA CONFIANÇA: Permitindo trade de {signal.confidence:.1f}% confiança apesar do risco")
                    else:
                        logger.warning(f"   ❌ Limite de posições/risco atingido. Risco: {current_risk:.2f}%. Não abrindo nova posição.")
                        continue 
            except Exception as e:
                logger.warning(f"   ⚠️ Erro verificando limite: {e}", exc_info=True)
            
            # Aprovado!
            filtered.append((symbol, signal))
            logger.info(f"   🎉 APROVADO! {symbol} (Confiança: {signal.confidence:.1f}%)")
            
        logger.info(f"🚀 Resultado filtragem: {len(filtered)}/{len(opportunities)} aprovados") 
        
        if filtered:
            approved_symbols = [symbol for symbol, _ in filtered]
            logger.info(f"✅ Símbolos aprovados: {approved_symbols}")
        else:
            logger.warning(f"⚠️ NENHUMA OPORTUNIDADE APROVADA - Risco atual: {current_risk:.2f}%")
            logger.info(f"💡 CONFIGURAÇÕES ATIVAS (config.py):")
            logger.info(f"   • MAX_TOTAL_RISK_PERCENT: {getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 'NÃO DEFINIDO')}")
            logger.info(f"   • MAX_PORTFOLIO_RISK_PERCENT: {getattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT', 'NÃO DEFINIDO')}")
            logger.info(f"   • Limite usado: {max_risk_limit:.1f}%")
            logger.info(f"   • ULTRA_SAFE_MODE: {getattr(Config, 'ULTRA_SAFE_MODE', False)}")
            logger.info(f"💡 DICAS PARA RESOLVER:")
            logger.info(f"   1. Aguardar redução do risco atual")
            logger.info(f"   2. Ajustar limite de risco no config.py")
            logger.info(f"   3. Fechar posições manualmente se necessário")
        
        return filtered

    async def initialize_telegram_bot_corrected(self):
        """Inicialização do bot Telegram"""
        try:
            if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
                logger.warning("⚠️ Credenciais Telegram não configuradas no config.py. Bot Telegram DESABILITADO.")
                self.telegram_bot_manager.application = None
                return

            if self.telegram_bot_manager.application:
                logger.info("🤖 Inicializando bot Telegram...")
                await self.telegram_bot_manager.initialize_handlers()
                if not self.telegram_bot_manager.application.running:
                    await self.telegram_bot_manager.application.initialize()
                    await self.telegram_bot_manager.application.start()
                    logger.info("✅ Bot Telegram inicializado")
                else:
                    logger.info("✅ Bot Telegram já está rodando. Ignorando inicialização.")
            else:
                logger.warning("⚠️ Aplicação Telegram não disponível (talvez devido a credenciais).")
                
        except Exception as e:
            logger.error(f"❌ Erro inicializando Telegram Bot: {e}", exc_info=True)

    async def update_market_conditions_simple(self):
        """Atualização das condições de mercado - VERSÃO SIMPLIFICADA"""
        try:
            all_klines = self.data_collector.get_all_klines_data()
            all_prices = self.data_collector.get_all_current_prices()
            
            if not all_klines or not all_prices:
                self.current_market_conditions.data_quality = 0.0
                self.current_market_conditions.risk_level = "EXTREME"
                return
            
            total_symbols = len(all_klines)
            valid_symbols = sum(1 for df in all_klines.values() if isinstance(df, pd.DataFrame) and len(df) >= 20)
            
            self.current_market_conditions.data_quality = valid_symbols / max(total_symbols, 1)
            self.current_market_conditions.active_symbols_count = valid_symbols
            self.current_market_conditions.timestamp = time.time()
            
            # Calcular volatilidade média de forma simplificada
            total_volatility = 0.0
            num_vol_symbols = 0
            
            # Usar os perfis de símbolos se existirem
            if hasattr(self.symbol_analyzer, 'symbol_profiles') and self.symbol_analyzer.symbol_profiles:
                for symbol_profile in self.symbol_analyzer.symbol_profiles.values():
                    if isinstance(symbol_profile, dict) and 'volatility' in symbol_profile:
                        vol = symbol_profile['volatility']
                        if vol > 0:
                            total_volatility += vol
                            num_vol_symbols += 1
            else:
                # Fallback: calcular volatilidade diretamente dos klines
                for symbol, klines in all_klines.items():
                    try:
                        if isinstance(klines, pd.DataFrame) and len(klines) >= 20:
                            close_prices = klines['close'].astype(float)
                            returns = close_prices.pct_change().dropna()
                            if len(returns) >= 10:
                                vol = returns.tail(20).std()
                                if vol > 0 and not pd.isna(vol):
                                    total_volatility += vol
                                    num_vol_symbols += 1
                    except Exception as e:
                        logger.debug(f"⚠️ Erro calculando volatilidade para {symbol}: {e}")
                        continue
            
            avg_volatility = (total_volatility / num_vol_symbols) if num_vol_symbols > 0 else 0.0
            self.current_market_conditions.overall_volatility = avg_volatility

            # Determinar nível de risco baseado na volatilidade
            if avg_volatility > 0.008:  # Alta volatilidade
                self.current_market_conditions.risk_level = "HIGH"
            elif avg_volatility > 0.004:  # Volatilidade média
                self.current_market_conditions.risk_level = "MEDIUM"
            else:  # Baixa volatilidade
                self.current_market_conditions.risk_level = "LOW"
            
            logger.debug(f"📊 Condições atualizadas: qualidade={self.current_market_conditions.data_quality:.1%}, "
                        f"risco={self.current_market_conditions.risk_level}, vol={avg_volatility:.6f}, "
                        f"símbolos={num_vol_symbols}")
            
        except Exception as e:
            logger.error(f"❌ Erro atualizando condições de mercado: {e}", exc_info=True)
            # Definir valores padrão em caso de erro
            self.current_market_conditions.data_quality = 0.5
            self.current_market_conditions.risk_level = "MEDIUM"
            self.current_market_conditions.overall_volatility = 0.0

    async def execute_intelligent_trades_individual(self, opportunities: List[Tuple[str, TradingSignal]]) -> int:
        """Execução de trades individuais - CORRIGINDO PORTFOLIO_MANAGER"""
        
        if not opportunities:
            return 0
        
        executed_count = 0
        logger.info(f"🚀 Executando {len(opportunities)} trades")
        
        # **CORREÇÃO CRÍTICA**: Verificar e ajustar configurações do portfolio_manager
        await self.fix_portfolio_manager_config()
        
        for symbol, signal in opportunities:
            if not symbol or symbol == "UNKNOWN":
                continue
            
            try:
                logger.info(f"📤 Executando trade: {symbol}")
                
                # Verificar saldo
                balance = await self.gate_api.get_futures_balance()
                available_usdt = balance.get('available', 0.0)
                if available_usdt <= 0:
                    available_usdt = balance.get('free', 0.0)

                min_balance_for_trade = getattr(Config, 'TARGET_USDT_FOR_NEW_TRADE', 10.0)
                if available_usdt < min_balance_for_trade:
                    logger.warning(f"   💰 Saldo insuficiente: {available_usdt:.2f}. Mínimo {min_balance_for_trade:.2f}. Pulando trade.")
                    continue
                
                # Obter a alavancagem máxima configurada
                max_leverage_config = getattr(Config, 'MAX_LEVERAGE', 1.0) 
                
                # Calcular a margem para este trade
                margin_allocation_percent = getattr(Config, 'MAX_MARGIN_ALLOCATION_PER_TRADE_PERCENT', 2.0)
                margin_to_allocate_usdt = available_usdt * (margin_allocation_percent / 100.0)

                # Ajustar a margem pela confiança do sinal
                confidence_factor = signal.confidence / 100.0
                adjusted_margin_usdt = margin_to_allocate_usdt * confidence_factor
                
                # Definir limites de margem
                min_trade_margin_usdt = 5.0
                max_trade_margin_usdt = min(available_usdt * 0.15, 100.0)
                
                final_margin_usdt = max(min_trade_margin_usdt, min(adjusted_margin_usdt, max_trade_margin_usdt))
                
                if final_margin_usdt < min_trade_margin_usdt:
                    logger.warning(f"   💰 Margem calculada ({final_margin_usdt:.2f} USDT) abaixo do mínimo ({min_trade_margin_usdt:.2f}). Pulando trade.")
                    continue

                # Calcular o valor nominal da posição
                nominal_position_value_usdt = final_margin_usdt * max_leverage_config
                
                # Calcular contratos com base no valor nominal e preço.
                contracts = nominal_position_value_usdt / signal.price
                
                # Ajuste de precisão para Gate.io
                contracts = round(contracts, 8)

                # Verificar se o número de contratos é razoável
                if contracts < 0.000001:
                    logger.warning(f"   📏 Posição muito pequena: {contracts:.8f} contratos. Pulando trade.")
                    continue
                
                logger.info(f"   📊 Margem Alocada: {final_margin_usdt:.2f} USDT | Valor Nominal da Posição: {nominal_position_value_usdt:.2f} USDT | Contratos: {contracts:.6f} @ {max_leverage_config}x alavancagem")
                
                # **VERIFICAÇÃO FINAL**: Usar verificação própria em vez do portfolio_manager
                can_open = await self.can_open_position_bypass_pm(symbol)
                if not can_open:
                    logger.warning(f"   ❌ Verificação final: Não pode abrir posição para {symbol}")
                    continue
                
                # Executar ordem
                side = "buy" if signal.action == "BUY" else "sell"
                
                # TENTAR definir a alavancagem para o símbolo antes de colocar a ordem.
                try:
                    set_leverage_success = await self.gate_api.set_leverage(symbol, int(max_leverage_config))
                    if not set_leverage_success:
                        logger.warning(f"   ⚠️ Falha ao definir alavancagem {int(max_leverage_config)}x para {symbol}. Usando padrão da conta/exchange.")
                except AttributeError:
                    logger.warning(f"   ❌ Erro: Método gate_api.set_leverage não existe. Prosseguindo com o padrão da conta/exchange.")
                except Exception as e:
                    logger.warning(f"   ❌ Erro ao tentar definir alavancagem para {symbol}: {e}. Prosseguindo com o padrão.", exc_info=True)

                result = await self.gate_api.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="market",
                    size=contracts,
                    reduce_only=False
                )
                
                if result and result.get('success', False):
                    executed_count += 1
                    self.stats['orders_successful'] += 1
                    
                    logger.info(f"   ✅ TRADE EXECUTADO: {symbol} {signal.action}")
                    
                    # Notificar sucesso (agora com info de SL/TP)
                    await self._notify_successful_trade_execution(symbol, signal, contracts, nominal_position_value_usdt)
                    
                else:
                    error_msg = result.get('message', 'Erro desconhecido') if result else 'Sem resposta'
                    logger.error(f"   ❌ Falha ao executar trade para {symbol}: {error_msg}")
                
            except Exception as e:
                logger.error(f"❌ Erro executando trade {symbol}: {e}", exc_info=True)
        
        logger.info(f"🏁 Execução concluída: {executed_count}/{len(opportunities)} sucessos")
        return executed_count

    async def fix_portfolio_manager_config(self):
        """CORREÇÃO CRÍTICA: Ajustar configurações do portfolio_manager para usar config.py"""
        try:
            logger.info("🔧 Corrigindo configurações do Portfolio Manager...")
            
            # **CORREÇÃO**: Forçar o portfolio_manager a usar valores do config.py
            max_total_risk_config = getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 30.0)
            max_positions_config = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 5)
            
            # Ajustar diretamente as propriedades do portfolio_manager
            if hasattr(self.portfolio_manager, 'max_total_risk_percent'):
                old_value = self.portfolio_manager.max_total_risk_percent
                self.portfolio_manager.max_total_risk_percent = max_total_risk_config
                logger.info(f"🔧 Portfolio Manager max_total_risk_percent: {old_value:.1f}% → {max_total_risk_config:.1f}%")
            
            if hasattr(self.portfolio_manager, 'max_portfolio_positions'):
                old_positions = self.portfolio_manager.max_portfolio_positions
                self.portfolio_manager.max_portfolio_positions = max_positions_config
                logger.info(f"🔧 Portfolio Manager max_portfolio_positions: {old_positions} → {max_positions_config}")
            
            logger.info("✅ Configurações do Portfolio Manager corrigidas!")
            
        except Exception as e:
            logger.error(f"❌ Erro corrigindo Portfolio Manager: {e}", exc_info=True)

    async def can_open_position_bypass_pm(self, symbol: str) -> bool:
        """Verificação direta de posição sem depender do portfolio_manager bugado"""
        try:
            # Calcular risco atual usando nosso método
            current_risk = await self.calculate_simplified_portfolio_risk()
            max_risk_limit = getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 30.0)
            
            if current_risk >= max_risk_limit:
                logger.warning(f"   🚫 Risco atual ({current_risk:.2f}%) >= limite config.py ({max_risk_limit:.1f}%)")
                return False
            
            # Verificar número de posições
            await self.portfolio_manager.update_account_info()
            positions_count = len(self.portfolio_manager.open_positions)
            max_positions = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 5)
            
            if positions_count >= max_positions:
                logger.warning(f"   🚫 Número de posições ({positions_count}) >= limite config.py ({max_positions})")
                return False
            
            # Verificar saldo
            balance = await self.gate_api.get_futures_balance()
            available_usdt = float(balance.get('available', 0.0))
            if available_usdt <= 0:
                available_usdt = float(balance.get('free', 0.0))
            
            min_balance = getattr(Config, 'MIN_BALANCE_FOR_NEW_TRADE', 20.0)
            if available_usdt < min_balance:
                logger.warning(f"   🚫 Saldo disponível ({available_usdt:.2f}) < mínimo config.py ({min_balance:.2f})")
                return False
            
            logger.info(f"   ✅ PODE ABRIR {symbol}: Risco={current_risk:.2f}%/{max_risk_limit:.1f}%, Posições={positions_count}/{max_positions}, Saldo={available_usdt:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro verificando se pode abrir {symbol}: {e}", exc_info=True)
            return False

    def _calculate_simple_position_size(self, available_usdt: float, signal: TradingSignal) -> float:
        """Cálculo simplificado de tamanho de posição"""
        risk_percent = getattr(Config, 'MAX_RISK_PER_TRADE_PERCENT', 3.0)
        base_size = available_usdt * (risk_percent / 100.0)
        confidence_factor = signal.confidence / 100.0
        adjusted_size = base_size * confidence_factor
        max_size = min(available_usdt * 0.05, 25.0)
        min_size = 5.0
        final_size = max(min_size, min(adjusted_size, max_size))
        return final_size

    async def optimize_portfolio_intelligently(self, opportunities: List[Tuple[str, Any]]) -> List[PortfolioAction]:
        """Otimização de portfolio"""
        
        if not opportunities:
            logger.debug("📊 Sem oportunidades para otimização")
            return []
        
        try:
            logger.info("🧠 Otimização de portfolio")
            actions = await self.portfolio_optimizer.optimize_portfolio(opportunities) 
            
            if actions:
                logger.info(f"⚡ {len(actions)} ações recomendadas")
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Erro na otimização: {e}", exc_info=True)
            return []

    async def execute_portfolio_optimizations(self, actions: List[PortfolioAction]) -> int:
        """Executa otimizações"""
        executed_count = 0
        if not actions:
            return 0
        
        logger.info(f"⚡ Executando {len(actions)} ações de otimização de portfólio...")
        for action in actions:
            try:
                if action.action_type == ActionType.CLOSE_POSITION:
                    logger.info(f"🔄 Otimização: Fechando {action.symbol} - Motivo: {action.reason}")
                    success = await self.portfolio_manager.close_single_position(action.symbol, action.reason)
                    if success:
                        executed_count += 1
                        logger.info(f"✅ Otimização: {action.symbol} fechado com sucesso.")
                elif action.action_type == ActionType.ADJUST_LEVERAGE:
                    logger.info(f"🛠️ Otimização: Ajustando alavancagem para {action.symbol} para {action.target_value}x")
                    try:
                        leverage_set = await self.gate_api.set_leverage(action.symbol, int(action.target_value))
                        if leverage_set:
                            executed_count += 1
                            logger.info(f"✅ Otimização: Alavancagem para {action.symbol} ajustada para {action.target_value}x.")
                        else:
                            logger.warning(f"⚠️ Otimização: Falha ao ajustar alavancagem para {action.symbol}.")
                    except AttributeError:
                        logger.warning(f"⚠️ Otimização: Método gate_api.set_leverage não implementado. Não foi possível ajustar alavancagem para {action.symbol}.")
                    except Exception as e:
                        logger.error(f"❌ Erro na otimização ao ajustar alavancagem para {action.symbol}: {e}", exc_info=True)

                else:
                    logger.warning(f"Ação de otimização desconhecida: {action.action_type}")
            except Exception as e:
                logger.error(f"❌ Erro executando ação de otimização para {action.symbol}: {e}", exc_info=True)
        
        logger.info(f"⚡ {executed_count}/{len(actions)} ações de otimização executadas com sucesso.")
        return executed_count

    async def get_remaining_opportunities(self, original_opportunities: List[Tuple[str, Any]], 
                                        executed_actions: List[PortfolioAction]) -> List[Tuple[str, Any]]:
        """Retorna oportunidades restantes"""
        return original_opportunities 

    # =====================================================================
    # MÉTODOS AUXILIARES DE GESTÃO DE RISCO
    # =====================================================================

    async def calculate_simplified_portfolio_risk(self) -> float:
        """Cálculo simplificado de risco do portfólio - USANDO DADOS REAIS DO PORTFOLIO MANAGER"""
        try:
            # Obter saldo total da conta via portfolio manager (já atualizado)
            await self.portfolio_manager.update_account_info()
            
            balance = await self.gate_api.get_futures_balance()
            total_equity = float(balance.get('equity', 0.0))
            
            if total_equity <= 0:
                logger.warning("⚠️ Equity total é zero, retornando risco 0%")
                return 0.0
            
            # **CORREÇÃO**: Usar o método do portfolio manager que já tem a lógica implementada
            try:
                portfolio_risk = await self.portfolio_manager.calculate_total_risk()
                logger.debug(f"📊 Risco calculado pelo Portfolio Manager: {portfolio_risk:.2f}%")
                return portfolio_risk
            except Exception as pm_error:
                logger.warning(f"⚠️ Falha no cálculo de risco do PM: {pm_error}. Usando cálculo simplificado.")
            
            # Fallback: cálculo manual se o portfolio manager falhar
            positions = list(self.portfolio_manager.open_positions.values())
            
            if not positions:
                logger.debug("📊 Nenhuma posição aberta, risco = 0%")
                return 0.0
            
            total_position_value = 0.0
            total_margin_used = 0.0
            
            for position in positions:
                try:
                    # Valor da posição (size * mark_price)
                    size = float(position.get('size', 0))
                    if size == 0:
                        continue
                        
                    symbol = position.get('contract', '')
                    if not symbol or symbol == 'UNKNOWN':
                        continue
                    
                    # Usar mark_price da posição (mais confiável que current_price)
                    mark_price = float(position.get('mark_price', 0))
                    if mark_price <= 0:
                        # Fallback para entry_price
                        mark_price = float(position.get('entry_price', 0))
                        if mark_price <= 0:
                            continue
                    
                    position_value = abs(size) * mark_price
                    margin_used = float(position.get('margin', 0))
                    
                    total_position_value += position_value
                    total_margin_used += margin_used
                    
                    logger.debug(f"📊 {symbol}: Size={size:.6f}, MarkPrice={mark_price:.4f}, PosValue={position_value:.2f}, Margin={margin_used:.2f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro processando posição {position}: {e}")
                    continue
            
            # Cálculo de risco: usar margem utilizada como proxy do risco exposto
            # Para futuros, a margem é uma representação mais precisa do capital em risco
            risk_percentage = (total_margin_used / total_equity) * 100.0
            
            logger.debug(f"📊 CÁLCULO RISCO MANUAL: Equity={total_equity:.2f}, MargemUsada={total_margin_used:.2f}, ValorPosições={total_position_value:.2f}, Risk={risk_percentage:.2f}%")
            
            return min(risk_percentage, 100.0)  # Máximo 100%
            
        except Exception as e:
            logger.error(f"❌ Erro no cálculo simplificado de risco: {e}", exc_info=True)
            # Em caso de erro, retornar valor conservador
            return 50.0  # Assume risco médio-alto para ser conservador

    async def can_open_new_position_intelligent(self, current_risk: float, max_risk_limit: float) -> bool:
        """Verificação inteligente se pode abrir nova posição - USA CONFIG.PY"""
        try:
            # Verificar se o risco atual permite nova posição
            if current_risk >= max_risk_limit:
                logger.debug(f"🚫 Risco atual ({current_risk:.2f}%) >= limite config.py ({max_risk_limit:.1f}%)")
                return False
            
            # Verificar número máximo de posições - USAR CONFIG.PY
            await self.portfolio_manager.update_account_info()
            positions_count = len(self.portfolio_manager.open_positions)
            
            # **CORREÇÃO**: Pegar do config.py em vez de usar portfolio_manager
            max_positions = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 5)
            # Fallback para MAX_OPEN_POSITIONS se MAX_CONCURRENT_POSITIONS não existir
            if max_positions == 5 and hasattr(Config, 'MAX_OPEN_POSITIONS'):
                max_positions = getattr(Config, 'MAX_OPEN_POSITIONS', 5)
            
            if positions_count >= max_positions:
                logger.debug(f"🚫 Número de posições ({positions_count}) >= limite config.py ({max_positions})")
                return False
            
            # Verificar saldo mínimo - USAR CONFIG.PY
            balance = await self.gate_api.get_futures_balance()
            available_usdt = float(balance.get('available', 0.0))
            if available_usdt <= 0:
                available_usdt = float(balance.get('free', 0.0))
            
            # **CORREÇÃO**: Pegar do config.py
            min_balance = getattr(Config, 'MIN_BALANCE_FOR_NEW_TRADE', 20.0)
            # Fallback para outras configurações se não existir
            if min_balance == 20.0 and hasattr(Config, 'MIN_BALANCE_USDT_TO_OPERATE'):
                min_balance = getattr(Config, 'MIN_BALANCE_USDT_TO_OPERATE', 50.0)
            
            if available_usdt < min_balance:
                logger.debug(f"🚫 Saldo disponível ({available_usdt:.2f}) < mínimo config.py ({min_balance:.2f})")
                return False
            
            logger.debug(f"✅ Pode abrir nova posição: Risco={current_risk:.2f}%/{max_risk_limit:.1f}%, Posições={positions_count}/{max_positions}, Saldo={available_usdt:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro verificando se pode abrir posição: {e}", exc_info=True)
            return False 

    # =====================================================================
    # INICIALIZAÇÃO E LOOP PRINCIPAL
    # =====================================================================

    async def initialize(self):
        """Inicialização com SL/TP MELHORADO usando config.py"""
        logger.info("🔄 Inicializando Bot com SL/TP MELHORADO (USANDO CONFIG.PY)...")
        
        # **DEBUG**: Verificar configurações carregadas ANTES de tudo
        logger.info("🔍 VERIFICAÇÃO INICIAL - Config.py carregado:")
        logger.info(f"   MAX_TOTAL_RISK_PERCENT = {getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 'NÃO ENCONTRADO')}")
        logger.info(f"   MAX_CONCURRENT_POSITIONS = {getattr(Config, 'MAX_CONCURRENT_POSITIONS', 'NÃO ENCONTRADO')}")
        logger.info(f"   ULTRA_SAFE_MODE = {getattr(Config, 'ULTRA_SAFE_MODE', 'NÃO ENCONTRADO')}")
        
        try:
            # APIs e WebSocket
            await self.gate_api.start_websockets()
            await self.portfolio_manager.initialize_account_info()
            
            # **CORREÇÃO CRÍTICA**: Ajustar portfolio_manager logo após inicialização
            await self.fix_portfolio_manager_config()
            
            # Bot Telegram
            await self.initialize_telegram_bot_corrected()
            
            # Descoberta inicial
            symbols = await self.adaptive_symbol_discovery_individual()
            
            # Dados iniciais
            if symbols:
                try:
                    klines_data = await self.data_collector.fetch_klines_parallel(
                        symbols, 
                        limit=getattr(Config, 'KLINES_LIMIT', 200)
                    )
                    self.data_collector.klines_data = klines_data
                    await self.data_collector.update_current_prices_batch(symbols)
                    await self.analyze_symbols_individually(symbols)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro nos dados iniciais: {e}", exc_info=True)
            
            # Gestão inicial com SL/TP melhorado
            logger.info("🎯 Executando gestão inicial com SL/TP MELHORADO...")
            await self.sltp_manager.monitor_all_positions_with_sltp()
            
            # Notificação de inicialização
            await self._send_startup_notification_with_sltp(symbols)
            
            logger.info("✅ Bot com SL/TP MELHORADO inicializado (USANDO CONFIG.PY)!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}", exc_info=True)
            return False

    async def _send_startup_notification_with_sltp(self, symbols: List[str]):
        """Notificação de inicialização com info do SL/TP"""
        try:
            balance = await self.gate_api.get_futures_balance()
            equity = balance.get('equity', 0.0) if balance else 0.0
            
            positions = await self.portfolio_manager.get_open_positions_ws()
            positions_count = len(positions) if positions else 0
            
            symbols_str = ', '.join(symbols[:5])
            if len(symbols) > 5:
                symbols_str += f" +{len(symbols)-5} outros"
            
            # Calcular risco atual para mostrar na notificação
            current_risk = 0.0
            try:
                current_risk = await self.calculate_simplified_portfolio_risk()
            except Exception as e:
                logger.warning(f"⚠️ Erro calculando risco para notificação: {e}")
                current_risk = 0.0

            # **CORREÇÃO**: Usar valores do config.py
            max_risk_limit = getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 30.0)
            if max_risk_limit == 30.0 and hasattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT'):
                max_risk_limit = getattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT', 10.0)
            
            ultra_safe_mode = getattr(Config, 'ULTRA_SAFE_MODE', False)
            
            message = (
                f"🎯 <b>BOT COM SL/TP MELHORADO INICIADO</b> 🎯\n"
                f"<b>✅ VERSÃO CORRIGIDA - SÍMBOLOS UNKNOWN RESOLVIDO</b>\n"
                f"<b>🔧 PORTFOLIO MANAGER CORRIGIDO PARA CONFIG.PY</b>\n\n"
                f"💰 Saldo: <b>{equity:.2f} USDT</b>\n"
                f"📊 Símbolos: <b>{len(symbols)}</b>\n"
                f"💼 Posições: <b>{positions_count}</b>\n"
                f"🎯 Ativos: <code>{symbols_str}</code>\n\n"
                f"🛡️ <b>Sistema SL/TP MELHORADO:</b>\n"
                f"• Stop Loss: <b>{self.sltp_manager.config.stop_loss_pct:+.1f}%</b>\n"
                f"• Take Profit: <b>{self.sltp_manager.config.take_profit_pct:+.1f}%</b>\n"
                f"• Trailing Stop: <b>{'✅' if self.sltp_manager.config.trailing_stop_enabled else '❌'}</b>\n"
                f"• Breakeven: <b>{'✅' if self.sltp_manager.config.breakeven_enabled else '❌'}</b>\n"
                f"• SL Emergencial: <b>{self.sltp_manager.config.emergency_sl_pct:+.1f}%</b>\n"
                f"• TP Emergencial: <b>{self.sltp_manager.config.emergency_tp_pct:+.1f}%</b>\n\n"
                f"📊 <b>Gestão de Risco (CORRIGIDA):</b>\n"
                f"• Risco atual: <b>{current_risk:.2f}%</b>\n"
                f"• Limite máximo: <b>{max_risk_limit:.1f}%</b> ({'ULTRA_SAFE' if ultra_safe_mode else 'CONFIG.PY'})\n"
                f"• Portfolio Manager: <b>CORRIGIDO</b> ✅\n"
                f"• Modo adaptativo: <b>{'Alto Risco' if current_risk >= max_risk_limit * 0.8 else 'Normal'}</b>\n"
                f"• Exceções alta confiança: <b>≥90%</b>\n\n"
                f"🚀 <b>Sistema 100% operacional!</b>\n"
                f"🔧 <b>CORREÇÕES APLICADAS:</b>\n"
                f"• ✅ Problema símbolos UNKNOWN resolvido\n"
                f"• ✅ Gestão de risco usando config.py (limite: {max_risk_limit:.1f}%)\n"
                f"• ✅ Portfolio Manager corrigido para usar config.py\n"
                f"• ✅ SymbolAnalyzer simplificado implementado\n"
                f"• ✅ Bypass de verificações bugadas implementado"
            )
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"❌ Erro na notificação de inicialização: {e}", exc_info=True)

    async def run_forever_individual(self):
        """Loop principal com SL/TP MELHORADO"""
        logger.info("🚀 Iniciando loop com SL/TP MELHORADO (USANDO CONFIG.PY)")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        try:
            while self.is_running:
                # Verificar horários
                current_utc_hour = datetime.now(timezone.utc).hour
                trading_hours = getattr(Config, 'TRADING_HOURS', {'start_hour': 0, 'end_hour': 23, 'weekend_trading': True})
                
                is_trading_time = (trading_hours['start_hour'] <= current_utc_hour <= trading_hours['end_hour'])
                is_weekend = datetime.now(timezone.utc).weekday() >= 5
                
                if not is_trading_time or (is_weekend and not trading_hours['weekend_trading']):
                    logger.info(f"😴 Fora do horário de operação: {current_utc_hour}h UTC. Fim de semana: {is_weekend}. Próximo ciclo em 5 minutos.")
                    await asyncio.sleep(300)
                    continue
                
                # Ciclo principal
                cycle_interval = getattr(Config, 'CYCLE_INTERVAL_SECONDS', 60)
                try:
                    success = await asyncio.wait_for(self.run_trading_cycle_individual(), timeout=cycle_interval * 2)
                    if success:
                        self.consecutive_failures = 0
                    else:
                        self.consecutive_failures += 1
                        
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout no ciclo de trading após {cycle_interval * 2} segundos! Falhas consecutivas: {self.consecutive_failures + 1}")
                    self.consecutive_failures += 1
                    
                except Exception as e:
                    logger.error(f"❌ Erro inesperado no ciclo de trading: {e}", exc_info=True)
                    self.consecutive_failures += 1
                
                # Verificar falhas consecutivas para shutdown de emergência
                if self.consecutive_failures >= getattr(Config, 'MAX_CONSECUTIVE_FAILURES', 3):
                    logger.critical(f"💥 {self.consecutive_failures} falhas consecutivas atingiram o limite de {getattr(Config, 'MAX_CONSECUTIVE_FAILURES', 3)}! Ativando shutdown de emergência.")
                    self.is_running = False
                    break
                
                # Log de status com SL/TP
                await self.log_individual_status()
                
                # Relatório SL/TP ocasional
                if self.cycle_count % 10 == 0:
                    await self.send_sltp_status_report()
                
                # Intervalo entre ciclos
                logger.info(f"⏳ Próximo ciclo em {cycle_interval}s")
                await asyncio.sleep(cycle_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrompido pelo usuário (Ctrl+C). Iniciando shutdown limpo.")
            self.is_running = False
        except Exception as e:
            logger.critical(f"❌ Erro fatal e inesperado no loop principal: {e}", exc_info=True)
            self.is_running = False
        finally:
            pass

    async def send_sltp_status_report(self):
        """Enviar relatório de status do SL/TP"""
        try:
            sltp_stats = await self.sltp_manager.get_sltp_statistics()
            
            if sltp_stats.get('total_closures', 0) == 0 and sltp_stats.get('active_trailing_stops', 0) == 0 and sltp_stats.get('active_breakeven', 0) == 0:
                logger.debug("ℹ️ Sem dados relevantes de SL/TP para enviar relatório.")
                return
            
            trailing_symbols = sltp_stats.get('trailing_symbols', [])
            breakeven_symbols = sltp_stats.get('breakeven_symbols', [])
            
            message = (
                f"📊 <b>RELATÓRIO SL/TP</b> (Ciclo {self.cycle_count})\n\n"
                f"🎯 <b>Estatísticas:</b>\n"
                f"• Total fechamentos: <b>{sltp_stats.get('total_closures', 0)}</b>\n"
                f"• Taxa de sucesso: <b>{sltp_stats.get('win_rate_pct', 0):.1f}%</b>\n"
                f"• PnL total: <b>{sltp_stats.get('total_pnl_usd', 0):+.2f} USDT</b>\n"
                f"• Stop Losses: <b>{sltp_stats.get('stop_losses', 0)}</b>\n"
                f"• Take Profits: <b>{sltp_stats.get('take_profits', 0)}</b>\n"
                f"• Trailing Stops: <b>{sltp_stats.get('trailing_stops', 0)}</b>\n\n"
                f"🎯 <b>Ativos:</b>\n"
                f"• Trailing ativo: <b>{len(trailing_symbols)}</b>"
            )
            
            if trailing_symbols:
                message += f" ({', '.join(trailing_symbols[:3])}{'...' if len(trailing_symbols) > 3 else ''})"
            
            message += f"\n• Breakeven ativo: <b>{len(breakeven_symbols)}</b>"
            
            if breakeven_symbols:
                message += f" ({', '.join(breakeven_symbols[:3])}{'...' if len(breakeven_symbols) > 3 else ''})"
            
            message += f"\n\n⏰ <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"❌ Erro enviando relatório SL/TP: {e}", exc_info=True)

    async def shutdown(self):
        """Shutdown com estatísticas finais SL/TP"""
        logger.info("🔄 Shutdown do sistema...")
        
        try:
            self.is_running = False
            
            # Enviar estatísticas finais
            sltp_stats = await self.sltp_manager.get_sltp_statistics()
            logger.info(f"📊 Estatísticas finais SL/TP: {sltp_stats}")
            
            # Parar Telegram Bot
            if self.telegram_bot_manager.application:
                try:
                    if self.telegram_bot_manager.application.running:
                        await self.telegram_bot_manager.application.stop()
                        await self.telegram_bot_manager.application.shutdown()
                except Exception as e:
                    logger.warning(f"⚠️ Erro parando Telegram Bot: {e}", exc_info=True)
            
            await self.gate_api.close()
            logger.info("✅ Shutdown concluído")
            
        except Exception as e:
            logger.error(f"❌ Erro no shutdown: {e}", exc_info=True)

    async def debug_all_configurations(self):
        """Debug completo de todas as configurações para identificar problemas"""
        logger.info("🔍 === DEBUG COMPLETO DAS CONFIGURAÇÕES ===")
        
        try:
            # 1. Configurações do Config.py
            logger.info("📋 CONFIGURAÇÕES DO CONFIG.PY:")
            logger.info(f"   MAX_TOTAL_RISK_PERCENT: {getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 'NÃO EXISTE')}")
            logger.info(f"   MAX_PORTFOLIO_RISK_PERCENT: {getattr(Config, 'MAX_PORTFOLIO_RISK_PERCENT', 'NÃO EXISTE')}")
            logger.info(f"   MAX_CONCURRENT_POSITIONS: {getattr(Config, 'MAX_CONCURRENT_POSITIONS', 'NÃO EXISTE')}")
            logger.info(f"   ULTRA_SAFE_MODE: {getattr(Config, 'ULTRA_SAFE_MODE', 'NÃO EXISTE')}")
            
            # 2. Configurações do Portfolio Manager
            logger.info("📊 CONFIGURAÇÕES DO PORTFOLIO MANAGER:")
            if hasattr(self.portfolio_manager, 'max_total_risk_percent'):
                logger.info(f"   portfolio_manager.max_total_risk_percent: {self.portfolio_manager.max_total_risk_percent}")
            else:
                logger.info(f"   portfolio_manager.max_total_risk_percent: NÃO EXISTE")
                
            if hasattr(self.portfolio_manager, 'max_portfolio_positions'):
                logger.info(f"   portfolio_manager.max_portfolio_positions: {self.portfolio_manager.max_portfolio_positions}")
            else:
                logger.info(f"   portfolio_manager.max_portfolio_positions: NÃO EXISTE")
            
            # 3. Risco atual
            current_risk = await self.calculate_simplified_portfolio_risk()
            logger.info(f"📈 RISCO ATUAL CALCULADO: {current_risk:.2f}%")
            
            # 4. Saldo disponível
            balance = await self.gate_api.get_futures_balance()
            logger.info(f"💰 SALDO ATUAL: {balance}")
            
            # 5. Posições abertas
            positions = await self.portfolio_manager.get_open_positions_ws()
            logger.info(f"💼 POSIÇÕES ABERTAS: {len(positions)}")
            for pos in positions:
                logger.info(f"   - {pos.get('contract', 'UNKNOWN')}: {pos.get('side', 'UNKNOWN')} {pos.get('size', 0)}")
            
            logger.info("🔍 === FIM DEBUG CONFIGURAÇÕES ===")
            
        except Exception as e:
            logger.error(f"❌ Erro no debug de configurações: {e}", exc_info=True)

    async def send_sltp_status_report(self):
        """Enviar relatório de status do SL/TP"""
        try:
            sltp_stats = await self.sltp_manager.get_sltp_statistics()
            
            if sltp_stats.get('total_closures', 0) == 0 and sltp_stats.get('active_trailing_stops', 0) == 0 and sltp_stats.get('active_breakeven', 0) == 0:
                logger.debug("ℹ️ Sem dados relevantes de SL/TP para enviar relatório.")
                return
            
            trailing_symbols = sltp_stats.get('trailing_symbols', [])
            breakeven_symbols = sltp_stats.get('breakeven_symbols', [])
            
            message = (
                f"📊 <b>RELATÓRIO SL/TP</b> (Ciclo {self.cycle_count})\n\n"
                f"🎯 <b>Estatísticas:</b>\n"
                f"• Total fechamentos: <b>{sltp_stats.get('total_closures', 0)}</b>\n"
                f"• Taxa de sucesso: <b>{sltp_stats.get('win_rate_pct', 0):.1f}%</b>\n"
                f"• PnL total: <b>{sltp_stats.get('total_pnl_usd', 0):+.2f} USDT</b>\n"
                f"• Stop Losses: <b>{sltp_stats.get('stop_losses', 0)}</b>\n"
                f"• Take Profits: <b>{sltp_stats.get('take_profits', 0)}</b>\n"
                f"• Trailing Stops: <b>{sltp_stats.get('trailing_stops', 0)}</b>\n\n"
                f"🎯 <b>Ativos:</b>\n"
                f"• Trailing ativo: <b>{len(trailing_symbols)}</b>"
            )
            
            if trailing_symbols:
                message += f" ({', '.join(trailing_symbols[:3])}{'...' if len(trailing_symbols) > 3 else ''})"
            
            message += f"\n• Breakeven ativo: <b>{len(breakeven_symbols)}</b>"
            
            if breakeven_symbols:
                message += f" ({', '.join(breakeven_symbols[:3])}{'...' if len(breakeven_symbols) > 3 else ''})"
            
            message += f"\n\n⏰ <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            
            await self.telegram_notifier.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"❌ Erro enviando relatório SL/TP: {e}", exc_info=True)
        """Shutdown com estatísticas finais SL/TP"""
        logger.info("🔄 Shutdown do sistema...")
        
        try:
            self.is_running = False
            
            # Enviar estatísticas finais
            sltp_stats = await self.sltp_manager.get_sltp_statistics()
            logger.info(f"📊 Estatísticas finais SL/TP: {sltp_stats}")
            
            # Parar Telegram Bot
            if self.telegram_bot_manager.application:
                try:
                    if self.telegram_bot_manager.application.running:
                        await self.telegram_bot_manager.application.stop()
                        await self.telegram_bot_manager.application.shutdown()
                except Exception as e:
                    logger.warning(f"⚠️ Erro parando Telegram Bot: {e}", exc_info=True)
            
            await self.gate_api.close()
            logger.info("✅ Shutdown concluído")
            
        except Exception as e:
            logger.error(f"❌ Erro no shutdown: {e}", exc_info=True)

# =====================================================================
# FUNÇÃO PRINCIPAL
# =====================================================================

async def main():
    """Função principal com SL/TP MELHORADO"""
    bot = None
    
    # Handler para sinais de interrupção (Ctrl+C)
    def signal_handler(signum, frame):
        logger.info(f"📡 Sinal de interrupção recebido (Signum: {signum}). Iniciando encerramento...")
        if bot:
            bot.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        bot = IntelligentBotGateIndividual()
        
        if not await bot.initialize():
            logger.critical("❌ Falha crítica na inicialização do bot. Encerrando o programa.")
            sys.exit(1)
        
        # Inicia o loop principal do bot
        await bot.run_forever_individual()
        
    except Exception as e:
        logger.critical(f"❌ Erro fatal e inesperado na função principal do programa: {e}", exc_info=True)
    finally:
        if bot:
            await bot.shutdown()
        
    logger.info("🏁 Bot com SL/TP MELHORADO finalizado (USANDO CONFIG.PY).")

if __name__ == "__main__":
    asyncio.run(main())