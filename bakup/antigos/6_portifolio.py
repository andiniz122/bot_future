#!/usr/bin/env python3
"""
Sistema de Portfolio e Risk Management
Gerencia posições, risco, P&L e execução de trades com segurança máxima
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import uuid
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('portfolio_system')

# =====================================================================
# ENUMS E CONSTANTES
# =====================================================================

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELED = "canceled"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

# =====================================================================
# CLASSES DE DADOS
# =====================================================================

@dataclass
class RiskConfig:
    """Configuração de risk management"""
    # Limites gerais
    max_portfolio_risk: float = 0.02  # 2% do portfolio por trade
    max_daily_loss: float = 0.05      # 5% perda máxima diária
    max_drawdown: float = 0.15        # 15% drawdown máximo
    
    # Limites de posição
    max_position_size: float = 0.1    # 10% do portfolio por posição
    max_positions_per_symbol: int = 1
    max_total_positions: int = 10
    
    # Limites de correlação
    max_correlation_exposure: float = 0.3  # 30% em ativos correlacionados
    correlation_threshold: float = 0.7
    
    # Stop loss e take profit
    default_stop_loss: float = 0.02   # 2%
    default_take_profit: float = 0.04 # 4%
    trailing_stop_activation: float = 0.015  # 1.5%
    trailing_stop_distance: float = 0.01     # 1%
    
    # Gestão de exposição
    max_leverage: float = 10.0
    margin_buffer: float = 0.2        # 20% buffer de margem
    
    # Timeouts
    position_timeout_hours: int = 24  # Fechar posições após 24h
    order_timeout_minutes: int = 5    # Cancelar ordens após 5min

@dataclass
class Position:
    """Posição individual"""
    id: str
    symbol: str
    side: PositionSide
    size: float                    # Tamanho da posição
    entry_price: float
    current_price: float = 0.0
    
    # Ordens de proteção
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    
    # Metadata
    entry_time: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    confidence: float = 0.0
    
    # Status
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk management
    max_loss_amount: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    def calculate_pnl(self) -> float:
        """Calcula P&L não realizado"""
        if self.current_price <= 0:
            return 0.0
            
        if self.side == PositionSide.LONG:
            pnl_per_unit = self.current_price - self.entry_price
        else:
            pnl_per_unit = self.entry_price - self.current_price
            
        self.unrealized_pnl = pnl_per_unit * self.size
        return self.unrealized_pnl
    
    def calculate_pnl_percentage(self) -> float:
        """Calcula P&L em percentual"""
        if self.entry_price <= 0:
            return 0.0
        return (self.calculate_pnl() / (self.entry_price * self.size)) * 100
    
    def should_stop_loss(self) -> bool:
        """Verifica se deve executar stop loss"""
        if not self.stop_loss_price:
            return False
            
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss_price
        else:
            return self.current_price >= self.stop_loss_price
    
    def should_take_profit(self) -> bool:
        """Verifica se deve executar take profit"""
        if not self.take_profit_price:
            return False
            
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit_price
        else:
            return self.current_price <= self.take_profit_price
    
    def update_trailing_stop(self, current_price: float):
        """Atualiza trailing stop"""
        if not self.trailing_stop_price:
            return
            
        self.current_price = current_price
        
        if self.side == PositionSide.LONG:
            # Para long, trailing stop sobe com o preço
            new_trailing = current_price * (1 - self.trailing_stop_distance)
            if new_trailing > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing
        else:
            # Para short, trailing stop desce com o preço
            new_trailing = current_price * (1 + self.trailing_stop_distance)
            if new_trailing < self.trailing_stop_price:
                self.trailing_stop_price = new_trailing
    
    def should_trailing_stop(self) -> bool:
        """Verifica se deve executar trailing stop"""
        if not self.trailing_stop_price:
            return False
            
        if self.side == PositionSide.LONG:
            return self.current_price <= self.trailing_stop_price
        else:
            return self.current_price >= self.trailing_stop_price
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Verifica se posição expirou"""
        return datetime.now() - self.entry_time > timedelta(hours=timeout_hours)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'trailing_stop_price': self.trailing_stop_price,
            'entry_time': self.entry_time.isoformat(),
            'strategy_id': self.strategy_id,
            'confidence': self.confidence,
            'status': self.status.value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'max_loss_amount': self.max_loss_amount,
            'risk_level': self.risk_level.value
        }

@dataclass
class PortfolioStats:
    """Estatísticas do portfolio"""
    total_balance: float = 0.0
    available_balance: float = 0.0
    used_margin: float = 0.0
    free_margin: float = 0.0
    
    # P&L
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Posições
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    
    # Risk metrics
    portfolio_risk: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Win rate
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Performance
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Trade:
    """Trade finalizado"""
    id: str
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percentage: float
    strategy_id: Optional[str] = None
    exit_reason: str = "manual"
    commission: float = 0.0

# =====================================================================
# SISTEMA DE RISK MANAGEMENT
# =====================================================================

class RiskManager:
    """Gerenciador de risco avançado"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_pnl_history = []
        self.drawdown_history = []
        
    def calculate_position_size(self, balance: float, risk_amount: float, 
                              entry_price: float, stop_loss_price: float) -> float:
        """Calcula tamanho da posição baseado no risco"""
        
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0.0
        
        # Calcular risco por unidade
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return 0.0
        
        # Tamanho baseado no risco
        position_size = risk_amount / risk_per_unit
        
        # Aplicar limites
        max_size_by_balance = balance * self.config.max_position_size / entry_price
        max_size_by_risk = balance * self.config.max_portfolio_risk / risk_per_unit
        
        return min(position_size, max_size_by_balance, max_size_by_risk)
    
    def calculate_stop_loss(self, entry_price: float, side: PositionSide, 
                          custom_stop: Optional[float] = None) -> float:
        """Calcula preço de stop loss"""
        
        stop_distance = custom_stop or self.config.default_stop_loss
        
        if side == PositionSide.LONG:
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    def calculate_take_profit(self, entry_price: float, side: PositionSide,
                            custom_tp: Optional[float] = None) -> float:
        """Calcula preço de take profit"""
        
        tp_distance = custom_tp or self.config.default_take_profit
        
        if side == PositionSide.LONG:
            return entry_price * (1 + tp_distance)
        else:
            return entry_price * (1 - tp_distance)
    
    def check_daily_loss_limit(self, daily_pnl: float, balance: float) -> bool:
        """Verifica se atingiu limite de perda diária"""
        
        daily_loss_limit = balance * self.config.max_daily_loss
        return daily_pnl <= -daily_loss_limit
    
    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """Verifica se atingiu limite de drawdown"""
        
        return current_drawdown >= self.config.max_drawdown
    
    def check_position_limits(self, positions: Dict[str, Position], 
                            new_symbol: str) -> Dict[str, Any]:
        """Verifica limites de posições"""
        
        # Contar posições atuais
        total_positions = len(positions)
        symbol_positions = sum(1 for p in positions.values() if p.symbol == new_symbol)
        
        # Verificar limites
        can_open = True
        reasons = []
        
        if total_positions >= self.config.max_total_positions:
            can_open = False
            reasons.append(f"Limite total de posições ({self.config.max_total_positions})")
        
        if symbol_positions >= self.config.max_positions_per_symbol:
            can_open = False
            reasons.append(f"Limite de posições por símbolo ({self.config.max_positions_per_symbol})")
        
        return {
            'can_open': can_open,
            'reasons': reasons,
            'total_positions': total_positions,
            'symbol_positions': symbol_positions
        }
    
    def calculate_correlation_risk(self, positions: Dict[str, Position], 
                                 new_symbol: str, correlation_matrix: pd.DataFrame) -> float:
        """Calcula risco de correlação"""
        
        if correlation_matrix.empty or new_symbol not in correlation_matrix.columns:
            return 0.0
        
        total_correlation_exposure = 0.0
        
        for position in positions.values():
            if position.symbol in correlation_matrix.columns:
                correlation = correlation_matrix.loc[new_symbol, position.symbol]
                
                if abs(correlation) >= self.config.correlation_threshold:
                    # Peso da posição existente
                    position_weight = abs(position.size * position.current_price)
                    total_correlation_exposure += position_weight * abs(correlation)
        
        return total_correlation_exposure
    
    def get_risk_level(self, pnl_percentage: float, confidence: float) -> RiskLevel:
        """Determina nível de risco da posição"""
        
        risk_score = 0
        
        # Baseado no P&L esperado
        if abs(pnl_percentage) > 10:
            risk_score += 2
        elif abs(pnl_percentage) > 5:
            risk_score += 1
        
        # Baseado na confiança
        if confidence < 0.5:
            risk_score += 2
        elif confidence < 0.7:
            risk_score += 1
        
        # Determinar nível
        if risk_score <= 1:
            return RiskLevel.LOW
        elif risk_score <= 2:
            return RiskLevel.MEDIUM
        elif risk_score <= 3:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME

# =====================================================================
# SISTEMA DE PORTFOLIO
# =====================================================================

class PortfolioManager:
    """Gerenciador principal do portfolio"""
    
    def __init__(self, config: RiskConfig, initial_balance: float = 10000.0):
        self.config = config
        self.risk_manager = RiskManager(config)
        
        # Estado do portfolio
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        
        # Estatísticas
        self.stats = PortfolioStats()
        self.daily_pnl_history = []
        self.balance_history = []
        
        # Database
        self.db_path = Path("portfolio.db")
        self._init_database()
        
        # Cache de preços
        self.price_cache = {}
        self.last_price_update = {}
        
    def _init_database(self):
        """Inicializa database do portfolio"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabela de posições
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    trailing_stop_price REAL,
                    entry_time TEXT NOT NULL,
                    strategy_id TEXT,
                    confidence REAL,
                    status TEXT NOT NULL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    max_loss_amount REAL,
                    risk_level TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            
            # Tabela de trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    strategy_id TEXT,
                    exit_reason TEXT,
                    commission REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            
            # Tabela de balanço histórico
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    balance REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_positions INTEGER NOT NULL,
                    daily_pnl REAL NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database do portfolio inicializada")
            
        except Exception as e:
            logger.error(f"❌ Erro inicializando database: {e}")
    
    async def update_prices(self, price_data: Dict[str, float]):
        """Atualiza preços das posições"""
        
        current_time = datetime.now()
        
        for symbol, price in price_data.items():
            self.price_cache[symbol] = price
            self.last_price_update[symbol] = current_time
            
            # Atualizar posições do símbolo
            for position in self.positions.values():
                if position.symbol == symbol:
                    position.current_price = price
                    position.calculate_pnl()
                    position.update_trailing_stop(price)
        
        # Atualizar estatísticas
        self._update_portfolio_stats()
    
    async def can_open_new_position_overall(self) -> bool:
        """Verifica se pode abrir novas posições (verificação geral)"""
        
        # Verificar limite de perda diária
        if self.risk_manager.check_daily_loss_limit(self.stats.daily_pnl, self.current_balance):
            logger.warning("❌ Limite de perda diária atingido")
            return False
        
        # Verificar drawdown
        if self.risk_manager.check_drawdown_limit(self.stats.current_drawdown):
            logger.warning("❌ Limite de drawdown atingido")
            return False
        
        # Verificar balance mínimo
        if self.current_balance <= self.initial_balance * 0.5:  # 50% do balance inicial
            logger.warning("❌ Balance muito baixo")
            return False
        
        return True
    
    async def can_open_position(self, symbol: str, side: PositionSide, 
                              price: float, risk_amount: float) -> Dict[str, Any]:
        """Verifica se pode abrir uma posição específica"""
        
        # Verificação geral primeiro
        if not await self.can_open_new_position_overall():
            return {
                'can_open': False,
                'reason': 'Limites gerais de risco atingidos',
                'suggested_size': 0
            }
        
        # Verificar limites de posições
        position_check = self.risk_manager.check_position_limits(self.positions, symbol)
        if not position_check['can_open']:
            return {
                'can_open': False,
                'reason': ', '.join(position_check['reasons']),
                'suggested_size': 0
            }
        
        # Calcular tamanho sugerido
        stop_loss_price = self.risk_manager.calculate_stop_loss(price, side)
        suggested_size = self.risk_manager.calculate_position_size(
            self.current_balance, risk_amount, price, stop_loss_price
        )
        
        if suggested_size <= 0:
            return {
                'can_open': False,
                'reason': 'Tamanho de posição calculado é zero',
                'suggested_size': 0
            }
        
        return {
            'can_open': True,
            'reason': 'Posição aprovada',
            'suggested_size': suggested_size,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': self.risk_manager.calculate_take_profit(price, side)
        }
    
    async def open_position(self, symbol: str, side: PositionSide, size: float,
                          price: float, strategy_id: str = None, 
                          confidence: float = 0.5) -> Optional[Position]:
        """Abre uma nova posição"""
        
        # Verificar se pode abrir
        risk_amount = self.current_balance * self.config.max_portfolio_risk
        check_result = await self.can_open_position(symbol, side, price, risk_amount)
        
        if not check_result['can_open']:
            logger.warning(f"❌ Não pode abrir posição {symbol}: {check_result['reason']}")
            return None
        
        # Usar tamanho sugerido se fornecido
        if size <= 0:
            size = check_result['suggested_size']
        
        # Criar posição
        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            current_price=price,
            strategy_id=strategy_id,
            confidence=confidence
        )
        
        # Definir stop loss e take profit
        position.stop_loss_price = self.risk_manager.calculate_stop_loss(price, side)
        position.take_profit_price = self.risk_manager.calculate_take_profit(price, side)
        
        # Configurar trailing stop se o movimento for favorável
        profit_threshold = price * self.config.trailing_stop_activation
        if side == PositionSide.LONG:
            if price >= position.entry_price + profit_threshold:
                position.trailing_stop_price = price * (1 - self.config.trailing_stop_distance)
        else:
            if price <= position.entry_price - profit_threshold:
                position.trailing_stop_price = price * (1 + self.config.trailing_stop_distance)
        
        # Calcular risco
        position.max_loss_amount = size * abs(price - position.stop_loss_price)
        position.risk_level = self.risk_manager.get_risk_level(0, confidence)
        
        # Adicionar ao portfolio
        self.positions[position.id] = position
        
        # Salvar no database
        await self._save_position_to_db(position)
        
        logger.info(f"✅ Posição aberta: {symbol} {side.value} {size:.4f} @ {price:.2f}")
        
        return position
    
    async def close_position(self, position_id: str, exit_price: float, 
                           exit_reason: str = "manual") -> Optional[Trade]:
        """Fecha uma posição"""
        
        if position_id not in self.positions:
            logger.warning(f"❌ Posição {position_id} não encontrada")
            return None
        
        position = self.positions[position_id]
        
        # Calcular P&L final
        position.current_price = exit_price
        final_pnl = position.calculate_pnl()
        pnl_percentage = position.calculate_pnl_percentage()
        
        # Criar trade
        trade = Trade(
            id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=final_pnl,
            pnl_percentage=pnl_percentage,
            strategy_id=position.strategy_id,
            exit_reason=exit_reason
        )
        
        # Atualizar balance
        self.current_balance += final_pnl
        
        # Atualizar posição
        position.status = PositionStatus.CLOSED
        position.realized_pnl = final_pnl
        position.unrealized_pnl = 0.0
        
        # Mover para trades fechados
        self.closed_trades.append(trade)
        
        # Remover das posições ativas
        del self.positions[position_id]
        
        # Salvar no database
        await self._save_trade_to_db(trade)
        await self._update_position_in_db(position)
        
        logger.info(f"✅ Posição fechada: {trade.symbol} P&L: {final_pnl:.2f} ({pnl_percentage:.2f}%)")
        
        return trade
    
    async def check_exit_conditions(self) -> List[Tuple[str, str]]:
        """Verifica condições de saída para todas as posições"""
        
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            exit_reason = None
            
            # Verificar stop loss
            if position.should_stop_loss():
                exit_reason = "stop_loss"
            
            # Verificar take profit
            elif position.should_take_profit():
                exit_reason = "take_profit"
            
            # Verificar trailing stop
            elif position.should_trailing_stop():
                exit_reason = "trailing_stop"
            
            # Verificar timeout
            elif position.is_expired(self.config.position_timeout_hours):
                exit_reason = "timeout"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason))
        
        return positions_to_close
    
    async def process_exits(self) -> List[Trade]:
        """Processa todas as saídas pendentes"""
        
        exits_to_process = await self.check_exit_conditions()
        trades_executed = []
        
        for position_id, exit_reason in exits_to_process:
            position = self.positions.get(position_id)
            if not position:
                continue
            
            # Usar preço atual ou preço de saída específico
            exit_price = position.current_price
            
            if exit_reason == "stop_loss" and position.stop_loss_price:
                exit_price = position.stop_loss_price
            elif exit_reason == "take_profit" and position.take_profit_price:
                exit_price = position.take_profit_price
            elif exit_reason == "trailing_stop" and position.trailing_stop_price:
                exit_price = position.trailing_stop_price
            
            # Executar saída
            trade = await self.close_position(position_id, exit_price, exit_reason)
            if trade:
                trades_executed.append(trade)
        
        return trades_executed
    
    def _update_portfolio_stats(self):
        """Atualiza estatísticas do portfolio"""
        
        # Calcular P&L total não realizado
        total_unrealized_pnl = sum(pos.calculate_pnl() for pos in self.positions.values())
        
        # Calcular P&L realizado total
        total_realized_pnl = sum(trade.pnl for trade in self.closed_trades)
        
        # Estatísticas de posições
        total_positions = len(self.positions)
        long_positions = sum(1 for pos in self.positions.values() if pos.side == PositionSide.LONG)
        short_positions = total_positions - long_positions
        
        # Estatísticas de trades
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calcular equity atual
        current_equity = self.current_balance + total_unrealized_pnl
        
        # Calcular drawdown
        if hasattr(self, 'peak_equity'):
            self.peak_equity = max(self.peak_equity, current_equity)
        else:
            self.peak_equity = current_equity
        
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # P&L diário (simplificado)
        daily_pnl = total_realized_pnl + total_unrealized_pnl
        
        # Atualizar stats
        self.stats = PortfolioStats(
            total_balance=self.current_balance,
            available_balance=self.current_balance - abs(total_unrealized_pnl),
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            daily_pnl=daily_pnl,
            total_positions=total_positions,
            long_positions=long_positions,
            short_positions=short_positions,
            current_drawdown=current_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            last_updated=datetime.now()
        )
    
    async def _save_position_to_db(self, position: Position):
        """Salva posição no database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (id, symbol, side, size, entry_price, current_price, stop_loss_price, 
                 take_profit_price, trailing_stop_price, entry_time, strategy_id, 
                 confidence, status, unrealized_pnl, realized_pnl, max_loss_amount, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.id, position.symbol, position.side.value, position.size,
                position.entry_price, position.current_price, position.stop_loss_price,
                position.take_profit_price, position.trailing_stop_price,
                position.entry_time.isoformat(), position.strategy_id, position.confidence,
                position.status.value, position.unrealized_pnl, position.realized_pnl,
                position.max_loss_amount, position.risk_level.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro salvando posição: {e}")
    
    async def _update_position_in_db(self, position: Position):
        """Atualiza posição no database"""
        await self._save_position_to_db(position)
    
    async def _save_trade_to_db(self, trade: Trade):
        """Salva trade no database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades 
                (id, symbol, side, size, entry_price, exit_price, entry_time, 
                 exit_time, pnl, pnl_percentage, strategy_id, exit_reason, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.id, trade.symbol, trade.side.value, trade.size,
                trade.entry_price, trade.exit_price, trade.entry_time.isoformat(),
                trade.exit_time.isoformat(), trade.pnl, trade.pnl_percentage,
                trade.strategy_id, trade.exit_reason, trade.commission
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro salvando trade: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retorna resumo completo do portfolio"""
        
        self._update_portfolio_stats()
        
        return {
            'balance': {
                'initial': self.initial_balance,
                'current': self.current_balance,
                'equity': self.current_balance + self.stats.total_unrealized_pnl,
                'available': self.stats.available_balance
            },
            'pnl': {
                'unrealized': self.stats.total_unrealized_pnl,
                'realized': self.stats.total_realized_pnl,
                'total': self.stats.total_unrealized_pnl + self.stats.total_realized_pnl,
                'daily': self.stats.daily_pnl
            },
            'positions': {
                'total': self.stats.total_positions,
                'long': self.stats.long_positions,
                'short': self.stats.short_positions,
                'details': [pos.to_dict() for pos in self.positions.values()]
            },
            'trading': {
                'total_trades': self.stats.total_trades,
                'winning_trades': self.stats.winning_trades,
                'losing_trades': self.stats.losing_trades,
                'win_rate': self.stats.win_rate
            },
            'risk': {
                'current_drawdown': self.stats.current_drawdown,
                'max_drawdown': self.stats.max_drawdown,
                'daily_loss_limit': self.config.max_daily_loss,
                'position_limit': self.config.max_total_positions
            }
        }
    
    async def execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Interface principal para executar trades (compatibilidade com outros módulos)"""
        
        try:
            action = signal.get('action', 'HOLD')
            price = signal.get('price', 0)
            confidence = signal.get('confidence', 0.5)
            strategy_id = signal.get('strategy_id', 'unknown')
            
            if action == 'HOLD' or price <= 0:
                return {'success': False, 'error': 'Sinal inválido'}
            
            # Determinar lado da posição
            side = PositionSide.LONG if action == 'BUY' else PositionSide.SHORT
            
            # Abrir posição
            position = await self.open_position(
                symbol=symbol,
                side=side,
                size=0,  # Será calculado automaticamente
                price=price,
                strategy_id=strategy_id,
                confidence=confidence
            )
            
            if position:
                return {
                    'success': True,
                    'position_id': position.id,
                    'symbol': symbol,
                    'side': side.value,
                    'size': position.size,
                    'price': price
                }
            else:
                return {'success': False, 'error': 'Não foi possível abrir posição'}
                
        except Exception as e:
            logger.error(f"Erro executando trade: {e}")
            return {'success': False, 'error': str(e)}

# =====================================================================
# EXEMPLO DE USO
# =====================================================================

async def example_usage():
    """Exemplo de uso do sistema de portfolio"""
    
    # Configuração de risco
    config = RiskConfig(
        max_portfolio_risk=0.02,    # 2% por trade
        max_daily_loss=0.05,        # 5% perda diária máxima
        max_total_positions=5,      # 5 posições máximas
        default_stop_loss=0.02,     # 2% stop loss
        default_take_profit=0.04    # 4% take profit
    )
    
    # Criar portfolio manager
    portfolio = PortfolioManager(config, initial_balance=10000.0)
    
    # Simular alguns trades
    print("=== SIMULAÇÃO DE PORTFOLIO ===")
    
    # Atualizar preços
    await portfolio.update_prices({
        'BTC_USDT': 45000.0,
        'ETH_USDT': 3000.0,
        'ADA_USDT': 1.5
    })
    
    # Abrir posições
    pos1 = await portfolio.open_position(
        symbol='BTC_USDT',
        side=PositionSide.LONG,
        size=0,  # Será calculado
        price=45000.0,
        strategy_id='momentum_strategy_1',
        confidence=0.75
    )
    
    pos2 = await portfolio.open_position(
        symbol='ETH_USDT',
        side=PositionSide.SHORT,
        size=0,
        price=3000.0,
        strategy_id='mean_reversion_1',
        confidence=0.6
    )
    
    # Simular mudanças de preço
    await portfolio.update_prices({
        'BTC_USDT': 46000.0,  # +2.22%
        'ETH_USDT': 2950.0,   # -1.67%
        'ADA_USDT': 1.5
    })
    
    # Verificar condições de saída
    exits = await portfolio.check_exit_conditions()
    print(f"Saídas pendentes: {exits}")
    
    # Processar saídas se houver
    if exits:
        trades = await portfolio.process_exits()
        print(f"Trades executados: {len(trades)}")
    
    # Mostrar resumo do portfolio
    summary = portfolio.get_portfolio_summary()
    
    print(f"\n=== RESUMO DO PORTFOLIO ===")
    print(f"Balance Inicial: ${summary['balance']['initial']:,.2f}")
    print(f"Balance Atual: ${summary['balance']['current']:,.2f}")
    print(f"Equity: ${summary['balance']['equity']:,.2f}")
    print(f"P&L Total: ${summary['pnl']['total']:,.2f}")
    print(f"P&L Não Realizado: ${summary['pnl']['unrealized']:,.2f}")
    print(f"Posições Ativas: {summary['positions']['total']}")
    print(f"Win Rate: {summary['trading']['win_rate']:.1f}%")
    print(f"Drawdown Atual: {summary['risk']['current_drawdown']:.2%}")

if __name__ == "__main__":
    asyncio.run(example_usage())