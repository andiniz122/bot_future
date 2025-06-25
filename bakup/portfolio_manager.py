# portfolio_manager.py - Versão FUTUROS Gate.io com WebSocket - CORRIGIDA
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union
from gate_api import GateAPI # Garanta que este import a API correta
import config_gate as Config # Importa o módulo como 'Config'
import config_ultra_safe as UltraSafeConfig # Importar config_ultra_safe para os novos parâmetros
from datetime import datetime, timedelta
from estrategia import TradingSignal, SignalStrength # Assegura que TradingSignal está disponível
from enum import Enum
import math

logger = logging.getLogger('portfolio_manager_gate')

class TradeReason(Enum):
    """Razões para trades"""
    STRATEGY_BUY_SIGNAL = "strategy_buy_signal"
    STRATEGY_SELL_SIGNAL = "strategy_sell_signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    RISK_MANAGEMENT = "risk_management"
    POSITION_CLOSE = "position_close"
    CLOSE_ORDER_SUBMITTED = "close_order_submitted" # Adicionado para rastrear ordens de fechamento

class GateFuturesPortfolioManager:
    def __init__(self, gate_api: GateAPI):
        self.gate_api = gate_api
        self.config = Config # Referencia o módulo Config diretamente
        
        # Estados da carteira de futuros Gate.io (atualizados via WebSocket)
        self.open_positions: Dict[str, Dict] = {}  # Posições ativas
        self.account_balance: Dict[str, Any] = {}  # Saldo de margem USDT
        self.usdt_balance: float = 0.0
        self.available_balance: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.current_prices: Dict[str, float] = {} # Usado para cálculos de PnL e size
        
        # Cache e controle
        self.initialized_account = False
        self.last_balance_update = 0
        self.last_positions_update = 0
        
        # Configurações de trading de futuros Gate.io (puxadas do Config.py ou UltraSafeConfig)
        self.autonomous_mode = getattr(Config, 'AUTONOMOUS_TRADING_ENABLED', True)
        
        # [NOVA LÓGICA] Definir limites com base em ULTRA_SAFE_MODE ou Config
        if UltraSafeConfig.ULTRA_SAFE_MODE:
            self.max_portfolio_positions = UltraSafeConfig.MAX_OPEN_POSITIONS #
            # Usando EMERGENCY_LOSS_PERCENT como max_total_risk e STOP_LOSS_PERCENT como max_single_position_risk
            self.max_total_risk_percent = abs(UltraSafeConfig.EMERGENCY_LOSS_PERCENT) # Convertendo para positivo
            self.max_single_position_risk_percent = abs(UltraSafeConfig.STOP_LOSS_PERCENT) # Convertendo para positivo
            self.max_leverage = UltraSafeConfig.MAX_LEVERAGE
            self.risk_per_trade = UltraSafeConfig.MAX_POSITION_SIZE_PERCENT / 100.0 # Tamanho da posição como risco por trade
            logger.info(f"🛡️ Portfolio Manager em modo ULTRA-SEGURO. Max Posições: {self.max_portfolio_positions}, Max Risco Total: {self.max_total_risk_percent}%, Max Risco por Posição: {self.max_single_position_risk_percent}%")
        else:
            self.max_portfolio_positions = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 3)
            self.max_leverage = getattr(Config, 'MAX_LEVERAGE', 30.0) # RISCO DA CARTEIRA
            self.risk_per_trade = getattr(Config, 'MAX_RISK_PER_TRADE_PERCENT', 1.5) / 100.0
            self.max_total_risk_percent = getattr(Config, 'MAX_TOTAL_RISK_PERCENT', 5.0)
            self.max_single_position_risk_percent = getattr(Config, 'MAX_RISK_PER_TRADE_PERCENT', 1.5)
            logger.info(f"💼 Portfolio Manager em modo padrão. Max Posições: {self.max_portfolio_positions}, Max Risco Total: {self.max_total_risk_percent}%, Max Risco por Posição: {self.max_single_position_risk_percent}%")

        # Histórico de trades
        self.trade_history: List[Dict] = []

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Converte valor para float de forma segura. Retorna default se falhar."""
        try:
            if value is None or value == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Erro convertendo '{value}' para float, usando {default}")
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Converte valor para int de forma segura. Retorna default se falhar."""
        try:
            if value is None or value == '':
                return default
            # Converte via float primeiro para lidar com strings decimais como '1.0'
            return int(float(value)) 
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Erro convertendo '{value}' para int, usando {default}")
            return default

    async def initialize_account_info(self):
        """Inicializa informações da conta de futuros Gate.io via WebSocket/REST."""
        logger.info("🔄 Inicializando conta de FUTUROS Gate.io via WebSocket/REST...")
        
        try:
            # Tentar obter saldos via API (que tentará WS ou REST)
            balances = await self.gate_api.get_futures_balance() #
            
            if balances:
                self.usdt_balance = self._safe_float(balances.get('equity'))
                self.available_balance = self._safe_float(balances.get('free'))
                self.unrealized_pnl = self._safe_float(balances.get('unrealPnl'))
                
                self.account_balance = balances # Guarda o dicionário completo
                self.last_balance_update = time.time()
                
                logger.info(f"💰 Saldo USDT inicial Gate.io: {self.usdt_balance:.2f}")
                logger.info(f"💵 Disponível: {self.available_balance:.2f}")
                logger.info(f"📈 PnL não realizado: {self.unrealized_pnl:.2f}")
                
                await self.sync_positions_with_websocket() # Sincroniza posições
                
                self.initialized_account = True
                logger.info("✅ Conta de FUTUROS Gate.io inicializada com sucesso.")
                
            else:
                logger.error("❌ Falha ao obter saldos da conta Gate.io. Saldo pode ser 0.")
                self.usdt_balance = 0.0
                self.available_balance = 0.0
                self.unrealized_pnl = 0.0
                
        except Exception as e:
            logger.error(f"❌ Erro inicializando conta de futuros Gate.io: {e}", exc_info=True)
            self.usdt_balance = 0.0
            self.available_balance = 0.0
            self.unrealized_pnl = 0.0

    async def update_account_info(self):
        """Atualiza informações da conta de futuros Gate.io via API (WS/REST)."""
        try:
            balances = await self.gate_api.get_futures_balance() #
            
            if balances:
                old_balance = self.usdt_balance
                old_available = self.available_balance
                old_pnl = self.unrealized_pnl
                
                self.usdt_balance = self._safe_float(balances.get('equity'))
                self.available_balance = self._safe_float(balances.get('free'))
                self.unrealized_pnl = self._safe_float(balances.get('unrealPnl'))
                
                if abs(self.usdt_balance - old_balance) > 1.0: # Loga se houver mudança significativa
                    logger.info(f"💰 Equity Gate.io: {old_balance:.2f} → {self.usdt_balance:.2f}")
                
                if abs(self.unrealized_pnl - old_pnl) > 0.5: # Loga se PnL mudar significativamente
                    logger.info(f"📈 PnL Gate.io: {old_pnl:.2f} → {self.unrealized_pnl:.2f}")
                    
                self.account_balance = balances
                self.last_balance_update = time.time()
                
                await self.sync_positions_with_websocket() # Atualiza posições
                
            else:
                logger.warning("⚠️ Nenhum saldo retornado na atualização da conta Gate.io. Mantendo valores anteriores.")
                
        except Exception as e:
            logger.error(f"❌ Erro atualizando conta de futuros Gate.io: {e}", exc_info=True)

    async def sync_positions_with_websocket(self):
        """Sincroniza posições de futuros Gate.io usando dados do WebSocket/REST da API."""
        try:
            # Chamada ao método get_open_positions_ws() da GateAPI para obter dados brutos de posições
            api_positions = await self.gate_api.get_open_positions_ws() #
            
            self.open_positions.clear() # Limpa as posições antigas no cache interno
            
            for pos_data in api_positions:
                contract = pos_data.get('contract')
                pos_size = self._safe_float(pos_data.get('size'))
                
                if contract and abs(pos_size) > 0: # Apenas posições com tamanho diferente de zero
                    side = 'LONG' if pos_size > 0 else 'SHORT'
                    
                    self.open_positions[contract] = {
                        'side': side,
                        'size': abs(pos_size),
                        'entry_price': self._safe_float(pos_data.get('entry_price')),
                        'mark_price': self._safe_float(pos_data.get('mark_price')),
                        'unrealized_pnl': self._safe_float(pos_data.get('unrealised_pnl')),
                        'margin': self._safe_float(pos_data.get('margin')),
                        'leverage': self._safe_float(pos_data.get('leverage'), 1.0),
                        'pnl_ratio': self._safe_float(pos_data.get('pnl_ratio', 0.0)), # Gate.io fornece isto
                        'status': 'OPEN',
                        'is_bot_managed': True, # Assume que todas as posições são gerenciadas pelo bot
                        'created_at': datetime.now().isoformat(), # Timestamp da última atualização da posição no cache
                        'exchange': 'GATE_IO'
                    }
                    
                    logger.debug(f"📊 Posição Gate.io Sincronizada: {contract} {side} {abs(pos_size)} @ {self.open_positions[contract]['entry_price']:.4f} "
                               f"(PnL: {self.open_positions[contract]['unrealized_pnl']:.2f})")
            
            if self.open_positions:
                logger.info(f"📋 {len(self.open_positions)} posições de futuros Gate.io sincronizadas no cache.")
            else:
                logger.info("📋 Nenhuma posição aberta de futuros Gate.io para sincronizar.")
            
            self.last_positions_update = time.time()
                    
        except Exception as e:
            logger.error(f"❌ Erro sincronizando posições de futuros Gate.io: {e}", exc_info=True)

    # MÉTODO FALTANTE NO SEU PORTFOLIO_MANAGER.PY (AGORA ADICIONADO)
    async def get_open_positions_ws(self) -> List[Dict]:
        """
        Retorna as posições abertas atualmente mantidas pelo Portfolio Manager.
        Este método é chamado por outras partes do bot (principalmente main.py),
        e o nome 'get_open_positions_ws' é esperado por lá.
        Ele retorna o cache interno, que é atualizado por sync_positions_with_websocket().
        """
        # Garante que as posições estejam o mais atualizadas possível no cache interno
        # O sync_positions_with_websocket() já é chamado por initialize_account_info() e update_account_info().
        # Podemos adicionar uma verificação de idade aqui se necessário, mas para evitar ciclos infinitos,
        # retornamos o que já temos no cache.
        
        # Filtra posições com size > 0, pois self.open_positions pode conter entradas de posições fechadas (size=0)
        # dependendo de como sync_positions_with_websocket lida com elas.
        active_positions = [
            pos for pos in self.open_positions.values()
            if self._safe_float(pos.get('size')) != 0.0 
        ]
        return active_positions

    async def calculate_position_size(self, symbol: str, signal_action: str, current_price: float,
                                    signal_data: Optional[Union[Dict, TradingSignal]] = None) -> Optional[float]: # Type Hint atualizado
        """Calcula tamanho da posição para futuros Gate.io baseado em gestão de risco"""
        if not self.initialized_account:
            await self.initialize_account_info()

        if signal_action not in ["BUY", "SELL"]:
            return None

        available_capital = self.available_balance
        
        if available_capital <= 0:
            logger.warning(f"❌ Capital insuficiente Gate.io para {symbol}: {available_capital:.2f}")
            return None

        risk_amount = available_capital * self.risk_per_trade # self.risk_per_trade já é 0-1
        
        confidence_multiplier = 1.0
        if isinstance(signal_data, TradingSignal):
            confidence = signal_data.confidence
            confidence_multiplier = max(0.5, min(1.5, confidence / 50.0))
        elif isinstance(signal_data, dict) and 'confidence' in signal_data:
            confidence = self._safe_float(signal_data.get('confidence'), 50)
            confidence_multiplier = max(0.5, min(1.5, confidence / 50.0))
        
        risk_amount *= confidence_multiplier
        
        leverage = min(self.max_leverage, 3.0) # Uso de max_leverage do Config
        
        notional_value = risk_amount * leverage
        
        # Tamanho em contratos (Gate.io usa inteiros para size)
        if current_price <= 0:
            logger.warning(f"❌ Preço atual de {symbol} é inválido ({current_price}), não é possível calcular o tamanho da posição.")
            return None
            
        position_size_raw = notional_value / current_price
        
        # Aplicar filtros de tamanho mínimo/máximo e step size
        adjusted_size = await self._apply_gate_symbol_filters(symbol, position_size_raw)
        
        if adjusted_size is not None and adjusted_size > 0:
            # Estimativa de margem usada para o log (valor nocional / alavancagem)
            estimated_margin = (adjusted_size * current_price) / leverage 
            logger.info(f"💡 {symbol} Gate.io: Tamanho={adjusted_size}, Margem≈{estimated_margin:.2f} USDT, "
                       f"Leverage={leverage}x, Risco={risk_amount:.2f}")
            return float(adjusted_size) # Retorna float para compatibilidade de tipo
        
        logger.warning(f"❌ Tamanho da posição ajustado resultou em zero ou None para {symbol}. Não é possível abrir trade.")
        return None

    async def _apply_gate_symbol_filters(self, symbol: str, size: float) -> Optional[int]:
        """Aplica filtros de símbolo para futuros Gate.io: min/max size e step size."""
        try:
            filters = await self.gate_api.get_symbol_filters(symbol) #
            if not filters:
                logger.warning(f"⚠️ Filtros não encontrados para {symbol} Gate.io, usando tamanho original")
                return self._safe_int(size) # Converte para int com safe_int
            
            lot_size_filter = filters.get('LOT_SIZE', {})
            min_size_contract = self._safe_float(lot_size_filter.get('minQty'), 1.0)
            max_size_contract = self._safe_float(lot_size_filter.get('maxQty'), 99999999.0)
            step_size_contract = self._safe_float(lot_size_filter.get('stepSize'), 1.0) # Adicionado stepSize

            # Arredondar a quantidade para o passo (stepSize) mais próximo
            if step_size_contract > 0:
                adjusted_size = round(size / step_size_contract) * step_size_contract
            else:
                adjusted_size = size # Se stepSize é zero, não ajusta por passo
            
            # Aplicar min/max de contrato
            adjusted_size = max(min_size_contract, min(adjusted_size, max_size_contract))
            
            # Gate.io Futures size é um INTEIRO
            final_size = self._safe_int(adjusted_size)

            if final_size <= 0:
                logger.warning(f"⚠️ Tamanho ajustado ({final_size}) para {symbol} é inválido após filtros. Retornando None.")
                return None

            return final_size

        except Exception as e:
            logger.error(f"❌ Erro aplicando filtros Gate.io para {symbol}: {e}", exc_info=True)
            # Em caso de erro, tenta retornar o tamanho original (convertido para int) como fallback.
            return self._safe_int(size) if size > 0 else None

    async def execute_trade(self, symbol: str, signal: Union[Dict[str, Any], TradingSignal]) -> Dict[str, Any]: # Retorna Dict para sucesso/erro
        """Executa trade de futuros Gate.io (abertura ou fechamento/inversão)."""
        try:
            # Normaliza o sinal para um dicionário para uso consistente
            if isinstance(signal, TradingSignal):
                action = signal.action
                current_price = signal.price
                confidence = signal.confidence
                signal_data = signal.indicators # Ou outros campos relevantes do TradingSignal
            elif isinstance(signal, dict):
                action = signal.get('action')
                current_price = signal.get('price')
                confidence = signal.get('confidence', 0)
                signal_data = signal
            else:
                raise ValueError(f"Formato de sinal inválido para {symbol}.")

            # Validações básicas
            if not action or current_price is None or self._safe_float(current_price) <= 0:
                logger.error(f"❌ Sinal inválido ou preço zero para {symbol} Gate.io: action={action}, price={current_price}")
                return {"success": False, "error": "Sinal ou preço inválido."}

            current_price = self._safe_float(current_price)
            current_position = await self.get_open_positions_ws() # Pega as posições atualizadas
            
            # Converter a lista de posições em um dicionário para fácil acesso pelo símbolo
            # (get_open_positions_ws já retorna lista de dict, precisamos de um dict por symbol)
            existing_position_map = {pos['contract']: pos for pos in current_position if pos.get('contract')}
            existing_position = existing_position_map.get(symbol) # Verifica se já existe uma posição para o símbolo

            # Lógica para lidar com posições existentes vs. novas trades
            if existing_position:
                return await self._handle_existing_position_gate(symbol, action, existing_position, signal_data)
            elif action in ["BUY", "SELL"]: # Apenas tenta abrir se não há posição existente
                return await self._open_new_position_gate(symbol, action, current_price, signal_data)
            
            # Se a ação não é BUY/SELL e não há posição existente para fechar, nada a fazer.
            logger.info(f"ℹ️ Ação '{action}' não aplicável ou nenhuma posição para {symbol}.")
            return {"success": False, "error": "Nenhuma ação executada."}

        except Exception as e:
            logger.error(f"❌ Erro executando trade Gate.io {symbol}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _handle_existing_position_gate(self, symbol: str, action: str, position: Dict, 
                                           signal_data: Dict) -> Dict[str, Any]: # Retorna Dict
        """Gerencia posição existente Gate.io (fechamento/inversão)"""
        current_side = position['side']
        
        # Lógica para fechar uma posição se o sinal for contrário
        if (current_side == 'LONG' and action == 'SELL') or \
           (current_side == 'SHORT' and action == 'BUY'):
            
            logger.info(f"🔄 Fechando posição {current_side} em {symbol} Gate.io por sinal contrário.")
            
            result = await self.close_single_position(symbol, TradeReason.STRATEGY_SELL_SIGNAL.value if action=='SELL' else TradeReason.STRATEGY_BUY_SIGNAL.value) # Chamar close_single_position
            
            if result and result.get('success'): # Verificar 'success' do close_single_position
                logger.info(f"✅ Ordem de fechamento para {symbol} Gate.io submetida com ID {result.get('order_id')}.")
                self._track_trade('CLOSE', symbol, TradeReason.STRATEGY_SELL_SIGNAL.value if action=='SELL' else TradeReason.STRATEGY_BUY_SIGNAL.value,
                                self._safe_float(position.get('mark_price')), self._safe_float(position.get('size')), signal_data)
                return {"success": True, "action": "CLOSED_POSITION", "order_id": result.get('order_id')}
            else:
                logger.warning(f"❌ Falha ao submeter ordem de fechamento para {symbol} Gate.io: {result}")
                return {"success": False, "error": f"Falha ao fechar posição: {result.get('message', 'Erro desconhecido')}"}
        
        logger.info(f"ℹ️ Posição {current_side} já existe para {symbol} Gate.io. Não há sinal contrário para fechar.")
        return {"success": False, "error": "Nenhuma ação executada para posição existente."}

    async def _open_new_position_gate(self, symbol: str, action: str, current_price: float, 
                                    signal_data: Dict) -> Dict[str, Any]: # Retorna Dict
        """Abre nova posição de futuros Gate.io"""
        # Verifica se pode abrir uma nova posição com base nos limites gerais (PortfolioManager)
        if not await self.can_open_new_position_overall(): #
            logger.warning(f"🚫 Não é possível abrir nova posição {symbol}. Limite de posições ou risco excedido.")
            return {"success": False, "error": "Limite de posições ou risco excedido."}

        position_size = await self.calculate_position_size(symbol, action, current_price, signal_data)
        
        if position_size is None or position_size <= 0:
            logger.warning(f"❌ Tamanho de posição inválido ou zero para {symbol} Gate.io. Não é possível abrir trade.")
            return {"success": False, "error": "Tamanho de posição inválido."}

        side = action.lower()
        position_side = 'LONG' if action == 'BUY' else 'SHORT'
        
        # Gate.io requer size negativo para short
        order_size_for_api = self._safe_int(position_size)
        if action == 'SELL':
            order_size_for_api = -order_size_for_api
        
        logger.info(f"🚀 Abrindo posição {position_side} Gate.io: {symbol} {abs(order_size_for_api)} @ {current_price:.4f}")
        
        order_result = await self.gate_api.place_order( #
            symbol=symbol,
            side=side,
            order_type="market", 
            size=order_size_for_api,
            price=None 
        )

        # Atualizar a verificação de sucesso para usar o retorno padronizado do place_order
        if order_result and order_result.get('success', False): #
            logger.info(f"✅ Ordem de abertura {position_side} {symbol} Gate.io submetida com ID {order_result.get('order_id')}.") #
            self._track_trade(action, symbol, TradeReason.STRATEGY_BUY_SIGNAL.value if action=='BUY' else TradeReason.STRATEGY_SELL_SIGNAL.value,
                            current_price, abs(self._safe_float(order_size_for_api)), signal_data)
            return {"success": True, "action": "OPENED_POSITION", "order_id": order_result.get('order_id')} #
        else:
            error_message = order_result.get('error', order_result.get('message', 'Unknown error')) if isinstance(order_result, dict) else 'Unknown error' #
            logger.warning(f"❌ Falha ao submeter ordem de abertura para {symbol} Gate.io: {error_message}. Resposta completa: {order_result}") #
            return {"success": False, "error": error_message} #

    async def close_all_positions(self, reason: str = "manual_close") -> int:
        """Fecha todas as posições de futuros Gate.io. Retorna o número de posições fechadas."""
        closed_count = 0
        
        # Faz uma cópia da lista de chaves, pois o dicionário pode ser modificado se o WS atualizar
        for symbol, position in list(self.open_positions.items()):
            try:
                # Apenas tenta fechar se a posição ainda estiver aberta (size != 0)
                if self._safe_float(position.get('size')) == 0.0:
                    logger.debug(f"Posição {symbol} já fechada. Pulando.")
                    continue

                logger.info(f"🔴 Fechando posição {symbol} Gate.io ({position['side']}) - Razão: {reason}...")
                
                # [NOVA LÓGICA] Chamar close_single_position
                result = await self.close_single_position(symbol, f"global_close_{reason}") #
                
                if result and result.get('success'): # Verificar 'success' do close_single_position
                    closed_count += 1
                    logger.info(f"✅ Ordem de fechamento para {symbol} Gate.io submetida. ID: {result.get('order_id')}")
                    self._track_trade('CLOSE', symbol, reason,
                                    self._safe_float(position.get('mark_price')), self._safe_float(position.get('size')))
                else:
                    logger.warning(f"❌ Falha ao submeter ordem de fechamento para {symbol} Gate.io: {result}")
                    
            except Exception as e:
                logger.error(f"❌ Erro fechando {symbol} Gate.io: {e}", exc_info=True)
        
        logger.info(f"📊 {closed_count} ordens de fechamento Gate.io submetidas.")
        return closed_count

    async def close_single_position(self, symbol: str, reason: str = "risk_management") -> Dict[str, Any]:
        """[NOVO MÉTODO] Fecha uma posição específica de futuros."""
        logger.info(f"🔴 Solicitando fechamento de posição para {symbol} (Razão: {reason})...")
        
        # Chama o método close_position da GateAPI, que já foi corrigido
        result = await self.gate_api.close_position(symbol)
        
        if result.get('success'):
            logger.info(f"✅ Pedido de fechamento para {symbol} enviado com sucesso. Order ID: {result.get('order_id')}")
            # Atualizar o estado local das posições após o fechamento
            await self.update_account_info()
            return {'success': True, 'message': 'Close order submitted', 'order_id': result.get('order_id')}
        else:
            logger.error(f"❌ Falha ao fechar posição para {symbol}: {result.get('message', 'Erro desconhecido')}")
            return {'success': False, 'message': result.get('message', 'Erro desconhecido'), 'full_response': result}

    async def get_position_pnl(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calcula PnL de uma posição de futuros Gate.io."""
        if symbol not in self.open_positions:
            return None

        position = self.open_positions[symbol]
        # Pega o preço atual do cache de current_prices, ou usa o mark_price da posição como fallback
        current_price = self.current_prices.get(symbol, self._safe_float(position.get('mark_price')))
        
        if current_price <= 0:
            logger.warning(f"⚠️ Preço atual para {symbol} é inválido ({current_price}). Não é possível calcular PnL.")
            return None

        # PnL já vem calculado do WebSocket para futuros Gate.io, mas garante float
        unrealized_pnl = self._safe_float(position.get('unrealized_pnl'))
        pnl_percentage = self._safe_float(position.get('pnl_ratio'))
        
        position_value = self._safe_float(position.get('size')) * current_price
        
        return {
            'pnl_percentage': pnl_percentage,
            'unrealized_pnl': unrealized_pnl,
            'current_price': current_price,
            'entry_price': self._safe_float(position.get('entry_price')),
            'position_value': position_value,
            'size': self._safe_float(position.get('size')),
            'side': position.get('side', 'UNKNOWN'),
            'leverage': self._safe_float(position.get('leverage'), 1.0),
            'exchange': 'GATE_IO'
        }

    async def monitor_positions_and_orders(self):
        """Monitora posições de futuros Gate.io via WebSocket (atualiza o cache interno)."""
        try:
            await self.update_account_info() # Isso já atualiza posições via sync_positions_with_websocket
            
            # Aqui você pode adicionar lógica para Stop Loss/Take Profit se não estiver no main
            # Ex: for symbol, position in self.open_positions.items():
            #        await self.check_position_risk_management(symbol, position)

            if self.open_positions:
                total_positions = len(self.open_positions)
                bot_managed = sum(1 for p in self.open_positions.values() if p.get('is_bot_managed', False))
                total_pnl = sum(self._safe_float(p.get('unrealized_pnl')) for p in self.open_positions.values())
                
                logger.debug(f"📋 {total_positions} posições futuros Gate.io ({bot_managed} bot) | "
                           f"PnL total: {total_pnl:.2f} USDT")

        except Exception as e:
            logger.error(f"❌ Erro monitorando posições de futuros Gate.io: {e}", exc_info=True)

    def _track_trade(self, action: str, symbol: str, reason: str, price: float, 
                    size: float, signal_data: Optional[Dict] = None):
        """Registra trade de futuros Gate.io no histórico."""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'reason': reason,
            'price': self._safe_float(price),
            'size': self._safe_float(size),
            'signal_confidence': self._safe_float(signal_data.get('confidence') if signal_data else 0),
            'account_balance': self.usdt_balance,
            'available_balance': self.available_balance,
            'type': 'FUTURES_GATE',
            'exchange': 'GATE_IO'
        }

        self.trade_history.append(trade_record)

        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        logger.info(f"🤖 GATE FUTURES TRADE: {action} {symbol} @ {price:.4f} | Size: {size:.4f} | {reason}")

    async def can_open_new_position_overall(self) -> bool:
        """
        Verifica se o Portfolio Manager pode abrir uma nova posição
        com base no limite de posições e risco total.
        Chamado pelo _open_new_position_gate
        """
        # Atualiza posições e saldo para ter os dados mais recentes
        await self.update_account_info() 

        current_positions_count = len(await self.get_open_positions_ws()) # Conta posições ativas
        
        # self.max_portfolio_positions já foi configurado na inicialização baseado em UltraSafeConfig ou Config
        if current_positions_count >= self.max_portfolio_positions: #
            logger.warning(f"🚫 Não pode abrir nova posição: Limite de posições ({current_positions_count}/{self.max_portfolio_positions}) atingido.") #
            return False
        
        total_risk_percent = await self.calculate_total_risk()
        # max_total_risk_percent já foi configurado (valor absoluto)
        if total_risk_percent >= self.max_total_risk_percent: #
            logger.warning(f"🚫 Não pode abrir nova posição: Risco total da carteira ({total_risk_percent:.2f}%) atinge ou excede limite de risco ({self.max_total_risk_percent}%)") #
            return False
        
        # Verificar se há capital disponível para uma nova trade
        # Usar MIN_BALANCE_USDT do UltraSafeConfig se ativo, senão default do Config
        min_usdt_needed_for_trade = UltraSafeConfig.MIN_BALANCE_USDT if UltraSafeConfig.ULTRA_SAFE_MODE else getattr(Config, 'MIN_BALANCE_USDT_TO_OPERATE', 50.0)
        if self.available_balance < min_usdt_needed_for_trade:
            logger.warning(f"⚠️ Saldo disponível insuficiente ({self.available_balance:.2f} USDT) para nova trade (min: {min_usdt_needed_for_trade:.2f} USDT).")
            return False

        logger.info(f"✅ Pode abrir nova posição. Posições abertas: {current_positions_count}/{self.max_portfolio_positions}. Saldo: {self.available_balance:.2f} USDT.")
        return True

    async def calculate_total_risk(self) -> float:
        """Calcula o risco total das posições abertas em porcentagem da equity."""
        try:
            positions = await self.get_open_positions_ws() # Pega posições atualizadas
            total_equity = self.usdt_balance # Pega equity do cache interno

            if total_equity <= 0:
                return 0.0 # Sem equity, sem risco
            
            total_risk_value_usd = 0.0
            
            for pos in positions:
                size = self._safe_float(pos.get('size'))
                entry_price = self._safe_float(pos.get('entry_price'))
                
                # Para futuros, o risco de uma posição é geralmente aproximado pelo seu valor nocional
                # ou pela margem inicial usada, ajustada por stop loss (se houver).
                # Aqui, para simplificar, vamos usar o valor nocional da posição
                # como uma proxy para o risco exposto.
                position_notional = abs(size) * entry_price
                total_risk_value_usd += position_notional
            
            total_risk_percent = (total_risk_value_usd / total_equity) * 100
            return total_risk_percent
            
        except Exception as e:
            logger.error(f"❌ Erro calculando risco total do portfólio: {e}", exc_info=True)
            return 100.0  # Assume risco máximo em caso de erro

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retorna resumo da carteira de futuros Gate.io"""
        # Verificar se WebSocket está conectado (usando o ws_client.connected)
        ws_status = 'DISCONNECTED'
        try:
            if hasattr(self.gate_api, 'ws_client') and hasattr(self.gate_api.ws_client, 'connected'):
                ws_status = 'CONNECTED' if self.gate_api.ws_client.connected else 'DISCONNECTED'
        except Exception:
            ws_status = 'UNKNOWN'

        summary = {
            'account_info': {
                'usdt_balance': self.usdt_balance,
                'available_balance': self.available_balance,
                'unrealized_pnl': self.unrealized_pnl,
                'total_positions': len(self.open_positions),
                'websocket_status': ws_status,
                'account_type': 'FUTURES_GATE',
                'exchange': 'GATE_IO'
            },
            'positions': {},
            'performance': {}
        }

        total_margin_used = 0.0
        total_pnl = 0.0
        total_position_value = 0.0

        for symbol, position in self.open_positions.items():
            current_price = self.current_prices.get(symbol, self._safe_float(position.get('mark_price')))
            
            if current_price <= 0:
                logger.warning(f"⚠️ Preço atual para {symbol} é inválido ({current_price}). Pulando no resumo.")
                continue

            position_size = self._safe_float(position.get('size'))
            position_value = position_size * current_price
            margin_used = self._safe_float(position.get('margin'))
            unrealized_pnl = self._safe_float(position.get('unrealized_pnl'))

            total_margin_used += margin_used
            total_pnl += unrealized_pnl
            total_position_value += position_value

            summary['positions'][symbol] = {
                'side': position.get('side', 'UNKNOWN'),
                'size': position_size,
                'entry_price': self._safe_float(position.get('entry_price')),
                'current_price': current_price,
                'position_value': position_value,
                'margin_used': margin_used,
                'leverage': self._safe_float(position.get('leverage'), 1.0),
                'unrealized_pnl': unrealized_pnl,
                'pnl_percentage': self._safe_float(position.get('pnl_ratio')),
                'is_bot_managed': position.get('is_bot_managed', False),
                'exchange': 'GATE_IO'
            }

        margin_utilization = (total_margin_used / self.usdt_balance * 100) if self.usdt_balance > 0 else 0.0

        summary['performance'] = {
            'total_equity': self.usdt_balance,
            'available_balance': self.available_balance,
            'total_margin_used': total_margin_used,
            'margin_utilization_pct': margin_utilization,
            'total_unrealized_pnl': total_pnl,
            'total_position_value': total_position_value,
            'pnl_percentage': (total_pnl / self.usdt_balance * 100) if self.usdt_balance > 0 else 0.0,
            'exchange': 'GATE_IO'
        }

        return summary

    def get_trade_history_summary(self, days: int = 7) -> Dict[str, Any]:
        """Resumo do histórico de trades de futuros Gate.io"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trade_history 
                        if datetime.fromisoformat(t['timestamp']) > cutoff_time]

        if not recent_trades:
            return {'message': f'Nenhum trade de futuros Gate.io nos últimos {days} dias', 'days': days}

        long_trades = [t for t in recent_trades if t['action'] == 'BUY']
        short_trades = [t for t in recent_trades if t['action'] == 'SELL']
        close_trades = [t for t in recent_trades if 'CLOSE' in t['action']] # 'CLOSE' ou 'CLOSE_ORDER_SUBMITTED'
        
        total_volume = sum(self._safe_float(t.get('price')) * self._safe_float(t.get('size')) for t in recent_trades)

        return {
            'period_days': days,
            'total_trades': len(recent_trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'close_trades': len(close_trades),
            'total_volume_usdt': total_volume,
            'trading_type': 'FUTURES_GATE',
            'websocket_efficiency': 'HIGH',
            'exchange': 'GATE_IO'
        }

    # ============ MÉTODOS DE COMPATIBILIDADE ============
    
    async def get_balance(self, asset: str) -> float:
        """Compatibilidade - para futuros Gate.io retorna saldo USDT disponível"""
        if not self.initialized_account:
            await self.initialize_account_info()
        
        if asset == 'USDT':
            return self.available_balance
        else:
            return 0.0

    async def sell_asset_to_raise_usdt(self, symbol_to_sell: str, current_price: float,
                                     quantity_to_sell: Optional[float] = None,
                                     reason: str = "capital_raise") -> bool:
        """Para futuros Gate.io, fecha posição para liberar margem."""
        if symbol_to_sell in self.open_positions:
            logger.info(f"💱 Fechando posição {symbol_to_sell} Gate.io para liberar margem...")
            
            # Chamar close_single_position
            result = await self.close_single_position(symbol_to_sell, reason) #
            
            if result and result.get('success'):
                logger.info(f"✅ Ordem de fechamento para {symbol_to_sell} Gate.io submetida para liberar margem. ID: {result.get('order_id')}")
                self._track_trade('CLOSE_ORDER_SUBMITTED', symbol_to_sell, reason,
                                  self._safe_float(current_price), self._safe_float(self.open_positions[symbol_to_sell].get('size')))
                return True
            else:
                logger.error(f"❌ Falha ao submeter ordem de fechamento para {symbol_to_sell} Gate.io: {result}")
                return False
        else:
            logger.warning(f"⚠️ Não há posição ativa em {symbol_to_sell} Gate.io para liberar margem.")
            return False

    # ============ MÉTODOS ADICIONAIS PARA ROBUSTEZ ============
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica a saúde do portfolio manager"""
        health_status = {
            'initialized': self.initialized_account,
            'last_balance_update': self.last_balance_update,
            'last_positions_update': self.last_positions_update,
            'current_time': time.time(),
            'positions_count': len(self.open_positions),
            'usdt_balance': self.usdt_balance,
            'available_balance': self.available_balance,
            'status': 'HEALTHY'
        }
        
        current_time = time.time()
        
        # Verificar se os dados estão atualizados (últimos 5 minutos)
        if current_time - self.last_balance_update > 300:
            health_status['status'] = 'STALE_DATA'
            health_status['warning'] = 'Dados de saldo desatualizados'
        
        if current_time - self.last_positions_update > 300:
            health_status['status'] = 'STALE_DATA'
            health_status['warning'] = 'Dados de posições desatualizados'
        
        return health_status
    
    async def force_refresh(self):
        """Força atualização de todos os dados"""
        logger.info("🔄 Forçando atualização de dados Gate.io...")
        try:
            await self.update_account_info()
            logger.info("✅ Dados Gate.io atualizados com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro na atualização forçada: {e}", exc_info=True)

# Alias para compatibilidade
PortfolioManager = GateFuturesPortfolioManager