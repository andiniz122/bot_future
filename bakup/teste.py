#!/usr/bin/env python3
"""
🛡️ VIGIA DE EMERGÊNCIA CORRIGIDO COM TELEGRAM - STOP LOSS 0.5% / TAKE PROFIT 10%
Monitora cada posição individualmente e fecha quando atinge os limites
VERSÃO REPARADA - Foco em PnL Não Realizado e execução da venda.

MONITORAMENTO DE PnL:
- 📊 PnL NÃO REALIZADO: Monitorado em tempo real (posições abertas) → Usado para SL/TP
- 💰 PnL REALIZADO: Registrado após fechamento → Usado para estatísticas finais
- ✅ CORRETO: Bot monitora PnL NÃO REALIZADO para determinar quando fechar posições

DEPENDÊNCIAS NECESSÁRIAS:
pip install aiohttp python-dotenv

CONFIGURAÇÃO .env:
TELEGRAM_BOT_TOKEN=seu_token_aqui
TELEGRAM_CHAT_ID=seu_chat_id_aqui
"""

import asyncio
import logging
import time
import os
import sys
import aiohttp
from datetime import datetime, timedelta, timezone # Importar timezone para consistência
from gate_api import GateAPI
from portfolio_manager import GateFuturesPortfolioManager

# Setup logging melhorado
logging.basicConfig(
    level=logging.INFO, # Nível padrão INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Inclua %(name)s
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'vigia_emergencia_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# [REPARO] Ajustar níveis de log para módulos específicos para DEBUG
logging.getLogger('emergency_watcher').setLevel(logging.INFO) # Mantenha em INFO para o principal
logging.getLogger('gate_api').setLevel(logging.DEBUG) # DEBUG para a API para ver todas as requisições/respostas
logging.getLogger('portfolio_manager_gate').setLevel(logging.DEBUG) # DEBUG para o PM para ver PnL e posições
logging.getLogger('telegram_notifier').setLevel(logging.INFO) # Para ver se as notificações estão saindo

logger = logging.getLogger('emergency_watcher')

class EmergencyTelegramNotifier:
    """Sistema de notificações Telegram para o bot de emergência"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            logger.info("📱 Sistema de notificações Telegram ATIVADO")
        else:
            logger.warning("⚠️ Credenciais Telegram não encontradas - Notificações DESABILITADAS")
    
    async def send_message(self, message: str, parse_mode: str = 'HTML'):
        """Envia mensagem via Telegram de forma assíncrona"""
        if not self.enabled:
            logger.debug("📱 Telegram desabilitado - mensagem não enviada")
            return False
        
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload) as response:
                    response_text = await response.text() # [REPARO] Capturar texto da resposta
                    if response.status == 200:
                        logger.info("📱 Mensagem Telegram enviada com sucesso")
                        return True
                    else:
                        logger.error(f"❌ Erro enviando Telegram: {response.status} - {response_text}") # [REPARO] Logar texto completo
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Erro no sistema Telegram: {e}")
            return False
    
    async def send_emergency_close(self, contract: str, reason: str, pnl_pct: float, 
                                 pnl_usd: float, side: str, entry_price: float, close_price: float):
        """Envia notificação específica de fechamento de emergência"""
        try:
            # Ícones baseados no resultado
            if reason == "STOP LOSS":
                icon = "🛑"
                color = "🔴"
                action_text = "STOP LOSS EMERGENCIAL"
            else:  # TAKE PROFIT
                icon = "🎯"
                color = "🟢"
                action_text = "TAKE PROFIT EMERGENCIAL"
            
            pnl_sign = "+" if pnl_usd >= 0 else ""
            
            message = (
                f"{icon} <b>🚨 BOT DE EMERGÊNCIA ATUOU! 🚨</b> {color}\n\n"
                f"⚡ <b>AÇÃO:</b> {action_text}\n"
                f"📊 <b>Ativo:</b> <code>{contract}</code>\n"
                f"📈 <b>Posição:</b> {side}\n"
                f"💰 <b>Entrada:</b> <code>{entry_price:.8f}</code>\n"
                f"💸 <b>Fechamento:</b> <code>{close_price:.8f}</code>\n\n"
                f"📈 <b>Resultado:</b> <code>{pnl_sign}{pnl_pct:.2f}%</code>\n"
                f"💵 <b>PnL:</b> <code>{pnl_sign}{pnl_usd:.4f} USDT</code>\n\n"
                f"🛡️ <b>Motivo:</b> <i>Sistema de proteção automática</i>\n"
                f"⏰ <b>Horário:</b> <code>{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}</code>\n\n"
                f"🤖 <i>Bot de Emergência - Proteção Ativa</i>"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Erro enviando notificação de fechamento: {e}")
    
    async def send_startup_notification(self, positions_count: int):
        """Envia notificação de inicialização do bot de emergência"""
        try:
            message = (
                f"🛡️ <b>BOT DE EMERGÊNCIA INICIADO</b> 🛡️\n\n"
                f"🔍 <b>Posições Monitoradas:</b> {positions_count}\n"
                f"🛑 <b>Stop Loss:</b> {self.STOP_LOSS_PCT}%\n" # [REPARO] Usar self.STOP_LOSS_PCT
                f"🎯 <b>Take Profit:</b> {self.TAKE_PROFIT_PCT}%\n" # [REPARO] Usar self.TAKE_PROFIT_PCT
                f"⚡ <b>Frequência:</b> Verificação a cada 5s\n\n"
                f"🚨 <b>Sistema de proteção ATIVO!</b>\n"
                f"💡 <i>Fechamento automático quando limites forem atingidos</i>\n\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}</code>"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Erro enviando notificação de startup: {e}")
    
    async def send_shutdown_notification(self, stats: dict, positions_remaining: int):
        """Envia notificação de encerramento do bot"""
        try:
            runtime_str = str(datetime.now() - stats.get('start_time', datetime.now())).split('.')[0]
            
            message = (
                f"🛑 <b>BOT DE EMERGÊNCIA ENCERRADO</b>\n\n"
                f"⏱️ <b>Tempo ativo:</b> {runtime_str}\n"
                f"🔍 <b>Verificações:</b> {stats.get('checks_performed', 0)}\n"
                f"📊 <b>Posições fechadas:</b> {stats.get('positions_closed', 0)}\n"
                f"🛑 <b>Stop Losses:</b> {stats.get('stop_losses', 0)}\n"
                f"🎯 <b>Take Profits:</b> {stats.get('take_profits', 0)}\n"
                f"💰 <b>PnL REALIZADO Total:</b> <code>{stats.get('total_realized_pnl_usd', 0):+.4f} USDT</code>\n"
                f"❌ <b>Erros:</b> {stats.get('errors', 0)}\n"
                f"📋 <b>Posições restantes:</b> {positions_remaining}\n\n"
                f"🛡️ <i>Sistema de proteção desativado</i>\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}</code>"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Erro enviando notificação de shutdown: {e}")
    
    async def send_emergency_close_all(self, total_positions: int, closed_count: int):
        """Envia notificação de fechamento de emergência total"""
        try:
            message = (
                f"🆘 <b>FECHAMENTO DE EMERGÊNCIA TOTAL!</b> 🆘\n\n"
                f"⚡ <b>Ação:</b> Usuário solicitou fechamento imediato\n"
                f"📊 <b>Posições encontradas:</b> {total_positions}\n"
                f"✅ <b>Posições fechadas:</b> {closed_count}\n"
                f"❌ <b>Falhas:</b> {total_positions - closed_count}\n\n"
                f"🚨 <b>TODAS AS POSIÇÕES FORAM PROCESSADAS!</b>\n"
                f"💡 <i>Verifique manualmente na exchange</i>\n\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}</code>"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Erro enviando notificação de emergência total: {e}")
    
    async def send_error_notification(self, error_msg: str, context: str = ""):
        """Envia notificação de erro crítico"""
        try:
            message = (
                f"❌ <b>ERRO NO BOT DE EMERGÊNCIA!</b>\n\n"
                f"🚨 <b>Erro:</b> <code>{error_msg}</code>\n"
                f"📍 <b>Contexto:</b> <i>{context}</i>\n\n"
                f"⚠️ <b>Verificação manual necessária!</b>\n"
                f"⏰ <code>{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}</code>"
            )
            
            await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Erro enviando notificação de erro: {e}")

class EmergencyWatcherCorrected:
    """Vigia CORRIGIDO que monitora posições com SL 0.5% e TP 10%"""

    def __init__(self):
        self.gate_api = GateAPI()
        self.portfolio_manager = GateFuturesPortfolioManager(self.gate_api)

        # Configurações de proteção (MAIS CONSERVADORAS para emergência)
        # [REPARO] Mantidos os valores que você definiu para este bot
        self.STOP_LOSS_PCT = -0.5    # 0.5% de perda
        self.TAKE_PROFIT_PCT = 10.0  # 10% de lucro

        # Estado das posições CORRIGIDO
        self.positions_data = {} # Cache local das posições
        self.start_time = datetime.now()
        self.is_running = False
        
        # Cache para instrumentos
        self.instruments_cache = {} # Usado para resolver símbolos UNKNOWN, se necessário
        
        # Sistema de notificações Telegram
        self.telegram = EmergencyTelegramNotifier()
        
        # Estatísticas
        self.stats = {
            'checks_performed': 0,
            'positions_closed': 0,
            'stop_losses': 0,
            'take_profits': 0,
            'errors': 0,
            'start_time': self.start_time,
            'total_realized_pnl_usd': 0.0,
            'realized_pnl_history': []
        }

        logger.warning("🛡️ VIGIA DE EMERGÊNCIA CORRIGIDO INICIALIZADO")
        logger.warning(f"🛑 STOP LOSS: {self.STOP_LOSS_PCT}% | 🎯 TAKE PROFIT: {self.TAKE_PROFIT_PCT}%")

    async def initialize(self):
        """Inicializa e verifica posições CORRIGIDO"""
        try:
            logger.info("🔄 Inicializando vigia CORRIGIDO...")
            
            # Inicializar portfolio manager
            # [REPARO] A chamada a initialize_account_info no PM já lida com o saldo e posições via WS/REST
            await self.portfolio_manager.initialize_account_info() 
            
            # Carregar cache de instrumentos para resolver símbolos
            await self.load_instruments_cache()
            
            # [REPARO] Remover analyze_positions_corrected daqui para evitar duplicação no run_watcher
            # A primeira análise será feita no primeiro ciclo do run_watcher
            
            return True
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}", exc_info=True)
            self.stats['errors'] += 1
            await self.telegram.send_error_notification(str(e), "Initialization Failure")
            return False

    async def load_instruments_cache(self):
        """Carrega cache de instrumentos para resolver símbolos UNKNOWN"""
        try:
            logger.info("📡 Carregando instrumentos disponíveis...")
            instruments = await self.gate_api.get_instruments_info() # Chama GateAPI para obter instrumentos
            
            if instruments:
                for inst in instruments:
                    if isinstance(inst, dict) and 'name' in inst:
                        symbol_name = inst['name']
                        self.instruments_cache[symbol_name] = inst
                        
            logger.info(f"📋 {len(self.instruments_cache)} instrumentos carregados no cache")
            
        except Exception as e:
            logger.error(f"❌ Erro carregando instrumentos: {e}")
            self.stats['errors'] += 1
            await self.telegram.send_error_notification(str(e), "Loading Instruments Cache")

    # [REPARO] `analyze_positions_corrected` foi removido para evitar duplicação.
    # Sua lógica foi incorporada e aprimorada em `update_positions_data_corrected`.
    # A responsabilidade principal de obter e sincronizar posições em tempo real é do PortfolioManager
    # e do método `update_positions_data_corrected` que o chama.

    def extract_contract_safely(self, position: Dict) -> Optional[str]: # [REPARO] Adicionado type hint
        """Extrai o símbolo do contrato de forma segura"""
        # [REPARO] Melhorar a robustez da extração do contrato
        contract_keys = ['contract', 'symbol', 'instrument', 'currency_pair'] 
        for key in contract_keys:
            if key in position and position[key]:
                contract = str(position[key]).strip()
                if contract and contract != "UNKNOWN" and len(contract) > 2:
                    return contract
        
        logger.warning(f"⚠️ Não foi possível extrair contrato de forma segura: {position}")
        return None # Retorna None se não encontrar

    def recover_symbol_from_position(self, position: Dict) -> Optional[str]: # [REPARO] Adicionado type hint
        """Tenta recuperar símbolo usando dados da posição. Apenas para símbolos UNKNOWN."""
        # [REPARO] Esta lógica é uma heurística fraca e deve ser um último recurso.
        # O ideal é que a API forneça o contrato corretamente.
        entry_price = self.safe_float(position.get('entry_price', 0))
        
        if entry_price > 0:
            for symbol_name, inst_data in self.instruments_cache.items():
                if inst_data.get('last') and abs(self.safe_float(inst_data['last']) - entry_price) / entry_price < 0.001: # 0.1% de tolerância
                    logger.warning(f"ℹ️ Símbolo recuperado por heurística: {symbol_name} (preço similar a {entry_price})")
                    return symbol_name
            # Heurísticas baseadas em preço (menos confiáveis)
            if 20000 <= entry_price <= 100000:  return "BTC_USDT"
            elif 1500 <= entry_price <= 5000:  return "ETH_USDT"
            elif 100 <= entry_price <= 300:  return "SOL_USDT"
            elif 0.3 <= entry_price <= 3:  return "XRP_USDT"
            elif 200 <= entry_price <= 800:  return "BNB_USDT"
        
        logger.warning(f"⚠️ Não foi possível recuperar símbolo para entry_price: {entry_price}. Posição: {position}")
        return None

    def safe_float(self, value: Any) -> float: # [REPARO] Adicionado type hint
        """Converte valor para float de forma segura"""
        try:
            if value is None or value == '':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Valor '{value}' não pôde ser convertido para float. Retornando 0.0.") # [REPARO] Logar erro
            return 0.0

    async def run_watcher(self):
        """Executa o monitoramento contínuo CORRIGIDO"""
        self.is_running = True
        logger.warning("👁️ INICIANDO VIGIA CORRIGIDO EM SEGUNDO PLANO")

        try:
            # [REPARO] Enviar notificação de startup no início do watcher, após a primeira análise de posições
            await self.update_positions_data_corrected() # Garante que positions_data está populado inicialmente
            await self.telegram.send_startup_notification(len(self.positions_data))

            while self.is_running:
                await self.check_positions_corrected()
                await asyncio.sleep(5)  # Verificar a cada 5 segundos (mais frequente)
        except KeyboardInterrupt:
            logger.warning("🛑 Vigia interrompido pelo usuário")
        except Exception as e:
            self.stats['errors'] += 1 # [REPARO] Contar erros
            logger.error(f"❌ Erro no vigia: {e}", exc_info=True)
            await self.telegram.send_error_notification(str(e), "Main Watcher Loop Error") # [REPARO] Notificar
        finally:
            self.is_running = False
            await self.final_report()
            await self.close() # [REPARO] Fechar conexões no final.

    async def check_positions_corrected(self):
        """Verifica cada posição contra SL e TP CORRIGIDO"""
        try:
            self.stats['checks_performed'] += 1
            
            # [REPARO] Atualizar dados das posições a cada checagem para ter o PnL mais atualizado
            await self.update_positions_data_corrected()

            if not self.positions_data:
                logger.debug("😴 Sem posições abertas - Vigia ativo")
                return

            logger.info(f"🔍 Verificando {len(self.positions_data)} posições...")
            logger.debug("📊 Monitorando PnL NÃO REALIZADO (posições abertas) para SL/TP")
            
            positions_to_close = []
            
            for contract, data in list(self.positions_data.items()): # [REPARO] Iterar sobre uma cópia para permitir exclusão
                
                # [REPARO] Puxar o PnL mais atualizado do portfolio_manager para esta verificação
                # O portfolio_manager.get_position_pnl() puxa os dados mais recentes do seu cache
                pnl_data_from_pm = await self.portfolio_manager.get_position_pnl(contract)
                if not pnl_data_from_pm:
                    logger.warning(f"⚠️ Não foi possível obter PnL de {contract} do PortfolioManager. Pulando checagem SL/TP para este ciclo.")
                    continue

                pnl_pct = pnl_data_from_pm['pnl_percentage']
                unrealized_pnl = pnl_data_from_pm['unrealized_pnl']
                current_price = pnl_data_from_pm['current_price']
                entry_price = pnl_data_from_pm['entry_price']

                # [REPARO] Atualizar os dados 'data' no cache local com os valores mais recentes
                data.update({
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': unrealized_pnl,
                    'last_check': time.time()
                })
                self.positions_data[contract] = data # Atualiza o cache principal

                logger.debug(f"📊 {contract}: PnL {pnl_pct:+.2f}% | "
                           f"SL: {self.STOP_LOSS_PCT}% | TP: {self.TAKE_PROFIT_PCT}%")

                # Verificar se precisa fechar a posição
                if pnl_pct <= self.STOP_LOSS_PCT:
                    reason = "STOP LOSS"
                    logger.critical(f"🛑 {reason} ATIVADO PARA {contract}: PnL {pnl_pct:+.2f}%") # [REPARO] Critical log
                    positions_to_close.append((contract, data, reason))
                    
                elif pnl_pct >= self.TAKE_PROFIT_PCT:
                    reason = "TAKE PROFIT"
                    logger.critical(f"🎯 {reason} ATIVADO PARA {contract}: PnL {pnl_pct:+.2f}%") # [REPARO] Critical log
                    positions_to_close.append((contract, data, reason))
            
            # Fechar posições que atingiram os limites
            for contract, data, reason in positions_to_close:
                # [REPARO] Chamar close_single_position_corrected
                success_close = await self.close_single_position_corrected(contract, data, reason)
                if not success_close:
                    logger.error(f"❌ Falha ao fechar posição {contract} após o acionamento de {reason}.")
                    self.stats['errors'] += 1 # [REPARO] Contar erro
                    await self.telegram.send_error_notification(f"Falha ao fechar SL/TP para {contract}", f"Acionamento {reason}")
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erro verificando posições: {e}", exc_info=True)
            
            # Enviar notificação de erro crítico via Telegram
            await self.telegram.send_error_notification(
                str(e), 
                f"check_positions_corrected - {len(self.positions_data)} posições monitoradas"
            )

    async def close_single_position_corrected(self, contract: str, data: Dict[str, Any], reason: str) -> bool: # [REPARO] Retorna bool
        """Fecha uma posição individual CORRIGIDO"""
        try:
            logger.critical(f"🛑 FECHANDO {contract}... Motivo: {reason}")
            logger.critical(f"   📊 PnL: {data['pnl_pct']:+.2f}% | Size: {data['size']:.6f}")

            position_side = data['side'].upper()
            close_side = "sell" if position_side == "LONG" else "buy"
            position_size = abs(self.safe_float(data['size']))
            
            if position_size <= 0:
                logger.warning(f"⚠️ Tamanho da posição para {contract} é zero ou inválido. Não há o que fechar.")
                # [REPARO] Remover do cache local se o tamanho for zero.
                if contract in self.positions_data:
                    del self.positions_data[contract]
                return True # Considerar "sucesso" porque não há nada para fechar
            
            logger.info(f"   🔄 Posição {position_side} -> Ordem {close_side.upper()} de {position_size:.6f}")

            # [REPARO] A GateAPI.close_position já encapsula o submit_futures_order com reduce_only
            # Basta chamar ela. Não precisamos de duas tentativas aqui.
            order_result = await self.gate_api.close_position(contract)
            
            if order_result and order_result.get('success', False):
                logger.critical(f"✅ {contract} FECHADO COM SUCESSO! ID: {order_result.get('order_id')}")
                await self.handle_successful_close(contract, data, reason)
                return True
            else:
                error_msg = order_result.get('message', 'Erro desconhecido') if order_result else 'Sem resposta'
                logger.error(f"❌ FALHA CRÍTICA: Não foi possível fechar {contract} via gate_api.close_position! Erro: {error_msg}. Resposta: {order_result.get('full_response', 'N/A')}")
                logger.critical(f"   📋 Dados da posição que falhou: {data}")
                return False

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erro no processo de fechamento de {contract}: {e}", exc_info=True)
            return False

    async def handle_successful_close(self, contract: str, data: Dict[str, Any], reason: str): # [REPARO] Adicionado type hint
        """Processa fechamento bem-sucedido"""
        try:
            # Calcular PnL REALIZADO final
            # [REPARO] PnL da posição que foi fechada
            realized_pnl_usd = self.safe_float(data['pnl_usd'])  
            realized_pnl_pct = self.safe_float(data['pnl_pct']) 
            
            # Atualizar estatísticas
            self.stats['positions_closed'] += 1
            self.stats['total_realized_pnl_usd'] += realized_pnl_usd
            
            # Adicionar ao histórico de PnL realizado
            self.stats['realized_pnl_history'].append({
                'contract': contract,
                'side': data['side'],
                'reason': reason,
                'entry_price': self.safe_float(data['entry_price']),
                'close_price': self.safe_float(data['current_price']),
                'realized_pnl_usd': realized_pnl_usd,
                'realized_pnl_pct': realized_pnl_pct,
                'timestamp': datetime.now(timezone.utc), # [REPARO] Usar UTC para consistência
                'closed_at': time.time()
            })
            
            if reason == "STOP LOSS":
                self.stats['stop_losses'] += 1
            elif reason == "TAKE PROFIT":
                self.stats['take_profits'] += 1
            
            # Enviar notificação via Telegram
            await self.telegram.send_emergency_close(
                contract=contract,
                reason=reason,
                pnl_pct=realized_pnl_pct,      
                pnl_usd=realized_pnl_usd,      
                side=data['side'],
                entry_price=self.safe_float(data['entry_price']),
                close_price=self.safe_float(data['current_price'])
            )
            
            # Remover do monitoramento
            if contract in self.positions_data:
                del self.positions_data[contract]
            
            # Log de sucesso com PnL realizado
            logger.critical(f"🎉 {contract} REMOVIDO DO MONITORAMENTO")
            logger.critical(f"   📈 PnL REALIZADO: {realized_pnl_pct:+.2f}% ({realized_pnl_usd:+.4f} USDT)")
            logger.critical(f"   💰 PnL Total Acumulado: {self.stats['total_realized_pnl_usd']:+.4f} USDT")
            logger.critical(f"   📱 Notificação Telegram enviada!")
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erro processando fechamento: {e}")
            await self.telegram.send_error_notification(str(e), f"Handle Successful Close for {contract}")

    async def update_positions_data_corrected(self):
        """Atualiza dados das posições CORRIGIDO - Puxa dados do PortfolioManager"""
        try:
            # [REPARO] O PortfolioManager já tem a lógica de sincronizar com WS/REST.
            # Basta chamar o método do PM que ele já deve ter atualizado o cache interno.
            await self.portfolio_manager.update_account_info() # Isso garante que o PM tem os dados mais recentes
            
            # Agora, pegue as posições ATUALIZADAS diretamente do cache do PortfolioManager
            # O get_open_positions_ws() do PM já retorna uma lista de posições ativas.
            raw_positions_from_pm = await self.portfolio_manager.get_open_positions_ws()
            current_contracts = set()

            if raw_positions_from_pm:
                logger.debug(f"🔄 Obtendo {len(raw_positions_from_pm)} posições atualizadas do PortfolioManager.")
                for position in raw_positions_from_pm:
                    contract = self.extract_contract_safely(position)
                    
                    if not contract: # Se não conseguiu extrair o contrato, pula
                        logger.warning(f"⚠️ Posição com contrato inválido no PM: {position}")
                        continue
                    
                    current_contracts.add(contract)
                    
                    # [REPARO] Extrair diretamente os dados mais recentes do 'position'
                    side = position.get('side', 'UNKNOWN').upper()
                    size_float = self.safe_float(position.get('size', 0))
                    entry_price = self.safe_float(position.get('entry_price', 0))
                    mark_price = self.safe_float(position.get('mark_price', 0))
                    unrealized_pnl = self.safe_float(position.get('unrealized_pnl', 0))
                    pnl_ratio = self.safe_float(position.get('pnl_ratio', 0))

                    if abs(size_float) < 0.000001: # Posição muito pequena ou fechada
                        continue # Não adiciona ao monitoramento se já fechada ou muito pequena

                    # [REPARO] Recalcular PnL% usando pnl_ratio da Gate.io, ou recalcular com mark_price se necessário
                    if pnl_ratio != 0:
                        pnl_pct = pnl_ratio * 100 # Gate.io pnl_ratio já é 0.XX, precisa * 100
                    elif entry_price > 0 and mark_price > 0:
                        if side == 'LONG':
                            pnl_pct = ((mark_price - entry_price) / entry_price) * 100
                        else:  # SHORT
                            pnl_pct = ((entry_price - mark_price) / entry_price) * 100
                    else:
                        pnl_pct = 0.0 # Fallback se não há dados de preço suficientes

                    # Adicionar/Atualizar no cache local do vigia
                    self.positions_data[contract] = {
                        'side': side,
                        'size': size_float,
                        'entry_price': entry_price,
                        'current_price': mark_price, # Mantenha o mark_price atualizado
                        'pnl_pct': pnl_pct,
                        'pnl_usd': unrealized_pnl,
                        'last_check': time.time(),
                        'raw_position': position # Manter a posição original completa para debug
                    }
                    logger.debug(f"🔄 Vigia cache: {contract} atualizado - PnL {pnl_pct:+.2f}% ({unrealized_pnl:+.4f} USDT)")

            # Remover posições que não estão mais ativas na Gate.io (ou seja, foram fechadas)
            # Iterar sobre uma cópia de self.positions_data para permitir modificação durante a iteração
            for contract_in_cache in list(self.positions_data.keys()):
                if contract_in_cache not in current_contracts:
                    logger.warning(f"✅ Posição {contract_in_cache} foi fechada na exchange e removida do monitoramento.")
                    del self.positions_data[contract_in_cache]

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erro atualizando posições: {e}", exc_info=True)
            await self.telegram.send_error_notification(
                str(e), 
                "update_positions_data_corrected"
            )

    async def process_new_position_corrected(self, position: Dict[str, Any], contract: str): # [REPARO] Adicionado type hint
        """Processa uma nova posição detectada CORRIGIDO"""
        # [REPARO] Esta função não é mais diretamente chamada se update_positions_data_corrected
        # está processando tudo de raw_positions_from_pm.
        # No entanto, mantida para clareza ou se houver outro fluxo.
        try:
            side = position.get('side', 'UNKNOWN').upper()
            size = self.safe_float(position.get('size', 0))
            entry_price = self.safe_float(position.get('entry_price', 0))
            mark_price = self.safe_float(position.get('mark_price', 0))
            unrealized_pnl = self.safe_float(position.get('unrealized_pnl', 0))
            pnl_ratio = self.safe_float(position.get('pnl_ratio', 0)) # [REPARO] Obter pnl_ratio
            
            if mark_price <= 0: mark_price = entry_price

            if pnl_ratio != 0:
                pnl_pct = pnl_ratio * 100 # [REPARO] Usar pnl_ratio
            elif entry_price > 0 and mark_price > 0:
                if side == 'LONG': pnl_pct = ((mark_price - entry_price) / entry_price) * 100
                else: pnl_pct = ((entry_price - mark_price) / entry_price) * 100
            else: pnl_pct = 0.0

            self.positions_data[contract] = {
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'current_price': mark_price,
                'pnl_pct': pnl_pct,
                'pnl_usd': unrealized_pnl,
                'last_check': time.time()
            }

            logger.critical(f"🆕 NOVA POSIÇÃO DETECTADA: {contract} {side}")
            logger.critical(f"   📊 Entry: {entry_price:.8f} | Atual: {mark_price:.8f}")
            logger.critical(f"   💰 Size: {size:.6f} | PnL: {pnl_pct:+.2f}% ({unrealized_pnl:+.4f} USDT)")

        except Exception as e:
            logger.error(f"❌ Erro processando nova posição: {e}", exc_info=True)
            self.stats['errors'] += 1
            await self.telegram.send_error_notification(str(e), f"Processing New Position for {contract}")

    async def emergency_close_all_corrected(self) -> int: # [REPARO] Retorna int
        """Fecha todas as posições imediatamente (emergência) CORRIGIDO"""
        try:
            logger.critical("🆘 INICIANDO FECHAMENTO DE EMERGÊNCIA TOTAL!")
            
            positions_to_close = await self.portfolio_manager.get_open_positions_ws()

            if not positions_to_close:
                logger.warning("ℹ️ Nenhuma posição para fechar no fechamento de emergência.")
                return 0

            logger.critical(f"🆘 FECHAMENTO DE EMERGÊNCIA DE {len(positions_to_close)} POSIÇÕES!")

            closed_count = 0
            tasks = [] # [REPARO] Usar tasks para fechar em paralelo e acelerar
            for pos in positions_to_close:
                contract = self.extract_contract_safely(pos)
                if not contract: continue

                size = self.safe_float(pos.get('size', 0))
                if abs(size) > 0.000001:
                    tasks.append(self.close_single_position_corrected(contract, pos, "GLOBAL_EMERGENCY_CLOSE")) # [REPARO] Chamar close_single_position_corrected

            results = await asyncio.gather(*tasks, return_exceptions=True) # [REPARO] Executar em paralelo
            
            for res in results:
                if res is True: closed_count += 1 # [REPARO] Contar sucessos

            if closed_count > 0:
                logger.critical(f"✅ EMERGÊNCIA: {closed_count} posições fechadas.")
                await self.telegram.send_emergency_close_all(len(positions_to_close), closed_count)
                self.positions_data.clear() # Limpa o cache local após fechamento total
            else:
                logger.error("❌ EMERGÊNCIA: Nenhuma posição foi fechada!")
                await self.telegram.send_error_notification(
                    "Falha no fechamento de emergência - Nenhuma posição fechada",
                    "emergency_close_all"
                )
            return closed_count

        except Exception as e:
            logger.error(f"❌ Erro no fechamento de emergência: {e}", exc_info=True)
            self.stats['errors'] += 1
            await self.telegram.send_error_notification(str(e), "Emergency Close All Error")
            return 0

    async def final_report(self):
        """Relatório final MELHORADO com PnL realizado"""
        try:
            runtime = datetime.now() - self.start_time
            
            logger.critical("\n" + "=" * 80)
            logger.critical("📊 RELATÓRIO FINAL DO VIGIA DE EMERGÊNCIA")
            logger.critical("=" * 80)
            logger.critical(f"⏱️  Tempo de execução: {str(runtime).split('.')[0]}")
            logger.critical(f"🔍 Verificações realizadas: {self.stats['checks_performed']}")
            logger.critical(f"📊 Posições fechadas: {self.stats['positions_closed']}")
            logger.critical(f"🛑 Stop Losses: {self.stats['stop_losses']}")
            logger.critical(f"🎯 Take Profits: {self.stats['take_profits']}")
            logger.critical(f"❌ Erros encontrados: {self.stats['errors']}")
            logger.critical(f"💰 PnL REALIZADO TOTAL: {self.stats['total_realized_pnl_usd']:+.4f} USDT")
            logger.critical(f"📋 Posições ainda monitoradas: {len(self.positions_data)}") # Aqui será 0 se tudo fechou
            
            # Mostrar histórico de PnL realizado
            if self.stats['realized_pnl_history']:
                logger.critical("💹 HISTÓRICO DE PnL REALIZADO:")
                for trade in self.stats['realized_pnl_history']:
                    logger.critical(f"   • {trade['contract']} {trade['side']}: "
                                  f"{trade['realized_pnl_pct']:+.2f}% "
                                  f"({trade['realized_pnl_usd']:+.4f} USDT) "
                                  f"- {trade['reason']} @ {trade['timestamp'].strftime('%H:%M:%S')}") # [REPARO] Formatar timestamp
            
            if self.positions_data:
                logger.critical("🔍 Posições restantes (PnL NÃO REALIZADO):")
                for contract, data in self.positions_data.items():
                    logger.critical(f"   • {contract} {data['side']}: PnL {data['pnl_pct']:+.2f}% ({data['pnl_usd']:+.4f} USDT)")
                    
            logger.critical("=" * 80)
            
            await self.telegram.send_shutdown_notification(self.stats, len(self.positions_data))
            
        except Exception as e:
            logger.error(f"❌ Erro no relatório final: {e}", exc_info=True)
            self.stats['errors'] += 1
            await self.telegram.send_error_notification(str(e), "Final Report Generation")

    async def close(self):
        """Fecha conexões"""
        try:
            self.is_running = False
            await self.gate_api.close() # Garante que a sessão aiohttp e websockets são fechadas
            logger.info("✅ Vigia encerrado")
        except Exception as e:
            logger.error(f"❌ Erro fechando conexões: {e}", exc_info=True)
            self.stats['errors'] += 1
            await self.telegram.send_error_notification(str(e), "Closing Connections")

async def main():
    """Função principal CORRIGIDA"""
    watcher = None

    try:
        print("🛡️ VIGIA DE EMERGÊNCIA CORRIGIDO COM TELEGRAM - GATE.IO")
        print("=" * 80)
        print(f"🛑 STOP LOSS: 0.5% | 🎯 TAKE PROFIT: 10%")
        print("🔧 VERSÃO REPARADA - Resolve problemas de PnL e venda") # [REPARO] Mensagem
        print("📱 NOVO: Notificações completas via Telegram")
        print("=" * 80)

        print("\n📊 INFORMAÇÕES IMPORTANTES:")
        print("• PnL NÃO REALIZADO: Lucro/prejuízo atual (posições abertas) - usado para SL/TP")
        print("• PnL REALIZADO: Lucro/prejuízo final (após fechar posição) - para estatísticas")
        print("• O bot monitora PnL NÃO REALIZADO (correto para SL/TP)")
        print("• PnL REALIZADO é registrado apenas após fechamento")
        
        # Verificar ambiente
        env = os.getenv("GATE_ENVIRONMENT", "live")
        if env == "live":
            print("\n⚠️ ATENÇÃO: Conta REAL!")
            print("O vigia irá fechar posições automaticamente!")
            confirm = input("Digite 'CONFIRMO' para continuar: ").strip()
            if confirm.upper() != "CONFIRMO":
                print("❌ Operação cancelada")
                return

        # Inicializar
        watcher = EmergencyWatcherCorrected()
        success = await watcher.initialize() # Chama initialize que faz setup inicial de PM e instrumentos
        if not success:
            print("❌ Falha na inicialização do vigia")
            return

        print(f"\n👁️ Vigia CORRIGIDO monitorando {len(watcher.positions_data)} posições...")
        if watcher.telegram.enabled:
            print("📱 Notificações Telegram ATIVAS - Você será avisado de todas as ações!")
        else:
            print("⚠️ Notificações Telegram DESABILITADAS - Configure TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no .env")
        print("Pressione Ctrl+C para parar e opcionalmente fechar todas as posições")
        print("=" * 80)
        
        # Iniciar vigia
        await watcher.run_watcher()

    except KeyboardInterrupt:
        print("\n⚠️ INTERRUPÇÃO DETECTADA!")
        if watcher and watcher.positions_data:
            print(f"Existem {len(watcher.positions_data)} posições sendo monitoradas.")
            confirm = input("Fechar TODAS as posições imediatamente? (S/N): ").strip().upper()
            if confirm == 'S':
                print("🛑 Fechando todas as posições...")
                closed = await watcher.emergency_close_all_corrected()
                print(f"✅ {closed} posições fechadas via emergência.")
            else:
                print("ℹ️ Posições mantidas abertas.")
        else:
            print("ℹ️ Nenhuma posição aberta.")
            
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        logger.error(f"❌ Erro crítico: {e}", exc_info=True)
    finally:
        if watcher: # Garante que o watcher é fechado mesmo em caso de erro
            await watcher.close()
        print("\n🏁 Vigia de emergência finalizado.")

if __name__ == "__main__":
    # [REPARO] Carregar .env antes de tudo para garantir que o TelegramNotifier inicialize corretamente
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())