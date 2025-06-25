#!/usr/bin/env python3
"""
Gate.io API Client - Versão 100% Compatível com main.py Corrigido
Focado em FUTURES com correções para tratamento de HTTP 201 e respostas de ordem
ADICIONADO: get_futures_tickers para compatibilidade com main.py
ADICIONADO: get_daily_volume_history para detecção de volume spike
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone, timedelta
import os
import aiohttp
from typing import List, Dict, Optional, Any, Callable, Union
from urllib.parse import urlencode
import websockets
import pandas as pd # Adicione esta linha

# Carregar variáveis do .env
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gate_api')

class GateAPI:
    """
    API Gate.io 100% compatível com main.py corrigido
    Carrega credenciais diretamente do .env
    """

    def __init__(self):
        self.api_key = os.getenv("GATE_API_KEY")
        self.api_secret = os.getenv("GATE_API_SECRET")
        self.environment = os.getenv("GATE_ENVIRONMENT", "live")

        if not all([self.api_key, self.api_secret]):
            missing = []
            if not self.api_key: missing.append("GATE_API_KEY")
            if not self.api_secret: missing.append("GATE_API_SECRET")
            raise ValueError(f"Credenciais Gate.io não encontradas no .env: {', '.join(missing)}")

        logger.info(f"✅ Credenciais Gate.io carregadas do .env (ambiente: {self.environment})")

        if self.environment == 'testnet':
            self.rest_url = "https://fx-api-testnet.gateio.ws"
            self.public_ws_url = "wss://ws-testnet.gate.io/v4/ws/futures/usdt"
            self.private_ws_url = "wss://ws-testnet.gate.io/v4/ws/futures/usdt"
        else:
            self.rest_url = "https://fx-api.gateio.ws"
            self.public_ws_url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
            self.private_ws_url = "wss://fx-ws.gateio.ws/v4/ws/usdt"

        self.base_headers = {
            'Content-Type': 'application/json',
            'KEY': self.api_key,
        }

        # A instância de GateWebSocketClient é criada aqui e passada para o DataCollector
        self.ws_client = self.GateWebSocketClient(
            parent_api=self,
            ws_url=self.public_ws_url,
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        self.rest_session: Optional[aiohttp.ClientSession] = None

        self._ws_account_data: Dict[str, Any] = {}
        self._ws_positions_data: Dict[str, Any] = {} # {contract: position_data}
        self._ws_orders_data: Dict[str, Any] = {}
        self._ws_tickers_data: Dict[str, Any] = {} # {contract: ticker_data}
        self._ws_instruments_data: List[Dict] = []

        self._last_ws_account_update: float = 0
        self._last_ws_positions_update: float = 0
        self._last_ws_orders_update: float = 0
        self._last_ws_tickers_update: float = 0
        self._last_ws_instruments_update: float = 0

        self._symbol_filters_cache: Dict[str, Dict] = {}
        self._instruments_cache: List[Dict] = [] # Cached list of all instruments
        self._cache_expiry = 300  # 5 minutos

        logger.info(f"🚀 API Gate.io FUTUROS inicializada - MODO WEBSOCKET (ambiente: {self.environment})")

    async def close(self):
        """Fecha todas as conexões ativas"""
        await self.ws_client.stop()
        if self.rest_session:
            await self.rest_session.close()
            self.rest_session = None # Reset session
        logger.info("Recursos da API Gate.io liberados")

    async def _handle_ws_account_update(self, message: Dict[str, Any]):
        """Processa updates de conta via WebSocket"""
        # Exemplo de payload: {"time":1606553885,"channel":"futures.balances","event":"update","result":{"total":"100","available":"90","unrealised_pnl":"10"}}
        if 'result' in message and isinstance(message['result'], dict):
            self._ws_account_data = message['result']
            self._last_ws_account_update = time.time()
            logger.debug(f"💰 WS Account Update: {json.dumps(self._ws_account_data)}")

    async def _handle_ws_positions_update(self, message: Dict[str, Any]):
        """Processa updates de posições via WebSocket"""
        # Exemplo de payload: {"time":1606553885,"channel":"futures.positions","event":"update","result":[{"contract":"BTC_USDT","size":"1",...}]}
        if 'result' in message and isinstance(message['result'], list):
            for pos in message['result']:
                contract = pos.get('contract')
                if contract:
                    # Gate.io envia posições com size 0 quando são fechadas
                    if float(pos.get('size', '0.0')) == 0:
                        if contract in self._ws_positions_data:
                            del self._ws_positions_data[contract]
                            logger.debug(f"📊 WS Position {contract} (size 0) removida.")
                    else:
                        self._ws_positions_data[contract] = pos
            self._last_ws_positions_update = time.time()
            logger.debug(f"📊 WS Positions: {len(self._ws_positions_data)} posições ativas.")

    async def _handle_ws_orders_update(self, message: Dict[str, Any]):
        """Processa updates de ordens via WebSocket"""
        # Exemplo de payload: {"time":1606553885,"channel":"futures.orders","event":"update","result":[{"id":"123","contract":"BTC_USDT",...}]}
        if 'result' in message and isinstance(message['result'], list):
            for order in message['result']:
                order_id = order.get('id')
                if order_id:
                    # Se a ordem está finalizada, remova-a do cache para economizar memória
                    if order.get('status') in ['finished', 'cancelled', 'failed']:
                        if str(order_id) in self._ws_orders_data:
                            del self._ws_orders_data[str(order_id)]
                            logger.debug(f"📋 WS Order {order_id} ({order.get('status')}) removida.")
                    else:
                        self._ws_orders_data[str(order_id)] = order
            self._last_ws_orders_update = time.time()
            logger.debug(f"📋 WS Orders: {len(self._ws_orders_data)} ordens.")

    async def _handle_ws_tickers_update(self, message: Dict[str, Any]):
        """Processa updates de preços/tickers via WebSocket"""
        # Exemplo de payload: {"time":1606553885,"channel":"futures.tickers","event":"update","result":{"contract":"BTC_USDT","last":"10000",...}}
        if 'result' in message and isinstance(message['result'], dict):
            ticker = message['result']
            contract = ticker.get('contract')
            if contract:
                self._ws_tickers_data[contract] = ticker
            self._last_ws_tickers_update = time.time()
            logger.debug(f"💹 WS Ticker: {contract} (last: {ticker.get('last')}) atualizado.")

    async def start_websockets(self):
        """Inicia WebSocket e subscreve a todos os canais necessários"""
        if not self.ws_client.connected:
            await self.ws_client.start()

        self.ws_client.add_callback("futures.balances", self._handle_ws_account_update)
        self.ws_client.add_callback("futures.positions", self._handle_ws_positions_update)
        self.ws_client.add_callback("futures.orders", self._handle_ws_orders_update)
        self.ws_client.add_callback("futures.tickers", self._handle_ws_tickers_update)

        # Assinar canais privados (requer autenticação)
        await self.ws_client.subscribe("futures.balances", "subscribe", ["USDT"], auth_required=True)
        await self.ws_client.subscribe("futures.positions", "subscribe", ["!all"], auth_required=True)
        await self.ws_client.subscribe("futures.orders", "subscribe", ["!all"], auth_required=True)

        # Subscrição a tickers padrão (BTC_USDT, ETH_USDT) para ter algum dado inicial
        await self.ws_client.subscribe_ticker(['BTC_USDT', 'ETH_USDT'])

        logger.info("🕐 Aguardando dados iniciais do WebSocket (5s)...")
        await asyncio.sleep(5)
        logger.info("✅ Aguardo inicial do WebSocket concluído.")

    async def get_all_spot_balances(self) -> List[Dict]:
        """Retorna o saldo de Futuros USDT. Garante retorno confiável mesmo se WebSocket não estiver populado."""
        # Preferência por dados de WS se recentes
        if self._ws_account_data and (time.time() - self._last_ws_account_update) < 15: # Reduzido para 15s
            try:
                available = float(self._ws_account_data.get('available', '0.0'))
                total = float(self._ws_account_data.get('total', '0.0'))
                unrealized_pnl = float(self._ws_account_data.get('unrealised_pnl', '0.0'))
                logger.debug(f"✅ Saldo USDT para futuros obtido via WebSocket (cache): {total:.2f}")
                return [{
                    'asset': 'USDT', 'free': available, 'locked': total - available,
                    'equity': total, 'unrealPnl': unrealized_pnl
                }]
            except Exception as e:
                logger.warning(f"⚠️ Erro processando saldo via WebSocket (cache): {e}. Tentando REST.", exc_info=True)

        logger.info("⏳ Aguardando dados de saldo via WebSocket ou tentando via REST...")
        try:
            # A API da Gate.io retorna /futures/usdt/accounts
            response = await self._rest_request('GET', '/futures/usdt/accounts', auth_required=True)
            if response.get('success', False) and isinstance(response.get('data'), dict):
                data = response['data']
                available = float(data.get('available', '0.0'))
                total = float(data.get('total', '0.0'))
                unrealized_pnl = float(data.get('unrealised_pnl', '0.0'))
                # Atualiza o cache WS com dados REST
                self._ws_account_data = data
                self._last_ws_account_update = time.time()
                logger.info(f"✅ Saldo USDT para futuros obtido via REST: {total:.2f}")
                return [{
                    'asset': 'USDT', 'free': available, 'locked': total - available,
                    'equity': total, 'unrealPnl': unrealized_pnl
                }]
            else:
                error_info = response.get('message', 'Unexpected response structure')
                logger.error(f"❌ Falha ao obter saldos via REST. Resposta inesperada: {error_info}")
        except Exception as e:
            logger.error(f"❌ Erro crítico ao obter saldo via REST: {e}", exc_info=True)

        logger.warning("❌ Falha total ao obter saldo de futuros. Retornando lista vazia.")
        return []

    async def get_futures_balance(self) -> dict:
        """🔥 CORRIGIDO: Retorna saldo total de USDT em Futuros (compatível com main.py)"""
        balances = await self.get_all_spot_balances()
        if balances and isinstance(balances, list) and len(balances) > 0:
            balance = balances[0]
            # Garantir que retorna os campos esperados pelo main.py
            return {
                'free': balance.get('free', 0.0),
                'equity': balance.get('equity', 0.0),
                'total': balance.get('equity', 0.0),  # main.py espera 'equity'
                'unrealPnl': balance.get('unrealPnl', 0.0)
            }
        return {'free': 0.0, 'equity': 0.0, 'total': 0.0, 'unrealPnl': 0.0}

    async def get_open_positions_ws(self) -> List[Dict]:
        """Obtém posições de futuros via WebSocket ou REST se WS falhar"""
        # Preferência por dados de WS se recentes
        if self._ws_positions_data and (time.time() - self._last_ws_positions_update) < 15: # Reduzido para 15s
            active_positions = []
            for pos in self._ws_positions_data.values():
                try:
                    pos_size = float(pos.get('size', '0.0'))
                    if abs(pos_size) > 0: # Apenas posições com size > 0
                        active_positions.append(pos)
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao processar posição do WS: {pos} - {e}", exc_info=True)
                    continue
            logger.debug(f"✅ Posições abertas obtidas via WebSocket: {len(active_positions)}")
            return active_positions
        
        logger.warning("❌ Dados de posições não disponíveis via WebSocket. Tentando via REST...")
        try:
            response = await self._rest_request('GET', '/futures/usdt/positions', auth_required=True)
            if response.get('success', False) and isinstance(response.get('data'), list):
                data = response['data']
                active_positions = []
                for pos in data:
                    try:
                        pos_size = float(pos.get('size', '0.0'))
                        if abs(pos_size) > 0:
                            active_positions.append(pos)
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao processar posição do REST: {pos} - {e}", exc_info=True)
                        continue
                logger.info(f"✅ Posições abertas obtidas via REST: {len(active_positions)}")
                # Atualiza o cache WS com dados REST
                self._ws_positions_data = {pos.get('contract'): pos for pos in active_positions if pos.get('contract')}
                self._last_ws_positions_update = time.time()
                return active_positions
            else:
                error_info = response.get('message', 'Unexpected response structure')
                logger.error(f"❌ Falha ao obter posições via REST. Resposta: {error_info}")
                return []
        except Exception as e:
            logger.error(f"❌ Erro crítico ao obter posições via REST: {e}", exc_info=True)
            return []

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual via WebSocket ou REST"""
        # Preferência por dados de WS se recentes
        ticker = self._ws_tickers_data.get(symbol)
        if ticker and 'last' in ticker and (time.time() - self._last_ws_tickers_update) < 10: # 10s de cache
            try:
                price = float(ticker['last'])
                if price > 0: return price
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Preço '{ticker.get('last')}' de {symbol} do WS não é float válido.")
                pass

        logger.debug(f"⏳ Ticker para {symbol} não disponível ou obsoleto no WS. Tentando subscrever e/ou REST.")
        if self.ws_client and self.ws_client.connected:
            # Subscrever o ticker se ainda não estiver subscrito (ou se for novo)
            # A _handle_ws_tickers_update preencherá _ws_tickers_data
            await self.ws_client.subscribe_ticker([symbol])
            await asyncio.sleep(0.5) # Pequeno atraso para o WS processar o subscribe e talvez enviar um update

            # Tentar novamente do cache WS após o subscribe
            ticker = self._ws_tickers_data.get(symbol)
            if ticker and 'last' in ticker:
                try:
                    price = float(ticker['last'])
                    if price > 0: return price
                except (ValueError, TypeError):
                    pass # Fallback para REST
        else:
            logger.warning("WebSocket não está conectado. Não foi possível subscrever o ticker.")

        logger.warning(f"⚠️ Preço de {symbol} não disponível via WebSocket. Tentando via REST...")
        try:
            response = await self._rest_request('GET', '/futures/usdt/tickers', auth_required=False, params={'contract': symbol})
            if response.get('success', False) and isinstance(response.get('data'), list) and len(response['data']) > 0 and response['data'][0].get('last'):
                price = float(response['data'][0]['last'])
                if price > 0:
                    # Atualiza o cache WS com dados REST para uso futuro
                    self._ws_tickers_data[symbol] = response['data'][0]
                    self._last_ws_tickers_update = time.time()
                    return price
            else:
                error_info = response.get('message', 'Unexpected response structure')
                logger.warning(f"❌ Preço de {symbol} não disponível via REST ou inválido. Resposta: {error_info}")
        except Exception as e:
            logger.error(f"❌ Erro REST ao obter preço para {symbol}: {e}", exc_info=True)

        logger.warning(f"❌ Preço de {symbol} não disponível via REST ou inválido. Retornando None.")
        return None

    async def get_current_prices_bulk(self, symbols: List[str]) -> Dict[str, float]:
        """Obtém múltiplos preços via WebSocket (cache) ou REST em lote para o que faltar"""
        prices = {}
        missing_symbols = []

        # Tenta obter preços do cache WebSocket primeiro
        for symbol in symbols:
            ticker = self._ws_tickers_data.get(symbol)
            if ticker and 'last' in ticker and (time.time() - self._last_ws_tickers_update) < 10: # 10s de cache
                try:
                    price = float(ticker['last'])
                    if price > 0: prices[symbol] = price
                    else: missing_symbols.append(symbol)
                except (ValueError, TypeError):
                    missing_symbols.append(symbol)
            else:
                missing_symbols.append(symbol)

        if missing_symbols:
            logger.debug(f"⏳ Faltando preços para {len(missing_symbols)} símbolos via WS (cache). Tentando subscrever e esperar.")
            if self.ws_client and self.ws_client.connected:
                # Subscrever os tickers ausentes para que o WS os envie
                await self.ws_client.subscribe_ticker(missing_symbols)
                await asyncio.sleep(1) # Pequeno atraso para o WS processar e enviar updates

                # Tentar novamente do cache WS após o subscribe
                still_missing = []
                for symbol in missing_symbols:
                    ticker = self._ws_tickers_data.get(symbol)
                    if ticker and 'last' in ticker:
                        try:
                            price = float(ticker['last'])
                            if price > 0: prices[symbol] = price
                            else: still_missing.append(symbol)
                        except (ValueError, TypeError): still_missing.append(symbol)
                    else: still_missing.append(symbol)
                missing_symbols = still_missing
            else:
                logger.warning("WebSocket não está conectado. Não foi possível subscrever tickers em massa.")

        if missing_symbols:
            logger.warning(f"⚠️ Nem todos os preços obtidos via WS ({len(prices)}/{len(symbols)}). Tentando REST bulk para {len(missing_symbols)} símbolos.")
            try:
                # Agora chamamos o novo método get_futures_tickers que já busca todos
                all_tickers_data = await self.get_futures_tickers()
                for ticker_data in all_tickers_data: # Já é uma lista de dicionários
                    contract = ticker_data.get('contract')
                    if contract and contract in symbols and 'last' in ticker_data:
                        try:
                            price = float(ticker_data['last'])
                            if price > 0:
                                prices[contract] = price
                                # Atualiza cache WS com dados REST
                                self._ws_tickers_data[contract] = ticker_data
                        except (ValueError, TypeError): pass
            except Exception as e:
                logger.error(f"❌ Erro REST bulk ao obter precios: {e}", exc_info=True)

        logger.debug(f"💹 {len(prices)}/{len(symbols)} preços obtidos (WS/REST).")
        return prices

    async def _init_rest_session(self):
        """Inicializa sessão REST apenas quando necessário e a reutiliza."""
        if self.rest_session is None or self.rest_session.closed:
            self.rest_session = aiohttp.ClientSession()

    def _sign_request(self, method: str, endpoint: str, query_string: str = '', body: str = '') -> dict:
        """Gera headers assinados para REST Gate.io"""
        timestamp = str(int(time.time()))
        endpoint_for_signature = endpoint
        # Garante que o endpoint começa com /api/v4 para a assinatura
        if not endpoint.startswith('/api/v4'):
            endpoint_for_signature = '/api/v4' + endpoint

        message = f"{method}\n{endpoint_for_signature}\n{query_string}\n{hashlib.sha512(body.encode('utf-8')).hexdigest()}\n{timestamp}"

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()

        headers = self.base_headers.copy()
        headers.update({
            'Timestamp': timestamp,
            'SIGN': signature
        })
        return headers

    async def _rest_request(self, method: str, endpoint: str, params: Optional[Union[Dict, List]] = None, auth_required: bool = True) -> Dict[str, Any]:
        """🔥 CORRIGIDO: Executa requisição REST com tratamento correto de HTTP 200 e 201 e retorno padronizado"""
        await self._init_rest_session()

        # Adicionar um pequeno atraso antes de cada requisição REST
        # Ajuste este valor conforme a necessidade e os limites da Gate.io
        await asyncio.sleep(0.1) # Por exemplo, 100ms de atraso

        try:
            full_url = f"{self.rest_url}/api/v4{endpoint}"

            query_string = ''
            body_for_sign = ''
            json_data = None

            if method.upper() == 'GET':
                if params:
                    query_string = urlencode(params)
                    full_url = f"{full_url}?{query_string}"
            else: # POST, PUT, DELETE
                if isinstance(params, dict):
                    body_for_sign = json.dumps(params)
                    json_data = params
                elif isinstance(params, list): # Handle list body if API expects it (e.g. batch orders)
                    body_for_sign = json.dumps(params)
                    json_data = params
                else: # If params is None or other types, treat as empty body
                    body_for_sign = ''
                    json_data = None


            headers = self._sign_request(method, endpoint, query_string, body_for_sign) if auth_required else {'Content-Type': 'application/json'}

            async with self.rest_session.request(method, full_url, headers=headers, json=json_data, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response_text = await response.text()

                try:
                    response_json = json.loads(response_text)

                    # Success cases: 200 OK, 201 Created
                    if response.status in [200, 201]:
                        # Log específico para 201 (Created) que indica ordem executada
                        if response.status == 201 and isinstance(response_json, dict):
                            status = response_json.get('status', 'unknown')
                            # Aprimorado para logs de ordem mais claros
                            if 'id' in response_json and 'contract' in response_json:
                                logger.info(f"✅ REST Sucesso (HTTP 201 - Ordem Criada/Preenchida): {response_json.get('contract')} - ID: {response_json.get('id')}. Status: {status}")
                            else:
                                logger.info(f"✅ REST Sucesso (HTTP 201 - Recurso Criado): {endpoint}")
                        else: # HTTP 200
                            logger.debug(f"✅ REST Sucesso (HTTP {response.status}): {endpoint}")

                        return {'success': True, 'data': response_json}
                    elif response.status == 429: # Tratar explicitamente o erro de limite de taxa
                        logger.error(f"❌ REST Error 429 (TOO_MANY_REQUESTS): {response_json.get('message', 'Request Rate Limit Exceeded')} for {endpoint}. Full response: {response_json}")
                        return {'success': False, 'code': 'TOO_MANY_REQUESTS', 'message': response_json.get('message', 'Request Rate Limit Exceeded'), 'full_response': response_json}
                    else:
                        # Error cases: 4xx, 5xx
                        if isinstance(response_json, dict):
                            code = response_json.get('label', str(response.status))
                            message = response_json.get('message', response_text)
                            logger.error(f"❌ REST Error {response.status} ({code}): {message} for {endpoint}. Full response: {response_json}")
                            return {'success': False, 'code': code, 'message': message, 'full_response': response_json}
                        else:
                            logger.error(f"❌ REST Error {response.status}: Unexpected response format for error. Endpoint: {endpoint}. Full response: {response_text}")
                            return {'success': False, 'code': str(response.status), 'message': response_text, 'full_response': response_text}

                except json.JSONDecodeError:
                    logger.error(f"❌ Failed to decode JSON (status: {response.status}): {response_text}. Endpoint: {endpoint}")
                    return {'success': False, 'code': 'JSON_DECODE_ERROR', 'message': response_text, 'full_response': response_text}

        except asyncio.TimeoutError:
            logger.error(f"❌ REST Timeout Error for {method} {endpoint}")
            return {'success': False, 'code': 'TIMEOUT', 'message': 'Request timed out'}
        except aiohttp.ClientError as e:
            logger.error(f"❌ REST Client Error for {method} {endpoint}: {e}", exc_info=True)
            return {'success': False, 'code': 'CLIENT_ERROR', 'message': str(e)}
        except Exception as e:
            logger.error(f"❌ Erro REST {method} {endpoint}: {e}", exc_info=True)
            return {'success': False, 'code': 'UNKNOWN_ERROR', 'message': str(e)}

    # 🔥 NOVO MÉTODO (renomeado e adaptado): get_futures_tickers
    async def get_futures_tickers(self) -> List[Dict]:
        """
        Obtém estatísticas de 24h para todos os contratos de futuros USDT.
        Retorna uma lista de dicionários, cada um contendo 'contract', 'vol_usdt_24h', 'last', etc.
        Compatível com o que main.py espera.
        """
        try:
            response = await self._rest_request('GET', '/futures/usdt/tickers', auth_required=False)

            tickers_list = []
            if response.get('success', False) and isinstance(response.get('data'), list):
                data = response['data']
                if logger.level <= logging.DEBUG:
                    logger.debug(f"DEBUG - Raw /futures/usdt/tickers response (first 2): {json.dumps(data[:min(len(data), 2)], indent=2)} ... (truncated)")

                for ticker in data:
                    contract = ticker.get('contract')
                    if contract and isinstance(contract, str):
                        try:
                            # Renomeia 'volume_24h_quote' para 'vol_usdt_24h' para compatibilidade com main.py
                            # Certifica-se de que 'last' e 'contract' existem
                            formatted_ticker = {
                                'contract': contract,
                                'vol_usdt_24h': float(ticker.get('volume_24h_quote', '0.0')),
                                'last': float(ticker.get('last', '0.0'))
                            }
                            # Adicionar outros campos relevantes se necessário pelo main.py/DataCollector
                            for key in ['change_24h', 'high_24h', 'low_24h', 'open_interest', 'funding_rate', 'mark_price', 'index_price']:
                                if key in ticker:
                                    formatted_ticker[key] = ticker[key]

                            tickers_list.append(formatted_ticker)
                        except (ValueError, TypeError) as conv_err:
                            logger.debug(f"⏭️ {contract}: Erro de conversão para volume/preço/etc. 24h: {conv_err} - Item: {ticker}")
                            continue
            logger.info(f"✅ {len(tickers_list)} tickers de futuros obtidos.")
            return tickers_list
        except Exception as e:
            logger.error(f"❌ Erro obtendo tickers de futuros: {e}", exc_info=True)
            return []

    # 🔥 NOVO MÉTODO: get_daily_volume_history
    async def get_daily_volume_history(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Obtém histórico de volume diário para um símbolo.
        Retorna um DataFrame do pandas com 'quote_asset_volume' (volume em USDT) para os últimos 'lookback_days'.
        """
        try:
            # Calcular o timestamp de início para os últimos N dias
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=lookback_days + 1) # +1 para garantir dados suficientes

            # A Gate.io tem um limite de 1000 klines por chamada.
            # Um dia tem 1440 minutos. Para 7 dias, precisamos de mais de 1000 klines, então teremos que iterar ou usar intervalo diário.
            # O intervalo '1d' seria o ideal, mas a Gate.io não tem '1d' para futuros, apenas '1m', '5m', '15m', '30m', '1h', '4h', '8h', '1d' (para spot).
            # Para futuros, o endpoint candlesticks suporta: 1m, 5m, 15m, 30m, 1h, 4h, 8h, 1d (sim, eles têm '1d' para futuros também, pela doc v4)
            
            # Usando intervalo '1d' para simplificar o cálculo do volume diário
            params = {
                'contract': symbol,
                'interval': '1d', # Intervalo de 1 dia
                'limit': lookback_days # Número de dias a buscar
            }
            response = await self._rest_request('GET', '/futures/usdt/candlesticks', params, auth_required=False)

            if response.get('success', False) and isinstance(response.get('data'), list):
                df = pd.DataFrame(response['data'])
                if df.empty:
                    logger.warning(f"⚠️ Nenhum dado de candlestick diário para {symbol}.")
                    return None
                
                # As colunas são [timestamp, open, high, low, close, volume, amount]
                # 'amount' é o volume em quote currency (USDT para pares _USDT)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df = df.set_index('timestamp')
                df['quote_asset_volume'] = pd.to_numeric(df['quote_asset_volume'], errors='coerce')
                
                # Filtrar apenas os dias relevantes
                df = df[df.index >= start_time]
                
                logger.debug(f"📈 Histórico de volume diário para {symbol} coletado: {len(df)} dias.")
                return df[['quote_asset_volume']] # Retorna apenas a coluna de volume

            logger.error(f"❌ Falha ao obter histórico de volume diário para {symbol}. Resposta: {response.get('message', 'Unknown error')}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro em get_daily_volume_history para {symbol}: {e}", exc_info=True)
            return None


    # 🔥 NOVO MÉTODO: set_leverage
    async def set_leverage(self, contract: str, leverage: int) -> bool:
        """
        Define a alavancagem para um contrato de futuros específico.
        A Gate.io permite definir a alavancagem por contrato/par.
        Endpoint: PUT /futures/usdt/positions/{contract} (para uma posição existente)
        ou POST /futures/usdt/settings (para o par em geral - mais complexo)
        Vamos usar o endpoint de posição, que também ajusta a alavancagem se a posição estiver aberta.
        """
        if not isinstance(leverage, int) or leverage <= 0:
            logger.error(f"❌ Alavancagem inválida: {leverage}. Deve ser um inteiro positivo.")
            return False

        try:
            # Primeiro, tente obter a posição atual para o contrato
            # Isso é necessário porque o endpoint PUT /positions/{contract} espera dados da posição
            # A Gate.io permite usar o endpoint GET /futures/usdt/positions para obter todas
            # e filtrar, ou GET /futures/usdt/positions/{contract} para uma específica.
            # A segunda é mais eficiente se a API for bem implementada.
            
            # Tentativa de obter a posição diretamente
            response_position = await self._rest_request('GET', f'/futures/usdt/positions', auth_required=True, params={'contract': contract})
            
            current_position_data = None
            if response_position.get('success', False) and isinstance(response_position.get('data'), list):
                # A API retorna uma lista, mesmo que com um item para um contrato específico
                for pos in response_position['data']:
                    if pos.get('contract') == contract:
                        current_position_data = pos
                        break
            
            if not current_position_data:
                logger.warning(f"⚠️ Não foi possível obter dados da posição para {contract} para definir alavancagem. Resposta: {response_position}. Provavelmente nenhuma posição aberta. Não definiremos alavancagem via PUT /positions.")
                # Se não há posição aberta, a Gate.io permite definir a alavancagem Padrão para o par via
                # POST /futures/usdt/settings
                # O endpoint é mais para ajustes de alavancagem para um usuário/par, não para uma posição específica.
                # Se main.py espera que 'set_leverage' defina para uma posição, e não existe,
                # então o método aqui falha.
                logger.error(f"❌ Não foi possível definir alavancagem para {contract} para {leverage}x: Nenhuma posição aberta para ajustar.")
                return False

            # Criar payload com a nova alavancagem (certifique-se de que a API aceita 'leverage' como string)
            # A Gate.io PUT /futures/usdt/positions/{contract} não aceita 'leverage' como parâmetro direto para alterar.
            # O campo 'leverage' é retornado, mas não é um campo editável via PUT nesse endpoint.
            # A alavancagem para uma posição é geralmente definida no momento da ordem.
            # Para alterar a alavancagem de um contrato sem ter uma posição aberta, o endpoint é /futures/usdt/settings.
            # A documentação da Gate.io para futuros (v4) sugere:
            # POST /futures/usdt/settings para definir alavancagem para o contrato.
            # Ex: {"contract": "BTC_USDT", "leverage": 10}
            
            # Vamos ajustar a chamada para usar o endpoint /settings
            # O main.py está chamando isso para definir alavancagem ANTES de place_order, então deve ser via settings.
            settings_payload = {
                "contract": contract,
                "leverage": str(leverage) # Gate.io API espera string
            }

            response = await self._rest_request(
                'POST',
                f'/futures/usdt/settings',
                params=settings_payload,
                auth_required=True
            )

            if response.get('success', False) and isinstance(response.get('data'), dict):
                # A resposta de /futures/usdt/settings/ pode não conter o campo 'leverage' diretamente,
                # apenas o sucesso da operação.
                logger.info(f"✅ Alavancagem padrão para {contract} definida para {leverage}x com sucesso via /settings.")
                return True
            else:
                error_message = response.get('message', 'Unknown error')
                logger.error(f"❌ Falha ao definir alavancagem padrão para {contract} via /settings: {error_message}. Full response: {response}")
                return False

        except Exception as e:
            logger.error(f"❌ Erro ao definir alavancagem para {contract} para {leverage}x: {e}", exc_info=True)
            return False

    async def close_position(self, symbol: str) -> dict:
        """🔥 CORRIGIDO: Fecha posição de futuros completamente na Gate.io"""
        positions = await self.get_open_positions_ws()

        target_position = None
        for pos in positions:
            if pos.get('contract') == symbol:
                target_position = pos
                break

        if not target_position:
            logger.warning(f"⚠️ Posição não encontrada para {symbol}. Não há o que fechar.")
            return {'success': False, 'code': 'POSITION_NOT_FOUND', 'message': 'Position not found'}

        try:
            current_size_str = target_position.get('size', '0.0')
            current_size = float(current_size_str)

            if current_size == 0:
                logger.info(f"ℹ️ Posição para {symbol} já está fechada (size=0).")
                return {'success': True, 'code': 'ALREADY_CLOSED', 'message': 'Position already closed'}

            closing_side = 'sell' if current_size > 0 else 'buy'
            closing_size = abs(current_size)

            logger.info(f"🔴 Submetendo ordem para fechar posição {symbol} ({'LONG' if current_size > 0 else 'SHORT'}) de size {closing_size}")

            # Note: Gate.io API para 'size' em submit_futures_order espera um valor nominal.
            # O `current_size` de uma posição pode ser a quantidade de base_currency ou o valor nominal.
            # Se for a quantidade da base_currency (ex: 1 AVAX), você pode precisar converter para o valor nominal em USDT
            # para o parâmetro `size` do place_order/submit_futures_order.
            # Para Gate.io, o campo 'size' nas posições é o "contrato" (quantidade de base_currency se for 1 contrato = 1 base).
            # No place_order, 'size' é o número de contratos a negociar.

            # Vamos assumir que `current_size` já é o número de contratos que precisa ser fechado.
            # Se 'size' em suas posições for o valor nominal em USDT, você precisaria de:
            # contracts_to_close = closing_size / current_price.
            # Mas o log anterior mostra 'size': 1 para AVAX, que é o número de contratos.

            result = await self.place_order(
                symbol=symbol,
                side=closing_side,
                order_type="market",
                size=closing_size, # Passa a quantidade de contratos a fechar
                reduce_only=True
            )

            # 🔥 VERIFICAÇÃO DE SUCESSO APRIMORADA PARA close_position
            if result and result.get('success', False):
                logger.info(f"✅ Ordem de fechamento para {symbol} submetida com sucesso. ID: {result.get('order_id')}")
                # Retorna um dicionário de sucesso, não o 'result' bruto que pode ser grande
                return {'success': True, 'message': 'Close order submitted successfully', 'order_id': result.get('order_id')}
            else:
                error_message = result.get('message', 'Unknown error') if isinstance(result, dict) else str(result)
                logger.error(f"❌ Falha ao submeter ordem de fechamento para {symbol}: {error_message}. Resposta completa: {result}")
                return {'success': False, 'code': 'CLOSE_ORDER_FAILURE', 'message': error_message}

        except Exception as e:
            logger.error(f"❌ Erro ao tentar fechar posição para {symbol}: {e}", exc_info=True)
            return {'success': False, 'code': 'EXCEPTION_CLOSING', 'message': str(e)}

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        """Cancela ordem"""
        endpoint = f"/futures/usdt/orders/{order_id}"
        # A Gate.io não exige 'contract' no endpoint DELETE para ordem, mas pode ser útil para log/contexto
        # params = {'contract': symbol} if symbol else {}

        result = await self._rest_request('DELETE', endpoint, auth_required=True) # Removido params
        # 🔥 VERIFICAÇÃO DE SUCESSO APRIMORADA PARA CANCEL_ORDER
        if result.get('success', False) and isinstance(result.get('data'), dict) and result['data'].get('id') == order_id:
            logger.info(f"✅ Ordem {order_id} cancelada com sucesso.")
            return {'success': True, 'order_id': order_id, 'response': result['data']}
        else:
            error_message = result.get('message', 'Unknown error') if isinstance(result, dict) else str(result)
            logger.error(f"❌ Falha ao cancelar ordem {order_id}: {error_message}. Resposta completa: {result}")
            return {'success': False, 'code': 'CANCEL_FAILURE', 'message': error_message, 'full_response': result}

    async def get_instruments_info(self) -> List[Dict]:
        """Obtém informações de instrumentos USDT futures"""
        now = time.time()

        if self._instruments_cache and (now - self._last_ws_instruments_update) < self._cache_expiry:
            logger.debug("✅ Retornando instrumentos FUTUROS do cache.")
            return self._instruments_cache

        logger.info("📡 Buscando informações de instrumentos FUTUROS da Gate.io via REST.")
        response = await self._rest_request('GET', '/futures/usdt/contracts', auth_required=False)

        if response.get('success', False) and isinstance(response.get('data'), list):
            self._instruments_cache = response['data']
            self._last_ws_instruments_update = now
            logger.info(f"📊 {len(self._instruments_cache)} instrumentos FUTUROS atualizados.")
            return self._instruments_cache

        logger.error(f"❌ Falha ao obter instrumentos de futuros. Resposta: {response.get('message', 'Unknown error')}. Full response: {response}")
        return []

    async def select_tradable_symbols(self, min_volume_usdt: float, min_notional: float, excluded_symbols: List[str]) -> List[str]:
        """🔥 CORRIGIDO: Seleção INTELIGENTE de símbolos com critérios muito mais relaxados"""
        instruments = await self.get_instruments_info()

        if not instruments:
            logger.error("❌ Nenhum instrumento de futuros retornado pela API")
            return []

        logger.info(f"🚀 SELEÇÃO INTELIGENTE DE FUTUROS GATE.IO (RELAXADA)")
        logger.info(f"   📊 Total instrumentos: {len(instruments)}")
        logger.info(f"   🎯 Critérios iniciais: Volume >= {min_volume_usdt:,.0f} USDT")

        # Chama o novo método get_futures_tickers
        stats_24h = await self.get_futures_tickers()
        stats_24h_map = {t['contract']: t for t in stats_24h}

        if not stats_24h_map:
            logger.warning("⚠️ tickers_data (24h stats) está vazio. A seleção por volume pode falhar.")

        tradable_symbols = await self._apply_gate_selection_criteria(
            instruments, stats_24h_map, min_volume_usdt, min_notional, excluded_symbols, "Critérios Relaxados"
        )

        if len(tradable_symbols) >= 5:  # 🔥 REDUZIDO: de 10 para 5
            logger.info(f"✅ Estratégia 1 funcionou: {len(tradable_symbols)} símbolos")
            await self._subscribe_to_symbols(tradable_symbols)
            return tradable_symbols

        logger.warning(f"⚠️ Estratégia 1 falhou ({len(tradable_symbols)} símbolos). Adaptando...")

        if stats_24h_map:
            volumes = [t['vol_usdt_24h'] for t in stats_24h if 'vol_usdt_24h' in t]
            volumes = [v for v in volumes if v > 0]
            volumes.sort(reverse=True)

            if volumes:
                # 🔥 RELAXADO: Usar volume do percentil 75 em vez de 25
                adaptive_volume = volumes[min(len(volumes) - 1, len(volumes)//4)]
                logger.info(f"🧠 Estratégia 2: Volume adaptativo {adaptive_volume:,.0f} USDT")

                tradable_symbols = await self._apply_gate_selection_criteria(
                    instruments, stats_24h_map, adaptive_volume, min_notional, excluded_symbols, "Volume Adaptativo"
                )

                if len(tradable_symbols) >= 3:  # 🔥 REDUZIDO: de 5 para 3
                    logger.info(f"✅ Estratégia 2 funcionou: {len(tradable_symbols)} símbolos")
                    await self._subscribe_to_symbols(tradable_symbols)
                    return tradable_symbols

        logger.warning(f"⚠️ Estratégia 2 falhou. Usando top símbolos por volume...")
        tradable_symbols = await self._select_top_volume_symbols_gate(instruments, stats_24h_map, 20, excluded_symbols)  # 🔥 REDUZIDO: de 50 para 20

        if len(tradable_symbols) >= 3:  # 🔥 REDUZIDO: de 5 para 3
            logger.info(f"✅ Estratégia 3 funcionou: {len(tradable_symbols)} símbolos")
            await self._subscribe_to_symbols(tradable_symbols)
            return tradable_symbols

        logger.warning(f"⚠️ Estratégia 3 falhou. Usando futuros principais...")
        tradable_symbols = await self._select_known_major_symbols_gate(instruments, excluded_symbols)

        if tradable_symbols:
            logger.info(f"✅ Estratégia 4 (fallback): {len(tradable_symbols)} futuros principais")
            await self._subscribe_to_symbols(tradable_symbols)
            return tradable_symbols

        logger.error(f"💥 TODAS as estratégias falharam! Retornando lista vazia.")
        return []

    # REMOVIDO: _get_24h_stats - AGORA É get_futures_tickers

    async def _apply_gate_selection_criteria(self, instruments: List[Dict], stats_24h: Dict,
                                           min_volume_usdt: float, min_notional: float,
                                           excluded_symbols: List[str], strategy_name: str) -> List[str]:
        """🔥 CORRIGIDO: Critérios muito mais relaxados para compatibilidade com main.py"""

        # 🔥 RELAXAMENTO EXTREMO: Threshold notional 10x maior para ser mais flexível
        threshold_notional = max(min_notional, 5000.0)  # MÍNIMO 5000 USDT agora

        logger.info(f"🎯 {strategy_name}: Volume >= {min_volume_usdt:,.0f}, Max Instrument Order Min <= {threshold_notional:,.0f}")

        selected = []
        processed = 0
        debug_rejections = {
            'volume_low': 0,
            'price_invalid': 0,
            'notional_exceeded': 0,
            'excluded': 0,
            'not_usdt': 0,
            'not_tradable': 0,
            'other': 0
        }

        for inst in instruments:
            name = inst.get('name')
            if not name:
                debug_rejections['other'] += 1
                continue

            if name in excluded_symbols:
                debug_rejections['excluded'] += 1
                logger.debug(f"⏭️ {name}: Na lista de exclusão")
                continue

            if not name.endswith('_USDT'):
                debug_rejections['not_usdt'] += 1
                logger.debug(f"⏭️ {name}: Não é par _USDT")
                continue

            trade_status = inst.get('trade_status')
            if trade_status is not None and trade_status != 'tradable':
                debug_rejections['not_tradable'] += 1
                logger.debug(f"⏭️ {name}: Status não é 'tradable' ({inst.get('trade_status')})")
                continue

            processed += 1

            try:
                stats = stats_24h.get(name, {})
                # Usar 'vol_usdt_24h' do get_futures_tickers
                volume_24h_quote = float(stats.get('vol_usdt_24h', '0.0'))

                if volume_24h_quote < min_volume_usdt:
                    debug_rejections['volume_low'] += 1
                    logger.debug(f"⏭️ {name}: Volume ({volume_24h_quote:,.0f}) < mínimo ({min_volume_usdt:,.0f})")
                    continue

                order_size_min = float(inst.get('order_size_min', '1.0'))
                last_price = float(stats.get('last', '0.0'))

                if last_price <= 0:
                    debug_rejections['price_invalid'] += 1
                    logger.debug(f"⏭️ {name}: Preço inválido ({last_price})")
                    continue

                # 🔥 CÁLCULO CORRETO: notional mínimo da ordem
                min_order_notional = order_size_min * last_price

                # 🔥 FLEXIBILIDADE EXTREMA: Para símbolos de alto volume, ser ainda mais flexível
                if volume_24h_quote > min_volume_usdt * 20:  # 20x maior que mínimo
                    effective_threshold = threshold_notional * 5  # 5x o threshold
                    logger.debug(f"📈 {name}: Volume muito alto, threshold aumentado para {effective_threshold:,.0f}")
                elif volume_24h_quote > min_volume_usdt * 5:  # 5x maior que mínimo
                    effective_threshold = threshold_notional * 2  # 2x o threshold
                    logger.debug(f"📈 {name}: Alto volume, threshold aumentado para {effective_threshold:,.0f}")
                else:
                    effective_threshold = threshold_notional

                if min_order_notional > effective_threshold:
                    debug_rejections['notional_exceeded'] += 1
                    logger.debug(f"⏭️ {name}: Notional ({min_order_notional:,.2f}) > limite ({effective_threshold:,.0f})")
                    continue

                # 🔥 APROVADO!
                selected.append(name)
                logger.debug(f"✅ {name}: APROVADO - Vol: {volume_24h_quote:,.0f}, Notional: {min_order_notional:,.2f}, Preço: {last_price:.6f}")

            except Exception as e:
                debug_rejections['other'] += 1
                logger.debug(f"❌ Erro processando {name}: {e}")
                continue

        # Log detalhado de rejeições para debug
        logger.info(f"📋 {strategy_name}: {len(selected)}/{processed} símbolos aprovados")
        logger.info(f"📊 Rejeições detalhadas:")
        logger.info(f"   • Volume baixo: {debug_rejections['volume_low']}")
        logger.info(f"   • Preço inválido: {debug_rejections['price_invalid']}")
        logger.info(f"   • Notional excedido: {debug_rejections['notional_exceeded']}")
        logger.info(f"   • Na lista de exclusão: {debug_rejections['excluded']}")
        logger.info(f"   • Não é _USDT: {debug_rejections['not_usdt']}")
        logger.info(f"   • Não tradable: {debug_rejections['not_tradable']}")
        logger.info(f"   • Outros erros: {debug_rejections['other']}")

        # 🔥 SE AINDA POUCOS SÍMBOLOS: Mostrar os melhores rejeitados para análise
        if len(selected) < 3:
            logger.warning(f"⚠️ Apenas {len(selected)} símbolos aprovados. Top rejeitados por notional:")

            rejected_candidates = []
            for inst in instruments[:50]:  # Top 50 por ordem
                name = inst.get('name')
                if (name and name.endswith('_USDT') and
                    inst.get('trade_status') == 'tradable' and
                    name not in excluded_symbols and
                    name not in selected):

                    stats = stats_24h.get(name, {})
                    volume_24h_quote = float(stats.get('vol_usdt_24h', '0.0'))
                    last_price = float(stats.get('last', '0.0'))
                    order_size_min = float(inst.get('order_size_min', '1.0'))

                    if volume_24h_quote >= min_volume_usdt and last_price > 0:
                        min_notional = order_size_min * last_price
                        rejected_candidates.append((name, min_notional, volume_24h_quote))

            # Ordenar por volume (maiores primeiro)
            rejected_candidates.sort(key=lambda x: x[2], reverse=True)

            logger.warning("   Top 5 rejeitados (ordenados por volume):")
            for name, notional, volume in rejected_candidates[:5]:
                logger.warning(f"     {name}: Notional {notional:,.0f} USDT, Volume {volume:,.0f} USDT")

        return selected

    async def _select_top_volume_symbols_gate(self, instruments: List[Dict], stats_24h: Dict,
                                            top_count: int, excluded_symbols: List[str]) -> List[str]:
        """Seleciona top símbolos Gate.io por volume"""
        logger.info(f"🔝 Selecionando top {top_count} futuros Gate.io por volume...")

        symbol_volumes = []

        for inst in instruments:
            name = inst.get('name')
            if not name or not name.endswith('_USDT') or name in excluded_symbols:
                continue

            trade_status = inst.get('trade_status')
            if trade_status is not None and trade_status != 'tradable':
                continue

            try:
                stats = stats_24h.get(name, {})
                # Usar 'vol_usdt_24h' do get_futures_tickers
                volume_24h_quote = float(stats.get('vol_usdt_24h', '0.0'))

                order_size_min = float(inst.get('order_size_min', '1.0'))
                last_price = float(stats.get('last', '0.0'))

                # 🔥 RELAXAMENTO: Aceitar símbolos com qualquer notional se o volume for alto
                if volume_24h_quote > 0 and last_price > 0 and order_size_min > 0:
                    symbol_volumes.append((name, volume_24h_quote))

            except Exception as e:
                logger.debug(f"Erro processando {name} para top volume: {e}", exc_info=True)
                continue

        symbol_volumes.sort(key=lambda x: x[1], reverse=True)
        selected = [symbol for symbol, volume in symbol_volumes[:top_count]]

        logger.info(f"🔝 Top futuros Gate.io selecionados: {len(selected)}")
        if selected:
            logger.info(f"   Exemplos: {selected[:min(10, len(selected))]}")

        return selected

    async def _select_known_major_symbols_gate(self, instruments: List[Dict], excluded_symbols: List[str]) -> List[str]:
        """Seleciona futuros principais conhecidos na Gate.io"""
        logger.info(f"🆘 Fallback: Selecionando futuros principais Gate.io...")

        major_futures = [
            'BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'ADA_USDT',
            'XRP_USDT', 'SOL_USDT', 'DOT_USDT', 'MATIC_USDT',
            'LINK_USDT', 'UNI_USDT', 'LTC_USDT', 'BCH_USDT',
            'ETC_USDT', 'TRX_USDT', 'AVAX_USDT', 'ATOM_USDT'
        ]

        available_instruments = {inst.get('name') for inst in instruments
                               if inst.get('name', '').endswith('_USDT')}

        selected = []
        for symbol in major_futures:
            if symbol in available_instruments and symbol not in excluded_symbols:
                selected.append(symbol)

        logger.info(f"🆘 Futuros principais Gate.io encontrados: {len(selected)}")
        if selected:
            logger.info(f"   Lista: {selected}")

        return selected

    async def _subscribe_to_symbols(self, symbols: List[str]):
        """Subscreve aos feeds de preços dos símbolos selecionados"""
        if symbols and len(symbols) > 0 and self.ws_client and self.ws_client.connected:
            try:
                await self.ws_client.subscribe("futures.tickers", "subscribe", symbols, auth_required=False)
                logger.info(f"📡 Subscrito a feeds de {len(symbols)} futuros Gate.io")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao subscrever tickers: {e}", exc_info=True)
        elif not self.ws_client or not self.ws_client.connected:
            logger.warning("Não foi possível subscrever símbolos. WebSocket não está conectado.")

    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> list:
        """Obtém klines para futuros Gate.io"""
        params = {
            'contract': symbol,
            'interval': interval,
            'limit': limit
        }
        response = await self._rest_request('GET', '/futures/usdt/candlesticks', params, auth_required=False)

        if response.get('success', False) and isinstance(response.get('data'), list):
            # Adicionar este log para inspecionar os dados brutos dos klines
            if not response['data']:
                logger.warning(f"⚠️ API retornou lista de klines VAZIA para {symbol} ({interval}, {limit}).")
            elif logger.level <= logging.DEBUG:
                logger.debug(f"DEBUG Klines Raw Data for {symbol} (first 2): {json.dumps(response['data'][:min(len(response['data']), 2)], indent=2)} ... (truncated)")
            return response['data']
        logger.error(f"❌ Falha ao obter klines {symbol}: {response.get('message', 'Unknown error')}. Full response: {response}")
        return []

    async def get_my_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Obtém histórico de trades de futuros"""
        params = {
            'contract': symbol, 'limit': limit
        }
        response = await self._rest_request('GET', '/futures/usdt/my_trades', params, auth_required=True)
        if response.get('success', False) and isinstance(response.get('data'), list):
            return response['data']
        logger.warning(f"Falha ao obter trades para {symbol}: {response.get('message', 'Unknown error')}. Full response: {response}")
        return []

    async def create_order(self, symbol: str, side: str, type_: str, quantity: float, price: Optional[float] = None) -> dict:
        """Alias para place_order"""
        return await self.place_order(symbol, side, type_, quantity, price)

    async def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Obtém filtros de símbolo para futuros Gate.io"""
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]

        instruments = await self.get_instruments_info()

        for inst in instruments:
            if inst.get('name') == symbol:
                filters = {
                    'LOT_SIZE': {
                        'minQty': float(inst.get('order_size_min', '1.0')),
                        'maxQty': float(inst.get('order_size_max', '99999999.0')),
                        'stepSize': float(inst.get('order_size_step', '1.0'))
                    },
                    'PRICE_FILTER': {
                        'tickSize': float(inst.get('order_price_round', '0.000001'))
                    },
                    'CONTRACT_INFO': {
                        'quanto_multiplier': float(inst.get('quanto_multiplier', '1.0')),
                        'leverage_min': float(inst.get('leverage_min', '1.0')),
                        'leverage_max': float(inst.get('leverage_max', '100.0'))
                    }
                }
                self._symbol_filters_cache[symbol] = filters
                return filters

        logger.warning(f"⚠️ Filtros para {symbol} não encontrados")
        return {}

    async def submit_futures_order(self, symbol: str, side: str, amount: float,
                                    price: float = None, order_type: str = "market",
                                    leverage: float = 1.0, reduce_only: bool = False) -> Dict[str, Any]:
        """Função CORRIGIDA e robusta para submeter ordens de futuros Gate.io."""
        try:
            logger.warning(f"📤 ORDEM {symbol}: {side.upper()} {amount} @ {price} (tipo: {order_type.upper()})")

            if amount <= 0: raise ValueError(f"Amount inválido: {amount}. Deve ser maior que zero.")

            if order_type.lower() == "market":
                if price is not None and price != 0:
                    logger.warning(f"⚠️ {symbol}: Preço ({price}) fornecido para ordem de mercado. Será definido como '0'.")
                price = 0.0
                tif_value = "ioc"
            elif order_type.lower() == "limit":
                if price is None or price <= 0: raise ValueError(f"Preço inválido para ordem limite: {price}. Deve ser maior que zero.")
                tif_value = "gtc"
            else: raise ValueError(f"Tipo de ordem inválido: {order_type}. Use 'market' ou 'limit'.")

            contract_info = await self.get_symbol_filters(symbol)
            if not contract_info:
                logger.error(f"❌ {symbol}: Informações do contrato não encontradas. Não é possível submeter a ordem.")
                return {'success': False, 'error': f"Informações do contrato para {symbol} não encontradas."}

            min_size = contract_info.get('LOT_SIZE', {}).get('minQty', 1.0)
            step_size = contract_info.get('LOT_SIZE', {}).get('stepSize', 1.0)

            # Garantir que a quantidade seja um múltiplo do step_size e no mínimo min_size
            if step_size > 0:
                # Quantidade final arredondada para o step_size mais próximo
                # E garantir que seja pelo menos min_size
                size_final = max(min_size, round(amount / step_size) * step_size)
            else:
                size_final = amount # Se step_size for 0, não faz arredondamento

            # Arredondamento final para o número de casas decimais do step_size
            # Isso é crucial para Gate.io que exige precisão.
            if step_size > 0:
                decimal_places = str(step_size)[::-1].find('.')
                if decimal_places == -1: decimal_places = 0 # É um número inteiro
                size_final = round(size_final, decimal_places)

            # A Gate.io espera size como um número, não necessariamente inteiro, mas com precisão
            # O exemplo no seu código anterior usava int(round(size_final)). Isso pode truncar.
            # Vamos usar float(size_final) ou apenas size_final.
            # O campo 'size' da Gate.io é o número de contratos. Para USDT perpétuos, 1 contrato é 1 base currency.
            # Ex: Para BTC_USDT, size 1 significa 1 BTC (valor nominal grande).
            # Para outros, 1 contrato pode ser 1 unidade da altcoin.
            # A menos que seja um contrato de US_Dollars (ex: BTC_USD_0326), onde size é $1.
            # A maioria dos contratos perpetuos USDT é 1 contrato = 1 base currency (ex: 1 BTC, 1 ETH).
            # Então, `amount` aqui deve ser a quantidade de cripto.

            if size_final <= 0: raise ValueError(f"Tamanho de ordem final inválido ({size_final}) após arredondamento para {symbol}.")

            order_data = {
                "contract": symbol,
                "size": size_final if side.lower() == "buy" else -size_final, # Passa como float
                "price": f"{price:.8f}" if price is not None else "0.00000000",
                "tif": tif_value,
                "auto_size": "",
                "iceberg": 0,
            }

            if reduce_only: order_data["reduce_only"] = True

            logger.info(f"📤 PAYLOAD {symbol}: {order_data}")

            endpoint = "/futures/usdt/orders"
            response = await self._rest_request("POST", endpoint, params=order_data, auth_required=True)

            # 🔥 VERIFICAÇÃO DE SUCESSO APRIMORADA PARA submit_futures_order
            if response.get('success', False) and isinstance(response.get('data'), dict):
                data = response['data']
                order_id = data.get('id')
                status = data.get('status', 'unknown')

                logger.warning(f"✅ ORDEM {symbol}: Sucesso! Order ID = {order_id}. Status: {status}")
                return {
                    'success': True, 'order_id': order_id, 'symbol': symbol, 'side': side,
                    'amount': amount, 'price': price, 'status': status, 'response': data
                }
            else:
                error_message = response.get('message', response.get('error', 'Unknown error'))
                logger.error(f"❌ Falha ao submeter ordem FUTUROS {side.upper()} {symbol}: {error_message}. Full response: {response}")
                return {
                    'success': False, 'error': error_message, 'symbol': symbol, 'side': side,
                    'amount': amount, 'price': price, 'payload_used': order_data, 'full_response': response
                }

        except ValueError as ve:
            logger.error(f"❌ Erro de validação da ordem {symbol}: {ve}")
            return {"success": False, "error": str(ve), 'symbol': symbol, 'side': side, 'amount': amount, 'price': price}
        except Exception as api_error:
            logger.error(f"❌ API ERROR ao submeter ordem para {symbol}: {api_error}", exc_info=True)
            error_msg = str(api_error).lower()
            if "invalid request body" in error_msg:
                logger.error(f"❌ {symbol}: Problema no formato do payload. Verifique os campos obrigatórios e tipos de dados.")
            elif "insufficient" in error_msg:
                logger.error(f"❌ {symbol}: Saldo insuficiente para a margem.")
            elif "size" in error_msg or "amount" in error_msg:
                logger.error(f"❌ {symbol}: Problema com o tamanho da ordem ({amount}). Verifique step size/min size do contrato.")

            return {
                'success': False, 'error': str(api_error), 'symbol': symbol, 'side': side,
                'amount': amount, 'price': price, 'payload_used': order_data if 'order_data' in locals() else 'Not available'
            }

    async def submit_futures_order_debug(self, symbol: str, side: str, amount: float,
                                     price: float = None, order_type: str = "market",
                                     leverage: float = 1.0, reduce_only: bool = False) -> Dict[str, Any]:
        """Submete ordem de futuros com debug detalhado para Gate.io"""
        # Este método não será diretamente usado pelo main.py, mas foi ajustado para
        # usar o novo formato de retorno do _rest_request
        try:
            logger.warning(f"🔍 DEBUG ORDER {symbol}: Preparando ordem...")
            logger.warning(f"🔍 DEBUG ORDER {symbol}: side={side}, amount={amount}, price={price}, type={order_type}")
            if amount <= 0: raise ValueError(f"Amount deve ser positivo: {amount}")

            if price is not None and price <= 0:
                logger.warning(f"⚠️ DEBUG ORDER {symbol}: Preço inválido {price}, convertendo para market order")
                price = None
                order_type = "market"

            instruments = await self.get_instruments_info()
            contract_info = None
            for inst in instruments:
                if inst.get('name') == symbol:
                    contract_info = inst
                    break
            if not contract_info:
                logger.warning(f"⚠️ DEBUG ORDER {symbol}: Símbolo não encontrado nos contratos")
                contract_info = {}
            logger.warning(f"🔍 DEBUG ORDER {symbol}: Contract info = {contract_info}")

            # Ajustar a quantidade com base no 'quanto_multiplier' se necessário
            # OBS: Este ajuste de quantidade deve ser feito ANTES de validar min_size/step_size
            # A Gate.io geralmente espera a quantidade de contratos. Para USDT perpétuos, 1 contrato = 1 base currency.
            # 'quanto_multiplier' é para contratos que não são 1:1, por exemplo, contratos inversos.
            # Se 'amount' já é a quantidade de BASE_CURRENCY, não precisa multiplicar.
            # No seu `submit_futures_order` acima, `amount` já é a quantidade de contratos.
            # Aqui, vamos garantir que a quantidade seja formatada corretamente como float.
            
            # Removido ajuste de multiplier aqui, pois `amount` já deve ser a quantidade de contratos esperada.
            # A precisão (casas decimais) é a parte mais importante.

            order_size_step = float(contract_info.get('order_size_step', '1.0'))
            order_size_min = float(contract_info.get('order_size_min', '1.0'))

            if order_size_step > 0:
                amount = max(order_size_min, round(amount / order_size_step) * order_size_step)
                decimal_places = str(order_size_step)[::-1].find('.')
                if decimal_places == -1: decimal_places = 0
                amount = round(amount, decimal_places)
            else:
                amount = max(order_size_min, amount) # Garante min_size se step_size for 0 ou inválido

            if amount <= 0: raise ValueError(f"Tamanho de ordem final inválido ({amount}) após arredondamento para {symbol}.")


            order_payload = {
                "contract": symbol,
                "size": amount if side.lower() == "buy" else -amount, # Passa como float, não int
                "price": str(price) if price is not None else "0",
                "tif": "ioc" if order_type.lower() == "market" else "gtc",
                "reduce_only": reduce_only,
                "iceberg": 0,
                "auto_size": ""
            }
            logger.warning(f"🔍 DEBUG ORDER {symbol}: Payload inicial = {order_payload}")

            # Ajustes finais baseados no tipo de ordem e lado
            if order_type.lower() == "market":
                order_payload.update({"price": "0", "tif": "ioc"})
            else: # limit
                if price is None: raise ValueError("Limit orders precisam de preço")
                order_payload.update({"price": f"{price:.8f}", "tif": "gtc"}) # Formata preço com 8 casas decimais

            logger.warning(f"🔍 DEBUG ORDER {symbol}: Payload final = {order_payload}")
            logger.warning(f"🔍 DEBUG ORDER {symbol}: Enviando para /futures/usdt/orders")

            endpoint = "/futures/usdt/orders"
            response = await self._rest_request("POST", endpoint, params=order_payload, auth_required=True)

            logger.warning(f"✅ DEBUG ORDER {symbol}: Resposta recebida = {response}")
            # Ajustado para o novo formato de retorno do _rest_request
            if response.get('success', False) and isinstance(response.get('data'), dict):
                data = response['data']
                return {
                    'success': True, 'order_id': data.get('id'), 'symbol': symbol, 'side': side,
                    'amount': amount, 'price': price, 'status': data.get('status', 'unknown'), 'response': data
                }
            else:
                error_message = response.get('message', response.get('error', 'Unknown error'))
                logger.error(f"❌ DEBUG ORDER {symbol}: Erro detalhado = {error_message}")
                logger.error(f"❌ DEBUG ORDER {symbol}: Payload que falhou = {locals().get('order_payload', 'N/A')}")
                return {
                    'success': False, 'error': error_message, 'symbol': symbol, 'side': side,
                    'amount': amount, 'price': price, 'full_response': response # Adicionado full_response para debug
                }

        except Exception as e:
            logger.error(f"❌ DEBUG ORDER {symbol}: Erro detalhado = {e}", exc_info=True)
            logger.error(f"❌ DEBUG ORDER {symbol}: Payload que falhou = {locals().get('order_payload', 'N/A')}")
            return {
                'success': False, 'error': str(e), 'symbol': symbol, 'side': side,
                'amount': amount, 'price': price
            }

    def validate_order_params(self, symbol: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Valida parâmetros de ordem antes de enviar"""
        try:
            contract_info = None
            if hasattr(self, '_instruments_cache') and self._instruments_cache:
                for inst in self._instruments_cache:
                    if inst.get('name') == symbol:
                        contract_info = inst
                        break
            if not contract_info:
                contract_info = {}
            logger.warning(f"🔍 VALIDATE {symbol}: Contract = {contract_info}")

            issues = []
            fixes = {}

            if 'order_size_min' in contract_info:
                min_size = float(contract_info['order_size_min'])
                if amount < min_size:
                    issues.append(f"Quantidade {amount} < mínimo {min_size}")
                    fixes['amount'] = min_size

            if 'order_size_max' in contract_info:
                max_size = float(contract_info['order_size_max'])
                if amount > max_size:
                    issues.append(f"Quantidade {amount} > máximo {max_size}")
                    fixes['amount'] = max_size

            if contract_info.get('trade_status') != 'tradable':
                issues.append(f"Contrato não está tradable: {contract_info.get('trade_status')}")

            logger.warning(f"🔍 VALIDATE {symbol}: Issues = {issues}")
            logger.warning(f"🔍 VALIDATE {symbol}: Fixes = {fixes}")

            return {
                'valid': len(issues) == 0, 'issues': issues, 'fixes': fixes, 'contract_info': contract_info
            }
        except Exception as e:
            logger.error(f"❌ VALIDATE {symbol}: Erro = {e}", exc_info=True)
            return {
                'valid': False, 'issues': [f"Erro de validação: {e}"], 'fixes': {}, 'contract_info': {}
            }

    async def test_single_order(self, symbol: str = "BTC_USDT", amount: float = 1.0):
        """Testa uma ordem individual para debug"""
        logger.warning(f"🧪 TESTE ORDER: Testando ordem para {symbol}")
        try:
            validation = self.validate_order_params(symbol, amount)
            logger.warning(f"🧪 TESTE ORDER: Validação = {validation}")

            if not validation['valid']:
                logger.warning(f"🧪 TESTE ORDER: Aplicando correções = {validation['fixes']}")
                if 'amount' in validation['fixes']: amount = validation['fixes']['amount']

            current_price = await self.get_current_price(symbol)
            logger.warning(f"🧪 TESTE ORDER: Preço atual = {current_price}")

            logger.warning(f"🧪 TESTE ORDER: Tentando market order...")
            result = await self.submit_futures_order_debug(
                symbol=symbol, side="BUY", amount=amount, order_type="market"
            )
            logger.warning(f"🧪 TESTE ORDER: Resultado = {result}")
            return result
        except Exception as e:
            logger.error(f"🧪 TESTE ORDER: Erro = {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def place_order(self, symbol: str, side: str, order_type: str, size: float, price: Optional[float] = None, reduce_only: bool = False) -> dict:
        """Função place_order CORRIGIDA para Gate.io"""
        return await self.submit_futures_order(
            symbol=symbol, side=side, amount=size, price=price, order_type=order_type, reduce_only=reduce_only
        )

    class GateWebSocketClient:
        """Cliente WebSocket otimizado para Gate.io usando o pacote 'websockets' (asyncio-native)"""

        def __init__(self, parent_api: 'GateAPI', ws_url: str, api_key: str, api_secret: str):
            self.parent_api = parent_api
            self.ws_url = ws_url
            self.api_key = api_key
            self.api_secret = api_secret
            self.connection: Optional[websockets.WebSocketClientProtocol] = None
            self.callbacks: Dict[str, List[Callable]] = {}
            self.connected = False
            self.authenticated = False
            self.reconnect_attempt = 0
            self.max_reconnect_attempts = 5
            self._message_id = 0
            self.subscribed_channels = set()
            self._listen_task: Optional[asyncio.Task] = None # Para controlar a tarefa de escuta

        async def start(self):
            """Inicia a conexão WebSocket e o loop de escuta"""
            if self.connected and self.connection:
                logger.info("WebSocket já está conectado.")
                return

            # Garantir que qualquer tarefa de escuta antiga seja cancelada
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
                try:
                    await self._listen_task # Esperar que ela seja cancelada
                except asyncio.CancelledError:
                    logger.debug("Tarefa de escuta anterior cancelada com sucesso.")
                except Exception as e:
                    logger.warning(f"Erro ao cancelar tarefa de escuta anterior: {e}")
                finally:
                    self._listen_task = None


            self.reconnect_attempt = 0
            while self.reconnect_attempt < self.max_reconnect_attempts:
                try:
                    self.connection = await websockets.connect(self.ws_url, ping_interval=30, ping_timeout=10)
                    self.connected = True
                    self.reconnect_attempt = 0
                    logger.info(f"🔗 Conectado ao WebSocket: {self.ws_url}")

                    # Iniciar a tarefa de escuta APENAS UMA VEZ
                    self._listen_task = asyncio.create_task(self._listen())
                    return
                except Exception as e:
                    self.reconnect_attempt += 1
                    logger.error(f"❌ Erro conectando ao WebSocket (tentativa {self.reconnect_attempt}/{self.max_reconnect_attempts}): {e}", exc_info=True)
                    await asyncio.sleep(min(30, 2 ** self.reconnect_attempt))

            logger.critical("💥 Falha persistente ao conectar ao WebSocket após várias tentativas.")
            self.connected = False

        async def stop(self):
            """Para o WebSocket"""
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    logger.debug("Tarefa de escuta cancelada durante o stop.")
                except Exception as e:
                    logger.warning(f"Erro ao esperar tarefa de escuta ser cancelada durante o stop: {e}")
                finally:
                    self._listen_task = None

            if self.connection:
                await self.connection.close()
                self.connection = None
            self.connected = False
            self.authenticated = False
            logger.info("🔌 WebSocket Gate.io desconectado")

        def _get_signature(self, channel: str, event: str, timestamp: int) -> str:
            """Gera assinatura HMAC-SHA512 para WS, conforme a documentação oficial para Futuros"""
            message = 'channel=%s&event=%s&time=%d' % (channel, event, timestamp)
            return hmac.new(
                self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha512
            ).hexdigest()

        async def _request(self, channel: str, event: Optional[str] = None, payload: Optional[List] = None, auth_required: bool = False):
            """Envia requisição WebSocket"""
            if not self.connection or not self.connected:
                logger.warning(f"⚠️ WS não conectado, não pode enviar requisição para {channel}. Tentando reconectar...")
                await self.start()
                if not self.connection or not self.connected:
                    logger.error(f"❌ Falha ao reconectar WS para enviar requisição para {channel}.")
                    return

            self._message_id += 1
            current_time = int(time.time())
            req_message = {
                "time": current_time, "channel": channel, "event": event, "payload": payload, "id": self._message_id
            }

            if auth_required:
                signature = self._get_signature(channel, event, current_time)
                req_message['auth'] = {"method": "api_key", "KEY": self.api_key, "SIGN": signature}

            try:
                await self.connection.send(json.dumps(req_message))
                logger.debug(f"📤 WS enviado: {json.dumps(req_message)}")
            except Exception as e:
                logger.error(f"❌ Erro enviando WS requisição: {e}", exc_info=True)

        async def subscribe(self, channel: str, event: Optional[str] = "subscribe", payload: Optional[List] = None, auth_required: bool = False):
            """Subscreve a canal/método da Gate.io"""
            self.subscribed_channels.add((channel, event, tuple(payload) if payload is not None else None, auth_required))
            await self._request(channel, event, payload, auth_required)

        async def subscribe_ticker(self, symbols: List[str]):
            """Subscreve preços de múltiplos símbolos"""
            for symbol in symbols:
                await self.subscribe("futures.tickers", "subscribe", [symbol], auth_required=False)

        def add_callback(self, channel_name: str, callback: Callable):
            """Adiciona callback para canal/método da Gate.io"""
            if channel_name not in self.callbacks: self.callbacks[channel_name] = []
            self.callbacks[channel_name].append(callback)

        async def _process_message(self, message: str):
            """Processa mensagem WebSocket"""
            try:
                data = json.loads(message)
                channel_name = data.get('channel')

                if data.get('event') == 'subscribe' and 'id' in data:
                    if data.get('result', {}).get('status') == 'success':
                        if channel_name in ["futures.balances", "futures.positions", "futures.orders", "futures.login"]:
                            self.authenticated = True
                            logger.info(f"🔐 Autenticação WebSocket Gate.io bem-sucedida via {channel_name}!")
                        logger.debug(f"WS request ID {data['id']} (channel: {channel_name}) confirmed: {data.get('result')}")
                    elif data.get('error'):
                        logger.error(f"WS request ID {data['id']} error (channel: {channel_name}): {data['error'].get('message', 'Unknown error')}. Full message: {data}")
                    return

                if data.get('error'):
                    logger.error(f"WS Error received (channel: {channel_name}): {data.get('error', {}).get('message', 'Unknown WS error')}. Full message: {data}")
                    return

                if channel_name and channel_name in self.callbacks:
                    for callback in self.callbacks[channel_name]:
                        asyncio.create_task(callback(data))
                elif 'method' in data and data['method'] in self.callbacks:
                    for callback in self.callbacks[data['method']]:
                        asyncio.create_task(callback(data))
                else:
                    logger.debug(f"Mensagem recebida para canal sem callback ou não reconhecido: {channel_name} (ou method: {data.get('method')}). Full message: {data}")
            except Exception as e:
                logger.error(f"❌ Erro ao processar mensagem WebSocket: {e}", exc_info=True)

        async def _listen(self):
            """Escuta o WebSocket em um loop contínuo"""
            while self.connected:
                try:
                    message = await self.connection.recv()
                    asyncio.create_task(self._process_message(message))
                except websockets.ConnectionClosed as e:
                    logger.warning(f"🔌 WebSocket Gate.io desconectado: {e}. Tentando reconectar em 5s...")
                    self.authenticated = False
                    self.connection = None
                    self.connected = False
                    break
                except asyncio.CancelledError:
                    logger.info("Tarefa de escuta WebSocket cancelada.")
                    break
                except Exception as e:
                    logger.error(f"❌ Erro inesperado no loop de escuta WebSocket: {e}", exc_info=True)
                    if self.connection:
                        try:
                            await self.connection.close()
                        except Exception as close_e:
                            logger.error(f"Erro ao fechar conexão após erro inesperado: {close_e}")
                    self.connection = None
                    self.connected = False
                    self.authenticated = False
                    break