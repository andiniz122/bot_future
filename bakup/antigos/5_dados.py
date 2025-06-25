#!/usr/bin/env python3
"""
Sistema de Coleta de Dados Avançado
Coleta dados históricos e em tempo real com cache inteligente
Suporta múltiplas exchanges e reconexão automática
"""

import asyncio
import logging
import aiohttp
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import sqlite3
import pickle
import gzip
from pathlib import Path
import time
import hashlib
from collections import defaultdict, deque
import hmac
import base64
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_system')

# =====================================================================
# CLASSES DE DADOS E CONFIGURAÇÃO
# =====================================================================

@dataclass
class DataConfig:
    """Configuração do sistema de dados"""
    # Cache settings
    cache_directory: str = "data_cache"
    max_cache_size_mb: int = 1000  # 1GB
    cache_ttl_hours: int = 24
    
    # Rate limiting
    requests_per_second: float = 10.0
    burst_requests: int = 20
    
    # WebSocket settings
    websocket_timeout: int = 30
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    
    # Data settings
    default_timeframe: str = "1h"
    max_candles_per_request: int = 1000
    data_validation: bool = True
    
    # Database settings
    db_path: str = "trading_data.db"
    enable_database: bool = True

@dataclass
class MarketData:
    """Estrutura padrão para dados de mercado"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    source: str = "gate.io"
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketData':
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume']),
            timeframe=data['timeframe'],
            source=data.get('source', 'gate.io')
        )

@dataclass
class TickerData:
    """Dados de ticker em tempo real"""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    source: str = "gate.io"

@dataclass
class DataCacheStats:
    """Estatísticas do cache"""
    total_symbols: int = 0
    total_candles: int = 0
    cache_size_mb: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

# =====================================================================
# SISTEMA DE CACHE INTELIGENTE
# =====================================================================

class IntelligentCache:
    """Cache inteligente com compressão e TTL"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.cache_dir = Path(config.cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.access_times = {}
        self.stats = DataCacheStats()
        
        # Rate limiting
        self.request_times = deque()
        
    def _get_cache_path(self, key: str) -> Path:
        """Gera caminho do arquivo de cache"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache.gz"
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtém dados do cache"""
        
        # Tentar cache em memória primeiro
        if key in self.memory_cache:
            self.access_times[key] = time.time()
            self.stats.hit_rate += 1
            return self.memory_cache[key]
        
        # Tentar cache em disco
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                # Verificar TTL
                file_age = time.time() - cache_path.stat().st_mtime
                if file_age > (self.config.cache_ttl_hours * 3600):
                    cache_path.unlink()  # Remover cache expirado
                    self.stats.miss_rate += 1
                    return None
                
                # Carregar dados comprimidos
                with gzip.open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    
                # Adicionar ao cache em memória
                self.memory_cache[key] = data
                self.access_times[key] = time.time()
                self.stats.hit_rate += 1
                return data
                
            except Exception as e:
                logger.warning(f"Erro lendo cache {key}: {e}")
                cache_path.unlink()  # Remover cache corrompido
        
        self.stats.miss_rate += 1
        return None
    
    async def set(self, key: str, data: Any, compress: bool = True):
        """Armazena dados no cache"""
        
        try:
            # Cache em memória
            self.memory_cache[key] = data
            self.access_times[key] = time.time()
            
            # Cache em disco comprimido
            if compress:
                cache_path = self._get_cache_path(key)
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Limpar cache se necessário
            await self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Erro salvando cache {key}: {e}")
    
    async def _cleanup_cache(self):
        """Limpa cache antigo e libera memória"""
        
        current_time = time.time()
        
        # Limpar cache em memória
        keys_to_remove = []
        for key, access_time in self.access_times.items():
            if current_time - access_time > 3600:  # 1 hora sem acesso
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.memory_cache.pop(key, None)
            self.access_times.pop(key, None)
        
        # Verificar tamanho do cache em disco
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache.gz"))
        cache_size_mb = cache_size / (1024 * 1024)
        
        if cache_size_mb > self.config.max_cache_size_mb:
            # Remover arquivos mais antigos
            cache_files = sorted(
                self.cache_dir.glob("*.cache.gz"),
                key=lambda x: x.stat().st_mtime
            )
            
            files_to_remove = cache_files[:len(cache_files) // 4]  # Remove 25%
            for file_path in files_to_remove:
                file_path.unlink()
        
        # Atualizar stats
        self.stats.cache_size_mb = cache_size_mb
        self.stats.last_updated = datetime.now()
    
    async def wait_for_rate_limit(self):
        """Implementa rate limiting"""
        
        current_time = time.time()
        
        # Remover timestamps antigos
        while self.request_times and current_time - self.request_times[0] > 1.0:
            self.request_times.popleft()
        
        # Verificar se precisa esperar
        if len(self.request_times) >= self.config.requests_per_second:
            sleep_time = 1.0 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(current_time)

# =====================================================================
# CLIENTE DE API PARA GATE.IO
# =====================================================================

class GateIOClient:
    """Cliente assíncrono para API da Gate.io"""
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.gateio.ws" if not testnet else "https://fx-api-testnet.gateio.ws"
        self.session = None
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'TradingBot/1.0',
                'Content-Type': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, method: str, url: str, query_string: str = "", payload: str = "") -> Dict[str, str]:
        """Gera assinatura para requests autenticados"""
        
        if not self.api_key or not self.secret_key:
            return {}
        
        timestamp = str(int(time.time()))
        string_to_sign = f"{method}\n{url}\n{query_string}\n{hashlib.sha512(payload.encode()).hexdigest()}\n{timestamp}"
        
        signature = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha512
        ).hexdigest()
        
        return {
            'KEY': self.api_key,
            'Timestamp': timestamp,
            'SIGN': signature
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                          data: Dict = None, authenticated: bool = False) -> Optional[Dict]:
        """Faz request HTTP com retry e rate limiting"""
        
        if not self.session:
            raise ValueError("Client não inicializado. Use 'async with'.")
        
        # Rate limiting simples
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # 100ms entre requests
            await asyncio.sleep(0.1)
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # Preparar parâmetros
        if params:
            query_string = urlencode(params)
            url += f"?{query_string}"
        else:
            query_string = ""
        
        # Payload para POST/PUT
        payload = ""
        if data:
            payload = json.dumps(data)
        
        # Autenticação se necessária
        if authenticated:
            auth_headers = self._generate_signature(method.upper(), endpoint, query_string, payload)
            headers.update(auth_headers)
        
        # Fazer request com retry
        for attempt in range(3):
            try:
                async with self.session.request(
                    method, url, headers=headers, data=payload if data else None
                ) as response:
                    
                    self.last_request_time = time.time()
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit hit, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API Error: {response.status} - {await response.text()}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return None
            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return None
        
        return None
    
    async def get_futures_tickers(self) -> List[Dict]:
        """Obtém todos os tickers de futuros"""
        
        result = await self._make_request("GET", "/api/v4/futures/usdt/tickers")
        return result if result else []
    
    async def get_klines(self, symbol: str, interval: str = "1h", 
                        limit: int = 1000, from_time: int = None, to_time: int = None) -> List[Dict]:
        """Obtém dados de klines (candlesticks)"""
        
        params = {
            'contract': symbol,
            'interval': interval,
            'limit': min(limit, 1000)  # Gate.io limit
        }
        
        if from_time:
            params['from'] = from_time
        if to_time:
            params['to'] = to_time
        
        result = await self._make_request("GET", "/api/v4/futures/usdt/candlesticks", params=params)
        return result if result else []
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Obtém orderbook"""
        
        params = {
            'contract': symbol,
            'limit': limit
        }
        
        return await self._make_request("GET", "/api/v4/futures/usdt/order_book", params=params)
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Obtém trades recentes"""
        
        params = {
            'contract': symbol,
            'limit': limit
        }
        
        result = await self._make_request("GET", "/api/v4/futures/usdt/trades", params=params)
        return result if result else []

# =====================================================================
# SISTEMA DE WEBSOCKETS
# =====================================================================

class WebSocketManager:
    """Gerenciador de WebSocket com reconexão automática"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.ws_url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        self.websocket = None
        self.subscriptions = set()
        self.callbacks = defaultdict(list)
        self.running = False
        self.reconnect_count = 0
        
    async def connect(self):
        """Conecta ao WebSocket"""
        
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                timeout=self.config.websocket_timeout,
                ping_interval=20,
                ping_timeout=10
            )
            self.running = True
            self.reconnect_count = 0
            logger.info("✅ WebSocket conectado")
            
            # Re-subscrever aos canais
            for subscription in self.subscriptions:
                await self._send_subscription(subscription)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro conectando WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Desconecta do WebSocket"""
        
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("WebSocket desconectado")
    
    async def _send_subscription(self, subscription: Dict):
        """Envia uma subscrição"""
        
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.send(json.dumps(subscription))
                logger.debug(f"Subscription sent: {subscription}")
            except Exception as e:
                logger.error(f"Erro enviando subscription: {e}")
    
    async def subscribe_tickers(self, symbols: List[str], callback: Callable):
        """Subscreve a tickers de símbolos"""
        
        subscription = {
            "method": "ticker.subscribe",
            "params": symbols,
            "id": int(time.time())
        }
        
        self.subscriptions.add(json.dumps(subscription, sort_keys=True))
        self.callbacks['ticker'].append(callback)
        
        if self.websocket:
            await self._send_subscription(subscription)
    
    async def subscribe_klines(self, symbols: List[str], interval: str, callback: Callable):
        """Subscreve a klines"""
        
        # Gate.io format: [symbol, interval]
        params = [[symbol, interval] for symbol in symbols]
        
        subscription = {
            "method": "kline.subscribe",
            "params": params,
            "id": int(time.time())
        }
        
        self.subscriptions.add(json.dumps(subscription, sort_keys=True))
        self.callbacks['kline'].append(callback)
        
        if self.websocket:
            await self._send_subscription(subscription)
    
    async def subscribe_trades(self, symbols: List[str], callback: Callable):
        """Subscreve a trades"""
        
        subscription = {
            "method": "trades.subscribe",
            "params": symbols,
            "id": int(time.time())
        }
        
        self.subscriptions.add(json.dumps(subscription, sort_keys=True))
        self.callbacks['trades'].append(callback)
        
        if self.websocket:
            await self._send_subscription(subscription)
    
    async def _handle_message(self, message: str):
        """Processa mensagens recebidas"""
        
        try:
            data = json.loads(message)
            
            # Verificar se é uma resposta de subscription
            if 'id' in data and 'result' in data:
                if data['result']['status'] == 'success':
                    logger.debug("Subscription confirmada")
                else:
                    logger.warning(f"Subscription failed: {data['result']}")
                return
            
            # Processar dados de mercado
            method = data.get('method', '')
            params = data.get('params', {})
            
            if method == 'ticker.update':
                await self._process_ticker_update(params)
            elif method == 'kline.update':
                await self._process_kline_update(params)
            elif method == 'trades.update':
                await self._process_trades_update(params)
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _process_ticker_update(self, params):
        """Processa update de ticker"""
        
        try:
            symbol = params[0]
            ticker_data = params[1]
            
            # Converter para formato padrão
            ticker = TickerData(
                symbol=symbol,
                price=float(ticker_data['last']),
                change_24h=float(ticker_data['change_percentage']),
                volume_24h=float(ticker_data['base_volume']),
                high_24h=float(ticker_data['high_24h']),
                low_24h=float(ticker_data['low_24h']),
                timestamp=datetime.now()
            )
            
            # Chamar callbacks
            for callback in self.callbacks['ticker']:
                try:
                    await callback(ticker)
                except Exception as e:
                    logger.error(f"Error in ticker callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing ticker update: {e}")
    
    async def _process_kline_update(self, params):
        """Processa update de kline"""
        
        try:
            symbol = params[0][0]
            interval = params[0][1]
            kline_data = params[1]
            
            # Converter para formato padrão
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(kline_data['t'])),
                open=float(kline_data['o']),
                high=float(kline_data['h']),
                low=float(kline_data['l']),
                close=float(kline_data['c']),
                volume=float(kline_data['v']),
                timeframe=interval
            )
            
            # Chamar callbacks
            for callback in self.callbacks['kline']:
                try:
                    await callback(market_data)
                except Exception as e:
                    logger.error(f"Error in kline callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing kline update: {e}")
    
    async def _process_trades_update(self, params):
        """Processa update de trades"""
        
        # Implementar se necessário
        pass
    
    async def listen(self):
        """Loop principal de escuta"""
        
        while self.running:
            try:
                if not self.websocket or self.websocket.closed:
                    if not await self.connect():
                        await asyncio.sleep(self.config.reconnect_delay)
                        continue
                
                # Escutar mensagens
                async for message in self.websocket:
                    await self._handle_message(message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self.running:
                    await self._handle_reconnect()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    await self._handle_reconnect()
    
    async def _handle_reconnect(self):
        """Gerencia reconexão automática"""
        
        self.reconnect_count += 1
        
        if self.reconnect_count <= self.config.max_reconnect_attempts:
            wait_time = min(self.config.reconnect_delay * (2 ** (self.reconnect_count - 1)), 60)
            logger.info(f"Reconectando em {wait_time}s (tentativa {self.reconnect_count})")
            await asyncio.sleep(wait_time)
        else:
            logger.error("Máximo de tentativas de reconexão atingido")
            self.running = False

# =====================================================================
# SISTEMA PRINCIPAL DE DADOS
# =====================================================================

class AdvancedDataSystem:
    """Sistema principal de coleta e gerenciamento de dados"""
    
    def __init__(self, config: DataConfig = None, api_key: str = "", secret_key: str = ""):
        self.config = config or DataConfig()
        self.cache = IntelligentCache(self.config)
        self.api_client = None
        self.websocket_manager = WebSocketManager(self.config)
        
        # Credenciais da API
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Armazenamento de dados em tempo real
        self.live_data = {}
        self.live_tickers = {}
        
        # Database
        self.db_path = Path(self.config.db_path)
        if self.config.enable_database:
            self._init_database()
        
        # Callbacks para dados em tempo real
        self.data_callbacks = defaultdict(list)
        
        # Estatísticas
        self.stats = {
            'requests_made': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'websocket_messages': 0,
            'errors': 0
        }
    
    def _init_database(self):
        """Inicializa database SQLite"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabela para dados OHLCV
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT DEFAULT 'gate.io',
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    UNIQUE(symbol, timestamp, timeframe)
                )
            ''')
            
            # Tabela para tickers
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ticker_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    change_24h REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    high_24h REAL NOT NULL,
                    low_24h REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    source TEXT DEFAULT 'gate.io',
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_data_symbol_time ON ticker_data(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database inicializada")
            
        except Exception as e:
            logger.error(f"❌ Erro inicializando database: {e}")
    
    async def start(self):
        """Inicia o sistema de dados"""
        
        logger.info("🚀 Iniciando sistema de dados...")
        
        # Inicializar cliente API
        self.api_client = GateIOClient(self.api_key, self.secret_key)
        
        # Configurar callbacks do WebSocket
        await self.websocket_manager.subscribe_tickers([], self._on_ticker_update)
        await self.websocket_manager.subscribe_klines([], "1h", self._on_kline_update)
        
        # Iniciar WebSocket em background
        asyncio.create_task(self.websocket_manager.listen())
        
        logger.info("✅ Sistema de dados iniciado")
    
    async def stop(self):
        """Para o sistema de dados"""
        
        logger.info("🛑 Parando sistema de dados...")
        
        await self.websocket_manager.disconnect()
        
        if self.api_client:
            await self.api_client.__aexit__(None, None, None)
        
        logger.info("✅ Sistema de dados parado")
    
    async def get_klines_data(self, symbol: str, timeframe: str = "1h", 
                             limit: int = 1000, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Obtém dados de klines com cache inteligente"""
        
        cache_key = f"klines_{symbol}_{timeframe}_{limit}"
        
        # Tentar cache primeiro
        if use_cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                self.stats['cache_hits'] += 1
                return pd.DataFrame(cached_data)
        
        # Se não encontrou no cache, buscar da API
        await self.cache.wait_for_rate_limit()
        
        try:
            async with GateIOClient(self.api_key, self.secret_key) as client:
                raw_data = await client.get_klines(symbol, timeframe, limit)
                
                if not raw_data:
                    self.stats['errors'] += 1
                    return None
                
                # Converter para formato padrão
                data_list = []
                for candle in raw_data:
                    try:
                        data_list.append({
                            'timestamp': datetime.fromtimestamp(int(candle['t'])),
                            'open': float(candle['o']),
                            'high': float(candle['h']),
                            'low': float(candle['l']),
                            'close': float(candle['c']),
                            'volume': float(candle['v'])
                        })
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Erro processando candle: {e}")
                        continue
                
                if not data_list:
                    return None
                
                # Criar DataFrame
                df = pd.DataFrame(data_list)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Validar dados se habilitado
                if self.config.data_validation:
                    df = self._validate_ohlcv_data(df)
                
                # Salvar no cache
                await self.cache.set(cache_key, df.to_dict('records'))
                
                # Salvar no database
                if self.config.enable_database:
                    await self._save_to_database(symbol, timeframe, df)
                
                self.stats['requests_made'] += 1
                self.stats['cache_misses'] += 1
                
                return df
                
        except Exception as e:
            logger.error(f"Erro obtendo klines para {symbol}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def get_multiple_klines(self, symbols: List[str], timeframe: str = "1h", 
                                 limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Obtém klines para múltiplos símbolos em paralelo"""
        
        tasks = []
        for symbol in symbols:
            task = self.get_klines_data(symbol, timeframe, limit)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                data_dict[symbol] = result
            elif isinstance(result, Exception):
                logger.warning(f"Erro obtendo dados para {symbol}: {result}")
        
        return data_dict
    
    async def get_live_ticker(self, symbol: str) -> Optional[TickerData]:
        """Obtém ticker mais recente"""
        
        # Retornar dados em tempo real se disponíveis
        if symbol in self.live_tickers:
            return self.live_tickers[symbol]
        
        # Fallback para API
        try:
            async with GateIOClient(self.api_key, self.secret_key) as client:
                tickers = await client.get_futures_tickers()
                
                for ticker in tickers:
                    if ticker.get('contract') == symbol:
                        return TickerData(
                            symbol=symbol,
                            price=float(ticker['last']),
                            change_24h=float(ticker.get('change_percentage', 0)),
                            volume_24h=float(ticker.get('volume_24h', 0)),
                            high_24h=float(ticker.get('high_24h', 0)),
                            low_24h=float(ticker.get('low_24h', 0)),
                            timestamp=datetime.now()
                        )
                        
        except Exception as e:
            logger.error(f"Erro obtendo ticker para {symbol}: {e}")
        
        return None
    
    async def get_available_symbols(self, min_volume: float = 1000000) -> List[str]:
        """Obtém lista de símbolos disponíveis"""
        
        cache_key = f"symbols_{min_volume}"
        
        # Tentar cache
        cached_symbols = await self.cache.get(cache_key)
        if cached_symbols:
            return cached_symbols
        
        try:
            async with GateIOClient(self.api_key, self.secret_key) as client:
                tickers = await client.get_futures_tickers()
                
                symbols = []
                for ticker in tickers:
                    try:
                        contract = ticker.get('contract', '')
                        volume_24h = float(ticker.get('volume_24h', 0))
                        
                        if '_USDT' in contract and volume_24h >= min_volume:
                            symbols.append(contract)
                            
                    except (KeyError, ValueError):
                        continue
                
                # Ordenar por volume
                symbols.sort(key=lambda s: float(next(
                    t.get('volume_24h', 0) for t in tickers 
                    if t.get('contract') == s
                )), reverse=True)
                
                # Cache por 1 hora
                await self.cache.set(cache_key, symbols)
                
                return symbols
                
        except Exception as e:
            logger.error(f"Erro obtendo símbolos: {e}")
            return []
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e limpa dados OHLCV"""
        
        if df.empty:
            return df
        
        # Remover valores inválidos
        df = df.dropna()
        
        # Verificar se OHLC são válidos
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['volume'] < 0)
        )
        
        if invalid_mask.any():
            logger.warning(f"Removendo {invalid_mask.sum()} candles inválidos")
            df = df[~invalid_mask]
        
        # Remover outliers extremos (opcional)
        for col in ['open', 'high', 'low', 'close']:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            df = df[(df[col] >= q01) & (df[col] <= q99)]
        
        return df
    
    async def _save_to_database(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Salva dados no database"""
        
        if not self.config.enable_database or df.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Preparar dados para inserção
            records = []
            for timestamp, row in df.iterrows():
                records.append((
                    symbol,
                    int(timestamp.timestamp()),
                    timeframe,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ))
            
            # Inserir com IGNORE para evitar duplicatas
            conn.executemany('''
                INSERT OR IGNORE INTO market_data 
                (symbol, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro salvando no database: {e}")
    
    async def _on_ticker_update(self, ticker: TickerData):
        """Callback para updates de ticker"""
        
        self.live_tickers[ticker.symbol] = ticker
        self.stats['websocket_messages'] += 1
        
        # Chamar callbacks registrados
        for callback in self.data_callbacks['ticker']:
            try:
                await callback(ticker)
            except Exception as e:
                logger.error(f"Erro em ticker callback: {e}")
    
    async def _on_kline_update(self, market_data: MarketData):
        """Callback para updates de kline"""
        
        # Armazenar dados em tempo real
        key = f"{market_data.symbol}_{market_data.timeframe}"
        if key not in self.live_data:
            self.live_data[key] = deque(maxlen=1000)
        
        self.live_data[key].append(market_data)
        self.stats['websocket_messages'] += 1
        
        # Chamar callbacks registrados
        for callback in self.data_callbacks['kline']:
            try:
                await callback(market_data)
            except Exception as e:
                logger.error(f"Erro em kline callback: {e}")
    
    def add_data_callback(self, data_type: str, callback: Callable):
        """Adiciona callback para dados em tempo real"""
        self.data_callbacks[data_type].append(callback)
    
    def get_system_stats(self) -> Dict:
        """Retorna estatísticas do sistema"""
        
        cache_stats = self.cache.stats.__dict__
        
        return {
            'api_stats': self.stats,
            'cache_stats': cache_stats,
            'websocket_connected': self.websocket_manager.running,
            'live_symbols': len(self.live_tickers),
            'live_data_streams': len(self.live_data)
        }

# =====================================================================
# EXEMPLO DE USO
# =====================================================================

async def example_usage():
    """Exemplo de como usar o sistema de dados"""
    
    # Configuração
    config = DataConfig(
        cache_directory="./data_cache",
        max_cache_size_mb=500,
        requests_per_second=8.0
    )
    
    # Inicializar sistema
    data_system = AdvancedDataSystem(config)
    await data_system.start()
    
    try:
        # Obter símbolos disponíveis
        symbols = await data_system.get_available_symbols(min_volume=5_000_000)
        print(f"Símbolos encontrados: {len(symbols)}")
        print(f"Top 5: {symbols[:5]}")
        
        # Obter dados históricos
        df = await data_system.get_klines_data("BTC_USDT", "1h", 100)
        if df is not None:
            print(f"\nDados BTC_USDT: {len(df)} candles")
            print(df.tail())
        
        # Obter múltiplos símbolos
        multi_data = await data_system.get_multiple_klines(symbols[:3], "1h", 50)
        print(f"\nDados múltiplos: {len(multi_data)} símbolos")
        
        # Callback para dados em tempo real
        async def on_ticker_update(ticker: TickerData):
            print(f"Ticker: {ticker.symbol} = ${ticker.price:.2f}")
        
        data_system.add_data_callback('ticker', on_ticker_update)
        
        # Subscrever a dados em tempo real
        await data_system.websocket_manager.subscribe_tickers(symbols[:5], data_system._on_ticker_update)
        
        # Rodar por 30 segundos
        await asyncio.sleep(30)
        
        # Estatísticas
        stats = data_system.get_system_stats()
        print(f"\nEstatísticas: {stats}")
        
    finally:
        await data_system.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())