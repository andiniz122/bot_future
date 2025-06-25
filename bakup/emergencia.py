#!/usr/bin/env python3
"""
🛡️ STOP LOSS SIMPLIFICADO E ROBUSTO
Versão que vai direto na API REST da Gate.io para máxima confiabilidade
"""
import asyncio
import logging
import aiohttp
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
import hmac
import hashlib
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_stop_loss')

class SimpleStopLoss:
    """Stop Loss direto via API REST - máxima simplicidade e confiabilidade"""
    
    def __init__(self):
        # Configurações da Gate.io
        self.api_key = os.getenv('GATE_API_KEY')
        self.api_secret = os.getenv('GATE_API_SECRET')
        self.base_url = "https://api.gateio.ws"
        
        # Configurações de proteção
        self.STOP_LOSS_PCT = -0.5  # 0.5% de perda
        self.TAKE_PROFIT_PCT = 10.0  # 10% de lucro
        
        # Estado
        self.session = None
        self.is_running = False
        self.stats = {
            'checks': 0,
            'positions_closed': 0,
            'stop_losses': 0,
            'take_profits': 0,
            'errors': 0
        }
        
        logger.info("🛡️ STOP LOSS SIMPLES INICIALIZADO")
        logger.info(f"🛑 Stop Loss: {self.STOP_LOSS_PCT}% | 🎯 Take Profit: {self.TAKE_PROFIT_PCT}%")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, method: str, url: str, query_string: str, payload: str) -> Dict[str, str]:
        """Gera assinatura para autenticação Gate.io"""
        timestamp = str(int(time.time()))
        
        # Criar string para assinatura
        message = f"{method}\n{url}\n{query_string}\n{hashlib.sha512(payload.encode()).hexdigest()}\n{timestamp}"
        
        # Gerar assinatura
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha512
        ).hexdigest()
        
        return {
            'KEY': self.api_key,
            'Timestamp': timestamp,
            'SIGN': signature
        }

    async def _api_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """Faz requisição direta para API Gate.io"""
        try:
            url = f"{self.base_url}{endpoint}"
            query_string = ""
            payload = ""
            
            if method == "GET" and params:
                query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                url += f"?{query_string}"
            
            if method in ["POST", "DELETE"] and data:
                payload = json.dumps(data)
            
            # Gerar headers de autenticação
            headers = self._generate_signature(method, endpoint, query_string, payload)
            headers['Content-Type'] = 'application/json'
            
            async with self.session.request(method, url, headers=headers, data=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"❌ API Error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return None

    async def get_positions(self) -> List[Dict]:
        """Obtém posições abertas diretamente da API"""
        try:
            logger.debug("📡 Buscando posições na API...")
            response = await self._api_request("GET", "/api/v4/futures/usdt/positions")
            
            if not response:
                return []
            
            # Filtrar apenas posições com size > 0
            positions = []
            for pos in response:
                size = float(pos.get('size', 0))
                if abs(size) > 0.000001:  # Posição válida
                    positions.append(pos)
            
            logger.debug(f"📊 {len(positions)} posições ativas encontradas")
            return positions
            
        except Exception as e:
            logger.error(f"❌ Erro obtendo posições: {e}")
            self.stats['errors'] += 1
            return []

    async def get_ticker_price(self, contract: str) -> Optional[float]:
        """Obtém preço atual do ticker"""
        try:
            response = await self._api_request("GET", f"/api/v4/futures/usdt/tickers", {"contract": contract})
            if response and len(response) > 0:
                return float(response[0].get('last', 0))
            return None
        except Exception as e:
            logger.debug(f"⚠️ Erro obtendo ticker de {contract}: {e}")
            return None

    def calculate_pnl_percentage(self, position: Dict, current_price: float) -> float:
        """Calcula PnL% de forma simples e direta"""
        try:
            entry_price = float(position.get('entry_price', 0))
            side = position.get('mode', '').upper()
            
            if entry_price <= 0 or current_price <= 0:
                return 0.0
            
            if side == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            elif side == 'SHORT':
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            else:
                return 0.0
                
            return pnl_pct
            
        except Exception as e:
            logger.error(f"❌ Erro calculando PnL: {e}")
            return 0.0

    async def close_position_direct(self, contract: str, size: float) -> bool:
        """Fecha posição diretamente via API"""
        try:
            logger.critical(f"🛑 Fechando posição {contract} (size: {size})")
            
            # Determinar side para fechamento (inverso da posição)
            side = "sell" if size > 0 else "buy"
            quantity = abs(size)
            
            order_data = {
                "contract": contract,
                "size": int(quantity),  # Gate.io espera int para size
                "price": "0",  # Market order
                "side": side,
                "reduce_only": True,
                "time_in_force": "ioc"  # Immediate or Cancel
            }
            
            response = await self._api_request("POST", "/api/v4/futures/usdt/orders", data=order_data)
            
            if response and response.get('id'):
                order_id = response['id']
                logger.critical(f"✅ Ordem de fechamento criada: {order_id}")
                return True
            else:
                logger.error(f"❌ Falha na criação da ordem: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro fechando posição {contract}: {e}")
            return False

    async def check_single_position(self, position: Dict) -> bool:
        """Verifica uma posição individual e aciona SL/TP se necessário"""
        try:
            contract = position.get('contract')
            size = float(position.get('size', 0))
            entry_price = float(position.get('entry_price', 0))
            side = position.get('mode', '').upper()
            
            # Obter preço atual
            current_price = await self.get_ticker_price(contract)
            if not current_price:
                logger.warning(f"⚠️ Não foi possível obter preço de {contract}")
                return False
            
            # Calcular PnL
            pnl_pct = self.calculate_pnl_percentage(position, current_price)
            
            logger.debug(f"📊 {contract}: {side} | PnL: {pnl_pct:+.3f}% | "
                        f"Entry: {entry_price:.6f} | Current: {current_price:.6f}")
            
            # Verificar stop loss
            if pnl_pct <= self.STOP_LOSS_PCT:
                logger.critical(f"🛑 STOP LOSS ATIVADO: {contract} | PnL: {pnl_pct:+.3f}%")
                success = await self.close_position_direct(contract, size)
                if success:
                    self.stats['stop_losses'] += 1
                    self.stats['positions_closed'] += 1
                    logger.critical(f"✅ Stop Loss executado com sucesso: {contract}")
                return success
            
            # Verificar take profit
            elif pnl_pct >= self.TAKE_PROFIT_PCT:
                logger.critical(f"🎯 TAKE PROFIT ATIVADO: {contract} | PnL: {pnl_pct:+.3f}%")
                success = await self.close_position_direct(contract, size)
                if success:
                    self.stats['take_profits'] += 1
                    self.stats['positions_closed'] += 1
                    logger.critical(f"✅ Take Profit executado com sucesso: {contract}")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro verificando posição: {e}")
            self.stats['errors'] += 1
            return False

    async def run_monitoring(self):
        """Loop principal de monitoramento"""
        self.is_running = True
        logger.critical("👁️ INICIANDO MONITORAMENTO SIMPLIFICADO")
        
        try:
            while self.is_running:
                self.stats['checks'] += 1
                logger.debug(f"🔍 Check #{self.stats['checks']}")
                
                # Obter posições atuais
                positions = await self.get_positions()
                
                if not positions:
                    logger.debug("😴 Nenhuma posição encontrada")
                else:
                    logger.info(f"📊 Verificando {len(positions)} posições...")
                    
                    # Verificar cada posição
                    for position in positions:
                        await self.check_single_position(position)
                
                # Aguardar próxima verificação
                await asyncio.sleep(2)  # 2 segundos
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoramento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}")
            self.stats['errors'] += 1
        finally:
            self.is_running = False
            await self.print_final_stats()

    async def print_final_stats(self):
        """Mostra estatísticas finais"""
        logger.critical("\n" + "=" * 50)
        logger.critical("📊 ESTATÍSTICAS FINAIS")
        logger.critical("=" * 50)
        logger.critical(f"🔍 Verificações: {self.stats['checks']}")
        logger.critical(f"📊 Posições fechadas: {self.stats['positions_closed']}")
        logger.critical(f"🛑 Stop Losses: {self.stats['stop_losses']}")
        logger.critical(f"🎯 Take Profits: {self.stats['take_profits']}")
        logger.critical(f"❌ Erros: {self.stats['errors']}")
        logger.critical("=" * 50)

async def main():
    """Função principal"""
    print("🛡️ STOP LOSS SIMPLIFICADO")
    print(f"🛑 Stop Loss: -0.5% | 🎯 Take Profit: 10%")
    print("⚡ Verificação a cada 2 segundos")
    print("\nPressione Ctrl+C para parar\n")
    
    async with SimpleStopLoss() as stop_loss:
        await stop_loss.run_monitoring()

if __name__ == "__main__":
    asyncio.run(main())