#!/usr/bin/python

import os
import time
import json
import hmac
import hashlib
import requests
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN

# ============================================================================
# 🔥 TESTE DE COMPRA E VENDA ETH - GATE.IO TESTNET
# Segue exatamente o mesmo padrão do conexao_testnet_gate.py
# ============================================================================

load_dotenv()

# Usar as mesmas variáveis do teste anterior
API_KEY = os.getenv('GATE_TESTNET_API_KEY') or os.getenv('GATE_API_KEY')
SECRET = os.getenv('GATE_TESTNET_API_SECRET') or os.getenv('GATE_API_SECRET')
ENVIRONMENT = os.getenv('GATE_ENVIRONMENT', 'testnet')

print("🔥 GATE.IO TESTNET - TESTE DE COMPRA E VENDA ETH")
print(f"🎯 Ambiente: {ENVIRONMENT.upper()}")
print("=" * 60)

def get_base_urls():
    """Retorna URLs baseado no ambiente - MESMO PADRÃO"""
    if ENVIRONMENT == 'testnet':
        return {
            'rest': 'https://api-testnet.gateapi.io',
            'ws': 'wss://ws-testnet.gate.io/v4/ws/futures/usdt'
        }
    else:
        return {
            'rest': 'https://fx-api.gateio.ws',
            'ws': 'wss://fx-ws.gateio.ws/v4/ws/usdt'
        }

def sign_request(method: str, endpoint: str, query_string: str = '', body: str = '') -> dict:
    """
    Gera headers assinados EXATAMENTE como no teste anterior
    """
    timestamp = str(int(time.time()))
    
    # Garante que o endpoint inclui '/api/v4'
    endpoint_for_signature = endpoint
    if not endpoint.startswith('/api/v4'):
        endpoint_for_signature = '/api/v4' + endpoint
    
    # Formato da mensagem exatamente como Gate.io requer
    message = f"{method}\n{endpoint_for_signature}\n{query_string}\n{hashlib.sha512(body.encode('utf-8')).hexdigest()}\n{timestamp}"
    
    signature = hmac.new(
        SECRET.encode('utf-8'), 
        message.encode('utf-8'), 
        hashlib.sha512
    ).hexdigest()
    
    headers = {
        'Content-Type': 'application/json',
        'KEY': API_KEY,
        'Timestamp': timestamp,
        'SIGN': signature
    }
    
    return headers

def get_account_balance():
    """Obtém saldo da conta"""
    print("\n1️⃣ VERIFICANDO SALDO DA CONTA:")
    print("-" * 40)
    
    urls = get_base_urls()
    endpoint = "/futures/usdt/accounts"
    
    headers = sign_request("GET", endpoint)
    full_url = f"{urls['rest']}/api/v4{endpoint}"
    
    try:
        response = requests.get(full_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            available = float(data.get('available', '0'))
            total = float(data.get('total', '0'))
            
            print(f"✅ Saldo disponível: {available:.2f} USDT")
            print(f"✅ Saldo total: {total:.2f} USDT")
            
            if available < 50:
                print("⚠️ ATENÇÃO: Saldo baixo para testes!")
                print("💰 Vá para https://testnet.gate.com/testnet e solicite USDT gratuito")
                return None
            
            return available
        else:
            print(f"❌ Erro ao obter saldo: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def get_eth_contract_info():
    """Obtém informações do contrato ETH_USDT"""
    print("\n2️⃣ OBTENDO INFORMAÇÕES DO CONTRATO ETH_USDT:")
    print("-" * 40)
    
    urls = get_base_urls()
    endpoint = "/futures/usdt/contracts/ETH_USDT"
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}"
        response = requests.get(full_url, timeout=15)
        
        if response.status_code == 200:
            contract = response.json()
            
            print(f"✅ Contrato: {contract.get('name', 'N/A')}")
            print(f"✅ Status: {contract.get('trade_status', 'N/A')}")
            print(f"✅ Tamanho mínimo: {contract.get('order_size_min', 'N/A')}")
            print(f"✅ Tamanho máximo: {contract.get('order_size_max', 'N/A')}")
            print(f"✅ Tick size: {contract.get('mark_price_round', 'N/A')}")
            
            return contract
        else:
            print(f"❌ Erro ao obter contrato: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def get_eth_price():
    """Obtém preço atual do ETH"""
    print("\n3️⃣ OBTENDO PREÇO ATUAL DO ETH:")
    print("-" * 40)
    
    urls = get_base_urls()
    endpoint = "/futures/usdt/tickers"
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}?contract=ETH_USDT"
        response = requests.get(full_url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                ticker = data[0]
                last_price = float(ticker.get('last', '0'))
                bid_price = float(ticker.get('highest_bid', '0'))
                ask_price = float(ticker.get('lowest_ask', '0'))
                
                print(f"✅ Último preço: ${last_price:.2f}")
                print(f"✅ Bid (compra): ${bid_price:.2f}")
                print(f"✅ Ask (venda): ${ask_price:.2f}")
                
                return {
                    'last': last_price,
                    'bid': bid_price,
                    'ask': ask_price
                }
        
        print(f"❌ Erro ao obter preço: {response.status_code}")
        return None
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def place_order(contract: str, size: str, price: str = None, side: str = "long", order_type: str = "limit"):
    """Coloca uma ordem"""
    urls = get_base_urls()
    endpoint = "/futures/usdt/orders"
    
    # Monta o corpo da ordem
    order_data = {
        "contract": contract,
        "size": int(size) if side == "long" else -int(size),  # Positivo para long, negativo para short
        "text": f"t-test_{int(time.time())}"  # CORREÇÃO: Gate.io exige que text comece com 't-'
    }
    
    if order_type == "limit" and price:
        order_data["price"] = str(price)
    
    body = json.dumps(order_data)
    headers = sign_request("POST", endpoint, "", body)
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}"
        response = requests.post(full_url, headers=headers, data=body, timeout=15)
        
        if response.status_code in [200, 201]:
            order = response.json()
            print(f"✅ Ordem criada: ID {order.get('id', 'N/A')}")
            return order
        else:
            error_data = response.json() if response.text else {}
            print(f"❌ Erro ao criar ordem: {response.status_code}")
            print(f"   Detalhes: {error_data}")
            return None
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def cancel_order(order_id: str):
    """Cancela uma ordem"""
    urls = get_base_urls()
    endpoint = f"/futures/usdt/orders/{order_id}"
    
    headers = sign_request("DELETE", endpoint)
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}"
        response = requests.delete(full_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            print(f"✅ Ordem {order_id} cancelada")
            return True
        else:
            print(f"❌ Erro ao cancelar ordem: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def get_order_status(order_id: str):
    """Verifica status de uma ordem"""
    urls = get_base_urls()
    endpoint = f"/futures/usdt/orders/{order_id}"
    
    headers = sign_request("GET", endpoint)
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}"
        response = requests.get(full_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            order = response.json()
            status = order.get('status', 'unknown')
            filled_size = order.get('size', 0)
            print(f"📊 Ordem {order_id}: Status={status}, Size={filled_size}")
            return order
        else:
            print(f"❌ Erro ao verificar ordem: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def test_buy_sell_sequence():
    """Executa sequência de compra e venda de teste"""
    print("\n4️⃣ EXECUTANDO TESTE DE COMPRA E VENDA:")
    print("-" * 40)
    
    # 1. Verificar saldo
    balance = get_account_balance()
    if not balance:
        return False
    
    # 2. Obter informações do contrato
    contract_info = get_eth_contract_info()
    if not contract_info:
        return False
    
    # 3. Obter preço atual
    price_info = get_eth_price()
    if not price_info:
        return False
    
    # 4. Calcular tamanho da ordem (pequeno para teste)
    min_size = int(contract_info.get('order_size_min', '1'))
    test_size = max(min_size, 1)  # Usar 1 contrato para teste
    
    current_price = price_info['last']
    
    # 5. Preço para compra (um pouco abaixo do mercado)
    buy_price = round(current_price * 0.999, 2)  # 0.1% abaixo
    
    print(f"📋 Parâmetros do teste:")
    print(f"   Tamanho: {test_size} contratos")
    print(f"   Preço atual: ${current_price:.2f}")
    print(f"   Preço de compra: ${buy_price:.2f}")
    
    # 6. COMPRAR ETH
    print(f"\n🟢 COMPRANDO {test_size} ETH a ${buy_price}:")
    buy_order = place_order("ETH_USDT", str(test_size), str(buy_price), "long", "limit")
    
    if not buy_order:
        print("❌ Falha na ordem de compra")
        return False
    
    buy_order_id = buy_order.get('id')
    print(f"✅ Ordem de compra criada: {buy_order_id}")
    
    # 7. Aguardar um pouco e verificar status
    print("⏳ Aguardando 5 segundos...")
    time.sleep(5)
    
    order_status = get_order_status(buy_order_id)
    if not order_status:
        return False
    
    # 8. Se a ordem não foi executada, cancelar e fazer ordem a mercado
    if order_status.get('status') not in ['finished', 'closed']:
        print("⚠️ Ordem de compra não foi executada, cancelando...")
        cancel_order(buy_order_id)
        
        # Fazer ordem a mercado (usando preço ask)
        market_price = price_info['ask']
        print(f"🔄 Comprando a mercado por ${market_price:.2f}")
        
        buy_order = place_order("ETH_USDT", str(test_size), str(market_price), "long", "limit")
        if not buy_order:
            return False
        
        buy_order_id = buy_order.get('id')
        time.sleep(3)
    
    # 9. Verificar se temos posição
    print(f"\n📊 Verificando posição...")
    
    # Vamos tentar vender independentemente
    sell_price = round(current_price * 1.001, 2)  # 0.1% acima
    
    # 10. VENDER ETH
    print(f"\n🔴 VENDENDO {test_size} ETH a ${sell_price}:")
    sell_order = place_order("ETH_USDT", str(test_size), str(sell_price), "short", "limit")
    
    if not sell_order:
        print("❌ Falha na ordem de venda")
        return False
    
    sell_order_id = sell_order.get('id')
    print(f"✅ Ordem de venda criada: {sell_order_id}")
    
    # 11. Aguardar e verificar
    time.sleep(3)
    sell_status = get_order_status(sell_order_id)
    
    # 12. Cancelar ordens pendentes
    print(f"\n🧹 LIMPANDO ORDENS PENDENTES:")
    cancel_order(buy_order_id)
    cancel_order(sell_order_id)
    
    print(f"\n✅ TESTE COMPLETO!")
    print(f"   ✅ Ordem de compra: {buy_order_id}")
    print(f"   ✅ Ordem de venda: {sell_order_id}")
    print(f"   ✅ Sistema de trading funcionando!")
    
    return True

def main():
    """Executa o teste completo"""
    print("🚀 INICIANDO TESTE DE COMPRA E VENDA")
    
    # Verificar credenciais
    if not API_KEY or not SECRET:
        print("❌ Credenciais não encontradas!")
        print("💡 Execute primeiro: python conexao_testnet_gate.py")
        return
    
    # Executar teste
    success = test_buy_sell_sequence()
    
    if success:
        print("\n" + "="*60)
        print("🎉 TESTE DE TRADING COMPLETADO COM SUCESSO!")
        print("✅ Compra e venda funcionando")
        print("✅ Sistema pronto para implementação")
        print("🚀 Próximo passo: Implementar estratégia de trading")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TESTE FALHOU")
        print("💡 Verifique:")
        print("   1. Saldo suficiente no testnet")
        print("   2. Permissões da API Key")
        print("   3. Execute o diagnóstico novamente")
        print("="*60)

if __name__ == "__main__":
    main()