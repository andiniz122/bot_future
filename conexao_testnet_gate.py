#!/usr/bin/python

import os
import time
import json
import hmac
import hashlib
import requests
from dotenv import load_dotenv

# ============================================================================
# 🔍 DEBUGGER DE CREDENCIAIS GATE.IO - VERSÃO PADRONIZADA
# Segue exatamente o mesmo padrão do gate_api.py
# ============================================================================

load_dotenv()

# CORREÇÃO: Usar as variáveis que estão no seu .env
API_KEY = os.getenv('GATE_TESTNET_API_KEY') or os.getenv('GATE_API_KEY')
SECRET = os.getenv('GATE_TESTNET_API_SECRET') or os.getenv('GATE_API_SECRET')
ENVIRONMENT = os.getenv('GATE_ENVIRONMENT', 'testnet')

print("🔍 GATE.IO CREDENTIAL DEBUGGER - VERSÃO PADRONIZADA")
print(f"🎯 Ambiente: {ENVIRONMENT.upper()}")
print("=" * 60)

def get_base_urls():
    """Retorna URLs baseado no ambiente - ATUALIZADAS conforme documentação oficial"""
    if ENVIRONMENT == 'testnet':
        return {
            'rest': 'https://api-testnet.gateapi.io',  # URL OFICIAL NOVA
            'ws': 'wss://ws-testnet.gate.io/v4/ws/futures/usdt'
        }
    else:
        return {
            'rest': 'https://fx-api.gateio.ws',  # Para futuros live
            'ws': 'wss://fx-ws.gateio.ws/v4/ws/usdt'
        }

def check_credential_format():
    """Verifica formato das credenciais"""
    print("1️⃣ VERIFICANDO FORMATO DAS CREDENCIAIS:")
    print("-" * 40)
    
    if not API_KEY or not SECRET:
        print("❌ Credenciais não encontradas no .env")
        print("📝 Verifique se o arquivo .env contém:")
        print("   GATE_API_KEY=sua_key")
        print("   GATE_API_SECRET=sua_secret")
        print("   GATE_ENVIRONMENT=testnet  # ou 'live'")
        return False
    
    print(f"✅ API Key encontrada: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"✅ Secret encontrada: {len(SECRET)} caracteres")
    print(f"✅ Ambiente: {ENVIRONMENT}")
    
    # Verificar se API Key tem formato correto
    if len(API_KEY) < 20:
        print("⚠️ API Key parece muito curta")
        return False
    
    if len(SECRET) < 30:
        print("⚠️ Secret Key parece muito curta")
        return False
    
    # Verificar caracteres válidos para secret (deve ser hex)
    valid_chars = set('0123456789abcdefABCDEF')
    if not all(c in valid_chars for c in SECRET):
        print("⚠️ Secret Key contém caracteres inválidos (deve ser hex)")
        return False
    
    print("✅ Formato das credenciais parece correto")
    return True

def sign_request_like_gate_api(method: str, endpoint: str, query_string: str = '', body: str = '') -> dict:
    """
    Gera headers assinados EXATAMENTE como no gate_api.py
    """
    timestamp = str(int(time.time()))
    
    # CORREÇÃO: Garante que o 'endpoint_for_signature' inclui '/api/v4'
    # Exatamente como no gate_api.py
    endpoint_for_signature = endpoint
    if not endpoint.startswith('/api/v4'):
        endpoint_for_signature = '/api/v4' + endpoint
    
    # Gate.io requires a specific message format for signing
    # EXATAMENTE como no gate_api.py
    message = f"{method}\n{endpoint_for_signature}\n{query_string}\n{hashlib.sha512(body.encode('utf-8')).hexdigest()}\n{timestamp}"
    
    signature = hmac.new(
        SECRET.encode('utf-8'), 
        message.encode('utf-8'), 
        hashlib.sha512
    ).hexdigest()
    
    # Headers base iguais ao gate_api.py
    headers = {
        'Content-Type': 'application/json',
        'KEY': API_KEY,
        'Timestamp': timestamp,
        'SIGN': signature
    }
    
    return headers, message, signature

def test_signature_generation():
    """Testa geração de assinatura com dados conhecidos"""
    print("\n2️⃣ TESTANDO GERAÇÃO DE ASSINATURA (PADRÃO GATE_API.PY):")
    print("-" * 40)
    
    # Dados de teste fixos
    method = "GET"
    endpoint = "/futures/usdt/accounts"  # Sem /api/v4
    query_string = ""
    body = ""
    
    print(f"📋 Dados de teste:")
    print(f"   Method: {method}")
    print(f"   Endpoint: {endpoint}")
    print(f"   Query: '{query_string}'")
    print(f"   Body: '{body}'")
    
    headers, message, signature = sign_request_like_gate_api(method, endpoint, query_string, body)
    
    print(f"   Endpoint Final: /api/v4{endpoint}")
    print(f"   Message: {repr(message)}")
    print(f"   Signature: {signature[:32]}...")
    print(f"   Timestamp: {headers['Timestamp']}")
    
    if len(signature) == 128:
        print("✅ Assinatura tem tamanho correto (128 hex chars)")
        return True
    else:
        print("❌ Assinatura tem tamanho incorreto")
        return False

def test_public_endpoint():
    """Testa endpoint público primeiro"""
    print("\n3️⃣ TESTANDO ENDPOINT PÚBLICO:")
    print("-" * 40)
    
    urls = get_base_urls()
    public_url = f"{urls['rest']}/api/v4/futures/usdt/contracts"
    
    print(f"📊 Testando: {public_url}")
    
    try:
        response = requests.get(public_url, timeout=15)
        
        if response.status_code == 200:
            contracts = response.json()
            print(f"✅ Endpoint público OK: {len(contracts)} contratos")
            
            # Mostrar alguns exemplos
            if contracts:
                print("   📋 Exemplos de contratos:")
                for i, contract in enumerate(contracts[:3]):
                    name = contract.get('name', 'N/A')
                    status = contract.get('trade_status', 'N/A')
                    print(f"      {i+1}. {name} (status: {status})")
            
            return True
        else:
            print(f"❌ Endpoint público falhou: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erro no endpoint público: {e}")
        return False

def test_private_endpoint():
    """Testa endpoint privado com autenticação"""
    print("\n4️⃣ TESTANDO ENDPOINT PRIVADO (AUTENTICAÇÃO):")
    print("-" * 40)
    
    urls = get_base_urls()
    endpoint = "/futures/usdt/accounts"
    method = "GET"
    
    # Usar a mesma função de assinatura do gate_api.py
    headers, message, signature = sign_request_like_gate_api(method, endpoint, "", "")
    
    print(f"📤 Detalhes da requisição:")
    print(f"   URL: {urls['rest']}/api/v4{endpoint}")
    print(f"   Method: {method}")
    print(f"   Timestamp: {headers['Timestamp']}")
    print(f"   API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"   Signature: {signature[:32]}...")
    print(f"   Message: {repr(message)}")
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}"
        response = requests.get(full_url, headers=headers, timeout=15)
        
        print(f"\n📊 Resposta:")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Autenticação funcionou!")
            print(f"   Dados da conta: {json.dumps(data, indent=2)}")
            
            # Extrair informações importantes
            if isinstance(data, dict):
                available = data.get('available', '0')
                total = data.get('total', '0')
                unrealized = data.get('unrealised_pnl', '0')
                print(f"\n💰 Resumo da conta USDT Futures:")
                print(f"   Available: {available}")
                print(f"   Total: {total}")
                print(f"   Unrealized PnL: {unrealized}")
            
            return True
            
        elif response.status_code == 401:
            error_data = response.json() if response.text else {}
            error_label = error_data.get('label', 'UNKNOWN')
            error_message = error_data.get('message', 'Unknown error')
            
            print(f"❌ Erro de autenticação: {error_label}")
            print(f"   Mensagem: {error_message}")
            print(f"   Response completo: {response.text}")
            
            # Diagnóstico específico
            if error_label == "INVALID_KEY":
                print("\n🔍 DIAGNÓSTICO INVALID_KEY:")
                print("   ⚠️ Possíveis causas:")
                print("   1. API Key incorreta ou expirada")
                print(f"   2. API Key não é para {ENVIRONMENT}")
                print("   3. API Key foi revogada")
                print("   4. Permissões insuficientes")
                
            elif error_label == "INVALID_SIGNATURE":
                print("\n🔍 DIAGNÓSTICO INVALID_SIGNATURE:")
                print("   ⚠️ Possíveis causas:")
                print("   1. Secret Key incorreta")
                print("   2. Problema na geração da assinatura")
                print("   3. Timestamp muito antigo/futuro")
                print("   4. Formato da mensagem incorreto")
                
            return False
        else:
            print(f"❌ Erro inesperado: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return False

def test_positions_endpoint():
    """Testa endpoint de posições"""
    print("\n5️⃣ TESTANDO ENDPOINT DE POSIÇÕES:")
    print("-" * 40)
    
    urls = get_base_urls()
    endpoint = "/futures/usdt/positions"
    method = "GET"
    
    headers, _, _ = sign_request_like_gate_api(method, endpoint, "", "")
    
    try:
        full_url = f"{urls['rest']}/api/v4{endpoint}"
        response = requests.get(full_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            positions = response.json()
            print(f"✅ Posições obtidas: {len(positions)} itens")
            
            # Filtrar posições ativas
            active_positions = []
            for pos in positions:
                size = float(pos.get('size', '0'))
                if abs(size) > 0:
                    active_positions.append(pos)
            
            print(f"📊 Posições ativas: {len(active_positions)}")
            
            if active_positions:
                print("   📋 Posições abertas:")
                for pos in active_positions:
                    contract = pos.get('contract', 'N/A')
                    size = pos.get('size', '0')
                    value = pos.get('value', '0')
                    pnl = pos.get('unrealised_pnl', '0')
                    print(f"      {contract}: Size={size}, Value={value}, PnL={pnl}")
            else:
                print("   ℹ️ Nenhuma posição ativa")
            
            return True
        else:
            error_data = response.json() if response.text else {}
            print(f"❌ Erro: {response.status_code} - {error_data}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def generate_config_guide():
    """Gera guia de configuração"""
    print("\n6️⃣ GUIA DE CONFIGURAÇÃO:")
    print("-" * 40)
    
    urls = get_base_urls()
    
    print(f"""
🔧 CONFIGURAÇÃO PARA {ENVIRONMENT.upper()}:

📁 ARQUIVO .env:
   GATE_API_KEY=sua_api_key_aqui
   GATE_API_SECRET=sua_secret_key_aqui
   GATE_ENVIRONMENT={ENVIRONMENT}

🔗 URLs UTILIZADAS (ATUALIZADAS):
   REST API: {urls['rest']}/api/v4
   WebSocket: {urls['ws']}

🔐 CRIAÇÃO DE API KEY - IMPORTANTE:
   {'TESTNET:' if ENVIRONMENT == 'testnet' else 'LIVE:'}
   1. Acesse: {'https://testnet.gate.com/myaccount/apiv4keys' if ENVIRONMENT == 'testnet' else 'https://www.gate.io/myaccount/apiv4keys'}
   2. {'Vá para aba: "Futures TestNet APIKeys"' if ENVIRONMENT == 'testnet' else 'Vá para aba: "Futures APIKeys"'}
   3. Clique em "Create API Key"
   4. Permissões necessárias:
      ✅ Futures Trading
      ✅ Read Account Info  
      ❌ Withdrawal (não necessário)
   
💰 FUNDOS TESTNET (se testnet):
   🔗 https://testnet.gate.com/testnet
   💰 Solicite USDT gratuito para testes
   
⚠️  ATENÇÃO CRÍTICA:
   - API Keys do TESTNET só funcionam no TESTNET
   - API Keys do LIVE só funcionam no LIVE  
   - São ambientes COMPLETAMENTE SEPARADOS
   
🚀 PRÓXIMOS PASSOS:
   1. CRIE API Keys no ambiente correto ({ENVIRONMENT.upper()})
   2. Atualize o arquivo .env com as novas keys
   3. Execute este teste novamente
   4. Se tudo OK, execute o sistema principal
    """)

def main():
    """Executa diagnóstico completo seguindo padrão gate_api.py"""
    print("🚀 INICIANDO DIAGNÓSTICO COMPLETO (PADRÃO GATE_API.PY)")
    
    # 1. Verificar formato
    if not check_credential_format():
        print("\n❌ PROBLEMA NO FORMATO DAS CREDENCIAIS")
        generate_config_guide()
        return
    
    # 2. Testar assinatura
    if not test_signature_generation():
        print("\n❌ PROBLEMA NA GERAÇÃO DE ASSINATURA")
        return
    
    # 3. Testar endpoint público
    if not test_public_endpoint():
        print("\n❌ PROBLEMA NO ACESSO AOS ENDPOINTS PÚBLICOS")
        print("   Verifique sua conexão de internet")
        return
    
    # 4. Testar autenticação
    if not test_private_endpoint():
        print("\n❌ PROBLEMA NA AUTENTICAÇÃO")
        generate_config_guide()
        return
    
    # 5. Testar posições
    if not test_positions_endpoint():
        print("\n⚠️ PROBLEMA NO ENDPOINT DE POSIÇÕES")
        print("   (Autenticação OK, mas posições falharam)")
    
    print("\n" + "="*60)
    print("🎉 DIAGNÓSTICO COMPLETO!")
    print("✅ Credenciais funcionando corretamente")
    print(f"✅ Ambiente {ENVIRONMENT.upper()} acessível")
    print("✅ Autenticação validada")
    print("🚀 Sistema pronto para trading!")
    print("="*60)

if __name__ == "__main__":
    main()