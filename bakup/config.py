# config.py - Configurações Unificadas para o Bot Gate.io

# ============ MODO DE SEGURANÇA ============
# Defina TRUE para ativar as configurações ULTRA-SEGURAS.
# Defina FALSE para usar as configurações padrão (mais flexíveis).
ULTRA_SAFE_MODE = False # Altere para True para ativar o modo ultra-seguro
NEWS_API_KEY="ec5a692f312e434e8a34bd25da9f4a19"
BTC_CONTEXT_UPDATE_INTERVAL_MINUTES = 5 # Por exemplo, atualiza o contexto BTC a cada 30 minutos

# ============ CONFIGURAÇÕES GERAIS DO BOT ============
# Intervalo padrão para atualização de klines e ciclo do bot (em segundos).
# O bot adaptativo pode ajustar isso dinamicamente.
KLINES_UPDATE_INTERVAL_SECONDS = 30 
CYCLE_INTERVAL_SECONDS = 60 # Default cycle interval for the bot's main loop

# Número máximo de falhas consecutivas antes de ativar o modo de recuperação profunda.
MAX_CONSECUTIVE_FAILURES = 3 

# ============ DESCOBERTA DE SÍMBOLOS (Valores Iniciais/Padrão) ============
# Estes valores são usados como ponto de partida pelo main.py e data_collector.py.
# O bot adaptativo no main.py os ajustará dinamicamente.

# Volume mínimo de 24h em USDT para um símbolo ser considerado (ponto de partida).
MIN_VOLUME_USDT_FOR_SELECTION = 10_000_000 # 10 milhões de USDT

# Valor nocional mínimo de uma ordem para um símbolo ser considerado.
MIN_NOTIONAL_FOR_SELECTION = 35.0

# Número máximo de símbolos a serem analisados/mantidos ativos.
# O bot adaptativo pode ajustar dinamicamente.
MAX_SYMBOLS_TO_ANALYZE = 10

# Número mínimo de símbolos ativos que o bot deve sempre tentar monitorar
MIN_ACTIVE_SYMBOLS = 5
# Número máximo de símbolos ativos para evitar sobrecarga de processamento e API
MAX_ACTIVE_SYMBOLS = 10

# ============ FALLBACK DE SEGURANÇA ============
# Símbolos manuais para usar quando a descoberta automática falha
MANUAL_SYMBOLS = [
    'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT',
    'XRP_USDT', 'ADA_USDT', 'MATIC_USDT', 'DOT_USDT',
    'LINK_USDT', 'UNI_USDT', 'LTC_USDT', 'AVAX_USDT'
]

# Símbolos ultra-seguros para modo de recuperação (usados no emergency_recovery)
RECOVERY_MODE_SYMBOLS = ['BTC_USDT', 'ETH_USDT']

# Símbolos a serem excluídos da análise e do trading (geral)
EXCLUDED_SYMBOLS = [] # Default empty, overridden in ULTRA_SAFE_MODE if active

# ============ CRITÉRIOS DE SINAL (Padrão para AdvancedSignalEngine) ============
# Estes são os critérios padrão que o AdvancedSignalEngine usará,
# mas o `main.py` pode sobrescrevê-los com base no modo adaptativo.

# Confiança mínima para um sinal ser considerado para execução de trade.
MIN_CONFIDENCE_TO_TRADE = 40.0

# Limiar para considerar um sinal como "muito forte".
MIN_CONFIDENCE_FOR_STRONG_SIGNAL = 60.0 

# Rácio de volume (volume atual / média móvel do volume) para um "spike" significativo.
MIN_VOLUME_SPIKE_RATIO = 1.5 

# Percentagem mínima de mudança de preço para um sinal ser relevante.
MIN_PRICE_CHANGE_THRESHOLD = 0.5 

# Limiares para RSI oversold e overbought.
MIN_RSI_OVERSOLD = 25 
MAX_RSI_OVERBOUGHT = 75 

# === CONFIGURAÇÃO BASE PARA O SIGNAL_ENGINE (STRATEGY_CONFIG) ===
# Este dicionário é passado para o AdvancedSignalEngine.__init__
# e é a base para as configurações da estratégia, que são ajustadas pelo modo adaptativo.
STRATEGY_CONFIG = {
    'ema_fast': 8,
    'ema_slow': 21,
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std': 2.0,
    'adx_period': 14,
    'volatility_lookback': 50,
    'ml_retrain_interval': 500, 
    'breakout_lookback': 20,
    'scalping_threshold': 0.003, 
    
    # Parâmetros de risco da estratégia (poderão ser sobrescritos pelo PortfolioManager ou ULTRA_SAFE_MODE)
    'max_portfolio_risk': 0.10, # 10% do portfólio
    'max_single_position': 0.02, # 2% do portfólio por posição
    'max_leverage': 10.0, # ajuste de risco.. teste nao tenho certeza
    
    # Parâmetros de Take Profit/Stop Loss
    'take_profit_ratio': 0.025, # 2.5%
    'stop_loss_ratio': -0.02 # -2%
}


# ============ CONFIGURAÇÕES DE PORTFOLIO MANAGER & RISCO ============
# Limite máximo de posições que o bot pode abrir concorrentemente.
MAX_CONCURRENT_POSITIONS = 5

# Risco máximo do capital total da carteira em todas as posições abertas (em porcentagem).
MAX_TOTAL_RISK_PERCENT = 30.0

# Risco máximo por trade individual (para cálculo de tamanho de posição, em porcentagem).
MAX_RISK_PER_TRADE_PERCENT = 1.0

# Alavancagem máxima padrão a ser usada se não especificado de outra forma.
MAX_LEVERAGE = 2.0

# Saldo mínimo em USDT para que o bot possa abrir um novo trade.
MIN_BALANCE_FOR_NEW_TRADE = 20.0

# Valor alvo em USDT para uma nova ordem de trade (usado para dimensionamento).
TARGET_USDT_FOR_NEW_TRADE = 10.0

# Configurações de SL/TP para o ImprovedSLTPManager (valores padrão)
STOP_LOSS_PCT = -2.0
TAKE_PROFIT_PCT = 2.5
TRAILING_STOP_ENABLED = True
BREAKEVEN_ENABLED = True
EMERGENCY_SL_PCT = -3.5
EMERGENCY_TP_PCT = 4.0
TRAILING_STOP_DISTANCE = 1.0
MAX_CLOSE_RETRIES = 3
QUICK_PROFIT_FOR_BREAKEVEN = 1.0


# ============ WEBSOCKET E CLIENTES API ============
WEBSOCKET_ENABLED = True 

# ============ DEBUG E LOGGING ============
DEBUG_MODE = True 
LOG_LEVEL = 'INFO' 

# ============ CONFIGURAÇÕES DE DADOS ============
# Limite de dados (klines) necessários para análise
KLINES_LIMIT = 200 # Gate.io tipicamente retorna no máximo 200 por requisição

# ============ HORÁRIOS DE OPERAÇÃO ============
# Define o horário de operação do bot em UTC.
# Para operar 24/7, defina start_hour=0, end_hour=23, weekend_trading=True
TRADING_HOURS = {
    'start_hour': 0,         # Começar às 0:00 UTC
    'end_hour': 23,          # Parar às 23:59 UTC
    'weekend_trading': True  # Operar nos finais de semana
}

# ============ DETECÇÃO DE VOLUME ANORMAL ============
ENABLE_VOLUME_SPIKE_DETECTION = True
# O volume atual deve ser X vezes maior que a média histórica para ser considerado um "spike"
VOLUME_SPIKE_MULTIPLIER = 2.0         # Ex: Volume atual 2x maior que a média
# Quantos dias para calcular o volume médio histórico para a detecção de spike
VOLUME_SPIKE_LOOKBACK_DAYS = 7        # Média dos últimos 7 dias
# Volume mínimo de 24h para um símbolo ser sequer considerado para detecção de spike
MIN_VOLUME_FOR_SPIKE_CANDIDATE = 1_000_000 # Ex: 1 milhão de USDT


# ====================================================================================================
#                          LÓGICA PARA ATIVAR O MODO ULTRA-SEGURO
# ====================================================================================================
if ULTRA_SAFE_MODE:
    print("\n🛡️ ULTRA_SAFE_MODE está ATIVO! As configurações abaixo estão sendo sobrescritas.")
    
    # 💰 GESTÃO DE RISCO ULTRA-CONSERVADORA
    MAX_POSITION_SIZE_PERCENT = 0.5      # 0.5% do saldo por posição
    STOP_LOSS_PERCENT = -0.8             # Stop loss rigoroso a -0.8%
    TAKE_PROFIT_PERCENT = 3.0            # Take profit conservador a +3.0%
    
    # Alinhando com as variáveis usadas no main.py
    MAX_CONCURRENT_POSITIONS = 3               # Máximo 3 posições abertas por vez
    MAX_TOTAL_RISK_PERCENT = 30.0              # Risco total max de 10% em modo ultra-seguro
    MAX_RISK_PER_TRADE_PERCENT = 0.5           # Max 0.5% risco por trade
    
    EMERGENCY_LOSS_PERCENT = -3.0        # Parada de emergência a -3%
    
    # 📊 CRITÉRIOS DE ENTRADA RIGOROSOS
    MIN_CONFIDENCE_TO_TRADE = 75.0       # Mínimo 75% de confiança
    MIN_BALANCE_FOR_NEW_TRADE = 100.0    # Mínimo 100 USDT para operar
    MIN_VOLUME_USDT_FOR_SELECTION = 20_000_000 # Mínimo 20M USDT volume 24h

    # 🔄 CONTROLE DE FREQUÊNCIA
    KLINES_UPDATE_INTERVAL_SECONDS = 15     
    CYCLE_INTERVAL_SECONDS = 120         # Ciclos a cada 2 minutos
    # MAX_DAILY_TRADES = 3                 # Not directly used by main.py but good to keep for reference
    # MAX_HOURLY_TRADES = 1                # Not directly used by main.py but good to keep for reference

    # 🛡️ SÍMBOLOS ULTRA-SEGUROS (apenas os mais estáveis)
    MANUAL_SYMBOLS = [ # Override MANUAL_SYMBOLS for ultra safe mode
        'BTC_USDT', 'ETH_USDT', 'BNB_USDT'
    ]

    # 🚫 SÍMBOLOS PROIBIDOS (alta volatilidade/risco)
    EXCLUDED_SYMBOLS = [ # Directly defining here to override or complement
        'USDC_USDT', 'BUSD_USDT', 'DAI_USDT', 'TUSD_USDT', 'USDD_USDT', 'FRAX_USDT', 
        'LUNA_USDT', 'UST_USDT', 'FTT_USDT', 
        'SHIB_USDT', 'PEPE_USDT', 'DOGE_USDT', 'FLOKI_USDT', # Memecoins
        'APE_USDT', 'SAND_USDT', 'MANA_USDT', # Altcoins voláteis
        'YFI_USDT', 'SUSHI_USDT', 'CAKE_USDT', # DeFi de alto risco
        'BTT_USDT', 'WIN_USDT', 'SUN_USDT' # Baixa liquidez Gate.io
    ]

    # 📈 INDICADORES TÉCNICOS CONSERVADORES (aplicados no STRATEGY_CONFIG base)
    TECHNICAL_INDICATORS = {
        'ema_fast': 20, 'ema_slow': 50, 'rsi_period': 21,
        'rsi_oversold': 25, 'rsi_overbought': 75,
        'bb_period': 30, 'bb_std': 2.5,
        'volume_threshold': 1.5,
        'trend_confirmation': 3
    }
    
    # 🔒 CONFIGURAÇÕES DE ALAVANCAGEM
    MAX_LEVERAGE = 1.0           
    LEVERAGE_FORBIDDEN = True    # Keep this if it triggers specific logic elsewhere

    # ⏰ HORÁRIOS DE OPERAÇÃO (UTC)
    TRADING_HOURS = {
        'start_hour': 0,         
        'end_hour': 23,          
        'weekend_trading': True  
    }

    # 📊 ANÁLISE DE MERCADO
    MARKET_ANALYSIS = {
        'min_data_points': 200,   
        'volatility_window': 30,  
        'trend_window': 50,       
        'volume_window': 20       
    }

    # 🛡️ SISTEMA DE ALERTAS
    ALERTS = {
        'telegram_enabled': True,     
        'email_enabled': False,        
        'log_level': 'INFO', # Adjusted for ULTRA_SAFE_MODE, to reduce debug logs
        'emergency_contacts': []       
    }
    LOG_LEVEL = ALERTS['log_level'] # Make sure global LOG_LEVEL is updated

    # 💾 BACKUP E LOGS
    BACKUP_CONFIG = {
        'save_trades': True,           
        'save_positions': True,        
        'log_rotation': True,          
        'max_log_files': 30,          
        'backup_interval_hours': 6     
    }

    # 🆘 CONFIGURAÇÕES DE EMERGÊNCIA
    EMERGENCY_CONFIG = {
        'auto_close_on_loss': True,        
        'close_all_on_emergency': True,    
        'emergency_cooldown_hours': 24,    
        'max_emergency_stops': 3,          
        'notify_on_emergency': True        
    }

    # 📱 CONFIGURAÇÕES DA API
    API_CONFIG = {
        'timeout_seconds': 30,         
        'max_retries': 3,             
        'retry_delay_seconds': 5,     
        'rate_limit_per_minute': 30   
    }

    # 🎯 METAS CONSERVADORAS
    TARGETS = {
        'daily_profit_target': 0.5,    
        'weekly_profit_target': 2.0,   
        'monthly_profit_target': 8.0,  
        'max_acceptable_loss': -1.0    
    }
    
    # Overwrite STRATEGY_CONFIG with ULTRA_SAFE values (recommended)
    STRATEGY_CONFIG['max_portfolio_risk'] = MAX_TOTAL_RISK_PERCENT / 100.0
    STRATEGY_CONFIG['max_single_position'] = MAX_POSITION_SIZE_PERCENT / 100.0
    STRATEGY_CONFIG['max_leverage'] = MAX_LEVERAGE
    STRATEGY_CONFIG['take_profit_ratio'] = TAKE_PROFIT_PERCENT / 100.0
    STRATEGY_CONFIG['stop_loss_ratio'] = STOP_LOSS_PERCENT / 100.0
    STRATEGY_CONFIG['min_confidence_to_trade'] = MIN_CONFIDENCE_TO_TRADE # Ensure this is also passed

    # Override SL/TP settings for ImprovedSLTPManager in ULTRA_SAFE_MODE
    STOP_LOSS_PCT = STOP_LOSS_PERCENT
    TAKE_PROFIT_PCT = TAKE_PROFIT_PERCENT
    EMERGENCY_SL_PCT = EMERGENCY_LOSS_PERCENT # Using the emergency loss for SL
    EMERGENCY_TP_PCT = TAKE_PROFIT_PERCENT * 1.5 # Example: higher TP for emergency
    TRAILING_STOP_ENABLED = False # Disable trailing stop in ultra safe mode for simplicity/control
    BREAKEVEN_ENABLED = True
    TRAILING_STOP_DISTANCE = 0.5 # Smaller distance if enabled
    MAX_CLOSE_RETRIES = 5 # More retries for critical closures
    QUICK_PROFIT_FOR_BREAKEVEN = 0.5 # Move to breakeven quicker

# 📋 VALIDAÇÕES GLOBAIS (apenas se este arquivo for executado diretamente)
def validate_config():
    """Valida algumas configurações para garantir consistência."""
    
    validations = []
    
    if ULTRA_SAFE_MODE:
        if MAX_CONCURRENT_POSITIONS > 3:
            validations.append("⚠️ MAX_CONCURRENT_POSITIONS é alto demais para ULTRA_SAFE_MODE (recomenda-se <= 3).")
        if abs(STOP_LOSS_PERCENT) < 0.5: # Stop loss muito apertado
            validations.append("⚠️ STOP_LOSS_PERCENT pode ser muito apertado, pode causar whipsaws.")
        if MIN_CONFIDENCE_TO_TRADE < 70.0: # Changed from MIN_CONFIDENCE_REQUIRED
            validations.append("⚠️ MIN_CONFIDENCE_TO_TRADE baixo demais para ULTRA_SAFE_MODE.")
        if MAX_LEVERAGE > 1.0:
            validations.append("⚠️ MAX_LEVERAGE > 1.0 detectado em ULTRA_SAFE_MODE.")
        
    else: # Modo padrão/mais flexível
        if MAX_CONCURRENT_POSITIONS < 1:
            validations.append("⚠️ MAX_CONCURRENT_POSITIONS deve ser pelo menos 1.")
        
    if MIN_VOLUME_USDT_FOR_SELECTION < 1_000_000:
        validations.append("⚠️ MIN_VOLUME_USDT_FOR_SELECTION baixo. Pode selecionar ativos pouco líquidos.")
    if MIN_NOTIONAL_FOR_SELECTION < 10.0:
        validations.append("⚠️ MIN_NOTIONAL_FOR_SELECTION baixo. Pode gerar ordens muito pequenas.")

    if validations:
        print("\n--- AVISOS DE CONFIGURAÇÃO ---")
        for msg in validations:
            print(msg)
        print("----------------------------")
        return False
    else:
        print("\n✅ Configurações validadas. Tudo pronto!")
        return True

# 📊 RESUMO DAS CONFIGURAÇÕES (apenas se este arquivo for executado diretamente)
def print_config_summary():
    """Imprime um resumo das configurações atuais."""
    
    print("\n=== RESUMO DAS CONFIGURAÇÕES ATUAIS ===")
    print(f"Modo ULTRA_SAFE: {'ATIVO' if ULTRA_SAFE_MODE else 'DESATIVADO'}")
    print(f"Intervalo de Ciclo: {CYCLE_INTERVAL_SECONDS}s")
    print(f"Máx. Falhas Consecutivas: {MAX_CONSECUTIVE_FAILURES}")
    print(f"Volume Mín. P/ Seleção: {MIN_VOLUME_USDT_FOR_SELECTION:,.0f} USDT")
    print(f"Notional Mín. P/ Seleção: {MIN_NOTIONAL_FOR_SELECTION:.1f} USDT")
    print(f"Máx. Símbolos P/ Análise: {MAX_SYMBOLS_TO_ANALYZE}")
    print(f"Símbolos Excluídos (base): {len(EXCLUDED_SYMBOLS)}")
    print(f"SL/TP padrão estratégia: {STRATEGY_CONFIG['stop_loss_ratio']*100:.2f}% / {STRATEGY_CONFIG['take_profit_ratio']*100:.2f}%")
    print(f"Máx. Posições Concorrentes: {MAX_CONCURRENT_POSITIONS}")
    print(f"Máx. Alavancagem: {MAX_LEVERAGE}x")
    print(f"Horário de Operação: {TRADING_HOURS['start_hour']}h-{TRADING_HOURS['end_hour']}h UTC (Fim de Semana: {TRADING_HOURS['weekend_trading']})")
    print("=" * 50)
    # VERIFICAÇÃO TEMPORÁRIA
    print(f"📊 CONFIG CARREGADO: MAX_CONCURRENT_POSITIONS = {MAX_CONCURRENT_POSITIONS}")
    print(f"📊 CONFIG CARREGADO: MAX_TOTAL_RISK_PERCENT = {MAX_TOTAL_RISK_PERCENT}")

if __name__ == "__main__":
    print_config_summary()
    validate_config()