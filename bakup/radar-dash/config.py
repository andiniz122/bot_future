"""
📅 Dashboard Macro v2.1 FRED - Configurações
Arquivo de configuração centralizado para o sistema
"""

import os
from typing import Dict, List

# =============================================================================
# 🔑 CONFIGURAÇÕES DE API
# =============================================================================

# FRED API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY", "4533c6f5e65f2377d74e594577b3eae9")
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Yahoo Finance não requer API key, mas podemos configurar timeouts
YAHOO_TIMEOUT = 10  # segundos

# =============================================================================
# 📊 CONFIGURAÇÕES DE ATIVOS
# =============================================================================

# Símbolos principais para análise
SYMBOLS = {
    'gold': 'GC=F',        # Ouro Futures
    'btc': 'BTC-USD',      # Bitcoin
    'dxy': 'DX-Y.NYB'      # US Dollar Index
}

# Símbolos adicionais (descomente para ativar)
ADDITIONAL_SYMBOLS = {
    # 'oil': 'CL=F',         # Petróleo WTI
    # 'spy': 'SPY',          # S&P 500 ETF  
    # 'eur': 'EURUSD=X',     # EUR/USD
    # 'vix': '^VIX',         # Volatility Index
    # 'tnx': '^TNX',         # 10Y Treasury
}

# Cores para visualização
ASSET_COLORS = {
    'gold': '#FFD700',      # Dourado
    'btc': '#FF8C00',       # Laranja
    'dxy': '#00CC66',       # Verde
    'oil': '#8B4513',       # Marrom
    'spy': '#4169E1',       # Azul Real
    'eur': '#FF69B4',       # Rosa
    'vix': '#DC143C',       # Vermelho
    'tnx': '#9370DB'        # Roxo
}

# =============================================================================
# ⚙️ CONFIGURAÇÕES DE CACHE E PERFORMANCE
# =============================================================================

CACHE_CONFIG = {
    "price_data_duration": 60,        # 1 minuto para dados de preço
    "fred_data_duration": 3600,       # 1 hora para dados FRED
    "max_price_history": 120,         # 2 horas de histórico (pontos de 1min)
    "max_angular_history": 50,        # 50 pontos de análise angular
    "max_alerts": 150                 # Máximo de alertas no cache
}

# Intervalos de atualização
UPDATE_INTERVALS = {
    "price_data": 30,         # 30 segundos para dados de preço
    "angular_analysis": 120,  # 2 minutos para análise angular
    "fred_data": 600,         # 10 minutos para dados FRED
    "alert_check": 300        # 5 minutos para verificação de alertas
}

# =============================================================================
# 🎯 CONFIGURAÇÕES DE ALERTAS ANGULARES
# =============================================================================

ANGULAR_ALERT_THRESHOLDS = {
    # Perfect Divergence
    "perfect_divergence": {
        "dxy_min_angle": 20,      # DXY deve subir pelo menos 20°
        "btc_max_angle": -20,     # BTC deve cair pelo menos -20°
        "min_strength": 0.6       # R² mínimo de 0.6
    },
    
    # Extreme Momentum
    "extreme_momentum": {
        "min_angle": 45,          # Ângulo mínimo de 45°
        "min_strength": 0.7       # R² mínimo de 0.7
    },
    
    # Trend Reversal
    "trend_reversal": {
        "min_angle_change": 45    # Mudança mínima de 45° entre períodos
    },
    
    # Bullish Convergence
    "bullish_convergence": {
        "btc_min_angle": 15,      # BTC sobe pelo menos 15°
        "gold_min_angle": 10,     # Ouro sobe pelo menos 10°
        "dxy_max_angle": -5       # DXY cai pelo menos -5°
    },
    
    # Bearish Avalanche
    "bearish_avalanche": {
        "btc_max_angle": -15,     # BTC cai pelo menos -15°
        "gold_max_angle": -10,    # Ouro cai pelo menos -10°
        "dxy_min_angle": 10       # DXY sobe pelo menos 10°
    }
}

# =============================================================================
# 🏛️ CONFIGURAÇÕES DE ALERTAS MACRO TRADICIONAIS
# =============================================================================

MACRO_ALERT_THRESHOLDS = {
    # FED Alert (Política Monetária)
    "fed_alert": {
        "dxy_min_change": 0.5,    # DXY sobe pelo menos 0.5%
        "btc_max_change": -1.0,   # BTC cai pelo menos -1.0%
        "timeframe_minutes": 15   # Em 15 minutos
    },
    
    # Crisis Alert
    "crisis_alert": {
        "dxy_max_change": -0.5,   # DXY cai pelo menos -0.5%
        "btc_min_change": 2.0,    # BTC sobe pelo menos 2.0%
        "timeframe_minutes": 15   # Em 15 minutos
    },
    
    # Volatility Alert
    "volatility_alert": {
        "dxy_threshold": 1.0,     # DXY move mais que 1.0%
        "btc_threshold": 3.0,     # BTC move mais que 3.0%  
        "gold_threshold": 1.5,    # Ouro move mais que 1.5%
        "timeframe_minutes": 15   # Em 15 minutos
    },
    
    # Flight to Quality
    "flight_to_quality": {
        "gold_min_change": 1.0,   # Ouro sobe pelo menos 1.0%
        "dxy_min_change": 0.3,    # DXY sobe pelo menos 0.3%
        "timeframe_minutes": 15   # Em 15 minutos
    }
}

# =============================================================================
# 📅 CONFIGURAÇÕES FRED
# =============================================================================

# Eventos econômicos críticos com configurações
CRITICAL_ECONOMIC_EVENTS = {
    "employment": {
        "impact_score": 95,
        "volatility_factor": 2.5,
        "keywords": ["payroll", "employment", "jobs", "unemployment"],
        "typical_time": "08:30",
        "category": "EMPLOYMENT"
    },
    "inflation": {
        "impact_score": 90,
        "volatility_factor": 2.0,
        "keywords": ["cpi", "pce", "inflation", "price index"],
        "typical_time": "08:30",
        "category": "INFLATION"
    },
    "monetary_policy": {
        "impact_score": 98,
        "volatility_factor": 3.0,
        "keywords": ["fed", "fomc", "federal funds", "interest rate"],
        "typical_time": "14:00",
        "category": "MONETARY_POLICY"
    },
    "gdp": {
        "impact_score": 85,
        "volatility_factor": 1.8,
        "keywords": ["gdp", "gross domestic product", "economic growth"],
        "typical_time": "08:30",
        "category": "GDP"
    },
    "housing": {
        "impact_score": 70,
        "volatility_factor": 1.3,
        "keywords": ["housing", "home sales", "construction"],
        "typical_time": "10:00",
        "category": "HOUSING"
    },
    "trade": {
        "impact_score": 75,
        "volatility_factor": 1.5,
        "keywords": ["trade balance", "exports", "imports"],
        "typical_time": "08:30",
        "category": "TRADE"
    }
}

# Configurações de alertas preventivos FRED
FRED_ALERT_CONFIG = {
    "pre_event_hours": 48,        # Alerta 48h antes de eventos críticos
    "high_impact_threshold": 80,  # Score mínimo para classificar como alto impacto
    "max_events_fetch": 21        # Buscar eventos dos próximos 21 dias
}

# =============================================================================
# 🌐 CONFIGURAÇÕES DE SERVIDOR
# =============================================================================

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

# CORS Configuration
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["GET"],
    "allow_headers": ["*"]
}

# =============================================================================
# 📱 CONFIGURAÇÕES DE INTERFACE
# =============================================================================

UI_CONFIG = {
    "default_period": "1d",
    "default_view": "prices",
    "auto_refresh_default": True,
    "refresh_interval_seconds": 30,
    "max_chart_points": 500,
    "animation_duration": 300
}

# Períodos disponíveis
AVAILABLE_PERIODS = [
    {"label": "Hoje", "value": "1d"},
    {"label": "5 Dias", "value": "5d"},
    {"label": "1 Mês", "value": "1mo"},
    {"label": "3 Meses", "value": "3mo"},
    {"label": "6 Meses", "value": "6mo"},
    {"label": "1 Ano", "value": "1y"},
    {"label": "2 Anos", "value": "2y"},
    {"label": "5 Anos", "value": "5y"},
    {"label": "10 Anos", "value": "10y"},
    {"label": "YTD", "value": "ytd"},
    {"label": "Máx", "value": "max"}
]

# Views disponíveis
AVAILABLE_VIEWS = [
    {"label": "Preços", "value": "prices", "icon": "📈"},
    {"label": "Volume", "value": "volume", "icon": "📊"},
    {"label": "Angular", "value": "angular", "icon": "📐"},
    {"label": "Calendário", "value": "calendar", "icon": "📅"},
    {"label": "Alertas", "value": "alerts", "icon": "🚨"},
    {"label": "Combinado", "value": "combined", "icon": "🔄"}
]

# =============================================================================
# 🔧 FUNÇÕES DE VALIDAÇÃO
# =============================================================================

def validate_config() -> bool:
    """Valida se as configurações estão corretas."""
    errors = []
    
    # Verifica API Key do FRED
    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        errors.append("⚠️ FRED API Key não configurada (usando dados simulados)")
    
    # Verifica se símbolos estão definidos
    if not SYMBOLS:
        errors.append("❌ Nenhum símbolo definido para análise")
    
    # Verifica thresholds
    if not ANGULAR_ALERT_THRESHOLDS:
        errors.append("❌ Thresholds angulares não definidos")
    
    if errors:
        print("🔧 Problemas de configuração encontrados:")
        for error in errors:
            print(f"   {error}")
        return False
    
    print("✅ Configuração validada com sucesso!")
    return True

def get_asset_config(asset: str) -> Dict:
    """Retorna configuração completa de um ativo."""
    return {
        "symbol": SYMBOLS.get(asset, ""),
        "color": ASSET_COLORS.get(asset, "#666666"),
        "name": asset.upper()
    }

# =============================================================================
# 📋 INFORMAÇÕES DO SISTEMA
# =============================================================================

SYSTEM_INFO = {
    "name": "Dashboard Macro FRED",
    "version": "2.1.0-FRED-CALENDAR",
    "description": "Sistema integrado de análise macro com Angular + FRED",
    "author": "Claude AI Assistant",
    "features": [
        "🎯 Análise Angular Avançada",
        "📐 Detecção de Padrões Geométricos",
        "🔄 Alertas de Reversão de Tendência",
        "📅 Calendário Econômico FRED",
        "⏰ Alertas Preventivos de Eventos", 
        "🏛️ Integração Federal Reserve",
        "📊 Correlação Eventos vs Movimentos"
    ]
}

if __name__ == "__main__":
    # Teste de configuração
    print(f"🚀 {SYSTEM_INFO['name']} v{SYSTEM_INFO['version']}")
    print(f"📋 {SYSTEM_INFO['description']}")
    print()
    validate_config()