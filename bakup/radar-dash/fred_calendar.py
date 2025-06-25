import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass, field # Importa 'field' para dataclasses
from config import FRED_API_KEY # Certifique-se de ter config.py com FRED_API_KEY = "SUA_CHAVE_AQUI"

logger = logging.getLogger(__name__)

@dataclass
class EconomicEvent:
    """Classe para representar um evento econômico com dados mais ricos."""
    release_id: str
    name: str
    date: datetime # Data do release
    time: Optional[str] # Horário estimado do release (ex: "08:30")
    importance: str  # HIGH, MEDIUM, LOW
    frequency: Optional[str] = None # Mensal, Semanal, etc.
    series_id: Optional[str] = None # ID da série FRED associada ao evento
    previous_value: Optional[float] = None
    forecast: Optional[float] = None # Difícil de obter do FRED, pode ser N/A
    actual: Optional[float] = None # Última observação da série
    impact_score: float = 0.0 # 0-100
    currency: str = "USD"
    category: str = "GENERAL"
    # Adicionando um campo para metadados adicionais, se necessário
    metadata: Dict[str, Any] = field(default_factory=dict) # Usa default_factory para mutáveis

class FREDEconomicCalendar:
    """
    Integração profissional com a API do FRED para calendário econômico.
    Foca em releases de alto impacto e busca as últimas observações.
    """
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("FRED API Key não fornecida. Por favor, configure FRED_API_KEY em config.py.")
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # Cache de releases e eventos para evitar múltiplas requisições da mesma informação
        self.cache: Dict[str, Any] = {
            "upcoming_events": [],
            "last_full_update": None, # Timestamp da última atualização completa
            "cache_duration_seconds": 21600 # Cache por 6 horas (21600 segundos) para eventos futuros
        }
        
        # Mapeamento de releases críticos com seus FRED series IDs e scores de impacto
        # Estes IDs são os que usaremos para buscar os valores anteriores/atuais.
        self.critical_release_series: Dict[str, Dict[str, Any]] = {
            # Non-Farm Payrolls - NFP (Muitas séries para NFP, usando um proxy comum)
            "PAYEMS": { 
                "name": "Non-Farm Payrolls", "series_id": "PAYEMS", # Total Nonfarm Payrolls: All Employees, Seasonally Adjusted
                "impact_score": 95, "importance": "HIGH", "category": "EMPLOYMENT", "volatility_factor": 2.5
            },
            # Consumer Price Index (CPI)
            "CPIAUCSL": { 
                "name": "Consumer Price Index", "series_id": "CPIAUCSL", # CPI for All Urban Consumers: All Items, Seasonally Adjusted
                "impact_score": 90, "importance": "HIGH", "category": "INFLATION", "volatility_factor": 2.0
            },
            # Federal Funds Rate (Taxa de Juros)
            "FEDFUNDS": { 
                "name": "Federal Funds Rate", "series_id": "FEDFUNDS", # Federal Funds Effective Rate
                "impact_score": 98, "importance": "HIGH", "category": "MONETARY_POLICY", "volatility_factor": 3.0
            },
            # Gross Domestic Product (GDP)
            "GDP": { 
                "name": "Gross Domestic Product", "series_id": "GDP", # Gross Domestic Product
                "impact_score": 85, "importance": "HIGH", "category": "GROWTH", "volatility_factor": 1.8
            },
            # Unemployment Rate
            "UNRATE": { 
                "name": "Unemployment Rate", "series_id": "UNRATE", # Civilian Unemployment Rate
                "impact_score": 80, "importance": "HIGH", "category": "EMPLOYMENT", "volatility_factor": 1.5
            },
            # Personal Consumption Expenditures (PCE Price Index)
            "PCEPI": { 
                "name": "PCE Price Index", "series_id": "PCEPI", # Personal Consumption Expenditures Price Index
                "impact_score": 85, "importance": "HIGH", "category": "INFLATION", "volatility_factor": 1.7
            },
            # Produção Industrial
            "INDPRO": {
                "name": "Industrial Production Index", "series_id": "INDPRO", # Industrial Production Index, Seasonally Adjusted
                "impact_score": 75, "importance": "MEDIUM", "category": "MANUFACTURING", "volatility_factor": 1.4
            },
            # Vendas no Varejo (Retail Sales)
            "RSXFS": {
                "name": "Retail Sales: Total", "series_id": "RSXFS", # Retail Sales: Total, Seasonally Adjusted
                "impact_score": 70, "importance": "MEDIUM", "category": "CONSUMPTION", "volatility_factor": 1.3
            }
            # Adicione mais conforme necessário, consultando o site do FRED para os `series_id`
        }
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Faz requisição para a API do FRED com tratamento de erros."""
        try:
            params.update({
                "api_key": self.api_key,
                "file_type": "json"
            })
            
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=10) # Adiciona timeout
            response.raise_for_status() # Lança HTTPError para 4xx/5xx responses
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Requisição FRED excedeu o tempo limite para {endpoint}.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de requisição FRED para {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON da resposta FRED para {endpoint}: {e}. Resposta: {response.text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado em _make_request para {endpoint}: {e}")
            return None

    def _get_series_observations(self, series_id: str, limit: int = 2) -> List[Dict]:
        """
        Busca as últimas observações para uma série específica do FRED.
        Args:
            series_id: O ID da série FRED (ex: "CPIAUCSL").
            limit: Número de observações mais recentes a buscar (para previous e actual).
        Returns:
            Lista de dicionários de observações.
        """
        params = {
            "series_id": series_id,
            "sort_order": "desc", # Ordem decrescente de data
            "limit": limit
        }
        data = self._make_request("series/observations", params)
        if data and "observations" in data:
            return data["observations"]
        return []

    def _get_release_info(self, release_id: str) -> Optional[Dict]:
        """Busca informações detalhadas de um release (ex: frequência)."""
        params = {"release_id": release_id}
        data = self._make_request("release", params)
        if data and "releases" in data and data["releases"]:
            return data["releases"][0]
        return None
    
    def get_upcoming_releases(self, days_ahead: int = 14) -> List[EconomicEvent]:
        """
        Obtém próximos releases econômicos importantes, incluindo dados anteriores/atuais.
        Utiliza cache para otimização.
        """
        # Verifica o cache
        if self.cache["last_full_update"] and \
           (datetime.now() - self.cache["last_full_update"]).total_seconds() < self.cache["cache_duration_seconds"]:
            logger.info("Retornando eventos FRED do cache.")
            return self.cache["upcoming_events"]

        logger.info(f"Buscando novos eventos FRED para os próximos {days_ahead} dias.")
        try:
            start_date_str = datetime.now().strftime("%Y-%m-%d")
            end_date_str = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            # Obter datas de releases de todas as séries (não apenas as críticas)
            # A API do FRED não permite filtrar por "importância" diretamente para releases/dates
            # Então, vamos buscar por "release dates" e depois tentar correlacionar com nossos `critical_release_series`
            params = {
                "realtime_start": start_date_str,
                "realtime_end": end_date_str,
                "include_release_dates_with_no_data": "false" # Apenas datas com dados
            }
            
            data = self._make_request("releases/dates", params)
            
            if not data or "release_dates" not in data:
                logger.warning("Nenhum dado de 'releases/dates' recebido do FRED.")
                self.cache["upcoming_events"] = [] # Limpa cache se falhar
                self.cache["last_full_update"] = datetime.now()
                return []
            
            events = []
            
            # Mapeia nomes de releases críticos para seus metadados (para classificação rápida)
            critical_names_map = {v["name"].lower(): k for k, v in self.critical_release_series.items()}

            for release in data["release_dates"]:
                release_id = release.get("release_id")
                release_name = release.get("release_name", "Unknown")
                release_date_str = release.get("date", "1970-01-01")
                
                # Tenta normalizar a data para ter um `datetime.date` para comparação
                try:
                    release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Data de release inválida: {release_date_str}. Pulando evento.")
                    continue

                # Classifica o release e tenta obter o series_id se for crítico
                importance = "LOW"
                impact_score = 30
                category = "GENERAL"
                series_id = None
                
                # Procura o release na nossa lista de críticos
                matched_series_key = critical_names_map.get(release_name.lower())
                if matched_series_key and matched_series_key in self.critical_release_series:
                    critical_info = self.critical_release_series[matched_series_key]
                    importance = critical_info["importance"]
                    impact_score = critical_info["impact_score"]
                    category = critical_info["category"]
                    series_id = critical_info["series_id"]
                
                # Busca as últimas observações se tivermos um series_id
                actual_val: Optional[float] = None
                previous_val: Optional[float] = None
                forecast_val: Optional[float] = None # FRED não tem forecast direto
                
                if series_id:
                    observations = self._get_series_observations(series_id, limit=2)
                    if observations:
                        # A observação mais recente é o 'actual'
                        try:
                            actual_val = float(observations[0]["value"])
                        except (ValueError, TypeError):
                            actual_val = None
                        
                        # A segunda observação mais recente é o 'previous'
                        if len(observations) > 1:
                            try:
                                previous_val = float(observations[1]["value"])
                            except (ValueError, TypeError):
                                previous_val = None

                # Tenta buscar frequência do release
                frequency = "Unknown"
                release_details = self._get_release_info(release_id)
                if release_details and "frequency" in release_details:
                    frequency = release_details["frequency"]

                event_obj = EconomicEvent(
                    release_id=release_id,
                    name=release_name,
                    date=release_date,
                    time=self._estimate_release_time(category),
                    importance=importance,
                    frequency=frequency,
                    series_id=series_id,
                    previous_value=previous_val,
                    forecast=forecast_val, # Permanece None sem fonte externa
                    actual=actual_val,
                    impact_score=impact_score,
                    category=category
                )
                
                events.append(event_obj)
            
            # Ordena por data e impacto
            events.sort(key=lambda x: (x.date, -x.impact_score))
            
            logger.info(f"Coletados e enriquecidos {len(events)} eventos econômicos próximos do FRED.")
            self.cache["upcoming_events"] = events
            self.cache["last_full_update"] = datetime.now()
            return events
            
        except Exception as e:
            logger.error(f"Erro ao obter e processar releases futuros do FRED: {e}")
            return []
    
    def _classify_release(self, release_name: str, release_id: str) -> tuple:
        """
        Classifica a importância e categoria de um release com base no nome e IDs críticos pré-definidos.
        Esta função é uma fallback/complemento para a lógica em `get_upcoming_releases`
        que já tenta usar `self.critical_release_series`.
        """
        name_lower = release_name.lower()
        
        # Procura na lista de releases críticos pelo nome (chave) ou nome (dentro do dict)
        for key, info in self.critical_release_series.items():
            if key.lower() in name_lower or info["name"].lower() in name_lower:
                return info["importance"], info["impact_score"], info["category"]
        
        # Fallback para classificação genérica se não for um dos críticos principais
        if any(keyword in name_lower for keyword in [
            "employment", "payroll", "jobs", "unemployment",
            "inflation", "cpi", "pce", "price index",
            "federal funds", "fomc", "fed", "interest rate",
            "gdp", "gross domestic"
        ]):
            return "HIGH", 80, self._get_category(name_lower)
        
        elif any(keyword in name_lower for keyword in [
            "housing", "retail", "consumer", "industrial",
            "durable goods", "trade", "treasury", "bond"
        ]):
            return "MEDIUM", 55, self._get_category(name_lower)
        
        else:
            return "LOW", 25, "GENERAL"
    
    def _get_category(self, name_lower: str) -> str:
        """Determina a categoria do evento."""
        if any(word in name_lower for word in ["employment", "payroll", "jobs", "unemployment", "labor"]):
            return "EMPLOYMENT"
        elif any(word in name_lower for word in ["inflation", "cpi", "pce", "price"]):
            return "INFLATION"
        elif any(word in name_lower for word in ["fed", "fomc", "funds", "rate", "monetary"]):
            return "MONETARY_POLICY"
        elif any(word in name_lower for word in ["gdp", "growth", "economic", "production"]):
            return "GROWTH"
        elif any(word in name_lower for word in ["housing", "home", "building"]):
            return "HOUSING"
        elif any(word in name_lower for word in ["trade", "export", "import", "balance"]):
            return "TRADE"
        elif any(word in name_lower for word in ["manufacturing", "industrial", "ism"]):
            return "MANUFACTURING"
        elif any(word in name_lower for word in ["consumer", "retail", "sales"]):
            return "CONSUMPTION"
        else:
            return "GENERAL"
    
    def _estimate_release_time(self, category: str) -> str:
        """Estima horário típico de release baseado na categoria (Horário de Nova York - ET)."""
        time_map = {
            "EMPLOYMENT": "08:30 ET",  
            "INFLATION": "08:30 ET",   
            "MONETARY_POLICY": "14:00 ET", 
            "GROWTH": "08:30 ET",      
            "HOUSING": "10:00 ET",     
            "TRADE": "08:30 ET",       
            "MANUFACTURING": "10:00 ET",
            "CONSUMPTION": "08:30 ET",
            "GENERAL": "10:00 ET"      
        }
        return time_map.get(category, "10:00 ET")
    
    def get_high_impact_events_today(self) -> List[EconomicEvent]:
        """Obtém eventos de alto impacto para hoje."""
        today_date = datetime.now().date()
        upcoming = self.get_upcoming_releases(days_ahead=1) # Busca releases até amanhã para pegar os de hoje
        
        return [
            event for event in upcoming 
            if event.date.date() == today_date and event.importance == "HIGH"
        ]
    
    def get_next_critical_event(self) -> Optional[EconomicEvent]:
        """Obtém o próximo evento crítico (impact_score >= 80)."""
        upcoming = self.get_upcoming_releases(days_ahead=30)
        critical_events = [
            event for event in upcoming 
            if event.impact_score >= 80 and event.date.date() >= datetime.now().date() # A partir de hoje
        ]
        # Retorna o primeiro evento crítico futuro, se houver
        return critical_events[0] if critical_events else None
    
    def generate_pre_event_alerts(self, hours_before: int = 24) -> List[Dict]:
        """Gera alertas preventivos antes de eventos importantes."""
        alerts = []
        upcoming = self.get_upcoming_releases(days_ahead=7) # Busca eventos para os próximos 7 dias
        
        for event in upcoming:
            # Considera eventos de alta ou média importância
            if event.importance in ["HIGH", "MEDIUM"]:
                
                # Calcula a diferença até a data do evento (desconsidera o tempo se não estiver na string de tempo)
                # Para ser mais preciso, deveria ter um timestamp completo do release
                # Por enquanto, compara apenas a data
                
                # Se o evento é hoje ou no futuro próximo
                if event.date.date() >= datetime.now().date():
                    
                    # Constrói o datetime completo para o evento (data + hora estimada)
                    event_datetime_str = f"{event.date.strftime('%Y-%m-%d')} {event.time.split(' ')[0]}" # Remove 'ET'
                    try:
                        event_datetime = datetime.strptime(event_datetime_str, "%Y-%m-%d %H:%M")
                    except ValueError:
                        logger.warning(f"Não foi possível parsear o tempo do evento {event.name}: {event_datetime_str}. Ignorando horário.")
                        event_datetime = event.date # Usa só a data

                    time_until_release = event_datetime - datetime.now()
                    hours_until = time_until_release.total_seconds() / 3600

                    # Gera alerta se o evento está dentro da janela `hours_before` e ainda não passou
                    if 0 < hours_until <= hours_before:
                        
                        alert = {
                            "type": "PRE_EVENT_ALERT",
                            "title": f"⏰ Evento Econômico Importante: {event.name}",
                            "message": f"Divulgação de {event.name} ({event.importance} Impacto) em {hours_until:.1f} horas ({event_datetime.strftime('%d/%m %H:%M %Z')}).",
                            "severity": "MEDIUM" if hours_until > 6 else "HIGH", # Mais perto, maior a severidade
                            "timestamp": datetime.now().isoformat(),
                            "event_data": {
                                "event_name": event.name,
                                "event_date": event.date.isoformat(),
                                "event_time": event.time,
                                "importance": event.importance,
                                "impact_score": event.impact_score,
                                "category": event.category,
                                "hours_until": hours_until,
                                "series_id": event.series_id,
                                "previous_value": event.previous_value,
                                "actual_value": event.actual, # O 'actual' será o último disponível antes do release
                                "forecast_value": event.forecast # Será None
                            },
                            "recommendations": self._generate_recommendations(event, hours_until)
                        }
                        
                        alerts.append(alert)
        
        return alerts
    
    def _generate_recommendations(self, event: EconomicEvent, hours_until: float) -> List[str]:
        """Gera recomendações de trading baseadas no tipo de evento e iminência."""
        recommendations = []
        
        if event.category == "EMPLOYMENT":
            recommendations.extend([
                "🔍 Monitorar DXY/BTC próximo ao NFP (Non-Farm Payrolls).",
                "📈 Esperar alta volatilidade em todos os ativos de risco.",
                "⚠️ Possível reversão de tendências se dados surpreenderem."
            ])
        elif event.category == "INFLATION":
            recommendations.extend([
                "🥇 Ouro pode ter movimento forte com dados de inflação (CPI/PCE).",
                "💵 DXY sensível a surpresas nos índices de preços.",
                "📊 Observar padrões de divergência DXY/BTC em torno da inflação."
            ])
        elif event.category == "MONETARY_POLICY":
            recommendations.extend([
                "🏛️ Decisões do FOMC (FED) podem gerar padrões angulares extremos.",
                "⚡ Volatilidade máxima esperada em todos os ativos.",
                "🎯 Monitorar 'perfect divergence' DXY/BTC com declarações do FED."
            ])
        elif event.category == "GROWTH":
            recommendations.extend([
                "📊 Dados de PIB (GDP) impactam diretamente o sentimento de risco.",
                "📈 Abertura de novas posições com cautela após o release."
            ])
        
        if hours_until <= 0.5: # 30 minutos antes
            recommendations.append("🚨 ALERTA CRÍTICO: Evento iminente em menos de 30 minutos - MÁXIMA ATENÇÃO!.")
        elif hours_until <= 2: # 2 horas antes
            recommendations.append("⚠️ ALERTA: Evento importante se aproxima - preparar para volatilidade.")
        elif hours_until <= 12: # 12 horas antes
            recommendations.append("🔔 Aviso: Evento de alto impacto nas próximas 12 horas.")
        
        return recommendations
    
    def correlate_with_angular_data(self, events: List[EconomicEvent], 
                                   angular_cache: Dict) -> Dict:
        """
        Correlaciona eventos econômicos com movimentos angulares históricos.
        Ainda em desenvolvimento, para um radar profissional real, esta lógica seria mais complexa.
        """
        correlations = {
            "event_impact_analysis": [],
            "volatility_predictions": {}, # Ex: prever se o evento causará alta ou baixa volatilidade
            "pattern_likelihood": {} # Ex: probabilidade de um certo padrão angular após o evento
        }
        
        try:
            # angular_history será uma lista de dicionários com 'timestamp', 'btc', 'gold', 'dxy'
            angular_history = angular_cache.get("angular_data", []) 

            # Converte a lista de dicionários para DataFrame para facilitar a busca por datas
            if angular_history:
                angular_df = pd.DataFrame(angular_history)
                # Garante que o timestamp é datetime e define como índice
                angular_df['timestamp'] = pd.to_datetime(angular_df['timestamp'])
                angular_df.set_index('timestamp', inplace=True)
                angular_df.sort_index(inplace=True)
            else:
                angular_df = pd.DataFrame()
            
            for event in events:
                event_date = event.date.date()
                
                if not angular_df.empty:
                    # Filtra dados angulares para o dia do evento ou nas 24h anteriores/posteriores
                    # Para simplificar, pegamos o dia do evento
                    event_day_data = angular_df.loc[
                        angular_df.index.date == event_date
                    ]
                    
                    if not event_day_data.empty:
                        # Analisa volatilidade angular no dia do evento (amplitude ou desvio padrão dos ângulos)
                        max_angle_change = 0
                        
                        for idx, data_row in event_day_data.iterrows():
                            for asset_key in ["btc", "gold", "dxy"]:
                                if asset_key in data_row and isinstance(data_row[asset_key], dict) and "angle" in data_row[asset_key]:
                                    angle_val = data_row[asset_key]["angle"]
                                    # Pega o ângulo absoluto como uma medida simples de volatilidade angular
                                    if abs(angle_val) > max_angle_change:
                                        max_angle_change = abs(angle_val)
                        
                        correlations["event_impact_analysis"].append({
                            "event_name": event.name,
                            "date": event_date.isoformat(),
                            "max_angular_volatility_observed": max_angle_change,
                            "event_impact_score": event.impact_score,
                            "event_category": event.category
                        })
            
            logger.info(f"Correlação angular concluída para {len(events)} eventos.")
            
        except Exception as e:
            logger.error(f"Erro na correlação angular: {e}")
        
        return correlations

# Função de utilidade para integração com o sistema principal
def integrate_fred_calendar(api_key: str, angular_cache: Dict) -> Dict:
    """
    Integra o calendário FRED e seus alertas com o sistema principal.
    Args:
        api_key (str): Sua chave de API do FRED.
        angular_cache (Dict): Cache de dados angulares (para correlação).
    Returns:
        Dict: Um dicionário contendo o calendário econômico e alertas.
    """
    try:
        calendar = FREDEconomicCalendar(api_key)
        
        # Obtém próximos eventos
        upcoming_events = calendar.get_upcoming_releases(days_ahead=14)
        
        # Gera alertas preventivos (se houver events válidos)
        pre_event_alerts = calendar.generate_pre_event_alerts(hours_before=48) if upcoming_events else []
        
        # Evento crítico próximo (se houver events válidos)
        next_critical = calendar.get_next_critical_event() if upcoming_events else None
        
        # Eventos importantes hoje (se houver events válidos)
        today_events = calendar.get_high_impact_events_today() if upcoming_events else []
        
        # Correlação com dados angulares
        correlations = calendar.correlate_with_angular_data(
            upcoming_events, 
            angular_cache # Passa o dicionário completo do cache angular
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "upcoming_events": [
                {
                    "name": event.name,
                    "date": event.date.isoformat(),
                    "time": event.time,
                    "importance": event.importance,
                    "impact_score": event.impact_score,
                    "category": event.category,
                    "days_until": (event.date.date() - datetime.now().date()).days if event.date.date() >= datetime.now().date() else 0,
                    "series_id": event.series_id,
                    "previous_value": event.previous_value,
                    "forecast": event.forecast,
                    "actual": event.actual
                }
                for event in upcoming_events[:20]  # Limita para o frontend
            ],
            "pre_event_alerts": pre_event_alerts,
            "next_critical_event": {
                "name": next_critical.name,
                "date": next_critical.date.isoformat(),
                "time": next_critical.time,
                "impact_score": next_critical.impact_score,
                "days_until": (next_critical.date.date() - datetime.now().date()).days if next_critical.date.date() >= datetime.now().date() else 0,
                "series_id": next_critical.series_id,
                "previous_value": next_critical.previous_value,
                "forecast": next_critical.forecast,
                "actual": next_critical.actual
            } if next_critical else None,
            "today_events": [
                {
                    "name": event.name,
                    "time": event.time,
                    "impact_score": event.impact_score,
                    "category": event.category,
                    "series_id": event.series_id,
                    "previous_value": event.previous_value,
                    "forecast": event.forecast,
                    "actual": event.actual
                }
                for event in today_events
            ],
            "correlations": correlations,
            "summary": {
                "total_upcoming": len(upcoming_events),
                "high_impact": len([e for e in upcoming_events if e.importance == "HIGH"]),
                "today_high_impact": len(today_events),
                "pre_alerts_count": len(pre_event_alerts)
            }
        }
        
    except Exception as e:
        logger.error(f"Erro na integração FRED: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "upcoming_events": [], "pre_event_alerts": [], "next_critical_event": None,
            "today_events": [], "correlations": {}, "summary": {} # Garante que as chaves existem
        }

# Exemplo de uso
if __name__ == "__main__":
    # Teste básico (requer API key do FRED em config.py)
    # CRIE UM ARQUIVO config.py NA MESMA PASTA COM:
    # FRED_API_KEY = "SUA_CHAVE_API_FRED_AQUI"
    # Você pode obter uma chave em https://fred.stlouisfed.org/docs/api/api_key.html
    
    # Simula cache angular (para teste de correlação)
    mock_angular_cache = {
        "angular_data": [
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "btc": {"angle": 25.5, "strength": 0.8},
                "gold": {"angle": -15.2, "strength": 0.6},
                "dxy": {"angle": 30.1, "strength": 0.9}
            },
             {
                "timestamp": (datetime.now() - timedelta(hours=0.5)).isoformat(),
                "btc": {"angle": 10.1, "strength": 0.7},
                "gold": {"angle": -5.0, "strength": 0.5},
                "dxy": {"angle": 15.0, "strength": 0.8}
            }
        ]
    }
    
    try:
        # Tenta carregar a API key do config.py
        from config import FRED_API_KEY
    except ImportError:
        logger.error("Arquivo config.py não encontrado ou FRED_API_KEY não definida. Por favor, crie config.py.")
        FRED_API_KEY = "DUMMY_KEY" # Chave dummy para evitar erro, mas não funcionará
    
    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE" or FRED_API_KEY == "DUMMY_KEY":
        logger.warning("FRED_API_KEY não configurada. A funcionalidade do FRED não irá operar corretamente.")

    result = integrate_fred_calendar(FRED_API_KEY, mock_angular_cache)
    print(json.dumps(result, indent=2, default=str))