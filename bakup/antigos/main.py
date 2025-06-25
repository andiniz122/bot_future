#!/usr/bin/env python3
"""
🤖 TRADING BOT - Sistema Integrador Principal
Orquestra todos os componentes do sistema de trading autônomo
"""

import asyncio
import logging
import signal
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import traceback
import time

# Imports dos módulos do sistema
from advanced_market_analyzer import AdvancedMarketAnalyzer
from autonomous_strategy_engine import AutonomousTradingSystem  
from adaptive_ml_system import AdaptiveMLSystem
from data_system import AdvancedDataSystem, DataConfig
from portfolio_system import PortfolioManager, RiskConfig

# =====================================================================
# CONFIGURAÇÃO GLOBAL DO SISTEMA
# =====================================================================

@dataclass
class BotConfig:
    """Configuração principal do bot"""
    # Configurações gerais
    bot_name: str = "AdvancedTradingBot"
    version: str = "1.0.0"
    environment: str = "testnet"  # testnet ou mainnet
    
    # Credenciais da API
    api_key: str = ""
    secret_key: str = ""
    
    # Configurações de execução
    main_loop_interval: int = 60          # Segundos entre ciclos principais
    data_update_interval: int = 30        # Segundos entre updates de dados
    ml_adaptation_interval: int = 3600    # Segundos entre adaptações ML (1h)
    portfolio_check_interval: int = 10    # Segundos entre checks de portfolio
    
    # Configurações de dados
    symbols_to_trade: List[str] = field(default_factory=lambda: [
        'BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'ADA_USDT', 'DOT_USDT'
    ])
    min_volume_filter: float = 5_000_000  # Volume mínimo em USDT
    max_symbols_active: int = 20
    
    # Configurações de portfolio
    initial_balance: float = 10000.0
    max_portfolio_risk: float = 0.02      # 2% por trade
    max_daily_loss: float = 0.05          # 5% perda diária máxima
    max_total_positions: int = 8
    
    # Configurações de logging
    log_level: str = "INFO"
    log_to_file: bool = True
    max_log_files: int = 30
    
    # Configurações de backup
    backup_interval_hours: int = 6
    max_backups: int = 10
    
    # Configurações de alertas
    enable_alerts: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Configurações de segurança
    emergency_stop_enabled: bool = True
    max_consecutive_losses: int = 5
    min_success_rate: float = 0.3
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'BotConfig':
        """Carrega configuração de arquivo JSON"""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logging.error(f"Erro carregando config: {e}")
            return cls()
    
    def save_to_file(self, config_path: str):
        """Salva configuração em arquivo JSON"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.__dict__, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Erro salvando config: {e}")

# =====================================================================
# SISTEMA DE ALERTAS
# =====================================================================

class AlertSystem:
    """Sistema de alertas críticos"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.alert_history = []
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutos entre alertas similares
        
    async def send_alert(self, level: str, message: str, details: Dict = None):
        """Envia alerta com rate limiting"""
        
        alert_key = f"{level}_{message[:50]}"
        current_time = time.time()
        
        # Rate limiting
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = current_time
        
        # Log do alerta
        if level == "CRITICAL":
            logging.critical(f"🚨 {message}")
        elif level == "WARNING":
            logging.warning(f"⚠️ {message}")
        else:
            logging.info(f"ℹ️ {message}")
        
        # Salvar histórico
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        self.alert_history.append(alert_record)
        
        # Manter apenas últimos 1000 alertas
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Enviar por Telegram se configurado
        if self.config.enable_alerts and self.config.telegram_bot_token:
            await self._send_telegram_alert(level, message, details)
    
    async def _send_telegram_alert(self, level: str, message: str, details: Dict):
        """Envia alerta via Telegram"""
        # Implementação do Telegram seria aqui
        # Por enquanto apenas log
        logging.info(f"📱 Telegram Alert: {level} - {message}")
    
    async def check_system_health(self, systems: Dict[str, Any]):
        """Verifica saúde geral do sistema"""
        
        alerts = []
        
        # Check Portfolio
        if 'portfolio' in systems:
            portfolio = systems['portfolio']
            summary = portfolio.get_portfolio_summary()
            
            if summary['risk']['current_drawdown'] > 0.1:
                alerts.append(("WARNING", f"Alto drawdown: {summary['risk']['current_drawdown']:.2%}"))
            
            if summary['trading']['win_rate'] < 30 and summary['trading']['total_trades'] > 10:
                alerts.append(("CRITICAL", f"Win rate baixo: {summary['trading']['win_rate']:.1f}%"))
        
        # Check Data System  
        if 'data_system' in systems:
            data_stats = systems['data_system'].get_system_stats()
            
            if data_stats['api_stats']['errors'] > 20:
                alerts.append(("WARNING", f"Muitos erros de API: {data_stats['api_stats']['errors']}"))
            
            if not data_stats['websocket_connected']:
                alerts.append(("CRITICAL", "WebSocket desconectado!"))
        
        # Check ML System
        if 'ml_system' in systems:
            ml_stats = systems['ml_system'].get_learning_statistics()
            success_rate = ml_stats['learning_state'].get('successful_predictions', 0) / max(1, ml_stats['learning_state'].get('total_trades_learned', 1))
            
            if success_rate < 0.4 and ml_stats['learning_state'].get('total_trades_learned', 0) > 20:
                alerts.append(("WARNING", f"ML performance baixa: {success_rate:.2%}"))
        
        # Enviar alertas
        for level, message in alerts:
            await self.send_alert(level, message)

# =====================================================================
# SISTEMA DE BACKUP
# =====================================================================

class BackupSystem:
    """Sistema de backup automático"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.last_backup = None
        
    async def create_backup(self, systems: Dict[str, Any]):
        """Cria backup completo do sistema"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # Backup Portfolio
            if 'portfolio' in systems:
                portfolio_data = systems['portfolio'].get_portfolio_summary()
                with open(backup_path / "portfolio.json", 'w') as f:
                    json.dump(portfolio_data, f, indent=2, default=str)
            
            # Backup ML System
            if 'ml_system' in systems:
                await systems['ml_system'].save_models()
                ml_stats = systems['ml_system'].get_learning_statistics()
                with open(backup_path / "ml_stats.json", 'w') as f:
                    json.dump(ml_stats, f, indent=2, default=str)
            
            # Backup Strategy System
            if 'strategy_system' in systems:
                strategy_status = systems['strategy_system'].get_system_status()
                with open(backup_path / "strategies.json", 'w') as f:
                    json.dump(strategy_status, f, indent=2, default=str)
            
            # Backup Configuração
            self.config.save_to_file(str(backup_path / "config.json"))
            
            self.last_backup = datetime.now()
            
            logging.info(f"✅ Backup criado: {backup_path}")
            
            # Limpar backups antigos
            await self._cleanup_old_backups()
            
        except Exception as e:
            logging.error(f"❌ Erro criando backup: {e}")
    
    async def _cleanup_old_backups(self):
        """Remove backups antigos"""
        
        try:
            backup_dirs = sorted([d for d in self.backup_dir.iterdir() if d.is_dir()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(backup_dirs) > self.config.max_backups:
                for old_backup in backup_dirs[self.config.max_backups:]:
                    import shutil
                    shutil.rmtree(old_backup)
                    logging.info(f"🗑️ Backup antigo removido: {old_backup}")
                    
        except Exception as e:
            logging.error(f"Erro limpando backups: {e}")
    
    def should_backup(self) -> bool:
        """Verifica se deve fazer backup"""
        if not self.last_backup:
            return True
        
        hours_since_backup = (datetime.now() - self.last_backup).total_seconds() / 3600
        return hours_since_backup >= self.config.backup_interval_hours

# =====================================================================
# SISTEMA INTEGRADOR PRINCIPAL
# =====================================================================

class TradingBotOrchestrator:
    """Orquestrador principal do sistema de trading"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.systems = {}
        self.running = False
        self.start_time = None
        
        # Componentes auxiliares
        self.alert_system = AlertSystem(config)
        self.backup_system = BackupSystem(config)
        
        # Estatísticas
        self.cycle_count = 0
        self.last_cycle_time = None
        self.error_count = 0
        self.last_errors = []
        
        # Emergency stop
        self.emergency_stop = False
        self.consecutive_errors = 0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configura sistema de logging"""
        
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Configurar formatação
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # File handler se habilitado
        handlers = [console_handler]
        
        if self.config.log_to_file:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configurar logging root
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        # Silenciar logs verbosos de terceiros
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        
    async def initialize_systems(self):
        """Inicializa todos os sistemas"""
        
        logging.info("🚀 Inicializando sistemas do trading bot...")
        
        try:
            # 1. Sistema de Dados
            data_config = DataConfig(
                cache_directory="./cache",
                requests_per_second=8.0,
                max_cache_size_mb=500
            )
            
            self.systems['data_system'] = AdvancedDataSystem(
                config=data_config,
                api_key=self.config.api_key,
                secret_key=self.config.secret_key
            )
            await self.systems['data_system'].start()
            logging.info("✅ Sistema de dados inicializado")
            
            # 2. Analisador de Mercado
            self.systems['market_analyzer'] = AdvancedMarketAnalyzer()
            logging.info("✅ Analisador de mercado inicializado")
            
            # 3. Sistema ML
            self.systems['ml_system'] = AdaptiveMLSystem(save_directory="ml_models")
            await self.systems['ml_system'].load_models()
            logging.info("✅ Sistema ML inicializado")
            
            # 4. Portfolio Manager
            risk_config = RiskConfig(
                max_portfolio_risk=self.config.max_portfolio_risk,
                max_daily_loss=self.config.max_daily_loss,
                max_total_positions=self.config.max_total_positions
            )
            
            self.systems['portfolio'] = PortfolioManager(
                config=risk_config,
                initial_balance=self.config.initial_balance
            )
            logging.info("✅ Portfolio manager inicializado")
            
            # 5. Sistema de Estratégias Autônomo
            self.systems['strategy_system'] = AutonomousTradingSystem(
                data_collector=self.systems['data_system'],
                portfolio_manager=self.systems['portfolio'],
                gate_api=self.systems['data_system'].api_client
            )
            await self.systems['strategy_system'].initialize()
            logging.info("✅ Sistema de estratégias inicializado")
            
            # Verificar saúde inicial
            await self.alert_system.check_system_health(self.systems)
            
            logging.info("🎉 Todos os sistemas inicializados com sucesso!")
            
        except Exception as e:
            logging.critical(f"❌ Erro crítico na inicialização: {e}")
            await self.alert_system.send_alert("CRITICAL", f"Falha na inicialização: {str(e)}")
            raise
    
    async def main_trading_loop(self):
        """Loop principal de trading"""
        
        logging.info("🔄 Iniciando loop principal de trading...")
        
        # Timers para diferentes tarefas
        last_data_update = 0
        last_ml_adaptation = 0
        last_portfolio_check = 0
        last_health_check = 0
        
        while self.running and not self.emergency_stop:
            try:
                cycle_start = time.time()
                self.cycle_count += 1
                
                current_time = time.time()
                
                # 1. Update de dados (a cada X segundos)
                if current_time - last_data_update >= self.config.data_update_interval:
                    await self._update_market_data()
                    last_data_update = current_time
                
                # 2. Check de portfolio (mais frequente)
                if current_time - last_portfolio_check >= self.config.portfolio_check_interval:
                    await self._check_portfolio_exits()
                    last_portfolio_check = current_time
                
                # 3. Ciclo principal de estratégias
                await self._run_strategy_cycle()
                
                # 4. Adaptação ML (menos frequente)
                if current_time - last_ml_adaptation >= self.config.ml_adaptation_interval:
                    await self._run_ml_adaptation()
                    last_ml_adaptation = current_time
                
                # 5. Health check (a cada 5 minutos)
                if current_time - last_health_check >= 300:
                    await self._system_health_check()
                    last_health_check = current_time
                
                # 6. Backup automático
                if self.backup_system.should_backup():
                    await self.backup_system.create_backup(self.systems)
                
                # 7. Log de estatísticas
                cycle_time = time.time() - cycle_start
                self.last_cycle_time = cycle_time
                
                if self.cycle_count % 10 == 0:  # Log a cada 10 ciclos
                    await self._log_system_stats()
                
                # Reset contador de erros consecutivos
                self.consecutive_errors = 0
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.config.main_loop_interval)
                
            except Exception as e:
                await self._handle_main_loop_error(e)
    
    async def _update_market_data(self):
        """Atualiza dados de mercado"""
        
        try:
            # Obter símbolos ativos
            if not hasattr(self, '_active_symbols'):
                self._active_symbols = await self.systems['data_system'].get_available_symbols(
                    min_volume=self.config.min_volume_filter
                )
                self._active_symbols = self._active_symbols[:self.config.max_symbols_active]
                logging.info(f"📊 Símbolos ativos: {len(self._active_symbols)}")
            
            # Obter preços atuais
            current_prices = {}
            for symbol in self._active_symbols:
                ticker = await self.systems['data_system'].get_live_ticker(symbol)
                if ticker:
                    current_prices[symbol] = ticker.price
            
            # Atualizar portfolio com preços
            if current_prices:
                await self.systems['portfolio'].update_prices(current_prices)
            
        except Exception as e:
            logging.error(f"Erro atualizando dados de mercado: {e}")
            raise
    
    async def _check_portfolio_exits(self):
        """Verifica e executa saídas de posições"""
        
        try:
            exits_processed = await self.systems['portfolio'].process_exits()
            
            if exits_processed:
                logging.info(f"🔄 {len(exits_processed)} posições fechadas")
                
                # Aprender com resultados
                for trade in exits_processed:
                    if 'ml_system' in self.systems:
                        # Criar features dummy para aprendizado
                        features_df = await self._create_trade_features(trade)
                        
                        await self.systems['ml_system'].learn_from_trade_result(
                            symbol=trade.symbol,
                            features=features_df,
                            action=trade.side.value,
                            result=trade.pnl_percentage / 100
                        )
            
        except Exception as e:
            logging.error(f"Erro verificando saídas: {e}")
            raise
    
    async def _run_strategy_cycle(self):
        """Executa ciclo principal das estratégias"""
        
        try:
            await self.systems['strategy_system'].run_autonomous_cycle()
            
        except Exception as e:
            logging.error(f"Erro no ciclo de estratégias: {e}")
            raise
    
    async def _run_ml_adaptation(self):
        """Executa adaptação do sistema ML"""
        
        try:
            # ML system já faz adaptação automática quando recebe feedback
            # Aqui podemos fazer adaptações adicionais se necessário
            
            ml_stats = self.systems['ml_system'].get_learning_statistics()
            logging.info(f"🧠 ML Stats: {ml_stats['learning_state']}")
            
        except Exception as e:
            logging.error(f"Erro na adaptação ML: {e}")
            raise
    
    async def _system_health_check(self):
        """Verifica saúde geral do sistema"""
        
        try:
            await self.alert_system.check_system_health(self.systems)
            
            # Check emergency stop conditions
            portfolio_summary = self.systems['portfolio'].get_portfolio_summary()
            
            # Condições de emergency stop
            if self.config.emergency_stop_enabled:
                drawdown = portfolio_summary['risk']['current_drawdown']
                win_rate = portfolio_summary['trading']['win_rate']
                total_trades = portfolio_summary['trading']['total_trades']
                
                if drawdown > 0.2:  # 20% drawdown
                    await self._trigger_emergency_stop(f"Alto drawdown: {drawdown:.2%}")
                elif win_rate < 20 and total_trades > 20:  # Win rate < 20% com trades suficientes
                    await self._trigger_emergency_stop(f"Win rate muito baixo: {win_rate:.1f}%")
                elif self.consecutive_errors >= self.config.max_consecutive_losses:
                    await self._trigger_emergency_stop(f"Muitos erros consecutivos: {self.consecutive_errors}")
            
        except Exception as e:
            logging.error(f"Erro no health check: {e}")
            raise
    
    async def _create_trade_features(self, trade) -> 'pd.DataFrame':
        """Cria features para aprendizado ML a partir de um trade"""
        
        try:
            # Obter dados históricos do símbolo
            klines = await self.systems['data_system'].get_klines_data(
                symbol=trade.symbol,
                timeframe="1h",
                limit=50
            )
            
            if klines is not None and not klines.empty:
                # Usar apenas última linha como features
                return klines.tail(1)
            else:
                # Features dummy se não tiver dados
                import pandas as pd
                return pd.DataFrame({
                    'close': [trade.entry_price],
                    'volume': [1000]
                })
                
        except Exception as e:
            logging.error(f"Erro criando features para trade: {e}")
            import pandas as pd
            return pd.DataFrame({'close': [0], 'volume': [0]})
    
    async def _handle_main_loop_error(self, error: Exception):
        """Trata erros do loop principal"""
        
        self.error_count += 1
        self.consecutive_errors += 1
        
        error_msg = f"Erro no loop principal (#{self.error_count}): {str(error)}"
        
        # Log detalhado do erro
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Adicionar ao histórico de erros
        self.last_errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'traceback': traceback.format_exc(),
            'cycle': self.cycle_count
        })
        
        # Manter apenas últimos 50 erros
        if len(self.last_errors) > 50:
            self.last_errors = self.last_errors[-50:]
        
        # Enviar alerta
        await self.alert_system.send_alert(
            "CRITICAL" if self.consecutive_errors >= 3 else "WARNING",
            error_msg,
            {'consecutive_errors': self.consecutive_errors}
        )
        
        # Aguardar antes de tentar novamente
        await asyncio.sleep(min(30, self.consecutive_errors * 5))
    
    async def _trigger_emergency_stop(self, reason: str):
        """Dispara parada de emergência"""
        
        self.emergency_stop = True
        
        critical_msg = f"🚨 EMERGENCY STOP ATIVADO: {reason}"
        logging.critical(critical_msg)
        
        await self.alert_system.send_alert("CRITICAL", critical_msg)
        
        # Fechar todas as posições
        try:
            for position_id in list(self.systems['portfolio'].positions.keys()):
                position = self.systems['portfolio'].positions[position_id]
                await self.systems['portfolio'].close_position(
                    position_id, 
                    position.current_price, 
                    "emergency_stop"
                )
            
            logging.critical("🔴 Todas as posições foram fechadas por emergency stop")
            
        except Exception as e:
            logging.critical(f"Erro fechando posições no emergency stop: {e}")
        
        # Criar backup final
        await self.backup_system.create_backup(self.systems)
    
    async def _log_system_stats(self):
        """Log das estatísticas do sistema"""
        
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        portfolio_summary = self.systems['portfolio'].get_portfolio_summary()
        
        logging.info(f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                    📊 SYSTEM STATUS                                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║ Uptime: {str(uptime).split('.')[0]:<20} │ Ciclos: {self.cycle_count:<15} │ Erros: {self.error_count:<10} ║
║ Balance: ${portfolio_summary['balance']['current']:<18.2f} │ P&L: ${portfolio_summary['pnl']['total']:<18.2f} │ Equity: ${portfolio_summary['balance']['equity']:<12.2f} ║
║ Posições: {portfolio_summary['positions']['total']:<19} │ Win Rate: {portfolio_summary['trading']['win_rate']:<13.1f}% │ Drawdown: {portfolio_summary['risk']['current_drawdown']:<9.2%} ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    async def start(self):
        """Inicia o sistema de trading"""
        
        logging.info(f"🚀 Iniciando {self.config.bot_name} v{self.config.version}")
        logging.info(f"🌍 Ambiente: {self.config.environment}")
        
        self.start_time = datetime.now()
        self.running = True
        
        try:
            await self.initialize_systems()
            await self.main_trading_loop()
            
        except KeyboardInterrupt:
            logging.info("👋 Interrupção pelo usuário")
        except Exception as e:
            logging.critical(f"❌ Erro crítico: {e}")
            await self.alert_system.send_alert("CRITICAL", f"Sistema falhou: {str(e)}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Para o sistema graciosamente"""
        
        logging.info("🛑 Parando sistema de trading...")
        
        self.running = False
        
        try:
            # Backup final
            await self.backup_system.create_backup(self.systems)
            
            # Salvar estado dos sistemas
            if 'ml_system' in self.systems:
                await self.systems['ml_system'].save_models()
            
            if 'strategy_system' in self.systems:
                await self.systems['strategy_system'].save_system_state("strategy_state.json")
            
            # Parar sistemas
            if 'data_system' in self.systems:
                await self.systems['data_system'].stop()
            
            logging.info("✅ Sistema parado com sucesso")
            
        except Exception as e:
            logging.error(f"Erro durante shutdown: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        status = {
            'bot_info': {
                'name': self.config.bot_name,
                'version': self.config.version,
                'environment': self.config.environment,
                'uptime_seconds': uptime.total_seconds(),
                'start_time': self.start_time.isoformat() if self.start_time else None
            },
            'execution': {
                'running': self.running,
                'emergency_stop': self.emergency_stop,
                'cycle_count': self.cycle_count,
                'last_cycle_time': self.last_cycle_time,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors
            },
            'systems': {}
        }
        
        # Status dos subsistemas
        for name, system in self.systems.items():
            try:
                if hasattr(system, 'get_system_stats'):
                    status['systems'][name] = system.get_system_stats()
                elif hasattr(system, 'get_system_status'):
                    status['systems'][name] = system.get_system_status()
                elif hasattr(system, 'get_portfolio_summary'):
                    status['systems'][name] = system.get_portfolio_summary()
                else:
                    status['systems'][name] = {'status': 'running'}
            except:
                status['systems'][name] = {'status': 'error'}
        
        return status

# =====================================================================
# FUNÇÃO PRINCIPAL
# =====================================================================

async def main():
    """Função principal"""
    
    # Carregar configuração
    config_path = "config.json"
    
    if Path(config_path).exists():
        config = BotConfig.load_from_file(config_path)
        logging.info(f"✅ Configuração carregada de {config_path}")
    else:
        config = BotConfig()
        config.save_to_file(config_path)
        logging.info(f"📄 Configuração padrão criada em {config_path}")
        logging.warning("⚠️ Configure suas credenciais de API no arquivo config.json!")
    
    # Criar e iniciar bot
    bot = TradingBotOrchestrator(config)
    
    # Setup signal handlers para parada graciosaa
    def signal_handler(signum, frame):
        logging.info(f"Sinal {signum} recebido, parando bot...")
        bot.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Iniciar bot
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        sys.exit(1)