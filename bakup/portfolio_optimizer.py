#!/usr/bin/env python3
"""
Portfolio Optimizer Aprimorado - Gestão Ativa das 3 Posições Existentes
"""

import time
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio # Adicionado para asyncio.sleep

# Importações de classes que podem ser usadas aqui, mas não estavam explicitamente importadas
# Supondo que PortfolioAction e ActionType venham de algum 'models' ou similar.
# Se elas estão em 'main.py' ou outro lugar, você precisará ajustar o import.
# Por simplicidade, vou assumir que estão no escopo ou que você as definirá aqui se necessário.
# Se PortfolioAction e ActionType não estão definidos, você pode precisar adicioná-los:
# from your_module_name import PortfolioAction, ActionType
# Ou defini-los aqui se forem específicos para este arquivo:
from dataclasses import dataclass

logger = logging.getLogger('enhanced_portfolio_optimizer')

# Definindo PortfolioAction e ActionType se não vierem de outro arquivo
# Estes são assumidos com base no seu uso em main.py e nesta classe.
class ActionType:
    CLOSE_AND_REPLACE = "CLOSE_AND_REPLACE"
    ADD_POSITION = "ADD_POSITION"
    CLOSE_PARTIAL = "CLOSE_PARTIAL"
    REBALANCE = "REBALANCE" # Adicionado caso seja um tipo de ação

@dataclass
class PortfolioAction:
    action_type: str
    reasoning: str
    expected_improvement: float = 0.0
    priority_score: float = 0.0
    target_position: Optional[str] = None # Símbolo da posição a ser fechada/ajustada
    new_opportunity: Optional[str] = None  # Símbolo da nova oportunidade (se for ADD_POSITION/REPLACE)

# =====================================================================
# CORREÇÃO: DEFINIÇÃO DA CLASSE BASE PortfolioOptimizer
# =====================================================================
class PortfolioOptimizer:
    """Classe base para otimização de portfolio."""
    def __init__(self, portfolio_manager, symbol_analyzer, gate_api):
        self.portfolio_manager = portfolio_manager
        self.symbol_analyzer = symbol_analyzer
        self.gate_api = gate_api
        # O data_collector e signal_engine são normalmente acessados via main.py ou passados.
        # Para evitar imports circulares, assumimos que são passados ou obtidos via portfolio_manager
        # ou que as funções que os usam receberão eles como parâmetros se necessário.
        self.data_collector = None # Será atribuído externamente ou via main.py
        self.signal_engine = None  # Será atribuído externamente ou via main.py

    async def optimize_portfolio(self, opportunities: List[Tuple[str, Any]]) -> List[PortfolioAction]:
        """
        Método base para otimização de portfolio.
        Este método será substituído por implementações mais específicas.
        """
        logger.warning("Método optimize_portfolio da classe base PortfolioOptimizer chamado. Implementação deve ser em uma subclasse.")
        return []

# =====================================================================
# FIM DA CORREÇÃO DA CLASSE BASE
# =====================================================================

class EnhancedPortfolioOptimizer(PortfolioOptimizer):
    """Portfolio Optimizer aprimorado para gestão ativa"""
    
    def __init__(self, portfolio_manager, symbol_analyzer, gate_api):
        super().__init__(portfolio_manager, symbol_analyzer, gate_api)
        
        # Configurações mais agressivas para gestão ativa
        self.min_improvement_threshold = 10.0    # Reduzido para 10%
        self.min_hold_time_minutes = 15          # Reduzido para 15 min
        self.max_position_age_hours = 12         # Máximo 12h por posição
        self.emergency_close_threshold = -3.0    # Fechar se < -3%
        self.profit_protection_threshold = 2.0   # Proteger lucro se > 2%
        
        # Contadores para tracking
        self.positions_analyzed = 0
        self.decisions_made = 0
        self.emergency_closes = 0

    # Sobrescrevendo o método da classe base 'optimize_portfolio'
    # Esta é a função que 'main.py' espera que seja chamada.
    async def optimize_portfolio(self, current_opportunities: List = None) -> List[PortfolioAction]:
        """Otimização ativa focada na gestão das posições existentes e novas oportunidades."""
        return await self.optimize_portfolio_actively(current_opportunities)

    async def optimize_portfolio_actively(self, current_opportunities: List = None) -> List[PortfolioAction]:
        """Otimização ativa focada na gestão das posições existentes"""
        
        logger.info("🧠 OTIMIZAÇÃO ATIVA DE PORTFOLIO INICIADA")
        
        actions = []
        
        # 1. PRIORIDADE MÁXIMA: Gerenciar posições existentes
        existing_actions = await self._manage_existing_positions_actively()
        actions.extend(existing_actions)
        
        # 2. Se há espaço livre, considerar novas oportunidades
        if current_opportunities:
            # Para usar _evaluate_new_opportunities, é preciso que self.data_collector e self.signal_engine
            # estejam setados. No main.py, eles são passados para o construtor do PortfolioOptimizer.
            # Se não estiverem, adicione a atribuição no __init__ do PortfolioOptimizer ou
            # passe-os para o construtor do EnhancedPortfolioOptimizer.
            # No contexto atual, a classe EnhancedPortfolioOptimizer não tem acesso direto a eles.
            # O mais seguro é passá-los durante a inicialização em main.py ou via DI.
            # Por enquanto, vou assumir que main.py irá atribuir self.data_collector e self.signal_engine
            # à instância de EnhancedPortfolioOptimizer após a criação, ou que eles já estão no self.gate_api
            # ou symbol_analyzer de alguma forma indireta.
            
            # ATENÇÃO: As linhas abaixo podem causar AttributeError se data_collector ou signal_engine
            # não estiverem acessíveis via 'self' nesta classe.
            # Você precisará garantir que 'self.data_collector' e 'self.signal_engine'
            # sejam definidos aqui (e preenchidos) ou removê-los se não forem usados.
            # Para este código, estou adicionando-os ao construtor do PortfolioOptimizer e à classe base
            # para resolver o erro.
            if self.data_collector and self.signal_engine: # Adicionado check para evitar AttributeError
                new_actions = await self._evaluate_new_opportunities(current_opportunities)
                actions.extend(new_actions)
            else:
                logger.warning("DataCollector ou SignalEngine não acessíveis para avaliar novas oportunidades.")

        # 3. Log detalhado das decisões
        await self._log_active_management_decisions(actions)
        
        return actions
    
    async def _manage_existing_positions_actively(self) -> List[PortfolioAction]:
        """Gestão ativa das posições existentes"""
        
        try:
            positions = await self.portfolio_manager.get_open_positions_ws()
            if not positions:
                logger.info("💼 Nenhuma posição para gerenciar")
                return []
            
            logger.info(f"📊 GESTÃO ATIVA: Analisando {len(positions)} posições existentes")
            
            actions = []
            
            for position in positions:
                symbol = position.get('contract')
                if not symbol or symbol == "UNKNOWN":
                    continue
                
                self.positions_analyzed += 1
                
                # Análise completa da posição
                action = await self._analyze_position_for_action(symbol, position)
                if action:
                    actions.append(action)
                    self.decisions_made += 1
            
            logger.info(f"📈 Gestão ativa: {len(actions)} decisões tomadas para {len(positions)} posições")
            return actions
            
        except Exception as e:
            logger.error(f"❌ Erro na gestão ativa: {e}")
            return []
    
    async def _analyze_position_for_action(self, symbol: str, position: Dict) -> Optional[PortfolioAction]:
        """Análise completa de uma posição para decidir ação"""
        
        try:
            logger.info(f"🔍 Análise ativa: {symbol}")
            
            # Obter dados da posição
            pnl_data = await self.portfolio_manager.get_position_pnl(symbol)
            if not pnl_data:
                logger.warning(f"   ⚠️ {symbol}: Sem dados de PnL")
                return None
            
            pnl_pct = pnl_data.get('pnl_percentage', 0)
            pnl_usd = pnl_data.get('pnl_usd', 0)
            size = float(position.get('size', 0))
            
            # Calcular idade da posição
            position_age_hours = self._calculate_position_age_hours_accurate(position)
            
            logger.info(f"   📊 {symbol}: PnL {pnl_pct:+.2f}% ({pnl_usd:+.2f} USDT)")
            logger.info(f"   ⏰ Idade: {position_age_hours:.1f}h | Size: {size}")
            
            # DECISÃO 1: Fechamento emergencial
            if pnl_pct <= self.emergency_close_threshold:
                self.emergency_closes += 1
                return PortfolioAction(
                    action_type=ActionType.CLOSE_PARTIAL, # Ou CLOSE_FULL se preferir
                    target_position=symbol,
                    expected_improvement=abs(pnl_pct),
                    reasoning=f"🚨 FECHAMENTO EMERGENCIAL: {symbol} com {pnl_pct:.2f}% de prejuízo",
                    priority_score=100  # Máxima prioridade
                )
            
            # DECISÃO 2: Proteção de lucro
            if pnl_pct >= self.profit_protection_threshold:
                # Verificar se deve proteger o lucro
                should_protect = await self._should_protect_profit(symbol, pnl_pct, position_age_hours)
                if should_protect:
                    return PortfolioAction(
                        action_type=ActionType.CLOSE_PARTIAL,
                        target_position=symbol,
                        expected_improvement=pnl_pct * 0.7,  # Proteger 70% do lucro
                        reasoning=f"🛡️ PROTEÇÃO DE LUCRO: {symbol} com {pnl_pct:.2f}% ({position_age_hours:.1f}h)",
                        priority_score=80
                    )
            
            # DECISÃO 3: Posição muito antiga
            if position_age_hours >= self.max_position_age_hours:
                return PortfolioAction(
                    action_type=ActionType.CLOSE_PARTIAL,
                    target_position=symbol,
                    expected_improvement=5.0,  # Liberar espaço vale 5 pontos
                    reasoning=f"⏳ POSIÇÃO ANTIGA: {symbol} há {position_age_hours:.1f}h - liberar espaço",
                    priority_score=60
                )
            
            # DECISÃO 4: Análise técnica para melhoria
            # Para esta função funcionar, self.data_collector e self.signal_engine precisam ser acessíveis
            # Se não estiverem, essa parte precisa ser revisada para como a EnhancedPortfolioOptimizer
            # obtém acesso a eles.
            if self.data_collector and self.signal_engine: # Adicionado check
                technical_action = await self._analyze_technical_improvement(symbol, pnl_pct)
                if technical_action:
                    return technical_action
            else:
                logger.debug(f"DataCollector ou SignalEngine não acessíveis para análise técnica para {symbol}.")

            # DECISÃO 5: Manter posição
            logger.info(f"   ✅ {symbol}: Manter posição (sem ação necessária)")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro analisando {symbol}: {e}")
            return None
    
    async def _should_protect_profit(self, symbol: str, pnl_pct: float, age_hours: float) -> bool:
        """Determina se deve proteger lucro baseado em múltiplos fatores"""
        
        # Proteger se lucro > 2% e posição > 2 horas
        if pnl_pct > 2.0 and age_hours > 2.0:
            return True
        
        # Proteger se lucro muito alto (>3%) independente do tempo
        if pnl_pct > 3.0:
            return True
        
        # Verificar se estamos em horário desfavorável
        current_hour = datetime.now().hour
        unfavorable_hours = [22, 23, 0, 1, 2, 3, 4, 5]  # Horários de baixa liquidez
        if current_hour in unfavorable_hours and pnl_pct > 1.5:
            return True
        
        # Proteger se posição está há muito tempo em lucro
        if age_hours > 6 and pnl_pct > 1.0:
            return True
        
        return False
    
    async def _analyze_technical_improvement(self, symbol: str, current_pnl: float) -> Optional[PortfolioAction]:
        """Analisa se há sinais técnicos para melhorar a posição"""
        
        try:
            # Obter dados técnicos atuais. self.data_collector e self.signal_engine devem estar acessíveis.
            # Eles foram adicionados ao __init__ da classe base PortfolioOptimizer.
            klines = self.data_collector.get_klines_data(symbol)
            current_price = self.data_collector.get_current_price(symbol)
            
            if not klines or not current_price:
                return None
            
            # Analisar sinal atual
            # O analyze do signal_engine espera um dicionário de klines e preços
            signals = self.signal_engine.analyze({symbol: klines}, {symbol: current_price})
            current_signal = signals.get(symbol)
            
            if not current_signal or current_signal.action == "HOLD":
                return None
            
            # Se sinal é contrário à posição atual e tem alta confiança
            # Precisa saber a direção da posição atual (LONG/SHORT)
            position_info = await self.portfolio_manager.get_open_positions_ws()
            current_position_size = 0
            for pos in position_info:
                if pos.get('contract') == symbol:
                    current_position_size = float(pos.get('size', 0))
                    break

            if current_position_size == 0:
                return None # Posição já fechada ou não encontrada

            is_long_position = current_position_size > 0
            
            signal_is_contrary = False
            if (is_long_position and current_signal.action == "SELL") or \
               (not is_long_position and current_signal.action == "BUY"):
                signal_is_contrary = True

            if signal_is_contrary and current_signal.confidence > 65: # Confiança alta para sinal contrário
                # Lógica simplificada - verificar se deve fechar
                if abs(current_pnl) < 0.5:  # Posição quase neutra
                    return PortfolioAction(
                        action_type=ActionType.CLOSE_PARTIAL, # Pode ser CLOSE_AND_REPLACE se o sinal for muito forte e queremos reverter
                        target_position=symbol,
                        expected_improvement=current_signal.confidence - 50, # Pontuação baseada na confiança do sinal
                        reasoning=f"📈 SINAL TÉCNICO: {symbol} - novo sinal {current_signal.action} "
                                f"com {current_signal.confidence:.1f}% confiança. PnL neutro ({current_pnl:.2f}%)",
                        priority_score=current_signal.confidence
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Erro na análise técnica para {symbol}: {e}")
            return None
    
    def _calculate_position_age_hours_accurate(self, position: Dict) -> float:
        """Cálculo mais preciso da idade da posição"""
        
        try:
            # Tentar diferentes campos de timestamp
            timestamp_fields = ['create_time', 'open_time', 'update_time', 'ctime']
            
            for field in timestamp_fields:
                if field in position:
                    create_time = float(position[field])
                    # Se timestamp está em milissegundos, converter
                    if create_time > 1e12:
                        create_time = create_time / 1000
                    
                    age_seconds = time.time() - create_time
                    age_hours = age_seconds / 3600
                    
                    # Validar se faz sentido (não pode ser negativo ou muito grande)
                    if 0 <= age_hours <= 168:  # Máximo 1 semana
                        return age_hours
            
            # Se não encontrou timestamp válido, assumir 2 horas
            logger.warning(f"Não foi possível determinar idade da posição, assumindo 2h")
            return 2.0
            
        except Exception as e:
            logger.debug(f"Erro calculando idade da posição: {e}")
            return 2.0
    
    async def _get_available_position_slots(self) -> int:
        """Retorna o número de slots de posição disponíveis."""
        open_positions = await self.portfolio_manager.get_open_positions_ws()
        # O max_positions_overall é uma propriedade do portfolio_manager ou de config
        # Assumindo que portfolio_manager.max_positions_overall é definido ou vindo de config.
        # Se não for, precisará ser passado para o PortfolioOptimizer ou definido aqui.
        max_positions = getattr(self.portfolio_manager, 'max_positions_overall', 3) # Valor padrão de 3
        return max_positions - len(open_positions)

    async def _analyze_opportunities(self, opportunities: List[Tuple[str, Any]]) -> List[Any]:
        """
        Analisa a "qualidade" das oportunidades para pontuá-las.
        ATENÇÃO: Este método depende da estrutura do 'opportunities' e do 'signal_engine'.
        Assumindo que `opportunities` é uma lista de `(symbol, signal_object)`.
        Assumindo que `signal_object` tem um atributo `opportunity_score`.
        """
        scored_opportunities = []
        for symbol, signal_obj in opportunities:
            # Assumindo que TradingSignal (signal_obj) tem um atributo 'opportunity_score'
            # ou que você calcula um aqui.
            # Se o signal_obj for um TradingSignal, ele já terá 'opportunity_score'.
            score = getattr(signal_obj, 'opportunity_score', 0.0)
            # Você pode aprimorar o score aqui com base em mais fatores:
            # Por exemplo: score += signal_obj.confidence * 0.5
            #               score += signal_obj.indicators.get('liquidity_score', 0) * 10
            
            # Criar um objeto temporário para armazenar o score e o símbolo
            @dataclass
            class ScoredOpportunity:
                symbol: str
                total_score: float
            
            scored_opportunities.append(ScoredOpportunity(symbol=symbol, total_score=score))
        
        # Ordenar por score, do maior para o menor
        scored_opportunities.sort(key=lambda x: x.total_score, reverse=True)
        return scored_opportunities
    
    async def _evaluate_new_opportunities(self, opportunities: List) -> List[PortfolioAction]:
        """Avalia novas oportunidades apenas se há espaço disponível"""
        
        try:
            # Verificar espaço disponível
            available_slots = await self._get_available_position_slots()
            if available_slots <= 0:
                logger.info("📊 Sem slots disponíveis para novas posições")
                return []
            
            # Analisar apenas as melhores oportunidades
            opportunity_scores = await self._analyze_opportunities(opportunities)
            
            # Filtrar apenas oportunidades excepcionais
            excellent_opportunities = [
                opp for opp in opportunity_scores 
                if opp.total_score >= 75  # Score muito alto
            ]
            
            actions = []
            for opp in excellent_opportunities[:available_slots]:
                action = PortfolioAction(
                    action_type=ActionType.ADD_POSITION,
                    new_opportunity=opp.symbol,
                    expected_improvement=opp.total_score,
                    reasoning=f"➕ NOVA OPORTUNIDADE EXCEPCIONAL: {opp.symbol} "
                            f"(score: {opp.total_score:.1f})",
                    priority_score=opp.total_score
                )
                actions.append(action)
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Erro avaliando novas oportunidades: {e}")
            return []
    
    async def _log_active_management_decisions(self, actions: List[PortfolioAction]):
        """Log detalhado das decisões de gestão ativa"""
        
        logger.info("📊 === RESULTADO DA GESTÃO ATIVA ===")
        logger.info(f"   📈 Posições analisadas: {self.positions_analyzed}")
        logger.info(f"   ⚡ Decisões tomadas: {self.decisions_made}")
        logger.info(f"   🚨 Fechamentos emergenciais: {self.emergency_closes}")
        logger.info(f"   📋 Ações totais: {len(actions)}")
        
        if actions:
            logger.info("   🎯 AÇÕES PLANEJADAS:")
            for i, action in enumerate(actions, 1):
                priority_icon = {
                    100: "🚨",  # Emergencial
                    80: "🛡️",   # Proteção
                    60: "⏳",   # Tempo
                    50: "📈"    # Técnico
                }.get(int(action.priority_score // 20) * 20, "⚡")
                
                logger.info(f"      {i}. {priority_icon} {action.reasoning}")
                logger.info(f"         Tipo: {action.action_type.value}")
                logger.info(f"         Prioridade: {action.priority_score:.0f}")
                logger.info(f"         Melhoria esperada: {action.expected_improvement:.1f}")
        else:
            logger.info("   ✅ Nenhuma ação necessária - portfolio está otimizado")

# INTEGRAÇÃO SIMPLIFICADA NO MAIN.PY:
# A parte de integração abaixo está aqui apenas como referência para o main.py
# Ela NÃO DEVE ser incluída neste arquivo 'portfolio_optimizer.py'.
# A integração deve ser feita no arquivo 'main.py' conforme a estrutura de módulos.
"""
class IntelligentBotGateIndividual:
    
    def __init__(self):
        # ... código existente ...
        # SUBSTITUIR o portfolio_optimizer padrão pelo aprimorado
        self.portfolio_optimizer = EnhancedPortfolioOptimizer(
            self.portfolio_manager, 
            self.symbol_analyzer, 
            self.gate_api
        )
        # É CRÍTICO que o data_collector e signal_engine sejam passados ou acessíveis
        # para o EnhancedPortfolioOptimizer, pois ele os usa diretamente.
        # Uma forma é passá-los no construtor ou atribuí-los após a criação.
        self.portfolio_optimizer.data_collector = self.data_collector
        self.portfolio_optimizer.signal_engine = self.signal_engine
    
    async def optimize_portfolio_intelligently(self, opportunities: List = None) -> List[PortfolioAction]:
        # Usar o otimizador aprimorado
        
        try:
            logger.info("🧠 INICIANDO GESTÃO ATIVA DE PORTFOLIO")
            
            # Usar o método de otimização ativa
            actions = await self.portfolio_optimizer.optimize_portfolio(opportunities)
            
            if actions:
                logger.info(f"⚡ {len(actions)} ações de gestão ativa recomendadas")
                for action in actions:
                    logger.info(f"   📋 {action.action_type.value}: {action.reasoning}")
            else:
                logger.info("✅ Portfolio gerenciado ativamente - nenhuma ação imediata necessária")
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Erro na gestão ativa de portfolio: {e}")
            return []
    
    async def execute_portfolio_optimizations(self, actions: List[PortfolioAction]) -> int:
        # Execução com prioridade correta
        
        if not actions:
            return 0
        
        # Ordenar por prioridade (maior primeiro)
        actions_sorted = sorted(actions, key=lambda x: x.priority_score, reverse=True)
        
        executed_count = 0
        
        logger.info(f"⚡ Executando {len(actions_sorted)} ações de gestão ativa")
        
        for action in actions_sorted:
            try:
                logger.info(f"🔄 Executando: {action.reasoning}")
                
                if action.action_type in [ActionType.CLOSE_PARTIAL, ActionType.CLOSE_AND_REPLACE]:
                    # Fechar posição
                    success = await self._execute_close_action(action)
                elif action.action_type == ActionType.ADD_POSITION:
                    # Adicionar posição
                    success = await self._execute_add_position(action)
                else:
                    logger.warning(f"⚠️ Ação não implementada: {action.action_type}")
                    continue
                
                if success:
                    executed_count += 1
                    await self._notify_optimization_action(action)
                    logger.info(f"   ✅ Ação executada com sucesso")
                    
                    # Aguardar um pouco entre execuções
                    await asyncio.sleep(2)
                else:
                    logger.warning(f"   ❌ Falha na execução da ação")
                
            except Exception as e:
                logger.error(f"❌ Erro executando ação: {e}")
        
        logger.info(f"📊 Gestão ativa concluída: {executed_count}/{len(actions_sorted)} sucessos")
        return executed_count
    
    async def _execute_close_action(self, action: PortfolioAction) -> bool:
        # Executa fechamento de posição
        try:
            symbol = action.target_position
            if not symbol:
                return False
            
            # Verificar se posição ainda existe
            positions = await self.portfolio_manager.get_open_positions_ws()
            if not any(p.get('contract') == symbol for p in positions):
                logger.warning(f"⚠️ Posição {symbol} não encontrada")
                return False
            
            # Executar fechamento
            close_result = await self.portfolio_manager.close_single_position(
                symbol, 
                reason=f"active_management_{action.action_type.value}"
            )
            
            return close_result.get('success', False)
            
        except Exception as e:
            logger.error(f"❌ Erro fechando posição: {e}")
            return False
"""