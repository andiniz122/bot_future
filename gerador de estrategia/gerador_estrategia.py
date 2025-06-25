import pandas as pd
import numpy as np
import asyncio
import logging
import time
from datetime import datetime, timedelta
from gate_api import ApiClient, Configuration, FuturesApi, FuturesOrder
import talib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple, Optional, Callable

# Configuração inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('strategy_generator')

class GateIOBacktester:
    """Gerador de estratégias com backtesting para Gate.io Futures"""
    
    def __init__(self, api_key: str, api_secret: str):
        # Configuração da API Gate.io
        config = Configuration(key=api_key, secret=api_secret)
        self.futures_api = FuturesApi(ApiClient(config))
        
        # Parâmetros do backtest
        self.initial_balance = 10000.0  # USDT
        self.leverage = 10
        self.trade_fee = 0.0005  # 0.05% por trade
        self.slippage = 0.001  # 0.1% de slippage
        
        # Dados históricos
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.symbol = "BTC_USDT"
        self.timeframe = "5m"
        
        # Resultados
        self.backtest_results = []
        self.best_strategies = []
        
        logger.info("✅ Gerador de estratégias inicializado")

    async def fetch_historical_data(self, symbol: str, timeframe: str, days: int = 30):
        """Busca dados históricos da Gate.io"""
        logger.info(f"📊 Buscando dados históricos para {symbol} ({timeframe})")
        self.symbol = symbol
        self.timeframe = timeframe
        
        try:
            # Calcular timestamps
            end = datetime.utcnow()
            start = end - timedelta(days=days)
            
            # Buscar dados
            candles = await self.futures_api.list_futures_candles(
                settle='usdt',
                contract=symbol,
                interval=timeframe,
                from_t=int(start.timestamp()),
                to_t=int(end.timestamp())
            )
            
            # Processar dados
            df = pd.DataFrame(candles)
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Converter tipos
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Adicionar indicadores básicos
            self.add_technical_indicators(df)
            
            self.historical_data[symbol] = df
            logger.info(f"✅ Dados históricos carregados: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Erro buscando dados históricos: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame):
        """Adiciona indicadores técnicos ao DataFrame"""
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Médias Móveis
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        
        # ATR (Average True Range)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        
        # Padrões de candle
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        return df

    def generate_basic_strategies(self) -> List[Dict]:
        """Gera um conjunto básico de estratégias para testar"""
        strategies = []
        
        # Estratégia 1: Duas velas verdes
        strategies.append({
            'name': 'Duas Velas Verdes',
            'entry_condition': lambda df, i: (
                df['close'].iloc[i-1] > df['open'].iloc[i-1] and 
                df['close'].iloc[i-2] > df['open'].iloc[i-2]
            ),
            'exit_condition': lambda df, i: (
                df['close'].iloc[i] < df['open'].iloc[i] or
                (df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i] > 0.005
            ),
            'stop_loss': lambda df, i: df['close'].iloc[i] * 0.99,
            'take_profit': lambda df, i: df['close'].iloc[i] * 1.015
        })
        
        # Estratégia 2: RSI Oversold
        strategies.append({
            'name': 'RSI Oversold',
            'entry_condition': lambda df, i: df['rsi'].iloc[i] < 30,
            'exit_condition': lambda df, i: df['rsi'].iloc[i] > 55,
            'stop_loss': lambda df, i: df['close'].iloc[i] * 0.98,
            'take_profit': lambda df, i: df['close'].iloc[i] * 1.02
        })
        
        # Estratégia 3: MACD Crossover
        strategies.append({
            'name': 'MACD Crossover',
            'entry_condition': lambda df, i: (
                df['macd'].iloc[i] > df['macd_signal'].iloc[i] and 
                df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]
            ),
            'exit_condition': lambda df, i: (
                df['macd'].iloc[i] < df['macd_signal'].iloc[i] and 
                df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]
            ),
            'stop_loss': lambda df, i: df['close'].iloc[i] * 0.99,
            'take_profit': lambda df, i: df['close'].iloc[i] * 1.02
        })
        
        # Estratégia 4: Bollinger Band Reversal
        strategies.append({
            'name': 'Bollinger Reversal',
            'entry_condition': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and 
                df['close'].iloc[i] > df['close'].iloc[i-1]
            ),
            'exit_condition': lambda df, i: df['close'].iloc[i] > df['bb_middle'].iloc[i],
            'stop_loss': lambda df, i: df['bb_lower'].iloc[i] * 0.99,
            'take_profit': lambda df, i: df['bb_middle'].iloc[i] * 1.01
        })
        
        logger.info(f"🧠 Geradas {len(strategies)} estratégias básicas")
        return strategies

    def generate_ml_strategy(self, df: pd.DataFrame):
        """Gera uma estratégia usando machine learning"""
        try:
            # Preparar dados
            df = df.copy().dropna()
            
            # Criar target (1 se próximo candle for positivo, 0 caso contrário)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Features
            features = [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                'sma_50', 'ema_20', 'atr', 'volume', 'volume_sma'
            ]
            
            X = df[features]
            y = df['target']
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Treinar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Avaliar
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"🤖 Modelo ML treinado - Acurácia: {accuracy:.2%}")
            logger.info(classification_report(y_test, y_pred))
            
            # Criar estratégia baseada no modelo
            return {
                'name': 'ML Strategy',
                'entry_condition': lambda df, i: (
                    model.predict([df.loc[df.index[i], features]])[0] == 1
                    if i < len(df) - 1 else False
                ),
                'exit_condition': lambda df, i: (
                    model.predict([df.loc[df.index[i], features]])[0] == 0
                    if i < len(df) - 1 else True
                ),
                'stop_loss': lambda df, i: df['close'].iloc[i] * 0.99,
                'take_profit': lambda df, i: df['close'].iloc[i] * 1.02
            }
            
        except Exception as e:
            logger.error(f"Erro gerando estratégia ML: {e}")
            return None

    def run_backtest(self, strategy: Dict, df: pd.DataFrame) -> Dict:
        """Executa backtest para uma estratégia"""
        if df.empty:
            return {}
            
        logger.info(f"🔍 Executando backtest para: {strategy['name']}")
        
        # Variáveis de estado
        balance = self.initial_balance
        position = None
        trades = []
        equity_curve = []
        max_drawdown = 0
        peak_equity = balance
        
        # Loop pelos dados históricos
        for i in range(2, len(df)):
            current_price = df['close'].iloc[i]
            
            # Fechar posição se necessário
            if position:
                # Verificar stop loss/take profit
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                if current_price <= stop_loss or current_price >= take_profit:
                    # Calcular resultado do trade
                    profit = (current_price - position['entry_price']) * position['size'] * position['direction']
                    profit -= profit * self.trade_fee * 2  # Taxa entrada + saída
                    
                    # Atualizar saldo
                    balance += profit
                    position = None
                    
                    # Registrar trade
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': df.index[i],
                        'symbol': self.symbol,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit
                    })
            
            # Condição de entrada
            if not position and strategy['entry_condition'](df, i):
                # Calcular tamanho da posição (5% do balanço)
                position_size = (balance * 0.05) * self.leverage / current_price
                
                # Direção (sempre long para estas estratégias)
                direction = 1
                
                # Preço de entrada com slippage
                entry_price = current_price * (1 + self.slippage) if direction > 0 else current_price * (1 - self.slippage)
                
                # Criar posição
                position = {
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'size': position_size,
                    'direction': direction,
                    'stop_loss': strategy['stop_loss'](df, i),
                    'take_profit': strategy['take_profit'](df, i)
                }
            
            # Registrar curva de equity
            current_equity = balance
            if position:
                unrealized = (current_price - position['entry_price']) * position['size'] * position['direction']
                current_equity += unrealized
            
            equity_curve.append(current_equity)
            
            # Calcular drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Métricas de desempenho
        total_return = (balance - self.initial_balance) / self.initial_balance
        num_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        avg_profit = sum(t['profit'] for t in trades) / num_trades if num_trades > 0 else 0
        
        result = {
            'strategy': strategy['name'],
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'trades': trades
        }
        
        self.backtest_results.append(result)
        logger.info(f"📈 Resultado: {strategy['name']} - Retorno: {total_return:.2%} | Trades: {num_trades} | Win Rate: {win_rate:.2%}")
        
        return result

    def optimize_strategy(self, base_strategy: Dict, df: pd.DataFrame, param_grid: Dict) -> Dict:
        """Otimiza parâmetros de uma estratégia usando grid search"""
        logger.info(f"⚙️ Otimizando estratégia: {base_strategy['name']}")
        best_result = None
        best_params = None
        
        # Gerar combinações de parâmetros
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(product(*param_grid.values()))
        
        for values in param_values:
            params = dict(zip(param_names, values))
            
            # Criar cópia da estratégia com novos parâmetros
            strategy = base_strategy.copy()
            strategy['name'] = f"{base_strategy['name']} {params}"
            
            # Atualizar condições com novos parâmetros
            if 'rsi_period' in params:
                df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
                strategy['entry_condition'] = lambda df, i: df['rsi'].iloc[i] < 30
                strategy['exit_condition'] = lambda df, i: df['rsi'].iloc[i] > 50
            
            # Executar backtest
            result = self.run_backtest(strategy, df)
            
            # Verificar se é o melhor resultado
            if not best_result or result['total_return'] > best_result['total_return']:
                best_result = result
                best_params = params
        
        logger.info(f"🏆 Melhor parâmetro: {best_params} - Retorno: {best_result['total_return']:.2%}")
        return best_result

    def find_best_strategies(self, top_n: int = 3) -> List[Dict]:
        """Encontra as melhores estratégias baseadas nos backtests"""
        if not self.backtest_results:
            return []
            
        # Ordenar por retorno total
        sorted_results = sorted(
            self.backtest_results, 
            key=lambda x: x['total_return'], 
            reverse=True
        )
        
        self.best_strategies = sorted_results[:top_n]
        
        logger.info("🏆 Melhores estratégias:")
        for i, strategy in enumerate(self.best_strategies):
            logger.info(f"{i+1}. {strategy['strategy']} - Retorno: {strategy['total_return']:.2%}")
        
        return self.best_strategies

    def plot_results(self, strategy_name: str):
        """Plota resultados de uma estratégia específica"""
        result = next((r for r in self.backtest_results if r['strategy'] == strategy_name), None)
        if not result:
            logger.error(f"Estratégia não encontrada: {strategy_name}")
            return
            
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(result['equity_curve'])
        plt.title(f"Curva de Equity: {strategy_name}")
        plt.xlabel("Períodos")
        plt.ylabel("Equity (USDT)")
        plt.grid(True)
        plt.show()
        
        # Plot trades
        if result['trades']:
            trade_returns = [t['profit'] / self.initial_balance for t in result['trades']]
            plt.figure(figsize=(12, 4))
            plt.bar(range(len(trade_returns)), trade_returns)
            plt.title("Retorno por Trade")
            plt.xlabel("Trade")
            plt.ylabel("Retorno (%)")
            plt.grid(True)
            plt.show()

    async def execute_live_trade(self, strategy: Dict, df: pd.DataFrame):
        """Executa uma trade ao vivo baseada na estratégia"""
        if not df.empty:
            i = len(df) - 1  # Último candle
            
            # Verificar condição de entrada
            if strategy['entry_condition'](df, i):
                logger.info(f"🚀 Sinal de entrada detectado para {strategy['name']}")
                
                try:
                    # Criar ordem
                    order = FuturesOrder(
                        contract=self.symbol,
                        size=int((self.initial_balance * 0.05) * self.leverage / df['close'].iloc[i]),
                        price=str(df['close'].iloc[i]),
                        tif='ioc'  # Immediate or Cancel
                    )
                    
                    # Enviar ordem
                    response = await self.futures_api.create_futures_order(
                        settle='usdt',
                        futures_order=order
                    )
                    
                    logger.info(f"✅ Ordem executada: {response}")
                    
                    # Configurar stop loss e take profit
                    stop_price = strategy['stop_loss'](df, i)
                    take_profit_price = strategy['take_profit'](df, i)
                    
                    # Aqui você implementaria a lógica para colocar ordens OCO
                    # (One Cancels the Other) na exchange
                    
                except Exception as e:
                    logger.error(f"Erro executando ordem: {e}")
            else:
                logger.info("⏳ Nenhum sinal de entrada detectado")

# Exemplo de uso
async def main():
    # Configurar com suas chaves API da Gate.io
    api_key = "SUA_API_KEY"
    api_secret = "SUA_API_SECRET"
    
    # Inicializar backtester
    backtester = GateIOBacktester(api_key, api_secret)
    
    # Buscar dados históricos
    df = await backtester.fetch_historical_data("BTC_USDT", "5m", days=90)
    
    if not df.empty:
        # Gerar estratégias básicas
        strategies = backtester.generate_basic_strategies()
        
        # Gerar estratégia de ML
        ml_strategy = backtester.generate_ml_strategy(df)
        if ml_strategy:
            strategies.append(ml_strategy)
        
        # Executar backtests
        for strategy in strategies:
            backtester.run_backtest(strategy, df)
        
        # Encontrar melhores estratégias
        best_strategies = backtester.find_best_strategies(top_n=3)
        
        # Plotar resultados da melhor estratégia
        if best_strategies:
            backtester.plot_results(best_strategies[0]['strategy'])
            
            # Executar trade ao vivo com a melhor estratégia
            await backtester.execute_live_trade(best_strategies[0], df)

if __name__ == "__main__":
    asyncio.run(main())