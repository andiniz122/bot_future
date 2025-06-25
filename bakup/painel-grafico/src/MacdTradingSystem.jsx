import React, { useMemo } from 'react';
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, Activity, Target, Play, Pause, Volume2, Clock } from 'lucide-react';

// Componente Card (duplicado aqui para ser independente, mas você pode centralizar se preferir)
const Card = ({ children, className = '', title = '' }) => (
  <div className={`bg-slate-800 rounded-xl border border-slate-700 ${className}`}>
    {title && (
      <div className="px-6 py-4 border-b border-slate-700">
        <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
      </div>
    )}
    <div className={title ? 'p-6' : 'p-6'}>
      {children}
    </div>
  </div>
);

const MacdTradingSystem = ({ data, timeframe = '5m' }) => {
  // 🧠 IMPLEMENTAÇÃO DA SUA LÓGICA DE TRADING
  const tradingAnalysis = useMemo(() => {
    if (!data || data.length < 3) return null;

    const validData = data.filter(d =>
      d.btc_macd !== undefined && d.btc_macd !== null &&
      d.btc_macd_signal !== undefined && d.btc_macd_signal !== null &&
      d.btc_macd_hist !== undefined && d.btc_macd_hist !== null &&
      !isNaN(d.btc_macd) && !isNaN(d.btc_macd_signal) && !isNaN(d.btc_macd_hist)
    );

    if (validData.length < 3) return null;

    // Garante que estamos pegando os últimos pontos válidos para análise
    const current = validData[validData.length - 1];
    const previous = validData[validData.length - 2];
    const beforePrevious = validData[validData.length - 3];

    // Variáveis para a lógica
    const macd = current.btc_macd;
    const macd_signal = current.btc_macd_signal;
    const macd_hist = current.btc_macd_hist;
    const prev_macd_hist = previous.btc_macd_hist;
    const prev_macd = previous.btc_macd;
    const prev_signal = previous.btc_macd_signal;

    // Detecção de cruzamentos
    const bullish_cross = prev_macd <= prev_signal && macd > macd_signal;
    const bearish_cross = prev_macd >= prev_signal && macd < macd_signal;
    
    // Tendências do histograma
    const hist_rising = macd_hist > prev_macd_hist;
    const hist_falling = macd_hist < prev_macd_hist;
    
    // Proximidade (consolidação)
    // Ajuste este valor conforme a sensibilidade desejada para "consolidação"
    // Um valor menor indica que MACD e Signal estão muito próximos.
    const close_threshold = Math.abs(macd - macd_signal) < 0.5; // Ajuste conforme a escala dos seus dados MACD
    
    // 🎯 LÓGICA DE RECOMENDAÇÃO (baseada na sua tabela)
    let recomendacao = "ESPERAR";
    let confidence = 0;
    let reasoning = "";
    let color = "yellow";
    let icon = Pause;

    // 1. COMPRAR: MACD cruza acima da signal E histograma passa para positivo (ou já é positivo e está crescendo)
    if (bullish_cross && macd_hist >= 0) { // MACD Hist >= 0 para incluir o momento do cruzamento para positivo
      recomendacao = "COMPRAR";
      confidence = 85;
      reasoning = "Cruzamento bullish confirmado com histograma positivo ou em ascensão.";
      color = "green";
      icon = TrendingUp;
    }
    // 2. VENDER: MACD cruza abaixo da signal E histograma passa para negativo (ou já é negativo e está caindo)
    else if (bearish_cross && macd_hist <= 0) { // MACD Hist <= 0 para incluir o momento do cruzamento para negativo
      recomendacao = "VENDER";
      confidence = 85;
      reasoning = "Cruzamento bearish confirmado com histograma negativo ou em queda.";
      color = "red";
      icon = TrendingDown;
    }
    // 3. COMPRAR: MACD e Signal ambos subindo, histograma positivo e ampliando (confirmação de tendência de alta)
    else if (macd > prev_macd && macd_signal > prev_signal && macd_hist > 0 && hist_rising) {
      recomendacao = "COMPRAR";
      confidence = 75;
      reasoning = "Tendência de alta forte com momentum crescente.";
      color = "green";
      icon = TrendingUp;
    }
    // 4. VENDER: MACD e Signal ambos caindo, histograma negativo e ampliando (confirmação de tendência de baixa)
    else if (macd < prev_macd && macd_signal < prev_signal && macd_hist < 0 && hist_falling) {
      recomendacao = "VENDER";
      confidence = 75;
      reasoning = "Tendência de baixa forte com momentum decrescente.";
      color = "red";
      icon = TrendingDown;
    }
    // 5. ESPERAR: MACD > Signal, mas histograma está caindo (momentum de alta enfraquecendo)
    else if (macd > macd_signal && hist_falling && !bullish_cross) { // Exclui o ponto exato do cruzamento bullish
      recomendacao = "ESPERAR";
      confidence = 40;
      reasoning = "Tendência de alta perdendo força - aguardar confirmação.";
      color = "yellow";
      icon = AlertTriangle;
    }
    // 6. ESPERAR: MACD < Signal, mas histograma está subindo (momentum de baixa enfraquecendo)
    else if (macd < macd_signal && hist_rising && !bearish_cross) { // Exclui o ponto exato do cruzamento bearish
      recomendacao = "ESPERAR";
      confidence = 40;
      reasoning = "Tendência de baixa perdendo força - possível reversão futura.";
      color = "yellow";
      icon = AlertTriangle;
    }
    // 7. ESPERAR: Consolidação lateral (MACD e Signal muito próximos ou histograma próximo de zero e sem tendência clara)
    else if (close_threshold || (Math.abs(macd_hist) < 0.1 && !hist_rising && !hist_falling)) { // Adiciona condição para hist próximo de zero
      recomendacao = "ESPERAR";
      confidence = 20;
      reasoning = "Consolidação lateral ou falta de momentum - aguardar direção.";
      color = "gray";
      icon = Pause;
    }
    // Default case (if none of the above conditions are met)
    else {
      recomendacao = "ESPERAR";
      confidence = 30;
      reasoning = "Condições de mercado indefinidas ou sem sinal claro.";
      color = "yellow";
      icon = Pause;
    }


    // Análise de volume (se disponível nos dados de entrada)
    let volume_confirmation = null;
    if (current.btc_volume && previous.btc_volume) {
      // Calcula a média de volume dos últimos 10 períodos
      const volume_avg = validData.slice(-10).reduce((sum, d) => sum + (d.btc_volume || 0), 0) / Math.min(10, validData.length);
      const volume_above_avg = current.btc_volume > volume_avg * 1.2; // Volume 20% acima da média

      if (volume_above_avg) {
        if (recomendacao === "COMPRAR") {
          confidence += 10;
          volume_confirmation = "Volume acima da média - reforça o sinal de compra.";
        } else if (recomendacao === "VENDER") {
          confidence += 10;
          volume_confirmation = "Volume acima da média - reforça o sinal de venda.";
        }
      } else {
        if (recomendacao === "COMPRAR" || recomendacao === "VENDER") {
          confidence -= 5; // Leve penalidade se o volume não confirmar
          volume_confirmation = "Volume baixo - pode enfraquecer a força do sinal.";
        }
      }
    }

    // Análise de timeframe
    let timeframe_reliability = "";
    switch(timeframe) {
      case '1m':
      case '5m':
        timeframe_reliability = "⚠️ Timeframe curto - mais ruído (sinal menos robusto)";
        confidence *= 0.8;
        break;
      case '15m':
      case '30m':
        timeframe_reliability = "📊 Timeframe médio - balanceado (sinal com boa relevância)";
        break;
      case '1h':
      case '4h':
        timeframe_reliability = "🎯 Timeframe confiável (sinal mais robusto)";
        confidence *= 1.1;
        break;
      case '1d':
        timeframe_reliability = "💎 Timeframe estratégico (sinal de alta relevância)";
        confidence *= 1.2;
        break;
      default:
        timeframe_reliability = "🕰️ Timeframe não especificado ou padrão (relevância média)";
        break;
    }

    confidence = Math.min(95, Math.max(5, confidence)); // Limitar entre 5-95%

    return {
      current,
      previous,
      recomendacao,
      confidence: Math.round(confidence),
      reasoning,
      color,
      icon,
      volume_confirmation,
      timeframe_reliability,
      signals: {
        bullish_cross,
        bearish_cross,
        hist_rising,
        hist_falling,
        close_threshold,
        macd_above_signal: macd > macd_signal,
        histogram_positive: macd_hist > 0
      },
      values: {
        macd: macd.toFixed(4),
        signal: macd_signal.toFixed(4),
        histogram: macd_hist.toFixed(4),
        hist_change: (macd_hist - prev_macd_hist).toFixed(4)
      }
    };
  }, [data, timeframe]); // Removido volume_data do array de dependências, pois agora é extraído de 'data'

  // Histórico de sinais (últimos 5)
  // Refatorado para usar 'data' diretamente para o histórico, não 'tradingAnalysis' que é para o ponto atual
  const signalHistory = useMemo(() => {
    if (!data || data.length < 10) return [];
    
    const history = [];
    // Itera sobre os últimos 10 pontos (ou menos se não houver 10) para buscar sinais passados
    for (let i = Math.max(0, data.length - 10); i < data.length - 1; i++) {
      const currentPoint = data[i];
      const previousPoint = data[i - 1];
      if (!currentPoint || !previousPoint) continue; // Garante que há dados suficientes para comparação

      // Recalcula as condições para cada ponto no histórico
      const macd = currentPoint.btc_macd;
      const macd_signal = currentPoint.btc_macd_signal;
      const macd_hist = currentPoint.btc_macd_hist;
      const prev_macd = previousPoint.btc_macd;
      const prev_signal = previousPoint.btc_macd_signal;
      const prev_macd_hist = previousPoint.btc_macd_hist;

      const bullish_cross = prev_macd <= prev_signal && macd > macd_signal;
      const bearish_cross = prev_macd >= prev_signal && macd < macd_signal;
      const hist_rising = macd_hist > prev_macd_hist;
      const hist_falling = macd_hist < prev_macd_hist;

      let action = null;
      let conf = 0;

      if (bullish_cross && macd_hist >= 0) {
        action = "COMPRAR";
        conf = 85;
      } else if (bearish_cross && macd_hist <= 0) {
        action = "VENDER";
        conf = 85;
      } else if (macd > prev_macd && macd_signal > prev_signal && macd_hist > 0 && hist_rising) {
        action = "COMPRAR";
        conf = 75;
      } else if (macd < prev_macd && macd_signal < prev_signal && macd_hist < 0 && hist_falling) {
        action = "VENDER";
        conf = 75;
      }

      if (action) {
        history.push({
          timestamp: currentPoint.timestamp,
          action: action,
          confidence: Math.round(conf), // Ajustar a confiança aqui também se necessário
          price: currentPoint.btc || 0
        });
      }
    }
    return history.slice(-5).reverse(); // Retorna os 5 últimos sinais, do mais recente para o mais antigo
  }, [data]);


  if (!tradingAnalysis) {
    return (
      <Card title="🤖 Sistema de Trading MACD" className="col-span-full">
        <div className="flex items-center justify-center h-40">
          <div className="text-center">
            <Activity className="w-8 h-8 text-slate-400 animate-pulse mx-auto mb-2" />
            <p className="text-slate-400">Aguardando dados para análise...</p>
          </div>
        </div>
      </Card>
    );
  }

  const IconComponent = tradingAnalysis.icon;

  return (
    <div className="space-y-6">
      {/* 🎯 RECOMENDAÇÃO PRINCIPAL */}
      <Card title="🤖 Sistema de Trading MACD - Recomendação Automática" className="col-span-full">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Recomendação Principal */}
          <div className="lg:col-span-2">
            <div className={`p-6 rounded-xl bg-gradient-to-r ${
              tradingAnalysis.color === 'green' ? 'from-green-500/20 to-green-600/10 border-green-500/30' :
              tradingAnalysis.color === 'red' ? 'from-red-500/20 to-red-600/10 border-red-500/30' :
              'from-yellow-500/20 to-yellow-600/10 border-yellow-500/30'
            } border-2`}>
              
              <div className="flex items-center space-x-4 mb-4">
                <div className={`p-3 rounded-full ${
                  tradingAnalysis.color === 'green' ? 'bg-green-500/20' :
                  tradingAnalysis.color === 'red' ? 'bg-red-500/20' :
                  'bg-yellow-500/20'
                }`}>
                  <IconComponent className={`w-8 h-8 ${
                    tradingAnalysis.color === 'green' ? 'text-green-400' :
                    tradingAnalysis.color === 'red' ? 'text-red-400' :
                    'text-yellow-400'
                  }`} />
                </div>
                <div>
                  <h3 className={`text-2xl font-bold ${
                    tradingAnalysis.color === 'green' ? 'text-green-400' :
                    tradingAnalysis.color === 'red' ? 'text-red-400' :
                    'text-yellow-400'
                  }`}>
                    {tradingAnalysis.recomendacao}
                  </h3>
                  <p className="text-slate-300 text-sm">{tradingAnalysis.reasoning}</p>
                </div>
              </div>

              {/* Barra de Confiança */}
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-300">Confiança do Sinal</span>
                  <span className="text-slate-100 font-bold">{tradingAnalysis.confidence}%</span>
                </div>
                <div className="bg-slate-600 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full transition-all duration-300 ${
                      tradingAnalysis.color === 'green' ? 'bg-green-500' :
                      tradingAnalysis.color === 'red' ? 'bg-red-500' :
                      'bg-yellow-500'
                    }`}
                    style={{width: `${tradingAnalysis.confidence}%`}}
                  ></div>
                </div>
              </div>

              {/* Alertas Adicionais */}
              <div className="space-y-2 text-sm">
                {tradingAnalysis.volume_confirmation && (
                  <div className="flex items-center space-x-2">
                    <Volume2 className="w-4 h-4 text-blue-400" />
                    <span className="text-slate-300">{tradingAnalysis.volume_confirmation}</span>
                  </div>
                )}
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-purple-400" />
                  <span className="text-slate-300">{tradingAnalysis.timeframe_reliability}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Valores Técnicos */}
          <div className="space-y-4">
            <div className="p-4 bg-slate-700/30 rounded-lg">
              <h4 className="text-sm font-medium text-slate-300 mb-3">📊 Valores MACD</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">MACD:</span>
                  <span className="text-slate-100 font-mono">{tradingAnalysis.values.macd}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Signal:</span>
                  <span className="text-slate-100 font-mono">{tradingAnalysis.values.signal}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Histogram:</span>
                  <span className={`font-mono ${Number(tradingAnalysis.values.histogram) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {tradingAnalysis.values.histogram}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Variação:</span>
                  <span className={`font-mono ${Number(tradingAnalysis.values.hist_change) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {tradingAnalysis.values.hist_change}
                  </span>
                </div>
              </div>
            </div>

            {/* Status dos Sinais */}
            <div className="p-4 bg-slate-700/30 rounded-lg">
              <h4 className="text-sm font-medium text-slate-300 mb-3">🔍 Status dos Sinais</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">MACD > Signal:</span>
                  <span className={tradingAnalysis.signals.macd_above_signal ? 'text-green-400' : 'text-red-400'}>
                    {tradingAnalysis.signals.macd_above_signal ? '✅' : '❌'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Histogram +:</span>
                  <span className={tradingAnalysis.signals.histogram_positive ? 'text-green-400' : 'text-red-400'}>
                    {tradingAnalysis.signals.histogram_positive ? '✅' : '❌'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Hist. Subindo:</span>
                  <span className={tradingAnalysis.signals.hist_rising ? 'text-green-400' : 'text-red-400'}>
                    {tradingAnalysis.signals.hist_rising ? '📈' : '📉'}
                  </span>
                </div>
                {tradingAnalysis.signals.bullish_cross && (
                  <div className="text-green-400 text-center">🟢 BULLISH CROSS</div>
                )}
                {tradingAnalysis.signals.bearish_cross && (
                  <div className="text-red-400 text-center">🔴 BEARISH CROSS</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* 📈 HISTÓRICO DE SINAIS */}
      {signalHistory.length > 0 && (
        <Card title="📈 Histórico de Sinais Recentes" className="col-span-full">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {signalHistory.map((signal, index) => (
              <div key={index} className={`p-3 rounded-lg border-l-4 ${
                signal.action === 'COMPRAR' ? 'bg-green-500/10 border-green-500' : 'bg-red-500/10 border-red-500'
              }`}>
                <div className={`text-sm font-bold ${
                  signal.action === 'COMPRAR' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {signal.action}
                </div>
                <div className="text-xs text-slate-400">{signal.timestamp}</div>
                <div className="text-xs text-slate-300">Conf: {signal.confidence}%</div>
                <div className="text-xs text-slate-300">${signal.price.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default MacdTradingSystem;