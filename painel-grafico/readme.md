# 📈 Trading Dashboard Pro Frontend

Interface React moderna e responsiva para o Bot de Trading com IA v2.1

**🌐 URL de Acesso: http://62.72.1.122:5173**

## 🚀 Características

### 🎯 Interface Principal
- **Dashboard em tempo real** com métricas de performance
- **Controle do bot** (start/stop) com feedback visual
- **Market Overview** para BTC, Gold e DXY
- **Gráficos interativos** com Recharts
- **Design responsivo** mobile-first

### 📊 Análise Técnica
- **RSI e MACD** em tempo real via WebSocket
- **Indicadores visuais** com gauges customizados
- **Alertas de cruzamentos** MACD automáticos
- **Performance charts** históricos

### 🧠 Análise de Sentimento
- **Sentiment tracking** BTC e PAXG
- **Fear & Greed Index** integration
- **Dados em tempo real** via WebSocket
- **Visualizações circulares** interativas

### 🔔 Sistema de Alertas
- **Alertas categorizados** por severidade
- **Notificações em tempo real**
- **Histórico de alertas** organizado
- **Status do sistema** visual

## 🛠️ Tecnologias Utilizadas

- **React 18** - Framework principal
- **Recharts** - Visualizações e gráficos
- **WebSocket API** - Dados em tempo real
- **CSS3** - Animações e estilos modernos
- **Responsive Design** - Mobile-first

## 📁 Estrutura do Projeto

```
src/
├── App.jsx                 # Componente principal do dashboard
├── index.js               # Entry point da aplicação
├── index.css              # Estilos globais e animações
└── reportWebVitals.js     # Monitoramento de performance

public/
├── index.html             # HTML principal com loading screen
├── manifest.json          # Configuração PWA
└── ...                    # Assets estáticos
```

## 🚀 Instalação e Execução

### Pré-requisitos
- Node.js 16 ou superior
- npm ou yarn
- Backend API rodando na porta 8000

### 1. Instalar dependências
```bash
npm install
```

### 2. Configurar ambiente
Crie um arquivo `.env`:
```bash
# Para servidor remoto
REACT_APP_API_URL=http://62.72.1.122:8000
REACT_APP_WS_URL=ws://62.72.1.122:8000
PORT=5173
HOST=0.0.0.0

# Para desenvolvimento local
# REACT_APP_API_URL=http://localhost:8000
# REACT_APP_WS_URL=ws://localhost:8000
```

### 3. Executar em desenvolvimento
```bash
npm start
```

### 4. Build para produção
```bash
npm run build
npm run serve
```

## 🔧 Configuração para Servidor

### Deploy Automático
Use o script de deploy fornecido:
```bash
chmod +x deploy.sh
sudo ./deploy.sh
```

### Deploy Manual
```bash
# 1. Build da aplicação
npm run build

# 2. Instalar serve globalmente
npm install -g serve

# 3. Servir arquivos estáticos
serve -s build -l 5173 -H 0.0.0.0
```

### Configuração Systemd
```bash
# Criar serviço
sudo nano /etc/systemd/system/trading-dashboard.service

# Conteúdo do arquivo:
[Unit]
Description=Trading Dashboard Pro Frontend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/trading-dashboard-pro
Environment=NODE_ENV=production
ExecStart=/usr/bin/npx serve -s build -l 5173 -H 0.0.0.0
Restart=on-failure

[Install]
WantedBy=multi-user.target

# Habilitar e iniciar
sudo systemctl daemon-reload
sudo systemctl enable trading-dashboard
sudo systemctl start trading-dashboard
```

## 🔗 URLs de Conexão

### Frontend
- **Desenvolvimento**: http://localhost:5173
- **Produção**: http://62.72.1.122:5173

### Backend API
- **REST API**: http://62.72.1.122:8000/api/*
- **Documentação**: http://62.72.1.122:8000/docs

### WebSocket Endpoints
- **Sentiment**: ws://62.72.1.122:8000/ws/sentiment
- **RSI/MACD**: ws://62.72.1.122:8000/ws/rsi-macd
- **OHLCV**: ws://62.72.1.122:8000/ws/ohlcv

## 🐳 Docker Deploy

### Build da imagem
```bash
docker build -t trading-dashboard:latest .
```

### Executar container
```bash
docker run -d \
  --name trading-dashboard \
  -p 5173:5173 \
  -e REACT_APP_API_URL=http://62.72.1.122:8000 \
  -e REACT_APP_WS_URL=ws://62.72.1.122:8000 \
  trading-dashboard:latest
```

## 📊 Componentes Principais

### TradingBotControl
Controla o bot de trading com botões start/stop e métricas principais.

### MarketOverview
Exibe preços atuais e variações dos ativos principais (BTC, Gold, DXY).

### RSIMACDDisplay
Mostra indicadores RSI e MACD em tempo real com sinais visuais.

### SentimentAnalysis
Gauges circulares para sentiment BTC/PAXG e Fear & Greed Index.

### PerformanceChart
Gráfico de linha mostrando evolução do P&L ao longo do tempo.

### RecentAlerts
Lista de alertas recentes categorizados por severidade.

## 🎨 Design System

### Cores Principais
- **Primary**: `#6366f1` (Indigo)
- **Secondary**: `#8b5cf6` (Purple)
- **Success**: `#10b981` (Green)
- **Warning**: `#f59e0b` (Amber)
- **Danger**: `#ef4444` (Red)

### Tipografia
- **Fonte Principal**: Inter
- **Fonte Mono**: Fira Code

### Animações
- Fade in/out
- Slide in (left/right)
- Scale in
- Pulse para status indicators
- Glow effects para hover states

## 📱 Responsividade

O dashboard é totalmente responsivo com breakpoints:
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

## 🔄 WebSocket Integration

O frontend usa WebSockets para dados em tempo real com reconexão automática:

```javascript
// URLs configuradas automaticamente baseadas no ambiente
const wsBaseURL = getWSBaseURL(); // ws://62.72.1.122:8000 ou ws://localhost:8000

// Hook customizado com reconexão automática
const useWebSocket = (url, onMessage) => {
  // Reconexão automática em caso de falha
  // Tratamento de erros
  // Status de conexão
};
```

## 🚨 Troubleshooting

### Frontend não carrega
```bash
# Verificar se o serviço está rodando
sudo systemctl status trading-dashboard

# Ver logs
sudo journalctl -u trading-dashboard -f

# Restart do serviço
sudo systemctl restart trading-dashboard
```

### WebSocket não conecta
```bash
# Verificar se# 📈 Trading Dashboard Pro Frontend

Interface React moderna e responsiva para o Bot de Trading com IA v2.1

## 🚀 Características

### 🎯 Interface Principal
- **Dashboard em tempo real** com métricas de performance
- **Controle do bot** (start/stop) com feedback visual
- **Market Overview** para BTC, Gold e DXY
- **Gráficos interativos** com Recharts
- **Design responsivo** mobile-first

### 📊 Análise Técnica
- **RSI e MACD** em tempo real via WebSocket
- **Indicadores visuais** com gauges customizados
- **Alertas de cruzamentos** MACD automáticos
- **Performance charts** históricos

### 🧠 Análise de Sentimento
- **Sentiment tracking** BTC e PAXG
- **Fear & Greed Index** integration
- **Dados em tempo real** via WebSocket
- **Visualizações circulares** interativas

### 🔔 Sistema de Alertas
- **Alertas categorizados** por severidade
- **Notificações em tempo real**
- **Histórico de alertas** organizado
- **Status do sistema** visual

## 🛠️ Tecnologias Utilizadas

- **React 18** - Framework principal
- **Recharts** - Visualizações e gráficos
- **WebSocket API** - Dados em tempo real
- **CSS3** - Animações e estilos modernos
- **Responsive Design** - Mobile-first

## 📁 Estrutura do Projeto

```
src/
├── App.jsx                 # Componente principal do dashboard
├── index.js               # Entry point da aplicação
├── index.css              # Estilos globais e animações
└── reportWebVitals.js     # Monitoramento de performance

public/
├── index.html             # HTML principal com loading screen
├── manifest.json          # Configuração PWA
└── ...                    # Assets estáticos
```

## 🚀 Instalação e Execução

### Pré-requisitos
- Node.js 16 ou superior
- npm ou yarn
- Backend API rodando na porta 8000

### 1. Instalar dependências
```bash
npm install
```

### 2. Executar em desenvolvimento
```bash
npm start
```

A aplicação estará disponível em: http://localhost:3000

### 3. Build para produção
```bash
npm run build
```

## 🔧 Configuração

### Backend API
O frontend está configurado para conectar com o backend em `localhost:8000`. 

Para alterar a URL da API, modifique o arquivo `package.json`:
```json
{
  "proxy": "http://seu-backend:8000"
}
```

### WebSocket Endpoints
- Sentiment: `ws://localhost:8000/ws/sentiment`
- RSI/MACD: `ws://localhost:8000/ws/rsi-macd`
- OHLCV: `ws://localhost:8000/ws/ohlcv`

## 📊 Componentes Principais

### TradingBotControl
Controla o bot de trading com botões start/stop e métricas principais.

### MarketOverview
Exibe preços atuais e variações dos ativos principais (BTC, Gold, DXY).

### RSIMACDDisplay
Mostra indicadores RSI e MACD em tempo real com sinais visuais.

### SentimentAnalysis
Gauges circulares para sentiment BTC/PAXG e Fear & Greed Index.

### PerformanceChart
Gráfico de linha mostrando evolução do P&L ao longo do tempo.

### RecentAlerts
Lista de alertas recentes categorizados por severidade.

## 🎨 Design System

### Cores Principais
- **Primary**: `#6366f1` (Indigo)
- **Secondary**: `#8b5cf6` (Purple)
- **Success**: `#10b981` (Green)
- **Warning**: `#f59e0b` (Amber)
- **Danger**: `#ef4444` (Red)

### Tipografia
- **Fonte Principal**: Inter
- **Fonte Mono**: Fira Code

### Animações
- Fade in/out
- Slide in (left/right)
- Scale in
- Pulse para status indicators
- Glow effects para hover states

## 📱 Responsividade

O dashboard é totalmente responsivo com breakpoints:
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

## 🔄 WebSocket Integration

O frontend usa WebSockets para dados em tempo real:

```javascript
// Hook customizado para WebSocket
const useWebSocket = (url, onMessage) => {
  // Reconexão automática
  // Tratamento de erros
  // Status de conexão
};
```

## 🚨 Tratamento de Erros

- **Error Boundary** para capturar erros React
- **Fallbacks** para dados indisponíveis
- **Loading states** para operações assíncronas
- **Reconexão automática** para WebSockets

## 🔐 Segurança

- **CORS** configurado no backend
- **Sanitização** de dados WebSocket
- **Error boundaries** para isolamento de falhas
- **Validação** de dados da API

## 🎯 Performance

- **Lazy loading** de componentes
- **Memoização** com useCallback/useMemo
- **Debouncing** de chamadas API
- **Otimização** de re-renders

## 📈 PWA Features

- **Manifest** configurado
- **Service Worker** (opcional)
- **Offline fallbacks**
- **Install prompts**

## 🧪 Testing

```bash
# Executar testes
npm test

# Cobertura de testes
npm test -- --coverage
```

## 📦 Build e Deploy

```bash
# Build otimizado
npm run build

# Análise do bundle
npm install -g serve
serve -s build
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/nova-feature`
3. Commit: `git commit -m 'Add nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Abra um Pull Request

## 📞 Suporte

- **Issues**: GitHub Issues
- **Documentação**: README.md
- **API Docs**: http://localhost:8000/docs

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

**Trading Dashboard Pro v6.0** - Built with ❤️ and React