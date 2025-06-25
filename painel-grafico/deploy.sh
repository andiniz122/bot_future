#!/bin/bash

# Script de Deploy para Trading Dashboard Pro
# Para usar no servidor 62.72.1.122:5173

set -e

echo "🚀 Iniciando deploy do Trading Dashboard Pro..."

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log colorido
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Verificar se Node.js está instalado
if ! command -v node &> /dev/null; then
    error "Node.js não está instalado. Instale a versão 16 ou superior."
fi

# Verificar versão do Node.js
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    error "Node.js versão 16 ou superior é necessária. Versão atual: $(node -v)"
fi

log "Node.js versão $(node -v) detectado ✅"

# Verificar se npm está instalado
if ! command -v npm &> /dev/null; then
    error "npm não está instalado."
fi

log "npm versão $(npm -v) detectado ✅"

# Definir diretórios
PROJECT_DIR="/opt/trading-dashboard-pro"
BACKUP_DIR="/opt/trading-dashboard-backup"
SERVICE_NAME="trading-dashboard"

# Função para fazer backup
backup_current() {
    if [ -d "$PROJECT_DIR" ]; then
        log "Fazendo backup da versão atual..."
        rm -rf "$BACKUP_DIR"
        cp -r "$PROJECT_DIR" "$BACKUP_DIR"
        log "Backup criado em $BACKUP_DIR ✅"
    fi
}

# Função para restaurar backup
restore_backup() {
    if [ -d "$BACKUP_DIR" ]; then
        warning "Restaurando backup devido a erro..."
        rm -rf "$PROJECT_DIR"
        cp -r "$BACKUP_DIR" "$PROJECT_DIR"
        log "Backup restaurado ✅"
    fi
}

# Criar diretório do projeto se não existir
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Fazer backup se existir versão anterior
backup_current

# Baixar/atualizar código fonte
log "Baixando código fonte..."
if [ -d ".git" ]; then
    info "Repositório Git encontrado, fazendo pull..."
    git pull origin main || error "Falha ao atualizar repositório"
else
    info "Clonando repositório..."
    # Substitua pela URL do seu repositório
    # git clone https://github.com/seu-usuario/trading-dashboard-pro.git .
    info "⚠️  Configure o repositório Git ou copie os arquivos manualmente"
fi

# Instalar dependências
log "Instalando dependências..."
npm ci --production || {
    error "Falha ao instalar dependências"
    restore_backup
}

# Configurar variáveis de ambiente
log "Configurando variáveis de ambiente..."
cat > .env << EOF
# Configuração de Produção
REACT_APP_API_URL=http://62.72.1.122:8000
REACT_APP_WS_URL=ws://62.72.1.122:8000
PORT=5173
HOST=0.0.0.0
GENERATE_SOURCEMAP=false
REACT_APP_VERSION=6.0.0
NODE_ENV=production
EOF

log "Arquivo .env configurado ✅"

# Build da aplicação
log "Fazendo build da aplicação..."
npm run build || {
    error "Falha no build da aplicação"
    restore_backup
}

log "Build concluído ✅"

# Instalar serve globalmente se não estiver instalado
if ! command -v serve &> /dev/null; then
    log "Instalando 'serve' globalmente..."
    npm install -g serve
fi

# Parar serviço existente se estiver rodando
if pgrep -f "serve.*build.*5173" > /dev/null; then
    log "Parando serviço existente..."
    pkill -f "serve.*build.*5173" || true
    sleep 2
fi

# Criar arquivo de serviço systemd
log "Configurando serviço systemd..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << EOF
[Unit]
Description=Trading Dashboard Pro Frontend
Documentation=https://github.com/seu-usuario/trading-dashboard-pro
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=$PROJECT_DIR
Environment=NODE_ENV=production
Environment=PORT=5173
Environment=HOST=0.0.0.0
ExecStart=/usr/bin/npx serve -s build -l 5173 -H 0.0.0.0
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

# Recarregar systemd e iniciar serviço
log "Recarregando systemd..."
sudo systemctl daemon-reload

log "Habilitando serviço para iniciar automaticamente..."
sudo systemctl enable $SERVICE_NAME

log "Iniciando serviço..."
sudo systemctl restart $SERVICE_NAME

# Aguardar alguns segundos e verificar status
sleep 5

if sudo systemctl is-active --quiet $SERVICE_NAME; then
    log "✅ Serviço iniciado com sucesso!"
    log "🌐 Dashboard disponível em: http://62.72.1.122:5173"
else
    error "❌ Falha ao iniciar o serviço. Verifique os logs com: sudo journalctl -u $SERVICE_NAME -f"
fi

# Configurar nginx como proxy reverso (opcional)
if command -v nginx &> /dev/null; then
    log "Configurando nginx como proxy reverso..."
    sudo tee /etc/nginx/sites-available/trading-dashboard > /dev/null << EOF
server {
    listen 80;
    server_name 62.72.1.122;

    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

    # Habilitar site
    sudo ln -sf /etc/nginx/sites-available/trading-dashboard /etc/nginx/sites-enabled/
    
    # Testar configuração
    sudo nginx -t && sudo systemctl reload nginx
    
    log "✅ Nginx configurado como proxy reverso"
    log "🌐 Dashboard também disponível em: http://62.72.1.122"
fi

# Verificar conectividade com backend
log "Verificando conectividade com backend..."
if curl -s -f "http://62.72.1.122:8000/api/status" > /dev/null; then
    log "✅ Backend respondendo corretamente"
else
    warning "⚠️  Backend não está respondendo em http://62.72.1.122:8000"
    warning "    Verifique se o backend está rodando antes de usar o dashboard"
fi

# Mostrar status final
log "📊 Status do Deploy:"
echo "  🔗 Frontend URL: http://62.72.1.122:5173"
echo "  🔗 Backend URL: http://62.72.1.122:8000"
echo "  📝 Logs: sudo journalctl -u $SERVICE_NAME -f"
echo "  🔄 Restart: sudo systemctl restart $SERVICE_NAME"
echo "  ⏹️  Stop: sudo systemctl stop $SERVICE_NAME"

log "🎉 Deploy concluído com sucesso!"

# Remover backup se tudo deu certo
rm -rf "$BACKUP_DIR"

log "✨ Trading Dashboard Pro está rodando em http://62.72.1.122:5173"