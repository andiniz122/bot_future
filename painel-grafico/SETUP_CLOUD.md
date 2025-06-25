# 🚀 Setup Trading Dashboard Pro na Nuvem

## 📋 Configuração para o servidor 62.72.1.122:5173

### ✅ Pré-requisitos

1. **Servidor com acesso SSH**
   - IP: `62.72.1.122`
   - Portas liberadas: `5173` (frontend), `8000` (backend)

2. **Software necessário:**
   ```bash
   # Node.js 18+
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Git
   sudo apt-get install git
   
   # PM2 (opcional, recomendado)
   sudo npm install -g pm2
   
   # Nginx (opcional, para proxy)
   sudo apt-get install nginx
   ```

### 🔧 Configuração do Projeto

1. **Clone e configuração inicial:**
   ```bash
   # Faça upload dos arquivos para o servidor ou clone do repositório
   cd /var/www/trading-dashboard-pro
   
   # Copie o arquivo de ambiente
   cp .env.example .env.production
   
   # Edite as configurações se necessário
   nano .env.production
   ```

2. **Configuração das URLs:**
   - Frontend: `http://62.72.1.122:5173`
   - API Backend: `http://62.72.1.122:8000`
   - WebSocket: `ws://62.72.1.122:8000`

### 🚀 Deploy Options

#### Opção 1: Deploy Automático (Recomendado)
```bash
# Torna o script executável
chmod +x deploy.sh

# Executa o deploy
./deploy.sh
```

#### Opção 2: Deploy Manual com PM2
```bash
# Instalar dependências
npm ci

# Build do projeto
npm run build

# Iniciar com PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

#### Opção 3: Deploy com Docker
```bash
# Build da imagem
docker build -t trading-dashboard-pro .

# Executar container
docker run -d \
  --name trading-dashboard-pro \
  --restart unless-stopped \
  -p 5173:5173 \
  trading-dashboard-pro
```

#### Opção 4: Deploy Simples
```bash
# Instalar dependências
npm ci

# Build do projeto
npm run build

# Iniciar servidor
npm run preview -- --host 0.0.0.0 --port 5173
```

### 🌐 Configuração do Nginx (Opcional)

1. **Copiar configuração:**
   ```bash
   sudo cp nginx.conf /etc/nginx/sites-available/trading-dashboard
   sudo ln -s /etc/nginx/sites-available/trading-dashboard /etc/nginx/sites-enabled/
   ```

2. **Testar e reiniciar:**
   ```bash
   sudo nginx -t
   sudo systemctl reload nginx
   ```

### 🔒 Configuração de Firewall

```bash
# UFW (Ubuntu)
sudo ufw allow 5173/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# iptables
sudo iptables -A INPUT -p tcp --dport 5173 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

### 📊 Monitoramento

#### Com PM2:
```bash
# Status dos processos
pm2 status

# Logs em tempo real
pm2 logs trading-dashboard-pro

# Monitor de recursos
pm2 monit

# Restart
pm2 restart trading-dashboard-pro
```

#### Com Docker:
```bash
# Status do container
docker ps

# Logs
docker logs trading-dashboard-pro -f

# Restart
docker restart trading-dashboard-pro
```

#### Verificações de saúde:
```bash
# Teste local
curl http://localhost:5173

# Teste externo
curl http://62.72.1.122:5173

# Teste WebSocket
wscat -c ws://62.72.1.122:8000/ws/sentiment
```

### 🔧 Troubleshooting

#### Problemas comuns:

1. **Porta 5173 ocupada:**
   ```bash
   # Encontrar processo
   lsof -i:5173
   
   # Matar processo
   sudo kill -9 <PID>
   ```

2. **Permissões de arquivo:**
   ```bash
   # Corrigir ownership
   sudo chown -R $USER:$USER /var/www/trading-dashboard-pro
   
   # Permissões corretas
   chmod 755 deploy.sh
   ```

3. **Problemas de memória:**
   ```bash
   # Limpar cache npm
   npm cache clean --force
   
   # Aumentar limite de memória Node.js
   export NODE_OPTIONS="--max-old-space-size=4096"
   ```

4. **WebSocket não conecta:**
   - Verificar se backend está rodando na porta 8000
   - Testar conexão direta: `telnet 62.72.1.122 8000`
   - Verificar firewall/proxy

5. **Build falha:**
   ```bash
   # Limpar node_modules
   rm -rf node_modules package-lock.json
   npm install
   
   # Build com debug
   npm run build -- --verbose
   ```

### 📈 URLs de Acesso

Após o deploy bem-sucedido:

- **Dashboard Principal:** http://62.72.1.122:5173
- **API Health Check:** http://62.72.1.122:8000/api/health
- **WebSocket Test:** ws://62.72.1.122:8000/ws/sentiment

### 🔄 Atualizações

Para atualizar o dashboard:

```bash
# Parar aplicação
pm2 stop trading-dashboard-pro

# Atualizar código
git pull origin main

# Reinstalar dependências se necessário
npm ci

# Rebuild
npm run build

# Reiniciar
pm2 start trading-dashboard-pro
```

### 📝 Logs

Localizações dos logs:

- **PM2:** `~/.pm2/logs/`
- **Aplicação:** `./logs/`
- **Nginx:** `/var/log/nginx/`
- **Sistema:** `/var/log/syslog`

### 🆘 Suporte

Em caso de problemas:

1. Verificar logs da aplicação
2. Testar conectividade de rede
3. Verificar status do backend
4. Confirmar configurações de firewall
5. Testar em modo de desenvolvimento local

### 📊 Performance

Para otimizar performance:

1. **Nginx:** Configure cache e compressão
2. **PM2:** Use cluster mode se necessário
3. **Docker:** Configure limites de recursos
4. **Sistema:** Monitor CPU, RAM e rede

---

**🎯 Objetivo:** Dashboard acessível em `http://62.72.1.122:5173` com conexões estáveis para backend em `http://62.72.1.122:8000`