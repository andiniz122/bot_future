# Multi-stage build para otimizar o tamanho da imagem

# Stage 1: Build da aplicação React
FROM node:18-alpine as build

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY package*.json ./

# Instalar dependências
RUN npm ci --only=production

# Copiar código fonte
COPY . .

# Definir variáveis de ambiente para build
ENV REACT_APP_API_URL=http://62.72.1.122:8000
ENV REACT_APP_WS_URL=ws://62.72.1.122:8000
ENV GENERATE_SOURCEMAP=false

# Build da aplicação
RUN npm run build

# Stage 2: Servidor de produção
FROM nginx:alpine

# Copiar configuração customizada do nginx
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 5173;
    server_name _;
    
    # Localização dos arquivos estáticos
    root /usr/share/nginx/html;
    index index.html;
    
    # Configuração para SPA (Single Page Application)
    location / {
        try_files \$uri \$uri/ /index.html;
        
        # Headers de cache para arquivos estáticos
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # Headers de segurança
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Configuração para WebSockets (proxy reverso se necessário)
    location /ws/ {
        proxy_pass http://62.72.1.122:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Configuração para API (proxy reverso se necessário)
    location /api/ {
        proxy_pass http://62.72.1.122:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Compressão gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
}
EOF

# Copiar arquivos buildados do stage anterior
COPY --from=build /app/build /usr/share/nginx/html

# Expor porta
EXPOSE 5173

# Comando para iniciar o nginx
CMD ["nginx", "-g", "daemon off;"]