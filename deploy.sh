#!/bin/bash

echo "===> Indo para o projeto"
cd /opt/trading-dashboard-pro || exit

echo "===> Puxando últimas alterações do GitHub"
git pull origin main

echo "===> Instalando dependências (caso haja mudanças)"
npm install

echo "===> Gerando build de produção"
npm run build

echo "===> Iniciando o front-end"
npm run serve
