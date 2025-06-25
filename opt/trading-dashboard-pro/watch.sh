#!/bin/bash

# Comando para monitorar os arquivos no diretório
chokidar 'src/**/*' -c './deploy.sh'   # Altere 'src/**/*' para o diretório que deseja monitorar, e 'deploy.sh' para o script a ser rodado