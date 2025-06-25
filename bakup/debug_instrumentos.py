#!/usr/bin/env python3
"""
Correção SIMPLES: Comentar a verificação de trade_status temporariamente
"""

import os
import shutil
from datetime import datetime

def aplicar_correcao_simples():
    """Remove temporariamente a verificação de trade_status"""
    
    gate_api_path = "gate_api.py"
    
    if not os.path.exists(gate_api_path):
        print(f"❌ {gate_api_path} não encontrado")
        return False
    
    # Fazer backup
    backup_path = f"{gate_api_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(gate_api_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Ler arquivo
    with open(gate_api_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar correção simples: comentar a verificação de trade_status
    corrections = [
        # Comentar a verificação de trade_status
        ("if inst.get('trade_status') != 'tradable':", "# TEMPORÁRIO: if inst.get('trade_status') != 'tradable':"),
        ("    skipped_reasons['not_tradable'] += 1", "    # TEMPORÁRIO: skipped_reasons['not_tradable'] += 1"),
        ("    continue", "    # TEMPORÁRIO: continue"),
        
        # Ou se estiver em outra parte do código
        ("if inst.get('trade_status') != 'tradable':\n                continue", "# TEMPORÁRIO: Verificação trade_status removida\n                pass"),
    ]
    
    changes_made = 0
    original_content = content
    
    # Tentar encontrar e comentar todas as verificações de trade_status
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Se encontrar verificação de trade_status
        if "inst.get('trade_status')" in line and "!=" in line and "'tradable'" in line:
            new_lines.append(f"            # TEMPORÁRIO: {line.strip()}")
            changes_made += 1
            
            # Comentar também as próximas linhas relacionadas
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('skipped_reasons') or lines[j].strip() == 'continue'):
                new_lines.append(f"            # TEMPORÁRIO: {lines[j].strip()}")
                j += 1
            i = j - 1
        else:
            new_lines.append(line)
        
        i += 1
    
    if changes_made > 0:
        new_content = '\n'.join(new_lines)
        
        # Escrever arquivo corrigido
        with open(gate_api_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ {changes_made} verificações de trade_status comentadas")
        print(f"✅ {gate_api_path} atualizado")
        return True
    else:
        print("ℹ️ Nenhuma verificação de trade_status encontrada para comentar")
        return False

def reverter_correcao():
    """Reverte a correção usando o backup mais recente"""
    
    import glob
    
    # Encontrar o backup mais recente
    backups = glob.glob("gate_api.py.backup_*")
    if not backups:
        print("❌ Nenhum backup encontrado")
        return False
    
    latest_backup = max(backups)
    shutil.copy2(latest_backup, "gate_api.py")
    print(f"✅ Revertido usando backup: {latest_backup}")
    return True

def main():
    """Menu principal"""
    
    print("🔧 CORREÇÃO SIMPLES - REMOVER VERIFICAÇÃO trade_status")
    print("=" * 55)
    print()
    print("Esta correção temporariamente remove a verificação de trade_status")
    print("para permitir que todos os instrumentos passem pelo filtro.")
    print()
    
    while True:
        print("Opções:")
        print("1. Aplicar correção (comentar verificação trade_status)")
        print("2. Reverter correção (restaurar backup)")
        print("3. Sair")
        
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == '1':
            if aplicar_correcao_simples():
                print("\n🎯 PRÓXIMOS PASSOS:")
                print("1. Execute: python3 main.py")
                print("2. Veja se símbolos são encontrados agora")
                print("3. Se funcionar, podemos ajustar os filtros adequadamente")
                break
            else:
                print("❌ Falha na aplicação da correção")
        
        elif choice == '2':
            if reverter_correcao():
                print("✅ Correção revertida")
            else:
                print("❌ Falha ao reverter")
        
        elif choice == '3':
            break
        
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    main()