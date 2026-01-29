# PotentialPlotter3D (ETAP Profiles) - 3D Surface + Ground Grid Overlay

Este projeto plota superficies 3D (X, Y, Z) a partir de arquivos `.dat` e sobrepoe a malha (grid) de aterramento lida de um arquivo Excel com segmentos (X1, Y1, X2, Y2).

O script gera:
- 1 grafico 3D por arquivo `.dat`
- 1 arquivo `.png` por grafico (mesmo nome do `.dat`, trocando a extensao)

## Requisitos

Python 3.9+ recomendado.

Bibliotecas:
- numpy
- pandas
- matplotlib

Instalacao:
```bash
pip install numpy pandas matplotlib openpyxl
```

## Arquivos de entrada

### 1) Arquivo Excel da malha (mesh_excel_path)

O Excel deve conter obrigatoriamente as colunas:

- X1, Y1, X2, Y2

Cada linha representa um segmento de reta:
- Ponto inicial: (X1, Y1)
- Ponto final:   (X2, Y2)

Celulas vazias sao substituidas por zero.

### 2) Arquivos .dat (superficie)

Cada arquivo `.dat` deve conter **pelo menos 3 colunas**, nesta ordem:

1. X
2. Y
3. Z

Formato esperado:
- Separador principal: TAB (`\t`)
- Sem cabecalho
- Uma linha = um ponto (X, Y, Z)
- Decimais podem ser `.` ou `,`
- Linhas vazias sao descartadas automaticamente
- Codificacao tentativa: utf-16 -> utf-8 -> latin1

## Saidas

Para cada `.dat`, o script salva:
- `NomeDoArquivo.dat` -> `NomeDoArquivo.png`

As imagens sao salvas com `dpi=300`.
