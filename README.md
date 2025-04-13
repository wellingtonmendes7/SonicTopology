# SonicTopology
Mapping symbolic sound patterns onto 3D structures

# -*- coding: utf-8 -*-
"""
Simbolismo Sonoro e Modelagem Tridimensional - Pipeline Completo

Este script realiza:
1. Seleção de múltiplos arquivos .obj
2. Extração de atributos geométricos com trimesh
3. Salvamento em CSV
4. Seleção do CSV de fonestemas codificados
5. Junção e treinamento com MLPClassifier
6. Exibição de relatório de desempenho

Requisitos: pandas, scikit-learn, trimesh, tkinter
"""

import trimesh
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from tkinter import filedialog, Tk, messagebox
import os

def calcular_rugosidade(mesh):
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    curvaturas = mesh.vertex_normals
    variabilidade = np.std(curvaturas)
    return float(variabilidade)

def extrair_atributos(path):
    mesh = trimesh.load_mesh(path)
    atributos = {
        'nome_arquivo': os.path.basename(path),
        'volume': float(mesh.volume),
        'area_superficial': float(mesh.area),
        'num_vertices': len(mesh.vertices),
        'num_arestas': len(mesh.edges),
        'num_faces': len(mesh.faces),
        'simetria_x': float(mesh.symmetry().get('x', np.nan)),
        'simetria_y': float(mesh.symmetry().get('y', np.nan)),
        'simetria_z': float(mesh.symmetry().get('z', np.nan)),
        'rugosidade': calcular_rugosidade(mesh),
    }
    return atributos

def selecionar_arquivos_obj():
    root = Tk()
    root.withdraw()
    caminhos = filedialog.askopenfilenames(title="Selecione arquivos .obj", filetypes=[("OBJ files", "*.obj")])
    return list(caminhos)

def selecionar_arquivo_csv(titulo):
    root = Tk()
    root.withdraw()
    caminho = filedialog.askopenfilename(title=titulo, filetypes=[("CSV Files", "*.csv")])
    return caminho

def main():
    print("=== Simbolismo Sonoro e Modelagem 3D ===")

    caminhos_obj = selecionar_arquivos_obj()
    if not caminhos_obj:
        messagebox.showerror("Erro", "Nenhum arquivo .obj selecionado.")
        return

    lista_atributos = []
    for path in caminhos_obj:
        try:
            atributos = extrair_atributos(path)
            print(f"Atributos extraídos de: {atributos['nome_arquivo']}")
            lista_atributos.append(atributos)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {path}: {e}")

    df_atributos = pd.DataFrame(lista_atributos)

    # Solicitar entrada manual de fonestema
    fonestemas = []
    print("\nInsira um fonestema para cada arquivo .obj:")
    for nome in df_atributos['nome_arquivo']:
        valor = input(f"Fonestema para {nome}: ")
        fonestemas.append(valor.strip())
    df_atributos['fonestema'] = fonestemas

    # Salvar atributos em CSV
    output_csv = "atributos_extraidos.csv"
    df_atributos.to_csv(output_csv, index=False)
    print(f"\nAtributos salvos em {output_csv}")

    # Selecionar codificação fonestêmica
    print("\nSelecione o CSV com codificação dos fonestemas:")
    caminho_fono = selecionar_arquivo_csv("Selecionar codificação fonestêmica")
    if not caminho_fono:
        messagebox.showerror("Erro", "Arquivo de fonestemas não selecionado.")
        return

    fono_df = pd.read_csv(caminho_fono)
    df_final = pd.merge(df_atributos, fono_df, on="fonestema")
    X = df_final.drop(columns=["nome_arquivo", "fonestema", "consoante_1", "consoante_2"])
    y = df_final["fonestema"]

    # Treinar modelo
    print("\nTreinando rede neural MLP...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # Relatório
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_test, y_pred))

    print("\nPipeline concluído com sucesso.")

if __name__ == "__main__":
    main()
