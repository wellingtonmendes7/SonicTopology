# SonicTopology
Classificador fonestêmico baseado em geometrias 3D
=================================================

**SonicTopology** é um software desenvolvido em Python que permite associar objetos tridimensionais (.obj) a padrões fonológicos recorrentes (fonestemas), extraindo atributos geométricos e treinando uma rede neural para reconhecer relações simbólicas entre forma e som.

Inspirado por estudos sobre simbolismo sonoro (Sapir, 1929; Imai & Kita, 2014), este pipeline automatiza o mapeamento entre ambientes virtuais e fonestemas, oferecendo uma ferramenta inovadora para experimentos computacionais em linguística e design simbólico.

## Características
---------
- Interface gráfica para seleção múltipla de arquivos `.obj`
- Extração automática de atributos geométricos (volume, área, simetria, rugosidade)
- Entrada manual de fonestemas
- Combinação com CSV contendo codificações fonológicas
- Treinamento de rede neural (MLPClassifier) com scikit-learn
- Geração de relatório de classificação

## Requisitos
--------------
1. Python 3.8+
2. Bibliotecas:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `trimesh`
   - `tkinter` (interface gráfica)

Instale as dependências com:

```bash
pip install pandas numpy scikit-learn trimesh
```

## Como usar
------------
1. Execute o script principal:
   ```bash
   python simbolismo_sonoro_pipeline_completo.py
   ```

2. Siga as etapas guiadas:
   - Selecione os arquivos `.obj`
   - Digite um fonestema para cada arquivo
   - Selecione o arquivo `.csv` com codificações fonológicas dos fonestemas

3. O script irá:
   - Calcular atributos dos modelos 3D
   - Combinar esses dados com as codificações fonológicas
   - Treinar uma rede neural MLP
   - Exibir um relatório de desempenho

## Output
--------
- `atributos_extraidos.csv`: contém os atributos geométricos + fonestemas inseridos
- Relatório de classificação no console com precisão, recall e f1-score por classe

## Example Attributes Extracted
| nome_arquivo       | volume | area_superficial | simetria_x | rugosidade | fonestema |
|--------------------|--------|------------------|------------|------------|-----------|
| cave_high.obj      | 10.32  | 25.61            | 0.89       | 0.12       | sl-       |
| cave_low.obj       | 5.12   | 14.92            | 0.45       | 0.33       | gl-       |

## Customização
---------------
- Altere a arquitetura da rede neural modificando os parâmetros de `MLPClassifier`:
```python
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
```
- Substitua o método de input manual para leitura automática de fonestemas se desejar trabalhar com arquivos já anotados.

## Limitações
-------------------
- Não realiza pré-processamento de malhas inválidas (usa convex hull quando necessário).
- A entrada de fonestemas é manual.
- Requer correspondência exata entre nomes no CSV e entradas digitadas.

## Autor
-------
Wellington Mendes (UFU, 2025)  
   - Email: wellington.mendes@ufu.br  
   - [Perfil Institucional](http://www.portal.ileel.ufu.br/pessoas/docentes/wellington-araujo-mendes-junior)  
   - [Google Scholar](https://scholar.google.com/citations?user=eI4709wAAAAJ&hl=pt-BR)

## Licença
-------
Este software é distribuído sob a Licença MIT. Livre para modificar e reutilizar.
