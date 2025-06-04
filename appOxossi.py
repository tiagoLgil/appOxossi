import sys
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import PyPDF2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget,
                           QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                           QLabel, QTextEdit, QLineEdit, QCheckBox, QComboBox,
                           QGroupBox, QPlainTextEdit, QProgressBar, QScrollArea,
                           QSplitter, QFrame, QMessageBox, QInputDialog, QSizePolicy,
                           QSpacerItem)
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tempfile
import datetime
from collections import Counter

# NOVO: Importações para geração de PDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import io

# Configurar matplotlib para modo não-interativo para evitar problemas com threading
import matplotlib
matplotlib.use('Agg')  # Usando 'Agg' em vez de 'Qt5Agg' para maior compatibilidade
plt.ioff()  # Desativar modo interativo para evitar problemas de threading

# Tente importar lingua, se não estiver disponível, use uma implementação simples
try:
    from lingua import Language, LanguageDetectorBuilder
    languages = [Language.PORTUGUESE, Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def detect_language(text):
        return detector.detect_language_of(text) == Language.PORTUGUESE
except ImportError:
    # Implementação simples para detecção de português (fallback)
    def detect_language(text):
        portuguese_words = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para']
        words = text.lower().split()
        portuguese_count = sum(1 for word in words if word in portuguese_words)
        return portuguese_count / len(words) > 0.2 if words else False

# Carregue o modelo spaCy para português
try:
    processamento = spacy.load('pt_core_news_md')
except OSError:
    try:
        # Tentar um modelo menor que provavelmente já está instalado
        processamento = spacy.load('pt_core_news_sm')
        print("Usando modelo spaCy pt_core_news_sm")
    except OSError:
        print("Modelo spaCy não encontrado. Tentando fazer o download...")
        try:
            from spacy.cli import download
            download('pt_core_news_sm')
            processamento = spacy.load('pt_core_news_sm')
        except Exception as e:
            print(f"Erro ao carregar o modelo spaCy: {e}")
            processamento = None

# NOVA FUNÇÃO: Análise Temporal (baseada no anospythonSEMséculo.py)
def is_likely_year(text, year, window=50):
    """Verifica se um número é provavelmente um ano baseado no contexto"""
    # Get the context around the year
    start = max(0, text.find(str(year)) - window)
    end = min(len(text), text.find(str(year)) + len(str(year)) + window)
    context = text[start:end].lower()

    # Keywords that suggest a year
    year_indicators = ['ano', 'em', 'de', 'século', 'década', 'período', 'era']

    # Check if any year indicator is in the context
    for indicator in year_indicators:
        if indicator in context:
            return True

    # Check if the year is part of a date format
    if re.search(r'\d{1,2}/\d{1,2}/' + str(year), context):
        return True

    # If no strong indicator is found, it's less likely to be a year
    return False

def roman_to_int(s):
    """Converte números romanos para inteiros"""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for i in range(len(s)):
        if i > 0 and roman_map[s[i]] > roman_map[s[i - 1]]:
            result += roman_map[s[i]] - 2 * roman_map[s[i - 1]]
        else:
            result += roman_map[s[i]]
    return result

def analisar_periodo_temporal(full_content):
    """
    Analisa o período temporal sobre o qual o texto fala
    Retorna dicionário com informações temporais
    """
    try:
        # Extract centuries mentioned in the format "século XVIII"
        centuries_str = re.findall(r'século ([IVX]+)', full_content)

        # Convert centuries to starting years (e.g., século XVIII -> 1700)
        century_years = [(roman_to_int(c) - 1) * 100 + 50 for c in centuries_str]

        # Extract all 4-digit numbers that could represent years
        years_str = re.findall(r'\b\d{4}\b', full_content)

        # Filter out years outside the range (1500 to 1850) and check if they're likely years
        years = [int(year) for year in years_str if 1500 <= int(year) <= 1850 and is_likely_year(full_content, year)]
        century_years = [int(year) for year in century_years if 1500 <= int(year) <= 1850]

        # Combine the years from both extractions
        all_years = years  # + century_years (como no código original)

        resultado = {
            'anos_encontrados': all_years,
            'total_anos': len(all_years),
            'periodo_encontrado': False,
            'media_anos': None,
            'desvio_padrao': None,
            'desvio_medio_absoluto': None,
            'periodo_principal': None,
            'periodo_alargado': None,
            'seculos_mencionados': centuries_str,
            'anos_seculo': century_years
        }

        if not all_years:
            resultado['mensagem'] = "Nenhum ano válido encontrado no período especificado (1500-1850)."
            return resultado

        # Calcular estatísticas
        minX = min(all_years)
        maxX = max(all_years)

        # Calculate and get the average year
        average_year = sum(all_years) / len(all_years)
        resultado['media_anos'] = round(average_year)

        # Calculate standard deviation
        std_dev = np.std(all_years)
        resultado['desvio_padrao'] = round(std_dev)

        # Calcular desvio médio absoluto (como no código original)
        dados2 = np.array(all_years)
        media2 = np.mean(dados2)
        desviosAbsolutos = np.abs(dados2 - media2)
        dma = np.mean(desviosAbsolutos)
        resultado['desvio_medio_absoluto'] = round(dma)

        # Determinar período principal usando DMA (como no código original)
        primoAno = average_year - dma
        FinaleAno = average_year + dma

        resultado['periodo_principal'] = (round(primoAno), round(FinaleAno))
        resultado['periodo_alargado'] = (minX, maxX)
        resultado['periodo_encontrado'] = True

        # Determinar se é um período específico ou amplo
        year_range = maxX - minX
        if year_range <= 2:
            resultado['tipo_periodo'] = 'específico'
        else:
            resultado['tipo_periodo'] = 'amplo'
            start_century = (minX // 100) + 1
            end_century = (maxX // 100) + 1
            resultado['seculos_periodo'] = (start_century, end_century)

        return resultado

    except Exception as e:
        return {
            'anos_encontrados': [],
            'total_anos': 0,
            'periodo_encontrado': False,
            'erro': str(e)
        }

# IMPLEMENTAÇÃO PERSONALIZADA DO PAGERANK (SEM SCIPY)
def custom_pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6):
    """
    Implementação customizada do PageRank que não depende do SciPy
    """
    if len(G) == 0:
        return {}

    # Inicializar valores
    n = len(G)
    x = {node: 1.0 / n for node in G.nodes()}

    # Iterar até convergência
    for _ in range(max_iter):
        x_last = x.copy()
        x = {node: 0.0 for node in G.nodes()}

        for node in G.nodes():
            rank = (1.0 - alpha) / n

            # Somar contribuições dos nós que apontam para este
            for neighbor in G.predecessors(node) if G.is_directed() else G.neighbors(node):
                if G.is_directed():
                    out_degree = G.out_degree(neighbor)
                else:
                    out_degree = G.degree(neighbor)

                if out_degree > 0:
                    rank += alpha * x_last[neighbor] / out_degree

            x[node] = rank

        # Verificar convergência
        err = sum(abs(x[node] - x_last[node]) for node in G.nodes())
        if err < n * tol:
            break

    return x

# Funções de processamento de texto adaptadas do código original
def limpeza(texto):
    texto = texto.lower()
    # Esta lista 'remover' é para limpeza geral de caracteres e frases específicas, não stopwords editáveis.
    remover = ["/","[","]", "-", "=", ">","'", '"',",",")","(","_","_"," parte "," forma ","revista","pesquisa","histórica"]
    for item in remover:
        texto = texto.replace(item,"  ")
    texto = texto.replace("século ","século_")
    texto = texto.replace("__","_")
    texto = re.sub(r"\d+","",texto)
    texto = texto.replace(":",". ")
    texto = texto.strip()
    return texto

def limpezaM(texto, stopwords_personalizadas=None):
    """
    Função de limpeza mais agressiva para textos grandes.
    Agora usa stopwords personalizadas passadas como argumento para remoção.
    """
    texto = texto.lower()
    texto = limpeza(texto) # Continua a limpeza básica (pontuação, etc.)

    if stopwords_personalizadas:
        # Criar um conjunto para busca mais rápida
        stopwords_set = set(stopwords_personalizadas)
        # Substituir stopwords por espaços, garantindo correspondência de palavra inteira
        for word in stopwords_set:
            # Usar regex para correspondência de palavra inteira (\b) e escapar caracteres especiais
            texto = re.sub(r'\b' + re.escape(word) + r'\b', ' ', texto)

    texto = re.sub(r"\d+", "", texto)
    texto = texto.strip()
    # Limpar múltiplos espaços que podem resultar das substituições
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def contarPalavras(lista):
    return Counter(lista)

def nTokens(texto):
    Ntokens = ["antigo sistema colonial","rio de janeiro","são paulo","rio grande de são pedro","Grão-Pará","Rio Grande","Santa Catarina","antigo regime","brasil colonial","américa portuguesa","companhia de jesus","nossa senhora"]
    for item in Ntokens:
        itemCorrigido = item.replace(" ","_")
        texto = texto.replace(item,itemCorrigido)
    return texto

def biTokenFinder(texto):
    texto = limpeza(texto)
    listTexto = texto.split(" ")
    listaDeBitokens = []
    anterior = ""
    for item in listTexto:
        atual = item
        if len(anterior) > 4 and len(atual) > 4:
            resultado = anterior + "_" + atual
            listaDeBitokens.append(resultado)
        anterior = atual

    resultados = contarPalavras(listaDeBitokens)
    paraExportar = []
    if len(resultados) > 0:
        biTokenMaisImportante = max(resultados, key=resultados.get)
        valores = resultados.values()
        valorDoMaisAlto = resultados[biTokenMaisImportante]
        paraExportar.append(biTokenMaisImportante)
        paraExportar.append(valorDoMaisAlto)
    else:
        paraExportar.append("-")
    return paraExportar

def substantivador(texto, exceto): # Removido stopwords_personalizadas, pois limpezaM já as remove
    if not processamento:
        return []

    lista = []
    resposta = processamento(texto)
    for token in resposta:
        if token.lemma_ == exceto:
            lista.append(token)
        elif token.pos_ == "NOUN":  # Adaptado para usar pos_ em vez de pos
            lista.append(token.lemma_)
    return lista

def filtrar_por_palavra(linhas, palavra_filtro):
    if not palavra_filtro:
        return linhas

    return [linha for linha in linhas if palavra_filtro.lower() in linha.lower()]

def classificar_macro_tema(texto, temas):
    if not temas:
        return {}

    resultados = {}
    for tema, palavras in temas.items():
        contagem = 0
        for palavra in palavras:
            padrao = r'\b' + re.escape(palavra.lower()) + r'\b'
            contagem += len(re.findall(padrao, texto.lower()))
        resultados[tema] = contagem

    return resultados

def listaParaRedeSubstantivos(texto, palavra_filtro=None, macro_temas=None, stopwords_personalizadas=None):
    # Aplica limpeza básica ou agressiva, que agora inclui a remoção de stopwords
    if len(texto) < 120000:
        # Para textos menores, a função limpeza não remove stopwords.
        # Se desejar remover stopwords também para textos menores,
        # seria necessário aplicar a lógica de stopwords aqui ou em 'limpeza'.
        # Por enquanto, mantém o comportamento original de 'limpeza' sem stopwords.
        texto_limpo = limpeza(texto)
        # Para consistência, podemos aplicar a remoção de stopwords aqui também para textos pequenos.
        if stopwords_personalizadas:
            stopwords_set = set(stopwords_personalizadas)
            words = texto_limpo.split()
            filtered_words = [word for word in words if word not in stopwords_set]
            texto_limpo = " ".join(filtered_words)
            texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip() # Limpar espaços extras
    else:
        texto_limpo = limpezaM(texto, stopwords_personalizadas) # Passa stopwords personalizadas

    texto_limpo = nTokens(texto_limpo)
    biToken = biTokenFinder(texto_limpo)
    biToken = biToken[0]
    biTokenSemEspaco = biToken.replace("_"," ")
    texto_limpo = texto_limpo.replace(biTokenSemEspaco, biToken)

    linhas = texto_limpo.split(". ")

    # Aplicar filtro de palavras se especificado
    if palavra_filtro:
        linhas = filtrar_por_palavra(linhas, palavra_filtro)

    # Classificar por macro-temas se especificado
    tema_resultados = {}
    if macro_temas:
        tema_resultados = classificar_macro_tema(texto_limpo, macro_temas)

    mistura = []
    for item in linhas:
        # substantivador agora não precisa mais de stopwords_personalizadas
        resposta = substantivador(item, biToken)
        if resposta and len(resposta) > 1:
            for palavraH in resposta:
                for palavraV in resposta:
                    if (palavraV is not palavraH and len(str(palavraV)) > 3 and
                        len(str(palavraH)) > 3 and len(str(palavraH)) < 14 and
                        len(str(palavraV)) < 14):

                        palavraV = str(palavraV)
                        palavraV = palavraV.replace("_"," ")
                        palavraH = str(palavraH)
                        palavraH = palavraH.replace("_"," ")
                        mistura.append(tuple((palavraV, palavraH)))

    return mistura, tema_resultados

def gerar_grafico_rede(arestas, limite_nos=100):
    G = nx.Graph()

    # Adicionar apenas as arestas mais frequentes para evitar gráficos muito densos
    contagem_arestas = {}
    for aresta in arestas:
        if aresta in contagem_arestas:
            contagem_arestas[aresta] += 1
        else:
            contagem_arestas[aresta] = 1

    # Ordenar arestas por frequência e adicionar apenas as top X
    arestas_ordenadas = sorted(contagem_arestas.items(), key=lambda x: x[1], reverse=True)
    arestas_top = [aresta for aresta, _ in arestas_ordenadas[:limite_nos]]

    G.add_edges_from(arestas_top)

    # Usar implementação customizada do PageRank
    try:
        pr = custom_pagerank(G, alpha=0.8)
    except Exception as e:
        print(f"Erro no cálculo do PageRank: {e}")
        # Fallback: usar grau dos nós como medida de importância
        pr = {node: G.degree(node) for node in G.nodes()}
        # Normalizar
        max_degree = max(pr.values()) if pr.values() else 1
        pr = {node: degree/max_degree for node, degree in pr.items()}

    # Extrair os nós mais importantes
    nos_importantes = sorted(pr, key=pr.get, reverse=True)[:15]

    return G, nos_importantes, pr

# Função para gerar e salvar a imagem do grafo
def salvar_grafico_rede(G, nos_importantes, pr, tamanho_figura=(10, 8)):
    # Criar uma nova figura para evitar problemas de threading
    fig = Figure(figsize=tamanho_figura)
    ax = fig.add_subplot(111)

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'Nenhum nó encontrado\nno grafo',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
    else:
        # Posição dos nós
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

        # Tamanhos de nós baseados no PageRank
        max_pr = max(pr.values()) if pr.values() else 1
        node_sizes = [pr.get(node, 0) * 8000 / max_pr + 500 for node in G.nodes()] # Adicionado offset para visibilidade

        # Cores dos nós - mais importantes em vermelho
        node_colors = ['#FF6F61' if node in nos_importantes[:6] else '#6A8EAE' for node in G.nodes()] # Tons mais cativantes

        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0.5, edgecolors='#333333')

        # Desenhar arestas com baixa opacidade
        nx.draw_networkx_edges(G, pos, ax=ax,
                              width=0.7, alpha=0.4, edge_color='#555555') # Cores mais suaves para arestas

        # Adicionar rótulos apenas para os nós mais importantes
        labels = {node: node for node in nos_importantes[:10]}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                              font_size=10, font_weight='bold', font_color='#333333')

    # Remover eixos para uma visualização mais limpa
    ax.set_axis_off()

    # Criar nome de arquivo com timestamp para evitar sobrescrever
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"rede_palavras_{timestamp}.png")

    # Salvar a figura com dpi maior para melhor qualidade
    fig.tight_layout()
    fig.savefig(filename, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True) # DPI maior e fundo transparente

    return filename

# NOVA FUNÇÃO: Gerar relatório PDF com análise temporal
def gerar_relatorio_pdf(resultado, arquivo_origem, palavra_filtro=None, caminho_saida=None):
    """
    Gera um relatório PDF completo com gráfico, resultados e análise temporal
    """
    if not caminho_saida:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_saida = os.path.join(os.path.expanduser("~"), f"relatorio_analise_texto_{timestamp}.pdf")

    # Criar documento PDF
    doc = SimpleDocTemplate(caminho_saida, pagesize=A4)
    elementos = []

    # Estilos
    estilos = getSampleStyleSheet()

    # Estilo personalizado para título
    estilo_titulo = ParagraphStyle(
        'CustomTitle',
        parent=estilos['Heading1'],
        fontSize=24, # Aumentado
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2C3E50') # Cor mais escura
    )

    # Estilo para subtítulos
    estilo_subtitulo = ParagraphStyle(
        'CustomSubtitle',
        parent=estilos['Heading2'],
        fontSize=16, # Aumentado
        spaceBefore=20, # Mais espaço acima
        spaceAfter=12,
        textColor=colors.HexColor('#C0392B') # Cor mais suave
    )

    # Estilo para texto normal
    estilo_normal = ParagraphStyle(
        'CustomNormal',
        parent=estilos['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        leading=14 # Espaçamento de linha
    )

    # Estilo para itens de lista (bullets)
    estilo_lista = ParagraphStyle(
        'CustomList',
        parent=estilos['Normal'],
        fontSize=10,
        spaceBefore=0,
        spaceAfter=3,
        leftIndent=36,
        firstLineIndent=-18,
        alignment=TA_LEFT
    )

    # CABEÇALHO DO RELATÓRIO
    elementos.append(Paragraph("Relatório de Análise de Texto Histórico", estilo_titulo))
    elementos.append(Spacer(1, 20))

    # Informações gerais
    data_atual = datetime.datetime.now().strftime("%d/%m/%Y às %H:%M")
    info_geral = f"""
    <b>Data do relatório:</b> {data_atual}<br/>
    <b>Arquivo analisado:</b> {os.path.basename(arquivo_origem) if arquivo_origem else 'N/A'}<br/>
    """

    if palavra_filtro:
        info_geral += f"<b>Filtro aplicado:</b> '{palavra_filtro}'<br/>"

    elementos.append(Paragraph(info_geral, estilo_normal))
    elementos.append(Spacer(1, 20))

    # NOVA SEÇÃO: ANÁLISE TEMPORAL
    if 'analise_temporal' in resultado and resultado['analise_temporal']['periodo_encontrado']:
        elementos.append(Paragraph("1. Análise do Período Temporal", estilo_subtitulo))

        temporal = resultado['analise_temporal']

        # Criar tabela com informações temporais
        dados_temporais = [['Métrica', 'Valor']]
        dados_temporais.append(['Total de anos identificados', str(temporal['total_anos'])])
        dados_temporais.append(['Ano médio', str(temporal['media_anos'])])
        dados_temporais.append(['Desvio padrão', str(temporal['desvio_padrao'])])
        dados_temporais.append(['Desvio médio absoluto', str(temporal['desvio_medio_absoluto'])])

        if temporal['periodo_principal']:
            periodo_principal = f"{temporal['periodo_principal'][0]} - {temporal['periodo_principal'][1]}"
            dados_temporais.append(['Período principal', periodo_principal])

        if temporal['periodo_alargado']:
            periodo_alargado = f"{temporal['periodo_alargado'][0]} - {temporal['periodo_alargado'][1]}"
            dados_temporais.append(['Período total', periodo_alargado])

        tabela_temporal = Table(dados_temporais)
        tabela_temporal.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')), # Verde escuro
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D4EFDF')), # Verde claro
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elementos.append(tabela_temporal)
        elementos.append(Spacer(1, 15))

        # Adicionar séculos mencionados se houver
        if temporal.get('seculos_mencionados'):
            seculos_texto = f"<b>Séculos mencionados:</b> {', '.join(temporal['seculos_mencionados'])}"
            elementos.append(Paragraph(seculos_texto, estilo_normal))
            elementos.append(Spacer(1, 15))

    # SEÇÃO 2: ESTATÍSTICAS DA REDE
    elementos.append(Paragraph("2. Estatísticas da Rede de Palavras", estilo_subtitulo))

    G = resultado['grafo']
    estatisticas = f"""
    <b>Número de nós (palavras):</b> {G.number_of_nodes()}<br/>
    <b>Número de conexões:</b> {G.number_of_edges()}<br/>
    <b>Densidade da rede:</b> {nx.density(G):.4f}<br/>
    """

    elementos.append(Paragraph(estatisticas, estilo_normal))
    elementos.append(Spacer(1, 15))

    # SEÇÃO 3: PALAVRAS MAIS IMPORTANTES
    elementos.append(Paragraph("3. Palavras Mais Importantes", estilo_subtitulo))

    nos_importantes = resultado['nos_importantes'][:15]
    pagerank = resultado['pagerank']

    # Criar tabela com as palavras mais importantes
    dados_tabela = [['Posição', 'Palavra', 'Importância (PageRank)']]

    for i, palavra in enumerate(nos_importantes, 1):
        importancia = pagerank.get(palavra, 0)
        dados_tabela.append([str(i), palavra, f"{importancia:.4f}"])

    tabela = Table(dados_tabela)
    tabela.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')), # Cinza-azul escuro
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')), # Cinza claro
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elementos.append(tabela)
    elementos.append(Spacer(1, 20))

    # SEÇÃO 4: ANÁLISE DE MACRO-TEMAS (se disponível)
    if resultado.get('tema_resultados'):
        elementos.append(Paragraph("4. Análise de Macro-temas", estilo_subtitulo))

        tema_resultados = resultado['tema_resultados']
        total_ocorrencias = sum(tema_resultados.values())

        if total_ocorrencias > 0:
            temas_ordenados = sorted(tema_resultados.items(), key=lambda x: x[1], reverse=True)

            # Criar tabela de temas
            dados_temas = [['Macro-tema', 'Ocorrências', 'Percentual']]

            for tema, contagem in temas_ordenados:
                porcentagem = (contagem / total_ocorrencias) * 100
                dados_temas.append([tema, str(contagem), f"{porcentagem:.1f}%"])

            tabela_temas = Table(dados_temas)
            tabela_temas.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')), # Azul
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D6EEF9')), # Azul claro
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elementos.append(tabela_temas)
            elementos.append(Spacer(1, 20))
        else:
            elementos.append(Paragraph("Nenhuma ocorrência de macro-temas encontrada.", estilo_normal))
            elementos.append(Spacer(1, 15))

    # QUEBRA DE PÁGINA ANTES DO GRÁFICO
    elementos.append(PageBreak())

    # SEÇÃO 5: VISUALIZAÇÃO DA REDE
    elementos.append(Paragraph("5. Visualização da Rede de Palavras", estilo_subtitulo))
    elementos.append(Spacer(1, 15))

    # Adicionar imagem do gráfico
    if 'imagem_path' in resultado and os.path.exists(resultado['imagem_path']):
        try:
            # Redimensionar imagem para caber na página
            img = Image(resultado['imagem_path'])
            img.drawHeight = 5.5*inch # Levemente maior
            img.drawWidth = 7.5*inch # Levemente maior
            img.hAlign = 'CENTER' # Centralizar imagem
            elementos.append(img)
        except Exception as e:
            elementos.append(Paragraph(f"Erro ao inserir imagem: {str(e)}", estilo_normal))
    else:
        elementos.append(Paragraph("Imagem do gráfico não disponível.", estilo_normal))

    elementos.append(Spacer(1, 20))

    # SEÇÃO 6: INTERPRETAÇÃO DOS RESULTADOS
    elementos.append(Paragraph("6. Interpretação dos Resultados", estilo_subtitulo))
    interpretacao = """
    Este relatório apresenta uma análise automatizada de texto histórico usando técnicas de processamento
    de linguagem natural e análise de redes. As palavras mais importantes foram identificadas através do
    algoritmo PageRank, que considera tanto a frequência quanto as conexões entre termos.
    <br/><br/>
    A análise temporal identifica o período histórico sobre o qual o texto fala, usando estatísticas
    baseadas nos anos mencionados no documento. O desvio médio absoluto é usado para determinar o
    período principal de foco do texto.
    <br/><br/>
    A rede de palavras mostra as relações semânticas entre os conceitos principais do texto, permitindo
    identificar temas centrais e sua interconexão. Os nós maiores e de cor mais intensa representam os termos mais
    centrais na estrutura conceptual do documento.
    """
    elementos.append(Paragraph(interpretacao, estilo_normal))

    # Gerar o PDF
    try:
        doc.build(elementos)
        return caminho_saida
    except Exception as e:
        raise Exception(f"Erro ao gerar PDF: {str(e)}")

# Classe para processamento em thread separada (ATUALIZADA com análise temporal e stopwords)
class ProcessadorThread(QThread):
    progresso_sinal = pyqtSignal(int)
    resultado_sinal = pyqtSignal(object)
    erro_sinal = pyqtSignal(str)

    def __init__(self, texto, palavra_filtro=None, macro_temas=None, stopwords=None):
        super().__init__()
        self.texto = texto
        self.palavra_filtro = palavra_filtro
        self.macro_temas = macro_temas
        self.stopwords = stopwords # NOVO PARÂMETRO

    def run(self):
        try:
            self.progresso_sinal.emit(10)

            # NOVA: Análise temporal
            analise_temporal = analisar_periodo_temporal(self.texto)
            self.progresso_sinal.emit(30)

            # Passar stopwords para a função de rede
            arestas, tema_resultados = listaParaRedeSubstantivos(
                self.texto,
                self.palavra_filtro,
                self.macro_temas,
                self.stopwords # NOVO PARÂMETRO
            )
            self.progresso_sinal.emit(60)

            G, nos_importantes, pagerank = gerar_grafico_rede(arestas)
            self.progresso_sinal.emit(80)

            # Gerar e salvar a imagem do grafo
            imagem_path = salvar_grafico_rede(G, nos_importantes, pagerank)
            self.progresso_sinal.emit(90)

            resultado = {
                'arestas': arestas,
                'grafo': G,
                'nos_importantes': nos_importantes,
                'pagerank': pagerank,
                'tema_resultados': tema_resultados,
                'imagem_path': imagem_path,
                'analise_temporal': analise_temporal
            }

            self.progresso_sinal.emit(100)
            self.resultado_sinal.emit(resultado)

        except Exception as e:
            import traceback
            traceback_info = traceback.format_exc()
            self.erro_sinal.emit(f"Erro no processamento: {str(e)}\n\n{traceback_info}")


# Interface gráfica
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#F5F5F5') # Fundo levemente cinza
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def limpar(self):
        self.axes.clear()
        self.draw()

    def carregar_imagem(self, imagem_path):
        # Limpar o eixo
        self.axes.clear()

        # Carregar e exibir a imagem
        img = plt.imread(imagem_path)
        self.axes.imshow(img)
        self.axes.set_axis_off()

        # Atualizar o canvas
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Oxossi Text Analyzer") # Novo título mais engajante
        self.setGeometry(100, 100, 1400, 900) # Janela maior

        # Definir uma folha de estilo customizada para um visual mais moderno
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ECEFF1; /* Azul Cinzento Claro */
            }
            QTabWidget::pane { /* The tab widget frame */
                border: 1px solid #CFD8DC;
                background-color: #FFFFFF;
            }
            QTabWidget::tab-bar {
                left: 5px; /* move to the right by 5px */
            }
            QTabBar::tab {
                background: #CFD8DC; /* Azul Cinzento */
                border: 1px solid #B0BEC5;
                border-bottom-color: #B0BEC5; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
                padding: 8px 15px;
                color: #2C3E50;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #FFFFFF; /* Branco */
                border-color: #B0BEC5;
                border-bottom-color: #FFFFFF; /* same as pane color */
            }
            QTabBar::tab:hover {
                background: #DDE2E6; /* Cinza mais claro ao passar o mouse */
            }

            QGroupBox {
                font-weight: bold;
                color: #2C3E50; /* Azul Escuro */
                border: 2px solid #B0BEC5; /* Borda Azul Suave */
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #FFFFFF; /* Fundo branco para os grupos */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #B0BEC5;
                border-radius: 3px;
                color: white;
            }
            QPushButton {
                background-color: #3F51B5; /* Azul Índigo */
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #5C6BC0; /* Índigo mais claro */
            }
            QPushButton#exportPdfButton { /* ID específico para o botão de PDF */
                background-color: #4CAF50; /* Verde */
            }
            QPushButton#exportPdfButton:hover {
                background-color: #66BB6A; /* Verde mais claro */
            }
            QPushButton[text="X"] { /* Estilo para o botão de exclusão de tema */
                background-color: #E74C3C; /* Vermelho */
                border-radius: 12px;
                padding: 0;
                min-width: 24px;
                max-width: 24px;
                min-height: 24px;
                max-height: 24px;
            }
            QPushButton[text="X"]:hover {
                background-color: #C0392B; /* Vermelho mais escuro */
            }
            QLineEdit, QTextEdit {
                border: 1px solid #CFD8DC; /* Borda leve */
                border-radius: 4px;
                padding: 5px;
                background-color: #FAFAFA; /* Fundo quase branco */
                color: #37474F; /* Texto escuro */
            }
            QTextEdit {
                background-color: #E0E0E0; /* Levemente mais escuro para texto de resultado */
                font-family: 'Consolas', 'Courier New', monospace; /* Monospace para resultados */
                font-size: 9pt;
            }
            QLabel {
                color: #37474F; /* Texto escuro para labels */
                font-weight: 500;
            }
            QProgressBar {
                border: 1px solid #90CAF9;
                border-radius: 5px;
                text-align: center;
                background-color: #E3F2FD;
                color: #2196F3;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 5px;
            }
            QCheckBox {
                color: #37474F;
            }
            QSplitter::handle {
                background-color: #B0BEC5;
                width: 8px; /* Mais largo para facilitar o clique */
            }
        """)

        # Tentar carregar um ícone
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oxossi_icon.png") # Novo nome de ícone
            if not os.path.exists(icon_path):
                # Criar um ícone simples se não existir
                fig, ax = plt.subplots(figsize=(1, 1), dpi=64)
                ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='#3F51B5', alpha=0.9)) # Círculo azul
                ax.text(0.5, 0.5, '🌿', fontsize=30, ha='center', va='center', color='white') # Emoji de folha
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.axis('off')
                plt.savefig(icon_path, dpi=64, format='png', transparent=True)
                plt.close(fig)

            self.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            print(f"Erro ao carregar/criar ícone: {e}")

        # Variáveis para armazenar dados
        self.texto_atual = ""
        self.resultado_atual = None
        self.imagem_atual_path = None
        self.arquivo_origem = None  # para guardar o caminho do arquivo original
        self.macro_temas = {
            "História Social": ["sociedade", "escravo", "família", "mulher", "homem", "criança", "matrimônio"],
            "História Cultural": ["cultura", "arte", "música", "literatura", "religião", "crença", "tradição", "costume"],
            "História Econômica": ["economia", "comércio", "mercado", "produção", "exportação", "importação", "fazenda", "engenho"],
            "História Política": ["política", "governo", "estado", "poder", "rei", "imperador", "presidente", "república", "império"]
        }

        # Stopwords padrão (baseadas na função limpezaM)
        self.stopwords = [
            "não", "sim", "em", "para", "mais", "muito", "quando", "forma", "a", "o",
            "um", "uma", "de", "em", "para", "por", "com", "talvez", "ainda", "dentro",
            "muito", "artigo", "revista", "pesquisa", "do", "pouco", "antes", "depois",
            "fora", "assim", "também", "que", "este", "esse", "isso", "mais", "de", "e",
            "da", "do", "das", "dos", "da", "como", "da", "entretanto", "também",
            "todavia", "porque", "assim", "logo", "portanto", "essa", "tem", "seguinte",
            "último", "simples", "alguma", "neste", "nesta", "nesse", "nessa", "história",
            "historiadores", "brasil", "onde", "até", "com", "contra", "desde", "entre",
            "para", "por", "sem", "sobre", "pelo", "pela", "pelas", "no", "na", "dias",
            "meses", "anos", "estudos", "revista", "parte", "foi", "ser", "ter", "estar",
            "seu", "sua", "seus", "suas", "já", "só", "pode", "podem", "ou", "mas", "se",
            "são", "foram", "bem", "mesmo", "após", "durante", "apenas", "cada", "todo",
            "toda", "todos", "todas", "outro", "outra", "outros", "outras"
        ]

        # Criar a interface
        self.setup_ui()

    def setup_ui(self):
        # Layout principal
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Criar tabs
        self.tabs = QTabWidget()
        self.tab_rede = QWidget()
        self.tab_temas = QWidget()
        self.tab_stopwords = QWidget() # NOVA ABA

        self.tabs.addTab(self.tab_rede, "Análise de Rede e Temporal") # Título da tab ajustado
        self.tabs.addTab(self.tab_temas, "Análise de Macro-temas")
        self.tabs.addTab(self.tab_stopwords, "Configurar Stopwords") # NOVA ABA

        # Configurar tab de rede de palavras
        self.setup_tab_rede()

        # Configurar tab de macro-temas
        self.setup_tab_temas()

        # Configurar tab de stopwords
        self.setup_tab_stopwords()

        # Adicionar tabs ao layout principal
        main_layout.addWidget(self.tabs)

        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.setCentralWidget(central_widget)

    def setup_tab_rede(self):
        layout = QVBoxLayout()

        # Área para selecionar e carregar arquivo - REDUZIDA e transformada em layout horizontal
        file_layout = QHBoxLayout()

        file_label = QLabel("Arquivo do Documento:")
        file_label.setFont(QFont("Arial", 10, QFont.Bold))
        file_layout.addWidget(file_label)

        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("Nenhum arquivo selecionado...")
        file_layout.addWidget(self.file_path, 1)  # Proporção 1 para expandir

        browse_button = QPushButton("Procurar Arquivo")
        browse_button.clicked.connect(self.browse_file)
        browse_button.setMaximumWidth(150)  # Limitar largura do botão
        file_layout.addWidget(browse_button)

        # Adicionar layout de arquivo diretamente, sem GroupBox
        layout.addLayout(file_layout)

        # Usar um splitter horizontal
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8) # Mais largo para arrastar

        # Área esquerda: controles + gráfico
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_frame.setFrameShadow(QFrame.Raised)
        left_layout = QVBoxLayout(left_frame)

        # Controles de filtro e processamento (acima do gráfico)
        controls_group = QGroupBox("Controles de Análise")
        controls_layout = QVBoxLayout()

        # Filtro de palavras
        filter_layout = QHBoxLayout()
        self.filtro_checkbox = QCheckBox("Aplicar filtro por palavra:")
        self.filtro_checkbox.setFont(QFont("Arial", 10))
        filter_layout.addWidget(self.filtro_checkbox)

        self.filtro_input = QLineEdit()
        self.filtro_input.setEnabled(False)
        self.filtro_input.setPlaceholderText("Ex: 'escravidão', 'política'...")
        self.filtro_checkbox.toggled.connect(self.filtro_input.setEnabled)
        filter_layout.addWidget(self.filtro_input)

        controls_layout.addLayout(filter_layout)

        # Botão de processamento
        processar_button = QPushButton("Iniciar Análise de Rede e Temporal")
        processar_button.clicked.connect(self.processar_rede)
        processar_button.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; background-color: #3F51B5; color: white; border-radius: 8px; font-size: 11pt; } QPushButton:hover { background-color: #5C6BC0; }")
        controls_layout.addWidget(processar_button)

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        # Canvas para o gráfico (abaixo dos controles)
        graph_group = QGroupBox("Visualização da Rede de Palavras")
        graph_layout = QVBoxLayout()

        self.canvas = MatplotlibCanvas(self, width=6, height=5)
        graph_layout.addWidget(self.canvas)

        # Botões para a imagem em layout horizontal
        image_buttons_layout = QHBoxLayout()

        # Botão para abrir a imagem externamente
        open_image_button = QPushButton("Abrir Imagem")
        open_image_button.clicked.connect(self.abrir_imagem_externamente)
        image_buttons_layout.addWidget(open_image_button)

        # Botão para exportar a imagem
        export_button = QPushButton("Exportar Imagem")
        export_button.clicked.connect(self.exportar_imagem)
        image_buttons_layout.addWidget(export_button)

        # Botão para exportar PDF
        export_pdf_button = QPushButton("Gerar Relatório PDF")
        export_pdf_button.setObjectName("exportPdfButton") # Definir object name para estilo
        export_pdf_button.clicked.connect(self.exportar_pdf)
        image_buttons_layout.addWidget(export_pdf_button)

        graph_layout.addLayout(image_buttons_layout)
        graph_group.setLayout(graph_layout)
        left_layout.addWidget(graph_group)

        splitter.addWidget(left_frame)

        # Área direita: apenas resultados
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_frame.setFrameShadow(QFrame.Raised)
        right_layout = QVBoxLayout(right_frame)

        # Resultados
        result_group = QGroupBox("Resultados Detalhados da Análise")
        result_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 9))  # Monospace para resultados
        self.result_text.setStyleSheet("background-color: #F8F8F8; color: #333333; border: 1px solid #E0E0E0; border-radius: 5px;")
        result_layout.addWidget(self.result_text)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        splitter.addWidget(right_frame)

        # Definir proporções iniciais do splitter (55% esquerda, 45% direita)
        splitter.setSizes([750, 650])

        layout.addWidget(splitter)

        self.tab_rede.setLayout(layout)

    def setup_tab_temas(self):
        layout = QVBoxLayout()

        # Área para editar macro-temas
        themes_group = QGroupBox("Configurar Macro-temas Personalizados")
        themes_layout = QVBoxLayout()

        # Área de rolagem para os temas
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget()
        self.themes_inner_layout = QVBoxLayout(scroll_content_widget) # Layout interno para temas
        scroll_area.setWidget(scroll_content_widget)
        themes_layout.addWidget(scroll_area)

        self.theme_editors = {} # Dicionário para armazenar QLineEdits dos temas
        self.carregar_temas_na_interface() # Carregar temas existentes

        # Botão para adicionar novo tema
        add_button = QPushButton("Adicionar Novo Tema")
        add_button.clicked.connect(self.adicionar_tema)
        themes_layout.addWidget(add_button)

        themes_group.setLayout(themes_layout)
        layout.addWidget(themes_group)

        # Área para selecionar arquivo - unificada com a aba de rede, mas mostrando aqui também
        file_layout = QHBoxLayout()

        file_label = QLabel("Arquivo do Documento:")
        file_label.setFont(QFont("Arial", 10, QFont.Bold))
        file_layout.addWidget(file_label)

        self.file_path_temas = QLineEdit() # QLineEdit separado para refletir a aba atual
        self.file_path_temas.setReadOnly(True)
        self.file_path_temas.setPlaceholderText("Nenhum arquivo selecionado...")
        file_layout.addWidget(self.file_path_temas, 1)  # Proporção 1 para expandir

        browse_button = QPushButton("Procurar Arquivo")
        # Usar lambda para passar um argumento para browse_file indicando que veio da aba de temas
        browse_button.clicked.connect(lambda: self.browse_file(is_tema_tab=True))
        browse_button.setMaximumWidth(150)  # Limitar largura do botão
        file_layout.addWidget(browse_button)

        layout.addLayout(file_layout)

        # Linha horizontal de botões para economizar espaço
        buttons_layout = QHBoxLayout()

        # Botão de processamento
        processar_button = QPushButton("Iniciar Análise de Macro-temas")
        processar_button.clicked.connect(self.processar_temas)
        processar_button.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; background-color: #3F51B5; color: white; border-radius: 8px; font-size: 11pt; } QPushButton:hover { background-color: #5C6BC0; }")
        buttons_layout.addWidget(processar_button)

        # Botão para exportar PDF na aba de temas
        export_pdf_temas_button = QPushButton("Gerar Relatório PDF")
        export_pdf_temas_button.setObjectName("exportPdfButton") # Definir object name para estilo
        export_pdf_temas_button.clicked.connect(self.exportar_pdf)
        buttons_layout.addWidget(export_pdf_temas_button)

        layout.addLayout(buttons_layout)

        # Área de resultados
        results_group = QGroupBox("Resultados Detalhados da Análise")
        results_layout = QVBoxLayout()

        self.results_temas = QTextEdit()
        self.results_temas.setReadOnly(True)
        self.results_temas.setFont(QFont("Consolas", 9))
        self.results_temas.setStyleSheet("background-color: #F8F8F8; color: #333333; border: 1px solid #E0E0E0; border-radius: 5px;")
        results_layout.addWidget(self.results_temas)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Adicionar um 'stretch' para que o conteúdo fique no topo
        layout.addStretch(1)

        self.tab_temas.setLayout(layout)

    def setup_tab_stopwords(self):
        layout = QVBoxLayout()

        # Área para editar stopwords
        stopwords_group = QGroupBox("Configurar Stopwords (Palavras a Serem Removidas)")
        stopwords_layout = QVBoxLayout()

        # Descrição
        descricao = QLabel("""
As stopwords são palavras comuns que geralmente não contribuem para a análise semântica
(como artigos, preposições, etc.). Você pode personalizar esta lista conforme sua análise.
        """)
        descricao.setWordWrap(True)
        descricao.setStyleSheet("QLabel { color: #555555; margin: 10px; }")
        stopwords_layout.addWidget(descricao)

        # Área de edição das stopwords
        self.stopwords_text = QTextEdit()
        self.stopwords_text.setFont(QFont("Consolas", 10))
        self.stopwords_text.setStyleSheet("background-color: #FAFAFA; border: 1px solid #CFD8DC; border-radius: 4px;")
        self.stopwords_text.setMaximumHeight(300)
        stopwords_layout.addWidget(self.stopwords_text)

        # Botões de ação em layout horizontal
        buttons_layout = QHBoxLayout()

        # Botão para restaurar stopwords padrão
        restore_button = QPushButton("Restaurar Padrão")
        restore_button.clicked.connect(self.restaurar_stopwords_padrao)
        restore_button.setStyleSheet("QPushButton { background-color: #FF9800; } QPushButton:hover { background-color: #FFB74D; }")
        buttons_layout.addWidget(restore_button)

        # Botão para salvar stopwords
        save_button = QPushButton("Salvar Configuração")
        save_button.clicked.connect(self.salvar_stopwords)
        save_button.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #66BB6A; }")
        buttons_layout.addWidget(save_button)

        # Botão para adicionar palavras comuns
        add_common_button = QPushButton("+ Adicionar Palavras Comuns")
        add_common_button.clicked.connect(self.adicionar_palavras_comuns)
        buttons_layout.addWidget(add_common_button)

        stopwords_layout.addLayout(buttons_layout)

        # Área de estatísticas
        stats_group = QGroupBox("Estatísticas das Stopwords")
        stats_layout = QVBoxLayout()

        self.stopwords_stats = QLabel()
        self.stopwords_stats.setFont(QFont("Arial", 9))
        self.stopwords_stats.setStyleSheet("QLabel { color: #333333; padding: 10px; }")
        stats_layout.addWidget(self.stopwords_stats)

        stats_group.setLayout(stats_layout)
        stopwords_layout.addWidget(stats_group)

        stopwords_group.setLayout(stopwords_layout)
        layout.addWidget(stopwords_group)

        # Carregar stopwords iniciais
        self.carregar_stopwords_na_interface()

        self.tab_stopwords.setLayout(layout)

    def carregar_temas_na_interface(self):
        # Limpar os editores de tema existentes
        for i in reversed(range(self.themes_inner_layout.count())):
            widget_to_remove = self.themes_inner_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
            else:
                layout_to_remove = self.themes_inner_layout.itemAt(i).layout()
                if layout_to_remove:
                    self.clear_layout(layout_to_remove)
                    self.themes_inner_layout.removeItem(layout_to_remove)

        self.theme_editors = {} # Resetar o dicionário

        for tema, palavras in self.macro_temas.items():
            tema_layout = QHBoxLayout()

            tema_label = QLabel(tema + ":")
            tema_label.setMinimumWidth(120)
            tema_layout.addWidget(tema_label)

            tema_edit = QLineEdit(", ".join(palavras))
            tema_layout.addWidget(tema_edit)

            # Adicionar um botão de exclusão para temas personalizados
            delete_button = QPushButton("X")
            delete_button.setFixedSize(QSize(24, 24))
            delete_button.setStyleSheet("QPushButton[text=\"X\"] { background-color: #E74C3C; font-weight: bold; border-radius: 12px; padding: 0; min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; } QPushButton[text=\"X\"]:hover { background-color: #C0392B; }")
            delete_button.clicked.connect(lambda _, t=tema: self.remover_tema(t))
            if tema in ["História Social", "História Cultural", "História Econômica", "História Política"]:
                delete_button.setVisible(False) # Esconder para temas padrão
            tema_layout.addWidget(delete_button)

            self.theme_editors[tema] = tema_edit
            self.themes_inner_layout.addLayout(tema_layout)

        self.themes_inner_layout.addStretch(1) # Empurrar conteúdo para o topo

    def clear_layout(self, layout):
        # Função auxiliar para limpar layouts aninhados
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def adicionar_tema(self):
        # Diálogo para adicionar novo tema
        tema_nome, ok = QInputDialog.getText(self, "Novo Macro-tema", "Digite o nome do novo macro-tema:")

        if ok and tema_nome and tema_nome not in self.macro_temas:
            self.macro_temas[tema_nome] = [] # Adicionar ao dicionário
            self.carregar_temas_na_interface() # Recarregar a seção de temas
        elif ok and tema_nome and tema_nome in self.macro_temas:
            QMessageBox.warning(self, "Aviso", "Este macro-tema já existe.")

    def remover_tema(self, tema_a_remover):
        if tema_a_remover in self.macro_temas:
            reply = QMessageBox.question(self, 'Confirmar Exclusão',
                                         f"Tem certeza que deseja remover o macro-tema '{tema_a_remover}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.macro_temas[tema_a_remover]
                self.carregar_temas_na_interface() # Recarregar a seção de temas

    def carregar_stopwords_na_interface(self):
        """Carrega as stopwords na interface de edição"""
        # Remover duplicatas e ordenar
        stopwords_unicas = sorted(list(set(self.stopwords)))

        # Organizar em colunas para melhor visualização (5 palavras por linha)
        linhas = []
        for i in range(0, len(stopwords_unicas), 5):
            linha = ", ".join(stopwords_unicas[i:i+5])
            linhas.append(linha)

        texto_stopwords = ",\n".join(linhas)
        self.stopwords_text.setText(texto_stopwords)

        # Atualizar estatísticas
        self.atualizar_estatisticas_stopwords()

    def salvar_stopwords(self):
        """Salva as stopwords editadas"""
        texto = self.stopwords_text.toPlainText()

        # Processar texto: dividir por vírgulas e quebras de linha
        palavras = []
        for linha in texto.split('\n'):
            for palavra in linha.split(','):
                palavra = palavra.strip().lower()
                if palavra and len(palavra) > 1: # Filtrar palavras muito curtas
                    palavras.append(palavra)

        # Remover duplicatas e atualizar
        self.stopwords = sorted(list(set(palavras)))

        # Recarregar na interface para mostrar a versão limpa
        self.carregar_stopwords_na_interface()

        QMessageBox.information(self, "Sucesso", f"Stopwords salvas com sucesso!\nTotal: {len(self.stopwords)} palavras únicas.")

    def restaurar_stopwords_padrao(self):
        """Restaura as stopwords para o padrão original"""
        reply = QMessageBox.question(self, 'Confirmar Restauração',
                                     "Tem certeza que deseja restaurar as stopwords padrão?\nIsso apagará todas as personalizações.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Stopwords padrão originais
            self.stopwords = [
                "não", "sim", "em", "para", "mais", "muito", "quando", "forma", "a", "o",
                "um", "uma", "de", "em", "para", "por", "com", "talvez", "ainda", "dentro",
                "muito", "artigo", "revista", "pesquisa", "do", "pouco", "antes", "depois",
                "fora", "assim", "também", "que", "este", "esse", "isso", "mais", "de", "e",
                "da", "do", "das", "dos", "da", "como", "da", "entretanto", "também",
                "todavia", "porque", "assim", "logo", "portanto", "essa", "tem", "seguinte",
                "último", "simples", "alguma", "neste", "nesta", "nesse", "nessa", "história",
                "historiadores", "brasil", "onde", "até", "com", "contra", "desde", "entre",
                "para", "por", "sem", "sobre", "pelo", "pela", "pelas", "no", "na", "dias",
                "meses", "anos", "estudos", "revista", "parte", "foi", "ser", "ter", "estar",
                "seu", "sua", "seus", "suas", "já", "só", "pode", "podem", "ou", "mas", "se",
                "são", "foram", "bem", "mesmo", "após", "durante", "apenas", "cada", "todo",
                "toda", "todos", "todas", "outro", "outra", "outros", "outras"
            ]
            self.carregar_stopwords_na_interface()

    def adicionar_palavras_comuns(self):
        """Abre diálogo para adicionar palavras comuns rapidamente"""
        dialog_text = """Digite palavras adicionais separadas por vírgula:
(Ex: governo, estado, poder, cidade, região)"""

        palavras_novas, ok = QInputDialog.getMultiLineText(self, "Adicionar Stopwords", dialog_text)

        if ok and palavras_novas.strip():
            # Processar novas palavras
            novas = []
            for linha in palavras_novas.split('\n'):
                for palavra in linha.split(','):
                    palavra = palavra.strip().lower()
                    if palavra and len(palavra) > 1:
                        novas.append(palavra)

            # Adicionar às stopwords existentes
            self.stopwords.extend(novas)
            self.stopwords = sorted(list(set(self.stopwords))) # Remover duplicatas

            # Recarregar interface
            self.carregar_stopwords_na_interface()

            QMessageBox.information(self, "Sucesso", f"{len(novas)} palavras adicionadas com sucesso!")

    def atualizar_estatisticas_stopwords(self):
        """Atualiza as estatísticas das stopwords"""
        total = len(self.stopwords)

        # Calcular estatísticas simples
        tamanhos = [len(word) for word in self.stopwords]
        tamanho_medio = sum(tamanhos) / len(tamanhos) if tamanhos else 0

        # Contar por tamanho
        por_tamanho = {}
        for tamanho in tamanhos:
            por_tamanho[tamanho] = por_tamanho.get(tamanho, 0) + 1

        stats_text = f"""📊 Estatísticas das Stopwords:
• Total de palavras: {total}
• Tamanho médio: {tamanho_medio:.1f} caracteres
• Palavras mais curtas: {min(tamanhos) if tamanhos else 0} caracteres
• Palavras mais longas: {max(tamanhos) if tamanhos else 0} caracteres

📈 Distribuição por tamanho:"""

        for tamanho in sorted(por_tamanho.keys()):
            stats_text += f"\n• {tamanho} caracteres: {por_tamanho[tamanho]} palavras"

        self.stopwords_stats.setText(stats_text)

    def browse_file(self, is_tema_tab=False):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Arquivo", "", "Arquivos PDF (*.pdf);;Arquivos de Texto (*.txt)", options=options
        )

        if file_path:
            if is_tema_tab:
                self.file_path_temas.setText(file_path)
            else:
                self.file_path.setText(file_path)

            # Salvar o caminho do arquivo original
            self.arquivo_origem = file_path

            # Carregar o conteúdo do arquivo
            self.carregar_arquivo(file_path)

    def carregar_arquivo(self, file_path):
        try:
            if file_path.lower().endswith('.pdf'):
                texto = self.extrair_texto_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    texto = file.read()

            self.texto_atual = texto

            # Mostrar estatísticas básicas em ambas as abas
            info = f"Arquivo carregado com sucesso:\n  • Nome: {os.path.basename(file_path)}\n"
            info += f"  • Tamanho: {len(texto):,} caracteres\n"
            info += f"  • Palavras (aprox.): {len(texto.split()):,}\n\n"
            info += "Pronto para iniciar a análise!\n\n"
            info += "Clique no botão 'Iniciar Análise...' na aba atual para gerar os resultados."

            self.result_text.setText(info)
            self.results_temas.setText(info)
            self.canvas.limpar() # Limpar qualquer gráfico anterior

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar o arquivo: {str(e)}")

    def extrair_texto_pdf(self, pdf_path):
        texto = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text_on_page = page.extract_text()
                    if text_on_page: # Adicionar apenas se houver texto
                        texto += text_on_page + "\n"
            if not texto.strip(): # Verificar se o texto extraído está vazio ou apenas espaços em branco
                 raise ValueError("Nenhum texto extraído do PDF. O PDF pode ser uma imagem escaneada ou estar criptografado.")
            return texto
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao processar PDF: {str(e)}")
            return ""

    def atualizar_macro_temas(self):
        # Atualizar o dicionário de macro-temas com base nos inputs
        for tema, editor in self.theme_editors.items():
            palavras_texto = editor.text().strip()
            if palavras_texto:
                self.macro_temas[tema] = [p.strip() for p in palavras_texto.split(',') if p.strip()] # Filtrar strings vazias
            else:
                self.macro_temas[tema] = []

    def processar_rede(self):
        if not self.texto_atual:
            QMessageBox.warning(self, "Aviso", "Nenhum texto carregado para processar. Por favor, selecione um arquivo.")
            return

        # Verificar se o filtro está ativado
        palavra_filtro = None
        if self.filtro_checkbox.isChecked():
            palavra_filtro = self.filtro_input.text().strip()
            if not palavra_filtro:
                QMessageBox.warning(self, "Aviso", "O filtro por palavra está ativado, mas nenhuma palavra foi especificada.")
                return

        # Atualizar macro-temas
        self.atualizar_macro_temas()

        # Mostrar barra de progresso
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.result_text.setText("Iniciando análise de rede e temporal... Isso pode levar alguns segundos.")
        self.canvas.limpar() # Limpar gráfico anterior

        # Iniciar processamento em thread separada (COM STOPWORDS)
        self.thread = ProcessadorThread(self.texto_atual, palavra_filtro, self.macro_temas, self.stopwords)
        self.thread.progresso_sinal.connect(self.atualizar_progresso)
        self.thread.resultado_sinal.connect(self.mostrar_resultado_rede)
        self.thread.erro_sinal.connect(self.mostrar_erro)
        self.thread.start()

    def processar_temas(self):
        if not self.texto_atual:
            QMessageBox.warning(self, "Aviso", "Nenhum texto carregado para processar. Por favor, selecione um arquivo.")
            return

        # Atualizar macro-temas
        self.atualizar_macro_temas()

        # Mostrar barra de progresso
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.results_temas.setText("Iniciando análise de macro-temas e temporal... Isso pode levar alguns segundos.")

        # Iniciar processamento em thread separada (COM STOPWORDS)
        self.thread = ProcessadorThread(self.texto_atual, None, self.macro_temas, self.stopwords)
        self.thread.progresso_sinal.connect(self.atualizar_progresso)
        self.thread.resultado_sinal.connect(self.mostrar_resultado_temas)
        self.thread.erro_sinal.connect(self.mostrar_erro)
        self.thread.start()

    def atualizar_progresso(self, valor):
        self.progress_bar.setValue(valor)

    def mostrar_erro(self, mensagem):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erro de Processamento", mensagem)

    def abrir_imagem_externamente(self):
        """Abre a imagem atual no visualizador padrão do sistema"""
        if self.imagem_atual_path and os.path.exists(self.imagem_atual_path):
            # Abrir com o aplicativo padrão do sistema
            try:
                if sys.platform == 'win32':
                    os.startfile(self.imagem_atual_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{self.imagem_atual_path}"')
                else:  # Linux e outros
                    os.system(f'xdg-open "{self.imagem_atual_path}"')
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Não foi possível abrir a imagem: {str(e)}")
        else:
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem de grafo disponível para abrir. Realize uma análise na aba 'Análise de Rede e Temporal' primeiro.")

    def exportar_imagem(self):
        """Exporta a imagem atual para um local escolhido pelo usuário"""
        if not self.imagem_atual_path or not os.path.exists(self.imagem_atual_path):
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem de grafo disponível para exportar. Realize uma análise na aba 'Análise de Rede e Temporal' primeiro.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Imagem da Rede de Palavras", os.path.expanduser("~/Rede_Palavras.png"),
            "Imagens PNG (*.png);;Todos os Arquivos (*)", options=options
        )

        if file_path:
            # Garantir que o arquivo tenha extensão .png
            if not file_path.lower().endswith('.png'):
                file_path += '.png'

            # Copiar a imagem para o destino escolhido
            import shutil
            try:
                shutil.copy2(self.imagem_atual_path, file_path)
                QMessageBox.information(self, "Sucesso", f"Imagem salva com sucesso em:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar a imagem: {str(e)}")

    # FUNÇÃO: Exportar PDF (unificada para ambas as abas)
    def exportar_pdf(self):
        """Exporta um relatório completo em PDF"""
        if not self.resultado_atual:
            QMessageBox.warning(self, "Aviso", "Nenhum resultado disponível para exportar. Por favor, processe o texto primeiro.")
            return

        # Diálogo para escolher onde salvar o PDF
        options = QFileDialog.Options()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_sugerido = f"relatorio_analise_texto_{timestamp}.pdf"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Relatório PDF",
            os.path.join(os.path.expanduser("~"), nome_sugerido),
            "Arquivos PDF (*.pdf);;Todos os Arquivos (*)", options=options
        )

        if file_path:
            # Garantir que o arquivo tenha extensão .pdf
            if not file_path.lower().endswith('.pdf'):
                file_path += '.pdf'

            try:
                # Obter palavra filtro se estiver ativa (apenas se for a aba de rede)
                palavra_filtro = None
                if self.tabs.currentWidget() == self.tab_rede and self.filtro_checkbox.isChecked():
                    palavra_filtro = self.filtro_input.text().strip()

                # Mostrar barra de progresso
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(50)

                # Gerar o PDF
                caminho_pdf = gerar_relatorio_pdf(
                    self.resultado_atual,
                    self.arquivo_origem,
                    palavra_filtro,
                    file_path
                )

                self.progress_bar.setValue(100)
                self.progress_bar.setVisible(False)

                # Mostrar mensagem de sucesso
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("PDF Exportado")
                msg.setText("Relatório PDF gerado com sucesso!")
                msg.setInformativeText(f"Arquivo salvo em:\n{caminho_pdf}")

                # Adicionar botão para abrir o PDF
                abrir_button = msg.addButton("Abrir PDF", QMessageBox.ActionRole)
                msg.addButton(QMessageBox.Ok)

                result = msg.exec_()

                # Se o usuário clicar em "Abrir PDF"
                if msg.clickedButton() == abrir_button:
                    if sys.platform == 'win32':
                        os.startfile(caminho_pdf)
                    elif sys.platform == 'darwin':  # macOS
                        os.system(f'open "{caminho_pdf}"')
                    else:  # Linux e outros
                        os.system(f'xdg-open "{caminho_pdf}"')

            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Erro", f"Erro ao gerar PDF:\n{str(e)}")

    # FUNÇÃO ATUALIZADA: Mostrar resultado da rede com análise temporal
    def mostrar_resultado_rede(self, resultado):
        self.resultado_atual = resultado
        self.progress_bar.setVisible(False)

        # Salvar o caminho da imagem
        self.imagem_atual_path = resultado['imagem_path']

        # Começar com a análise temporal
        texto_resultado = ""

        # SEÇÃO 1: Análise Temporal
        if 'analise_temporal' in resultado and resultado['analise_temporal']['periodo_encontrado']:
            temporal = resultado['analise_temporal']
            texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
            texto_resultado += "║                    ANÁLISE DO PERÍODO TEMPORAL                ║\n"
            texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"

            texto_resultado += f"📅 Anos encontrados: {temporal['total_anos']}\n"
            texto_resultado += f"📊 Média dos anos: {temporal['media_anos']}\n"
            texto_resultado += f"📈 Desvio padrão: {temporal['desvio_padrao']}\n"
            texto_resultado += f"📏 Desvio médio absoluto (DMA): {temporal['desvio_medio_absoluto']}\n\n"

            if temporal['periodo_principal']:
                texto_resultado += f"🎯 PERÍODO PRINCIPAL: {temporal['periodo_principal'][0]} - {temporal['periodo_principal'][1]}\n"

            if temporal['periodo_alargado']:
                texto_resultado += f"📅 Período total: {temporal['periodo_alargado'][0]} - {temporal['periodo_alargado'][1]}\n\n"

            if temporal.get('seculos_mencionados'):
                texto_resultado += f"🏛️ Séculos mencionados: {', '.join(temporal['seculos_mencionados'])}\n\n"

            # Mostrar alguns anos encontrados como exemplo
            if temporal['anos_encontrados']:
                anos_exemplo = temporal['anos_encontrados'][:10]
                texto_resultado += f"📋 Exemplos de anos identificados: {', '.join(map(str, anos_exemplo))}"
                if len(temporal['anos_encontrados']) > 10:
                    texto_resultado += f" (e mais {len(temporal['anos_encontrados']) - 10} anos)"
                texto_resultado += "\n\n"
        elif 'analise_temporal' in resultado and 'erro' in resultado['analise_temporal']:
             texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
             texto_resultado += "║                    ANÁLISE DO PERÍODO TEMPORAL                ║\n"
             texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"
             texto_resultado += f"❌ Erro na análise temporal: {resultado['analise_temporal']['erro']}\n\n"
        else:
             texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
             texto_resultado += "║                    ANÁLISE DO PERÍODO TEMPORAL                ║\n"
             texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"
             texto_resultado += "ℹ️ Nenhum período temporal identificado.\n\n"

        # SEÇÃO 2: Análise de Macro-temas
        if resultado['tema_resultados'] and sum(resultado['tema_resultados'].values()) > 0:
            texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
            texto_resultado += "║                    ANÁLISE DE MACRO-TEMAS                    ║\n"
            texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"

            tema_resultados = resultado['tema_resultados']
            total_ocorrencias = sum(tema_resultados.values())

            temas_ordenados = sorted(tema_resultados.items(), key=lambda x: x[1], reverse=True)

            for i, (tema, contagem) in enumerate(temas_ordenados, 1):
                porcentagem = (contagem / total_ocorrencias) * 100
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📌"
                texto_resultado += f"{emoji} {tema}: {contagem} ocorrências ({porcentagem:.1f}%)\n"

            texto_resultado += "\n"
        else:
            texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
            texto_resultado += "║                    ANÁLISE DE MACRO-TEMAS                    ║\n"
            texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"
            texto_resultado += "ℹ️ Nenhuma ocorrência de macro-temas encontrada ou configurada.\n\n"

        # SEÇÃO 3: Rede de Palavras
        texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
        texto_resultado += "║                    REDE DE PALAVRAS                          ║\n"
        texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"

        nos = resultado['nos_importantes']
        G = resultado['grafo']

        # Estatísticas da rede
        texto_resultado += f"🌐 Número de nós (palavras únicas): {G.number_of_nodes()}\n"
        texto_resultado += f"🔗 Número de conexões (relações): {G.number_of_edges()}\n"
        texto_resultado += f"📊 Densidade da rede: {nx.density(G):.4f}\n\n"

        # Top 10 palavras mais importantes
        texto_resultado += "🏆 TOP 10 PALAVRAS MAIS IMPORTANTES (por PageRank):\n\n"

        for i, no in enumerate(nos[:10], 1):
            pagerank = resultado['pagerank'].get(no, 0)
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            texto_resultado += f"{emoji} {no} (Importância: {pagerank:.4f})\n"

        texto_resultado += "\n"

        # SEÇÃO 4: Informações de Exportação
        texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
        texto_resultado += "║                    OPÇÕES DE EXPORTAÇÃO                      ║\n"
        texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"

        texto_resultado += f"🖼️ Imagem do grafo temporariamente salva em:\n   {self.imagem_atual_path}\n\n"
        texto_resultado += "💡 Utilize os botões abaixo para:\n"
        texto_resultado += "   • 'Abrir Imagem' - visualizar o gráfico em tamanho completo\n"
        texto_resultado += "   • 'Exportar Imagem' - salvar a imagem em local personalizado\n"
        texto_resultado += "   • 'Gerar Relatório PDF' - criar um relatório completo em PDF\n\n"

        # Adicionar informação sobre filtros aplicados
        if hasattr(self, 'filtro_checkbox') and self.filtro_checkbox.isChecked():
            palavra_filtro = self.filtro_input.text().strip()
            if palavra_filtro:
                texto_resultado += f"🔍 Filtro de Palavra Aplicado: '{palavra_filtro}'\n\n"

        texto_resultado += "✅ Análise concluída com sucesso! Resultados prontos para visualização."

        self.result_text.setText(texto_resultado)

        # Carregar a imagem no canvas
        try:
            self.canvas.carregar_imagem(self.imagem_atual_path)
        except Exception as e:
            self.result_text.append(f"\n\n❌ Erro ao exibir a imagem no visualizador: {str(e)}\n")
            self.result_text.append("Por favor, use o botão 'Abrir Imagem' para visualizar externamente.")

    # FUNÇÃO ATUALIZADA: Mostrar resultado dos temas com análise temporal
    def mostrar_resultado_temas(self, resultado):
        self.resultado_atual = resultado  # IMPORTANTE: salvar resultado para PDF
        self.progress_bar.setVisible(False)

        texto_resultado = ""

        # SEÇÃO 1: Análise Temporal também na aba de temas
        if 'analise_temporal' in resultado and resultado['analise_temporal']['periodo_encontrado']:
            temporal = resultado['analise_temporal']
            texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
            texto_resultado += "║                    ANÁLISE DO PERÍODO TEMPORAL                ║\n"
            texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"

            texto_resultado += f"📅 Anos encontrados: {temporal['total_anos']}\n"
            texto_resultado += f"📊 Média dos anos: {temporal['media_anos']}\n"
            texto_resultado += f"📈 Desvio padrão: {temporal['desvio_padrao']}\n"
            texto_resultado += f"📏 Desvio médio absoluto (DMA): {temporal['desvio_medio_absoluto']}\n\n"

            if temporal['periodo_principal']:
                texto_resultado += f"🎯 PERÍODO PRINCIPAL: {temporal['periodo_principal'][0]} - {temporal['periodo_principal'][1]}\n"

            if temporal['periodo_alargado']:
                texto_resultado += f"📅 Período total: {temporal['periodo_alargado'][0]} - {temporal['periodo_alargado'][1]}\n\n"

            if temporal.get('seculos_mencionados'):
                texto_resultado += f"🏛️ Séculos mencionados: {', '.join(temporal['seculos_mencionados'])}\n\n"
        elif 'analise_temporal' in resultado and 'erro' in resultado['analise_temporal']:
             texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
             texto_resultado += "║                    ANÁLISE DO PERÍODO TEMPORAL                ║\n"
             texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"
             texto_resultado += f"❌ Erro na análise temporal: {resultado['analise_temporal']['erro']}\n\n"
        else:
             texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
             texto_resultado += "║                    ANÁLISE DO PERÍODO TEMPORAL                ║\n"
             texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"
             texto_resultado += "ℹ️ Nenhum período temporal identificado.\n\n"

        # SEÇÃO 2: Análise de Macro-temas
        tema_resultados = resultado['tema_resultados']

        if tema_resultados and sum(tema_resultados.values()) > 0:
            texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
            texto_resultado += "║                    ANÁLISE DE MACRO-TEMAS                    ║\n"
            texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"

            # Calcular o total de ocorrências para percentuais
            total_ocorrencias = sum(count for _, count in tema_resultados.items())
            if total_ocorrencias == 0:
                total_ocorrencias = 1  # Evitar divisão por zero

            # Mostrar todos os temas com seus percentuais, ordenados por relevância
            temas_ordenados = sorted(tema_resultados.items(), key=lambda x: x[1], reverse=True)

            for i, (tema, contagem) in enumerate(temas_ordenados, 1):
                porcentagem = (contagem / total_ocorrencias) * 100
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📌"
                texto_resultado += f"{emoji} {tema}: {contagem} ocorrências ({porcentagem:.1f}%)\n"

                # Listar palavras-chave associadas a este tema
                palavras = self.macro_temas.get(tema, [])
                if palavras:
                    texto_resultado += "   Palavras-chave configuradas: " + ", ".join(palavras) + "\n\n"

            # Adicionar estatísticas do texto
            texto_resultado += f"\nEstatísticas do texto analisado:\n"
            texto_resultado += f"Tamanho: {len(self.texto_atual):,} caracteres\n"
            texto_resultado += f"Aproximadamente {len(self.texto_atual.split()):,} palavras\n\n"

            # Mostrar palavras mais importantes se disponíveis
            if 'nos_importantes' in resultado and resultado['nos_importantes']:
                texto_resultado += "🏆 TOP 10 PALAVRAS MAIS IMPORTANTES (da rede de palavras):\n"
                for i, palavra in enumerate(resultado['nos_importantes'][:10], 1):
                    texto_resultado += f"{i}. {palavra}\n"
                texto_resultado += "\n"

            # Se há uma imagem disponível, mencioná-la
            if 'imagem_path' in resultado and resultado['imagem_path']:
                self.imagem_atual_path = resultado['imagem_path']
                texto_resultado += f"🖼️ Imagem do grafo disponível na aba 'Análise de Rede e Temporal'.\n"

            texto_resultado += "\n✅ Análise concluída com sucesso! Resultados prontos para visualização."
            texto_resultado += "\nPara gerar um relatório completo, use o botão 'Gerar Relatório PDF'."

            self.results_temas.setText(texto_resultado)
        else:
            texto_resultado += "╔═══════════════════════════════════════════════════════════════╗\n"
            texto_resultado += "║                    ANÁLISE DE MACRO-TEMAS                    ║\n"
            texto_resultado += "╚═══════════════════════════════════════════════════════════════╝\n\n"
            texto_resultado += "ℹ️ Nenhuma ocorrência de macro-temas encontrada ou configurada.\n"
            texto_resultado += "Verifique a configuração das palavras-chave para cada macro-tema."
            self.results_temas.setText(texto_resultado)


# Script de inicialização com mais tratamento de erros e prevenção contra detecção como malware
if __name__ == "__main__":
    # Verificar se o sistema operacional é suportado
    if sys.platform not in ['win32', 'darwin', 'linux']:
        print(f"Aviso: Sistema operacional {sys.platform} pode não ser totalmente compatível.")

    # Lidar com o aviso do Wayland antes de criar a aplicação
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ["QT_QPA_PLATFORM"] = "wayland"

    # Garantir que diretórios temporários existam e tenham permissões corretas
    try:
        temp_dir = tempfile.gettempdir()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    except Exception as e:
        print(f"Aviso: Não foi possível verificar diretório temporário: {e}")

    # Garantir que matplotlib use um backend não interativo em todas as situações
    try:
        matplotlib.use('Agg')  # Forçar o uso do backend Agg (mais compatível)
    except:
        pass

    # Definir configurações do aplicativo para evitar detecção como malware
    app = QApplication(sys.argv)
    app.setApplicationName("Oxossi Text Analyzer")
    app.setApplicationVersion("1.0.2") # Atualizada para incluir melhorias visuais e de UX
    app.setOrganizationName("HistoriaBR")
    app.setOrganizationDomain("historiabr.edu")

    # Configuração para evitar problemas de thread em ambientes Windows
    if sys.platform == 'win32':
        # Desabilitar verificação de ameaças em tempo real para este processo (não funciona sem elevação)
        # Alternativa: adicionar ao caminho de exclusão do Windows Defender
        pass

    # Verificar se o modelo spaCy está disponível antes de iniciar
    if not processamento:
        from PyQt5.QtWidgets import QMessageBox
        # Usar QApplication.instance() para evitar criar um novo QApplication se um já existir
        app_temp = QApplication.instance() if QApplication.instance() else QApplication([])
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Aviso - Modelo spaCy Ausente")
        msg.setText("O modelo de linguagem 'pt_core_news_sm' do spaCy não foi encontrado.")
        msg.setInformativeText("O aplicativo funcionará com funcionalidade limitada (análise de substantivos pode ser afetada).\n\nPara instalar, feche o aplicativo e execute no terminal:\n\n<b>python -m spacy download pt_core_news_sm</b>")
        msg.exec_()

    # Verificar se ReportLab está disponível
    try:
        import reportlab
    except ImportError:
        from PyQt5.QtWidgets import QMessageBox
        app_temp = QApplication.instance() if QApplication.instance() else QApplication([])
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Aviso - ReportLab Ausente")
        msg.setText("A biblioteca 'ReportLab' não foi encontrada.")
        msg.setInformativeText("A funcionalidade de exportação de relatório PDF não estará disponível.\n\nPara instalar, feche o aplicativo e execute no terminal:\n\n<b>pip install reportlab</b>")
        msg.exec_()

    # Verificar se NumPy está disponível (necessário para análise temporal)
    try:
        import numpy
    except ImportError:
        from PyQt5.QtWidgets import QMessageBox
        app_temp = QApplication.instance() if QApplication.instance() else QApplication([])
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Aviso - NumPy Ausente")
        msg.setText("A biblioteca 'NumPy' não foi encontrada.")
        msg.setInformativeText("A funcionalidade de análise temporal pode estar limitada ou não funcionar.\n\nPara instalar, feche o aplicativo e execute no terminal:\n\n<b>pip install numpy</b>")
        msg.exec_()

    # Iniciar aplicação com tratamento de exceções
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        error_msg = f"Erro fatal ao iniciar a aplicação: {str(e)}\n\nDetalhes:\n{traceback.format_exc()}"
        print(error_msg)

        # Tentar mostrar diálogo de erro se possível
        try:
            from PyQt5.QtWidgets import QMessageBox
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Erro Fatal da Aplicação")
            error_dialog.setText("Ocorreu um erro inesperado e a aplicação será encerrada.")
            error_dialog.setInformativeText("Por favor, tente reiniciar. Se o problema persistir, contate o suporte.")
            error_dialog.setDetailedText(error_msg)
            error_dialog.exec_()
        except:
            # Se falhar, pelo menos imprimimos o erro no console
            pass
