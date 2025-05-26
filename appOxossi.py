import sys
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import PyPDF2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
                           QLabel, QTextEdit, QLineEdit, QCheckBox, QComboBox,
                           QGroupBox, QPlainTextEdit, QProgressBar, QScrollArea,
                           QSplitter, QFrame, QMessageBox, QInputDialog, QSizePolicy,
                           QSpacerItem)
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tempfile
import datetime
from collections import Counter

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
    remover = ["/","[","]", "-", "=", ">","'", '"',",",")","(","_","_"," parte "," forma ","revista","pesquisa","histórica"]
    for item in remover:
        texto = texto.replace(item,"  ")
    texto = texto.replace("século ","século_")
    texto = texto.replace("__","_")
    texto = re.sub(r"\d+","",texto)
    texto = texto.replace(":",". ")
    texto = texto.strip()
    return texto

def limpezaM(texto):
    texto = texto.lower()
    texto = limpeza(texto)
    remover = ["  não ", " sim ", " em ", " para ", "  mais ", " muito ", " quando", " forma ", " a ", " o ", " um ", " uma ", " de ", " em ", " para ", " por ", " com ", " talvez ", " ainda ", " dentro ", " muito ", "  artigo ", " revista ", " pesquisa ", " do ", " pouco ", " antes ", " depois ", " fora ", " assim ", " também", " que ", " este ", " esse ", " isso ", " mais ", " de ", " e ", " da ", " do ", " das ", " dos ", " da ", " como ", " da ", " entretanto ", " também ", " todavia", " porque ", " assim ", " logo ", " portanto ", " essa ", " tem ", " seguinte ", " último ", " simples ", " alguma ", " neste ", "  nesta ", " nesse ", " nessa ", " história ", " historiadores ", " brasil ", " onde ", " até ", " com ", " contra ", " desde ", " entre ", " para ", " por ", " sem ", " sobre ", " pelo ", " pela ", " pelas ", " no ", " na"," dias ", " meses ", " anos ", " estudos "," revista "," parte "]
    for item in remover:
        texto = texto.replace(item,"  ")
    texto = re.sub(r"\d+","",texto)
    texto = texto.strip()
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

def substantivador(texto, exceto):
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

def listaParaRedeSubstantivos(texto, palavra_filtro=None, macro_temas=None):
    if len(texto) < 120000:
        texto = limpeza(texto)
    else:
        texto = limpezaM(texto)
    
    texto = nTokens(texto)
    biToken = biTokenFinder(texto)
    biToken = biToken[0]
    biTokenSemEspaco = biToken.replace("_"," ")
    texto = texto.replace(biTokenSemEspaco, biToken)
    
    linhas = texto.split(". ")
    
    # Aplicar filtro de palavras se especificado
    if palavra_filtro:
        linhas = filtrar_por_palavra(linhas, palavra_filtro)
    
    # Classificar por macro-temas se especificado
    tema_resultados = {}
    if macro_temas:
        tema_resultados = classificar_macro_tema(texto, macro_temas)
    
    mistura = []
    for item in linhas:
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
        node_sizes = [pr.get(node, 0) * 8000 / max_pr for node in G.nodes()]
        
        # Cores dos nós - mais importantes em vermelho
        node_colors = ['red' if node in nos_importantes[:6] else 'skyblue' for node in G.nodes()]
        
        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, ax=ax, 
                              node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Desenhar arestas com baixa opacidade
        nx.draw_networkx_edges(G, pos, ax=ax, 
                              width=0.5, alpha=0.3)
        
        # Adicionar rótulos apenas para os nós mais importantes
        labels = {node: node for node in nos_importantes[:10]}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                              font_size=10, font_weight='bold')
    
    # Remover eixos para uma visualização mais limpa
    ax.set_axis_off()
    
    # Criar nome de arquivo com timestamp para evitar sobrescrever
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"rede_palavras_{timestamp}.png")
    
    # Salvar a figura com dpi maior para melhor qualidade
    fig.tight_layout()
    fig.savefig(filename, format='png', dpi=120, bbox_inches='tight')
    
    return filename

# Classe para processamento em thread separada
class ProcessadorThread(QThread):
    progresso_sinal = pyqtSignal(int)
    resultado_sinal = pyqtSignal(object)
    erro_sinal = pyqtSignal(str)
    
    def __init__(self, texto, palavra_filtro=None, macro_temas=None):
        super().__init__()
        self.texto = texto
        self.palavra_filtro = palavra_filtro
        self.macro_temas = macro_temas
        
    def run(self):
        try:
            self.progresso_sinal.emit(20)
            arestas, tema_resultados = listaParaRedeSubstantivos(
                self.texto, 
                self.palavra_filtro, 
                self.macro_temas
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
                'imagem_path': imagem_path
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
        self.fig = Figure(figsize=(width, height), dpi=dpi)
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
        
        self.setWindowTitle("Processador de Textos Históricos")
        self.setGeometry(100, 100, 1200, 800)
        
        # Tentar carregar um ícone (para ajudar com o problema do malware)
        try:
            # Criar um ícone simples se necessário
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
            if not os.path.exists(icon_path):
                # Criar um ícone básico usando matplotlib (sem PIL)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.6, color='blue', alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.axis('off')
                plt.savefig(icon_path, dpi=64, format='png')
                plt.close(fig)
            
            self.setWindowIcon(QIcon(icon_path))
        except:
            # Se não conseguir, continue sem um ícone
            pass
        
        # Variáveis para armazenar dados
        self.texto_atual = ""
        self.resultado_atual = None
        self.imagem_atual_path = None
        self.macro_temas = {
            "História Social": ["sociedade", "escravo", "família", "mulher", "homem", "criança", "matrimônio"],
            "História Cultural": ["cultura", "arte", "música", "literatura", "religião", "crença", "tradição", "costume"],
            "História Econômica": ["economia", "comércio", "mercado", "produção", "exportação", "importação", "fazenda", "engenho"],
            "História Política": ["política", "governo", "estado", "poder", "rei", "imperador", "presidente", "república", "império"]
        }
        
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
        
        self.tabs.addTab(self.tab_rede, "Rede de Palavras")
        self.tabs.addTab(self.tab_temas, "Análise de Macro-temas")
        
        # Configurar tab de rede de palavras
        self.setup_tab_rede()
        
        # Configurar tab de macro-temas
        self.setup_tab_temas()
        
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
        
        file_label = QLabel("Arquivo:")
        file_layout.addWidget(file_label)
        
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        file_layout.addWidget(self.file_path, 1)  # Proporção 1 para expandir
        
        browse_button = QPushButton("Procurar")
        browse_button.clicked.connect(self.browse_file)
        browse_button.setMaximumWidth(100)  # Limitar largura do botão
        file_layout.addWidget(browse_button)
        
        # Adicionar layout de arquivo diretamente, sem GroupBox
        layout.addLayout(file_layout)
        
        # Usar um splitter horizontal
        splitter = QSplitter(Qt.Horizontal)
        
        # Área esquerda: controles + gráfico
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        
        # Controles de filtro e processamento (acima do gráfico)
        controls_group = QGroupBox("Controles")
        controls_layout = QVBoxLayout()
        
        # Filtro de palavras
        filter_layout = QHBoxLayout()
        self.filtro_checkbox = QCheckBox("Filtrar por palavra:")
        filter_layout.addWidget(self.filtro_checkbox)
        
        self.filtro_input = QLineEdit()
        self.filtro_input.setEnabled(False)
        self.filtro_input.setPlaceholderText("Digite a palavra para filtrar...")
        self.filtro_checkbox.toggled.connect(self.filtro_input.setEnabled)
        filter_layout.addWidget(self.filtro_input)
        
        controls_layout.addLayout(filter_layout)
        
        # Botão de processamento
        processar_button = QPushButton("Processar Texto")
        processar_button.clicked.connect(self.processar_rede)
        processar_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        controls_layout.addWidget(processar_button)
        
        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)
        
        # Canvas para o gráfico (abaixo dos controles)
        graph_group = QGroupBox("Visualização da Rede")
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
        export_button = QPushButton("Exportar")
        export_button.clicked.connect(self.exportar_imagem)
        image_buttons_layout.addWidget(export_button)
        
        graph_layout.addLayout(image_buttons_layout)
        graph_group.setLayout(graph_layout)
        left_layout.addWidget(graph_group)
        
        splitter.addWidget(left_frame)
        
        # Área direita: apenas resultados
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        
        # Resultados
        result_group = QGroupBox("Resultados")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        splitter.addWidget(right_frame)
        
        # Definir proporções iniciais do splitter (60% esquerda, 40% direita)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        self.tab_rede.setLayout(layout)
    
    def setup_tab_temas(self):
        layout = QVBoxLayout()
        
        # Área para editar macro-temas - mantida como está por ser funcionalidade importante
        themes_group = QGroupBox("Configurar Macro-temas")
        themes_layout = QVBoxLayout()
        
        # Layout para cada tema
        self.theme_editors = {}
        
        for tema, palavras in self.macro_temas.items():
            tema_layout = QHBoxLayout()
            
            tema_label = QLabel(tema + ":")
            tema_layout.addWidget(tema_label)
            
            tema_edit = QLineEdit(", ".join(palavras))
            tema_layout.addWidget(tema_edit)
            
            self.theme_editors[tema] = tema_edit
            themes_layout.addLayout(tema_layout)
        
        # Botão para adicionar novo tema
        add_button = QPushButton("Adicionar Novo Tema")
        add_button.clicked.connect(self.adicionar_tema)
        themes_layout.addWidget(add_button)
        
        themes_group.setLayout(themes_layout)
        layout.addWidget(themes_group)
        
        # Área para selecionar arquivo - REDUZIDA e transformada em layout horizontal simples
        file_layout = QHBoxLayout()
        
        file_label = QLabel("Arquivo:")
        file_layout.addWidget(file_label)
        
        self.file_path_temas = QLineEdit()
        self.file_path_temas.setReadOnly(True)
        file_layout.addWidget(self.file_path_temas, 1)  # Proporção 1 para expandir
        
        browse_button = QPushButton("Procurar")
        browse_button.clicked.connect(lambda: self.browse_file(True))
        browse_button.setMaximumWidth(100)  # Limitar largura do botão
        file_layout.addWidget(browse_button)
        
        # Adicionar layout de arquivo diretamente, sem GroupBox
        layout.addLayout(file_layout)
        
        # Linha horizontal de botões para economizar espaço
        buttons_layout = QHBoxLayout()
        
        # Botão de processamento
        processar_button = QPushButton("Processar Macro-temas")
        processar_button.clicked.connect(self.processar_temas)
        buttons_layout.addWidget(processar_button)
        
        # Adicionar mais botões se necessário no futuro
        # ...
        
        layout.addLayout(buttons_layout)
        
        # Área de resultados
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout()
        
        self.results_temas = QTextEdit()
        self.results_temas.setReadOnly(True)
        results_layout.addWidget(self.results_temas)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.tab_temas.setLayout(layout)
    
    def adicionar_tema(self):
        # Diálogo para adicionar novo tema
        tema_nome, ok = QInputDialog.getText(self, "Novo Tema", "Nome do macro-tema:")
        
        if ok and tema_nome:
            # Adicionar à interface
            tema_layout = QHBoxLayout()
            
            tema_label = QLabel(tema_nome + ":")
            tema_layout.addWidget(tema_label)
            
            tema_edit = QLineEdit("")
            tema_layout.addWidget(tema_edit)
            
            # Adicionar ao dicionário e à interface
            self.theme_editors[tema_nome] = tema_edit
            self.macro_temas[tema_nome] = []
            
            # Adicionar ao layout
            self.tab_temas.layout().itemAt(0).widget().layout().addLayout(tema_layout)
    
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
            
            # Mostrar estatísticas básicas
            info = f"Arquivo carregado: {os.path.basename(file_path)}\n"
            info += f"Tamanho: {len(texto)} caracteres\n"
            info += f"Aproximadamente {len(texto.split())} palavras\n"
            
            self.result_text.setText(info)
            self.results_temas.setText(info)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar o arquivo: {str(e)}")
    
    def extrair_texto_pdf(self, pdf_path):
        texto = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    texto += page.extract_text() + "\n"
            return texto
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao processar PDF: {str(e)}")
            return ""
    
    def atualizar_macro_temas(self):
        # Atualizar o dicionário de macro-temas com base nos inputs
        for tema, editor in self.theme_editors.items():
            palavras_texto = editor.text().strip()
            if palavras_texto:
                self.macro_temas[tema] = [p.strip() for p in palavras_texto.split(',')]
            else:
                self.macro_temas[tema] = []
    
    def processar_rede(self):
        if not self.texto_atual:
            QMessageBox.warning(self, "Aviso", "Nenhum texto carregado para processar.")
            return
        
        # Verificar se o filtro está ativado
        palavra_filtro = None
        if self.filtro_checkbox.isChecked():
            palavra_filtro = self.filtro_input.text().strip()
            if not palavra_filtro:
                QMessageBox.warning(self, "Aviso", "Filtro ativado mas nenhuma palavra especificada.")
                return
        
        # Atualizar macro-temas
        self.atualizar_macro_temas()
        
        # Mostrar barra de progresso
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Iniciar processamento em thread separada
        self.thread = ProcessadorThread(self.texto_atual, palavra_filtro, self.macro_temas)
        self.thread.progresso_sinal.connect(self.atualizar_progresso)
        self.thread.resultado_sinal.connect(self.mostrar_resultado_rede)
        self.thread.erro_sinal.connect(self.mostrar_erro)
        self.thread.start()
    
    def processar_temas(self):
        if not self.texto_atual:
            QMessageBox.warning(self, "Aviso", "Nenhum texto carregado para processar.")
            return
        
        # Atualizar macro-temas
        self.atualizar_macro_temas()
        
        # Mostrar barra de progresso
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Iniciar processamento em thread separada
        self.thread = ProcessadorThread(self.texto_atual, None, self.macro_temas)
        self.thread.progresso_sinal.connect(self.atualizar_progresso)
        self.thread.resultado_sinal.connect(self.mostrar_resultado_temas)
        self.thread.erro_sinal.connect(self.mostrar_erro)
        self.thread.start()
    
    def atualizar_progresso(self, valor):
        self.progress_bar.setValue(valor)
    
    def mostrar_erro(self, mensagem):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erro", mensagem)
    
    def abrir_imagem_externamente(self):
        """Abre a imagem atual no visualizador padrão do sistema"""
        if self.imagem_atual_path and os.path.exists(self.imagem_atual_path):
            # Abrir com o aplicativo padrão do sistema
            if sys.platform == 'win32':
                os.startfile(self.imagem_atual_path)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{self.imagem_atual_path}"')
            else:  # Linux e outros
                os.system(f'xdg-open "{self.imagem_atual_path}"')
        else:
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem disponível para abrir.")
    
    def exportar_imagem(self):
        """Exporta a imagem atual para um local escolhido pelo usuário"""
        if not self.imagem_atual_path or not os.path.exists(self.imagem_atual_path):
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem disponível para exportar.")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Imagem", os.path.expanduser("~/Rede_Palavras.png"), 
            "Imagens PNG (*.png)", options=options
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
    
    def mostrar_resultado_rede(self, resultado):
        self.resultado_atual = resultado
        self.progress_bar.setVisible(False)
        
        # Salvar o caminho da imagem
        self.imagem_atual_path = resultado['imagem_path']
        
        # Mostrar resultados no texto
        nos = resultado['nos_importantes']
        texto_resultado = "Palavras mais importantes na rede:\n\n"
        
        for i, no in enumerate(nos[:10], 1):
            pagerank = resultado['pagerank'].get(no, 0)
            texto_resultado += f"{i}. {no} (Importância: {pagerank:.4f})\n"
        
        # Adicionar estatísticas da rede
        G = resultado['grafo']
        texto_resultado += f"\nEstatísticas da rede:\n"
        texto_resultado += f"Número de nós: {G.number_of_nodes()}\n"
        texto_resultado += f"Número de conexões: {G.number_of_edges()}\n"
        
        # Adicionar resultados de macro-temas se disponíveis
        if resultado['tema_resultados']:
            texto_resultado += "\nRelevância para macro-temas:\n"
            tema_ordenado = sorted(resultado['tema_resultados'].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for tema, contagem in tema_ordenado:
                texto_resultado += f"{tema}: {contagem} ocorrências\n"
        
        # Adicionar informação sobre a imagem salva
        texto_resultado += f"\nImagem do grafo salva em:\n{self.imagem_atual_path}\n"
        texto_resultado += "Use o botão 'Abrir Imagem' para visualizar em tamanho completo."
        
        self.result_text.setText(texto_resultado)
        
        # Carregar a imagem no canvas
        try:
            self.canvas.carregar_imagem(self.imagem_atual_path)
        except Exception as e:
            self.result_text.append(f"\nErro ao exibir a imagem: {str(e)}\n")
            self.result_text.append("Use o botão 'Abrir Imagem' para visualizar.")
    
    def mostrar_resultado_temas(self, resultado):
        self.progress_bar.setVisible(False)
        
        # Mostrar resultados de macro-temas
        tema_resultados = resultado['tema_resultados']
        
        if tema_resultados:
            texto_resultado = "Análise de Macro-temas:\n\n"
            
            # Calcular o total de ocorrências para percentuais
            total_ocorrencias = sum(count for _, count in tema_resultados.items())
            if total_ocorrencias == 0:
                total_ocorrencias = 1  # Evitar divisão por zero
            
            # Mostrar todos os temas com seus percentuais, ordenados por relevância
            temas_ordenados = sorted(tema_resultados.items(), key=lambda x: x[1], reverse=True)
            
            for tema, contagem in temas_ordenados:
                porcentagem = (contagem / total_ocorrencias) * 100
                texto_resultado += f"{tema}: {contagem} ocorrências ({porcentagem:.1f}%)\n"
                
                # Listar palavras-chave associadas a este tema
                palavras = self.macro_temas.get(tema, [])
                if palavras:
                    texto_resultado += "   Palavras-chave: " + ", ".join(palavras) + "\n\n"
            
            # Adicionar estatísticas do texto
            texto_resultado += f"\nEstatísticas do texto analisado:\n"
            texto_resultado += f"Tamanho: {len(self.texto_atual)} caracteres\n"
            texto_resultado += f"Aproximadamente {len(self.texto_atual.split())} palavras\n"
            
            # Mostrar palavras mais importantes se disponíveis
            if 'nos_importantes' in resultado and resultado['nos_importantes']:
                texto_resultado += "\nPalavras mais importantes no texto:\n"
                for i, palavra in enumerate(resultado['nos_importantes'][:10], 1):
                    texto_resultado += f"{i}. {palavra}\n"
            
            # Se há uma imagem disponível, mencioná-la
            if 'imagem_path' in resultado and resultado['imagem_path']:
                self.imagem_atual_path = resultado['imagem_path']
                texto_resultado += f"\nImagem do grafo disponível na aba 'Rede de Palavras'."
            
            self.results_temas.setText(texto_resultado)
        else:
            self.results_temas.setText("Nenhum resultado de macro-tema encontrado.")


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
    app.setApplicationName("Processador de Textos Históricos")
    app.setApplicationVersion("1.0.0")
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
        app_temp = QApplication([]) if not QApplication.instance() else QApplication.instance()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Aviso - Modelo spaCy")
        msg.setText("O modelo de linguagem spaCy não foi encontrado.")
        msg.setInformativeText("O aplicativo funcionará com funcionalidade limitada.\n\nPara instalar o modelo, execute:\npython -m spacy download pt_core_news_sm")
        msg.exec_()
    
    # Iniciar aplicação com tratamento de exceções
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        error_msg = f"Erro ao iniciar aplicação: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        # Tentar mostrar diálogo de erro se possível
        try:
            from PyQt5.QtWidgets import QMessageBox
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Erro Fatal")
            error_dialog.setText("A aplicação encontrou um erro e precisa ser fechada.")
            error_dialog.setDetailedText(error_msg)
            error_dialog.exec_()
        except:
            # Se falhar, pelo menos imprimimos o erro no console
            pass
