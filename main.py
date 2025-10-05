import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Configura o backend do Matplotlib para não interativo (essencial para servidores)
plt.switch_backend('Agg')
import streamlit as st # NOVO: Importe o Streamlit
import io # Para lidar com o gráfico em memória

plt.switch_backend('Agg')

from langchain.prompts import MessagesPlaceholder
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =================================================================
# 1. Configuração do Ambiente e Inicialização do Streamlit
# =================================================================

import os

# 1. Título do Aplicativo (Opcional, mas útil)
st.title("Configuracao chave OPENAI_API_KEY")
st.markdown("---")

# 2. Coleta da Chave API (Inicialização da Chave)
# Use st.session_state para um gerenciamento de estado mais robusto que os.environ
if "openai_key_valid" not in st.session_state:
    st.session_state["openai_key_valid"] = False

# Tenta obter a chave da sessão, se já foi inserida
user_api_key = st.session_state.get("user_api_key", "")

# Cria o campo de entrada para o usuário
user_api_key_input = st.text_input(
    "Insira sua OpenAI API Key:",
    value=user_api_key,
    type="password"
)

if user_api_key_input:
    # Salva a chave na sessão e no ambiente para a biblioteca OpenAI
    st.session_state["user_api_key"] = user_api_key_input
    os.environ["OPENAI_API_KEY"] = user_api_key_input
    st.session_state["openai_key_valid"] = True
    st.success("Chave API configurada com sucesso!")
    # Recarrega o app para prosseguir (opcional, mas limpa a interface)
    #st.rerun() 
else:
    # Se a chave não foi inserida, exibe uma mensagem de aviso
    st.warning("Por favor, insira sua chave API para liberar a aplicação.")

# 3. Lógica Principal do Aplicativo (Portão de Execução)
# O bloco 'if' garante que o código só é executado SE a chave for válida.
if st.session_state["openai_key_valid"]:
    
    # IMPORTANTE: A partir daqui, você pode inicializar o cliente OpenAI
    # (ou qualquer outra biblioteca que use a chave API).
    # Exemplo:
    # from openai import OpenAI
    # client = OpenAI()
    
    st.subheader("🎉 Voce já pode utilizar a Aplicação!")
    
    # O restante da sua aplicação (widgets, chamadas de API, etc.)
    # deve vir DENTRO deste bloco 'if'.
    #st.text_area("Seu input para o modelo:", "Olá, como posso ajudar?")
    # ... outros elementos da UI
    

    # Define o título da aplicação Streamlit
    st.set_page_config(page_title="Agente de Análise de Dados com LangChain/Streamlit", layout="wide")
    st.title("🤖 Agente de Análise de Dados (LangChain + Streamlit)")
    st.caption("Faça upload de um CSV e comece a analisar. A visualização de clusterização está disponível via Tool.")

    # 2. Configuração das Ferramentas
    # O Streamlit não deve rodar plt.show(), mas sim exibir o objeto Figure.
    # A Tool agora salva o gráfico em um buffer de memória e o coloca no session_state.

    @tool
    def run_plotting_code(plotting_code: str) -> str:
        """
        Executa um bloco de código Python Matplotlib/Seaborn para gerar um gráfico. 
        O código DEVE usar o DataFrame 'df' e NÃO DEVE conter plt.show() ou plt.close().
        
        Exemplo de código: 'sns.histplot(df['coluna'], kde=True)'
        """
        # --- NOVO: RE-IMPORTAÇÃO EXPLÍCITA PARA GARANTIR ESCOPO ---
        # Isso garante que plt e sns estejam disponíveis localmente
        # mesmo que o LangChain execute em um escopo restrito.
        import matplotlib.pyplot as plt # Importação necessária para plt
        import seaborn as sns           # Importação necessária para sns
        # -----------------------------------------------------------
        if 'df' not in st.session_state or st.session_state.df is None:
            return "ERRO: DataFrame não carregado. Não é possível plotar."
        
        df = st.session_state.df # O código será executado com acesso a este df
        
        # Prepara o ambiente de execução
        local_scope = {'df': df, 'plt': plt, 'sns': sns} 
        
        # Cria uma nova figura no início para garantir que o Streamlit capture o contexto
        fig, ax = plt.subplots(figsize=(8, 6))
        local_scope['fig'] = fig
        local_scope['ax'] = ax
        
        try:
            # Executa o código de plotagem. O código deve usar 'ax' e não criar uma nova figura.
            exec(plotting_code, globals(), local_scope) 
            
            # O código de plotagem é executado e o objeto 'fig' é populado.
            
            # Armazena a figura no Session State para exibição pelo Streamlit
            st.session_state['general_plot_fig'] = fig 
            
            # Fecha o objeto Matplotlib localmente
            plt.close(fig) 

            # === CORREÇÃO EXTRA (SE NECESSÁRIO) ===
            # st.rerun() # Adicione esta linha APENAS se o problema persistir
            # =======================================
            # NOVO RETURN: O Agente deve interpretar isso como sucesso
            return "Plotagem finalizada. Por favor, gere uma resposta amigável para o usuário sobre a visualização estar pronta no frontend."

            
        except Exception as e:
            # Se o código do LLM falhar (coluna inexistente, erro de sintaxe)
            plt.close('all') # Limpa qualquer figura aberta por erro
            return f"ERRO ao executar o código de plotagem: {e}. O código que causou o erro foi: {plotting_code}. Por favor, corrija e tente novamente ou sugira outra análise."

    @tool
    def run_clustering_analysis() -> str:
        """
        Executa a Análise de Componentes Principais (PCA) seguida pelo algoritmo K-Means 
        para identificar agrupamentos nos dados usando o DataFrame na sessão 'df'.
        
        A Tool gera o gráfico e o armazena no Streamlit Session State para exibição.
            
        Retorna uma string confirmando a ação e instruindo o usuário.
        """
        # Acessa o DataFrame 'df' do Streamlit Session State
        if 'df' not in st.session_state or st.session_state.df is None:
            return "ERRO: O DataFrame ainda não foi carregado. Por favor, carregue um arquivo CSV primeiro."
        
        df = st.session_state.df # Pega o DataFrame da sessão

        try:
            # st.write("\n[EXECUÇÃO DA TOOL] Rodando PCA/KMeans e gerando gráfico...")
            
            # O restante da lógica de execução permanece o mesmo
            df_cluster = df.drop(['Time', 'Class'], axis=1, errors='ignore').select_dtypes(include=['number'])
            
            if df_cluster.empty:
                return "AVISO: Não há colunas numéricas suficientes para o PCA/KMeans. Não foi possível gerar o gráfico."

            # Reduzindo a dimensionalidade
            pca = PCA(n_components=2)
            df_pca = pca.fit_transform(df_cluster)
            
            # Aplicando o algoritmo KMeans
            # (N=2 é um chute inicial, em um cenário real, seria melhor usar Elbow Method)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df_pca)
            
            # NOVO: Armazena APENAS os dados brutos e metadata no estado da sessão
            st.session_state['clustering_data'] = {
                'pca_components': df_pca,
                'clusters': clusters,
                'variance_ratio': pca.explained_variance_ratio_
            }

            # Garante que não haja figura antiga no state
            st.session_state['clustering_plot'] = None 
            
            # O LLM confirma que os dados para plotagem foram gerados
            return "Plotagem finalizada. Os dados de agrupamentos (PCA/K-Means) foram calculados com sucesso. A visualização de clusterização foi gerada e será exibida."
            
        except Exception as e:
            return f"ERRO ao executar a análise de clusterização: {e}"

    # =================================================================
    # 3. Carregamento e Inicialização do DataFrame (Streamlit)
    # =================================================================

    # Inicializa o estado da sessão
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'agent_executor' not in st.session_state:
        st.session_state.agent_executor = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'clustering_plot' not in st.session_state:
        st.session_state.clustering_plot = None

    # Função de upload (substitui a carga estática local)
    def load_data_and_init_agent(uploaded_file):
        try:
            # Carrega o DF diretamente da memória (melhor prática Streamlit)
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Informação visual no sidebar
            st.sidebar.success(f"CSV carregado! {len(df)} linhas, {len(df.columns)} colunas.")
            st.sidebar.dataframe(df.head(), use_container_width=True)
            
            # Inicialização do Agente (movido para cá para ser chamado APENAS após o upload)
            # Configure sua chave de API do OpenAI - DEVE SER UMA VARIÁVEL DE AMBIENTE
            
            llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")

            # INSTRUÇÃO ADICIONAL CRÍTICA NO PROMPT DO AGENTE PANDAS
            CUSTOM_PREFIX = """Você é um assistente de análise de dados. Você tem acesso a um DataFrame Pandas chamado 'df', Matplotlib ('plt') e Seaborn ('sns').
            Você é um agente NÃO-GUI (sem janela gráfica).

            1.  **VISUALIZAÇÃO (TODOS os gráficos, exceto clusterização):** Para qualquer tipo de gráfico (histograma, scatter, boxplot, etc.),
                **VOCÊ DEVE SEMPRE** usar a Tool `run_plotting_code`. O argumento `plotting_code` deve ser o código Python **completo** (uma única linha ou um bloco) 
                que **usa as variáveis `df`, `plt`, `sns` e `ax`**. NUNCA use `plt.show()` ou `plt.close()` dentro do código que você fornece a esta Tool.
                * Exemplo de chamada: run_plotting_code(plotting_code="sns.scatterplot(x='V1', y='V2', data=df, ax=ax)")
            2.  **VISUALIZAÇÃO DE CLUSTERIZAÇÃO:** Para análise K-Means/PCA, **VOCÊ DEVE SEMPRE** usar a Tool `run_clustering_analysis`.
            3.  **CÁLCULOS/DADOS:** Para todas as outras perguntas (média, filtro, contagem), use o código Python/Pandas (`python_repl_ast`). 
                **SEMPRE** chame **plt.close('all')** após qualquer código de plotagem do Pandas.
            """

            # 1. Tool principal (análise textual/cálculos do DataFrame)
            pandas_agent_executor = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                # Adicionando o prefixo personalizado
                prefix=CUSTOM_PREFIX 
            )

            # 2. Tool de Clusterização 
            clustering_tool = run_clustering_analysis
            general_plotting_tool = run_plotting_code # NOVO: Tool Universal de Plotagem

            # 3. Criação da Memória
            # O Streamlit cuida de manter o estado, a memória do LangChain armazena o histórico do LLM
            memory = ConversationBufferWindowMemory(
                k=5, 
                memory_key="chat_history", 
                return_messages=True 
            )

            # 4. Configuração das Mensagens do Prompt
            chat_history_message = MessagesPlaceholder(variable_name="chat_history")
            agent_scratchpad = MessagesPlaceholder(variable_name="agent_scratchpad")
            
            # 5. Agente Pai: Combina todas as ferramentas
            from langchain.agents import initialize_agent
            
            tools = [
                clustering_tool, 
                general_plotting_tool, # NOVO: Adiciona a tool de plotagem
                pandas_agent_executor.tools[0] # Pega a Tool 'python_repl_ast' do agente Pandas
            ]

            agent_executor = initialize_agent(
                tools,
                llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False, # Definir como False no Streamlit para evitar poluição visual
                memory=memory,
                agent_kwargs={
                    "extra_prompt_messages": [chat_history_message, agent_scratchpad],
                    "input_variables": ["input", "agent_scratchpad", "chat_history"],
                }
            )
            
            st.session_state.agent_executor = agent_executor
            
            # Primeira mensagem para o chat
            st.session_state.messages.append({"role": "assistant", "content": f"O DataFrame com {len(df)} linhas foi carregado. Estou pronto para sua análise. Posso calcular estatísticas, ou tente a tool de clusterização com o comando: **run_clustering_analysis()**"})
            
        except Exception as e:
            st.error(f"Erro na inicialização do agente: {e}. Verifique se a variável OPENAI_API_KEY está configurada no seu ambiente.")

    # Widget de Upload na Sidebar
    uploaded_file = st.sidebar.file_uploader("1. Carregue seu Arquivo CSV", type="csv")

    if uploaded_file is not None and st.session_state.df is None:
        # Se um arquivo foi carregado e o DF ainda não está na sessão
        load_data_and_init_agent(uploaded_file)
    elif uploaded_file is None and st.session_state.df is not None:
        # Se o DF já está na sessão (por exemplo, após um refresh), mantenha-o.
        pass
    elif st.session_state.df is None:
        # Se não há arquivo e nem DF, exibe a instrução.
        st.info("Por favor, use o painel lateral para carregar um arquivo CSV e iniciar o agente.")


    # =================================================================
    # 4. Interface de Chat (Substitui o Loop Interativo)
    # =================================================================

    # Função para interagir com o agente
    def ask_agent(query):
        if st.session_state.agent_executor is None:
            return "Agente não inicializado. Por favor, carregue o arquivo CSV primeiro."
            
        try:
            # AQUI o agente faz a chamada e usa as tools, incluindo a run_clustering_analysis
            response = st.session_state.agent_executor.invoke({"input": query})['output']
            return response
        except Exception as e:
            return f"Ocorreu um erro no Agente: {e}"


    # Exibe o histórico de mensagens
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Exibe o gráfico se a tool 'run_clustering_analysis' o gerou
    # Função de plotagem que garante que a figura é criada no ciclo de renderização do Streamlit
    def display_clustering_plot():
        data = st.session_state.get('clustering_data')
        if data is None:
            return # Não há dados para plotar

        # Cria a figura (garantindo que ela é criada no contexto Streamlit)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Usa os dados salvos pela Tool
        df_pca = data['pca_components']
        clusters = data['clusters']
        variance_ratio = data['variance_ratio']

        ax.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        ax.set_title('Agrupamentos K-Means após Redução PCA (2 Componentes)')
        ax.set_xlabel(f'Componente Principal 1 ({variance_ratio[0]*100:.2f}%)')
        ax.set_ylabel(f'Componente Principal 2 ({variance_ratio[1]*100:.2f}%)')
        
        st.subheader("Visualização de Agrupamento Gerada pela Tool:")
        st.pyplot(fig) # Streamlit renderiza a figura
        plt.close(fig) # Fecha a figura Matplotlib para liberar memória

        # Adiciona um botão para limpar os dados e o gráfico
        if st.button("Limpar Gráfico de Clusterização", key="clear_plot_btn"):
            st.session_state.clustering_data = None
            st.rerun() # Re-executa para remover o gráfico 

    # Exibe o gráfico de Clusterização (usando dados para recriar a figura)
    if st.session_state.get('clustering_data') is not None:
        # A lógica de display_clustering_plot (que recria a fig e chama st.pyplot)
        # deve ser chamada aqui.
        display_clustering_plot()
        st.divider()

    # =================================================================
    # BLOCO DE EXIBIÇÃO DE PLOTAGEM GERAL
    # =================================================================

    # Exibe o gráfico de Propósito Geral (histogramas, scatter, etc.)
    # Verifica o nome exato da variável de estado 'general_plot_fig'
    if st.session_state.get('general_plot_fig') is not None:
        st.subheader("Visualização de Gráfico Gerada pelo Agente:")
        
        # Recupera o objeto Figura
        fig = st.session_state['general_plot_fig']
        
        # Comando CRÍTICO: Plota a figura no Streamlit
        st.pyplot(fig) 
        
        # Adiciona um botão para limpar o gráfico da tela
        if st.button("Limpar Gráfico", key="clear_general_plot_btn"):
            # Limpa o objeto da memória do Streamlit e do Matplotlib
            st.session_state['general_plot_fig'] = None
            plt.close(fig) 
            st.rerun() # Força o Streamlit a redesenhar a página sem o gráfico

    st.divider() # Mantém o separador

    # main.py (Lógica de entrada do usuário)
    if st.session_state.df is not None:
        if prompt := st.chat_input("Sua Pergunta de Análise:"):
            
            # === A CORREÇÃO CRÍTICA: LIMPAR ESTADO NO INÍCIO ===
            # Limpa o gráfico anterior. Se a Tool for chamada, ela salvará um novo objeto.
            st.session_state['general_plot_fig'] = None
            st.session_state['clustering_data'] = None
            # ===================================================

            # 1. Armazena o estado atual do objeto de plotagem ANTES de chamar o agente
            # Esta linha agora é desnecessária se o estado foi limpo acima, mas manteremos
            # a verificação de "depois" para o RERUN.
            # plot_state_before = st.session_state.get('general_plot_fig') is not None
            
            # 2. Adiciona a pergunta do usuário e exibe
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # 3. Processa a pergunta com o Agente (Tempo de espera longo)
            with st.chat_message("assistant"):
                with st.spinner("Pensando e executando análises..."):
                    response = ask_agent(prompt) # Executa o agente

                # 4. Adiciona a resposta do Agente (texto)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response) # Exibe o texto imediatamente

            # 5. VERIFICAÇÃO CRÍTICA E RERUN
            # Verifica se existe um objeto de plotagem geral NOVO na sessão (que só existirá
            # se a Tool foi chamada e salvou um novo gráfico após a limpeza inicial)
            is_general_plot_ready = st.session_state.get('general_plot_fig') is not None
            is_clustering_ready = st.session_state.get('clustering_data') is not None
            
            # O RERUN é chamado apenas se uma Tool de plotagem foi executada (e salvou o objeto)
            if is_general_plot_ready or is_clustering_ready:
                st.rerun()

    # O script termina aqui.
else:
    # Garante que o usuário só veja a interface de coleta da chave
    pass # Deixa o código rodar com a mensagem de aviso (warning) acima