import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.pydantic_v1 import BaseModel, Field

# 1. Configuração do Ambiente
# Configure sua chave de API do OpenAI
# Certifique-se de que a variável de ambiente OPENAI_API_KEY está configurada
# os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_AQUI" 

# 2. Configuração das Ferramentas
# O LangChain já possui um agente especializado para DataFrames do Pandas.
# Ele traduz automaticamente a pergunta do usuário em operações Pandas.
# Não é necessário criar uma ferramenta de execução de código do zero.

# Carregue o arquivo CSV que o usuário disponibilizou
# A função abaixo pode ser adaptada para receber qualquer nome de arquivo
def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return "Arquivo não encontrado. Por favor, verifique o caminho."

# Exemplo de uso: Carregue um arquivo de exemplo
file_path = "./data/creditcard.csv"  # Substitua pelo caminho do seu arquivo CSV
df = load_csv_data(file_path)

if isinstance(df, str):
    print(df) # Exibe a mensagem de erro se o arquivo não for encontrado
else:
    # 3. Criação do Agente
    # Inicialize o modelo de linguagem (GPT-4)
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")

    # Crie o agente de DataFrame. O LangChain faz a magia aqui,
    # conectando o LLM para raciocinar e o DataFrame para ser a "fonte de dados".
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        allow_dangerous_code=True  # Adicione esta linha
)

    # Função para interagir com o agente
    def ask_agent(query):
        try:
            # O agente executa a pergunta e retorna a resposta
            response = agent.run(query)
            return response
        except Exception as e:
            return f"Ocorreu um erro: {e}"

    # --- Exemplos de Perguntas ---
    print("\nPerguntas de Exemplo:")

    # Descrição dos Dados
    print(ask_agent("Qual o tipo de dado de cada coluna?"))
    print(ask_agent("Qual a média, mediana e desvio padrão da coluna 'Amount'?")) # Substitua 'idade' por uma coluna do seu CSV

    # Relações entre Variáveis
    print(ask_agent("Mostre a correlação entre todas as variáveis numéricas."))

    # Análise e Visualização
    print(ask_agent("Crie um histograma para a coluna 'Time'.")) # Substitua 'salario' por uma coluna do seu CSV
    
    # Anomalias
    print(ask_agent("Identifique se existem valores atípicos (outliers) na coluna 'Amount' e explique como eles afetam a média."))

    # Conclusões
    # Para gerar conclusões, você pode fazer uma pergunta mais aberta.
    print(ask_agent("Baseado na análise, quais são as principais conclusões que você obteve sobre os dados?"))