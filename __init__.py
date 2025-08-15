from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import mysql.connector, sys, os, threading, re

class Database:
    def __init__(self):
        self.db_params = {'host': "localhost", 'user': "web", 'password': "web"}
        self.db_name = "webcfc_go_jaguar"
        self.connection = self.start_connect()
        self.schema = dict()
        if self.connection:
            self.get_schema()

    def start_connect(self):
        try:
            conn = mysql.connector.connect(database=self.db_name, **self.db_params, connection_timeout=10)
            if conn.is_connected():
                return conn
        except mysql.connector.Error as e:
            print(f"❌ Falha ao conectar em '{self.db_name}': {e}")
            return None

    def get_schema(self):
        query_schema = f"""
            SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.db_name}';
        """
        cursor = self.connection.cursor()
        cursor.execute(query_schema)
        
        for table, column, col_type in cursor.fetchall():
            if table not in self.schema:
                self.schema[table] = {}
            self.schema[table][column] = col_type
        cursor.close()

    def get_schema_for_prompt(self, table_name="aulas_praticas"):
        if table_name not in self.schema:
            return f"-- Tabela '{table_name}' não encontrada."
        
        columns = [f"  `{col}` {dtype}" for col, dtype in self.schema[table_name].items()]
        return f"CREATE TABLE `{table_name}` (\n" + ",\n".join(columns) + "\n);"

    def execute_query(self, query):
        if not query.strip().upper().startswith("SELECT"):
            return "Erro: Apenas consultas SELECT são permitidas.", None

        conn = None
        try:
            conn = mysql.connector.connect(database=self.db_name, **self.db_params)
            cursor = conn.cursor()
            cursor.execute(query)
            header = [i[0] for i in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return results, header
        except mysql.connector.Error as e:
            print(f"❌ Erro ao executar a query: {e}")
            return f"Erro de SQL: {e}", None
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

class DeepSeekAgent:
    def __init__(self):
        self.db = Database()
        
        self.model_repo_id = "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF"
        self.model_filename = "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
        
        self.local_model_dir = "models"
        self.model_path = os.path.join(self.local_model_dir, self.model_filename)
        self.model = None
        
        self.setup_model()
    
    def setup_model(self):
        if not os.path.exists(self.model_path):
            self.download_model()
        self.load_model()

    def download_model(self):
        print(f"📥 Baixando o modelo '{self.model_filename}' (aprox. 9.7 GB)...")
        os.makedirs(self.local_model_dir, exist_ok=True)
        try:
            hf_hub_download(repo_id=self.model_repo_id, filename=self.model_filename, local_dir=self.local_model_dir)
            print("✅ Download concluído com sucesso!")
        except Exception as e:
            print(f"❌ Erro fatal ao baixar o modelo: {e}")
            sys.exit(1)

    def load_model(self):
        print(f"🧠 Carregando modelo '{self.model_filename}' com llama-cpp-python...")
        try:
            self.model = Llama(model_path=self.model_path, n_gpu_layers=0, n_ctx=4096, verbose=False)
            print("✅ Modelo de IA carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro fatal ao carregar o modelo: {e}")
            sys.exit(1)

    def _call_llm(self, messages, temperature=0.0):
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=temperature,
            stop=["--", ";"]
        )
        return response['choices'][0]['message']['content'].strip()

    def _generate_sql(self, user_prompt):
        print("⚙️  Etapa 1: Gerando consulta SQL...")
        schema_prompt = self.db.get_schema_for_prompt("aulas_praticas")
        
        system_message = f"""Você é um especialista em MySQL. Sua tarefa é criar uma consulta SQL para responder à pergunta do usuário com base no esquema da tabela fornecido.
            - Retorne APENAS a consulta SQL, sem explicações ou qualquer outro texto.
            - A coluna `data` é do tipo DATETIME. Para comparar anos, use a função `YEAR(data)`.

            ### ESQUEMA
            {schema_prompt}
            """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"-- User Question: {user_prompt}\n-- SQL Query:"}
        ]
        
        generated_text = self._call_llm(messages, temperature=0.0)
        
        sql_query = generated_text.replace("```sql", "").replace("```", "").strip()
        if not sql_query.endswith(';'):
            sql_query += ';'
            
        return sql_query

    def _generate_human_response(self, user_prompt, sql_query, query_result, header):
        print("⚙️  Etapa 2: Gerando resposta humanizada...")
        result_str = f"Cabeçalho: {header}\nResultados:\n{query_result}"
        
        system_message = f"""Você é um assistente de sistema. Sua tarefa é analisar o resultado de uma consulta SQL e fornecer uma resposta clara e amigável para o usuário.

        - A pergunta original do usuário foi: "{user_prompt}"
        - A consulta SQL executada foi: `{sql_query}`
        - O resultado da consulta é:
        {result_str}

        Com base nesses dados, formule uma resposta concisa e direta."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Por favor, explique o que esses resultados significam em relação à minha pergunta."}
        ]
        
        return self._call_llm(messages, temperature=0.2)

    def generate_response(self, prompt):
        sql_query = self._generate_sql(prompt)
        print(f"🔍 SQL Gerado: {sql_query}")

        query_result, header = self.db.execute_query(sql_query)
        if header is None:
            return f"Desculpe, não consegui executar a consulta. O erro foi: {query_result}"
        
        print(f"📊 Resultado do Banco: {query_result}")

        human_response = self._generate_human_response(prompt, sql_query, query_result, header)
        return human_response

    def get_question(self):
        return input("\nFaça sua pergunta sobre as aulas práticas (ou 'sair'): ")

if __name__ == "__main__":
    print("🚀 Iniciando DeepSeek Agent (Coder V2)...")
    deepseek_agent = DeepSeekAgent()

    while True:
        pergunta = deepseek_agent.get_question()
        if pergunta.strip().lower() == "sair":
            print("Encerrando o programa.")
            break
        
        resposta = deepseek_agent.generate_response(pergunta)
        print("\n🤖 DeepSeek:", resposta)