from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import mysql.connector, sys, os, threading, json, pprint

class Database:
    def __init__(self):
        self.db_params = {'host': "localhost", 'user': "web", 'password': "web"}
        self.db_name = "webcfc_go_jaguar"
        self.connection = self.start_connect()
        self.get_schema()
        self.tables = ["aulas_praticas"]
        self.data_tables = dict()
        self.data_ready = threading.Event() 

    def start_connect(self):
        try:
            conn = mysql.connector.connect(database=self.db_name, **self.db_params, connection_timeout=10)
            if conn.is_connected():
                return conn
        except mysql.connector.Error as e:
            print(f"❌ Falha ao conectar em '{self.db_name}': {e}")
            return None

    def get_schema(self):
        if not self.connection: return
        self.schema = dict()
        query_schema = f"""
            SELECT DISTINCT TABLE_NAME, COLUMNS.COLUMN_NAME, COLUMNS.COLUMN_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.db_name}';
        """
        cursor = self.connection.cursor()
        cursor.execute(query_schema)
        
        for data_schema in cursor.fetchall():
            if data_schema[0] not in self.schema:
                self.schema[data_schema[0]] = {}
            self.schema[data_schema[0]][data_schema[1]] = data_schema[2]
        
        self.tables = list(self.schema.keys())
        cursor.close()

    def get_data_tables_parallel(self, max_workers=5):
        """Este método agora roda em uma thread separada e sinaliza quando termina."""
        print("⚙️  Iniciando busca de dados do banco em segundo plano...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_table, table): table for table in self.tables}
            for future in as_completed(futures):
                table, results = future.result()
                self.data_tables[table] = results
        
        # Sinaliza que os dados estão prontos
        self.data_ready.set()
        print("✅ Dados do banco de dados carregados com sucesso!")
    
    def fetch_table(self, table_name):
        conn = mysql.connector.connect(database=self.db_name, **self.db_params)
        cursor = conn.cursor()
        try:
            colunas = list(self.schema[table_name].keys())
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY RAND() LIMIT 5")
            results = cursor.fetchall()

            mapped_results = [dict(zip(colunas, row)) for row in results]

            print(f"   -> Tabela '{table_name}': {len(results)} linhas encontradas.")
            return table_name, mapped_results
        except mysql.connector.Error as e:
            print(f"❌ Erro ao buscar '{table_name}': {e}")
            return table_name, []
        finally:
            cursor.close()
            conn.close()

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

class DeepSeekAgent:
    def __init__(self):
        # 1. Inicia a classe de banco de dados
        self.db = Database()
        
        # 2. Inicia a busca de dados em uma thread separada
        self.db_thread = threading.Thread(target=self.db.get_data_tables_parallel, daemon=True)
        self.db_thread.start()
        
        # O código abaixo continua executando enquanto os dados são buscados
        print("🧠 Carregando modelo de IA...")
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DeepSeek1.5B")
        self.check_model()
    
    def check_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.listdir(self.model_path):
            self.download_model()
        self.load_model()
    
    def download_model(self):
        try:
            snapshot_download(repo_id=self.model_name, local_dir=self.model_path, local_dir_use_symlinks=False)
        except Exception as e:
            print(f"Erro ao baixar o modelo: {e}")
            sys.exit(1)

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cpu")
            print("✅ Modelo de IA carregado.")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            sys.exit(1)

    def generate_response(self, prompt):
        """Gera uma resposta, aguardando os dados do DB se necessário."""
        try:
            # Espera até que a thread do DB termine (se ainda não terminou)
            # print("...Verificando se os dados do banco estão prontos...")
            self.db.data_ready.wait(timeout=30) # Espera por até 30 segundos
            
            if not self.db.data_ready.is_set():
                return "Ainda estou carregando os dados do banco. Tente novamente em um instante."
            
            final_prompt = f"""
                --- INÍCIO DAS DIRETIVAS ---
                # Persona e Domínio
                Você é o "Assistente WebCFC", um assistente de IA especialista no sistema de gerenciamento de Centros de Formação de Condutores (autoescolas).
                # Regras
                1.  **Relevância:** Avalie se a pergunta do usuário é sobre o sistema WebCFC. Se não for (ex: política, esporte, etc.), recuse educadamente com a mensagem: "Desculpe, minha função é fornecer informações exclusivamente sobre o sistema WebCFC. Não tenho conhecimento sobre outros assuntos."
                2.  **Contexto:** Para perguntas relevantes, use o ESQUEMA GERAL para entender a estrutura do banco e os DADOS RELEVANTES ENCONTRADOS para obter os detalhes específicos.
                3.  **Veracidade:** Sua resposta deve se basear ESTRITAMENTE nas informações fornecidas. Não invente dados. Se a informação não estiver no contexto, informe que não foi encontrada no sistema.
                --- FIM DAS DIRETIVAS ---

                --- DADOS RELEVANTES ENCONTRADOS PARA ESTA PERGUNTA ---
                {self.db.data_tables['aulas_praticas']}
                --- FIM DOS DADOS RELEVANTES ---

                Com base em TODAS as regras e informações acima, responda à pergunta do usuário.

                Pergunta: "{prompt}"
                Assistente WebCFC:
            """
            
            pprint.pprint(final_prompt)

            generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            resposta = generator(final_prompt, max_new_tokens=150, temperature=0.7)[0]["generated_text"]
            resposta_limpa = resposta.replace(final_prompt, "").strip()
            return resposta_limpa
        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            return None

    def get_question(self):
        return input("\nFaça sua pergunta (ou digite 'sair'): ")

if __name__ == "__main__":
    print("🚀 Iniciando DeepSeek Agent...")
    deepseek_agent = DeepSeekAgent()

    # O loop principal pode começar imediatamente
    while True:
        pergunta = deepseek_agent.get_question()
        if pergunta.strip().lower() == "sair":
            print("Encerrando o programa.")
            break
        
        resposta = deepseek_agent.generate_response(pergunta)
        if resposta:
            print("\n🤖 DeepSeek:", resposta)
        else:
            print("Desculpe, não consegui gerar uma resposta.")

# db = Database()
# db.get_data_tables_parallel()

# pprint.pprint(db.data_tables)