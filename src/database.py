import pandas as pd
from sqlalchemy import create_engine, inspect, text
import urllib.parse

class DatabaseConnection:
    def __init__(self, db_type, host=None, port=None, database=None, username=None, password=None):
        self.db_type = db_type.lower()
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        self._connect()
    
    def _connect(self):
        try:
            if self.db_type == "postgresql":
                connection_string = f"postgresql://{self.username}:{urllib.parse.quote_plus(self.password)}@{self.host}:{self.port}/{self.database}"
            elif self.db_type == "mysql":
                connection_string = f"mysql+pymysql://{self.username}:{urllib.parse.quote_plus(self.password)}@{self.host}:{self.port}/{self.database}"
            elif self.db_type == "sqlserver":
                driver = "{ODBC Driver 17 for SQL Server}"
                connection_string = f"mssql+pyodbc://{self.username}:{urllib.parse.quote_plus(self.password)}@{self.host}:{self.port}/{self.database}?driver={driver}"
            elif self.db_type == "sqlite":
                connection_string = f"sqlite:///{self.database}"
            else:
                raise ValueError(f"サポートされていないデータベースタイプ: {self.db_type}")
            
            self.engine = create_engine(connection_string)
            self.engine.connect()
            
        except Exception as e:
            raise Exception(f"データベース接続エラー: {str(e)}")
    
    def get_table_schema(self):
        try:
            inspector = inspect(self.engine)
            schema_info = []
            
            for table_name in inspector.get_table_names():
                schema_info.append(f"テーブル: {table_name}")
                columns = inspector.get_columns(table_name)
                
                for column in columns:
                    col_type = str(column['type'])
                    nullable = "NULL可" if column['nullable'] else "NOT NULL"
                    schema_info.append(f"  - {column['name']}: {col_type} ({nullable})")
                
                primary_keys = inspector.get_pk_constraint(table_name)
                if primary_keys['constrained_columns']:
                    schema_info.append(f"  主キー: {', '.join(primary_keys['constrained_columns'])}")
                
                foreign_keys = inspector.get_foreign_keys(table_name)
                for fk in foreign_keys:
                    schema_info.append(f"  外部キー: {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
                
                schema_info.append("")
            
            return "\n".join(schema_info)
            
        except Exception as e:
            raise Exception(f"スキーマ取得エラー: {str(e)}")
    
    def execute_query(self, sql_query):
        try:
            sql_query = sql_query.strip()
            
            if sql_query.upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                raise ValueError("参照クエリのみ実行可能です")
            
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query))
                
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return df
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            raise Exception(f"クエリ実行エラー: {str(e)}")
    
    def close(self):
        if self.engine:
            self.engine.dispose()