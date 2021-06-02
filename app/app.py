from os import stat_result
import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import joblib

class DataBase():
    global USER, PASS, HOST, DATABASE
    USER = 'app_ETL_MP'
    HOST = '20.69.216.47'
    PASS = 'gxwLsSgSQxQ'
    DATABASE = 'etl'
    
    def __init__(self):
        self.user = "app_ETL_MP"
        self.host = '20.69.216.47'
        self.passwrd = 'gxwLsSgSQxQ'
        self.database = 'etl'

    def _get_db(self):
        db = None
        try:
            db = self._connect()
        except mysql.connector.Error as err:
            print("Erro ao conectar no banco de dados: ", err)
        return db

    def _connect(self):
        db = mysql.connector.connect(
            host = self.host,
            user = self.user,
            passwd = self.passwrd, 
            database = self.database
        )
        return db

    def get_apps(self):
        apps = pd.DataFrame({})
        try:
            mydb = self._get_db()
            cursor = mydb.cursor()
            apps = pd.read_sql("SELECT A.APPID, A.POSITION, A.COUNTRY, A.DATE, A.SCORE, A.CATEGORY, B.APPDETAILPAGEVISITS, B.APPDETAILPAGEINSTALLS, B.TITLE FROM (SELECT * FROM appposition WHERE store = 'google' AND country = 'br') AS A INNER JOIN (SELECT * FROM channel_acquisition WHERE country = 'br') AS B ON A.APPID = B.APPID AND A.DATE = B.DATE", con = mydb)            
            cursor.close()
            mydb.close()
            return apps
        
        except mysql.connector.Error as err:
            print('Erro na consulta dos APPs registrados no banco de dados: {}'.format(err))

        try:
            cursor.close()
        except:
            print("O cursor de execução do Banco de Dados já se encontra fechado!")
        try:
            mydb.close()
        except:
            print("A conexão com o Banco de Dados já se encontra fechada!")

class DataHandler():
    def __init__(self):
        pass

    def treat_data(self,app):
        app['score'] = app['score'].astype(float)
        st.write(app)
        app['appDetailPageVisits'] = app['appDetailPageVisits'].astype(float)
        app['appDetailPageInstalls'] = app['appDetailPageInstalls'].astype(float)
        app = app.groupby(['appId','date','position','score','category']).sum().reset_index()
        app =  app.rename({'appDetailPageInstalls':"Aquisição de Usuários", "appDetailPageVisits":"Visitas ao Perfil", "category":"Category"},axis = 1)
        app['Aquisição de Usuários'] = app['Aquisição de Usuários'].replace(0.0, np.nan)
        app['Visitas ao Perfil'] = app['Visitas ao Perfil'].replace(0.0, np.nan)
        aplicativos_unicos = app['appId'].unique()
        final_df = pd.DataFrame({})
        for app_name in aplicativos_unicos:
            print(app_name)
            app_actual = app[app['appId'] == app_name]
            app_actual_mean = app_actual['Aquisição de Usuários'].mean()
            app_actual_visitas = app_actual['Visitas ao Perfil'].mean()
            app_actual['Aquisição de Usuários'] = app_actual['Aquisição de Usuários'].fillna(app_actual_mean)
            app_actual['Visitas ao Perfil'] = app_actual['Visitas ao Perfil'].fillna(app_actual_visitas)
            final_df = pd.concat([final_df, app_actual])
        final_df = final_df.dropna()
        final_df['position_category'] = final_df['position'].apply(self.position_category)
        final_df['score_cat'] = final_df['score'].apply(self.notas_categoria)
        return final_df
    
    def create_dummies(self, df, column_name):
        dummies = pd.get_dummies(df[column_name],prefix=column_name)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(column_name, axis = 1)
        return df
    
    def create_new_positions(self, df,value):
        valores_originais = df['Aquisição de Usuários'].values
        novos_valores = valores_originais + (valores_originais * value) / 100
        novos_valores_list = []
        for value in novos_valores:
            novos_valores_list.append((int(value)))
        return valores_originais, novos_valores_list
    
    def fix_positions(self, predictions):
        result = [round(pred) for pred in predictions]
        return pd.Series(result)
    
    def position_category(self, position):
        resultado = ''
        if position == 1:
            resultado = 'Top 1'
        elif position >= 2 and position <= 5:
            resultado = 'Top 5'
        elif position > 5 and position <= 10:
            resultado = 'Top 10'
        elif position > 10 and position <= 20:
            resultado = 'Top 20'
        elif position > 20 and position <= 50:
            resultado = 'Top 50'
        elif position > 50 and position <= 100:
            resultado = 'Top 100'
        else:
            resultado = 'Top 200'
        return resultado
    
    def notas_categoria(self, nota):
        resultado = None
        if nota >= 4:
            resultado = 'positiva'
        elif nota < 3:
            resultado = 'negativa'
        elif nota >= 3 and nota < 4:
            resultado = 'neutra'
        return resultado        

class Application():
    def __init__(self):
        self.db = DataBase()
        self.apps = pd.read_csv('dados.csv')#self.db.get_apps()
        self.data_handler = DataHandler()
        
    def _render_header(self):
        st.markdown('<h1 style = "border-radius: 50px; color: #FFFFFF;text-align:center;text-transform: uppercase; background:-webkit-linear-gradient(#FD6E1B, #DB2F40);">Zortar - MVP</h1>',unsafe_allow_html=True)
        st.text("")
        st.text("")
        st.markdown("<p style = 'text-align: center; color: #7E7E7E; font-size: 20px'>Plataforma em estágio MVP, que permite a predição da categoria dos próximos dias.</p>", unsafe_allow_html=True )
    
    def _render_escolha_apps(self):
        app = self.apps
        app = app.drop('Unnamed: 0', axis = 1)
        app = self.data_handler.treat_data(app)
        app = self.data_handler.create_dummies(app,'Category')
        app = self.data_handler.create_dummies(app, 'score_cat')
        app = self.data_handler.create_dummies(app, 'position_category')
        aplicativo = st.selectbox('ALGOOO', app['appId'].unique())
        aplicativo_escolhido = app[app['appId'] == aplicativo]
        st.write(aplicativo_escolhido)
        st.write(aplicativo_escolhido.shape)
        st.write(aplicativo_escolhido.columns)
        return aplicativo_escolhido
        ##depois que seleciona o app
    
    def _render_new_positions(self, aplicativo_escolhido):
        dates = aplicativo_escolhido['date']
        aquisicao_de_usuarios = aplicativo_escolhido['Aquisição de Usuários'].values
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        loaded_model = joblib.load(open('lgb_regressor (1).pkl', 'rb'))
        aplicativo_escolhido = aplicativo_escolhido.set_index("date")
        aplicativo_escolhido = aplicativo_escolhido.drop(['appId','position'],axis = 1)
        st.write(aplicativo_escolhido)
        st.write(aplicativo_escolhido.columns)
        predictions = loaded_model.predict(aplicativo_escolhido)
        st.write(predictions)
        """
        aquisicao = aplicativo_escolhido['Aquisição de Usuários'].values
        if st.button("Deseja visualizar com mais downloads?"):
            value = st.slider("Quantas downloads vc quer?", 5,50,step = 5)
            st.write(value)
            valores_originais, novos_valores = self.data_handler.create_new_positions(aplicativo_escolhido, value)
            st.write(valores_originais)
            aplicativo_escolhido['Aquisição de Usuários'] = novos_valores
        
        aplicativo_escolhido = aplicativo_escolhido.set_index("date")
        aplicativo_escolhido = aplicativo_escolhido.drop(['appId','position'],axis = 1)
        st.write(aplicativo_escolhido)
        predictions = loaded_model.predict(aplicativo_escolhido)
        fixed_predictions = self.data_handler.fix_positions(predictions)
        st.write(predictions)
        result = pd.DataFrame({})
        result['date'] = dates
        result['predictions'] = fixed_predictions
        st.write(result)
        """
        
    def render_app(self):
        self._render_header()
        aplicativo_escolhido = self._render_escolha_apps()
        self._render_new_positions(aplicativo_escolhido)

mvp = Application()
mvp.render_app()