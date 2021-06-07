import math
import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import joblib
from fbprophet import Prophet
import matplotlib.pyplot as plt 
import altair as alt
import plotly.graph_objects as go
class DataBase():
    global USER, PASS, HOST, DATABASE
    USER = st.secrets['USER']#'app_ETL_MP'
    HOST = st.secrets['HOST']#20.69.216.47'
    PASS = st.secrets['PASS']#'gxwLsSgSQxQ'
    DATABASE = st.secrets['DATABASE'] #'etl'
    
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
  
    def create_new_positions(self, df,value, model):
        df_copy = df.copy()
        aquisicao_originais = df['Aquisição de Usuários'].values
        visitas_originais = df['Visitas ao Perfil'].values
        novos_valores_aquisicao = aquisicao_originais + (aquisicao_originais * value) / 100
        novas_visitas_originais = visitas_originais + (visitas_originais * value) / 100
        df_copy['Aquisição de Usuários'] = novos_valores_aquisicao.astype(int)
        df_copy['Visitas ao Perfil'] = novas_visitas_originais.astype(int)
        df_copy = df_copy.drop(['appId', 'position'],axis = 1)
        predictions = model.predict(df_copy)
        predictions = [1 if pred <= 1 else pred for pred in predictions]
        rounded_predictions = np.round(predictions)
        predictions = np.abs(rounded_predictions)
        predictions_series = self.fix_outliers(predictions)
        df_final = pd.DataFrame({'date': df.index, "position": predictions_series})
        df_final = df_final.set_index('date')
        df_final['appId'] = str(df['appId'].iloc[0])
        return df_final
      

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
    
    def fix_outliers(self, predictions):
        outliers = []
        index = {}
        i = 0
        mae = 6
        preds_mean = np.mean(predictions,0)
        preds_std = np.std(predictions, 0)
        for pred in predictions:
            if (pred > (preds_mean + preds_std * 2)):
                index[i] = pred
                outliers.append(pred)
            elif (pred < (preds_mean - preds_std * 2)):
                index[i] = pred
                outliers.append(pred)
            i += 1
        predictions_series = pd.Series(predictions)
        print(index)
        if len(index) != 0:
            for key in index:
                for ind, val in zip(predictions_series.index, predictions_series):
                    if int(key) == (ind):
                        value = index[key]
                        fixed_position = round((mae + preds_mean + preds_std +value)/(4))
                        predictions_series.iloc[ind] = fixed_position
        
        return predictions_series
   
    def _plot_forecast(self, data, X, train_size, flag_new_preds = False, df_concorrente = pd.DataFrame({})):
        data['yhat'] = round(data['yhat'],0)
        
        yhat = data['yhat'].tolist()

        data['yhat'] = [1 if pred <= 1 else pred for pred in yhat]
     
        data['yhat_lower'] = round(data['yhat_lower'],0)
    
        yhat_lower = data['yhat_lower'].tolist()
  
        data['yhat_lower'] = [1 if pred <= 1 else pred for pred in yhat_lower]
     
        data['yhat_upper'] = round(data['yhat_upper'],0)
      
        data['yhat_lower'] = np.abs(data['yhat_lower'])
        data['yhat_upper'] = np.abs(data['yhat_upper'])
        if df_concorrente.shape[0] != 0:
            df_concorrente['yhat'] = round(df_concorrente['yhat'], 0)
            yhat_concorrente = df_concorrente['yhat'].tolist()
            df_concorrente['yhat'] = [1 if pred <= 1 else pred for pred in yhat_concorrente]
            df_concorrente['yhat_lower'] = round(df_concorrente['yhat_lower'],0)
            yhat_lower_concorrente = df_concorrente['yhat_lower'].tolist()
            df_concorrente['yhat'] = [1 if pred <= 1 else pred for pred in yhat_lower_concorrente]
            df_concorrente['yhat_upper'] = round(df_concorrente['yhat_upper'], 0)
            df_concorrente['yhat_lower'] = np.abs(df_concorrente['yhat_lower'])
        """
        fig, ax = plt.subplots(figsize = (20,10))
        ax.plot(X[0:train_size].index, X[0:train_size].values, label = 'Valores Observados', color = '#2574BF')
        ax.plot(data['ds'][-10:], data['yhat'][-10:].values, label = 'Previsões de Posição de Categoria', alpha = 0.7, color = 'red')
        ax.plot(X[train_size:train_size+5].index, X[train_size: train_size + 5].values, label = 'Valores Reais', color = 'green')
        ax.fill_between(data['ds'][-10:], data['yhat_lower'][-10:].values, data['yhat_upper'][-10:].values, color = 'k', alpha = 0.1)
        ax.set_title("Forecasting Posicao")
        ax.set_xlabel("Data")
        ax.set_ylabel("Posição")
        ax.legend()
        st.pyplot(fig)
        """
        _df = pd.DataFrame({})
        if flag_new_preds != False:
            _df = data.set_index('ds')
            X = _df['position']
           
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = X[0:train_size].index, y= X[0:train_size].values,
                                 mode = 'lines',name='Observados',  connectgaps=True, line = dict(color = '#1A88FF')))
        fig.add_trace(go.Scatter(x =data['ds'][-10:],y = data['yhat_lower'][-10:].values,mode = 'lines',fill= None, name = 'LI', showlegend = False, line = dict(color = 'lightgray')))
        fig.add_trace(go.Scatter(x = data['ds'][-10:], y = data['yhat_upper'][-10:].values, opacity = 0.1, showlegend = False, fill = 'tonexty', name = 'LS', mode ='lines', line=dict(color = 'lightgray')))

        fig.add_trace(go.Scatter(
            x = data['ds'][-10:],
            y = data['yhat'][-10:].values, opacity = 0.7,
            mode = 'lines', name = 'Previsões de Posição de Categoria', line = dict(color = 'firebrick', dash = 'dot'),  connectgaps=True
        ))
        if df_concorrente.shape[0] != 0:
            aplicativos_concorrentes = df_concorrente['appId'].unique()
            colors = ["#0BD6D4","#6658CB",'#FACA00','#6DD230',"#FF5A60"]
            colors_prediction = ['#d63e0b','#cb9958','#9a00fa','#d2306e','#60ff5a']
            i = 0
            for app in aplicativos_concorrentes:
                app_atual = df_concorrente[df_concorrente['appId'] == app]
                fig.add_trace(go.Scatter(
                    x = app_atual[0:train_size]['ds'],
                    y = app_atual[0:train_size]['position'], mode = 'lines', name = f'Observados do {app}', opacity = 0.28, line = dict(color=colors[i])
                ))
                fig.add_trace(go.Scatter(
                    x = app_atual['ds'][-10:],
                    y = app_atual['yhat'][-10:].values, mode = 'lines', name = f"Previsão {app}", connectgaps=True, opacity = 0.33, line = dict(color = colors_prediction[i], dash = 'dot')
                ))
                i += 1
        fig.update_xaxes(rangeslider_visible=True)
        st.markdown(f"<p style = 'text-align: center'> Previsões da Posição da Categoria do Aplicativo: <b style = 'color: #1A88FF'>{data['appId'].iloc[0]}</b></p>", True)
        fig.update_layout(
                          showlegend = True,
                          legend=dict(
                                orientation="v",
                                yanchor="bottom",
                                x=1,
                                y = 1.03,
                                xanchor="right",
                              
                        ),
                        xaxis_title = 'Data',
                        yaxis_title = 'Posição',
                        height = 600,
                        width = 800,
                        hovermode = 'x unified',
                        template = 'plotly_white',
                        plot_bgcolor="white",
                        )
        st.plotly_chart(fig,use_container_width=True)
        
        data['ds'] = data['ds'].dt.strftime("%d/%m/%Y")
        previsoes = data[['ds','yhat','yhat_lower','yhat_upper']][-5:]
        previsoes = previsoes.rename(columns = {"ds":"Data"})
        previsoes = previsoes.rename(columns = {"yhat":"Posição Prevista"})
        previsoes = previsoes.rename(columns = {'yhat_lower': 'Posição min prevista'})
        previsoes = previsoes.rename(columns = {'yhat_upper': 'Posição máx prevista'})
        st.write(previsoes) 
        return previsoes                       
    def _forecast_concorrentes(self, app):
        app = app.set_index("date")
        app_names = app['appId'].unique()
        prophet_forecast_df = pd.DataFrame({})
        for app_name_concorrente in app_names:
            app_atual = app[app['appId'] == app_name_concorrente]
            ts = app_atual[['position']]
            X = ts['position']
            train_size = int(len(X) * 0.90)
            df_train = pd.DataFrame({'ds': X.index,
                                    'y': X.values})
            prophet = Prophet(changepoint_prior_scale=0.3, holidays_prior_scale=0.3,
                            n_changepoints= 150, seasonality_mode='additive')
            prophet.fit(df_train)
            future = prophet.make_future_dataframe(10)
            p = prophet.predict(future)
            p['appId'] = app_name_concorrente
            p['position'] = pd.Series(app_atual['position'].tolist())
            prophet_forecast_df = pd.concat([prophet_forecast_df, p])
        return prophet_forecast_df
    
    def forecast_data(self, app, df_original = None, df_concorrente = None):
        ts = app[['position']]
        X = ts['position']
        train_size = int(len(X) * 0.90)
        train_set, test_set = X[0: train_size], X[train_size:]
        df_train = pd.DataFrame({'ds': X.index,
                                 'y': X.values})
        prophet = Prophet(changepoint_prior_scale=0.3, holidays_prior_scale=0.3,
                          n_changepoints= 150, seasonality_mode='additive')
        prophet.fit(df_train)
        future = prophet.make_future_dataframe(10)
        p = prophet.predict(future)
        flag = False
        p = pd.concat([p, pd.DataFrame(df_original['position'].reset_index().drop('date',axis = 1))],axis = 1)
        flag = True 
        p['appId'] = app['appId'].iloc[0]
        self._plot_forecast(p, X, int(len(X)), flag, df_concorrente)
        
  
        
class Application():
    def __init__(self):
        self.db = DataBase()
        self.apps = self.db.get_apps()
        self.data_handler = DataHandler()
        
    def _render_header(self):
        st.image('assets/Rankmyvidente (1).png')
        st.markdown('<h1 style = "border-radius: 50px; color: #FFFFFF;text-align:center;text-transform: uppercase; background:-webkit-linear-gradient(#1A88FF, #0bd6d4);">RankMyVidente</h1>',unsafe_allow_html=True)
        st.text("")
        st.text("")
        st.markdown("<p style = 'text-align: center; color: #7E7E7E; font-size: 20px'>Plataforma em estágio MVP, que permite verificar a posição da categoria dos próximos dias.</p>", unsafe_allow_html=True )

  
    def _render_escolha_apps(self):
        app = self.apps
        app = self.data_handler.treat_data(app)
        app = self.data_handler.create_dummies(app,'Category')
        app = self.data_handler.create_dummies(app, 'position_category')
        app = self.data_handler.create_dummies(app, 'score_cat')
        st.markdown("<p style = 'text-align: center'>Por favor, selecione o aplicativo que deseja realizar a predição da posição nós próximos dias.</p>", True)
        aplicativo = st.selectbox('', app['appId'].unique())
        df_concorrente = pd.DataFrame({})
        st.markdown("<p style = 'text-align: center'>Por favor, selecione até 5 aplicativos concorrentes para análise.</p>", True)
        aplicativos_concorrentes = st.multiselect("",app['appId'].unique())
        aplicativo_escolhido = app[app['appId'] == aplicativo]
        if len(aplicativos_concorrentes) != 0:
            for app_concorrente in aplicativos_concorrentes:
                 app_atual = app[app['appId'] == app_concorrente]
                 app_atual['date'] = pd.to_datetime(app_atual['date'])
                 df_concorrente = pd.concat([df_concorrente, app_atual])
        return aplicativo_escolhido, df_concorrente
        ##depois que seleciona o app
    
    def _render_new_positions(self, aplicativo_escolhido, df_concorrente):
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        loaded_model = joblib.load(open('lgb_regressor (1).pkl', 'rb'))
        aplicativo_escolhido = aplicativo_escolhido.set_index("date")
        value = st.slider("Informe a porcentagem, de Downloads a mais para Predição do seu Aplicativo.", 0,25,step = 5)
        if df_concorrente.shape[0] != 0:
            df_concorrente = self.data_handler._forecast_concorrentes(df_concorrente)
        if value != 0:
            predictions = self.data_handler.create_new_positions(aplicativo_escolhido, value, loaded_model)  
            self.data_handler.forecast_data(predictions, aplicativo_escolhido, df_concorrente)
        else: 
            self.data_handler.forecast_data(aplicativo_escolhido, aplicativo_escolhido, df_concorrente)  
      
    def render_app(self):
        self._render_header()
        aplicativo_escolhido, df_concorrente = self._render_escolha_apps()
        self._render_new_positions(aplicativo_escolhido, df_concorrente)

mvp = Application()
mvp.render_app()