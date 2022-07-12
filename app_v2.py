import streamlit as st
import numpy as np
import pandas as pd 
import altair as alt

from PIL import Imidade
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_pidade_config(pidade_title='Insuficiência Cardíaca')

st.title('Predição de Insuficiência Cardíaca')
st.subheader('Essa aplicação usa a técnica: ***Random Forest***')
st.write('---')

st.sidebar.header('Diretório')
app = st.sidebar.selectbox('', ['Explorar Dados', 'Predição de Mortalidade', 'Citação'])

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

if app == 'Explore Data':
    about_expander = st.beta_expander('Sobre',expanded=True)
    with about_expander:
        img = Imidade.open('heartattack.jpg')
        st.imidade(img)
        st.write("""
                	As doenças cardiovasculares (DCVs) são a causa número 1 de morte em todo o mundo,
		tirando cerca de 17,9 milhões de vidas a cada ano, o que representa 31% de todas as mortes 
		em todo o mundo. A insuficiência cardíaca é um evento comum causado por DVCs e este conjunto 
		de dados contém 12 recursos que podem ser usados para prever mortalidade por insuficiência 
		cardíaca.
                """)

    st.subheader('**Explore o conjunto de dados**')
    col1, col2 = st.beta_columns(2)
    selectbox_options = col1.selectbox('Transforme', ['Head','Tail', 
                                                        'Describe','Shape', 
                                                        'DTypes', 'Value Count'])
    if selectbox_options == 'Head':
        input_count = col2.number_input('Count', 5, 50, help='min=5, max=50')
        st.write(df.head(input_count))
    elif selectbox_options == 'Tail':
        input_count = col2.number_input('Count', 5, 50, help='min=5, max=50')
        st.write(df.tail(input_count))
    elif selectbox_options == 'Describe':
        st.write(df.describe())
    elif selectbox_options == 'Shape':
        st.write(df.head())
        st.write('Shape: ', df.shape)
    elif selectbox_options == 'DTypes':
        st.write(df.dtypes)
    
    st.write('---')
    numeric_df = df.select_dtypes(['float64', 'int64'])
    numeric_cols = numeric_df.columns

    st.subheader('**Filtrar colunas com Multiseleção**')
    st.write("""Esse recurso é para comparar determinadas colunas no conjunto de dados.
				Você pode adicionar apenas as colunas que deseja comparar e explorar.
                """)
    feature_selection = st.multiselect('', options=numeric_cols)
    df_features = df[feature_selection]
    st.write(df_features)
    st.write('---')

    st.sidebar.subheader('Configurações de Visualização')
    y_axis = st.sidebar.selectbox('Selecione o eixo Y', ['idade', 'fracao_ejecao', 
                                                    'tempo'])
    x_axis = st.sidebar.selectbox('Selecione o eixo X', ['plaquetas', 'creatinina_fosfoquinase', 
                                                    'creatinina_serica', 'sodio_serico'])
    label = st.sidebar.selectbox('Selecione rótulo', ['morte', 'anemia', 'diabetes', 
                                                    'pressão_arterial_elevada', 'sexo', 
                                                    'tabaco'])
    st.subheader('**Visualização**')
    st.write("""Personalize o eixo x e y através das configurações de visualização da barra lateral.
				Você também pode selecionar recursos binários como rótulos que estarão na forma	de 
				uma cor.""")
    select_graph = st.sidebar.radio('Selecione o Gráfico', ('point', 'bar', 'area', 'line'))

    col1, col2, col3 = st.beta_columns([.5,.5,1])
    graph_hgt = col1.slider('Height', 200, 600, 400, step=10)
    graph_wgt = col2.slider('Width',400, 800, 600, step=10)
        
    df = df.loc[(df.creatinina_fosfoquinase < 800) & (df.plaquetas < 500000) & 
                (df.creatinina_serica < 2.2) & (df.idade >= 40)]

    chart = alt.Chart(data=df, mark=select_graph).encode(alt.X(x_axis, scale=alt.Scale(zero=False)), 
                                                            alt.Y(y_axis, scale=alt.Scale(zero=False)),color=label).properties(
        height=graph_hgt,width=graph_wgt)
    st.write(chart)
    
    if y_axis == 'idade' and x_axis == 'plaquetas' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de plaquetas variando de 150,000 - 300,000 e idade 58 - 75')
    elif y_axis == 'idade' and x_axis == 'creatinina_fosfoquinase' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de creatinina fosfoquinase variando de 100 - 250 e idade 55 - 70')
    elif y_axis == 'idade' and x_axis == 'creatinina_serica' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de creatinina sérica variando de 1.2 - 1.9 e idade 50 - 75')
    elif y_axis == 'idade' and x_axis == 'sodio_serico' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha uma contagem de sódio sérico variando de 134 - 140 e idade 55 - 80')
    
    elif y_axis == 'fracao_ejecao' and x_axis == 'plaquetas' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de plaquetas variando de 150,000 - 250,000 e contagem da fração de ejeção de 10 - 30') 
    elif y_axis == 'fracao_ejecao' and x_axis == 'creatinina_fosfoquinase' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de creatinina fosfoquinase variando de 50 - 175 e contagem da fração de ejeção de 20 - 30') 
    elif y_axis == 'fracao_ejecao' and x_axis == 'creatinina_serica' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de creatinina sérica variando de 1.8 - 2 e contagem da fração de ejeção de 20 - 40') 
    elif y_axis == 'fracao_ejecao' and x_axis == 'sodio_serico' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de soro_sódio variando de 134 - 138 e contagem da fração de ejeção de 20 - 40') 
        
    elif y_axis == 'tempo' and x_axis == 'plaquetas' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de plaquetas variando de 150,000 - 350,000 e tempo de acompanhamento inferior a 50 dias') 
    elif y_axis == 'tempo' and x_axis == 'creatinina_fosfoquinase' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de creatinina fosfoquinase variando de 50 - 250, 550 - 600, e tempo de acompanhamento inferior a 50 dias') 
    elif y_axis == 'tempo' and x_axis == 'creatinina_serica' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de creatinina sérica variando de 0.9 - 1.5 e tempo de acompanhamento inferior a 50 dias') 
    elif y_axis == 'tempo' and x_axis == 'sodio_serico' and label == 'morte':
        st.write('A maioria dos pacientes falecidos tinha contagem de soro_sódio variando de 134 - 140 e tempo de acompanhamento inferior a 100 dias') 
        

elif app == 'Predição de Mortalidade':
    st.sidebar.subheader('User Input Features')

    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    X = df.drop('morte', axis=1)
    y = df['morte']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

    def user_input_features():
        display = ("Mulher (0)", "Homem (1)")
        options = list(range(len(display)))
        sexo = st.sidebar.radio("Sexo", options, format_func=lambda x: display[x])

        tabaco = st.sidebar.checkbox('Tabaco')
        if tabaco:
            tabaco = 1
        pressão_arterial_elevada = st.sidebar.checkbox('Hipertensivo')
        if pressão_arterial_elevada:
            pressão_arterial_elevada = 1
        diabetes = st.sidebar.checkbox('Diabético')
        if diabetes:
            diabetes = 1
        anemia = st.sidebar.checkbox('Anêmico')
        if anemia:
            anemia = 1
            
        idade = st.sidebar.slider('idade', 40, 95, 60)
        fracao_ejecao = st.sidebar.slider('Fração de Ejeção', 14, 80, 38)
        sodio_serico = st.sidebar.slider('Sódio Sérico', 113, 148, 136)
        
        creatinina_fosfoquinase = st.sidebar.number_input('Creatinina Fosfoquinase', 23, 7861, 581)
        plaquetas = st.sidebar.number_input('Contagem de plaquetas', 25100.00, 850000.00, 263358.00, help='25100 < input < 850000')
        creatinina_serica = st.sidebar.number_input('Creatinina Serica', 0.5, 9.4, 1.3)
        tempo = st.sidebar.number_input('Período de acompanhamento (Days)', 4, 285, 130)
        data = {'idade': idade,
                'anemia': anemia,
                'creatinina_fosfoquinase': creatinina_fosfoquinase,
                'diabetes': diabetes,
                'fracao_ejecao': fracao_ejecao,
                'pressão_arterial_elevada': pressao_arterial_elevada,
                'plaquetas': plaquetas,
                'creatinina_serica': creatinina_serica,
                'sodio_serico': sodio_serico,
                'sexo': sexo,
                'tabaco': tabaco,
                'tempo': tempo
                }
        features = pd.DataFrame(data, index=[0])
        return features

    user_data = user_input_features()
    st.subheader('**Parâmetros de entrada**')
    st.write(user_data)
    my_expander = st.beta_expander('Check dataset')
    with my_expander:
        st.write(df)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    user_result = classifier.predict(user_data)

    st.title('')
    st.subheader('**Conclusão:**')
    pred_button = st.button('Predição')
    if pred_button:
        if user_result[0] == 0:
            st.success('Paciente sobreviveu durante o período de acompanhamento')
        else:
            st.error('Paciente falecido durante o período de acompanhamento')

else:
    st.header('**Referencia**')