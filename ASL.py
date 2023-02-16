#-----librerías----------------------------------------------------------------------------------------

# Importamos las librerias para trabajar
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle 
from unicodedata import name
from PIL import Image 

# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import folium


#-----lectura del dataset--------------------------------------------------------------------------
lang = pd.read_csv('lang.csv')
coord = pd.read_csv('coord.csv') 
dflang = pd.read_csv('TerritoryLanguage.txt')
sl_train = pd.read_csv('sign_mnist_train.csv')


#-----configuracion de página--------------------------------------------------------------------------

st.set_page_config(page_title='Idiomas x el mundo', layout='centered', page_icon='🌇')

#-----empieza la app-----------------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; '>Proyecto Final</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; '>Idiomas por el mundo</h1>", unsafe_allow_html=True)
st.image("https://www.avantilanguageschool.com/wp-content/uploads/2015/03/bandera-idiomas.jpg")
st.text("Fuente imagen: https://www.avantilanguageschool.com/")

st.markdown("")
st.markdown("")

#-----Configuración de bloques---------------------------------------------------------------------------
bloques = st.tabs(["Recursos necesarios", "Tratamiento de datos", "Análisis Exploratorio", "Data science con ASL"])

#-----Recursos necesarios---------------------------------------------------------------------------
with bloques[0]:

    st.image("https://www.redeszone.net/app/uploads-redeszone.net/2017/02/curso-de-python-online.jpg")
    st.text("Fuente imagen: https://www.redeszone.net/")

    #-----Librerías-----------
    st.markdown("<h2>Librerías utilizadas y importación del dataset</h2>", unsafe_allow_html=True)
    st.code('''# Importamos las librerías para trabajar
import wget
import os
import gzip
import pandas as pd
import numpy as np
import zipfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Gráficos e imágenes
import IPython.display as display
from IPython.display import IFrame
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

# Web Scrapping
# Selenium
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Machine learning
# Sklearn
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Modelos supervisados
from sklearn.linear_model import LogisticRegression # Regresión logística
from sklearn.neighbors import KNeighborsClassifier # Vecinos cercanos
from sklearn.tree import DecisionTreeClassifier, plot_tree # Árboles de decisión
from sklearn import svm # Máquinas vectores de soporte
from sklearn.ensemble import RandomForestClassifier # Bosques aleatorios

# Modelos no supervisados
from sklearn.decomposition import PCA # Análisis de componentes principales''')
    
    st.markdown("")

    #-----Dataset utilizado-----------
    st.markdown("##### El dataset utilizado esta en este [link](https://unicode-org.github.io/cldr-staging/charts/38/supplemental/territory_language_information.html)")
    dataset = Image.open('dataset.png')
    st.image(dataset)


#-----Tratamiento de datos---------------------------------------------------------------------------
with bloques[1]:
    
    st.image("https://www.finereport.com/en/wp-content/uploads/2019/12/01.jpg")
    st.text("Fuente imagen: https://www.finereport.com/")

    #-----Dataset crudo-----------
    st.markdown("")
    st.markdown("<h2>Tratamiento de los datos</h2>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h5>Dataset crudo</h5>", unsafe_allow_html=True)
    st.dataframe(dflang.head(15))
    st.markdown("")
    st.markdown("")

    st.markdown("<h5>Pasos seguidos para tratar el dataset</h5>", unsafe_allow_html=True)
    st.markdown("")
    clean1 = st.checkbox('Reemplazamos los valores NaN por los valores anteriores')
    if clean1:
        st.code('''dflang = dflang.fillna(method='ffill')''')

    clean2 = st.checkbox('Borramos las columnas y las filas no deseadas')
    if clean2:
        st.code('''dflang.drop(['Code', 'Terr. Literacy', 'Code.1', 'Report Bug'], axis = 1, inplace=True)
dflang = dflang[dflang['Language'] != 'add new']''')

    clean3 = st.checkbox('Cambiamos nombres a las columnas')
    if clean3:
        st.code('''dflang.rename(columns = {"Lang. Pop.":"Speakers", "Pop.%":"Population%"}, inplace=True)''')

    clean4 = st.checkbox('Reemplazamos algunos símbolos del dataset')
    if clean4:
        st.code('''dflang['Language'] = dflang['Language'].str.replace(' {O}','')
dflang['Language'] = dflang['Language'].str.replace(' {OR}','')
dflang['Language'] = dflang['Language'].str.replace(' {OR}','')
dflang['Language'] = dflang['Language'].str.replace(' {OD}','')
dflang['Speakers'] = dflang['Speakers'].str.replace(',','')
dflang['Population%'] = dflang['Population%'].str.replace('%','')
dflang['Literacy%'] = dflang['Literacy%'].str.replace('%','')
dflang['Written%'] = dflang['Written%'].str.replace('%','')''')
        
    clean5 = st.checkbox('Pasamos todas las columnas que tocan a variables númericas')
    if clean5:
        st.code('''lang['Speakers'] = lang['Speakers'].astype(float)
lang['Population%'] = lang['Population%'].astype(float)
lang['Literacy%'] = lang['Literacy%'].astype(float)
lang['Written%'] = lang['Written%'].astype(float)''')
    
    dflang = dflang.fillna(method='ffill')
    dflang.drop(['Code', 'Terr. Literacy', 'Code.1', 'Report Bug'], axis = 1, inplace=True)
    dflang = dflang[dflang['Language'] != 'add new']
    dflang.rename(columns = {"Lang. Pop.":"Speakers", "Pop.%":"Population%"}, inplace=True)
    dflang['Language'] = dflang['Language'].str.replace(' {O}','')
    dflang['Language'] = dflang['Language'].str.replace(' {OR}','')
    dflang['Language'] = dflang['Language'].str.replace(' {OR}','')
    dflang['Language'] = dflang['Language'].str.replace(' {OD}','')
    dflang['Speakers'] = dflang['Speakers'].str.replace(',','')
    dflang['Population%'] = dflang['Population%'].str.replace('%','')
    dflang['Literacy%'] = dflang['Literacy%'].str.replace('%','')
    dflang['Written%'] = dflang['Written%'].str.replace('%','')
    dflang = dflang[dflang['Speakers'] != 'Lang. Pop.']

    st.markdown("")
    st.markdown("")

    #-----Dataset limpio-----------
    st.markdown("<h5>Dataset limpio</h5>", unsafe_allow_html=True)
    st.markdown("")
    st.dataframe(dflang.head(15))

    st.markdown("")
    st.markdown("")

    #-----Web scrapping coordenadas-----------
    st.markdown("<h5>Hacemos un webscrapping con Selenium para sacar las coordenadas de los países a través de google maps y creamos un nuevo dataset</h5>", unsafe_allow_html=True)
    st.markdown("")
    st.video('gmaps.mp4')

    with st.expander("Código"):
        st.code('''# Creamos una lista de los paises del estudio
paises = dflang['Territory'].unique()

# Creamos los links
maps = []
for pais in paises:
    maps.append('https://maps.google.com/maps?q=' + pais)
    
# Inicializamos el chrome
options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")
driver = webdriver.Chrome('Proyecto/chromedriver')
driver.get('https://maps.google.com');

# Sacamos las coordenadas de los países a través de Web Scrapping con selenium y google maps
lat = []
lon = []

for link in maps:
    driver.get(link)
    coord = driver.find_element(by=By.CSS_SELECTOR, value = 'meta[itemprop=image]').get_attribute('content')
    lat.append(coord.split('?center=')[1].split('&zoom=')[0].split('%2C')[0])
    lon.append(coord.split('?center=')[1].split('&zoom=')[0].split('%2C')[1])''')

    st.markdown("")
    st.markdown("")

    #-----Merge datasets-----------
    st.markdown("<h5>Resultado al juntar datasets</h5>", unsafe_allow_html=True)
    st.markdown("")
    st.dataframe(lang.head(15))


#-----Análisis exploratorio---------------------------------------------------------------------------
with bloques[2]:

    st.image('https://blog.datawrapper.de/wp-content/uploads/2022/08/colorblind-f2-copy-1024x512.png')
    st.text("Fuente imagen: https://www.datawrapper.de")
    st.markdown("")
    st.markdown("<h2>Análisis exploratorio</h2>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("<h5>Mostramos el número total de países que se hablan los idiomas más populares</h5>", unsafe_allow_html=True)
    i1 = Image.open('i1.png')
    st.image(i1)
    st.markdown('**Vemos claramente como el inglés es el idioma internacional**.')

    st.markdown("")
    st.markdown("")

    st.markdown("<h5>Mostramos en un mapa los países dónde que se habla el idioma seleccionado</h5>", unsafe_allow_html=True)
    lang_select = st.selectbox("Escoge el idioma",['English', 'Chinese', 'Hindi', 'Spanish', 'Arabic', 'Urdu', 'Bangla', 'French', 'Portuguese', 'Russian'])
    x_lang = lang_select if lang_select != None else 'English'
    x_language = lang[lang['Language'] == x_lang]

    # Mapa interactivo: Idioma y localización
    map = folium.Map(location=[0,0], zoom_start=1.5, tiles='CartoDB positron')
   
    for lat, lon in zip(x_language['Latitude'], x_language['Longitude']):
        folium.Marker([lat, lon], 
                    icon=folium.CustomIcon(icon_image='https://i.imgur.com/CYx04oC.png', icon_size=(10,10))).add_to(map)
 
    st_folium(map, width=700, height=450)

    st.markdown("")

    st.markdown("<h5>Mostramos el número total de hablantes de los idioma más populares</h5>", unsafe_allow_html=True)
    i2 = Image.open('i2.png')
    st.image(i2)
    st.markdown('**Ahora vemos que en términos de población el inglés se ve superado por el chino**.')


#-----Data science con ASL---------------------------------------------------------------------------
with bloques[3]:

    st.image('https://i.blogs.es/2b36a7/algoritmo/1366_2000.png')
    st.text("Fuente imagen: https://www.xataka.com/")
    st.markdown("")
    st.markdown("<h2>Ciéncia de datos con lenguaje de signos</h2>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown('''<h6>En esta parte del proyecto vamos a probar distintos modelos predictivos con un dataset de lenguaje de signos.
Trabajaremos con dos datasets, Train y Test.</h6>''', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h5>Mostramos por ejemplo el dataset Train</h5>", unsafe_allow_html=True)   
    st.dataframe(sl_train.head(10))

    st.markdown('''* Nuestro dataset `Train` tiene `27.455 observaciones` y `785 variables`.
* Nuestro dataset `Test` tiene `7.172 observaciones` y también como no `785 variables`.''')
    st.markdown('''* Podemos ver que todas las columnas son información para los píxeles excepto la columna `label`. 
* Tenemos entonces `784 pixeles` por `imágen` y suponemos que la columna `label` sera la letra del abecedario a la cual hacen referencia las otras columnas.
* Por último, la información de cada píxel ocupa 1 byte de memoria ya que van de 0 a 255 ''')

    st.markdown("")
    st.markdown("")

    st.markdown("<h5>Mostramos ahora una fila del dataframe Train en formato imagen pasando de un array 1D a 2D.</h5>", unsafe_allow_html=True)
    i3 = Image.open('i3.png')
    st.image(i3)

    with st.expander("Código"):
        st.code('''# Introducimos todo el abecedario en la lista 'label_map'
label_map = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Mostramos ahora una fila del dataframe Train en formato imagen pasando de un array 1D a 2D.
ax = plt.figure(figsize=(4,4)) # Tamaño de la figura
solo_image = pd.DataFrame(X_train).iloc[0].values.reshape([28,28]) # Creamos la variable 'solo_image' con un 2Darray de 28x28 píxeles a partir de 1Darray de 784 píxeles.
plt.imshow(solo_image, cmap='gray_r') # Gráficamos la variable solo_image con escala de grises
plt.title(label_map[y_train[1]], fontsize=15, pad=15); # El título de la imagen será la letra que contiene y_train''')

    st.markdown("")
    st.markdown("")

    #-----PCA-----------
    st.markdown("<h3>Compresión de las imágenes: Reducción con PCA</h3>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('''El análisis de componentes principales [PCA](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales) es una técnica de reducción de dimensionalidad lineal utilizada para describir un conjunto de datos con variables correlacionadas en un número más pequeño de variables no correlacionadas llamados componentes.

Uno de los casos de uso de PCA es que se puede usar para la compresión de imágenes, una técnica que minimiza el tamaño en bytes de una imagen mientras mantiene la mayor calidad posible. Comprimiendo la imágen entonces, reduciremos el nombre de variables del dataset y por lo tanto reduciremos los costos computaciones con el uso de modelos predictivos más adelante.''')
    
    st.markdown("")
    st.markdown("")

    st.markdown("<h5>Creamos un grafico que nos mostrará la varianza en función del número de componentes de un análisis de componentes principales (PCA).</h5>", unsafe_allow_html=True)
    
    i4 = Image.open('i4.png')
    st.image(i4)
    st.markdown("Vemos que cuando el número de componentes llega a los ``400`` se estabiliza y los datos ya tienden al 100% de la explicación de la varianza.")

    st.markdown("")
    st.markdown("")

    st.markdown("<h5>Probamos ahora utilizar una PCA con diferentes componentes</h5>", unsafe_allow_html=True)

    i5 = Image.open('i5.png')
    st.image(i5)
    st.markdown('''Confirmamos que si existe diferencia en la calidad de la imagen dependiendo del número de componentes.

Nos parece una buena opción tomar como referencia ``400`` como el número de componentes a aplicar con PCA para el uso de los modelos predictivos.''')

    st.markdown("")

    st.markdown("<h2>Preparación de los modelos</h2>", unsafe_allow_html=True)
    with st.expander("Ver pasos"):
        st.markdown('''* **1.** Dividir el dataset en ``Train`` para entrenar el modelo y en ``Test`` para evaluarlo. ✅
* **2.** Dividir cada dataset anterior en dos, uno con la variable de respuesta ``Y`` y el otro con la/las variables predictoras ``X``. ✅
* **3.** Convertir variables categóricas a numéricas con la ayuda de un encoder. las variables ya son números. ❎
* **4.** Escalar los datos siempre que sea necesario. Ya tienen la misma escala entre 0 y 255 ❎
* **5.** Definir el modelo. ✅
* **6.** Entrenar el modelo. ✅
* **7.** Evaluar el modelo. ✅''')

    st.markdown("")

    st.markdown("<h5>Comparación de los modelos</h5>", unsafe_allow_html=True)
    
    i6 = Image.open('i6.png')
    st.image(i6)
    st.markdown('''Primero comentar que como la variable que queremos predecir es una **``variable cualitativa``** todos los modelos utilizados son de **``clasificación``**.

El medolo que nos da una mejor precisión es el SVM aun que es el que más tiempo necesita.''')

    st.markdown("")

    st.markdown("<h2>Predicción de imágenes</h2>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h5>Imagénes del propio dataset</h5>", unsafe_allow_html=True)

    i7 = Image.open('i7.png')
    st.image(i7)

    st.markdown("")
    st.markdown("")

    st.markdown("<h5>Imágenes própias</h5>", unsafe_allow_html=True)

    i8 = Image.open('i8.png')
    st.image(i8)

    st.markdown("")
    st.markdown("")

    st.markdown("<h2>Conclusiones</h2>", unsafe_allow_html=True)
    st.markdown('''* Al final para los modelos no se ha utilizado el PCA. Al estar trabajando con imágenes y no con un dataset numérico como por ejemplo un dataset de predicción de cáncer no se tuvo en cuanta otro factor más, la posición de los píxeles.
* Se necesitarian aplicar redes neuronales convolucionales para tratar este tipo de casos. Podría ser la continuación de este proyecto que no lo descarto para el futuro.''')
