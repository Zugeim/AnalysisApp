# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:31:20 2024

@author: Imanol
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visual_exploration import VisualExploration
import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def show_description(df, key):
    options = sorted(df.columns.tolist())
    
    traspuesta = st.checkbox("Invertir estadisticos", key=key+'_traspuesta_table')
        
    cols = st.multiselect("Selecciona las columnas a mostrar en la" +
                          " descripción y los graficos:",
                          options=options,
                          key=key+'_multiselect')
    if not cols:
        cols = options # Mostrar todas las columnas
    
    if is_numeric_dtype(df[options[0]]):
        st.table(VisualExploration(df[cols]).numerical_description(t_num=traspuesta))
    else:
        st.table(VisualExploration(df[cols]).categorical_description(t_cat=traspuesta))
        
    return cols

def filter_dataframe(df, key, cut=False):
    modify = st.checkbox("Añadir filtros", key=key+'_checkbox')

    if not modify and cut:
        return df.head(10)
    
    if not modify and not cut:
        return df
    

    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar dataframe en ",
                                           sorted(df.columns),
                                           key=key+'multiselect')
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores para {column}",
                    df[column].unique(),
                    default=list(df[column].unique(),),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valores para {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores para {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring o regex en {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def show_df_info(ve):
    st.markdown(f'El DataFrame tiene {ve.n_rows} filas y {ve.n_columns} columnas.')
    st.markdown('<div style="text-align: center; font-size: 20px;"><u>Columnas numéricas</u></div>', unsafe_allow_html=True)
    st.markdown(f"{ve.num_columns}")
    st.markdown('<div style="text-align: center; font-size: 20px;"><u>Columnas categóricas</u></div>', unsafe_allow_html=True)
    st.markdown(f"{ve.cat_columns}")
    
def show_plots(df, cols, key):
    graph = st.checkbox("Ver los graficos", key=key+'_graph')
    
    if graph: 
        traspuesta = st.checkbox("Invertir graficos", key=key+'_traspuesta_graph')
        
        if is_numeric_dtype(df[cols[0]]):
            plots = ['histogram', 'boxplot', 'violinplot',
                     'heatmap', 'corr', 'pairplot']
        else:
            plots = ['donut', 'value_counts']
            
        # Mostrar captions y estadísticas de los datos numéricos
        seleccion = st.selectbox("Selecciona una opción:", plots,)
        
        if not seleccion:
            seleccion = plots[0]
    
        # Mostrar el DataFrame filtrado por las columnas seleccionadas
        if seleccion == 'histogram':
            fig = VisualExploration(df[cols]).show_histograms(traspuesta)
            st.pyplot(fig)
        if seleccion == 'boxplot':
            fig = VisualExploration(df[cols]).show_boxplots(traspuesta)
            st.pyplot(fig)
        if seleccion == 'violinplot':
            fig = VisualExploration(df[cols]).show_violinplots(traspuesta)
            st.pyplot(fig)
        if seleccion == 'heatmap':
            fig = VisualExploration(df[cols]).show_heatmap()
            st.pyplot(fig)
        if seleccion == 'corr':
            st.table(df[cols].corr())

        if seleccion == 'donut':
            fig = VisualExploration(df[cols]).show_donuts(traspuesta)
            st.pyplot(fig)
        if seleccion == 'value_counts':
            fig = VisualExploration(df[cols]).show_value_counts(traspuesta)
            st.pyplot(fig)
        if seleccion == 'pairplot':
            fig = VisualExploration(df[cols]).show_pairplot()
            st.pyplot(fig)
    
def show_numeric_columns(ve):
    df = ve.df_num.copy()
    if df.shape[1] > 0:
        key = 'numeric'
        cols = show_description(df, key)
        show_plots(df, cols, key)
    else:
        st.markdown("NO HAY DATOS CATEGÓRICOS")
        
def show_category_columns(ve):
    df = ve.df_cat.copy()
    key = 'category'
    if df.shape[1] > 0:
        cols = show_description(df, key)
        show_plots(df, cols, key)
    else:
        st.markdown("NO HAY DATOS CATEGÓRICOS")
        
def change_name_columns(ve):
    options = ve.tot_columns
    selected_option = st.selectbox("Selecciona la columna que quieras" +
                                  " cambiar de nombre: ", options)
    new_name = st.text_input("Escribe el nuevo nombre: ")
    
    if st.button("Cambiar nombre de la columna"):
        if new_name:
            ve.change_name_col(selected_option, new_name)
        else:
            st.error("Falta poner un nuevo nombre")
        
    return ve.df.copy()

def change_type_columns(ve):
    
    type_list = ['category', 'int', 'float', 'bool', 'datetime', 'string']
    col_options = ve.tot_columns
    
    change_options = ('Cambio de tipo de varias columnas',
                      'Cambio de tipo columna a columna (Con personalización' +
                      ' de nombre de categorias)')
    selected_option = st.radio("Selecciona como cambiar el tipo de las columnas:",
                               change_options)
    
    if selected_option == change_options[0]:
        cols = st.multiselect("Selecciona las columnas que quieras cambiar" +
                              " de tipo:", col_options)
        new_type = st.selectbox("Selecciona a que tipo cambiar", type_list)
        
        if new_type == 'datetime':
            date_format = st.text_input('Escribe el fromato de fecha (Ej: %m/%d/%Y')
        else:
            date_format = None
        
        if st.button("Cambiar tipos de las columnas"):
            ve.change_type(cols, new_type, date_format=date_format)
            st.success("Tipo de columna cambiado correctamente.")
            
    if selected_option == change_options[1]:
        col = st.selectbox("Selecciona la columna que quieras cambiar de tipo:",
                            col_options)
        new_type = st.selectbox("Selecciona a que tipo cambiar una columna",
                                type_list)
        
        if col and new_type:
            new_categories = None
            if new_type == 'category':
                categories = ve.df[col].unique().tolist()
                new_categories = [str(c) for c in categories]
                st.markdown("Escribe el nombre de cada categoria:\\n" +
                            "(Si no escribes nada se guardará con el valor" +
                            " por defecto")

                for i, v in enumerate(categories):
                    text_value = st.text_input(f"Nuevo nombre de categoría {v}:",
                                               key=f"cat_input_{i}")
                    new_categories[i] = text_value if text_value else str(categories[i])

            if new_type == 'datetime':
                date_format = st.text_input('Escribe el fromato de fecha (Ej: %m/%d/%Y)')
            else:
                date_format = None             

            if st.button("Cambiar tipo de la columna"):
                if new_categories:
                    if len(new_categories) != len(categories):
                        st.error("El número de nuevas categorías debe" +
                                 " coincidir con el número de categorías" +
                                 " originales.")  
                ve.change_type(cols=col, new_type=new_type,
                               new_categories=new_categories,
                               date_format=date_format)
                st.success("Tipo de columna cambiado correctamente.")
    
    st.dataframe(ve.df.dtypes)
    
    return ve.df.copy()

def edit_null_values(ve):
    options = ['Elimina las filas con valores nulos',
               'Elimina las columnas con valores nulos',
               'Imputar valores con la media',
               'Imputar valores con la mediana',
               'Imputar valores con la moda',
               'Imputar valores con usando el valor anterior',
               'Imputar valores con usando el valor anterior',]
    imputations = options[2:]
    drops = options[:2]
    num_opts = sorted(ve.num_columns)
    tot_opts = sorted(ve.tot_columns)
        
    edit_nulls = st.radio("Selecciona como tratar los valores nulos:",
                            options)
    
    if edit_nulls in drops:
        selected_cols = st.multiselect("Selecciona si quieres comprobar"+
                                        " en una columna concreta:", 
                                        tot_opts)
        selected_cols = selected_cols if selected_cols else tot_opts
        
        if st.button("Eliminar"):
                axis = drops.index(edit_nulls)
                ve.drop_na(axis, selected_cols)
                return ve.df.copy()
        
    if edit_nulls in imputations:
        selected_cols = st.multiselect("Selecciona las columna para" +
                                      " imputar valores:", num_opts)
        if st.button("Imputar"):
            if edit_nulls in imputations and selected_cols:
                imp = imputations.index(edit_nulls)
                ve.input_vals(selected_cols, imp)
                return ve.df.copy()
            elif edit_nulls in imputations:
                st.error("No has puesto las columna/s a imputar")
                
    return ve.df.copy()
                
def show_df_with_rows_range(ve):
    df = ve.df.copy()
    columns = ve.tot_columns
    selected_cols = st.multiselect("Selecciona las columnas que quieras ver:",
                                   columns, key='df_with_rows')
    rows_range = st.slider("Selecciona el rango de filas a mostrar",
                           0, ve.n_rows, (0, 10))
    
    df_filtered = df.iloc[rows_range[0]:rows_range[1],:]
    
    if selected_cols:
        st.dataframe(df_filtered[selected_cols])
    else:
        st.dataframe(df_filtered)
        
def show_duplicates(ve):
    df = ve.show_duplicates()
    number_of_duplicates = VisualExploration(df).n_rows
    st.header(f'Registros duplicados: {number_of_duplicates}')
    st.dataframe(df)
    

def drop_duplicates(ve):
    if st.button("Eliminar duplicados"):
        ve.drop_duplicates()
        
    show_duplicates(ve)
        
    return ve.df.copy()

def normalize(ve):
    columns = sorted(ve.num_columns)
    selected_columns = st.multiselect("Selecciona que columnas normalizar:",
                                      columns)
    if st.button('Normalizar'):
        ve.normalize_columns(selected_columns)
        
    return ve.df.copy()

def standarize(ve):
    columns = sorted(ve.num_columns)
    selected_columns = st.multiselect("Selecciona que columnas estandarizar:", 
                                      columns)
    if st.button('Estandarizar'):
        ve.standarize_columns(selected_columns)
        
    return ve.df.copy()

def normalize_and_standarize(ve):
    df = ve.df.copy()
    
    options = ('Normalizar (rango [0,1])',
               'Estandarizar (Media = 0, Desviación estándar = 1)')
    selected_option = st.radio("Selecciona una opción:", options)
    
    if selected_option == options[0]:
        df = normalize(ve)
        
    if selected_option == options[1]:
        df = standarize(ve)
        
    return df
   
def delete_outliers(ve, col):
    ve.delete_column_outliers(col) 
    return ve.df.copy()
    
def winsorizing_outliers(ve, col):
    ve.winsorizing_column_outliers(col) 
    return ve.df.copy()
    
def handling_outliers(ve):
    columns = sorted(ve.num_columns)
    selected_column = st.selectbox("Selecciona de que columna ver outliers:", 
                                      columns)
    
    outliers = ve.return_column_outliers(selected_column)
    n_outliers = len(outliers[0])
    st.markdown(f'Tiene {n_outliers} outliers')
    st.dataframe(outliers[0])
    
    if st.button('Eliminar filas que contienen los outliers'):
        return delete_outliers(ve, selected_column)
        
    if st.button('Capar outliers'):
        return winsorizing_outliers(ve, selected_column)
    
    return ve.df.copy()
 
def strings_correction(ve):
    columns = sorted(ve.return_string_columns())
    selected_column = st.selectbox("Selecciona que columna de string corregir:",
                                   columns)
    
    st.markdown('Valores unicos de la columna')
    st.markdown(f'{ve.df[selected_column].unique()}')

    if st.button('Eliminar espacios delante y detras'):
        ve.string_strip(selected_column)
        
    if st.button('Poner en minúsculas'):
        ve.string_lower(selected_column)
        
    if st.button('Poner en mayúsculas'):
        ve.string_upper(selected_column)
    
    old_string = st.text_input('Escribe el fragmento de texto a corregir')
    new_string = st.text_input('Escribe el nuevo fragmento de texto')
                               
    if st.button('Reemplazar fragmento de texto'):
        ve.string_replace(selected_column, old_string, new_string)
    
    return ve.df.copy()
    
            
def facet_grids(df, rows, category, hue, type_p, key='facet', variable2=None):
    return VisualExploration(df).show_facet_grid(rows=rows, category=category,
                                                 hue=hue, type_p=type_p,
                                                 variable2=variable2)
def show_data_exploration_graphs(ve):
    col1, col2, col3, col4 = st.columns(4)
    
    plot_types_1 = ['histogram', 'kdeplot']
    plot_types_2 = ['scatterplot']
    plot_types = plot_types_1 + plot_types_2
    
    df = filter_dataframe(ve.df.copy(), key='exploration_graphs')
    
    n_cols = sorted(VisualExploration(df).num_columns)
    c_cols = sorted(VisualExploration(df).cat_columns)
    c_cols.insert(0, 'None')
    
    with col1:
        selection1 = st.multiselect("Variables numéricas principales:", n_cols)
    with col2:
        selection2 = st.selectbox("Variables categóricas por las que dividir:",
                                  c_cols)
    with col3:
        selection3 = st.selectbox("Variables categóricas con las que colorear:",
                                  c_cols)
    with col4:
        selection4 = st.selectbox("Selecciona tipo de gráfica:", plot_types)
        
        if selection4 in plot_types_2:
            second_n = st.selectbox("Variable numérica secundaria:", n_cols)
        else:
            second_n = None
            
        
    category = None if selection2 == 'None' else selection2
    hue = None if selection3 == 'None' else selection3
        
    if selection1 and category:
        figs = facet_grids(df, selection1, category, hue, type_p=selection4,
                           variable2=second_n)
        for i, fig in enumerate(figs):
            st.pyplot(fig)
            
def show_data_exploration_3D(ve):
    col1, col2, col3, col4 = st.columns(4)
    
    df = filter_dataframe(ve.df.copy(), key='exploration_3D')
    t_cols = sorted(VisualExploration(df).tot_columns)
    t_cols.insert(0, 'None')
    
    with col1:
        selection1 = st.selectbox("Eje 1:", t_cols)
    with col2:
        selection2 = st.selectbox("Eje 2", t_cols)
    with col3:
        selection3 = st.selectbox("Eje 3:", t_cols)
    with col4:
        selection4 = st.selectbox("Color:", t_cols)
    
        
    axis_1 = None if selection1 == 'None' else selection1
    axis_2 = None if selection2 == 'None' else selection2
    axis_3 = None if selection3 == 'None' else selection3
    axis_4 = None if selection4 == 'None' else selection4
    
    if axis_1 and axis_2 and axis_3 and axis_4:
        st.plotly_chart(VisualExploration(df).rotable_3d(axis_1, axis_2,
                                                         axis_3, axis_4))

def save_df(df):
    file = st.text_input("Escribe el nombre del archivo (sin .csv): ")
    path = st.text_input("Escribe la ruta de guardado (si no escribes nada" +
                         " se guardará en la carpeta donde este el archivo): ")
    if st.button("Guardar dataframe"):
        if file:
            file = file + '.csv'
            file = "./" + file if not path else os.path.join(path, file)
            file = file
            
            VisualExploration(df).save_df(file)
            st.success(f"Guardado en {file}.")
        else:
            st.error("Falta poner un nombre al archivo")
            
def initial_exploration(ve):
        options, data = st.columns([1, 4])
        
        with options:
            exploration_options = ['Tabla de datos',
                                   'Descripción',
                                   'Datos numéricos',
                                   'Datos categóricos']
            
            selected_exploration = st.radio('', exploration_options,
                                            key="init_explore")
        
        
        with data:
            
            if selected_exploration == exploration_options[0]:
                st.dataframe(filter_dataframe(ve.df.copy(),
                                              key='tab1',
                                              cut=True))
                
            if selected_exploration == exploration_options[1]:
                show_df_info(ve)
                
            if selected_exploration == exploration_options[2]:    
                show_numeric_columns(ve)
                
            if selected_exploration == exploration_options[3]:
                show_category_columns(ve)


def data_cleaning(ve):
    options, data, table = st.columns([1, 1, 2])
        
    with options:
        cleaning_options = ['Manejo de Valores Faltantes',
                            'Detección y Eliminación de Duplicados',
                            'Corrección de Tipos de Datos',
                            'Normalización y Estandarización',
                            'Manejo de Outliers',
                            'Corrección de Errores en Datos',
                            'Renombrado de Columnas']
        
        cleaning_functions = {
            cleaning_options[0]: edit_null_values,
            cleaning_options[1]: drop_duplicates,
            cleaning_options[2]: change_type_columns,
            cleaning_options[3]: normalize_and_standarize,
            cleaning_options[4]: handling_outliers,
            cleaning_options[5]: strings_correction,
            cleaning_options[6]: change_name_columns
            }
        
        selected_cleaning = st.radio('', cleaning_options,
                                     key="cleaning")
        
    with data:
        df = cleaning_functions[selected_cleaning](ve)    
            
    with table:
        show_df_with_rows_range(VisualExploration(df.copy()))
 
    return df.copy()


# Configurar el layout a "wide"
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Proyecto Examen")

# Verificar si el botón ha sido presionado
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False  # Inicializa el estado como no cargado

# Subir un archivo CSV
uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
sep = st.text_input('Separador (por defecto ";"):')

# Crear un botón para cargar el DataFrame
if st.button("Cargar DataFrame"):
    sep = sep if sep else ";"  # Usar ";" como separador por defecto
    if uploaded_file is not None:
        try:
            # Cargar el DataFrame
            df = pd.read_csv(uploaded_file, sep=sep)  # Cambia el separador si es necesario
            st.success("DataFrame cargado correctamente.")
            
            # Almacenar el DataFrame en el estado de sesión
            st.session_state.data_loaded = True
            st.session_state.df = df  # Almacenar el DataFrame en el estado de sesión

        except Exception as e:
            st.error(f"No se pudo cargar el DataFrame. Error: {e}")
    else:
        st.warning("Por favor, selecciona un archivo.")

# Mostrar pestañas solo si se ha cargado el DataFrame
if st.session_state.data_loaded:
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Exploración Inicial",
                                            "Limpieza de datos",
                                            "Visualización",
                                            "Visualización 3D",
                                            "Guardado"])

    with tab1:
        ve = VisualExploration(st.session_state.df)
        initial_exploration(ve)
        
    with tab2:
        ve = VisualExploration(st.session_state.df)
        st.session_state.df = data_cleaning(ve)
   
    with tab3:  
        ve = VisualExploration(st.session_state.df)
        show_data_exploration_graphs(ve)
        
    with tab4:
        ve = VisualExploration(st.session_state.df)
        show_data_exploration_3D(ve)
            
    with tab5: 
        save_df(st.session_state.df)

