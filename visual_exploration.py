# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:45:07 2024

@author: Imanol
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
import plotly.graph_objs as go
from matplotlib.colors import to_hex
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class VisualExploration():
    mi_paleta = {'Navy Blue': '#1b2e3c',
                 'Crimson': '#4b0000',
                 'Black': '#0c0c1e',
                 'Cream': '#f3e3e2'}
    data_types = []

    def __init__(self, df):
        self.df = self.process_df(df)
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
    
    @staticmethod
    def process_df(df):
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        df.columns = df.columns.str.strip()
        return df
      
    def split_df(self):
        self.df_num = self.df.select_dtypes(include="number")
        self.df_cat = self.df.select_dtypes(exclude="number")
        self.num_columns = self.df_num.columns.tolist()
        self.cat_columns = self.df_cat.columns.tolist()
        
    def show_info(self):
        self.df.info()
        
    def numerical_description(self, t_num=False):
        # Descripción de columnas numéricas
        des_num = self.df_num.describe().T
        des_num["Tipos"] = self.df_num.dtypes
        des_num["Nulos"] = self.df_num.isna().sum()
        des_num["Únicos"] = self.df_num.nunique()
        des_num = des_num.T if t_num else des_num

        return des_num
      
        
    def categorical_description(self, t_cat=False):
        # Descripción de columnas categóricas
        des_cat = self.df_cat.describe(include="all").T
        des_cat["Tipos"] = self.df_cat.dtypes
        des_cat["Nulos"] = self.df_cat.isna().sum()
        des_cat = des_cat.T if t_cat else des_cat
            
        return des_cat
            
    def show_describe(self, t_num=False, t_cat=False):
        """Mostrar descripción de los datos divididos en categoricos
        y numéricos
        """
        # Mostrar tamaño del DataFrame
        print("** Datos del DataFrame **")
        print(f"El DataFrame tiene {self.n_rows} filas y {self.n_columns} columnas.\n")

        # Mostrar las columnas
        print("** Columnas numéricas **")
        print(f"{self.num_columns}\n")
        
        print("** Columnas categóricas **")
        print(f"{self.cat_columns}\n")
        
        print("** Descripción de las columnas numéricas **")
        display(self.numerical_description(t_num))
        print("\n")
        
        print("** Descripción de las columnas categóricas **")
        display(self.categorical_description(t_cat))
        print("\n")

    def show_head(self, n=5):
        """Mostrar las primeras n filas del DataFrame"""
        print(f"** Primeras {n} filas del DataFrame **")
        return self.df.head(n)

    def show_tail(self, n=5):
        """Mostrar las ultimas n filas del DataFrame"""
        print(f"** Ultimas {n} filas del DataFrame **")
        return self.df.tail(n)

    def _configure_plot_simple(self, name, traspuesta, variable, axes, idx):

        mi_paleta = VisualExploration.mi_paleta

        plot_axe = 'y' if traspuesta else 'x'

        if name == 'histograma':
            plot_config = {plot_axe: self.df_num[variable],
                            'kde':True,
                            'color': mi_paleta['Navy Blue'],
                            'edgecolor': mi_paleta['Black'],
                            'ax':axes[idx]}
            sns.histplot(**plot_config)

        if name == 'boxplot':
            plot_config = {plot_axe: self.df_num[variable],
                           'color': mi_paleta['Black'],
                           'whiskerprops':{'color': mi_paleta['Cream'],
                                            'linewidth': 1},
                           'capprops':{'color': mi_paleta['Cream'], 'linewidth': 1},
                           'medianprops':{'color': mi_paleta['Cream'], 'linewidth': 1},
                           'flierprops':{
                               'marker': 'o',
                               'markerfacecolor': mi_paleta['Cream'],
                               'markersize': 6,
                               'markeredgecolor': mi_paleta['Black'],
                               'markeredgewidth': 0.5},
                           'ax':axes[idx]
                            }
            sns.boxplot(**plot_config)

        if name == 'violinplot':
            plot_config = {plot_axe: self.df_num[variable],
                           'color': mi_paleta['Navy Blue'],
                           'inner_kws':dict(box_width=15, whis_width=2,
                                 color=mi_paleta['Black']),
                           'ax':axes[idx]
                           }
            sns.violinplot(**plot_config)

        if name == 'value_counts':
            plot_config = {plot_axe:variable, 'data':self.df_cat,
                           'color': mi_paleta['Black'], 'ax':axes[idx]}
            sns.countplot(**plot_config)
            
        if name == 'donut':
            etiquetas = self.df_cat[variable].unique().tolist()
            conteos = self.df_cat[variable].value_counts()

            # Gráfico de pastel
            plot_config = {'x': conteos, 'labels': etiquetas,
                           'colors':sns.color_palette("bright"),
                           'textprops': {'color': mi_paleta['Black'],
                                         'fontsize':8}}
            axes[idx].pie(**plot_config)

            # Círculo central para el donut
            circulo_central = plt.Circle((0, 0), 0.7, color=mi_paleta['Crimson'])
            axes[idx].add_artist(circulo_central)

            # Leyenda
            #axes[idx].legend(labels=etiquetas, loc='upper left')
            
        if name == 'facetgrid':
            axes[idx] = sns.FacetGrid(self.df, col=self.category)
            axes[idx].map(sns.histplot, variable)

    def _format_axes(self, axes, variable, idx, traspuesta, plot_type):
        mi_paleta = VisualExploration.mi_paleta

        # Ajustar fondo
        axes[idx].set_facecolor(mi_paleta['Crimson'])

        # Borrar ejes
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['left'].set_visible(False)
        axes[idx].spines['bottom'].set_visible(False)

        # Personalizar ticks
        axes[idx].tick_params(bottom=False, left=False)
        axes[idx].tick_params(axis='y', colors=mi_paleta['Cream'], labelsize=8)
        axes[idx].tick_params(axis='x', colors=mi_paleta['Cream'], labelsize=8)

        # Personalizar labels
        x, y = ('', variable) if traspuesta else (variable, '')
        axes[idx].set_ylabel(y, fontsize=10, color=mi_paleta['Cream'], fontstyle='italic')
        axes[idx].set_xlabel(x, fontsize=10, color=mi_paleta['Cream'], fontstyle='italic')

        # Cambiamos el valor de la traspuesta si las gaficas son violin o box
        traspuesta = not traspuesta if plot_type in ['violinplot', 'boxplot'] else traspuesta

        # Eliminar el grid o añadirlo
        axes[idx].set_axisbelow(True)
        if traspuesta:
            axes[idx].xaxis.grid(True, color=mi_paleta['Navy Blue'], linewidth=1)
            axes[idx].yaxis.grid(False)
        else:
            axes[idx].xaxis.grid(False)
            axes[idx].yaxis.grid(True, color=mi_paleta['Navy Blue'], linewidth=1)

    def _finalize_plot(self, suptitle):
        mi_paleta = VisualExploration.mi_paleta

        plt.suptitle(suptitle, fontsize=16, weight='bold', color=mi_paleta['Cream'], x=0.5, y=0.95)

        plt.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

    def _show_plot(self, traspuesta, suptitle, type_name, cols):
        """Mostrar el tipo de gráfica solicitada"""
        categoricals = ['value_counts', 'donut']
 
        data = self.df_cat if type_name in categoricals else self.df_num
        n = len(data.columns)
        n_cols = min(4, n)
        if cols:
            n_cols = cols
        n_rows = int(np.ceil(n / n_cols))
        
        if n_cols == 1 and n_rows == 1:
            fig, axes = plt.subplots(2, 1, figsize=(15, 5 * n_rows))
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
        axes = axes.ravel()

        # Fondo de la figura color personalizado
        fig.patch.set_facecolor(VisualExploration.mi_paleta['Crimson'])

        for idx, variable in enumerate(data.columns):
            self._configure_plot_simple(type_name, traspuesta, variable, axes, idx)
            self._format_axes(axes, variable, idx, traspuesta, type_name)

        for ax in axes[n:]:
            ax.set_visible(False)  # Ocultar ejes adicionales

        #self._finalize_plot(suptitle)
        return fig

    def show_histograms(self, traspuesta=False, cols=None):
        return self._show_plot(traspuesta, "Frecuencia de Variables Numéricas",
                        "histograma", cols)

    def show_boxplots(self, traspuesta=True, cols=None):
        return self._show_plot(traspuesta, "Frecuencia de Variables Numéricas",
                        "boxplot", cols)

    def show_violinplots(self, traspuesta=True, cols=None):
        return self._show_plot(traspuesta, "Frecuencia de Variables Numéricas",
                        "violinplot", cols)

    def show_value_counts(self, traspuesta=False, cols=None):
        return self._show_plot(traspuesta, "Frecuencia de Variables Categóricas",
                        "value_counts", cols)

    def show_heatmap(self):
        """Mostrar mapa de calor de correlación"""
        mi_paleta = VisualExploration.mi_paleta

        # Crear una colormap personalizada con los colores deseados
        colores = [mi_paleta['Navy Blue'], mi_paleta['Black']]
        mi_cmap = LinearSegmentedColormap.from_list('mi_cmap', colores)

        corr = self.df_num.corr()

        fig, axes = plt.subplots()

        # Fondo de la figura color personalizado
        fig.patch.set_facecolor(VisualExploration.mi_paleta['Crimson'])

        heatmap = sns.heatmap(corr, cmap=mi_cmap, ax=axes)

        # Cambiar el color de los ticks
        axes.tick_params(left=False, bottom=False)
        plt.xticks(color=mi_paleta['Cream'], fontstyle='italic', fontsize=4)
        plt.yticks(color=mi_paleta['Cream'], fontstyle='italic', fontsize=4)

        # Acceder a la barra de color
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors=mi_paleta['Cream'], labelsize=4)

        #self._finalize_plot("Matriz de Correlación")
        
        return fig
        
    def show_donuts(self, cols=None):    
        return self._show_plot(False, "Distribución de Variables Categóricas",
                        "donut", cols)
        
    def show_pairplot(self):
        """Mostrar mapa de calor de correlación"""
        mi_paleta = VisualExploration.mi_paleta

        p = sns.pairplot(self.df, corner=True,
                 plot_kws={'color': mi_paleta['Black']},
                 diag_kws={'color': mi_paleta['Black']})

        p.fig.patch.set_facecolor(mi_paleta['Crimson'])
        for ax in p.axes.flatten():
            if ax is not None:
                ax.set_facecolor(mi_paleta['Crimson'])
                
                ax.tick_params(bottom=False, left=False)
                ax.tick_params(axis='y', colors=mi_paleta['Cream'],
                               labelsize=6)
                ax.tick_params(axis='x', colors=mi_paleta['Cream'],
                               labelsize=6)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                
                ax.xaxis.label.set_color(mi_paleta['Cream'])
                ax.yaxis.label.set_color(mi_paleta['Cream'])
                ax.xaxis.label.set_fontstyle('italic')
                ax.yaxis.label.set_fontstyle('italic')

        return p
        
    def show_facet_grid(self, rows, category, hue=None, type_p='histogram',
                        variable2=None):
        mi_paleta = VisualExploration.mi_paleta
        
        fig = []
        
        title =  True
        for (idx,variable) in enumerate(self.df[rows]):
            
            options = {'histogram': {'func':sns.histplot, 'args':[variable]},
                       'scatterplot': {'func':sns.scatterplot, 'args':[variable,
                                                                    variable2]},
                       'kdeplot': {'func':sns.kdeplot, 'args':[variable]}}
            
            g = sns.FacetGrid(self.df, col=category, hue=hue, palette=sns.dark_palette("#79C"),
                              height=4, aspect=1, dropna=False)
            g.fig.patch.set_facecolor(mi_paleta['Crimson'])
            
            g.map(options[type_p]['func'], *options[type_p]['args'])
        
            # Personalizar cada eje dentro del FacetGrid
            for ax in g.axes.flat:
                # Cambiar el color de fondo
                ax.set_facecolor(mi_paleta['Crimson'])
                
                # Cambiar el color de los ticks
                ax.tick_params(colors=mi_paleta['Cream'])
                
                # Cambiar el color de las etiquetas de los ejes
                ax.set_xlabel(ax.get_xlabel(), color=mi_paleta['Cream'])
                ax.set_ylabel('', color=mi_paleta['Cream'])
        
                if title:
                    ax.set_title(ax.get_title(), color=mi_paleta['Cream'])
                else:
                    ax.set_title('', color=mi_paleta['Cream'])
        
            title = False
            
            if hue:
                  g.add_legend(title=hue)
            
            fig.append(g)
            
        return fig
        
    @staticmethod
    def cat_2_num(col):
        if VisualExploration.is_categorical(col.dtypes):
            new_col = col.astype('category')
            new_col = col.cat.codes
            return new_col
        
    # Función para pintar las reglas en 3D
    def rotable_3d(self, col1, col2, col3, cat_col):
        datos = self.df.copy()
        # Reseteo el índice de los datos originales
        datos.reset_index(inplace=True)
        
        datos[[col1, col2, col3, cat_col]].apply(self.cat_2_num)
        
    
        # Crear el scatter plot en 3D con Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=datos[col1],
            y=datos[col2],
            z=datos[col3],
            mode='markers',
            marker=dict(
                size=10,
                color=datos[cat_col],  # Color según el valor de la categoría
                colorscale='Blues',  # Escala de color
                opacity=0.8,
                colorbar=dict(  # Configuración de la barra de colores
                    title='Valores',  # Título de la barra de colores
                    titleside='right',  # Lado del título
                    tickvals=[],  # Puedes añadir valores específicos para ticks aquí
                    ticktext=[],  # Puedes añadir texto para ticks aquí
                    )
            ),
            text='<br>' + \
                 col1 + ": " + datos[col1].astype(str) + '<br>' + \
                 col2 + ": " + datos[col2].astype(str) + '<br>' + \
                 col3 + ": " + datos[col3].astype(str),
            hoverinfo='text'  # Mostrar texto en el menú emergente
        )])
    
        # Configuración del diseño del gráfico
        fig.update_layout(
            scene=dict(
                xaxis_title=col1,
                yaxis_title=col2,
                zaxis_title=col3,
            ),
            title='Scatter Plot 3D',
            width=800,
            height=1200,
        )
        
        return fig
        
    @staticmethod
    def is_categorical(var_type):
        return var_type in ('object', 'category', 'string', 'datetime')
        
        
    @staticmethod
    def adjust_col(col, new_type):
        if (VisualExploration.is_categorical(col.dtypes) and 
            not VisualExploration.is_categorical(new_type)):
            col = col.astype('category').cat.codes
        return col  
        
    def change_type(self, cols, new_type, new_categories=None, date_format=None):
        type_map = {
            'int': lambda x: pd.to_numeric(x, errors='coerce').astype('int64'),
            'float': lambda x: pd.to_numeric(x, errors='coerce').astype('float'),
            'category': lambda x: x.astype('category'),
            'datetime': lambda x: pd.to_datetime(x, errors='coerce',
                                                 format = date_format),
            'string': lambda x: x.astype('string')
            }
        
        
        try:
            data = self.df[cols].copy()
            print(data)
            data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            data = data.apply(lambda x: VisualExploration.adjust_col(x, new_type))
            print(cols)
            print(data[cols])
                
            data = data.apply(lambda col: type_map[new_type](col))
            self.df[cols] = data.copy()
            if new_type == 'category' and new_categories:
                col_cat = pd.DataFrame()
                col_cat[cols] = self.df[cols].copy()
                self.df[cols] = pd.Categorical(col_cat[cols]).rename_categories(new_categories)
            self.split_df()
        except KeyError:
            raise KeyError(f"Tipo '{new_type}' no soportado para la columna '{cols}'.")    
        except Exception as e:
                raise Exception(f"Error al convertir la columna '{cols}': {e}")      
        
    def change_name_col(self, col, new_name):
        self.df.rename(columns={col:new_name}, inplace=True)
        self.split_df()
        
    def save_df(self, path):
        self.df.to_csv(path, index=False, sep=';')
        
    def input_vals(self, cols, select):     
        imputations = {
        0: lambda col: self.df[col].fillna(self.df[col].mean()),     # Media
        1: lambda col: self.df[col].fillna(self.df[col].median()),   # Mediana
        2: lambda col: self.df[col].fillna(self.df[col].mode()[0]),  # Moda
        3: lambda col: self.df[col].fillna(method='ffill'),          # Valor anterior
        4: lambda col: self.df[col].fillna(method='bfill')           # Valor posterior
        }
        
        if select in imputations:
            for col in cols:
                self.df[col] = imputations[select](col)
        else:
            raise ValueError("Valor de select no válido. Debe ser 0, 1, 2, 3 o 4.")
            
    def drop_na(self, axis, cols=None):
        if axis==0 or (axis==1 and not cols):
            self.df.dropna(axis=axis, subset=cols, inplace=True)
        
        if axis==1 and cols:
            cols = cols if isinstance(cols, list) else [cols]
            for col in cols:
                if self.df[col].isnull().any():
                    self.df.drop(col, axis=1, inplace=True)
                
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
        
    def drop_duplicates(self, cols=None):
        if cols:
            cols = cols if isinstance(cols, list) else [cols]
        self.df.drop_duplicates(subset=cols, inplace=True)
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
        
    def show_duplicates(self, cols=None, keep='first'):
        df = self.df.copy()
        return df[df.duplicated(subset=cols, keep=keep)]
        
        
    def delete_columns(self, cols):
        if cols != None and len(cols) == 1:
            aux_cols = []
            aux_cols.append(cols)
            cols = aux_cols.copy()
        self.df.drop(columns=cols, inplace=True)
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
        
    def standarize_columns(self, cols):
        scaler = StandardScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])
        
    def normalize_columns(self, cols):
        normalizer = MinMaxScaler()
        self.df[cols] = normalizer.fit_transform(self.df[cols])
        
    def return_column_outliers(self, col):
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
        outliers = self.df[outlier_condition]
        
        return outliers[col], outlier_condition
        
    def delete_column_outliers(self, col):
        _, outlier_condition, = self.return_column_outliers(col)
        self.df = self.df[~outlier_condition]
        
    def winsorizing_column_outliers(self, col):
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        self.df[col] = np.where(self.df[col] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR,
                                self.df[col])
        self.df[col] = np.where(self.df[col] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR,
                                self.df[col])
    
    def return_string_columns(self):
        return self.df.select_dtypes(include=['object', 'string'])
        
    def string_strip(self, col):
        self.df[col] = self.df[col].str.strip()
        
    def string_upper(self, col):
        self.df[col] = self.df[col].str.upper()
    
    def string_lower(self, col):
        self.df[col] = self.df[col].str.lower()
        
    def string_replace(self, col, old_string, new_string):
        self.df[col] = self.df[col].str.replace(old_string, new_string)

if __name__ == "__main__":
    df = pd.read_csv("data.csv", sep=";")
    ve = VisualExploration(df)
    
    print(ve.df['Target'].unique())
    ve.string_replace('Target', 'out', 'free')
    print(ve.df['Target'].unique())
    print(df.columns.str.strip())
