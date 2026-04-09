from utils import db_connect
engine = db_connect()

# your code here


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split


#Imported Necessary Libraries // Librerias necesarias para el proyecto. 

# Load Database / Cargar base de datos.
df = pd.read_csv('/workspaces/Tutorial-Construye-un-modelo-de-regresion-lineal-usando-pandas-y-python/data/raw/demographic_health_data.csv')
total_data = df 
total_data.head()

print(total_data.columns.tolist())
# Print columns to select the target variable for the health study due to socioeconomics issues. / Imprimir las columnas para seleccionar la variable
# objetivo del estudio de salud en cuanto a aspectos socioeconomicos. 

#The chosen variable it is "anycondition_prevalence" (now used as target), because it resumes all possible health problems of the population / La 
# variable elegida es "anycondition_prevalence" (ahora usada como objetivo), porque resume todos los posibles problemas de salud de la población.

# Database features / Características de la base de datos
print(total_data.shape)
print(total_data.describe())
print(total_data.info())

#Eliminate duplicates / Eliminar duplicados. 
total_data[total_data.duplicated(keep=False)]
total_data = total_data.drop_duplicates()
total_data.shape

#Check for null values / Verificar valores nulos.
total_data.isnull().sum()

#Determinate irrelevant features / Determinar características irrelevantes.
# As we chose "anycondition_prevalence" as target variable, we will eliminate the features and variables linked to any ohter case. 
# Focusing on the socioconomis aspects. Also duplicates of information stored in different columns and ways. /
# Como elegimos "anycondition_prevalence" como variable objetivo, eliminaremos las características y variables vinculadas a cualquier otro caso.
# Enfocándonos en los aspectos socioconomicos. También duplicados de información almacenada en diferentes columnas y formas.

total_data = total_data.drop([ 'fips', 'STATE_FIPS', 'CNTY_FIPS'], axis=1) 

# No predictive features / Características no predictivas.

total_data = total_data.drop([
    '0-9', '19-Oct', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+',
    'White-alone pop', 'Black-alone pop',
    'Native American/American Indian-alone pop',
    'Asian-alone pop', 'Hawaiian/Pacific Islander-alone pop',
    'Two or more races pop',
    'Less than a high school diploma 2014-18',
    'High school diploma only 2014-18',
    "Some college or associate's degree 2014-18",
    "Bachelor's degree or higher 2014-18",
    'POVALL_2018',
    'Civilian_labor_force_2018', 'Employed_2018', 'Unemployed_2018',
    'Population Aged 60+', 'Total Population', 'POP_ESTIMATE_2018',], axis=1) 

# % version replaced features / Características reemplazadas por porcentajes.

total_data = total_data.drop([
    'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)',
    'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)',
    'Active General Surgeons per 100000 Population 2018 (AAMC)',
    'Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)',
    'MEDHHINC_2018',          
    'PCTPOV017_2018',          
    'PCTPOV517_2018',          
], axis=1)

#Almos perfect duplicates / Casi duplicados.

total_data = total_data.drop([
    'R_NATURAL_INC_2018',                    # = R_birth - R_death
    'R_NET_MIG_2018',                        # = R_DOM + R_INTL
    'N_POP_CHG_2018',
    'GQ_ESTIMATES_2018',
    'CI90LBINC_2018', 'CI90UBINC_2018',      # intervalos de confianza del ingreso
    'Med_HH_Income_Percent_of_State_Total_2018',
    'county_pop2018_18 and older',
], axis=1)

# Derived by another variables or redundant metrics./ Metricas derivadas de otras variables o redundantes.

total_data = total_data.drop([
    'R_birth_2018',
    'R_DOMESTIC_MIG_2018',
    'R_INTERNATIONAL_MIG_2018',
], axis=1)

# Irrelevant for the target variable / Irrelevantes para la variable objetivo.

total_data = total_data.drop([
    'R_death_2018',
    'anycondition_Lower 95% CI', 'anycondition_Upper 95% CI', 'anycondition_number',
    'Obesity_prevalence', 'Obesity_Lower 95% CI', 'Obesity_Upper 95% CI', 'Obesity_number',
    'Heart disease_prevalence', 'Heart disease_Lower 95% CI',
    'Heart disease_Upper 95% CI', 'Heart disease_number',
    'COPD_prevalence', 'COPD_Lower 95% CI', 'COPD_Upper 95% CI', 'COPD_number',
    'diabetes_prevalence', 'diabetes_Lower 95% CI',
    'diabetes_Upper 95% CI', 'diabetes_number',
    'CKD_prevalence', 'CKD_Lower 95% CI', 'CKD_Upper 95% CI', 'CKD_number',
], axis=1)

# Cronic conditions correlated to the target variable, but not useful for the model. / 
# Condiciones crónicas correlacionadas con la variable objetivo, pero no útiles para el modelo.

total_data.info()
print(f"\nDimensiones: {total_data.shape}")   
print(total_data.head())


fig, axis = plt.subplots(1, 1, figsize=(8, 5))

# Urban-rural classification is the main categorical variable / La clasificación urbano-rural es la variable categórica principal
sns.histplot(ax = axis, data = total_data, x = "Urban_rural_code")

# Adjust layout / Ajustar el diseño
plt.tight_layout()

# Show the plot / Mostrar el gráfico
plt.show()

fig, axis = plt.subplots(4, 2, figsize = (12, 16), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

sns.histplot(ax = axis[0, 0], data = total_data, x = "anycondition_prevalence")
sns.boxplot(ax = axis[1, 0],  data = total_data, x = "anycondition_prevalence")

sns.histplot(ax = axis[0, 1], data = total_data, x = "PCTPOVALL_2018")
sns.boxplot(ax = axis[1, 1],  data = total_data, x = "PCTPOVALL_2018")

sns.histplot(ax = axis[2, 0], data = total_data, x = "Median_Household_Income_2018")
sns.boxplot(ax = axis[3, 0],  data = total_data, x = "Median_Household_Income_2018")

sns.histplot(ax = axis[2, 1], data = total_data, x = "Unemployment_rate_2018")
sns.boxplot(ax = axis[3, 1],  data = total_data, x = "Unemployment_rate_2018")

# Adjust layout / Ajustar el diseño
plt.tight_layout()

# Show the plot / Mostrar el gráfico
plt.show()

# Numerical - Numerical Analysis / Análisis numérico - numérico

# Create subplot canvas / Crear lienzo de subgráficos
fig, axis = plt.subplots(4, 2, figsize = (12, 18))

# Create Plots / Crear gráficos
sns.regplot(ax = axis[0, 0], data = total_data, x = "PCTPOVALL_2018", y = "anycondition_prevalence")
sns.heatmap(total_data[["anycondition_prevalence", "PCTPOVALL_2018"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = total_data, x = "Median_Household_Income_2018", y = "anycondition_prevalence").set(ylabel = None)
sns.heatmap(total_data[["anycondition_prevalence", "Median_Household_Income_2018"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = total_data, x = "Unemployment_rate_2018", y = "anycondition_prevalence").set(ylabel = None)
sns.heatmap(total_data[["anycondition_prevalence", "Unemployment_rate_2018"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)

sns.regplot(ax = axis[2, 1], data = total_data, x = "Active Physicians per 100000 Population 2018 (AAMC)", y = "anycondition_prevalence").set(ylabel = None)
sns.heatmap(total_data[["anycondition_prevalence", "Active Physicians per 100000 Population 2018 (AAMC)"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 1]).set(ylabel = None)

# Adjust layout / Ajustar el diseño
plt.tight_layout()

# Show the plot / Mostrar el gráfico
plt.show()

# Map states to US Census regions / Mapear estados a regiones censales de EE.UU.
census_region_map = {
    'Alabama': 'South',     'Alaska': 'West',         'Arizona': 'West',         'Arkansas': 'South',
    'California': 'West',   'Colorado': 'West',        'Connecticut': 'Northeast', 'Delaware': 'South',
    'Florida': 'South',     'Georgia': 'South',        'Hawaii': 'West',           'Idaho': 'West',
    'Illinois': 'Midwest',  'Indiana': 'Midwest',      'Iowa': 'Midwest',          'Kansas': 'Midwest',
    'Kentucky': 'South',    'Louisiana': 'South',      'Maine': 'Northeast',       'Maryland': 'South',
    'Massachusetts': 'Northeast', 'Michigan': 'Midwest', 'Minnesota': 'Midwest',  'Mississippi': 'South',
    'Missouri': 'Midwest',  'Montana': 'West',         'Nebraska': 'Midwest',      'Nevada': 'West',
    'New Hampshire': 'Northeast', 'New Jersey': 'Northeast', 'New Mexico': 'West', 'New York': 'Northeast',
    'North Carolina': 'South', 'North Dakota': 'Midwest', 'Ohio': 'Midwest',      'Oklahoma': 'South',
    'Oregon': 'West',       'Pennsylvania': 'Northeast', 'Rhode Island': 'Northeast', 'South Carolina': 'South',
    'South Dakota': 'Midwest', 'Tennessee': 'South',   'Texas': 'South',           'Utah': 'West',
    'Vermont': 'Northeast', 'Virginia': 'South',       'Washington': 'West',       'West Virginia': 'South',
    'Wisconsin': 'Midwest', 'Wyoming': 'West'
}
total_data['census_region'] = total_data['STATE_NAME'].map(census_region_map)

fig, axis = plt.subplots(figsize = (10, 5))

sns.countplot(data = total_data, x = "Urban_rural_code", hue = "census_region")

# Show the plot / Mostrar el gráfico
plt.show()


# Factorize the categorical variables / Factorizar las variables categóricas
total_data["COUNTY_NAME"]    = pd.factorize(total_data["COUNTY_NAME"])[0]
total_data["STATE_NAME"]     = pd.factorize(total_data["STATE_NAME"])[0]
total_data["census_region"]  = pd.factorize(total_data["census_region"])[0]

# Select key variables to keep the heatmap readable / Seleccionar variables clave para mantener el mapa de calor legible
key_vars = [
    "anycondition_prevalence",
    "PCTPOVALL_2018", "Median_Household_Income_2018", "Unemployment_rate_2018",
    "Percent of adults with less than a high school diploma 2014-18",
    "Percent of adults with a bachelor's degree or higher 2014-18",
    "Percent of Population Aged 60+",
    "% White-alone", "% Black-alone",
    "Active Physicians per 100000 Population 2018 (AAMC)",
    "Active Primary Care Physicians per 100000 Population 2018 (AAMC)",
    "Total Hospitals (2019)", "ICU Beds_x",
    "Urban_rural_code", "census_region",
]

fig, axes = plt.subplots(figsize = (14, 12))

sns.heatmap(total_data[key_vars].corr(), annot = True, fmt = ".2f", annot_kws = {"size": 8})

plt.tight_layout()

# Draw Plot / Mostrar el gráfico
plt.show()


# Pairplot restricted to the most informative variables / Pairplot restringido a las variables más informativas
pairplot_vars = [
    "anycondition_prevalence",
    "PCTPOVALL_2018",
    "Median_Household_Income_2018",
    "Unemployment_rate_2018",
    "Percent of Population Aged 60+",
    "Active Physicians per 100000 Population 2018 (AAMC)",
    "Percent of adults with less than a high school diploma 2014-18",
]

sns.pairplot(data = total_data[pairplot_vars])

total_data.describe()


fig, axes = plt.subplots(2, 3, figsize = (15, 10))

sns.boxplot(ax = axes[0, 0], data = total_data, y = "anycondition_prevalence")
sns.boxplot(ax = axes[0, 1], data = total_data, y = "PCTPOVALL_2018")
sns.boxplot(ax = axes[0, 2], data = total_data, y = "Median_Household_Income_2018")
sns.boxplot(ax = axes[1, 0], data = total_data, y = "Unemployment_rate_2018")
sns.boxplot(ax = axes[1, 1], data = total_data, y = "Active Physicians per 100000 Population 2018 (AAMC)")
sns.boxplot(ax = axes[1, 2], data = total_data, y = "Percent of Population Aged 60+")

plt.tight_layout()

plt.show()


# Stats for anycondition_prevalence / Estadísticas para anycondition_prevalence
anycondition_stats = total_data["anycondition_prevalence"].describe()
anycondition_stats

# IQR for anycondition_prevalence / Rango intercuartílico para anycondition_prevalence

anycondition_iqr = anycondition_stats["75%"] - anycondition_stats["25%"]
upper_limit = anycondition_stats["75%"] + 1.5 * anycondition_iqr
lower_limit = anycondition_stats["25%"] - 1.5 * anycondition_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(anycondition_iqr, 2)}")

# Clean the outliers / Limpiar los outliers

total_data = total_data[total_data["anycondition_prevalence"] > 0]

# Stats for PCTPOVALL_2018 / Estadísticas para PCTPOVALL_2018
pctpov_stats = total_data["PCTPOVALL_2018"].describe()
pctpov_stats

# IQR for PCTPOVALL_2018 / Rango intercuartílico para PCTPOVALL_2018

pctpov_iqr = pctpov_stats["75%"] - pctpov_stats["25%"]
upper_limit = pctpov_stats["75%"] + 1.5 * pctpov_iqr
lower_limit = pctpov_stats["25%"] - 1.5 * pctpov_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(pctpov_iqr, 2)}")

# Clean the outliers / Limpiar los outliers

total_data = total_data[total_data["PCTPOVALL_2018"] <= upper_limit]

# Stats for Median_Household_Income_2018 / Estadísticas para Median_Household_Income_2018

income_stats = total_data["Median_Household_Income_2018"].describe()
income_stats

# IQR for Median_Household_Income_2018 / Rango intercuartílico para Median_Household_Income_2018

income_iqr = income_stats["75%"] - income_stats["25%"]
upper_limit = income_stats["75%"] + 1.5 * income_iqr
lower_limit = income_stats["25%"] - 1.5 * income_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(income_iqr, 2)}")

# Stats for Unemployment_rate_2018 / Estadísticas para Unemployment_rate_2018

unemp_stats = total_data["Unemployment_rate_2018"].describe()
unemp_stats

# IQR for Unemployment_rate_2018 / Rango intercuartílico para Unemployment_rate_2018

unemp_iqr = unemp_stats["75%"] - unemp_stats["25%"]
upper_limit = unemp_stats["75%"] + 1.5 * unemp_iqr
lower_limit = unemp_stats["25%"] - 1.5 * unemp_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(unemp_iqr, 2)}")

# Count NaN values / Contar valores NaN
total_data.isnull().sum().sort_values(ascending = False)

from sklearn.preprocessing import MinMaxScaler

num_variables = [col for col in total_data.columns if col != "anycondition_prevalence"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(total_data[num_variables])
df_scal = pd.DataFrame(scal_features, index = total_data.index, columns = num_variables)
df_scal["anycondition_prevalence"] = total_data["anycondition_prevalence"]
df_scal.head()

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scal.drop("anycondition_prevalence", axis = 1)
y = df_scal["anycondition_prevalence"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)
# chi2 needs: (1) non-negative X, (2) categorical y
scaler = MinMaxScaler()
X_train_pos = scaler.fit_transform(X_train)
X_test_pos  = scaler.transform(X_test)
y_train_bins = pd.qcut(y_train, q = 5, labels = False, duplicates = "drop")
selection_model = SelectKBest(chi2, k = 10)
selection_model.fit(X_train_pos, y_train_bins)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(
    selection_model.transform(X_train_pos), columns = X_train.columns.values[ix]
)
X_test_sel = pd.DataFrame(
    selection_model.transform(X_test_pos), columns = X_train.columns.values[ix]
)
X_train_sel.head()

# Load train and test datasets / Cargar los conjuntos de datos de entrenamiento y prueba
train_data = pd.read_csv("../data/processed/clean_train.csv")
test_data  = pd.read_csv("../data/processed/clean_test.csv")

train_data.head()


# Visualization of features vs anycondition_prevalence / Visualización de características vs anycondition_prevalence
total_data_lr = pd.concat([train_data, test_data])

selected_features = [col for col in train_data.columns if col != "anycondition_prevalence"]
n = len(selected_features)
fig, axis = plt.subplots(n * 2, 1, figsize = (10, n * 4))

for i, feature in enumerate(selected_features):
    sns.regplot(ax = axis[i * 2],     data = total_data_lr, x = feature, y = "anycondition_prevalence")
    sns.heatmap(total_data_lr[["anycondition_prevalence", feature]].corr(),
                annot = True, fmt = ".2f", ax = axis[i * 2 + 1], cbar = False)

plt.tight_layout()
plt.show()

# Split features and target variable / Separar características y variable objetivo

X_train = train_data.drop(["anycondition_prevalence"], axis = 1)
y_train = train_data["anycondition_prevalence"]
X_test  = test_data.drop(["anycondition_prevalence"], axis = 1)
y_test  = test_data["anycondition_prevalence"]

# Model training / Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters / Parámetros del modelo

print(f"Intercepto (a): {model.intercept_}")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"Coeficiente {feature}: {coef}")

    # Predictions / Predicciones

y_pred = model.predict(X_test)
y_pred

# Model evaluation / Evaluación del modelo

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")


# As R² = 0.6852939531042808, the value is very close to the acceptable between 0.7 & 1.0. Still a trustable model. / 
# Como R² = 0,6852939531042808, el valor está muy cerca del rango aceptable entre 0,7 y 1,0. Sigue siendo un modelo fiable. 


