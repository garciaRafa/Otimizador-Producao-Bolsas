import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(layout="wide")
st.title("💰 Otimizador de Lucro de Produção de Bolsas")
st.markdown("---")

# ----------------------------------------------------------------------
# 1. FUNÇÃO DE CÁLCULO DE TEMPO E OTIMIZAÇÃO
# ----------------------------------------------------------------------

TEMPOS_POR_CATEGORIA = {
    'G': 0.45,  # 9h / 20 bolsas
    'M': 0.40,  # 8h / 20 bolsas
    'P': 0.30   # 6h / 20 bolsas
}

def calcular_e_otimizar(df_modelos, horas_total, arredondar_resultado):
    # 1. Preparar Lucro (Vetor C)
    df_modelos['Lucro Unitário (R$)'] = df_modelos['Preço de Venda (R$)'] - df_modelos['Preço de Custo (R$)']
    
    # Linprog MINIMIZA, então invertemos o sinal do Lucro para MAXIMIZAR
    lucros_c = -(df_modelos['Lucro Unitário (R$)']).values
    
    # 2. Preparar Restrição de Tempo (Matriz A_ub e b_ub)
    # Calcula o tempo unitário com base na Categoria
    df_modelos['Tempo por Unidade (h)'] = df_modelos['Categoria'].map(TEMPOS_POR_CATEGORIA).fillna(0)
    
    A_ub = df_modelos['Tempo por Unidade (h)'].values.reshape(1, -1) # Coeficientes do Tempo
    b_ub = np.array([horas_total]) # Limite Total de Horas

    # 3. Preparar Limites de Produção (Bounds)
    # Combina Mínimo e Máximo de Venda em uma lista de tuplas [(min1, max1), (min2, max2), ...]
    bounds = df_modelos[['Mín. Venda (Unidades)', 'Máx. Venda (Unidades)']].values
    # Converte para o formato de tuplas que o SciPy espera
    bounds_list = [tuple(b) for b in bounds]
    
    # 4. Executar a Otimização
    try:
        resultado = linprog(
            c=lucros_c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            bounds=bounds_list, 
            method='highs'
        )
    except ValueError as e:
        return None, f"Erro nos dados de entrada: {e}"
    
    # 5. Processar Resultados
    if resultado.success:
        lucro_maximo = -resultado.fun
        quantidades = resultado.x
        
        if arredondar_resultado:
            quantidades = np.round(quantidades).astype(int)
            # RE-CALCULAR o lucro máximo com as quantidades arredondadas
            lucro_maximo = np.dot(quantidades, df_modelos['Lucro Unitário (R$)'].values)
            
        tempo_usado = np.dot(df_modelos['Tempo por Unidade (h)'].values, quantidades)

        df_modelos['Produção Ideal (Unidades)'] = quantidades
        df_modelos['Tempo Total (h)'] = df_modelos['Tempo por Unidade (h)'] * quantidades

        return df_modelos, lucro_maximo, tempo_usado, resultado.message
    else:
        return None, resultado.message


# ----------------------------------------------------------------------
# 2. INTERFACE STREAMLIT
# ----------------------------------------------------------------------

# 2.1 Configuração Inicial da Tabela (Modelos Padrão)
dados_iniciais = {
    'Modelo': ['B1 G', 'B2 M', 'B1 P', 'Novo Modelo'],
    'Categoria': ['G', 'M', 'P', 'G'],
    'Preço de Venda (R$)': [45.58, 37.18, 23.10, 50.00],
    'Preço de Custo (R$)': [26.81, 21.87, 13.59, 25.00],
    'Mín. Venda (Unidades)': [18, 16, 12, 10],
    'Máx. Venda (Unidades)': [36, 28, 24, 50]
}
df_base = pd.DataFrame(dados_iniciais)

# 2.2 Sidebar (Recursos Globais)
with st.sidebar:
    st.header("⚙️ Recursos e Configurações")
    horas_total = st.number_input(
        "Total de Horas de Trabalho Disponíveis (Mês)", 
        min_value=1.0, 
        value=300.0, 
        step=1.0,
        format="%f"
    )
    arredondar_resultado = st.checkbox(
        "Arredondar produção para números inteiros", 
        value=True, 
        help="A produção ideal (x) será arredondada para o inteiro mais próximo. O lucro será recalculado."
    )
    st.markdown("---")


# 2.3 Área Principal (Edição dos Modelos)
st.subheader("👜 Edição dos Parâmetros de Produção")
st.info("⚠️ Edite os valores abaixo para definir os parâmetros e limites de cada modelo. Adicione novas linhas se necessário.")

# Configuração de Colunas para o Data Editor
column_config = {
    "Categoria": st.column_config.SelectboxColumn(
        "Categoria",
        options=list(TEMPOS_POR_CATEGORIA.keys()),
        required=True,
    ),
    "Preço de Venda (R$)": st.column_config.NumberColumn(format="R$ %.2f", min_value=0.01),
    "Preço de Custo (R$)": st.column_config.NumberColumn(format="R$ %.2f", min_value=0.01),
    "Mín. Venda (Unidades)": st.column_config.NumberColumn(format="%d", min_value=0, step=1),
    "Máx. Venda (Unidades)": st.column_config.NumberColumn(format="%d", min_value=1, step=1),
}

df_editado = st.data_editor(
    df_base,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True
)

st.markdown("---")

# 2.4 Botão de Execução
if st.button('🚀 Calcular Plano de Produção Ótimo', type="primary"):
    
    # 1. Substituir valores Nulos em colunas críticas para evitar falhas no cálculo
    df_temp = df_editado.copy()

    # Preenche 'Categoria' com 'G' (Grande) para NaN, evitando erro no .map()
    df_temp['Categoria'] = df_temp['Categoria'].fillna('G')

    cols_numericas = ['Preço de Venda (R$)', 'Preço de Custo (R$)', 'Mín. Venda (Unidades)', 'Máx. Venda (Unidades)']
    for col in cols_numericas:
        if col in df_temp.columns:
            # Garante que Min e Max são pelo menos 0, e Venda/Custo são pelo menos 0.01
            df_temp[col] = df_temp[col].fillna(0)

    df_validado = df_temp.dropna(subset=['Modelo']).copy()            
    
    if df_validado.empty:
        st.error("Por favor, preencha pelo menos um modelo válido na tabela.")
    else:
        with st.spinner('Otimizando o plano de produção...'):
            resultados = calcular_e_otimizar(df_validado, horas_total, arredondar_resultado)
        
        # 3. EXIBIÇÃO DOS RESULTADOS
        
        if resultados[0] is not None:
            df_otimo, lucro_maximo, tempo_usado, mensagem = resultados
            
            st.success("Cálculo concluído com sucesso!")
            
            col1, col2 = st.columns(2)
            
            # Coluna 1: Métricas Chave
            with col1:
                st.metric(
                    "💰 Lucro Máximo Mensal", 
                    f"R$ {lucro_maximo:,.2f}"
                )
                
                # Exibe o tempo usado em relação ao total
                delta_tempo = round(tempo_usado - horas_total, 2)
                st.metric(
                    "Horas de Trabalho Usadas", 
                    f"{tempo_usado:,.2f}h de {horas_total}h",
                    delta=f"{delta_tempo}h de sobra/falta",
                    delta_color="inverse"
                )

            # Coluna 2: Plano de Produção Detalhado
            with col2:
                st.subheader("Plano de Produção Ótimo")
                
                df_final = df_otimo[[
                    'Modelo', 
                    'Categoria', 
                    'Produção Ideal (Unidades)', 
                    'Lucro Unitário (R$)',
                    'Tempo Total (h)'
                ]].sort_values(by='Lucro Unitário (R$)', ascending=False)
                
                st.dataframe(df_final, use_container_width=True, hide_index=True)
                
        else:
            st.error(f"Não foi possível encontrar uma solução ótima: {resultados[1]}")
            st.info("Verifique se as restrições de venda mínima são alcançáveis com as horas disponíveis.")