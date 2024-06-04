### Reference Script for Processing the Database and Returning Cohort Graphs ###

## Basic Rules:
# File must have two columns: 'Date' and 'ID'
# ID can be any unique identifier (CPF, RG, Unique ID...)
# Date can be, well, any date
# File extension can be .csv, .xlsx, .xlsm, .xlsb, .xls


## Libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import timedelta
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import openai
import os
from dotenv import load_dotenv

load_dotenv('secrets.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

## Receives CSV or Excel file and returns Pandas Dataframe
def load_data(csv):
    # Identify format

    # Load Data

    ## Testing
    data = pd.read_csv(csv)
    data['Data'] = pd.to_datetime(data['Data'])
    data['cohort'] = data.groupby('CPF')['Data'].transform('min')
    data['sale_yearmonth'] = data['Data'].dt.to_period('M') 
    data['cohort_yearmonth'] = data['cohort'].dt.to_period('M')
    cohort_data = data.groupby(['cohort_yearmonth', 'sale_yearmonth']).agg(n_brokers=('CPF', 'nunique')).reset_index()
    cohort_data['period_number'] = (cohort_data['sale_yearmonth'] - cohort_data['cohort_yearmonth']).apply(lambda x: x.n)
    cohort_pivot = cohort_data.pivot_table(index='cohort_yearmonth', columns='period_number', values='n_brokers', aggfunc='sum')
    cohort_sizes = cohort_pivot.iloc[:, 0] 
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)

    return retention_matrix, cohort_sizes


## Receives Raw Pandas Dataframe and returns DF ready for plotting insights
def format_data():
    # Identify Columns (Which one is an ID and which is Date)

    # Create Cohort Column Full-Date, and Cohort Year-Month column

    # 

    return # Formatted_data


## Receives formatted DataFrame and generate graphs
def plot_graphs(retention_matrix, cohort_sizes):
    # Plot Main Graph formatted

    # Add logo if there is logo (Optional)

    ## Testing
    # Cores Alice para cmap
    pink_colors = ['#ffffff', '#f9ccec', '#f487cf', '#e40494']  # Claro para escuro

    # Criando colormap personalizado
    pink_cmap = LinearSegmentedColormap.from_list("CustomPink", pink_colors, N=256)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(retention_matrix, annot=True, fmt=".0%", cmap=pink_cmap, vmax=0.25, ax=ax) # Mudar vmax a depender dos valores de retenção para melhor visualização

    # Adicionando tamanho dos cohorts no gráfico
    for i, size in enumerate(cohort_sizes):
        ax.annotate(f'Brokers: {int(size)}', xy=(0, i), xycoords='data', textcoords='offset points', xytext=(55, -25),
                    ha='center', va='center', color='white', weight='bold', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.1))

    ax.set_title('Análise de Cohort - % de Retenção Brokers')
    ax.set_ylabel('Cohort (Mês da primeira venda)')
    ax.set_xlabel('Meses após primeira venda')

    # Save the figure
    plt.savefig('retention_heatmap.png')

    # Return the figure object
    return fig


## Receives graphs and generate insights
def llm_analysis(retention_matrix, fig, prompt):
    # Create LLM Instance

    # Generate Response by feeding the graph (and file - maybe)
        # Convert retention matrix to a CSV string
    retention_matrix_csv = retention_matrix

    # Save the figure as a PNG file
    fig_path = 'retention_heatmap.png'
    fig.savefig(fig_path)

    # Initialize OpenAI with your API key
    llm = OpenAI(api_key=OPENAI_API_KEY)

    # Create the prompt
    question = f"""{prompt}\n\nRetention Matrix:\n{retention_matrix_csv}\n\n
    Figure saved at {fig_path}
    """

    template = """ Question: {question}
    
    Answer: Let's analyze it step by step.
    """

    prompt_final = PromptTemplate.from_template(template)

    # Langchain chain
    llm_chain = prompt_final | llm

    # Query the LLM
    response = llm_chain.invoke(question)

    return response


## Receives CSV or Excel file, calls all functions, returns graphs plotted and LLM analysis
def main():
    csv_path = 'ludke_alice.csv'
    retention_matrix, cohort_sizes = load_data(csv_path)
    fig = plot_graphs(retention_matrix, cohort_sizes)
    prompt = "Analyze the retention trends and provide insights based on the cohort analysis."
    llm_response = llm_analysis(retention_matrix, fig, prompt)

    plt.show()
    return fig, llm_response

if __name__ == "__main__":
    fig, llm_response = main()
    with open("output.txt", "w") as f:
        f.write(llm_response)
    print("Output written to output.txt")
