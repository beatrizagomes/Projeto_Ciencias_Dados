import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import nltk
import contractions
import spacy
import re
from nltk.tokenize import word_tokenize
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import nltk
import contractions
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyecharts import options as opts
from pyecharts.charts import Map
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from collections import Counter
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from pprint import pprint

# Função para limpeza de texto
def clean_text(text_string, customlist=[]):
    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', text_string)

    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    # Removing the punctuations
    string = re.sub(r'[^\w\s]', '', string)

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in string.split() if word not in stop_words]

    # Applying custom stop words
    final_words = list(set(filtered_words) - set(customlist))

    # Tokenization
    tokens = word_tokenize(' '.join(final_words))

    # Remove numbers
    tokens = [word for word in tokens if word.isalpha()]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    # Fix contractions
    final_string = ' '.join([contractions.fix(word) for word in stemmed_words])

    return final_string

def remove_punctuation(text):
        # Função para remover pontuações, incluindo ponto e vírgula
        return text.replace(';', '').replace(',', '').replace('.', '')

def perform_year_chart(df, country_filter):

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Artciles from {country_filter}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Drop rows with NaN values in "Authors with affiliations" column
    df_cleaned = df.dropna(subset=["Authors with affiliations"])

    # Filter dataframe based on selected country
    selected_country_df = df_cleaned[df_cleaned["Authors with affiliations"].str.lower().str.contains(country_filter.lower(), na=False)]

    # Extract and count occurrences of years
    year_counts = selected_country_df["Year"].value_counts().sort_index()
    year_counts_df = pd.DataFrame({"Year": year_counts.index, "Article Count": year_counts.values})

    # Bar chart using Altair with a customized style (orange bars)
    chart = alt.Chart(year_counts_df).mark_bar(opacity=0.7, color='#FFA500').encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Article Count:Q', title='Article Count'),
        tooltip=['Year:O', 'Article Count:Q'],
    ).properties(
        width=1400,
        height=600,
        title=f"Number of Articles per Year in {country_filter.capitalize()}"
    ).configure_title(
        fontSize=18,
        anchor='start',
        offset=15
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    st.altair_chart(chart, use_container_width=True)

    selected_articles = df[df["Authors with affiliations"].str.contains(country_filter, case=False, na=False)]
    st.subheader(f"List of Articles from {country_filter}")
    # Modifique esta linha para formatar a coluna "Year"
    st.write(selected_articles[["Author full names", "Title", "Year", "Abstract"]].replace({r',': ''}, regex=True).astype({"Year": str}))



# Função para realizar a análise bibliométrica
def perform_abstarct_analysis_one(df, country_filter):

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Most common words in the Abstract")

    # Drop rows with NaN values in "Authors with affiliations" column
    df_cleaned = df.dropna(subset=["Authors with affiliations"])

    # Filter dataframe based on selected country
    selected_country_df = df_cleaned[df_cleaned["Authors with affiliations"].str.lower().str.contains(country_filter.lower())]

    st.markdown("<br>", unsafe_allow_html=True)  # Adiciona uma linha em branco

    top = ["Top 3", "Top 5", "Top 10", "Top 30"]

    # top-level filters
    top_filter = st.selectbox("Select the Top Words", pd.unique(top))

    st.markdown("<br>", unsafe_allow_html=True)

    country_df = selected_country_df
    column_name = 'clean_Abstract'

    selected_data = country_df[column_name].dropna()

    # Count the frequency of each word
    word_freq = Counter(selected_data.str.split().sum())

    # Prepare data for plotting
    labels, values = zip(*word_freq.items())

    # Sort the values in descending order
    indSort = sorted(range(len(values)), key=lambda k: values[k], reverse=True)

    # Rearrange the data
    labels = [labels[i] for i in indSort]
    values = [values[i] for i in indSort]

    # Create the plot
    if top_filter == "Top 3":
        chart = alt.Chart(pd.DataFrame({'labels': labels[:3], 'values': values[:3]})).mark_bar().encode(
            x='labels:N',
            y='values:Q',
            tooltip=['labels', 'values']
        ).properties(
            width=600,
            height=400,
            title='Top 3 Most Frequent Words'
        )
        st.altair_chart(chart, use_container_width=True)

    if top_filter == "Top 5":
        chart = alt.Chart(pd.DataFrame({'labels': labels[:5], 'values': values[:5]})).mark_bar().encode(
            x='labels:N',
            y='values:Q',
            tooltip=['labels', 'values']
        ).properties(
            width=600,
            height=400,
            title='Top 5 Most Frequent Words'
        )
        st.altair_chart(chart, use_container_width=True)

    if top_filter == "Top 10":
        chart = alt.Chart(pd.DataFrame({'labels': labels[:10], 'values': values[:10]})).mark_bar().encode(
            x='labels:N',
            y='values:Q',
            tooltip=['labels', 'values']
        ).properties(
            width=600,
            height=400,
            title='Top 10 Most Frequent Words'
        )
        st.altair_chart(chart, use_container_width=True)

    if top_filter == "Top 30":
        chart = alt.Chart(pd.DataFrame({'labels': labels[:30], 'values': values[:30]})).mark_bar().encode(
            x='labels:N',
            y='values:Q',
            tooltip=['labels', 'values']
        ).properties(
            width=600,
            height=400,
            title='Top 30 Most Frequent Words'
        )
        st.altair_chart(chart, use_container_width=True)


def perform_abstarct_analysis_two(df, country_filter):

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Most common words in the Abstract")

    st.markdown("<br>", unsafe_allow_html=True)
        
    # Drop rows with NaN values in "Authors with affiliations" column
    df_cleaned = df.dropna(subset=["Authors with affiliations"])

    # Filter dataframe based on selected country
    selected_country_df = df_cleaned[df_cleaned["Authors with affiliations"].str.lower().str.contains(country_filter.lower())]

    # Count the frequency of each word
    word_freq = Counter(selected_country_df['clean_Abstract'].str.split().sum())

    # Prepare data for plotting
    labels, values = zip(*word_freq.items())

    # Sort the values in descending order
    indSort = sorted(range(len(values)), key=lambda k: values[k], reverse=True)

    # Rearrange the data
    labels = [labels[i] for i in indSort]
    values = [values[i] for i in indSort]

    # Create the plot using Altair
    chart = alt.Chart(pd.DataFrame({'labels': labels[:30], 'values': values[:30]})).mark_bar().encode(
        x='labels:N',
        y='values:Q',
        tooltip=['labels', 'values']
    ).properties(
        width=600,
        height=400,
        title='Top 30 Most Frequent Words'
    )
    st.altair_chart(chart, use_container_width=True)

def perform_wordcloud(df, country_filter):

    
    # Drop rows with NaN values in "Authors with affiliations" column
    df_cleaned = df.dropna(subset=["Authors with affiliations"])

    # Filter dataframe based on selected country
    selected_country_df = df_cleaned[df_cleaned["Authors with affiliations"].str.lower().str.contains(country_filter.lower())]

    # Wordcloud for Author Keywords and Index Keywords
    keywords_text = " ".join(selected_country_df["Author Keywords"].dropna()) + " " + " ".join(selected_country_df["Index Keywords"].dropna())
    combined_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)

    # Criar duas colunas
    col1, col2 = st.columns(2)

    # Coluna 1: WordCloud
    col1.markdown("<br>", unsafe_allow_html=True)
    col1.subheader("Wordcloud of the Keywords")
    col1.markdown("<br>", unsafe_allow_html=True)

    col1.image(combined_wordcloud.to_array(), use_column_width=True)

    # Contar a ocorrência de palavras em "Author Keywords" e "Index Keywords"
    author_keywords_count = Counter(" ".join(selected_country_df["Author Keywords"].dropna().apply(remove_punctuation)).lower().split())
    index_keywords_count = Counter(" ".join(selected_country_df["Index Keywords"].dropna().apply(remove_punctuation)).lower().split())

    # Adicionar as contagens das duas colunas
    combined_keywords_count = author_keywords_count + index_keywords_count

    # Excluir stop words da contagem
    stop_words = set(stopwords.words("english"))  # Mude para o idioma desejado
    combined_keywords_count = {word: count for word, count in combined_keywords_count.items() if word not in stop_words}

    # Top 15 palavras-chave combinadas
    top_combined_keywords = dict(sorted(combined_keywords_count.items(), key=lambda x: x[1], reverse=True)[:15])

    # Create the plot using Altair
    chart = alt.Chart(pd.DataFrame(list(top_combined_keywords.items()), columns=['Word', 'Count'])).mark_bar().encode(
        x=alt.X('Word:N', sort='-y'),
        y='Count:Q',
        tooltip=['Word', 'Count']
    ).properties(
        width=600,
        height=400,
        title='Top 15 Combined Keywords'
    )

    col2.markdown("<br>", unsafe_allow_html=True)
    col2.subheader("Bar Graph of the Keywords")
    col2.markdown("<br>", unsafe_allow_html=True)
    col2.altair_chart(chart, use_container_width=True)

def perform_wordcloud_two(df, country_filter):

    # Drop rows with NaN values in "Authors with affiliations" column
    df_cleaned = df.dropna(subset=["Authors with affiliations"])

    # Filter dataframe based on selected country
    selected_country_df = df_cleaned[df_cleaned["Authors with affiliations"].str.lower().str.contains(country_filter.lower())]

    # Wordcloud for Author Keywords and Index Keywords
    keywords_text = " ".join(selected_country_df["Author Keywords"].dropna()) + " " + " ".join(selected_country_df["Index Keywords"].dropna())
    combined_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Keywords: Wordcloud and Bar Graph")
    st.markdown("<br>", unsafe_allow_html=True)

    st.image(combined_wordcloud.to_array(), use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Contar a ocorrência de palavras em "Author Keywords" e "Index Keywords"
    author_keywords_count = Counter(" ".join(selected_country_df["Author Keywords"].dropna().apply(remove_punctuation)).lower().split())
    index_keywords_count = Counter(" ".join(selected_country_df["Index Keywords"].dropna().apply(remove_punctuation)).lower().split())

    # Adicionar as contagens das duas colunas
    combined_keywords_count = author_keywords_count + index_keywords_count

    # Excluir stop words da contagem
    stop_words = set(stopwords.words("english"))  # Mude para o idioma desejado
    combined_keywords_count = {word: count for word, count in combined_keywords_count.items() if word not in stop_words}

    # Top 15 palavras-chave combinadas
    top_combined_keywords = dict(sorted(combined_keywords_count.items(), key=lambda x: x[1], reverse=True)[:15])

    # Create the plot using Altair
    chart = alt.Chart(pd.DataFrame(list(top_combined_keywords.items()), columns=['Word', 'Count'])).mark_bar().encode(
        x=alt.X('Word:N', sort='-y'),
        y='Count:Q',
        tooltip=['Word', 'Count']
    ).properties(
        width=600,
        height=400,
        title='Top 15 Combined Keywords'
    )

    st.altair_chart(chart, use_container_width=True)

# Definição da função preprocess_data
def preprocess_data(documents):
    # Define the list of stopwords in English
    stop_words = stopwords.words('english')

    # Tokenize the documents and remove stopwords
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]

    # Return the list of tokenized and preprocessed texts
    return texts

# Função para realizar a análise de tópicos
def perform_topics(df, country_filter):
    # Drop rows with NaN values in "Authors with affiliations" column
    df_cleaned = df.dropna(subset=["Authors with affiliations"])

    # Filter dataframe based on selected country
    selected_country_df = df_cleaned[df_cleaned["Authors with affiliations"].str.lower().str.contains(country_filter.lower())]

    # Preprocessamento dos documentos
    documents = selected_country_df["Abstract"].tolist()
    processed_texts = preprocess_data(documents)

    # Criação do dicionário e do corpus
    id2word = corpora.Dictionary(processed_texts)
    texts = processed_texts
    corpus = [id2word.doc2bow(text) for text in texts]

    # Configuração do modelo LDA
    num_topics = 4
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10, alpha='auto', per_word_topics=True)

    # Impressão dos tópicos
    pprint(lda_model.print_topics())

    # Visualização do modelo de tópicos usando pyLDAvis
    dictionary = Dictionary(processed_texts)
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Topics for {country_filter}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Exibir a visualização no Streamlit
    st.components.v1.html(pyLDAvis.prepared_data_to_html(vis_data), height=800)


# Função principal
def main():
    st.set_page_config(
        page_title="World Migration",
        page_icon="🔍",
        layout="wide",
    )

    # Carregar dados
    df = pd.read_csv("scopus.csv")

    # Títulos
    st.title("Migration in the World")

    # Escolher entre um país e dois países
    analysis_option = st.radio("Choose Analysis Option:", ["One Country", "Two Countries"])

    # Variável para filtrar países
    country_filter = ""

    countries = ["United States", "Germany", "United Kingdom", "Hong Kong", "Australia", "France", "China", "Canada", "Brazil", "Spain", "Italy", "Russian Federation", "Bangladesh", "India"]
    
    if analysis_option == "One Country":
        country_filter = st.selectbox("Select the Country", pd.unique(countries))

        # Criar mapa com Pyecharts
        perform_pyecharts_map(df, countries)

        # Limpeza do Abstract
        perform_abstract_cleaning(df)

        # Opção para escolher o tipo de gráfico
        graph_filter = st.selectbox("Choose the Chart", ["Articles per Year", "Abstract", "Keywords", "Topics"], key="graph_filter")

        if graph_filter == "Articles per Year":
            # Análise Ano
            perform_year_chart(df, country_filter)

        elif graph_filter == "Abstract":
            # Analise Abstract
            perform_abstarct_analysis_one(df, country_filter)

        elif graph_filter == "Keywords":
            # Analise wordcloud
            perform_wordcloud(df, country_filter)

        elif graph_filter == "Topics":
            # Analise Topicos
            perform_topics(df, country_filter)



    elif analysis_option == "Two Countries":
        # Escolher dois países
        selected_countries = st.multiselect("Select Two Countries", countries)

        # Verificar se pelo menos dois países foram selecionados
        if len(selected_countries) == 2:
            # Criar mapa com Pyecharts
            perform_pyecharts_map(df, selected_countries)

            # Limpeza do Abstract
            perform_abstract_cleaning(df)

            # Opção para escolher o tipo de gráfico
            graph_filter = st.selectbox("Choose the Chart", ["Articles per Year", "Abstract", "Keywords", "Topics"], key="graph_filter")

            # Criar duas colunas
            col1, col2 = st.columns(2)

            # Análise bibliométrica para o primeiro país
            with col1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"Analysis for {selected_countries[0]}")

                if graph_filter == "Articles per Year":
                    # Análise Ano
                    perform_year_chart(df, selected_countries[0])

                elif graph_filter == "Abstract":
                    # Analise Abstract
                    perform_abstarct_analysis_two(df, selected_countries[0])

                elif graph_filter == "Keywords":
                    # Analise wordcloud
                    perform_wordcloud_two(df, selected_countries[0])

                elif graph_filter == "Topics":
                    # Analise Topicos
                    perform_topics(df, selected_countries[0])
                

            # Análise bibliométrica para o segundo país
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"Analysis for {selected_countries[1]}")

                if graph_filter == "Articles per Year":
                    # Análise Ano
                    perform_year_chart(df, selected_countries[1])

                elif graph_filter == "Abstract":
                    # Analise Abstract
                    perform_abstarct_analysis_two(df, selected_countries[1])

                elif graph_filter == "Keywords":
                    # Analise wordcloud
                    perform_wordcloud_two(df, selected_countries[1])

                elif graph_filter == "Topics":
                    # Analise Topicos
                    perform_topics(df, selected_countries[1])

    
# Função para criar o mapa interativo usando Pyecharts
def perform_pyecharts_map(df, countries):
    df = df.dropna(subset=["Authors with affiliations"])

    country_counts = {country: 0 for country in countries}
    for country in countries:
        country_counts[country] = df[df["Authors with affiliations"].str.contains(country, case=False)].shape[0]

    data_for_pyecharts = list(zip([c for c in country_counts.keys()], list(country_counts.values())))

    # Pyecharts map
    c = (
        Map()
        .add("Article Count", data_for_pyecharts, "world")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(max_=max(country_counts.values())),
        )
    )

    # Exibir o gráfico Pyecharts no Streamlit
    st.components.v1.html(c.render_embed(), height=600, scrolling=True)


# Função para realizar a limpeza do campo 'Abstract'
def perform_abstract_cleaning(df):
    # Realizar a limpeza do campo 'Abstract'
    df['clean_Abstract'] = df['Abstract'].apply(lambda x: clean_text(x, customlist=['migration', 'migrant']))

# Executar o código principal
if __name__ == "__main__":
    main()
