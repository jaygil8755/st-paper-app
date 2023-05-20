from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import streamlit as st  

import pandas as pd  
import requests
from bs4 import BeautifulSoup as bs
import base64
import time
import numpy as np
import random
from io import StringIO

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from mecab import MeCab
import konlpy    
from konlpy.tag import Okt  
import nltk
from nltk.corpus import stopwords

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.colors as mcolors
import plotly.express as px
import seaborn as sns
font_path = "./font/NanumBarunGothic.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

import pyLDAvis.gensim_models
import regex
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models import LdaModel

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
  
st.set_page_config(layout='wide', page_title="crawling papers")
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.sidebar:
    choose = option_menu("App Menu", ["Crawl", "Analyze", "LDA TM", "Bertopic", "Contact"],
                         icons=['box arrow in down', 'bar-chart', 'card-text', 'chat-text','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "Crawl":
    st.subheader("LLM 기반 연구 논문 분석 자동화 서비스")
    st.markdown("""
        #### 논문 크롤링 자동화 """)
    expander_bar = st.expander("Quick Guide")
    expander_bar.markdown("""
    1. 수집할 논문의 키워드를 입력하세요. 
    2. RISS에서 수집된 논문 수를 확인하고 '크롤링 시작'을 체크하세요.
    3. 국내학술논문과 석박사 논문을 크롤링합니다.(제목, 저자, 연도, 발행기관, 학술지, 상세링크, 초록이 수집됩니다.)
    http://www.riss.kr/index.do
    4. 수집된 논문을 Pandas 데이터프레임으로 확인하고 csv 파일로 저장합니다.
    
    ** 본 서비스는 교육/연구용으로 제공되는 것으로 결과에 대해 어떠한 책임도 지지 않습니다. 
    저작권에 대한 책임도 이용자 본인에게 있습니다.**
    """)
    with st.form("form"):
      keyword=st.text_input('검색할 논문의 키워드를 입력하세요(예:토픽 모델링): ', key='keyword') 
      submit = st.form_submit_button("검색")

   
    if submit:
        
        st.write(f'{keyword}에 대한 논문을 수집합니다.')    
    
        HEADERS={'User-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}
    
        url1 = f'http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&query={keyword}&queryText=&iStartCount=0&iGroupView=5&icate=bib_t&colName=re_a_kor&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&pageScale=10&orderBy=&fsearchMethod=search&isFDetailSearch=N&sflag=1&searchQuery={keyword}&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&resultKeyword={keyword}&pageNumber=1&p_year1=&p_year2=&dorg_storage=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&language_code=&ccl_code=&language=&inside_outside=&fric_yn=&image_yn=&regnm=&gubun=&kdc=&ttsUseYn='
        url2 = f'http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&query={keyword}&queryText=&iStartCount=0&iGroupView=5&icate=re_a_kor&colName=bib_t&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&pageScale=10&orderBy=&fsearchMethod=search&isFDetailSearch=N&sflag=1&searchQuery={keyword}&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&resultKeyword={keyword}&pageNumber=1&p_year1=&p_year2=&dorg_storage=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&language_code=&ccl_code=&language=&inside_outside=&fric_yn=&image_yn=&regnm=&gubun=&kdc=&ttsUseYn='
    
        result1 = requests.get(url1, headers=HEADERS)
        result2 = requests.get(url2, headers=HEADERS)
    
        if result1.status_code == 200:
            soup1 = bs(result1.text, 'html.parser')
            max_num1 = soup1.find('span', class_='num').text
            max_num1= max_num1.replace(',','')
            #print(f'총 {max_num1}개의 학술논문이 검색되었습니다.')
            #st.markdown(f'총 {max_num1}개의 학술논문이 검색되었습니다.')         
        else :
          print('다음에 다시 시도해주세요')
    
        if result2.status_code == 200:
            soup2 = bs(result2.text, 'html.parser')
            max_num2 = soup2.find('span', class_='num').text
            max_num2= max_num2.replace(',','')
            #print(f'총 {max_num2}개의 학위논문이 검색되었습니다.')
        else :
          print('다음에 다시 시도해주세요')
        
        st.info(f'총 {max_num1}개의 학술논문과 {max_num2}개의 학위논문이 검색되었습니다.')
        
        if 'max_num1' not in st.session_state:
            st.session_state['max_num1'] = max_num1
            
        if 'max_num2' not in st.session_state:
            st.session_state['max_num2'] = max_num2
    
    if st.button('크롤링 시작') :
        
        with st.spinner('논문을 수집하고 있습니다....'):
           tab1, tab2 = st.tabs(["학술논문", "학위논문"])

           with tab1:
              HEADERS={'User-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}
                          
              url1 = f'http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&query={keyword}&queryText=&iStartCount=0&iGroupView=5&icate=bib_t&colName=re_a_kor&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&pageScale={st.session_state.max_num1}&orderBy=&fsearchMethod=search&isFDetailSearch=N&sflag=1&searchQuery={keyword}\&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&resultKeyword={keyword}&pageNumber=1&p_year1=&p_year2=&dorg_storage=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&language_code=&ccl_code=&language=&inside_outside=&fric_yn=&image_yn=&regnm=&gubun=&kdc=&ttsUseYn='
              result1 = requests.get(url1, headers=HEADERS)
              soup1 = bs(result1.text, 'html.parser')
              contents1 = soup1.find_all('div', class_='cont ml60')

              title = []
              writer =[]
              publisher = []
              year = []
              journal = []
              link = []
              abstracts = []

              for cont in contents1:
                  title.append(cont.find('p', class_='title').text)
                  writer.append(cont.find('span', class_='writer').text)
                  publisher.append(cont.find('span', class_='assigned').text)
                  year.append(cont.find('p', class_='etc').find_all('span')[2].text)
                  journal.append(cont.find('p', class_='etc').find_all('span')[3].text)
                  link.append("https://www.riss.kr"+cont.find('p', class_='title').find('a')['href'])

                  if cont.find('p', class_='preAbstract'):
                      abstracts.append(cont.find('p', class_='preAbstract').text)
                  else :
                      abstracts.append('No_Abstracts')

              df1 = pd.DataFrame(
              {'title':title,
              'writer': writer,
              'publisher': publisher,
              'year': year,
              'journal': journal,
              'link': link,
              'abstracts': abstracts}
      )
              st.balloons()
              st.success("학술 논문 수집에 성공하였습니다.")

              st.dataframe(df1)
                       
           with tab2:      
              HEADERS={'User-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}
                   
              url2 = f'http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&query={keyword}&queryText=&iStartCount=0&iGroupView=5&icate=re_a_kor&colName=bib_t&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&pageScale={st.session_state.max_num2}&orderBy=&fsearchMethod=search&isFDetailSearch=N&sflag=1&searchQuery={keyword}&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&resultKeyword={keyword}&pageNumber=1&p_year1=&p_year2=&dorg_storage=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&language_code=&ccl_code=&language=&inside_outside=&fric_yn=&image_yn=&regnm=&gubun=&kdc=&ttsUseYn='
              result2 = requests.get(url2, headers=HEADERS)
              soup2 = bs(result2.text, 'html.parser')
              contents2 = soup2.find_all('div', class_='cont ml60')
    
              title = []
              writer =[]
              publisher = []
              year = []
              journal = []
              link = []
              abstracts = []

              for cont in contents2:
                  title.append(cont.find('p', class_='title').text)
                  writer.append(cont.find('span', class_='writer').text)
                  publisher.append(cont.find('span', class_='assigned').text)
                  year.append(cont.find('p', class_='etc').find_all('span')[2].text)
                  journal.append(cont.find('p', class_='etc').find_all('span')[3].text)
                  link.append("https://www.riss.kr"+cont.find('p', class_='title').find('a')['href'])

                  if cont.find('p', class_='preAbstract'):
                      abstracts.append(cont.find('p', class_='preAbstract').text)
                  else :
                      abstracts.append('No_Abstracts')

              df2 = pd.DataFrame(
              {'title':title,
              'writer': writer,
              'publisher': publisher,
              'year': year,
              'journal': journal,
              'link': link,
              'abstracts': abstracts}
          )
              st.balloons()
              st.success("학위 논문 수집에 성공하였습니다.")

              st.dataframe(df2)
        
        st.write('학술 논문과 학위 논문을 하나로 만듭니다.')
        df = pd.concat([df1,df2], ignore_index=True)
    
        st.dataframe(df)
        if 'df' not in st.session_state:
            st.session_state['df'] = df
        st.session_state['df'] = df
        
        time.sleep(1)
    
    if st.checkbox ('키워드 미포함 또는 중복 데이터 삭제'):
    
        st.write('제목과 초록에 키워드가 없는 논문은 삭제합니다.')
#         keyword='토픽 ?모델링'
        # 정규표현식으로 띄어쓰기가 0개 이상인 단어 찾기
        keyword = st.session_state.keyword.replace(" ", " ?")
        
    #     df=df[(df['title'].str.contains(keyword))|(df['abstracts'].str.contains(keyword))] 
        st.session_state['df']=st.session_state['df'][(st.session_state['df']['title'].str.contains(keyword))|(st.session_state['df']['abstracts'].str.contains(keyword))]
    
        time.sleep(1)   
    
        st.write('중복데이터를 검사합니다..')
    
        df_double = st.session_state['df'][st.session_state['df'].duplicated(subset=['title', 'writer'], keep=False)]
        st.dataframe(df_double)
        st.session_state['df'].drop_duplicates(subset= ['title', 'writer'], keep='first', inplace=True, ignore_index=True)
    
        st.write('중복데이터를 제거했습니다.')
        st.write('최종 논문 수:', len(st.session_state['df']))
    
        st.dataframe(st.session_state['df'])
    
    if st.checkbox ('데이터 저장'):
    
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(st.session_state['df'])

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f'{keyword}.csv',
            mime='text/csv',
        )

if choose == "Analyze":
    st.header("논문데이터 분석과 시각화")

    uploaded_file = st.file_uploader("수집한 csv파일을 업로드하세요.")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        df = pd.read_csv(uploaded_file)

        st.write('처음 5개 데이터')
        st.dataframe(df.head())
        st.write('마지막 5개 데이터')
        st.dataframe(df.tail())
        st.write('데이터의 수')
        st.write(len(df))
        
    st.markdown('**저자**에서 특정 단어를 포함하고 있는 논문만 검색할 수 있습니다. ')
    keyword_writer =st.text_input('저자에 포함할 단어를 입력해주세요: ')
    
    if keyword_writer :
        df_=df[df['writer'].str.contains(keyword_writer)]
        st.write(f'총 {len(df_)}편이 검색되었습니다.')
        st.dataframe(df[df['writer'].str.contains(keyword_writer)])
    
    st.markdown('**제목**에서 특정 단어를 포함하고 있는 논문만 검색할 수 있습니다. ')
    keyword_title =st.text_input('제목에 포함할 단어를 입력해주세요: ')
    
    if keyword_title :
        df_= df[df['title'].str.contains(keyword_title, case=False)]
        st.write(f'총 {len(df_)}편이 검색되었습니다.')
        st.dataframe(df[df['title'].str.contains(keyword_title, case=False)])
    
    st.markdown('**초록**에서 특정 단어를 포함하는 논문만 검색할 수 있습니다.')
    keyword_abs =st.text_input('초록에 포함할 단어를 입력해주세요: ')
    
    if keyword_abs :
        df_= df[df['abstracts'].str.contains(keyword_abs, case=False)]
        st.write(f'총 {len(df_)}편이 검색되었습니다.')
        st.dataframe(df[df['abstracts'].str.contains(keyword_abs, case=False)])
    
    vis_checked=st.checkbox('데이터 시각화')
    
    if vis_checked :
      
      tab1, tab2, tab3 = st.tabs(["journal bar chart - 상위 빈도 20개", "publisher pie chart - 상위 빈도 20개", "year histogram"])

      with tab1:
          fig1_1 = px.bar(df['journal'].value_counts().head(20), orientation='h', labels={'index': '학술지/학위', 'value': '논문 수'})
          fig1_1.update_layout(showlegend=True, 
                               legend_title='학술지/학위별 논문수')

          st.plotly_chart(fig1_1)

      with tab2:
          freq = df['publisher'].value_counts().head(20)
          fig2_2 = px.pie(values=freq, 
                    names=freq.index, 
                    title='학회/학교별 논문 수')

          fig2_2.update_traces(textposition='inside', 
                         textinfo='percent+label', 
                         marker=dict(colors=px.colors.sequential.YlGnBu))
          st.plotly_chart(fig2_2)

      with tab3:
          fig3_3 = px.histogram(df, 
                           x='year', 
                           nbins=25, 
                           color_discrete_sequence=['pink'], 
                           title='연도별 논문 수',
                           labels={'year': '발행 연도', 'count': '발행 논문수'})

          st.plotly_chart(fig3_3)

if choose == "LDA TM":
  
   
    st.header('LDA 토픽모델링')
    uploaded_file = st.file_uploader("수집한 csv파일을 업로드하세요.")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        df = pd.read_csv(uploaded_file)
        st.write('처음 5개 데이터 확인')
        st.dataframe(df.head())
    
        st.markdown('한글 제목과 초록만 추출하여 저장하기')
            
        df_text = df.get('title') + ' '+ df.get('abstracts')
        df_kor_text = df_text.str.replace('[^가-힣]',' ', regex=True).replace('\s+',' ', regex=True)
    
        text = df_kor_text.to_list()
        st.write('[확인] 처음 3개 text 시예시 (제목 + 초록)', text[:3])
  
        if st.checkbox('텍스트 빈도 분석 : 워드 클라우드', value=False):
    
            with st.spinner('텍스트에서 명사를 추출하고있습니다.....'): 
    
                kiwi=Kiwi() 
                extract_pos_list = ["NNG", "NNP", "NNB", "NR", "NP"]
                stopwords = Stopwords()
          
                tokenized_doc=[]
                for words in text:
                    nouns_ = [] 
                    for word in kiwi.tokenize(words,  stopwords=stopwords):
                        if word[1] in extract_pos_list and len(word[0]) > 1 :
                            nouns_.append(word[0])     
                    tokenized_doc.append(nouns_)
                Bigram_Model = gensim.models.Phrases(tokenized_doc, min_count=2, delimiter=" ")
                bigram_tokenized_doc = []
                for doc in tokenized_doc:
                    a = Bigram_Model[doc]
                    bigram_tokenized_doc.append(a)
                    
            st.write('[확인] 첫번째 논문에 등장하는 단어들은', bigram_tokenized_doc[0])

            dictionary = corpora.Dictionary(bigram_tokenized_doc)
            dictionary.filter_extremes(no_below=5, no_above=0.4)
            st.write('#자주 등장하거나 등장횟수가 적은 명사 제외한 단어는:', len(dictionary))
            
    
            st.session_state.bigram_tokenized_doc = bigram_tokenized_doc
            st.session_state.dictionary = dictionary
            st.session_state.text = text
            st.write('가장 빈도수가 높은 단어 Top 10:')
            
            st.dataframe(pd.DataFrame(st.session_state.dictionary.most_common(10), columns=['Word', 'Count']))
    
            with st.expander('3개의 워드클라우드를 생성'):
    
                top_nouns_from_corpora = dict(st.session_state.dictionary.most_common(100))
    
                left_column, middle_column, right_column = st.columns(3)
    
                with left_column:
                    wordcloud = WordCloud (width=700, height=600,
                                   background_color='white',prefer_horizontal=1.0, random_state = 20,font_path = "./font/NanumBarunGothic.ttf")
                    wc = wordcloud.generate_from_frequencies(top_nouns_from_corpora)
                    fig = plt.figure()
                    plt.imshow(wc, interpolation="bilinear")     
                    plt.axis('off')    
                    left_column.pyplot(fig)
    
    
                with middle_column:
                    wordcloud = WordCloud (width=700, height=600,
                                   background_color='black', prefer_horizontal=1.0, random_state = 21,font_path = "./font/NanumGothic.ttf")
                    wc = wordcloud.generate_from_frequencies(top_nouns_from_corpora)
                    fig = plt.figure()
                    plt.imshow(wc, interpolation="bilinear")     
                    plt.axis('off')     
                    middle_column.pyplot(fig)
    
                with right_column:
                    wordcloud = WordCloud (width=700, height=600,
                                   background_color='white',prefer_horizontal=1.0, random_state = 22,font_path = "./font/NanumPen.ttf")
                    wc = wordcloud.generate_from_frequencies(top_nouns_from_corpora)
                    fig = plt.figure()
                    plt.imshow(wc, interpolation="bilinear")     
                    plt.axis('off')     
                    right_column.pyplot(fig)
  
   
            COLORS = [color for color in mcolors.XKCD_COLORS.values()]
    
            def show_coherence(corpus, dictionary, start=4, end=11):
                iter_num = []
                per_value = []
                coh_value = []
    
                for i in range(start, end + 1):
    
                    model = LdaModel(corpus=corpus, id2word=dictionary,
                             chunksize=1000, num_topics=i,
                             random_state=7)
                    iter_num.append(i)
                    pv = model.log_perplexity(corpus)
                    per_value.append(pv)
    
                    cm = CoherenceModel(model=model, corpus=corpus, 
                                        coherence='u_mass')
                    cv = cm.get_coherence()
                    coh_value.append(cv)
                    print(f'num_topics: {i}, perplexity: {pv:0.3f}, coherence: {cv:0.3f}')
    
                left_column, right_column = st.columns(2)
    
                with left_column:
   
                    fig1=plt.figure()  
                    plt.plot(iter_num, per_value, 'g-')
                    plt.xlabel("num_topics")
                    plt.ylabel("perplexity")
                    st.pyplot(fig1)
    
                with right_column:
    
                    fig2=plt.figure()
                    plt.plot(iter_num, coh_value, 'r--')
                    plt.xlabel("num_topics")
                    plt.ylabel("coherence")
                    st.pyplot(fig2)
    
            with st.spinner('최적의 토픽 수를 찾는 중....'):
                corpus = [st.session_state.dictionary.doc2bow(text) for text in st.session_state.bigram_tokenized_doc]
                id2word = st.session_state.dictionary
                show_coherence(corpus, id2word)
    
        NUM = st.number_input('토픽 수를 선택하세요', min_value=4, max_value=11, value=7, step=1)
        st.session_state.num =NUM
    
        start = st.checkbox('토픽모델링 시작!', value=False)
    
        if start:
            st.write(NUM, '개의 토픽을 찾습니다.' )
    
            with st.spinner('LDA 모델 훈련 중 ...'):
    
                corpus = [st.session_state.dictionary.doc2bow(text) for text in st.session_state.bigram_tokenized_doc]
                model = gensim.models.LdaModel(corpus, id2word=st.session_state.dictionary, num_topics=st.session_state.num)
    
                topics = model.show_topics(formatted=False, num_words=50,
                                                         num_topics=st.session_state.num, log=False)
    
            with st.expander('Topic Word-Weighted Summaries'):
                topic_summaries = {}
                for topic in topics:
                    topic_index = topic[0]
                    topic_word_weights = topic[1]
                    topic_summaries[topic_index] = ' + '.join(
                        f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
                for topic_index, topic_summary in topic_summaries.items():
                    st.markdown(f'**Topic {topic_index}**: _{topic_summary}_')
    
            COLORS = [color for color in mcolors.XKCD_COLORS.values()]
            colors = random.sample(COLORS, k=st.session_state.num)
            with st.expander('Top N Topic Keywords Wordclouds'):
                cols = st.columns(3)
                for index, topic in enumerate(topics):
                    wc = WordCloud(font_path=font_path, width=700, height=600,
                                   background_color='white',prefer_horizontal=1.0,
                                   color_func=lambda *args, **kwargs: colors[index])
                    with cols[index % 3]:
                        wc.generate_from_frequencies(dict(topic[1]))
                        st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)
 
            if hasattr(model, 'inference'):  # gensim Nmf has no 'inference' attribute so pyLDAvis fails
                with st.spinner('Creating pyLDAvis Visualization ...'):
                    py_lda_vis_data = pyLDAvis.gensim_models.prepare(model, corpus, st.session_state.dictionary)
                    py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
                with st.expander('pyLDAvis', expanded=True):
                    components.html(py_lda_vis_html, width=1300, height=800)   
        
if choose == "Bertopic":
    st.header('Bertopic (LLM)')
    st.markdown('''임베딩 모델 이름을 확인하세요..
    model1 = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    model2 = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    model3 = "sentence-transformers/all-mpnet-base-v2"
    model4 = "jhgan/ko-sroberta-multitask"
    ''')
    model = st.radio ("어떤 모델을 선택하시겠습니까?", ['model1', 'model2', 'model3', 'model4' ])
    tab1, tab2, tab3, tab4 = st.tabs(['model1', 'model2', 'model3', 'model4'])

    with tab1:
        if model == 'model1':
            model1 = BERTopic.load("./model/my_model1")
            st.header("Visualizations")
            st.plotly_chart(model1.visualize_topics())
            st.plotly_chart(model1.visualize_barchart(top_n_topics = 9990, n_words = 9999))
            st.plotly_chart(model1.visualize_heatmap())
            st.plotly_chart(model1.visualize_hierarchy())
            st.plotly_chart(model1.visualize_term_rank())

    with tab2:
        if model == 'model2':
            model1 = BERTopic.load("./model/my_model2")
            st.header("Visualizations")
            st.plotly_chart(model2.visualize_topics())
            st.plotly_chart(model2.visualize_barchart(top_n_topics = 9990, n_words = 9999))
            st.plotly_chart(model2.visualize_heatmap())
            st.plotly_chart(model2.visualize_hierarchy())
            st.plotly_chart(model2.visualize_term_rank())
            
    with tab3:
        if model == 'model3':
            model1 = BERTopic.load("./model/my_model3")
            st.header("Visualizations")
            st.plotly_chart(model3.visualize_topics())
            st.plotly_chart(model3.visualize_barchart(top_n_topics = 9990, n_words = 9999))
            st.plotly_chart(model3.visualize_heatmap())
            st.plotly_chart(model3.visualize_hierarchy())
            st.plotly_chart(model3.visualize_term_rank())
    with tab4:
        if model == 'model4':
            model1 = BERTopic.load("./model/my_model2")
            st.header("Visualizations")
            st.plotly_chart(model4.visualize_topics())
            st.plotly_chart(model4.visualize_barchart(top_n_topics = 9990, n_words = 9999))
            st.plotly_chart(model4.visualize_heatmap())
            st.plotly_chart(model4.visualize_hierarchy())
            st.plotly_chart(model4.visualize_term_rank())

    
    
#     uploaded_file = st.file_uploader("수집한 csv파일을 업로드하세요.")
#     if uploaded_file is not None:
#         bytes_data = uploaded_file.getvalue()
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         string_data = stringio.read()
#         df = pd.read_csv(uploaded_file)
#         st.write('처음 5개 데이터 확인')
#         st.dataframe(df.head())
           
# #         df_with_abs = df[df.abstracts != 'No_Abstracts']
# #         df_with_abs = df_with_abs.abstracts.str.replace('[^가-힣]',' ', regex=True).replace('\s+',' ', regex=True)
    
# #         text = df_with_abs.to_list()
#         text = df.title.to_list()
# #         st.write('초록이 있는 논문 수', len(text))
    
#         # 토크나이저에 명사만 추가한다
#         st.write('명사를 추출합니다..')

#         extract_pos_list = ["NNG", "NNP", "NNB", "NR", "NP"]
#         stopwords = Stopwords()
#         stopwords.add(('토픽', 'NNG'))
#         stopwords.add(('모델링', 'NNG'))
        
#         class CustomTokenizer:
#             def __init__(self, kiwi):
#                 self.kiwi = kiwi
#             def __call__(self, text):
#                 result = list()
#                 for word in self.kiwi.tokenize(text,  stopwords=stopwords):
#                     # 명사이고, 길이가 2이상인 단어이고, 불용어 리스트에 없으면 추가하기
#                     if word[1] in extract_pos_list and len(word[0]) > 1 :
#                          result.append(word[0])
#                 return result
                
         
#         custom_tokenizer = CustomTokenizer(Kiwi())
#         vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=300)
# #         st.write('CountVertorizer  생성 완료!', vectorizer)

#         with st.expander('xlm-r-100langs-bert 모델'):
#           with st.spinner('BERTopic 모델 훈련 중 ...'):
#             model1 = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", \
#             vectorizer_model=vectorizer,
#             nr_topics=10, # 문서를 대표하는 토픽의 갯수
#             top_n_words=10,
#             calculate_probabilities=True)

#             # tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
#             # model = AutoModel.from_pretrained("beomi/kcbert-base")

#             # model1 = BERTopic(embedding_model=model, language="korean", 
#             #                   # top_n_words=10, nr_topics= Nr_topics, 
#             #                   calculate_probabilities=False, verbose=False)
#             # model1.fit(text)
#             topics, probs = model1.fit_transform(text)
        
#             st.dataframe(model1.get_topic_info())

    
            # with st.expander('beomi/kcbert-base 모델'):
                
            #     st.write('시작...')
                
            #     tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
            #     model = AutoModel.from_pretrained("beomi/kcbert-base")
                
            #     model1 = BERTopic(embedding_model=model, language="korean", 
            #                       # top_n_words=10, nr_topics= Nr_topics, 
            #                       calculate_probabilities=False, verbose=False)
            #     # model1.fit(text)
            #     topics, probs = model1.fit_transform(text)
            
            #     st.dataframe(model1.get_topic_info())
            #     st.plotly_chart(model1.visualize_barchart(top_n_topics=7))    
       
            #     for i in range(Nr_topics):
            #       st.write(i,'번째 토픽 :', model1.get_topic(i))
    
            # with st.expander('all-mpnet-base-v2 모델'):
        
           
            #     sentence_model = SentenceTransformer("all-mpnet-base-v2")
            #     model2 = BERTopic(embedding_model=sentence_model, language="multiligual", nr_topics= Nr_topics, calculate_probabilities=True, verbose=False)
            #     topics, probs = model2.fit_transform(text)
        
            #     st.table(model2.get_topic_info().head(7))
            #     fig3, ax = model2.visualize_barchart()
            #     st.pyplot(fig3)
        
            #     fig4, ax = model2.visualize_barchart(top_n_topics=Nr_topics)
            #     st.pyplot(fig4)
        
            #     for i in range(N_topics):
            #       st.write(i,'번째 토픽 :', model.get_topic(i))



# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
