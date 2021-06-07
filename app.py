import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import requests
from bs4 import BeautifulSoup as bs
import base64


import time


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}

# st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.sidebar.header('메뉴를 선택하세요')
menu1 = ["Riss 논문 수집(한글)", "Word Cloud", "Topic Modeling", "Pubmed 논문 수집(영문)"]
choice = st.sidebar.selectbox("Menu",menu1)
# menu2 = ["Pubmed 논문 수집(Eng)"]
# choice = st.sidebar.selectbox("Menu",menu2)

if choice == "Riss 논문 수집(한글)":
    st.title("논문 데이터 자동 수집 어플리케이션")
    st.subheader("Web App for Crawling Academic Papers and Data Analysis")
    st.markdown("""
        #### --- 누구나 쉽게 자신만의 데이터분석 웹 어플리케이션을 만들 수 있다.

        """)
    expander_bar = st.beta_expander("Quick Guide")
    expander_bar.markdown("""
    1. 왼쪽 사이드바 메뉴에서 서비스 메뉴를 선택할 수 있습니다. 
    2. 먼저 논문 정보 수집을 선택한 후, 검색어와 수집할 논문개수를 입력합니다. 
    3. 수집된 논문은 Pandas DataFrame으로 볼 수 있고, csv 파일로 저장할 수 있습니다.
    4. 워드클라우드와 토픽모델링 메뉴에서는 저장한 파일을 업로드하여 결과를 확인해볼 수 있습니다. (한글논문만)

    ** 본 서비스는 교육/연구용으로 제공되는 것으로 결과에 대해 어떠한 책임도 지지 않습니다. 
    저작권에 대한 책임도 이용자 본인에게 있습니다.**
    """)
    
    
    st.write("1. 검색 키워드 입력")
    

    keyword = st.text_input("검색할 논문의 키워드를 입력하세요(예:로봇, 인공지능+교육): ")
    if st.checkbox("검색 결과 확인!"):
        url = f"http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&queryText=&strQuery={keyword}&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&p_year1=&p_year2=&iStartCount=0\
        &orderBy=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&ccl_code=&inside_outside=&fric_yn=&image_yn=&gubun=&kdc=&ttsUseYn=&fsearchMethod=&sflag=1&isFDetailSearch=N&pageNumber=&resultKeyword=&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&icate=re_a_kor&colName=re_a_kor&pageScale=100\
        &isTab=Y&regnm=&dorg_storage=&language=&language_code=&clickKeyword=&relationKeyword=&query={keyword}"
        result = requests.get(url, headers=headers)
        if result.status_code == 200:

            soup = bs(result.content, 'html.parser')
            max_num = soup.find('span', class_='num').text
            st.text(f"총{max_num}개의 논문이 검색되었습니다. ")
        else :
            st.text("woops! 다음에 다시 시도해주세요.")
            
    st.write("2. 검색할 논문 개수 입력")

    number= st.text_input("이 중 몇 개를 가져올까요?(최대 1,000개):")
    st.text("아래 크롤링 시작! 버튼을 클릭하면 크롤링이 시작되고 잠시 후 결과가 나타납니다.!""")
    
    @st.cache
    def get_info():
    
        url = f"http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&queryText=&strQuery={keyword}&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&p_year1=&p_year2=&iStartCount=0&orderBy=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&ccl_code=&inside_outside=&fric_yn=&image_yn=&gubun=&kdc=&ttsUseYn=&fsearchMethod=&sflag=1&isFDetailSearch=N&pageNumber=&resultKeyword=&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&icate=re_a_kor&colName=re_a_kor&pageScale={number}&isTab=Y&regnm=&dorg_storage=&language=&language_code=&clickKeyword=&relationKeyword=&query={keyword}"


        result = requests.get(url, headers=headers)
        soup = bs(result.content, 'html.parser')
        contents = soup.find_all('div', class_='cont')

        title =[]
        writer = []
        society = []
        year = []
        journal = []
        link =[]
        abstracts=[]


        for cont in contents:

            title_temp = cont.find('p', class_='title').text
            #print(title_temp)
            title.append(title_temp)
            writer_temp = cont.find('span', class_ = 'writer').text
            writer.append(writer_temp)

            society.append(cont.find('span', class_ = 'assigned').text)

            year.append(cont.find('p', class_='etc').find_all('span')[2].text)  # <p class='etc'>에서 3번째 span tag에 있는 텍스트

            journal.append(cont.find('p', class_='etc').find_all('span')[3].text)

            link.append("http://www.riss.kr"+cont.find('p', class_='title').a['href']) # <p, class='title'>의 a 태그의 'href' 속성 값

            if cont.find('p', class_='preAbstract') :
                abstracts.append(cont.find('p', class_='preAbstract').text)

            else:
                abstracts.append("초록이 없습니다.")

        df = pd.DataFrame(
            {"title":title,
             "writer":writer,
             "society":society,
             "year":year,
             "journal":journal,
             "link":link,
            "abstracts":abstracts}
        )

        return df
    
    df = get_info()
    
    st.write("3. 검색 결과 확인")

        
    if st.checkbox("크롤링 시작!"):
        
        st.write("**논문 제목, 저자, 학회, 발행연도, 발행기관, 상세링크, 요약문**을 수집합니다.")
        
        st.dataframe(df)
        st.balloons()
        st.success("논문 수집에 성공하였습니다.")
        
    st.write("4. 빈도 분석 시각화")
    
    st.text("발행연도, 저자, 발행기관별로 빈도수를 분석합니다.! ")
    
    counts = st.selectbox('Select Columns',('select','year','writer','society'))
 

    font_path = "./font/NanumGothic.ttf"

    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    if counts=="select":
        pass

    if counts=="writer":
        st.write(df["writer"].value_counts())
        df['writer'].value_counts().head(10).plot(kind="barh")
        st.pyplot()

    if counts=="year":
        st.write(df["year"].value_counts())
        df['year'].value_counts().head(10).plot(kind="pie", label='', autopct='%.1f%%', shadow=True, cmap="rainbow", counterclock=False, startangle=90)
        st.pyplot()

    if counts=="society":
        st.write(df["society"].value_counts())
        df['society'].value_counts().head(5).plot(kind="pie", label='', autopct='%.1f%%', shadow=True, cmap="Blues", counterclock=False, startangle=90)
        st.pyplot()
        
     
    st.write("5. csv 파일로 저장")
    st.text("파일 이름은 '키워드+논문개수'로 자동 생성됩니다 예)'인공지능10.csv")
    
#     def get_table_download_link(df):

#         csv = df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#         href = f'<a href="data:file/csv;base64,{b64}" download="file.csv">Download csv file</a>'
#         return href
   
    if st.checkbox("논문 저장"):
        
        select=st.radio('어떤 파일로 저장할까요?', ['그냥 안할래요', 'excel파일','csv파일','둘 다'])

        if select == 'excel파일':
            df.to_excel(f'{keyword}{number}.xlsx', index=False)

        elif select == 'csv파일':
            df.to_csv(f'{keyword}{number}.csv', index=False)

        elif select == '둘 다':
            df.to_csv(f'{keyword}{number}.csv', index=False)
            df.to_excel(f'{keyword}{number}.xlsx', index=False)
        else :
            pass

#         st.markdown(get_table_download_link(df), unsafe_allow_html=True)
 

    st.write("저장된 파일을 확인하시고, 사이드 메뉴에서 워드클라우드도 시도해보세요!! :sunglasses:")
    
elif choice == "Word Cloud":
    
    st.write("### 크롤링으로 저장한 csv파일을 불러와서 분석을 시작해봅시다! ###")
    data_file = st.file_uploader("Upload CSV",type=['csv'])

    if data_file is not None:
        df = pd.read_csv(data_file)
        st.dataframe(df)


    if st.checkbox("단어 빈도수 분석"):

        st.text("제목에서 사용된 단어 분석 : 상위 50개 명사 출력")

        from konlpy.tag import Okt  
        okt=Okt() 

        title = df["title"]

        nouns=[]
        stop_words=''
        for words in title: 
            for noun in okt.nouns(words):
                if noun not in stop_words and len(noun)>1: 
                    nouns.append(noun) 


        from collections import Counter

        num_top_nouns = 50
        nouns_counter = Counter(nouns)
        top_nouns = dict(nouns_counter.most_common(num_top_nouns))
        st.text (top_nouns)

        st.text("텍스트분석에서 제거할 단어, 즉 불필요한 단어를 띄어쓰기 기준으로 적어주세요.")

        nouns=[]
        stop_words= st.text_input("불용어: ")


        for words in title: 
            for noun in okt.nouns(words):
                if noun not in stop_words and len(noun)>1: 
                    nouns.append(noun) 

        num_top_nouns = 200
        nouns_counter = Counter(nouns)
        top_nouns = dict(nouns_counter.most_common(num_top_nouns))


    if st.checkbox("WordCloud 생성"):
        
        bg_color = st.selectbox("배경색을 선택하세요",['white','yellow','green','grey', 'black'])
    
        max_number = st. slider("최대 글자 수를 설정해보세요.", 50, 200)
        
        
        random_number = st.slider("슬라이드를 움직여 바뀌는 그림을 확인해보세요.",0,100)


        wordcloud = WordCloud (
        max_words= max_number,  
        background_color = bg_color,  
        random_state = random_number,

        font_path = "./font/NanumBarunGothic.ttf")

        wc = wordcloud.generate_from_frequencies(top_nouns)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(wc, interpolation="bilinear")     
        plt.axis('off')    ## 가로, 세로축을 별도로 표시하지 않음.
        # plt.savefig('bts.png')    ## 그림을 저장함.
        st.pyplot()
        


elif choice == "Topic Modeling":
    

    
    st.write("### 토픽모델링을 시작해봅시다! ###")
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.dataframe(df)


    if st.checkbox("토픽모델링"):  
        
        from konlpy.tag import Okt  

        import gensim.corpora as corpora
        from gensim.models.ldamodel import LdaModel 

        okt=Okt() 

        title_text = df.title

        stop_words=""

        tokenized_corpus=[]

        for words in title_text: 
            title_nouns = [] 
            for noun in okt.nouns(words):
                if noun not in stop_words and len(noun)>1: 
                    title_nouns.append(noun)

            tokenized_corpus.append(title_nouns)


        dictionary = corpora.Dictionary(tokenized_corpus)
        corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]
        
        import gensim
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=7)
        st.write(model.print_topics())


    if st.checkbox("토픽모델링 시각화"):  

        import pyLDAvis
        import pyLDAvis.gensim as gensimvis
        prepared_data = gensimvis.prepare(model, corpus, dictionary)
    #         pyLDAvis.display(prepared_data)

        html_string = pyLDAvis.prepared_data_to_html(prepared_data)
        from streamlit import components
        components.v1.html(html_string, width=800, height=400, scrolling=True)


elif choice == "Pubmed 논문 수집(영문)":
    
        

    keyword1 = st.text_input("검색할 논문의 키워드를 입력하세요(예:corona19): ")
    number1 = st.number_input("검색할 논문의 개수를 입력하세요: ", min_value=10, max_value=1000, step=10)
    number1 =  int(number1/10)
    
    if st.checkbox("검색 결과 확인!"):
    
        url = f'https://pubmed.ncbi.nlm.nih.gov/?term={keyword1}&page={number1}'
        result = requests.get(url, headers=headers)
        if result.status_code == 200:

            st.success("이제, 논문 크롤링을 할 수 있습니다.")
        else :
            st.text("woops! 다음에 다시 시도해주세요.")

 
    
    @st.cache
    def get_pubmed():
    
        url = f'https://pubmed.ncbi.nlm.nih.gov/?term={keyword1}&page={number1}'

        result = requests.get(url, headers=headers)
        soup = bs(result.content, 'html.parser')
        

        title =[]
        author = []
        journal_citation = []
        PMID = []
        link = []

        for i in range(1, number1+1):

            url = f'https://pubmed.ncbi.nlm.nih.gov/?term={keyword1}&page={number1}'
            result = requests.get(url, headers=headers)
            soup = bs(result.content, 'html.parser')
            contents = soup.find_all('div', class_='docsum-content')

            for info in contents:

                title.append(info.find('a', class_="docsum-title").text.strip()) 
                author.append(info.find('span', class_='docsum-authors short-authors').text.strip())
                journal_citation.append(info.find('span', class_='docsum-journal-citation full-journal-citation').text.strip())
                PMID.append(info.find('span', class_="citation-part").text.strip().split(':')[1].strip())
                link.append("https://pubmed.ncbi.nlm.nih.gov/"+info.find('a',class_="docsum-title")['href'])

            time.sleep(0.5)


        df1 = pd.DataFrame(
                {"title":title,
                 "author":author,
                 "journal_citation":journal_citation,
                 "PMID":PMID,
                 "link":link}
            )

        return df1
    
    df1 = get_pubmed()

    if st.checkbox("크롤링 시작!"):
        
        st.write("**논문 제목, 저자, 저널인용횟수, PMID, 상세링크** 정보를 수집합니다.")
        
        st.dataframe(df1)
        st.balloons()
        st.success("논문 수집에 성공하였습니다.")
        
    st.write("파일로 저장")
    st.text("파일 이름은 '키워드+논문개수'로 자동 생성됩니다 예)'인공지능10.csv")
    

    if st.checkbox("논문 저장"):
        
        select=st.radio('어떤 파일로 저장할까요?', ['그냥 안할래요', 'excel파일','csv파일','둘 다'])

        if select == 'excel파일':
            df1.to_excel(f'{keyword1}{number1}.xlsx', index=False)

        elif select == 'csv파일':
            df1.to_csv(f'{keyword1}{number1}.csv', index=False)

        elif select == '둘 다':
            df1.to_csv(f'{keyword1}{number1}.csv', index=False)
            df1.to_excel(f'{keyword1}{number1}.xlsx', index=False)
        else :
            pass

    
