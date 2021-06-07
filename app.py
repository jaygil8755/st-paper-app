
import streamlit as st

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import base64
import time


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}

# st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.sidebar.header('메뉴를 선택하세요')
menu1 = ["Riss 논문 수집(한글)", "Pubmed 논문 수집(영문)"]
choice = st.sidebar.selectbox("Menu",menu1)

st.title("논문 데이터 자동 수집 어플리케이션")

if choice == "Riss 논문 수집(한글)":

    st.subheader("riss에서 국내학술논문을 수집하여 파일로 저장할 수 있어요")

    expander_bar = st.beta_expander("Quick Guide")
    expander_bar.markdown("""
    1. 왼쪽 사이드바 메뉴에서 서비스 메뉴를 선택할 수 있습니다. 
    2. 먼저 논문 정보 수집을 선택한 후, 검색어와 수집할 논문개수를 입력합니다. 
    3. 수집된 논문은 Pandas DataFrame으로 볼 수 있고, csv 파일로 저장할 수 있습니다.
   
    ** 본 서비스는 교육/연구용으로 제공되는 것으로 결과에 대해 어떠한 책임도 지지 않습니다. 
    저작권에 대한 책임도 이용자 본인에게 있습니다.**
    """)
    expander_bar.video('https://youtu.be/ch83Zvl2icM') 
    
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
  
    st.write("4. csv 파일로 저장")
   
    @st.cache
    def download_link(object_to_download, download_filename, download_link_text):

        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


    if st.button('Download Dataframe as CSV'):
        tmp_download_link = download_link(df, f'{keyword}.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    

elif choice == "Pubmed 논문 수집(영문)":
    
    st.subheader("Pubmed에서 해외학술논문을 수집하여 파일로 저장할 수 있어요")
    
        

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
    


    if st.checkbox("크롤링 시작!"):
        
        st.write("**논문 제목, 저자, 저널인용횟수, PMID, 상세링크** 정보를 수집합니다.")
        df1 = get_pubmed()
        
        st.dataframe(df1)
        st.balloons()
        st.success("논문 수집에 성공하였습니다.")
        


    @st.cache
    def download_link(object_to_download, download_filename, download_link_text):
 
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


   
    st.write("csv 파일로 저장")

    if st.button('Download Dataframe as CSV'):
        tmp_download_link = download_link(df1, f'{keyword1}.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
