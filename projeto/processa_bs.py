# Bibliotecas
import  os
import  subprocess

import  pandas   as pd
import  numpy    as np
import  re

import  joblib
import  json

from    tqdm                  import tqdm
from    unidecode             import unidecode

from    gensim.models.doc2vec import TaggedDocument
from    spacy                 import load
from    config                import FOLDER_BS

import  warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Constantes
NLP = load("pt_core_news_lg")
CLASSIFIER = joblib.load(r"scripts\models\model_clf2.pkl")
MODEL_D2V = joblib.load(r"scripts\models\model_d2v.pkl")

SUBST = re.compile('\\x0c')
SUBST2 = re.compile(r"(\W)|(\S*@\S*\s?)|(http\S+)|(www\S+)|(\d+)|(\\n)|[ªº_]")
RE_MINUS = re.compile(r'(\s([a-z][A-Z]{2,})|(\sDAS?\s)|(\sDES?\s)|(\sDOS?\s))')
RE_MAIUS = re.compile(r'(([A-Z]{2,}\s?){2,})')

PCFS_CODIGO = pd.read_excel(r"scripts\peritos_codigo.xlsx")

# Funções
def pdf_to_dataframe(fname) -> pd.DataFrame:
    """Convert online pdf to text using a complete URL, pdftotext module and tempfile. Convert text to DataFrame.

    Args:
        fname (str): The complete url.

    Raises:
        subprocess.CalledProcessError: Could not convert using pdftotext.

    Returns:
        pd.DataFrame: a text DataFrame.
    """
    # Transforma de PDF para TXT
    txtfn = fname.replace('.pdf', '.txt')
    if not os.path.isfile(txtfn):
        try:
            cmd = r'pdftotext -layout -raw -enc UTF-8 "{}"'.format(fname)
            output, error = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()
            if error:
                raise subprocess.CalledProcessError # type: ignore
        except subprocess.CalledProcessError as e:
            print('[Called Process Error] Could not convert using pdftotext. Technical Details given below.')
            print(str(e))

    # Transforma de TXT para DataFrame
    with open(fname.replace('.pdf', '.txt'), 'r', encoding="utf8") as file:
        corpus = []
        for lin in file:
            corpus.append(lin)

    df_texto = pd.DataFrame(corpus, columns=['texto'])
    df_texto.texto.replace(SUBST, "", regex=True, inplace=True) # type: ignore

    return df_texto


def processa_df(df_texto: pd.DataFrame, file_path: str, fname: str):
    """ Faz o pré-processamento do DataFrame\n
    Aplica o doc2vec, classificando cada linha, sendo: [1: "linhas a ignorar", 2: "inicio de portaria", 3: "texto da portaria"]\n
    Separa as portaria apenas com as linhas classificadas como 2 e 3\n
    É utilizado o NLP para identificar pessoas no documento, depois verifica se o nome da pessoa é de um PCF

    Args:
        df_texto (pd.DataFrame): Text in a DataFrame.
        file_path (str): URL completa do arquivo.
        fname (str): File name.

    Returns:
        list: 1º lista --> Texto portaria, Código portaria e Caminho do arquivo.\n
        2º lista --> Código PCF e Código portaria.
    """
    # Pre-processamento do texto do DataFrame
    df_texto['texto_limpo'] = df_texto.texto.replace(SUBST2, " ", regex = True) # type: ignore
    df_texto.texto_limpo.apply(lambda x: unidecode(x))
    df_texto['texto_limpo2'] = df_texto.texto_limpo.apply(lambda x: x.split())

    # doc2vec
    tagged_doc = [TaggedDocument(texto, [i]) for i, texto in enumerate(df_texto.texto_limpo2)]
    x = np.array([MODEL_D2V.infer_vector(tagged_doc[i][0], alpha=0.25, min_alpha=0.07) for i in range(len(tagged_doc))])

    # Classifica cada linha do texto
    df_texto["predict_lin"] = CLASSIFIER.predict(x)

    # Separa cada portaria
    lst_portarias_txtlimpo = []
    txt_port1 = ''
    lst_portarias_txt = []
    txt_port2 = ''
    for lin in df_texto.iterrows():
        if lin[1].predict_lin == 3:

            txt_port1 += lin[1].texto_limpo
            txt_port2 += lin[1].texto

        elif lin[1].predict_lin == 2:

            lst_portarias_txtlimpo.append(txt_port1)
            txt_port1 = lin[1].texto_limpo
            lst_portarias_txt.append(txt_port2)
            txt_port2 = lin[1].texto

    lst_portarias_txtlimpo.append(txt_port1)
    lst_portarias_txt.append(txt_port2)
    
    pcfs_codigo = PCFS_CODIGO.copy()
    pcfs_codigo.nome_perito = pcfs_codigo.nome_perito.apply(lambda x: unidecode(x).lower())    
    nome_codigo = pd.DataFrame()

    # passa de portaria em portaria aplicando o NLP
    for portaria in lst_portarias_txtlimpo:
        
        for r in re.findall(RE_MINUS, portaria):
            portaria = portaria.replace(r[0], r[0].lower())
        for r in re.findall(RE_MAIUS, portaria):
            portaria = portaria.replace(r[0], r[0].title())

        # Aplica o NLP na portaria
        # Transforma em DataFrame os que forem classificados como PER = Pessoa
        doc = NLP(portaria)
        for ent in doc.ents:
            if ent.label_ == 'PER':

                nome_procurado = unidecode(ent.text).lower()
                pcfs_codigo[nome_procurado] = pcfs_codigo.nome_perito.apply(lambda x: x in nome_procurado)

        # nome_codigo recebe um DataFrame com os nomes e os codigos dos PCFs que estão na portaria 
        nome_codigo = pcfs_codigo[pcfs_codigo.iloc[:,2:].any(axis=1)][["nome_perito", "codigo_de_barras_perito"]]

    df_auxi = pd.DataFrame({"texto": lst_portarias_txt,"texto_limpo": lst_portarias_txtlimpo})
    df_auxi.texto_limpo = df_auxi.texto_limpo.apply(lambda x: x.lower())  

    # verifica qual PCF esta na portaria e coloca o código do PCF correspondente
    for j in range(len(nome_codigo)):

        nome = nome_codigo.iloc[j]["nome_perito"]
        cod = nome_codigo.iloc[j]["codigo_de_barras_perito"]
        df_auxi[nome] = df_auxi.texto_limpo.apply(lambda x: cod if nome in x else 0)

    # apenas as linhas que tem algum código de PCF ficam
    df_auxi = df_auxi[df_auxi.iloc[:,2:].any(axis=1)]

    # junta os códigos da mesma linha em uma nova coluna "codigos"
    df_auxi["codigos"] = df_auxi.apply(lambda row: [valor for valor in row[2:] if pd.notnull(valor) and valor > 0], axis=1).tolist()
    df_auxi.reset_index(inplace=True, drop=True)

    lst_txt_codPort_path = []
    lst_codPCF_codPort = []
    for i in range(len(df_auxi)):

        # Cria um dicionário com o Texto da portaria, Código unico da portaria e o Caminho do arquivo
        dict1 = {"Texto": df_auxi.texto[i], "CodigoPortaria": fname+f"-{i+1:03d}", "Path": file_path}
        lst_txt_codPort_path.append(dict1)

        for cod in df_auxi.codigos[i]:

            # Cria um dicionário com o Código do PCF e o Código único da portaria
            dict2 = {"CodigoPCF": cod, "CodigoPortaria": fname+f"-{i+1:03d}"}
            lst_codPCF_codPort.append(dict2)

    return lst_txt_codPort_path, lst_codPCF_codPort


def processa_file(pasta: str, txt_codPort_path, codPCF_codPort, reprocessa = False):
    """Abre a pasta e processa arquivo por arquivo\n
    Caso o arquivo já tenha sido processado é pulado

    Args:
        pasta (str): Caminho da pasta que contém os documentos.
        txt_codPort_path: texto, código único e caminho da portaria.
        codPCF_codPort: código do PCF e código único da portaria.

    Returns:
        list: 1º lista --> Texto portaria, Código portaria e Caminho do arquivo.\n
        2º lista --> Código PCF e Código portaria.
    """
    for dirpath, _, filenames in os.walk(pasta):
        print(dirpath)

        # Abre arquivos já processados 
        pross_json = dirpath + '\\~processados.json'
        if os.path.isfile(pross_json) and not reprocessa:
            with open(pross_json, 'r') as json_file:
                processados_pasta = json.load(json_file)
        else:
            processados_pasta = []

        # Processa arquivos da pasta
        for file in tqdm(filenames):

            # Pula arquivos que não são pdf
            if file[-3:] not in 'pdf':
                pass

            # Pula arquivos ja processados
            elif file[:-4] in processados_pasta:
                pass
            
            # processa arquivos
            else:
                path = dirpath + "\\" + file
                txtfn = file.replace('.pdf', '')
                df = processa_df(pdf_to_dataframe(path), path, txtfn)
                if df:
                    txt_codPort_path = txt_codPort_path.append(df[0])
                    codPCF_codPort = codPCF_codPort.append(df[1])

                # Adiciona o arquivo à lista de processados
                processados_pasta.append(file[:-4])
            
        # Salva na pasta a lista de arquivos já processados
        with open(pross_json, 'w') as json_file:
            json.dump(processados_pasta, json_file)

    return txt_codPort_path, codPCF_codPort


def processa_portaria(reprocessa = False):

    # Recupera dados já processados
    pross_h5 = FOLDER_BS + '\\processados.h5'
    if os.path.isfile(pross_h5) and not reprocessa:
        txt_codPort_path = pd.read_hdf(pross_h5, 'txt_codPort_path')
        codPCF_codPort = pd.read_hdf(pross_h5, 'codPCF_codPort')
    else:
        txt_codPort_path = pd.DataFrame()
        codPCF_codPort = pd.DataFrame()

    # processa arquivo
    txt_codPort_path, codPCF_codPort = processa_file(FOLDER_BS, txt_codPort_path, codPCF_codPort, reprocessa)
    txt_codPort_path.reset_index(drop=True, inplace=True)
    codPCF_codPort.reset_index(drop=True, inplace=True)
    
    # salva processamento
    txt_codPort_path.to_hdf(pross_h5, key = 'txt_codPort_path', mode = 'w')
    codPCF_codPort.to_hdf(pross_h5, key = 'codPCF_codPort', mode = 'a')
    PCFS_CODIGO.to_hdf(pross_h5, key = 'nomePCF_codPCF', mode = 'a')

if __name__ == "__main__":
    processa_portaria()