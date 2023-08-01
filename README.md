# processa-doc-PF

Este projeto de como objetivo identificar nomes de peritos em textos publicados em PDF.

Para me auxiliar nessa identificação:
1) Treinei um modelo de doc2vec com os textos dos PDFs para gerar vetores do documento
2) Treinei um modelo de classificação (HistGradientBoostingClassifier) utilizando os vetores do doc2vec para classificar o texto do PDF.
3) Salvei os modelos com os melhores resultados
4) Após transformar o pdf em texto, aplico os modelos que me retornam o texto classificado e de acordo com a classificação separo as portarias
5) Em cada portaria aplico o NLP e utilizo apenas as strings classificadas como "PERSON", comparo com uma lista de nomes de peritos para verificar se aquela string
é um nome de um perito
6) Caso afirmativo adiciono o texto da portaria, o nome do perito e o link do PDF em um DataFrame.
