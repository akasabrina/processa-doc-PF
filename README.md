# processa-doc-PF

Este projeto de como objetivo identificar nomes de peritos em textos publicados em PDF.

Para me auxiliar nessa identificação:
1) Treinei um modelo de doc2vec com os textos dos PDFs para gerar vetores do documento
2) Treinei um modelo de classificação (HistGradientBoostingClassifier) utilizando os vetores do doc2vec para classificar o texto do PDF.
3) Salvei os modelos com os melhores resultados
4) Após transformar o pdf em texto, aplico os modelos criado que devem me retornam o texto classificado, e de acordo com a classificação separo as portarias.
5) Em cada portaria aplico o NLP (da biblioteca spacy) e utilizo apenas as strings classificadas como "PERSON", comparo com uma lista de nomes de peritos para verificar se aquela string
é um nome de um perito (Guardo essa comparação em um DataFrame para utlizar no próximo passo)
6) Caso afirmativo adiciono o texto da portaria, o nome do perito e o link do PDF em um DataFrame.

descrição:
- doc2vec: É um algotirmo não-supervisionado para gerar vetores para frases, paragrafos ou documentos (Representações distribuídas de sentenças e documento)
- HistGradientBoostingClassifier: Árvore de classificação de aumento de gradiente baseada em histograma. Para grandes conjuntos de dados (n_samples >= 10.000)
- NLP spacy: Algoritmo para reconhecimento de entidades nomeadas. O reconhecedor de entidade identifica intervalos rotulados de tokens não sobrepostos. O algoritmo baseado em transição usado codifica certas suposições que são eficazes para tarefas de reconhecimento de entidades nomeadas “tradicionais”
