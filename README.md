# processa-doc-PF

Este projeto de como objetivo identificar nomes de peritos em textos publicados em PDF.

Para me auxiliar nessa identificação:
1) Treino um modelo de doc2vec com os textos dos PDFs para gerar vetores do documento
2) Treino um modelo de classificação (HistGradientBoostingClassifier) utilizando os vetores do doc2vec para classificar o texto do PDF.
3) Salvo os modelos com os melhores resultados.
4) Após transformar o pdf em texto, aplico os modelos criado que devem me retornar o texto classificado, e de acordo com a classificação separo as portarias.
5) Utilizo um modelo de NLP ja treinado (biblioteca spacy) para então treinar novamente com os nomes dos peritos para melhorar o resultado do modelo.
6) Em cada portaria aplico o NLP que treinei que me retorna as "entidades nomeadas" que ele identificou, comparo com uma lista de nomes de peritos para verificar se aquela "entidade" é um nome de um perito.
7) Caso afirmativo adiciono o texto da portaria, o nome do perito e o link do PDF em um DataFrame.

descrição:
- doc2vec: É um algotirmo não-supervisionado para gerar vetores para frases, paragrafos ou documentos (Representações distribuídas de sentenças e documento)
- HistGradientBoostingClassifier: Árvore de classificação de aumento de gradiente baseada em histograma. Para grandes conjuntos de dados (n_samples >= 10.000)
- NLP spacy: Algoritmo para reconhecimento de entidades nomeadas. O reconhecedor de entidade identifica intervalos rotulados de tokens não sobrepostos. O algoritmo baseado em transição usado codifica certas suposições que são eficazes para tarefas de reconhecimento de entidades nomeadas “tradicionais”
