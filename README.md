# processa-doc-PF

O processa-doc-PF é um projeto desenvolvido para identificar nomes de peritos em textos extraídos de documentos PDF. A solução utiliza técnicas de Processamento de Linguagem Natural (NLP) e aprendizado de máquina para realizar a extração e classificação de informações relevantes, organizando os dados em um formato estruturado.

## Funcionalidades
**1. Geração de Vetores com doc2vec**

Os textos extraídos dos PDFs são processados com o modelo doc2vec, que converte documentos em representações vetoriais distribuídas. Isso permite capturar a semântica dos textos para análises subsequentes.

**2. Classificação com HistGradientBoostingClassifier**

Após a vetorização, um modelo de classificação baseado no HistGradientBoostingClassifier é treinado para identificar e classificar o conteúdo dos PDFs. Os modelos com melhor desempenho são salvos para utilização futura.

**3. Extração e Classificação de Portarias**

Após a transformação dos PDFs em texto, os modelos treinados são aplicados para classificar o texto e separar as portarias relevantes.
Um modelo de NLP pré-treinado (utilizando a biblioteca spaCy) é ajustado com uma lista personalizada de nomes de peritos para aprimorar o reconhecimento.

**4. Reconhecimento de Entidades Nomeadas**

- O modelo NLP identifica "entidades nomeadas" nos textos das portarias.
- As entidades detectadas são comparadas com uma lista de nomes de peritos.
- Quando há uma correspondência, o texto da portaria, o nome do perito e o link do PDF são armazenados em um DataFrame.

## Benefícios do Ajuste do Modelo NLP
O modelo NLP padrão apresentou dificuldades em identificar nomes completos de peritos. Para superar essa limitação, o modelo foi ajustado com uma lista personalizada de nomes de peritos, resultando em uma melhoria significativa no desempenho, alcançando 90% de precisão na identificação correta.

## Principais Tecnologias Utilizadas
- **doc2vec:** Algoritmo não supervisionado que gera representações vetoriais distribuídas para frases, parágrafos ou documentos, facilitando a análise semântica.
- **HistGradientBoostingClassifier:** Modelo de classificação de árvore baseado em aumento de gradiente com histograma, ideal para grandes conjuntos de dados (n_samples ≥ 10.000).
- **spaCy:** Biblioteca de NLP utilizada para reconhecimento de entidades nomeadas. Seu algoritmo baseado em transição foi ajustado para tarefas específicas do projeto.
 
## Fluxo de Trabalho
1. Extração do texto dos PDFs.
2. Vetorização do texto com doc2vec.
3. Treinamento e aplicação do HistGradientBoostingClassifier para classificar os documentos.
4. Separação e identificação das portarias classificadas.
5. Ajuste e aplicação do modelo spaCy para identificar os nomes de peritos.
6. Armazenamento dos dados estruturados em um DataFrame (texto da portaria, nome do perito e link do PDF).

*Este projeto foi desenvolvido inteiramente durante meu estágio na Policia Federal.*
