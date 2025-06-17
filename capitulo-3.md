---
title: Capítulo 3 
layout: page
---

# O Processo por trás de uma Inteligência Artificial

## A essência da IA: Entrada, Saída e o Aprendizado de Máquina

Na computação, de modo geral, o caminho para obter um resultado é claro
e direto: temos a **entrada**, a **aplicação de regras predefinidas**, e
a **saída**, que é o resultado dessa operação.

Podemos visualizar esse processo como um fluxo linear e determinístico:

> Entrada → Regras Predefinidas → Saída

Aqui, as regras – ou instruções – são explícitas e fixas, ditando
exatamente como a entrada deve ser processada para gerar a saída
desejada. Pense na receita de um bolo: as entradas são os ingredientes,
e a saída, o bolo em si. Podemos chamar estas regras de **algoritmos**,
nome dado a uma sequência de passos lógicos e ordenados aplicados para
resolver um problema ou realizar uma tarefa.

No universo da Inteligência Artificial, a dinâmica é um pouco diferente.
Em vez de regras programadas manualmente, a IA aprende a descobrir
padrões a partir de exemplos. Aqui temos:

> Entrada → Saída → Regras descobertas e aprendidas!

Esse processo de aprendizado é o que chamamos de **Aprendizado de
Máquina** no Brasil, ou, como é mais amplamente conhecido, **Machine
Learning (ML)**.

Imagine que você quer ensinar um sistema a diferenciar gatos de
cachorros:

-   **Entrada:** Milhares de fotos de animais.

<!-- -->

-   **Saída:** Os rótulos "gato" ou "cachorro".

<!-- -->

-   **Processo:** A IA analisa sozinha características como formato das
    orelhas, textura da pelagem e tamanho do focinho, descobrindo como
    distinguir as espécies.

Aqui também temos algoritmos, mas, em vez de seguirem regras fixas, eles
seguem cálculos e otimizações pré-definidas. A capacidade de identificar
padrões não se limita a fotos de animais — ela é crucial em muitas
áreas. No contexto de saúde pública, por exemplo, podemos contar com
informações como:

-   Histórico de milhões de consultas e exames
-   Dados climáticos e fatores socioeconômicos
-   Informações de sensores e dispositivos

Um especialista, por mais experiente que seja, enfrenta limitações ao
processar manualmente a imensa quantidade de informações disponíveis. A
IA, por outro lado, consegue analisar milhares de dados, identificar
padrões complexos e conexões sutis que poderiam passar despercebidos.
Dessa forma, ela **complementa o conhecimento humano**, ampliando a
visão além da experiência individual do especialista. Com essa riqueza
de informações, os resultados se tornam mais personalizados e adaptáveis
ao contexto clínico e às particularidades locais, aprimorando
significativamente a tomada de decisões.

<!--# to-do: SLIDES - algo que dê essa ideia de oque nós sabemos VS. o quetem no mundo -->

O segredo está em como esses algoritmos aprendem!

------------------------------------------------------------------------

## Navegando no Mundo da IA: Machine Learning, Deep Learning, GenAI e LLM

**Inteligência Artificial** é um campo vasto e repleto de
possibilidades, organizado em diversas camadas. Para entender o que está
por trás de sistemas como o ChatGPT ou simplesmente compreender melhor
todo o *hype* em torno da IA, precisamos passar por pelo menos três
camadas: **Machine Learning**, **Deep Learning** e **IA Generativa**.

<!-- to-do: SLIDES - o frame das bolinhas -->

Nos próximos capítulos, nos aprofundaremos em cada uma delas. Mas, para
iniciarmos nossa jornada, vamos começar com o **Machine Learning**.

### Machine Learning: A Fundação do Aprendizado por Dados

É aqui que tudo começa. O **ML** são as técnicas utilizadas para ensinar
as máquinas a partir de dados. Para isso, temos uma variedade de
algoritmos de aprendizagem – cada um com seus pré-requisitos, pontos
fortes e limitações. Profissionais de IA, como cientistas de dados,
precisam conhecer e dominar essas opções. Em termos aplicados, estes
algoritmos nos permitem:

-   Criar **Sistemas de Recomendação**, como os da Netflix ou Spotify.

<!-- -->

-   Detectar **fraudes**, monitorando padrões suspeitos em transações
    bancárias.

<!-- -->

-   **Segmentar clientes**, agrupando-os com base em características
    comuns para otimizar ações de marketing, equilibrando personalização
    de serviços, objetivos estratégicos e execução em escala.

O próximo conceito que vamos discutir é o **Deep Learning**, ou, em
português, **Aprendizado Profundo**.

### Deep Learning: A Evolução para Problemas Complexos

Em termos técnicos, o **Deep Learning** trata-se de mais uma opção na
"prateleira" de ML, mas na prática pode ser visto como uma evolução —
podemos pensar nele como uma **versão turbinada do Machine Learning**.
Pois, enquanto algoritmos tradicionais de ML funcionam bem com padrões
básicos e de média complexidade, o Deep Learning se destaca na resolução
de **problemas de alta complexidade**.

O Deep Learning se inspira na estrutura do cérebro humano, com várias
interconexões. Lembra do modelo Entrada → Regras Aprendidas → Saída?
Aqui, temos **múltiplas camadas de aprendizado**, o que justifica o
termo "Deep" (Profundo).

> Entrada → Regras Aprendidas → … → Regras Aprendidas → Saída

Na prática, essa capacidade faz do Deep Learning uma escolha
interessante para tarefas que envolvem desafios mais complexos, como:

-   **Reconhecimento de imagens** avançado.

<!-- -->

-   **Tradução automática de áudios** com maior precisão.

<!-- -->

-   De modo geral, procurar **superar os humanos em cenários** em que o
    Machine Learning tradicional ainda não tinha conseguido, como o
    modelo de Deep Learning desenvolvido pela DeepMind, uma empresa do
    Google, que venceu jogadores de Go — um dos jogos mais difíceis do
    mundo. Em 2016, o modelo denominado **AlphaGo** conseguiu alcançar o
    feito de derrotar o lendário campeão Lee Sedol, sem dúvida mais um
    grande marco na história da IA.

Agora, vamos seguir para a próxima camada: a **IA Generativa**, ou, em
inglês, *Generative AI* (ou GenAI).

### IA Generativa: A Capacidade de Criação

Aqui a IA não apenas reconhece padrões, mas também é capaz de
**criá-los**. A **IA Generativa** produz texto, imagens, sons e até
código de programação. Algumas aplicações práticas incluem:

-   **Geradores de imagens** que criam ilustrações a partir de uma
    descrição textual.

<!-- -->

-   **Composições de músicas**, tanto melodia quanto letra, a partir de
    referências dadas pelo usuário.

<!-- -->

-   **Modelos de texto** como o ChatGPT, que escrevem desde receitas de
    bolo até documentos jurídicos.

Dentro da GenAI, vale destacar os **Large Language Models (LLMs)** ou,
em português, Grandes Modelos de Linguagem — modelos treinados
especificamente para lidar com a linguagem natural. **GPT**, o modelo
por trás do ChatGPT, é um LLM e também um ótimo exemplo do que este tipo
de modelo pode oferecer.

Mas lembre-se: a **IA Generativa**, apesar do nome, é uma aplicação
avançada do **Deep Learning**, focada em criar novos conteúdos, que, por
sua vez, está dentro do guarda-chuva maior do **Machine Learning**, que
faz parte do campo da **Inteligência Artificial**.

------------------------------------------------------------------------

## As Etapas Essenciais do Aprendizado de uma IA: Preparo de Dados, Treino, Teste e Avaliação

O aprendizado de um modelo de Machine Learning não é mágica, mas sim um
método sistemático que nos permite transformar dados em conhecimento.
Vamos conhecer quatro de suas principais etapas: **preparação dos dados,
treinamento, teste e avaliação**. Vamos falar sobre cada uma delas.

### 1. Preparação dos Dados: A Base do Conhecimento

Tudo começa com a preparação dos dados, pois raramente eles chegam
prontos para uso. Treinar um modelo com dados ruins é como estudar para
uma prova com um livro cheio de páginas rasgadas — você perde
informações cruciais. Por isso, a primeira etapa consiste em os dados
passarem por um verdadeiro “banho de loja”: **corrigir erros, padronizar
formatos, preencher lacunas e garantir que representem a realidade de
forma equilibrada**. Além disso, essa fase permite um olhar mais
profundo sobre as informações em si, pois acabamos passando por etapas
como a Análise Exploratória de Dados, onde aprendemos com os dados por
meio de ferramentas como a Estatística Descritiva. Um exemplo: se
treinarmos um modelo para diagnosticar doenças usando majoritariamente
dados de homens jovens, ele pode falhar gravemente ao analisar mulheres
ou idosos — como já aconteceu com alguns algoritmos de diagnóstico
cardíaco no passado.

### 2. Treinamento do Modelo: O Coração do Aprendizado

Com os dados preparados, começamos o aprendizado em si, ou a fase de
**treinamento**. Aqui, o algoritmo analisa os dados para aprender seus
padrões e ajustar seus parâmetros internos. Cada algoritmo aprende de
uma maneira diferente. Alguns, inclusive, possuem configurações extras
chamadas **hiperparâmetros** — como "ajustes finos" que controlam
detalhes do aprendizado. O processo de otimizar esses ajustes para
melhorar o desempenho do modelo é chamado de *tuning*. Podemos pensar
nisso como uma experiência culinária: os ingredientes são os dados, a
receita representa os algoritmos e, mesmo usando os mesmos insumos, o
resultado pode variar conforme ajustes como "sal a gosto" — esses são os
hiperparâmetros. Todos esses refinamentos, combinados, fazem toda a
diferença na qualidade do modelo. E por isso, a variedade de opções e o
portfólio de algoritmos são essenciais para garantir que o modelo tenha
determinadas qualidades. Talvez a mais importante delas seja o **poder
de generalização**, ou seja, a capacidade do modelo de aplicar o que
aprendeu a novos dados, sem perder performance. Um modelo bem
generalizado reconhece gatos em fotos de diferentes raças, poses e
fundos. Se o modelo não generaliza bem, ele pode estar simplesmente
"decorando" os dados, o que chamamos de **superajuste** ou, em inglês,
*overfitting*. É como um estudante que sabe de cor todas as respostas de
um livro sem realmente entender os conceitos. Na hora da prova, se a
pergunta for ligeiramente diferente, ele não saberá responder. É como um
aluno chamado Chaves que só sabe responder quando o problema é com
maçãs, mas não com laranjas.

### 3. Teste do Modelo: Validando o Conhecimento

Para avaliar a capacidade do modelo de lidar com novos cenários, usamos
um conjunto de dados que ele nunca viu antes — o chamado **conjunto de
teste**. A sacada aqui é que, antes de iniciar todo o processo de
aprendizagem, segmentamos os dados em dois grupos: **treino e teste**. O
primeiro é utilizado para fazer tudo o que comentamos até aqui,
basicamente, ensinar o modelo. E os dados de teste são utilizados
posteriormente para verificar se o que o modelo aprendeu é suficiente
para manter bons resultados neste cenário inédito. Voltando ao exemplo
do estudante, é como apresentar questões completamente novas e avaliar
se ele ou ela mantém as performances obtidas até então. Existem outras
metodologias que contribuem para a criação de modelos eficazes, como
dados de validação e validação cruzada, entre outros. Tudo sempre com o
mesmo objetivo: garantir que o aprendizado seja generalizável.

### 4. Avaliação do Desempenho: Medindo o Sucesso e os Desafios

O que nos leva à quarta e última etapa: como medir o sucesso do
aprendizado? Não se trata apenas de acertar, pois existem **tipos
diferentes de acertos e de erros**.

Em diagnósticos médicos, um **falso negativo** ocorre quando uma doença
presente não é detectada, o que pode ter consequências graves. Por outro
lado, um **falso positivo** leva à identificação incorreta de uma doença
inexistente, resultando em exames desnecessários e ansiedade para o
paciente. O equilíbrio entre essas falhas depende do contexto da doença.
Existem métricas que nos ajudam a avaliar o desempenho do modelo. Duas
métricas comuns são: a **precisão**, que indica a proporção de casos
positivos reais corretamente classificados, e o **recall**, que mede a
capacidade de identificar todos os casos reais capturados. Inclusive,
existem outros nomes para estas métricas, como erro tipo I e II ou
sensibilidade e especificidade. Por fim, é importante ressaltar que este
processo é **iterativo**: seja porque o profissional responsável pode
revisitar as etapas, corrigindo erros e reforçando acertos, ou porque os
dados em si podem ser utilizados no retreino do modelo conforme novas
informações ficam disponíveis. E assim, o processo de modelagem,
treinamento ou simplesmente o “ciclo de aprendizado de máquina” se
completa — mas raramente se encerra. Modelos são refinados, dados são
atualizados e métricas reavaliadas, em um processo contínuo e iterativo
de melhoria.
