---
title: 4. Aprendizado de Máquina (ML), suas segmentações e algoritmos clássicos
layout: page
editor_options: 
  markdown: 
    wrap: sentence
---

{: .no_toc }

## Índice
{: .no_toc .text-delta }

- TOC
{:toc}

<!-- to-do: ter um frame de ML para ir reforçando e resgatando ao longo do livro -->


## Tipos de Aprendizado no ML: Como a IA Aprende?

Já imaginou como a IA aprende?
Por trás das decisões inteligentes dos sistemas, há diferentes formas de aprendizado de máquina.
Duas das mais tradicionais e fundamentais são o **Aprendizado Supervisionado** e o **Aprendizado Não Supervisionado**.

### Aprendizado Supervisionado: Aprendendo com Exemplos

No **Aprendizado Supervisionado**, a IA aprende por meio de **exemplos previamente rotulados**.
Ou seja, usando dados que já possuem a "resposta" que nos interessa, o modelo irá aprender a prever qual deveria ser a resposta em situações futuras onde essa resposta não existe.

**Pense assim:**

-    Queremos classificar novas imagens tendo como base outras imagens que já estavam classificadas (ex: "isto é um gato", "isto é um cachorro").

-    Queremos rotular clientes como "bons" ou "maus pagadores" com base nos padrões observados em clientes anteriores para os quais temos essa identificação.

-    Ou ainda fazer a previsão do tempo tendo como base de aprendizado o histórico dos resultados climáticos.

É sobre ter exemplos rotulados, aprender os padrões e aplicá-los em cenários novos.
Essa abordagem nos permite **predizer números e categorias**.

### Aprendizado Não Supervisionado: Descobrindo Padrões Ocultos

Já no **Aprendizado Não Supervisionado**, a IA recebe dados **sem identificação prévia (sem rótulos)** e precisa encontrar padrões por conta própria.
A máquina busca estruturas e relações inerentes aos dados.

**Por exemplo:**

-    Podemos querer **segmentar clientes**, de modo a criar grupos com comportamentos de compra similares, permitindo a personalização de estratégias de marketing.

-   Ou **identificar transações suspeitas** ao reconhecer padrões diferentes do habitual, evitando possíveis fraudes e garantindo maior segurança financeira.

-   Uma aplicação muito comum: a **organização de músicas** que aplicativos fazem, criando *playlists* com base em características como ritmo, melodia e instrumentos.

Nesse tipo de aprendizado, o sistema de IA explora padrões escondidos nos dados — sem direcionamentos ou rótulos prévios — e revela estruturas que, talvez, nem sequer havíamos cogitado.

### Outras Abordagens: Expandindo as Possibilidades do ML

Existem também outras abordagens importantes que complementam as tradicionais:

-   **Aprendizado por Reforço:** Funciona por tentativa e erro, ajustando decisões com base nos *feedbacks* recebidos.
    É a estratégia utilizada para ensinar computadores a jogar jogos (como o **AlphaGo** que comentamos anteriormente, que venceu jogadores de Go), ou robôs a caminhar em diferentes terrenos.
    Outro exemplo são os **Sistemas de Publicidade Online** (como Google Ads) – onde algoritmos ajustam campanhas publicitárias com base no engajamento dos usuários, aprendendo quais anúncios funcionam melhor e otimizando automaticamente os investimentos.

-   **Aprendizado Semi-Supervisionado:** Combina aspectos do supervisionado e do não supervisionado.
    Ele utiliza um pequeno conjunto de dados rotulados para guiar o aprendizado sobre um grande volume de dados não rotulados.
    Essa estratégia é útil quando a rotulação manual de dados é cara ou difícil.
    Já notou como o **Google Photos** melhora ao corrigir suas sugestões?
    Isso é aprendizado semi-supervisionado em ação!
    Um sistema de reconhecimento de imagens que organiza automaticamente fotos e sugere identificação de rostos com base em um pequeno conjunto de imagens rotuladas.
    Os usuários podem confirmar ou corrigir esses rótulos, ajudando o modelo a melhorar suas previsões ao longo do tempo.

Ao entender essas abordagens, conseguimos visualizar como a IA aprende e melhora suas decisões em diferentes contextos, tornando-se cada vez mais eficaz e adaptável.

Inclusive, aplicações modernas podem combinar diferentes formas de aprendizado.
Um ótimo exemplo são os **sistemas de recomendação de filmes**.
Imagine que uma plataforma sugira *Black Mirror* após você assistir à série *Ruptura*.
Ela pode estar combinando:

-   **Aprendizado Supervisionado:** Baseando-se nas suas avaliações passadas.

<!-- -->

-   **Aprendizado Não Supervisionado:** Comparando seu comportamento com o de outros usuários com gostos similares.

<!-- -->

-   **Aprendizado por Reforço:** Observando sua reação – se você assistiu até o fim ou abandonou a série.

Vamos agora aprender um pouco mais sobre as particularidades das duas segmentações que conhecemos como Machine Learning Tradicional.

------------------------------------------------------------------------

## ML - Aprendizado Supervisionado em Detalhes: Classificação e Regressão

No Aprendizado Supervisionado, estamos diante de uma situação bastante familiar: temos uma pergunta clara e temos históricos das respostas para essa pergunta.
É como ensinar uma criança a identificar animais — mostramos uma imagem e dizemos se é um gato ou um cachorro.
A IA aprende da mesma forma: analisando dados que já vêm com suas respectivas "**respostas corretas**", também chamadas de **rótulos**.

Essa abordagem é extremamente comum em aplicações do dia a dia.
Por exemplo, em serviços de assinatura, podemos querer identificar padrões de *churn*, ou seja, prever quais clientes têm maior probabilidade de cancelar sua assinatura com base em comportamentos anteriores.
Para isso, alimentamos o modelo com dados históricos de clientes já classificados, e ele aprende a reconhecer os padrões que diferenciam os perfis.
Da mesma forma, se quisermos prever o valor de aluguel de uma casa, fornecemos informações, como tamanho e localização, sobre casas que sabemos esse valor, para que o modelo aprenda essa relação.

Existem dois grandes tipos de problemas nesse contexto:

-   **Classificação:** quando queremos prever **categorias**, ou seja, valores discretos (ex: inadimplente ou adimplente, paciente doente ou saudável, documento confidencial ou público).
-   **Regressão:** quando a previsão envolve **valores numéricos** (ex: preço de uma casa, vida útil de um produto eletrônico, ou ainda cálculo de tempo de entrega).

E para endereçar estes desafios, temos algoritmos como **Árvores de Decisão** e **Random Forest**, frequentemente utilizados para aprendizagem supervisionada.

As **Árvores de Decisão**, por exemplo, são como fluxogramas inteligentes, oferecendo uma abordagem visual das regras mapeadas.
Imagine que você quer prever o preço de uma casa.
Uma Árvore de Decisão faria perguntas diretas: 'Tem mais de 100m²?
Tem vista para o mar?'
 - e daria o valor com base nessas regras claras.

Já a **Random Forest** é um algoritmo que combina centenas dessas árvores, cada uma com perguntas ligeiramente diferentes, tornando as regras menos explícitas, mas garantindo previsões mais precisas e robustas.
Para o nosso exemplo, seria como consultar 100 corretores diferentes: cada um com seus critérios, e no final você tira a média.
O primeiro método é transparente, o segundo, menos interpretável, mas mais preciso, ou seja, acerta mais.

Ambos os algoritmos funcionam tanto para modelos de regressão quanto de classificação, a partir de pequenos ajustes.
No entanto, existem outros que não são abrangentes, como é o caso da **Regressão Linear**, que funciona com problemas de regressão, e a **Regressão Logística**, que funciona para problemas de classificação — confuso, né?
É aqui que vemos algumas intersecções entre áreas como a Estatística, Econometria e *Advanced Analytics* em geral, que por meio de processos e premissas um pouco diferentes, acabam por utilizar metodologias parecidas.

O fato é que existem muitos, muitos outros algoritmos, cada um com características próprias, adequando-se melhor a diferentes tipos de dados e desafios.
Falaremos mais sobre isso no final do capítulo.

### Avaliando o Desempenho no Aprendizado Supervisionado

Mas como avaliamos os resultados desses algoritmos?
Por meio de **métricas**, que são como "notas" que mostram se as previsões estão certas ou erradas.

Para **modelos de regressão**, que preveem números (como preço de casas ou temperatura), medimos o **erro** – a diferença entre o valor previsto e o real.
O segredo é equilibrar dois fatores:

-   **Viés:** Representa o quanto o seu modelo está consistentemente errado em suas previsões. É como um atirador que sempre erra para o mesmo lado.
-   **Variância:** Mede o quão as previsões do seu modelo mudam quando você usa diferentes conjuntos de dados de treinamento. É como um atirador com disparos espalhados ao redor do alvo.

Resumindo: **viés é erro consistente; variância é sensibilidade excessiva aos dados**.
Por exemplo, imagine um modelo tentando prever o preço de casas.
Se ele tiver alto viés, pode estimar R\$ 500 mil para casas que, na verdade, custam R\$ 300 mil, ignorando características relevantes como localização e tamanho.
Já um modelo com alta variância pode ser tão sensível a pequenos detalhes que alterará as estimativas drasticamente com mínimas variações nos dados.

Já para **modelos de classificação** (como "spam" ou "não-spam"), olhamos para as categorias, mas a questão é que os erros podem ser diferentes:

-   **Falsos Positivos:** Quando o modelo diz "sim", mas era "não" (por exemplo: marcar um e-mail importante como *spam*).
-   **Falsos Negativos:** Quando diz "não", mas era "sim" (por exemplo: deixar um *spam* passar para a caixa de entrada).

Note que, dependendo do contexto, uma métrica pode ser mais relevante do que outra.
Compreender o Aprendizado Supervisionado é essencial para mergulhar no universo da IA.
No próximo capítulo, vamos explorar a situação oposta: quando a IA precisa aprender sozinha, sem respostas prontas, encontrando padrões ocultos nos dados.

------------------------------------------------------------------------

## ML - Aprendizado Não Supervisionado em Detalhes: Agrupamento, Associação e Redução de Dimensão

E se, ao invés de mostrar as respostas à IA, simplesmente entregássemos os dados e disséssemos: "Descubra o que pode ser aprendido daqui"?
Esse é o espírito do **Aprendizado Não Supervisionado**.
Aqui, os dados não têm rótulos ou respostas certas.
A missão do algoritmo é buscar estrutura, identificar padrões e extrair sentido das informações disponíveis.
Vamos citar três possibilidades deste tipo de aprendizagem: **agrupamento, associação e redução de dimensão**.

### Agrupamento (Clustering): Segmentando em Grupos Naturais

Vamos falar primeiro do **agrupamento**, também conhecido como "*clustering*".
Aqui, queremos segmentar os elementos analisados em grupos.
No contexto de análise de clientes com base em comportamentos de compra, podemos, por exemplo, agrupar consumidores que adquirem itens de luxo e produtos exclusivos.
Uma parte deles talvez priorizando promoções e compras econômicas, outro sendo mais impulsivo (com pouco tempo entre a consulta do item e a compra), e um terceiro que faz muitas buscas, inclusive oscilando entre o *desktop* e o aplicativo do celular.
O algoritmo pode ainda considerar muitas outras informações: padrões de devolução de produtos, horários de navegação em um site, métodos de pagamento...
Serão os dados que indicarão o que nos ajuda a segmentar.

Em termos de algoritmos, o **K-means** é um dos mais conhecidos.
Nele, definimos previamente o número de grupos desejados – e a qualidade desses agrupamentos pode ser posteriormente avaliada por meio de técnicas como o método do cotovelo.
Mas ele não é o único, existem alternativas como o **DBSCAN**, que identifica agrupamentos com base na densidade dos dados, e o **agrupamento hierárquico**, que constrói uma estrutura de grupos em forma de árvore – inclusive é um dos meus algoritmos preferidos, pois é uma técnica que permite visualizar a construção dos grupos, quase como uma Análise Exploratória Avançada.

### Associação: Desvendando Relações entre Itens

Mas nem só de agrupamento vivem os modelos não supervisionados.
Podemos também trabalhar com **associação**, usada para identificar padrões de co-ocorrência entre itens.
Um exemplo clássico desse conceito está na análise de compras: se um cliente adquire determinado produto, quais são os itens mais propensos a serem comprados juntos?
Nosso resultado é algo como: "Clientes que compram esta marca de goiabada também compram estes queijos".

### Redução de Dimensionalidade: Simplificando a Complexidade

A terceira abordagem é a **redução de dimensionalidade**, especialmente útil quando lidamos com bases de dados enormes, cheias de variáveis.
Para simplificar, pense nas variáveis como colunas em uma planilha do Excel.
Técnicas como o **PCA (Análise de Componentes Principais)** ajudam a reduzir o número de colunas, construindo um número menor de novas colunas que são capazes de representar suficientemente bem as colunas originais.
Podemos começar com 100 colunas e, ao final, ter apenas 4 que preservam o que realmente importa, eliminando redundâncias e ruídos.
Parece mágica, mas é melhor: é matemática, álgebra linear!
Podemos pensar no PCA como compactar uma foto: reduz o tamanho, mas mantém o essencial visível.

### Avaliando o Desempenho no Aprendizado Não Supervisionado

E como fica a avaliação dos modelos?
Nos modelos não supervisionados, a avaliação funciona de forma diferente.
Como não há “resposta certa”, usamos métricas relacionadas à **coesão** (o quão semelhantes são os itens dentro de um grupo) e **separação** (o quão distintos são os grupos entre si).

Um bom exemplo de métodos não supervisionados são os agrupamentos de imagens do Pinterest, que organizam fotos pelo estilo visual, temas, cores predominantes e *hashtags*.
Isso garante que fotos similares sejam agrupadas em segmentos distintos – e permite, inclusive, que o usuário identifique novos segmentos de interesse!
Foi assim que eu descobri, por exemplo, o conceito de Mind Maps e CheatSheets, basicamente diagramas visuais que conectam ideias de maneira resumida.
E claro, isso se aplica em outras áreas e contextos, como: organização de documentos, estratégia logística urbana e estudos genéticos.

Aprender sem rótulos é algo poderoso e, muitas vezes, surpreendente.
Mas quando temos tantas abordagens diferentes - é natural que surja a dúvida: como saber qual algoritmo usar?
Será que existe uma escolha certa?
Vamos desvendar isso no próximo capítulo!

------------------------------------------------------------------------

## Como escolher o melhor algoritmo de Machine Learning?

"O algoritmo perfeito existe?" Se você espera uma resposta simples, prepare-se para a realidade: no Machine Learning, **tudo depende do contexto do problema** – a verdade é que "DEPENDE" é a resposta certa para quase todas as perguntas.
Neste material, usaremos esta palavra ao menos uma vez por capítulo!

Isso porque algoritmos são como ferramentas, e a escolha depende do objetivo que queremos alcançar, dos dados disponíveis e das restrições do problema.
A arte está em escolher uma ferramenta que faça sentido para o seu cenário.

Podemos pensar em atributos como:

-   **Escalabilidade:** Tenho dados limitados ou um grande volume?

<!-- -->

-   **Desempenho:** Minha prioridade é obter respostas rápidas (como no processamento de dados em tempo real) ou posso focar na precisão do resultado analítico?

<!-- -->

-   **Recursos:** Qual o custo computacional aceitável?

<!-- -->

-   **Interpretabilidade:** Preciso entender como a decisão foi tomada?

Em relação a este último ponto, algoritmos mais simples — como **Regressão Linear, Regressão Logística e Árvores de Decisão** — se destacam justamente pela transparência e facilidade de aplicabilidade.
Exigem menos dados e recursos computacionais e possuem a vantagem de serem interpretáveis.
Em áreas como saúde, finanças e direito, entender como a IA chegou a uma decisão pode ser tão importante quanto a precisão do resultado.

Por outro lado, existem modelos que nos distanciam da interpretabilidade, mas garantem maior precisão.
É o caso das abordagens conhecidas como ***bagging*** e ***boosting***, que trabalham com a combinação de diferentes algoritmos e estratégias.
O exemplo mais conhecido é a **Random Forest**, que é uma técnica de *bagging*, pois, como falamos, ela utiliza várias Árvores de Decisão para obter seus resultados.
Existem outros algoritmos famosos como **XGBoost** e **Gradient Boosting**, que melhoram a precisão dos modelos ao corrigir os erros progressivamente.
Inclusive, estes algoritmos são amplamente utilizados em competições de *machine learning* em sites como o Kaggle, onde a otimização da precisão é o grande objetivo.

Mas existem algoritmos ainda mais complexos, como as **Redes Neurais Profundas**, que são capazes de modelar cenários extremamente complexos.
Contudo, aqui, há a necessidade de encontrar um equilíbrio: pois quanto mais poderosa a IA, menos conseguimos entender como ela decide, além de demandar mais dados, tempo de treinamento e poder computacional.

Portanto, a questão não é qual algoritmo é o melhor, mas qual é o mais adequado para o problema que você deseja resolver.
Em muitos casos, um modelo simples pode ser suficiente e ter a vantagem de ser interpretável – o que ajuda a desenharmos ações *data-driven*.
Em outros, é necessário recorrer a abordagens mais sofisticadas para sermos capazes de endereçar o desafio.

Nos próximos capítulos, vamos entender mais sobre essas abordagens mais avançadas, nomeadamente sobre as redes neurais.
