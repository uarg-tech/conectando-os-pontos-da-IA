---
title: Capítulo 6 - Fundamentos e Evolução da Inteligência Artificial Generativa
layout: page
editor_options: 
  markdown: 
    wrap: sentence
---

# O que é Inteligência Artificial Generativa (GenAI) ?

A inteligência artificial saiu do campo da ficção científica e deixou de ser um conceito técnico distante e para se tornar parte do nosso cotidiano.
Tudo mudou, de forma definitiva, no final de 2022 com o lançamento do ChatGPT.

Foi um marco: o aplicativo que mais rapidamente conquistou usuários na história.
Em apenas 5 dias, já somava 1 milhão de usuários.
Para se ter uma ideia, a Netflix levou anos para atingir esse número, enquanto o ChatGPT fez isso em menos de uma semana.

<!-- to-do: adicionar uma visualização comparativa-->

Esse fenômeno reflete o impacto e a acessibilidade da IA Generativa.
Antes, a IA parecia algo distante, presente apenas em laboratórios e grandes corporações.
Mas de repente, qualquer pessoa passou a poder testar, criar e interagir diretamente com um modelo de IA avançado, o que acelerou não apenas a adoção, mas também a curiosidade e o desenvolvimento de novas aplicações.
Desde então, o interesse só cresce e a presença da IA na sociedade se expande rapidamente.

No entanto, essa familiaridade com chatbots e modelos de linguagem também trouxe uma percepção distorcida de que IA se resume a sistemas como o ChatGPT, ou ainda que se limita apenas à IA Generativa.
O que, como vimos, está longe de ser verdade.
Na realidade, está mais para o contrário: dentro da IA, temos o Aprendizado de Máquina, com o Deep Learning como uma técnica de modelagem e a IA Generativa como uma opção de abordagem.
E dentro da IA Generativa, existem outros conceitos até chegarmos em soluções como o ChatGPT, como vamos explorar ao longo deste capítulo.

Mas antes, quero comentar sobre o que exatamente tornou tudo isso possível: o que diferencia essa nova geração de IA?

A Inteligência Artificial Generativa abrange modelos capazes de criar conteúdos originais em vez de simplesmente replicar informações existentes, a GenAI combina informações de maneira inédita.
Um palestrante recifense que eu gosto muito, chamado Murilo Gun, diz que a criatividade é habilidade de combinar coisas conhecidas de maneiras novas.
Pois bem, nestes termos, temos a "criatividade" de diferentes formas.

Diferente das IAs tradicionais, que apenas analisam ou classificam dados, a IA Generativa cria.
Ela escreve textos, desenha imagens, compõe músicas, gera vozes, ou até inventa a foto de um cachorro que nunca existiu!

Os princípios que possibilitam isso já foram discutidos aqui.
Já que no coração de toda essa inovação está o Deep Learning — especialmente arquiteturas como os Transformers, que veremos mais adiante.

Em resumo, por trás dessa “criatividade”, há pura matemática: a IA calcula a probabilidade de palavras se combinarem de forma plausível, semelhante a um estagiário seguindo instruções sem saber exatamente o que está fazendo.
Citei "palavras", mas isso vale também para pixels de imagens, notas musicais ou qualquer outro tipo de dado com que estivermos trabalhando.
E a partir destas combinações, acabamos por gerar novos textos, figuras ou composições.
Assim, a IA passa a ser capaz de sugerir, simular, criar!

É sem dúvida o começo de algo novo: uma era em que máquinas criam — um território que, até pouco tempo, era exclusivo da inteligência humana.
Mas afinal, o que permite que esses modelos gerem conteúdos de maneira tão coerente?
É isso que vamos descobrir no próximo capítulo.

# O que é um LLM e como ele aprende: dados, tokens e padrões

Na IA Generativa, um modelo se destaca: os LLMs — Large Language Models ou Grandes Modelos de Linguagem.
Eles são o cérebro por trás de sistemas como o ChatGPT, capazes de interpretar linguagem e gerar respostas com fluidez.

E como esses modelos aprendem?
Para responder, vale antes entender uma questão fundamental:

> o que é linguagem para uma máquina?

Para uma máquina, linguagem é um conjunto de símbolos que deve ser decifrado e modelado matematicamente para gerar respostas coerentes.

Uma das abordagens mais comuns é a tokenização, que divide palavras em unidades menores chamadas tokens.
Por exemplo, "ChatGPT" pode ser fragmentado em ["Chat", "G", "PT"], permitindo que a IA analise melhor as variações linguísticas.
Existem outras técnicas: como stemming, que reduz palavras à sua raiz, e lematização, que identifica a forma padrão das palavras, ou ainda os Word Embeddings, que convertem palavras em vetores numéricos, permitindo o cálculo de relações semânticas e contextuais.
Com essa abordagem, a IA pode reconhecer que ‘doce’ e ‘sobremesa’ têm uma relação conceitual, enquanto 'pipoca' e 'cinema’ compartilham proximidade contextual.

Agora, imagine ler milhares de frases por dia.
Com o tempo, você perceberia que "gostaria de" é frequentemente seguido por um verbo no infinitivo, como "viajar", "aprender" ou "comer".
Os LLMs fazem algo semelhante—mas em escala massiva, analisando as palavras extraídas de livros, sites, artigos e fóruns na internet.
Mas não necessariamente seguindo regras gramaticais rígidas, mas sim aprendendo padrões estatísticos, capturando nuances e variações na linguagem.

De modo geral a ideia do treinamento deste tipo de rede funciona como um quebra-cabeça estatístico.
O modelo recebe textos com partes ocultas e precisa prever os tokens que vêm a seguir.
Se errar, ajusta internamente seus bilhões de parâmetros—os mesmos "botões de ajuste" que mencionamos no capítulo anterior.
Esse processo se repete muitas vezes, exigindo supercomputadores operando por meses.
Para você ter uma ideia, estima-se que o treinamento do GPT-3 custou cerca de US\$ 12 milhões em energia, consumindo o equivalente a uma cidade de 50 mil habitantes!
E este é um modelo anterior ao lançamento do ChatGPT!

Mas voltando ao aprendizado das LLMs, o ponto é que todas estas estratégias, dados e processamentos permitem que esses modelos captem contextos, ironias, ambiguidades … tornando suas respostas mais naturais e coerentes.

Se perguntarmos sobre “pizza de calabresa leva queijo?”
, a resposta pode ser SIM, já que tradicionalmente leva, mas se adicionarmos a palavra “São Paulo”, isso irá modificar a resposta, a menção à cidade ativa regiões específicas da rede, fazendo com que o NÃO supere o SIM.

Mas como os LLMs conseguem manter coerência em textos longos, analisando páginas inteiras, por exemplo?
A resposta está na arquitetura Transformer — que será o foco do próximo capítulo.

# A Arquitetura Transformer: A revolução dos Modelos de Linguagem

O segredo por trás da fluência do ChatGPT e de outros modelos de linguagem tem nome: Transformer.
E não tem nada a ver com o filme — trata-se de uma arquitetura revolucionária apresentada em 2017, em um artigo de apenas nove páginas que transformou para sempre o campo da inteligência artificial.

O artigo “Attention is All You Need” (Atenção é tudo o que você precisa) propôs uma ideia simples, mas poderosa: e se, ao invés de seguir a ordem das palavras, o modelo pudesse observar todas elas ao mesmo tempo?
Nascia ali o mecanismo de self-attention, algo como “atenção ao eu”.

Imagine a seguinte frase: 'O médico disse ao paciente que ele deveria descansar.' O modelo precisa entender a quem 'ele' se refere.
Enquanto modelos mais antigos (sem o mecanismo de atenção) teriam grande dificuldade com essa ambiguidade, focando apenas na proximidade das palavras, o Transformer, com seu mecanismo de atenção própria, onde cada palavra observa todas as outras para entender seu papel no contexto, consegue observar todas as palavras simultaneamente, medindo matematicamente suas relações e resolvendo a questão.

Essa arquitetura funciona como um maestro que não só coordena cada instrumento (palavra), mas prevê como a sinfonia vai evoluir.
Quanto maior o número de camadas e parâmetros, mais refinada essa 'regência' se torna — o que explica a precisão crescente dos modelos mais avançados.
Mas note que a revolução não se limitou ao texto.

Hoje, os Transformers são a base de diversos sistemas de IA, impulsionando desde a recomendação de conteúdo até a geração de imagens.
Ainda assim, sua essência e maior força continuam nos textos — não à toa, assistentes conversacionais como ChatGPT, Copilot, Gemini, Claude e tantos outros dominam diálogos com uma naturalidade simulada impressionante.

Agora, imagine conseguir este tipo de efeito de maneira integrada: textos, imagens, sons, tudo em um único modelo.
Esse tipo de integração já é realidade — e se chama IA Multimodal, tema do nosso próximo capítulo.

# IA Multimodal: GenAI além do texto

Até aqui falamos de modelos que lidam principalmente com texto.
Mas o mundo real é composto por muitos outros elementos, como imagens, sons, vídeos, movimentos, sinais sensoriais...
É nesse contexto que a IA Multimodal nasce.

O termo “multimodal” se refere à capacidade de um sistema de inteligência artificial de processar e combinar múltiplos tipos de entradas.
Ou seja: é uma IA que aprende a ver, ouvir, ler … E cruzar tudo isso!
Mesmo que os formatos sejam originalmente diferentes, aproximando a IA da maneira como nós, humanos, interagimos com o mundo.

> E como a combinação dos diferentes tipos é feita?

Em vez de uma única rede neural, esses modelos combinam várias redes especializadas em cada tipo de dado: uma para texto, outra para imagens, outra para áudio, e assim por diante.
O que conecta e harmoniza todas essas entradas?
Quer arriscar?
Sim, um Transformer!
Mas aqui, um Transformer adaptado para funcionar como um grande tradutor dos demais modelos.

E esse tradutor transforma cada tipo de entrada em uma linguagem numérica comum — uma espécie de “língua franca digital”.
Texto vira token, imagem vira vetor de pixels, som vira espectrograma.
E depois tudo é convertido em números que o modelo consegue interpretar e relacionar em conjunto.
E os avanços recentes são impressionantes.
Hoje, modelos como o Gemini, por exemplo, não apenas analisam vídeos e respondem a perguntas sobre o que acontece em cada cena.
Ou ainda o Claude, que combina texto e código de programação, permitindo a geração de códigos, análise de vulnerabilidades, ou elaboração de documentações.
E temos novas ferramentas surgindo o tempo todo, ampliando ainda mais as prateleiras de possibilidades.

Vamos considerar um exemplo mais concreto: Imagine um sistema médico equipado com IA Multimodal.
Ele recebe a radiografia de tórax de um paciente, cruza com o histórico clínico e analisa o áudio da consulta médica – onde identifica um padrão de tosse.
E com base na fusão de todos esses dados, a IA retorna: “Padrão sugestivo de pneumonia.
Recomenda-se a seguinte lista de exames confirmatórios.”

E se essa integração puder também gerar conteúdo combinado — imagens, textos, sons — a partir de um simples comando?
É aí que entra a Inteligência Artificial Generativa Multimodal, tema do nosso próximo capítulo.

# GenIA em ação: Modelos, Produtos e Engenharia de Prompt

Assistentes como ChatGPT, Gemini e Claude já fazem parte do nosso cotidiano, evoluindo rapidamente e impressionando pela capacidade de responder perguntas, resumir conteúdos, gerar ideias e auxiliar em tarefas de escrita e programação.

As interfaces que utilizamos para acessar tais recursos, por outro lado, não costumam apresentar grandes modificações.
O que nos leva a uma reflexão importante: a diferença entre o "cérebro" por trás destas ferramentas, e a forma como interagimos com eles.

Essa distinção se resume a dois conceitos fundamentais: modelo e produto.

-   O modelo é o algoritmo de IA em si, treinado com grandes volumes de dados.
-   O produto é a aplicação que faz uso desses modelos e os entrega de forma mais acessívelao usuário final.

**Modelos**

O ChatGPT, por exemplo, é a interface que utilizamos como produto, mas seu “cérebro” é um modelo de Deep Learning chamado GPT, que possui diferentes versões.

O modelo Generative Pre-trained Transformer (Transformador Pré-treinado Generativo), ou simplesmente GPT, é baseado em uma arquitetura criada em 2017 que revolucionou o campo do processamento de linguagem natural.

O primeiro modelo GPT foi lançado pela OpenAI em 2018, seguido pelo GPT-2 em 2019 e o GPT-3 em 2020.
A interface ChatGPT foi lançada em novembro de 2022, popularizando o modelo GPT-3.5 entre o público geral.
Em março de 2023 veio o GPT-4, e em maio de 2024 o GPT-4o — cada versão trazendo melhorias em precisão, criatividade, desempenho e, mais recentemente, em compreensão multimodal (texto, imagem, áudio e vídeo).

Enquanto o modelo evolui tecnicamente, o produto aprimora a experiência do usuário (UX): integra recursos multimodais, aumenta a velocidade de resposta e até permite a criação de GPTs personalizados — equivalentes a ajustar o comportamento do modelo conforme a necessidade de cada usuário ou empresa.

Essa personalização pode acontecer de diferentes formas.
No nível mais simples, temos parâmetros como a "temperatura", que regula o grau de imprevisibilidade nas respostas: valores altos geram respostas mais criativas e incomuns, enquanto valores baixos tornam a saída mais literal e previsível.
Em um nível mais avançado, existe o fine-tuning (ajuste fino), onde o modelo é retreinado com dados específicos para se especializar em determinada área ou tarefa, como atendimento médico, jurídico ou suporte técnico.

Por fim, temos um elo fundamental entre modelo e produto: as instruções que damos à IA.
Ou simplesmente, o prompt.
Já vimos como uma simples palavra pode alterar toda a resposta – lembra do exemplo da pizza de calabresa?

Por isso, a chamada “Engenharia de Prompt” tornou-se uma habilidade valorizada, cercada de boas práticas.
Mas, como diz o professor Eduardo Barbosa: “Chamar isso de engenharia é até injusto — tá mais para uma alquimia.”

A relação entre modelos e produtos se estende para além do ChatGPT, envolvendo outras gigantes como Google (Gemini), Anthropic (Claude) e Microsoft (GitHub Copilot).
Além disso, alguns produtos usam mais de um modelo, ou ainda modelos que aparecem em vários produtos, como os modelos da OpenAI que são utilizados pela Microsoft em seus produtos integrados.

A relação entre modelos e produtos se estende para além do ChatGPT, com diferentes estratégias no mercado:

-   o Google desenvolve seus próprios modelos (como o Gemini) para seus produtos;
-   a Anthropic criou o Claude como modelo independente; ou ainda
-   a Microsoft incorpora modelos da OpenAI em produtos como GitHub Copilot e Microsoft 365.

Essa diversidade mostra como a mesma tecnologia base pode ser implementada de formas distintas, dependendo da estratégia de cada empresa.

E qual o melhor modelo?
Qual a melhor solução?
A resposta é quase sempre a mesma: depende!
Existem diversos critérios que podem ser considerados, desde desempenho técnico até custo e privacidade, muitas opções já disponíveis e tantas outras que estão sendo desenvolvidas neste exato momento.
O ideal é explorar as ferramentas, combinando comparativos de desempenho dos modelos, estratégias de prompt e experimentação de diferentes soluções.

É importante lembrar uma limitação fundamental: por mais poderosos que sejam, os LLMs organizam, resumem e expandem ideias com base em padrões estatísticos de linguagem, não em compreensão real.
Isso significa que podem falhar em informações factuais, apresentando o que é mais provável, mas não necessariamente o correto ou ideal para uma dada situação.

Agora que entendemos melhor a diferença entre modelos e produtos, podemos buscar ferramentas com recursos de checagem, explorar personalizações técnicas ou até desenvolver nossas próprias soluções.
Com essa base, estamos prontos para dar o próximo passo: entender como os Agentes Autônomos ampliam ainda mais o poder da GenAI.

# Agentes Autônomos: Quando a IA decide e age por conta própria

Se o ChatGPT fosse um estagiário que apenas respondesse perguntas, os Agentes Autônomos seriam funcionários plenos: não só falam, mas fazem.
Eles enviam e-mails, atualizam planilhas, pesquisam na web e executam tarefas complexas – tudo sem intervenção humana.
E esta é uma das revoluções mais recentes da IA Generativa!

Imagine acordar com seu agente já tendo respondido e-mails, organizado sua agenda e corrigido um bug no código — tudo isso sem comandos diretos.
Essa é a proposta: transformar a IA de uma ferramenta reativa em uma assistente proativa, ganhando tempo e produtividade ao transformar informações em ações concretas.

Como eles funcionam?
A lógica é baseada em três pilares fundamentais:

-   Percepção: O agente coleta e interpreta informações do ambiente, como dados de sistemas, mensagens ou mudanças em arquivos.
-   Decisão: Com base nas informações coletadas, avalia diferentes opções e escolhe a melhor estratégia de ação.
-   Ação: Executa a solução escolhida, podendo interagir com múltiplas ferramentas e sistemas simultaneamente.

Um exemplo prático: no atendimento ao cliente, o agente analisa automaticamente o histórico do cliente, identifica o problema reportado e decide entre diferentes soluções — como processar um reembolso, agendar suporte técnico ou enviar instruções personalizadas — tudo sem intervenção humana.

Exemplos vão desde automatizar a criação de apresentações e análise de dados até agentes de customer service que resolvem tickets automaticamente.

Com o avanço da IA, esses agentes estão ficando mais sofisticados e presentes em diversos setores.
O futuro aponta para sistemas cada vez mais autônomos, capazes de aprender com suas próprias ações e otimizar processos continuamente com o mínimo de intervenção humana.

**Os desafios e limitações**

No entanto, ainda há limitações significativas.
Esses sistemas podem cometer erros, tomar decisões inadequadas ou apresentar comportamentos inesperados.
Imagine se um agente de vendas manipula dados para atingir metas**,** ou se um sistema de RH rejeita candidatos com base em vieses não intencionais?

Tentar pré-programar todas as exceções seria inviável — uma lição que já aprendemos com os sistemas especialistas dos anos 80, que falharam justamente por essa rigidez.

Além disso, questões de segurança e privacidade se tornam críticas quando agentes têm acesso a dados sensíveis e podem tomar ações com impacto real no mundo físico e digital.

Por isso, mesmo com a crescente autonomia, a supervisão humana continua indispensável, especialmente em decisões críticas.
À medida que esses agentes se tornam mais poderosos, surgem questões éticas fundamentais sobre responsabilidade, transparência e controle — não à toa, regulamentações como a Lei de IA da União Europeia estão sendo desenvolvidas globalmente para garantir um uso responsável da IA.

Essas reflexões nos conduzem naturalmente ao próximo tema: os desafios éticos da inteligência artificial — uma conversa cada vez mais urgente à medida que delegamos decisões críticas a sistemas que aprendem e agem por conta própria.
