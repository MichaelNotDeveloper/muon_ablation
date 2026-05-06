date : 2026-05-05

Autor : Vladimir Golubev

Status : #finished

Tags : #polar #spectrum

References and links :
- [CASPR Without Accumulation is Muon](https://leloykun.github.io/ponder/caspr-wo-accum-is-muon/)
- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
- [Steepest Descent Under Schatten-p Norms](https://leloykun.github.io/ponder/steepest-descent-schatten-p/)
- 

February 21, 2025 ·  Franz Louis Cesista


В начале автор рассказывает про Muon (Muon - An optimizer for hidden layers in neural networks) и Newton-Schulz.

Объясняет, зачем нужно ортогонализировать градиенты Мюона (то есть в чем его идея), ссылаясь на то, что это продолжение идеи Shampoo или CASPR без аккумуляции preconditioners ([CASPR Without Accumulation is Muon](https://leloykun.github.io/ponder/caspr-wo-accum-is-muon/)).

## Steepest Descent

Далее автор говорит о том, что о Мюоне можно думать как об алгоритме, который делает спуск по наиболее крутому направлению по спектральной норме. И объяснение заключается в том, что это операторная норма в случае если входы и выходы в евклидовой норме (что является адекватным допущением). Мысли об этом были и раньше для предшественников Muon ([Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)).

## Approximate Semi-Orthogonalization

Несмотря на то, что Newton-Schulz лишь приближенно вычисляет Polar Decomposition, Muon-у этого достаточно (*от себя: иногда это даже эффективнее — [[Insights on Muon from Simple Quadratics]]*).

Автор утверждает, что это так, потому что Muon может быть переопределен, как наиболее крутой спуск по Schatten-p Norms. Автор ссылается на [Steepest Descent Under Schatten-p Norms](https://leloykun.github.io/ponder/steepest-descent-schatten-p/).

И затем в защиту своего тезиса автор заявляет, что все направления приближающие оригинальный градиент к его semi-orthogonalization (Polar Decomposition) это хорошие направления.

*пока что выглядит как рукомахание. Статью, на которую ссылка, кратко пролистал. Там выглядит как просто то, что объясняется, что идет крутой спуск по пространству спектральных значений матрицы. И его норма — это shatten-p норма, норма оператора, дуальная нормам входа, обычным $\ell$ нормам*.

## What’s the problem with the original coefficients of Muon?

Далее автор говорит о том, что Newton-Schulz создает слишком много шума в сингулярных значениях и показывает график с большой вариацией. Автор утверждает, что что-то посчитал и Shatten-32 норма ничего не меняет и поэтому Muon можно интерпретировать как наиболее крутой спуск по Schatten-32 норме (*что? ладно*).

Далее автор говорит о том, что проблема возникает из-за неоптимальности коэффициентов и вслед за авторами [[THE POLAR EXPRESS OPTIMAL MATRIX SIGN METHODS AND THEIR APPLICATION TO THE MUON ALGORITHM]] — предлагает подбирать коэффициенты для всех шагов свои, то есть как матрицу $3 \times NUM\_STEPS$. Для этого можно использовать градиентный спуск (ну почему нет. Кстати идея как у Meta Learning). 

Далее автор утверждает, что прироста не было на GPT-2-small, но прирост от этого метода был получен на GPT-2-medium (*напомню, что Polar Express — [[THE POLAR EXPRESS OPTIMAL MATRIX SIGN METHODS AND THEIR APPLICATION TO THE MUON ALGORITHM]] проверяли свой метод на GPT-2-large*). Автор интерполирует свои результаты и на наиболее крупные модели. И вероятно они в чем-то правы, так как не самые уверенные подтверждения, но все же, имеются в статье по polar express.

*Мой concern: надо исключить эффект переобучения под задачу. То есть такой подбор коэффициентов не выглядит оправданным математически. Возможно происходит fit под конкретную задачу/архитектуру, а не в общем*.

## Утверждения автора
1. In early training, the ‘steepness’ of the curve matters more than noise reduction. This is because the stable rank of the gradients tends to be smaller in early training.
2. But noise reduction matters more for longer training runs. I.e., having a smaller variance in the resulting singular values after NS iterations results in lower loss overall.

*Я кажется понял, что он имеет в виду. Steepness изначально заложен в Muon. По этой причине он хорош на первых итерациях. И автор утверждает, что если уменьшать шум на последних итерациях, то сходимость будет лучше. Наверное, я с этим соглашусь. У этого утверждения ограниченное обоснование (его почти нет, только очень натянутое через подбор коэффициентов для уменьшения шума, но по сути в рамках нашей задачи, по всей видимости), но при этом оно звучит логично и вяжется с результатами других работ*.
