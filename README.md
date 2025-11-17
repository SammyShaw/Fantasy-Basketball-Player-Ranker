# Fantasy-Basketball-Player-Ranker 2.0
Improvements on existing fantasy basketball ranking algorithms (e.g., ESPN, BasketballMonster) using PCA and SHAW (Structured Hierarchical Adjusted Weights) for category leagues. 


Fantasy Basketball category leagues typically rank players by standardizing their statistics across nine (or more) categories and summing the resulting values. This approach implicitly assumes that each category contributes equally to player value and that statistical categories accumulate independently. Prior attempts to improve on this framework exist, but only a small number have shown empirically defensible gains (e.g., Rosenof).

In this paper, I propose a simple and intuitive alternative based on the empirical covariation of statistical categories. Fantasy categories do not accumulate independently; they co-vary within players, and this covariation is structured by three broad player archetypes: guards, bigs, and, to a lesser extent, wings. Because conventional nine-category leagues emphasize categories that disproportionately reward guards, a clear implication follows: weighting the dominant cluster of covarying categories yields more accurate and robust player rankings. I demonstrate this method and evaluate it using a straightforward, if static, matchup-based approach, showing substantial improvements over traditional Z-score systems.

