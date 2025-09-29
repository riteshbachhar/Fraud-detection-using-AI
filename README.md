<h1> Deep Learning for Anti-Money Laundering: Detecting Suspicious Transactions </h1>

<h3>Team Members:Heba Bou KaedBey, Min Shi, Ritesh Bachhar, Khanh Nguyen </h3>

Define Problem: Money laundering is the process of conversion of illicit money which comes out of the crime which is then intermixed with the licit money to make appear legitimate, and it becomes very difficult to distinguish the legitimate money from the illegitimate one.

Fraud and money laundering continue to burden financial institutions, with indirect costs far exceeding direct losses. In North America, each dollar lost to fraud cost firms \$4.45 in 2023, factoring in investigation, recovery, and reputational harm. Meanwhile, Panama Papers and Swiss leaks spotlighted how offshore structures facilitate laundering, blending illicit funds with legitimate ones to obscure their origin. As schemes become more sophisticated, robust data-driven models are essential for timely detection and prevention.

Dataset: The SAML-D dataset comprises 12 characteristics and 28 typologies, selected based on existing datasets, academic literature, and interviews with AML specialists, which highlights its broad applicability. It contains 9.5 million entries, however, the ``isLaundering" labels are highly imbalanced, with only 0.15\% representing actual money laundering activity. In addition, the data set includes 15 graphical network structures that model transaction flows within these typologies. Although some structures are shared between types, they differ significantly in parameters to increase complexity and challenge detection efforts.

Goal: Build a deep learning model on SAML-D to detect suspicious financial transactions, reducing false positives while improving recall in anti-money laundering systems. We also want to try using OpenAI’a GPT to generate natural language explanations for each flagged transaction. And if time permits deploy to AWS ( probably S3).


<h3>Citations:</h3>
