<h1> Deep Learning for Anti-Money Laundering: Detecting Suspicious Transactions </h1>

<h3>Team Members: Heba Bou KaedBey, Min Shi, Ritesh Bachhar, Khanh Nguyen </h3>

Define Problem: Money laundering is the process of conversion of illicit money which comes out of the crime which is then intermixed with the licit money to make appear legitimate, and it becomes very difficult to distinguish the legitimate money from the illegitimate one.



<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#Dataset">Dataset</a></li>
    <li><a href="#Citations">Citations</a></li>
    <li><a href="#Code-Description">Code Description</a></li>
</ul>

---

<h3 id="Introduction">Introduction</h3>

Fraud and money laundering continue to burden financial institutions, with indirect costs far exceeding direct losses. In North America, each dollar lost to fraud cost firms $4.45 in 2023, factoring in investigation, recovery, and reputational harm. Meanwhile, Panama Papers and Swiss leaks spotlighted how offshore structures facilitate laundering, blending illicit funds with legitimate ones to obscure their origin. As schemes become more sophisticated, robust data-driven models are essential for timely detection and prevention.

**Goal:** Build a deep learning model on SAML-D to detect suspicious financial transactions, reducing false positives while improving recall in anti-money laundering systems. We also want to try using OpenAI’a GPT to generate natural language explanations for each flagged transaction. And if time permits deploy to AWS ( probably S3).

---

<h3 id="Dataset">Dataset</h3>

The <a href="https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/data">SAML-D dataset</a> is a synthetic dataset of 9.5 million tranactions, developed using insights from AML specialists through semi-structured interviews, analysis of existing datasets, and a comprehensive literature review. The dataset includes 28 typologies, 11 normal and 17 suspicious, modelled to reflect real-world senarios, with overlapping behaviours between normaal and suspicious transactions to increase complexity and challenge detection efforts. The dataset includes 676,912 unique accounts conducting transactions across 18 geographic locations, using 13 currencies and 7 payment methods. Among these, 0.104% of the transactions (9,873 entires) are labeled as suspicious representing various money laundering techniques such as layering, rapid fund movement, and transactions to high risk locations.

---

<h3 id="Citations">Citations</h3>
<ul>
<li>LexisNexis Risk Solutions. <a href=https://risk.lexisnexis.com/about-us/press-room/press-release/20240424-tcof-financial-services-lending>Every Dollar Lost to a Fraudster Costs North America's Financial Institutions $4.45</a>. LexisNexis Risk Solutions, 24 Apr. 2024.</li>
<li> B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan and G. Aksu, "Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset," 2023 IEEE International Conference on e-Business Engineering (ICEBE), Sydney, Australia, 2023, pp. 47-54, doi: 10.1109/ICEBE59045.2023.00028.</li>
<li>Oztas, Berkan, et al. "Tab-AML: A Transformer Based Transaction Monitoring Model for Anti-Money Laundering." 2025 IEEE Conference on Artificial Intelligence (CAI). IEEE, 2025.</li>
</ul>


---

<h3 id="Code-Description">Code Description</h3>
