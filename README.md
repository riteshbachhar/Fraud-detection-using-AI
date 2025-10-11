<h1> Deep Learning for Anti-Money Laundering: Detecting Suspicious Transactions </h1>

<h3>Team Members: Heba Bou KaedBey, Min Shi, Ritesh Bachhar, Khanh Nguyen </h3>

Define Problem: Money laundering is the process of conversion of illicit money which comes out of the crime which is then intermixed with the licit money to make appear legitimate, and it becomes very difficult to distinguish the legitimate money from the illegitimate one.



<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#Dataset">Dataset</a></li>
    <li><a href="#Preprocessing">Preprocessing</a></li>
        <ul>
            <li><a href="#Recasting">Memory Optimization via Integer Recasting</a></li>
            <li><a href="#Train-Test-split">Custom Train-test split</a></li>
        </ul>
    <li><a href="Features">Time‑Based and Graph‑Inspired Feature Engineering</a></li>
    <li><a href="#References">References</a></li>
    <li><a href="#Code-Description">Code Description</a></li>
</ul>

---

<h3 id="Introduction">Introduction</h3>

Fraud and money laundering continue to burden financial institutions, with indirect costs far exceeding direct losses. In North America, each dollar lost to fraud cost firms $4.45 in 2023, factoring in investigation, recovery, and reputational harm. Meanwhile, Panama Papers and Swiss leaks spotlighted how offshore structures facilitate laundering, blending illicit funds with legitimate ones to obscure their origin. As schemes become more sophisticated, robust data-driven models are essential for timely detection and prevention.

**Goal:** Build a deep learning model on SAML-D to detect suspicious financial transactions, reducing false positives while improving recall in anti-money laundering systems. We also want to try using OpenAI’a GPT to generate natural language explanations for each flagged transaction. And if time permits deploy to AWS (probably S3).

---

<h3 id="Dataset">Dataset</h3>

The <a href="https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/data">SAML-D dataset</a> is a synthetic dataset of 9.5 million transactions, developed using insights from AML specialists through semi-structured interviews, analysis of existing datasets, and a comprehensive literature review. The dataset includes 28 typologies, 11 normal and 17 suspicious, modelled to reflect real-world senarios, with overlapping behaviours between normaal and suspicious transactions to increase complexity and challenge detection efforts. The dataset includes 676,912 unique accounts conducting transactions across 18 geographic locations, using 13 currencies and 7 payment methods. Among these, 0.104% of the transactions (9,873 entires) are labeled as suspicious representing various money laundering techniques such as layering, rapid fund movement, and transactions to high risk locations.

The dataset includes the following 12 features:

- `Time`: Time of transaction, formatted as HH:MM:ss
- `Date`: Date of transaction, formatted as YYYY:MM:DD; ranges from October 6, 2022 to August 22, 2023
- `Sender_account`: Unique identifier for the sender's account
- `Receiver_account`: Unique identifier for the receiver's account
- `Amount`: Transaction amount; currency: British pounds (£)
- `Payment_currency`: Currency used by the sender
- `Received_currency`: Currency received by the receiver; may differ from the sender's currency
- `Sender_bank_location`: Country where the sender's bank is located
- `Receiver_bank_location`: Country where the receiver's bank is located
- `Payment_type`: Type of payment, such as credit card, debit card, cash, etc.
- `Is_laundering`: Target variable indicating whether the transaction is laundering (`1`) or not (`0`)
- `Laundering_type`: A categorical feature that includes 28 laundering typologies, derived from literature and semi-structured interviews with AML specialists.


---

<h3 id="Preprocessing">Preprocessing</h3>

<h4 id="Recasting">Memory Optimization via Integer Recasting</h4>

We implemented a function to optimize memory usage by downcasting integer columns. Columns originally typed as `int64` or `int32` were recast to the smallest suitable type (`int8`, `int16`, or `int32`) according to the following logic:

- If the maximum value in a column is less than 127, recast it to `int8`.
- If the maximum value is less than 32,768, recast it to `int16`.
- If the maximum value is less than 2,147,483,647, recast it to `int32`.
- Otherwise, keep the column as `int64`.

We excluded the `Sender_account` and `Receiver_account` columns from this recasting process.

<h4 id="Train-Test-split">Custom Train-test split</h4>

We customized the train–validation–test split to respect the temporal nature of the data. Since many of our engineered features rely on 30‑day rolling windows, it was essential to partition the dataset chronologically rather than randomly.

The full dataset spans 320 days. We designated the first 250 days for training, leaving a 70‑day holdout period. Of this holdout, the first 35 days were used for validation and the final 35 days for testing. This corresponds to a split of approximately 78.17% / 10.99% / 10.84% for training, validation, and test sets, respectively. The distribution of the positive class (class `1`) across these splits was 0.00101 / 0.00114 / 0.00116, which shows that the rare‑event class remained consistently represented across all partitions.

This temporal splitting strategy serves two purposes:
- Prevents data leakage: ensuring that information from the future does not inadvertently influence the training process.
- Maintains class stratification: preserving the relative balance of rare and common classes across all subsets, which is critical for reliable model evaluation.

---

<h3 id="Features">Time‑Based and Graph‑Inspired Feature Engineering</h3>

From the 12 original features, we derived 8 additional temporal variables from the existing `Time` and `Date` attributes. These include: `year`, `month`, `day_of_month`, `day_of_year`, `day_of_week`, `hour`, `minute`, and `second`. Together, these features allow the model to capture seasonality, periodicity, and fine‑grained temporal patterns in transaction behavior.

In addition, we engineered several domain‑specific features designed to capture structural and behavioral signals of anomalous activity:

- `fanin_30d`: The count of unique incoming counterparties over a rolling 30‑day window. This proved to be the single most predictive feature, reflecting the diversity of inbound connections.
- `fan_in_out_ratio`: The ratio of inbound to outbound counterparties over 30 days, highlighting imbalances in transactional relationships.
- `fanin_intensity_ratio`: The `fanin_30d` value normalized by the daily number of received transactions, concentration of inbound activity.
- `amount_dispersion_std`: The standard deviation of transaction amounts per sender, capturing volatility in counterparties' transfer sizes.
- `sent_to_received_ratio_monthly`: The ratio of total received to total sent amounts within a month. Ratios trending toward 1 may indicate circular or balancing behavior that warrants scrutiny
- `back_and_forth_transfers`: The number of transfers exchanged between a sender and receiver within a single calendar day. This is a directed metric: A → B is treated as distinct from B → A.
- `circular_transaction_count`: The number of transactions that eventually return to the original sender, forming a cycle. Cycles may span multiple steps and extend across several days, making them a strong indicator of layering or obfuscation.

---

<h3 id="References">References</h3>
<ul>
<li>LexisNexis Risk Solutions. <a href=https://risk.lexisnexis.com/about-us/press-room/press-release/20240424-tcof-financial-services-lending>Every Dollar Lost to a Fraudster Costs North America's Financial Institutions $4.45</a>. LexisNexis Risk Solutions, 24 Apr. 2024.</li>
<li> B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan and G. Aksu, "Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset," 2023 IEEE International Conference on e-Business Engineering (ICEBE), Sydney, Australia, 2023, pp. 47-54, doi: 10.1109/ICEBE59045.2023.00028.</li>
<li>Oztas, Berkan, et al. "Tab-AML: A Transformer Based Transaction Monitoring Model for Anti-Money Laundering." 2025 IEEE Conference on Artificial Intelligence (CAI). IEEE, 2025.</li>
</ul>


---

<h3 id="Code-Description">Code Description</h3>

- [config.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/config.py): include various paths such as `DATAPATH`, `SAMPLE_DATAPATH`, and others. Use the configuration file in your notebook
- [notebooks/colab_setup.ipynb](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/notebooks/colab_setup.ipynb): instructions for setting up Google Colab
- [notebooks/dataset_download.ipynb](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/notebooks/dataset_download.ipynb): generate a sample dataset (first N rows) from the full dataset for quick experimentation
- [data/raw/sample_SAML-D.csv](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/data/raw/sample_SAML-D.csv): a sample dataset comprising the first 500,000 transactions
- [utils/download_dataset.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/utils/download_dataset.py): download the full dataset, but it requires the Kaggle API
