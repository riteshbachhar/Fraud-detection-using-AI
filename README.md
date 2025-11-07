<h1> Deep Learning for Anti-Money Laundering: Detecting Suspicious Transactions </h1>

<h3>Team Members: Heba Bou KaedBey, Min Shi, Ritesh Bachhar, Khanh Nguyen </h3>

Define Problem: Money laundering is the process of conversion of illicit money which comes out of the crime which is then intermixed with the licit money to make appear legitimate, and it becomes very difficult to distinguish the legitimate money from the illegitimate one.



<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#Dataset">Dataset Overview</a></li>
        <ul>
            <li><a href="#EDA">Exploratory Data Analysis</a></li>
            <li><a href="#Features">Time‑Based and Graph‑Inspired Feature Engineering</a></li>
            <li><a href="#Recasting">Memory Optimization via Integer Recasting</a></li>
            <li><a href="#Train-Test-split">Custom Train-test split</a></li>
        </ul>
    <li><a href="#Baseline">Baseline Model: XGBoost</a></li>
        <ul>
            <li><a href="#Baseline-Architecture">Architecture</a></li>
            <li><a href="#Baseline-Preprocess">Preprocess</a></li>
            <li><a href="#Baseline-Result">Result</a></li>
        </ul>
    <li><a href="#Transformer">Transformer</a></li>
        <ul>
            <li><a href="#Transformer-Architecture">Architecture</a></li>
            <li><a href="#Transformer-Preprocess">Preprocess</a></li>
            <li><a href="#Transformer-Result">Result</a></li>
        </ul>
    <li><a href="#GNN">Graph Neural Network</a></li>
        <ul>
            <li><a href="#GNN-Architecture">Architecture</a></li>
            <li><a href="#GNN-Preprocess">Preprocess</a></li>
            <li><a href="#GNN-Result">Result</a></li>
        </ul>
    <li><a href="#References">References</a></li>
    <li><a href="#Code-Description">Code Description</a></li>
</ul>

---

<h3 id="Introduction">Introduction</h3>

Fraud and money laundering continue to impose significant burdens on financial institutions, with indirect costs often far surpassing direct monetary losses. In North America alone, every dollar lost to fraud in 2023 translated into an average total cost of $4.45, once investigation, recovery efforts, and reputational damage were accounted for. High-profile exposés such as the Panama Papers and Swiss Leaks have further illuminated how offshore financial structures enable laundering by commingling illicit funds with legitimate assets, effectively obscuring their criminal origins.

As laundering schemes grow increasingly sophisticated, the need for robust, data-driven detection systems becomes more urgent. Timely identification and prevention now hinge on scalable models capable of parsing complex transaction patterns and adapting to evolving typologies.

Money laundering typically unfolds in three stages: placement, layering, and integration (see Figure 1). While the placement and integration phases are often concealed and difficult to detect, the layering stage, characterized by a series of transactions between accounts, is more amenable to scrutiny. This phase presents a critical opportunity for intervention, as it involves the movement and transformation of funds designed to break the audit trail. With global laundering estimated at 2–5% of GDP ($800B–$2T), the stakes are high. High-recall detection is not merely desirable—it is essential for identifying suspicious patterns early, minimizing false negatives, and enabling timely escalation to investigative teams. In this context, precision can be sacrificed to ensure that potentially illicit activity is flagged, even at the cost of increased review workload.

In this project, our objective is to develop deep learning models using the SAML-D dataset to identify suspicious financial transactions. We aim to reduce false positives while enhancing recall within anti-money laundering (AML) systems. Additionally, we plan to explore the use of OpenAI's GPT to generate natural language explanations for each flagged transaction, improving interpretability for compliance teams. If time permits, we will deploy the solution to AWS, likely leveraging S3 for storage and scalability.

<!-- Fraud and money laundering continue to burden financial institutions, with indirect costs far exceeding direct losses. In North America, each dollar lost to fraud cost firms $4.45 in 2023, factoring in investigation, recovery, and reputational harm. Meanwhile, Panama Papers and Swiss leaks spotlighted how offshore structures facilitate laundering, blending illicit funds with legitimate ones to obscure their origin. As schemes become more sophisticated, robust data-driven models are essential for timely detection and prevention.

**Goal:** Build a deep learning model on SAML-D to detect suspicious financial transactions, reducing false positives while improving recall in anti-money laundering systems. We also want to try using OpenAI’a GPT to generate natural language explanations for each flagged transaction. And if time permits deploy to AWS (probably S3). -->

<p float="center">
  <img src="/Figures/Money_Laundering_Cycle.png" width="900" />
</p>
<p align="center"><b>Figure 1. Three stages of Money Laundering Cycle.
</b></p>

---

<h3 id="Dataset">Dataset Overview</h3>

The <a href="https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/data">SAML-D dataset</a> is a synthetic dataset of 9.5 million transactions, developed using insights from AML specialists through semi-structured interviews, analysis of existing datasets, and a comprehensive literature review. The dataset includes 28 typologies, 11 normal and 17 suspicious, modelled to reflect real-world senarios, with overlapping behaviours between normaal and suspicious transactions to increase complexity and challenge detection efforts. The dataset includes 676,912 unique accounts conducting transactions across 18 geographic locations, using 13 currencies and 7 payment methods. Among these, 0.104% of the transactions (9,873 entires) are labeled as suspicious representing various money laundering techniques such as layering, rapid fund movement, and transactions to high risk locations (see Figure 2).

<p float="center">
  <img src="/Figures/imbalanced_class.png" width="450" />
<p align="center"><b>Figure 2. The data is highly imbalanced, with only 0.1% of transactions labeled as suspicious.
</b></p>

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

<h4 id="EDA">Exploratory Data Analysis</h4>

28 transaction typologies were identified through structured, semi‑structured interviews with eight subject‑matter experts in anti‑money‑laundering. 
- Eleven typologies were classified as normal: `normal small fan‑out`, `normal fan‑out`, `normal fan‑in`, `normal group`, `normal cash withdrawal`, `normal cash deposit`, `normal periodic`, `normal plus mutual`, `normal mutual`, `normal forward`, and `normal single large`.
- Seventeen were classified as suspicious: `structuring`, `cash withdrawal`, `deposit‑send`, `smurfing`, `layered fan‑in`, `layered fan‑out`, `stacked bipartite`, `behavioral change 1`, `behavioral change 2`, `bipartite`, `cycle`, `fan‑in`, `gather‑scatter`, `scatter‑gather`, `single large`, `fan‑out`, `and over‑invoicing`.

The top 20 laundering types by log count highlight `structuring`, `cash withdrawal`, `deposit‑send`, and `smurfing` as the most frequent suspicious typologies (see Figure 3). By median log amount, `over‑invoicing` dominates the chart and `single large` ranked among the top three, showing that high‑magnitude events drive a different signal than high‑frequency events. Together, these patterns show that frequency‑based and magnitude‑based indicators capture distinct risk strata and therefore both should be incorporated into the detection pipeline.

<p float="center">
  <img src="/Figures/laundering_type.png" width="1000" />
  <img src="/Figures/amount_by_laundering_type.png" width="1000" />
</p>
<p align="center"><b>Figure 3. Laundering types (top) and median log-transformed transaction amounts by type (bottom).
</b></p>

We then examined sender and receiver locations. The majority of both senders and receivers were located in the `UK`, while transactions involving other countries were more evenly distributed. The dataset designated `Mexico`, `Turkey`, `Morocco`, and the `UAE` as higher‑risk countries; these classifications were applied to the analysis rather than inferred directly from the data. Notably, only `Morocco` and `Mexico` appeared among the top 20 receiver bank locations, whereas all four countries appeared in the top 20 sender locations, suggesting asymmetric flows that merit separate sender‑ and receiver‑based scrutiny (see Figure 4).

<p float="center">
  <img src="/Figures/top_receiver_locations.png" width="1000" />
  <img src="/Figures/top_sender_locations.png" width="1000" />
</p>
<p align="center"><b>Figure 4. Histograms of receiver bank locations (top) and sender bank locations (bottom).
</b></p>

Most of the top 20 money‑transfer routes are domestic (UK → UK), but the top‑20 list also includes several UK → designated high‑risk‑country corridors, indicating concentrated outbound flows along specific cross‑border routes. To better understand cross‑border behavior, we examined currency mismatches: among the top 20 currency pairs only GBP → GBP is a same‑currency pair; all other top pairs involved currency conversion. Currency conversion therefore appeared to be a persistent characteristic of high‑volume corridors and may interact with typology and routing signals to reveal elevated risk (see Figure 5).

<p float="center">
  <img src="/Figures/top_20_money_transfer.png" width="1000" />
  <img src="/Figures/top_20_currency_pairs.png" width="1000" />
</p>
<p align="center"><b>Figure 5. Top 20 money transfer routes (top) and currency pairs (bottom).
</b></p>

<h4 id="Features">Time‑Based and Graph‑Inspired Feature Engineering</h4>

From the 12 original features, we derived 8 additional temporal variables from the existing `Time` and `Date` attributes. These include: `year`, `month`, `day_of_month`, `day_of_year`, `day_of_week`, `hour`, `minute`, and `second`. Together, these features allow the model to capture seasonality, periodicity, and fine‑grained temporal patterns in transaction behavior.

In addition, we engineered several domain‑specific features (see Figure 6) designed to capture structural and behavioral signals of anomalous activity:

- `fanin_30d`: The count of unique incoming counterparties over a rolling 30‑day window. This proved to be the single most predictive feature, reflecting the diversity of inbound connections.
- `fan_in_out_ratio`: The ratio of inbound to outbound counterparties over 30 days, highlighting imbalances in transactional relationships.
- `fanin_intensity_ratio`: The `fanin_30d` value normalized by the daily number of received transactions, concentration of inbound activity.
- `amount_dispersion_std`: The standard deviation of transaction amounts per sender, capturing volatility in counterparties' transfer sizes.
- `sent_to_received_ratio_monthly`: The ratio of total received to total sent amounts within a month. Ratios trending toward 1 may indicate circular or balancing behavior that warrants scrutiny
- `back_and_forth_transfers`: The number of transfers exchanged between a sender and receiver within a single calendar day. This is a directed metric: A → B is treated as distinct from B → A.
- `circular_transaction_count`: The number of transactions that eventually return to the original sender, forming a cycle. Cycles may span multiple steps and extend across several days, making them a strong indicator of layering or obfuscation.

<p float="center">
  <img src="/Figures/fanin.JPG" width="250" />
  <img src="/Figures/fanout.JPG" width="250" />
  <img src="/Figures/circular_transaction.JPG" width="300" />
</p>
<p align="center"><b>Figure 6. Transaction topology examples used in feature engineering: (left) fanin (aggregation into a hub), (mid) fanout (dispersion from a hub), and (right) circular_transaction (directed cycle returning to origin).
</b></p>

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

<h3 id="Baseline">Baseline Model: XGBoost</h3>

<h4 id="Baseline-Architecture">Architecture</h4>

We selected XGBoost as our baseline for its resistance to overfitting: its tree‑based architecture handles outliers well, and built‑in regularization and shrinkage mitigate overfitting on noisy transactional data.

We tuned hyperparameters to maximize the F1 score, then applied the best configuration to the final model. Retraining on the combined training and validation sets maximized the available learning signal. We generated class predictions on the held‑out test set and evaluated performance using the confusion matrix, F1 score, precision, and recall. To interpret model behavior and quantify each feature's contribution, we computed SHAP values and ranked features by their mean absolute SHAP importance.

Parameters considered:
- `max_depth`: 3, 5, 10
- `learning_rate`: 0.1, 0.05, 0.01
- `subsample`: 0.3, 0.5, 1

<h4 id="Baseline-Preprocess">Preprocess</h4>

Using the correlation matrix (see Figure 7), we identified and removed eight features that were highly collinear with other predictors, resulting in a final set of 15 features for modeling and analysis. This reduction improved interpretability and reduced redundancy in the feature set prior to training and validation.

<p float="center">
  <img src="/Figures/corr_mat.png" width="1000" />
</p>
<p align="center"><b>Figure 7. Half correlation matrix for the 23 features. Features with high correlations were excluded from model tuning.</b></p>

We trained the model and used a held‑out validation set to tune hyperparameters. We performed a grid search over `max_depth`, `learning_rate`, and `subsample` to identify the best-performing configurations.

**Metrics and rationale:** We selected precision, recall, and F1 score as our primary success metrics because the data were highly imbalanced, with only 0.1% of records in the positive class. We prioritized metrics that reflect ranking quality and the tradeoff between false positives and false negatives rather than overall accuracy. Top five hyperparameter combinations were selected by highest validation F1; precision, recall, and training time for these configurations were reported in Table 1 to illustrate tradeoffs between model performance and computational cost.

**Operational considerations:** We also accounted for training time and GPU resource usage given the dataset size. We used early stopping on the validation metric, recorded wall time for each configuration, and favored hyperparameter combinations that delivered meaningful metric improvements relative to their computational cost.

<p float="center">
  <img src="/Figures/xgboost_hypertuning.JPG" width="800" />
</p>

<h4 id="Baseline-Result">Result</h4>

We trained an XGBoost model on the combined training and validation sets, selecting hyperparameters that yielded the highest F1 score during validation: `max_depth` = 10, `learning_rate` = 0.01, and `subsample` = 1. 

The resulting model achieved a high precision of 0.887, indicating that the majority of flagged transactions were indeed suspicious, an encouraging outcome for reducing false positives and minimizing unnecessary compliance reviews. However, recall remained relatively low at 0.413, suggesting that the model failed to identify a significant portion of true suspicious cases. This trade-off reflects a common challenge in anti-money laundering systems, where precision is often favored to avoid overwhelming investigators, but at the cost of missing subtle or novel laundering behaviors.
The average precision score was 0.5586, reflecting moderate performance across varying decision thresholds and highlighting the model's sensitivity to class imbalance. These metrics are visualized in Figure 8, which presents the confusion matrix (left) and the precision–recall curve (right).

<p float="center">
  <img src="/Figures/XGBoost_confusion_mat.png" width="350" />
  <img src="/Figures/xgboost_pr_curve.png" width="450" />
</p>
<p align="center"><b>Figure 8. Confusion matrix (left) and Precision–Recall curve (right) for the XGBoost model, illustrating classification performance and trade-offs between precision and recall.
</b></p>

We assessed feature importance using SHAP values (see Figure 9). The SHAP summary identified the strongest signals and their directions. The top six features contributing most to classification were `back_and_forth_transfers`, `fanin_30d`, `sent_to_received_ratio_monthly`, `fanin_intensity_ratio`, `Amount`, `circular_transaction_count`, and `currency_mismatch`. Other features had smaller SHAP contributions and do not materially influence classification relative to these variables. These results highlighted behavioral patterns to prioritize during feature engineering and threshold selection and warrant further investigation to confirm they align with domain expertise and are not artifacts of sampling or labeling.

<p float="center">
  <img src="/Figures/xgboost_shap_summary.png" width="1000" />
</p>
<p align="center"><b>Figure 9. SHAP summary plot for the XGBoost model, showing the impact of each feature on model output. Each point represents a SHAP value for a single prediction. Color indicates the feature value (e.g., red = high, blue = low). Features are ranked by mean absolute SHAP value, highlighting their overall importance.</b></p>

We also visualized SHAP values for four key features: `back_and_forth_transfers`, `circular_transaction_count`, `currency_mismatch`, and `high_risk_sender` (see Figure 10). In each plot, color intensity represented the logarithmic scale of transaction amounts, helping contextualize feature impact across varying transaction sizes.

The feature `back_and_forth_transfers` captured the number of transfers between the same sender and receiver within a single day. We observed that low values (1 or 2 transfers) were associated with positive SHAP values, indicating the model leaned toward predicting class `1`, i.e. suspicious or laundering-related activity. In contrast, values of 3 or more were linked to negative SHAP values, suggesting the model interpreted these as normal transactions. This pattern may reflect the model's sensitivity to unusually short, reciprocal transfer chains.

Interestingly, `circular_transaction_count` = 1 strongly corresponded to negative SHAP values, reinforcing the model's confidence in classifying such transactions as normal. However, higher counts did not consistently yield positive SHAP values, indicating that this feature alone may not be a strong predictor of laundering behavior without additional context.

Both `currency_mismatch` and `high_risk_sender` exhibited consistently positive SHAP values across the dataset, suggesting that their presence reliably increased the model's likelihood of flagging a transaction as suspicious. These features likely captured regulatory red flags—such as inconsistent currency flows or involvement of known high-risk entities—that contribute meaningfully to the model's decision-making.

<p float="center">
  <img src="/Figures/xgboost_shap_back_and_forth_transfers.png" width="400" />
  <img src="/Figures/xgboost_shap_circular_transaction_count.png" width="400" />
  <img src="/Figures/xgboost_shap_currency_mismatch.png" width="400" />
  <img src="/Figures/xgboost_shap_high_risk_sender.png" width="400" />
</p>

**Figure 10. SHAP values plotted against  (top left) `back_and_forth_transfers`,  (top right) `circular_transaction_count`,  (bottom left) `currency_mismatch`,  (bottom right) `high_risk_sender`.  Color intensity reflects the logarithmic scale of transaction amounts.**

---

<h3 id="Transformer">Transformer</h3>

The transformer was selected as one of the deep learning models, drawing inspiration from the Tab-AML Tranformer in "Tab-AML: A Transformer Based Transaction Monitoring Model for Anti-Money Laundering." This choice was motivated by the Transformer's strong ability to capture complex relationships among categorical and numerical features in financial data. 

Each Categorical feature is first transformed into a learned embedding vector. These embeddings are then processed by two encoder modules:
- A micro encoder that focuses on the sender-receiver relationship, learning direct transaction-level dependancies. 
- A macro-encoder that integrates all categorical and continuous features to learn broader contextual and temporal patterns. 

Through multi-head self attention residual attention and shared embedding, the Transformer captures complex relationships among transaction features. Residual attention ensure that previously learned information is preserved through multiple layers, enabling the model to understand complex financial behaviors at differnt levels of detail. Thus, it is suitable for detecting indirect money laundering activities.

<h4 id="Transformer-Architecture">Architecture</h4>

- Categorical Features: Each feature is embedded into a vector split into shared and individual components.
- Micro Attention Encoder: Learns relationships between selected key features (sender and receiver account).
- Macro Attention Encoder: Aggregates contextual information across all categorical features. 
- MLP: Combines flattened Transformer output and scaled continuous features through three GELU-activated linear layers with dropout. 
- Loss Function: Uses Focal Loss to focus learning on the minority class. 
- Sampler: A weightedRandomSampler balances the training data by oversampling rare laundering examples.

<p float="center">
  <img src="/Figures/transformer_diagram.jpg" width="500" />
</p>
<p align="center"><b>Figure 11. Overview of Transformer Model architecture.</b></p>

<h4 id="Transformer-Preprocess">Preprocess</h4>

- Dates and times are merged into a unified timestamp, and temporal features (day, month, year, hour, weekday, weekend) are derived. 
- The amount feature is log-transformed to stabilize variance. 
- High-cardinality categorical columns (Sender_account, Receiver_account) are hashed into 50,000 buckets for efficient embedding. 
- Continuous features are standardized using StandardScaler. 

<h4 id="Transformer-Result">Result</h4>

---

<h3 id="GNN">Graph Neural Network</h3>

Traditional rule-based methods have proven inadequate for detecting the sophisticated money laundering patterns prevalent in today's financial system. Recent work has demonstrated that Graph Neural Networks can effectively learn from graph-structured data, achieving substantial improvements in financial fraud detection. Building on these advances, our objective is to develop a Graph Neural Network model that detects money laundering activities with greater than 85% recall while maintaining operational precision, leveraging the explicit network structures embedded in the SAML-D dataset.

**Advantages of GNN:**

- **Relational reasoning:** Captures dependencies between account
- **Structural pattern recognition:** Explicitly models network typologies
- **Information propagation:** Aggregates multi-hop neighborhood features

**Graph Construction & GNN Models:**
We propose to construct a transaction graph where each account represents a node and each transaction forms a directed edge between accounts. Node features capture aggregate account characteristics, including transaction statistics, network metrics, behavioral patterns, and temporal statistics. Edge features encode transaction-specific attributes such as amount, inter-arrival time, geographic properties, and payment type.

We propose to investigate two primary GNN architectures: **(1) GraphSAGE**, which enables scalable neighborhood sampling and aggregation for efficient learning on large graphs, and **(2) Graph Attention Networks~(GAT)**, which learn adaptive attention weights over neighbors to identify the most relevant connections for risk assessment and provide interpretability for regulatory compliance.

<h4 id="GNN-Architecture">Architecture</h4>

- GRUCell: The GRUCell layer updates node hidden states by integrating current node features with previous temporal information, capturing sequential dependencies in transaction data.
- GATConv (GNN1): The first GATConv layer aggregates neighbor information using attention mechanisms, refining node representations based on relevant sender-receiver relationships.
- GATConv (GNN1): The second GATConv layer further enhances node features through additional message passing, deepening the model's ability to detect complex laundering patterns.
- Linear (lin): The linear layer combines concatenated sender, receiver, and edge features to produce a single logit for binary edge classification, determining the likelihood of laundering.

Money laundering is fundamentally a network-based problem, not an individual one. Thus, we implemented a temporal neural network to flag suspicious transactions (see Figure 12). We first convert transactions into graphs using a 7-day window, where each node represents an account, and edges represent the transactions. To prevent data leakage, we only calculated the account-level statistics over that specific time window. Forward pass through the network goes as follows: The model performs a temporal update for each node via GRU cell. This updates node's current state $h_{t-1}$ with current node features ($X_t$), learning node-level temporal dynamics. Next, the proposed hidden state is passed through 3-layer GraphSAGE network. This step updates the node features by a method called message passing. This allows the nodes to learn about their neighbourhood and create a final embedding $h_{t}$. Finally, in the classification step, the model predicts each transaction. The hidden state of the sender node and the hidden state of the receiver node are concatenated with the transaction feature and passed through an MLP. This calculates a single logit, which is compared with the true label to compute a loss function.


<p float="center">
  <img src="/Figures/TGNN_diagram.JPG" width="1000" />
</p>
<p align="center"><b>Figure 12. Architecture of the Temporal Graph Neural Network (TGNN).</b></p>

<h4 id="GNN-Preprocess">Preprocess</h4>

- We aggregate the dataset into seven-day intervals to capture temporal network dynamics and structural patterns, reducing computational overhead during graph-based processing. For each interval, we compute account-level statistics, including transaction frequency, fan-in/fan-out degrees, and median transaction amounts. Transaction amounts are log-transformed to mitigate skewness and enhance the applicability of deep learning models for detecting complex laundering typologies.
- Edges are constructed based on sender-receiver pairs, incorporating edge attributes like log-transformed amounts, currency mismatch indicators, cross-border flags, and day of week to enrich the relational context.

<h4 id="GNN-Result">Result</h4>

---

<h3 id="References">References</h3>
<ul>
<li>United Nations Office on Drugs and Crime. Money Laundering. United Nations, <a href=https://www.unodc.org/unodc/en/money-laundering/overview.html>https://www.unodc.org/unodc/en/money-laundering/overview.html</a>. Accessed 29 Oct. 2025.
</li>
<li>LexisNexis Risk Solutions. <a href=https://risk.lexisnexis.com/about-us/press-room/press-release/20240424-tcof-financial-services-lending>Every Dollar Lost to a Fraudster Costs North America's Financial Institutions $4.45</a>. LexisNexis Risk Solutions, 24 Apr. 2024.</li>
<li>Grigorescu, Petre-Cornel, and Antoaneta Amza. "Explainable Feature Engineering for Multi-class Money Laundering Classification." Informatica Economică, vol. 29, no. 1, 2025, pp. 64–78, doi: 10.24818/issn14531305.</li>
<li>B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan and G. Aksu, "Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset," 2023 IEEE International Conference on e-Business Engineering (ICEBE), Sydney, Australia, 2023, pp. 47-54, doi: 10.1109/ICEBE59045.2023.00028.</li>
<li>Oztas, Berkan, et al. "Tab-AML: A Transformer Based Transaction Monitoring Model for Anti-Money Laundering." 2025 IEEE Conference on Artificial Intelligence (CAI). IEEE, 2025.</li>
<li>Phan, T.T.T. (2025). Leveraging Graph Neural Networks and Optimization Algorithms to Enhance Anti-money Laundering Systems. In: Le Thi, H.A., Le, H.M., Nguyen, Q.T. (eds) Advances in Data Science and Optimization of Complex Systems. ICAMCS 2024. Lecture Notes in Networks and Systems, vol 1311. Springer.</li>
</ul>

---

<h3 id="Code-Description">Code Description</h3>

- [config.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/config.py): include various paths such as `DATAPATH`, `SAMPLE_DATAPATH`, and others. Use the configuration file in your notebook
- [notebooks/colab_setup.ipynb](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/notebooks/colab_setup.ipynb): instructions for setting up Google Colab
- [notebooks/dataset_download.ipynb](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/notebooks/dataset_download.ipynb): generates a sample dataset (first N rows) from the full dataset for quick experimentation
- [data/raw/sample_SAML-D.csv](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/data/raw/sample_SAML-D.csv): a sample dataset comprising the first 500,000 transactions
- [utils/download_dataset.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/utils/download_dataset.py): download the full dataset, but it requires the Kaggle API
- [custom_split_polars.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/custom_split_polars.py): returns three Polars DataFrames: training, validation, and test sets. The input `df` must be a Polars DataFrame.
- [recast_polars.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/recast_polars.py): downcasts integer columns to smaller integer dtypes (for example, from `Int64` to `Int32` or `Int16`) to reduce memory usage. The input `df` must be a Polars DataFrame and the function returns the recast Polars DataFrame.
- [circular_transaction_count_polars.py](https://github.com/hebabkb/Deep-Learning-for-Anti-Money-Laundering-Detecting-Suspicious-Transactions/blob/main/circular_transaction_count_polars.py): computes circular_transaction_count using a calendar‑month sliding window, ensuring rustworkx is installed.
