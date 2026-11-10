#!/usr/bin/env python3
"""
Script to generate the complete faculty-aligned docs/manuscript.tex
for SWIFT-U5 in strict compliance with:
1. Faculty 38-page Research Phenomenon Writing Guidelines (Method-based paper)
2. Phase 6 Results Freeze (docs/phase6_results_freeze.md)
3. Version 1.2 Claim Alignment Log (docs/phase6_final_claim_alignment_log.md)
"""
import sys
from pathlib import Path

def generate_manuscript():
    tex = r"""\documentclass[a4paper,fleqn]{cas-sc}

\usepackage[numbers,sort&compress]{natbib}
\bibliographystyle{model1-num-names}
\usepackage{array}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{xurl}
\usepackage{float}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\graphicspath{{figures/}}

% Inline figures (CAS redefines floats; avoid caption package conflicts)
\makeatletter
\providecolor{scolor}{gray}{0.2}
\newcommand{\casinlinefig}[4][\linewidth]{%
  \par\vskip8pt
  \refstepcounter{figure}%
  {\sffamily\small\centering
  \makebox[\textwidth][c]{\includegraphics[width=#1,height=0.48\textheight,keepaspectratio]{#2}}\par\vskip6pt
   \textbf{\color{scolor}\figurename~\thefigure:}~#3\par}%
  \label{#4}%
  \par\vskip8pt
}
\newcommand{\casinlinefigtall}[4][\linewidth]{%
  \par\vskip8pt
  \refstepcounter{figure}%
  {\sffamily\small\centering
  \makebox[\textwidth][c]{\includegraphics[width=#1,height=0.58\textheight,keepaspectratio]{#2}}\par\vskip6pt
   \textbf{\color{scolor}\figurename~\thefigure:}~#3\par}%
  \label{#4}%
  \par\vskip8pt
}
\makeatother

% References: single-column bibliography section
\makeatletter
\renewcommand{\bibsection}{%
  \clearpage
  \section*{References}\vspace{0.25\baselineskip}%
}
\AtBeginDocument{%
  \setlength{\bibsep}{0pt plus 0.2pt}%
}
\makeatother

\begin{document}
\let\WriteBookmarks\relax
\def\printorcid{}

\shorttitle{Survey-Weighted Child Mortality Risk Stratification in Bangladesh}
\shortauthors{Chowdhury et al.}

\title[mode=title]{SWIFT-U5: A Survey-Weighted and Temporally Calibrated Machine Learning Framework for Population-Level Child Mortality Risk Stratification in Bangladesh}

\author[1]{Md. Rafiul Azam Chowdhury}
\author[1]{Rezaul Mursalin Khan}
\author[1]{Fatema Tuz Zohura Rifa}
\author[1]{Tanjim Ahmed Irfad}
\author[1]{Arafat Hussain Nahid}
\author[1]{Ohidujjaman}
\affiliation[1]{organization={Department of Computer Science and Engineering, United International University},
            addressline={},
            city={Dhaka},
            postcode={1212},
            country={Bangladesh}}

\begin{abstract}
Under-five child mortality remains a critical global health challenge under United Nations Sustainable Development Goal 3.2, particularly in low- and middle-income countries like Bangladesh, where survival disparities reflect multi-factorial maternal, socioeconomic, environmental, and healthcare access vulnerabilities. Although Demographic and Health Survey (DHS) microdata provide large nationally representative birth cohorts, published machine learning applications in child mortality prediction suffer from three pervasive methodological weaknesses: retrospective survival leakage from postnatal proxies (such as breastfeeding duration and maternal recall of recent child illness symptoms), complete disregard of complex multi-stage survey sampling designs, and reliance on uncalibrated default decision thresholds ($\tau = 0.5$) under in-distribution random splits that conceal temporal distribution shift. To resolve these challenges, this study formulates, implements, and prospectively benchmarks an audited, survey-weighted, and drift-calibrated machine learning framework---termed \textbf{SWIFT-U5} (\textbf{S}urvey-\textbf{W}eighted, \textbf{I}mportance-corrected, \textbf{F}orward-validated \textbf{T}emporal Risk Model for Under-Five Mortality)---for population-level risk stratification and community outreach prioritization simulation in Bangladesh. SWIFT-U5 restricts predictors to pre-birth maternal reproductive history and durable household assets, eliminating survival-contingent proxies, and couples gradient tree boosting with three sequential methodological modules: transductive covariate-shift density weighting with temporal recency decay (Module~A), empirical Bayes division-level logit shrinkage (Module~B), and drift-aware Platt recalibration (Module~C). We harmonized an exposure-complete national cohort of $51{,}489$ live births and $2{,}338$ documented under-five deaths across six sequential Bangladesh Demographic and Health Survey (BDHS) rounds spanning 18 years (2004 to 2022), partitioned into historical development (2004--2014, $N = 29{,}686$), validation/calibration anchoring (2017--18, $N = 8{,}684$), and locked prospective out-of-time testing (2022, $N = 13{,}119$). On the locked BDHS 2022 test cohort ($N = 13{,}119$, $408$ deaths, survey-weighted prevalence $2.99\%$), primary SWIFT-U5 achieved a survey-weighted ROC-AUC of $0.625$ [95\% PSU cluster-bootstrap CI: 0.587--0.663] versus $0.638$ [95\% CI: 0.603--0.673] for uncalibrated baseline XGBoost. In descriptive comparison between configurations, SWIFT-U5 restored probability calibration, achieving a survey-weighted Brier score of $0.0287$ compared to $0.1649$ for uncalibrated class-rebalanced XGBoost, outperforming the uninformative constant-prevalence null benchmark of $0.0290$ ($\text{BSS} = +0.0103$), with a calibration slope of $0.806$ and intercept of $-0.905$. In simulated retrospective community outreach prioritization at 10\% capacity ($1{,}312$ records prioritized containing $107$ unweighted recorded deaths, corresponding to approximately $97$ weighted true-positive event equivalents under DHS sampling weights), the model achieved a survey-weighted Sensitivity of $24.60\%$, Specificity of $90.45\%$, Precision (PPV) of $7.70\%$, $\text{NNS} = 13.0$, and a weighted estimated prioritization yield of approximately 77 per 1{,}000 prioritized records ($77.0$ per $1{,}000$; $2.57\times$ efficiency over random selection), rising to an estimated yield of $123.3$ per $1{,}000$ prioritized ($\text{NNS} = 8.1$) in high-burden Rangpur division. At the validation-locked Youden threshold ($\tau_w = 0.04546$), the model achieved $47.18\%$ weighted sensitivity, $70.61\%$ specificity, $58.90\%$ balanced accuracy, and $69.91\%$ weighted accuracy (contrasting with a $97.01\%$ trivial all-negative baseline). Subgroup audits demonstrated comparable ranking sensitivity across rural ($23.3\%$) and urban ($28.5\%$) strata, while concentrating outreach yield among socioeconomically disadvantaged households. Importantly, these findings reflect retrospective risk-ranking performance on cross-sectional survey-recorded deaths rather than demonstrated prospective clinical screening effectiveness; because the outcome is recorded death by maternal interview and living children have incomplete follow-up, prospective field evaluation among eligible living children is required before operational deployment claims can be made.
\end{abstract}

\begin{keywords}
Under-five mortality \sep Child survival risk prediction \sep Demographic and Health Surveys \sep Machine learning in public health \sep Survey-weighted learning \sep Temporal validation \sep Probability calibration \sep Population risk stratification
\end{keywords}

{\hfuzz=120pt\maketitle}

\section{Introduction}
\label{sec:introduction}

Under-five child mortality (U5M) remains a profound global public health challenge, directly targeted by United Nations Sustainable Development Goal (SDG) 3.2, which calls on all nations to reduce under-five mortality to at least as low as 25 per 1,000 live births by 2030 \cite{who2023,unicef2023,black2010,you2015}. In low- and middle-income countries (LMICs), child mortality disparities reflect intricate interactions among biological vulnerability, maternal reproductive history, household socioeconomic deprivation, water, sanitation, and hygiene (WASH) infrastructure, and regional healthcare access barriers \cite{chowdhury2013,arifeen2013,omran2005}. Over the past three decades, Bangladesh has achieved extraordinary public health gains, reducing national under-five mortality from approximately $133$ deaths per $1{,}000$ live births in 1993--94 to $29.9$ per $1{,}000$ in 2022 \cite{niport2024}. However, as national mortality rates have declined, the remaining burden of child deaths has become increasingly concentrated in the vulnerable neonatal period (which accounts for over 70\% of all under-five deaths in Bangladesh) and within historically disadvantaged geographic regions, such as Sylhet and Rangpur \cite{khan2017,khan2021,mazumder2024causes,mazumder2024stagnant}. In this contemporary epidemiological landscape, traditional one-size-fits-all maternal-child health interventions face diminishing marginal returns. Consequently, there is an urgent public health imperative for predictive machine learning systems that can leverage standardized household demographic surveys to accurately stratify mortality risk at the population level, identify high-risk clusters, and guide community healthcare worker outreach before adverse birth outcomes occur.

Despite extensive research exploring supervised machine learning for child mortality prediction from Demographic and Health Survey (DHS) microdata across Africa and South Asia \cite{oduse2024,naznin2025,mabula2025,bitew2020,saroj2022,asmare2026,fahim2026}, existing studies exhibit critical methodological limitations that invalidate their real-world applicability (Table~\ref{tab:prior_work_positioning}). First, several published benchmarks introduce severe retrospective survival leakage by including postnatal covariates that directly proxy survival status. Most prominently, Naznin et al. \cite{naznin2025} incorporated ``currently breastfeeding'' and ``duration of breastfeeding'' as predictive features for under-five mortality in BDHS 2022 microdata; because deceased infants cannot be breastfed at maternal interview, tree algorithms trivially split on feeding duration, manufacturing an illusion of near-perfect classification ($\text{ROC-AUC} = 0.9979$) that collapses completely in prospective deployment. Similarly, retrospective child illness variables (e.g., fever, diarrhea, or acute respiratory infection in the two weeks preceding the survey) are administered exclusively to living children in DHS protocols, inducing structural missingness that acts as a deterministic survival indicator. Second, previous studies widely employ synthetic oversampling techniques (such as SMOTE \cite{chawla2002}) across pooled cohorts prior to validation partitioning, artificially distorting class prevalence, degrading posterior probability calibration, and producing overconfident decision rules. Third, existing machine learning pipelines uniformly ignore the complex multi-stage survey architecture of DHS microdata (clustering, stratification, and unequal sampling probabilities), violating standard i.i.d. assumptions and introducing substantial parameter bias \cite{macnell2023,lumley2004,oh2026}. Fourth, prior models rely overwhelmingly on k-fold random cross-validation, randomly shuffling live births from the same survey round across training and test splits. This practice conceals macroeconomic transitions, secular health-system expansions, and temporal distribution shifts, creating severe out-of-distribution performance inflation. Finally, published models evaluate performance under arbitrary default decision thresholds ($\tau = 0.5$) that classify virtually all children as non-events under severe class imbalance ($33:1$), while omitting probability calibration metrics (Brier score, calibration slope, and calibration-in-the-large) and actionable public health capacity metrics (Number Needed to Screen, NNS).

\begin{table}[pos=htbp]
  \centering
  \caption{Methodological characteristics reported or identifiable in recent DHS machine-learning studies on child mortality. ``Not rep.'' indicates that the cited study did not describe the corresponding design element; ``Sens.'' indicates evaluation only as an auxiliary sensitivity check; ``Full'' indicates complete integration into training, validation, and testing.}
  \label{tab:prior_work_positioning}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\centering\arraybackslash}p{0.08\linewidth}*{6}{>{\centering\arraybackslash}X}@{}}
    \toprule
    Study & Target Domain & Wts & Temp & Calib & Dec & Fair & Leak-Free \\
    \midrule
    Oduse et al. (2024) \cite{oduse2024} & U5M (Nigeria) & Not rep. & Not rep. & Not rep. & Not rep. & Not rep. & No \\
    Naznin et al. (2025) \cite{naznin2025} & U5M (BDHS 2022) & Not rep. & Not rep. & Not rep. & Not rep. & Not rep. & \textbf{No (Severe)} \\
    Mabula et al. (2025) \cite{mabula2025} & U5M (Tanzania) & Not rep. & Not rep. & Not rep. & Not rep. & Not rep. & Unclear \\
    Bitew et al. (2020) \cite{bitew2020} & U5M (Ethiopia) & Not rep. & No & Not rep. & Not rep. & Not rep. & Limited \\
    Haque \& Kabir (2026) \cite{haque2026} & Stunting (BDHS) & Not rep. & Yes & Not rep. & Not rep. & Yes & Partial \\
    Fahim et al. (2026) \cite{fahim2026} & U5M (BDHS 4-wave) & Sens. & Yes & Yes & Yes & Yes & Yes \\
    \textbf{SWIFT-U5 (This Study)} & \textbf{U5M (BDHS 6-wave)} & \textbf{Full} & \textbf{Yes (5 Cuts)} & \textbf{Yes} & \textbf{Yes} & \textbf{Yes} & \textbf{Yes (Audited)} \\
    \bottomrule
  \end{tabularx}
\end{table}

To resolve these widespread methodological deficiencies, the primary objective of this research is to develop, formalize, and prospectively benchmark an audited, survey-weighted, and drift-calibrated machine learning framework---termed \textbf{SWIFT-U5} (\textbf{S}urvey-\textbf{W}eighted, \textbf{I}mportance-corrected, \textbf{F}orward-validated \textbf{T}emporal Risk Model for Under-Five Mortality)---for population-level child mortality risk stratification and community outreach simulation in Bangladesh. To achieve this objective, SWIFT-U5 fundamentally redesigns the predictive modeling pipeline across four pillars. First, it enforces an audited pre-birth predictor timing protocol, restricting candidate features to maternal reproductive history, antenatal care compliance, and durable household assets while strictly eliminating all survival-contingent postnatal proxies. Second, it integrates DHS complex survey sampling weights, primary sampling unit (PSU) clustering, and administrative stratification end-to-end throughout loss function optimization, cross-validation partitioning, and bootstrap inference. Third, it introduces a modular three-stage prospective architecture: Module~A applies transductive covariate-shift density weighting with exponential recency decay to counter longitudinal distribution drift; Module~B implements an empirical Bayes shrinkage head that regularizes division-specific risk logits; and Module~C applies drift-aware Platt recalibration fitted on out-of-fold validation predictions from a strictly held-out penultimate survey wave to restore true posterior probability scaling. Fourth, it implements a prospective forward-temporal validation lifecycle that evaluates models on future unseen survey rounds, preventing temporal leakage and isolating genuine out-of-time generalizability.

The empirical scope of this study encompasses an exposure-complete master cohort of $51{,}489$ live births and $2{,}338$ documented under-five deaths assembled across six sequential BDHS survey rounds spanning 18 years (2004, 2007, 2011, 2014, 2017--18, and 2022). The proposed framework is designed specifically for public health resource allocation, demographic vulnerability mapping, and simulated outreach prioritization, where local healthcare supervisors allocate limited community health worker visits to households with the highest predicted risk. We explicitly delineate the scientific boundaries of this investigation: the analytical outcome is recorded under-five death by maternal interview, and living children under five years of age have incomplete follow-up (right-censoring); furthermore, Module~A incorporates unlabeled target covariates within a transductive domain adaptation framework. Consequently, simulated outreach prioritization metrics represent retrospective ranking concordance on survey-recorded vulnerabilities rather than demonstrated prospective clinical efficacy. Prospective field trials enrolling living children are required before operational healthcare delivery claims can be substantiated. Based on these verified foundations, this study establishes six distinct contributions:
\begin{enumerate}
    \item \textbf{End-to-End Complex Survey Integration:} Unlike prior benchmarks that evaluate survey weights post-hoc or omit them entirely, SWIFT-U5 integrates Horvitz-Thompson sampling weights (\texttt{v005}), PSU clustering (\texttt{v021}), and geographic stratification (\texttt{v022}) into loss function weighting, cluster-grouped validation folds (\texttt{GroupKFold}), and design-based cluster bootstrapping.
    \item \textbf{Audited Predictor Timing Protocol:} We establish a rigorous temporal feature eligibility framework, eliminating postnatal feeding and illness recall variables that induce structural data leakage, while validating durable household assets as structural background proxies.
    \item \textbf{Controlled $2 \times 2$ Factorial Validation Benchmark:} Across five sequential forward-temporal cuts, we systematically isolate the statistical inflation induced by in-distribution random cross-validation versus forward temporal testing, and unweighted versus survey-weighted learning.
    \item \textbf{Drift-Aware Modular Recalibration Architecture:} We formulate, implement, and validate the three-module SWIFT-U5 architecture (Modules~A, B, and~C), reversing severe logit distortion under extreme class imbalance and restoring probability calibration on locked prospective test data.
    \item \textbf{Granular Division-Level Policy Capacity Layer:} Moving beyond monolithic national averages, we evaluate division-level predictability across all eight administrative divisions of Bangladesh, providing prioritized capacity tables with 95\% PSU cluster-bootstrap confidence intervals for prioritization workload (NNS) and survey-weighted estimated prioritization yield.
    \item \textbf{Comprehensive Algorithmic Equity Audit:} We conduct a formal demographic equity audit across wealth quintiles, urban/rural residence, and maternal education levels conforming to TRIPOD+AI \cite{collins2024} and PROBAST \cite{wolff2019} reporting standards, explicitly designating small-sample strata as Insufficient Evidence rather than certified fairness.
\end{enumerate}

\section{Conventional Methods}
\label{sec:conventional_methods}

To establish a rigorous technical foundation for understanding SWIFT-U5, this section presents the established statistical and machine learning methodologies from which our framework is derived, detailing their working principles, mathematical formulations, and the specific technical limitations that motivate our proposed approach.

\subsection{Conventional Supervised Learning for Child Mortality Risk Prediction}
\label{sec:conv_ml}

Supervised machine learning models for binary child survival risk prediction seek to learn a mapping $f: \mathcal{X} \to \mathbb{R}$ from a $d$-dimensional feature space $\mathcal{X} \subseteq \mathbb{R}^d$ to a continuous decision score or log-odds of mortality $y \in \{0, 1\}$. In conventional epidemiology and machine learning practice, gradient boosted decision tree (GBDT) ensembles (such as XGBoost \cite{chen2016} and LightGBM \cite{ke2017}) and regularized logistic regression have emerged as standard benchmark architectures \cite{saroj2022,mabula2025,bitew2020}. 

For a training dataset $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, standard gradient tree boosting minimizes an empirical objective function comprising a convex loss function $\ell$ and an ensemble model complexity penalty $\Omega$:
\begin{equation}
  \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \ell(y_i, f(\mathbf{x}_i; \boldsymbol{\theta})) + \sum_{k=1}^K \Omega(f_k),
  \label{eq:standard_obj}
\end{equation}
where $f(\mathbf{x}_i) = \sum_{k=1}^K f_k(\mathbf{x}_i)$ represents the sum of $K$ regression trees, and $\Omega(f_k) = \gamma T_k + \frac{1}{2}\lambda \sum_{j=1}^{T_k} w_{kj}^2$ penalizes tree leaf count $T_k$ and leaf weight magnitudes $w_{kj}$. Under standard binary classification, $\ell$ corresponds to binary cross-entropy (logistic loss):
\begin{equation}
  \ell(y_i, f(\mathbf{x}_i)) = - \Big[ y_i \log \sigma(f(\mathbf{x}_i)) + (1 - y_i) \log(1 - \sigma(f(\mathbf{x}_i))) \Big],
  \label{eq:bce}
\end{equation}
where $\sigma(z) = (1 + \exp(-z))^{-1}$ is the sigmoid activation function. 

To address the extreme class imbalance characteristic of rare child mortality events (where national prevalence $\pi \approx 3\%$), conventional implementations universally apply a heuristic cost-sensitive scale-position multiplier $\omega = \frac{N - N_{\text{pos}}}{N_{\text{pos}}} \approx \frac{1 - \pi}{\pi}$ to the positive class loss:
\begin{equation}
  \ell_{\text{rebal}}(y_i, f(\mathbf{x}_i)) = - \Big[ \omega \cdot y_i \log \sigma(f(\mathbf{x}_i)) + (1 - y_i) \log(1 - \sigma(f(\mathbf{x}_i))) \Big].
  \label{eq:bce_weighted}
\end{equation}
\textit{Limitations:} While scale-position rebalancing encourages tree splits to isolate rare death events and improves uncalibrated pairwise rank discrimination ($\text{ROC-AUC}$), it introduces a massive constant log-odds shift:
\begin{equation}
  f_{\text{rebal}}(\mathbf{x}) \approx f(\mathbf{x}) + \log(\omega).
  \label{eq:logit_shift}
\end{equation}
Because $\omega \approx 32.4$ under $33:1$ imbalance, raw output probabilities $\hat{p}_i = \sigma(f_{\text{rebal}}(\mathbf{x}_i))$ are systematically inflated by several hundred percent, destroying posterior probability calibration and rendering the raw probabilities useless for clinical triage.

\subsection{Complex Survey Sampling and Design Effects}
\label{sec:conv_survey}

Demographic and Health Surveys do not draw simple random samples (SRS) from the population. Instead, they employ a two-stage stratified cluster sampling design \cite{croft2023,dhsrecode2018}. In Stage 1, primary sampling units (PSUs, designated as \texttt{v021}), corresponding to census enumeration areas containing approximately 100--250 households, are selected with probability proportional to size within administrative strata (\texttt{v022}). In Stage 2, a fixed number of households (typically 30) are systematically selected within each sampled PSU.

To obtain unbiased population parameters, observations are assigned Horvitz-Thompson sampling weights $w_i^{\text{DHS}} = \pi_i^{-1}$, representing the inverse probability of selection, adjusted for household non-response and post-stratification \cite{lumley2004}:
\begin{equation}
  \hat{\mu}_{\text{HT}} = \frac{\sum_{i=1}^N w_i^{\text{DHS}} y_i}{\sum_{i=1}^N w_i^{\text{DHS}}}.
  \label{eq:ht_mean}
\end{equation}
The survey design induces intra-cluster correlation among children born in the same community, resulting in design-based variance inflation quantified by the Design Effect ($\text{DEFF}$):
\begin{equation}
  \text{DEFF} = \frac{\text{Var}_{\text{complex}}(\hat{\mu})}{\text{Var}_{\text{SRS}}(\hat{\mu})} \approx 1 + \rho (m - 1),
  \label{eq:deff}
\end{equation}
where $\rho$ is the intra-cluster correlation coefficient and $m$ is the average cluster size.

\textit{Limitations:} Standard machine learning libraries (such as scikit-learn and standard GBDT packages) treat observations as independent and identically distributed ($i.i.d.$). Fitting unweighted objective functions optimizes sample-frequency accuracy rather than population-representative risk, biasing predictions toward oversampled geographic strata and underestimating true standard errors by ignoring PSU clustering.

\subsection{Distribution Shift and Covariate Adaptation}
\label{sec:conv_covariate_shift}

When predictive models are trained on historical survey rounds and deployed on future cohorts, the underlying data distribution undergoes temporal drift. Formally, let $P_{\text{source}}(\mathbf{x}, y)$ denote the joint distribution of features and mortality outcomes in historical development waves, and let $P_{\text{target}}(\mathbf{x}, y)$ denote the distribution in the prospective deployment cohort. Under the classic covariate shift assumption \cite{shimodaira2000,sugiyama2012}:
\begin{equation}
  P_{\text{source}}(y \mid \mathbf{x}) = P_{\text{target}}(y \mid \mathbf{x}), \quad \text{while } P_{\text{source}}(\mathbf{x}) \neq P_{\text{target}}(\mathbf{x}).
  \label{eq:cov_shift_assumption}
\end{equation}
Under covariate shift, the standard empirical risk minimizer on source data converges to a sub-optimal target decision boundary. The theoretical importance-weighting solution weights each source instance by the target-to-source density ratio:
\begin{equation}
  \beta(\mathbf{x}) = \frac{P_{\text{target}}(\mathbf{x})}{P_{\text{source}}(\mathbf{x})},
  \label{eq:density_ratio}
\end{equation}
minimizing the weighted empirical risk:
\begin{equation}
  \hat{R}_{\beta}(f) = \frac{1}{N_{\text{source}}} \sum_{i=1}^{N_{\text{source}}} \beta(\mathbf{x}_i) \ell(y_i, f(\mathbf{x}_i)).
  \label{eq:weighted_risk}
\end{equation}
In conventional machine learning, density ratios can be estimated via probabilistic classification \cite{sugiyama2012}. By training a domain classifier $g(\mathbf{x}) = P(S = 1 \mid \mathbf{x})$ to distinguish target instances ($S = 1$) from source instances ($S = 0$), Bayes' rule yields:
\begin{equation}
  \beta(\mathbf{x}) = \frac{P(\mathbf{x} \mid S=1)}{P(\mathbf{x} \mid S=0)} = \frac{P(S=1 \mid \mathbf{x})}{P(S=0 \mid \mathbf{x})} \cdot \frac{P(S=0)}{P(S=1)} = \frac{g(\mathbf{x})}{1 - g(\mathbf{x})} \cdot \frac{N_{\text{source}}}{N_{\text{target}}}.
  \label{eq:domain_ratio}
\end{equation}
\textit{Limitations:} In longitudinal demographic surveys spanning multiple decades, unconstrained density ratios $\beta(\mathbf{x})$ exhibit severe variance explosion when target covariates fall into sparse source regions. Furthermore, conventional importance weighting assumes stationary time gaps, ignoring the continuous monotonic decay of feature relevance over decades of health-system development.

\subsection{Post-Hoc Probability Calibration}
\label{sec:conv_calibration}

Probability calibration assesses the agreement between predicted risk estimates $\hat{p} = \hat{P}(Y=1 \mid \mathbf{x})$ and actual observed empirical event rates:
\begin{equation}
  P(Y = 1 \mid \hat{P} = p) = p, \quad \forall p \in [0, 1].
  \label{eq:perfect_cal}
\end{equation}
In clinical risk prediction, post-hoc calibration transforms uncalibrated raw model margins $s_i = f(\mathbf{x}_i)$ into calibrated probabilities. The most widely established parametric calibration method is Platt scaling \cite{platt1999}, which fits a univariate logistic transformation:
\begin{equation}
  \hat{p}_i^{\text{Platt}} = \sigma(A \cdot s_i + B) = \frac{1}{1 + \exp(-(A \cdot s_i + B))},
  \label{eq:platt_scaling}
\end{equation}
where scalar parameters $A$ and $B$ are estimated via maximum likelihood on a held-out validation dataset $\mathcal{D}_{\text{val}}$:
\begin{equation}
  \arg\min_{A, B} - \sum_{j \in \mathcal{D}_{\text{val}}} \Big[ y_j \log \hat{p}_j^{\text{Platt}} + (1 - y_j) \log(1 - \hat{p}_j^{\text{Platt}}) \Big].
  \label{eq:platt_mle}
\end{equation}
\textit{Limitations:} Conventional Platt scaling assumes that validation and test sets share identical outcome prevalence and logit distributions. When deployed across a multi-year temporal gap, secular declines in baseline child mortality violate this assumption, causing standard Platt scaling to suffer from calibration drift (large non-zero calibration intercepts $\alpha \ll 0$). Moreover, under severe class imbalance, unconstrained optimization of $(A, B)$ frequently collapses predicted probabilities toward the training prevalence mean, flattening risk gradients.

\subsection{Small-Area Estimation and Regional Shrinkage}
\label{sec:conv_shrinkage}

In national health surveys, administrative divisions exhibit substantial disparities in sample sizes and death event counts. In small divisions with sparse event counts (e.g., Khulna with 34 deaths in 2022), sample event rates $\hat{\mu}_d = \frac{n_{d, \text{pos}}}{N_d}$ suffer from high sampling variance, whereas national estimates $\bar{\mu}$ possess low variance but mask regional heterogeneity. 

Classical small-area estimation resolves this variance-bias tradeoff using James-Stein or empirical Bayes shrinkage estimators \cite{casella1985,efron2012}:
\begin{equation}
  \hat{\theta}_d^{\text{EB}} = \kappa_d \hat{\mu}_d + (1 - \kappa_d) \bar{\mu},
  \label{eq:eb_shrinkage}
\end{equation}
where the shrinkage weight $\kappa_d \in [0, 1]$ is determined by the ratio of between-division variance $\tau^2$ to within-division sampling variance $\sigma_d^2 / N_d$:
\begin{equation}
  \kappa_d = \frac{\tau^2}{\tau^2 + \frac{\sigma_d^2}{N_d}}.
  \label{eq:shrinkage_intensity}
\end{equation}
\textit{Limitations:} Conventional empirical Bayes estimators operate strictly as univariate linear post-processing tools on observed prevalence counts. They are not integrated into multi-stage machine learning architectures to regularize tree ensemble decision margins across regional administrative strata.

\subsection{Limitations and Research Gap Motivating SWIFT-U5}
\label{sec:conv_gap}

In summary, conventional methods suffer from four unresolved structural limitations when applied to demographic child mortality prediction:
\begin{enumerate}
    \item Standard machine learning models optimize sample-frequency loss functions without survey design corrections, leading to biased parameter estimates;
    \item Class imbalance rebalancing distorts model logits by a constant offset, generating severely miscalibrated predicted probabilities;
    \item Density-ratio covariate adaptation suffers from weight instability and lacks mechanisms to incorporate temporal recency decay across decadal survey rounds;
    \item Conventional calibration and shrinkage methods operate in isolation rather than as an integrated, drift-aware prospective learning pipeline.
\end{enumerate}
These limitations establish the specific technical research gap that motivates the design and mathematical formulation of the SWIFT-U5 framework.

\section{Proposed Method — SWIFT-U5}
\label{sec:proposed_method}

To address the limitations of conventional methodologies, we propose \textbf{SWIFT-U5} (\textbf{S}urvey-\textbf{W}eighted, \textbf{I}mportance-corrected, \textbf{F}orward-validated \textbf{T}emporal Risk Model for Under-Five Mortality). This section details the working principles, mathematical formulations, algorithmic workflow, and distinctive characteristics of the proposed framework.

\subsection{Working Principle}
\label{sec:swift_working_principle}

The SWIFT-U5 architecture is organized as an end-to-end prospective machine learning pipeline that transforms raw multi-wave DHS microdata into calibrated, division-adjusted risk scores and operational outreach triage lists. As illustrated in Figure~\ref{fig:workflow}, the operational workflow comprises four sequential processing stages:
\begin{enumerate}
    \item \textbf{Input Space and Predictor Auditing:} Predictors are strictly restricted to durable pre-birth maternal reproductive characteristics, clinical antenatal care, and household physical infrastructure, eliminating all survival-contingent proxies. Continuous features are median-imputed and standardized using statistics computed exclusively on training folds; categorical variables are one-hot encoded with explicit missingness indicators.
    \item \textbf{Module A (Transductive Covariate-Shift Density Weighting with Recency Decay):} For all historical training records, raw DHS sampling weights are adjusted by combining an exponential temporal recency decay function with a transductively estimated domain density ratio, producing composite instance weights $w_i^*$.
    \item \textbf{Survey-Weighted Base Model Training:} A gradient tree boosting ensemble (XGBoost) is trained by minimizing survey-weighted empirical risk under instance weights $w_i^*$, capturing non-linear feature interactions under complex design constraints.
    \item \textbf{Module B (Empirical Bayes Division Shrinkage Head):} Raw out-of-fold logit predictions from the base model are regularized across administrative divisions, shrinking noisy regional calibration parameters toward national empirical averages.
    \item \textbf{Module C (Drift-Aware Platt Recalibration):} The division-shrunk logits are mapped to true population probabilities via a parametric Platt scaling transformation fitted on held-out validation predictions from an independent penultimate survey wave.
    \item \textbf{Operational Decision and Prioritization Layer:} Calibrated posterior probabilities are evaluated against a validation-locked decision threshold ($\tau_w$) and ranked to simulate capacity-constrained community health outreach triage.
\end{enumerate}

\casinlinefig[\linewidth]{workflow_diagram.png}{System architecture of the SWIFT-U5 framework: Survey-weighted temporal validation, dual-method consensus feature discovery, Platt recalibration, and composite risk stratification across BDHS waves.}{fig:workflow}

\subsection{Technical Formulation}
\label{sec:swift_formulation}

\subsubsection{Module A: Recency-Decayed Covariate Shift Adaptation}
\label{sec:form_module_a}
To reconcile multi-decade historical training microdata with contemporary target populations, Module~A formulates a composite instance weight $w_i^*$ for each historical live birth record $i \in \mathcal{D}_{\text{train}}$. The formulation integrates three distinct factors: baseline DHS survey sampling weights, an exponential temporal recency discount, and a bounded transductive covariate density ratio.

Let $t_i \in \{2004, 2007, 2011, 2014\}$ denote the survey round of training record $i$, and let $t_{\text{target}} = 2022$ denote the prospective target survey round. The exponential temporal recency weight $\gamma(t_i)$ is defined as:
\begin{equation}
  \gamma(t_i) = \exp\Big( -\lambda (t_{\text{target}} - t_i) \Big),
  \label{eq:recency_decay}
\end{equation}
where $\lambda > 0$ is the temporal decay parameter. Setting $\lambda = 0.05$ establishes an effective half-life of $t_{1/2} = \frac{\ln(2)}{\lambda} \approx 13.9$ years, appropriately down-weighting distant survey rounds (e.g., 2004 records receive weight $\exp(-0.05 \times 18) \approx 0.407$) while preserving recent historical evidence.

To model covariate shift $P_{\text{train}}(\mathbf{x}) \neq P_{\text{target}}(\mathbf{x})$, we construct an unlabeled transductive domain dataset:
\begin{equation}
  \mathcal{D}_{\text{domain}} = \Big\{ (\mathbf{x}_i, S_i = 0) \Big\}_{i \in \mathcal{D}_{\text{train}}} \cup \Big\{ (\mathbf{x}_j, S_j = 1) \Big\}_{j \in \mathcal{D}_{\text{target}}},
  \label{eq:domain_data}
\end{equation}
where $S \in \{0, 1\}$ indicates domain membership, and $\mathcal{D}_{\text{target}}$ contains strictly unlabeled feature vectors from the 2022 survey cohort without mortality target labels ($Y_{2022}$). A regularized logistic domain classifier $g(\mathbf{x}) = \hat{P}(S=1 \mid \mathbf{x})$ is trained by minimizing $L_2$-penalized cross-entropy:
\begin{equation}
  \min_{\mathbf{w}_g, b_g} - \sum_{k \in \mathcal{D}_{\text{domain}}} \Big[ S_k \log \sigma(\mathbf{w}_g^T \mathbf{x}_k + b_g) + (1 - S_k) \log(1 - \sigma(\mathbf{w}_g^T \mathbf{x}_k + b_g)) \Big] + \frac{1}{2} C_g \|\mathbf{w}_g\|_2^2.
  \label{eq:domain_mle}
\end{equation}
The raw density ratio is computed via Bayes' theorem:
\begin{equation}
  r(\mathbf{x}_i) = \frac{g(\mathbf{x}_i)}{1 - g(\mathbf{x}_i)} \cdot \frac{N_{\text{train}}}{N_{\text{target}}}.
  \label{eq:raw_density}
\end{equation}
To prevent weight instability and variance explosion from extreme feature outliers, $r(\mathbf{x}_i)$ is bounded within a robust clipping interval $[r_{\min}, r_{\max}] = [0.1, 10.0]$:
\begin{equation}
  r_{\text{clipped}}(\mathbf{x}_i) = \min\Big( \max(r(\mathbf{x}_i), \, 0.1), \, 10.0 \Big).
  \label{eq:density_clip}
\end{equation}
The composite instance weight $w_i^*$ combines the normalized DHS sample weight $w_i^{\text{survey}}$, the recency discount $\gamma(t_i)$, and the clipped density ratio:
\begin{equation}
  w_i^* = w_i^{\text{survey}} \cdot \exp\Big( -\lambda (t_{\text{target}} - t_i) \Big) \cdot r_{\text{clipped}}(\mathbf{x}_i).
  \label{eq:composite_weight}
\end{equation}
Finally, composite weights are normalized across the historical training cohort to sum to the empirical development sample size $N_{\text{dev}} = 29{,}686$:
\begin{equation}
  \tilde{w}_i^* = w_i^* \times \frac{N_{\text{dev}}}{\sum_{j \in \mathcal{D}_{\text{dev}}} w_j^*}, \quad \text{such that } \sum_{i \in \mathcal{D}_{\text{dev}}} \tilde{w}_i^* = N_{\text{dev}}.
  \label{eq:norm_composite_weight}
\end{equation}

\subsubsection{Survey-Weighted Base Model Formulation}
\label{sec:form_base_model}
The base learner is a gradient boosted decision tree ensemble $f_{\text{base}}(\mathbf{x}; \boldsymbol{\theta})$ trained on the development partition $\mathcal{D}_{\text{dev}}$ using the composite instance weights $\tilde{w}_i^*$. The objective function minimizes survey-weighted empirical cross-entropy:
\begin{equation}
  \mathcal{L}_{\text{SWIFT}}(\boldsymbol{\theta}) = - \sum_{i \in \mathcal{D}_{\text{dev}}} \tilde{w}_i^* \Big[ y_i \log \sigma(f(\mathbf{x}_i; \boldsymbol{\theta})) + (1 - y_i) \log(1 - \sigma(f(\mathbf{x}_i; \boldsymbol{\theta}))) \Big] + \sum_{k=1}^K \Omega(f_k).
  \label{eq:swift_objective}
\end{equation}
Because instance weights $\tilde{w}_i^*$ directly embed target population representation and recency importance, the model objective optimizes population risk rather than unweighted sample classification.

\subsubsection{Module B: Empirical Bayes Division Shrinkage Head}
\label{sec:form_module_b}
Bangladesh's administrative geography comprises eight divisions exhibiting substantial heterogeneity in baseline under-five mortality (e.g., Sylhet at $44.3$‰ vs. Khulna at $21.7$‰ in 2022). Fitting unconstrained division-specific calibration intercepts introduces high variance in smaller divisions. Module~B formulates an empirical Bayes shrinkage head that regularizes division-level risk logit offsets toward the national mean.

Let $z_i = \text{logit}(f_{\text{base}}(\mathbf{x}_i)) \in \mathbb{R}$ denote the base model margin score for child $i$, and let $d(i) \in \{1, \dots, D\}$ denote the child's administrative division ($D = 7$ in harmonized longitudinal models; $D = 8$ in contemporary models). For each division $d$, we define a localized logistic calibration model:
\begin{equation}
  \text{logit}\big(P(Y = 1 \mid z_i, d(i) = d)\big) = \alpha_d + \beta_d z_i.
  \label{eq:div_cal_model}
\end{equation}
Unconstrained maximum likelihood estimates $(\tilde{\alpha}_d, \tilde{\beta}_d)$ are computed on validation predictions. Under an empirical Bayes Gaussian prior $\alpha_d \sim \mathcal{N}(\bar{\alpha}_{\text{national}}, \tau_\alpha^2)$ and $\beta_d \sim \mathcal{N}(\bar{\beta}_{\text{national}}, \tau_\beta^2)$, the posterior mean shrinkage estimator is given by:
\begin{equation}
  \hat{\alpha}_d = \frac{n_d}{n_d + \kappa} \tilde{\alpha}_d + \frac{\kappa}{n_d + \kappa} \bar{\alpha}_{\text{national}}, \quad
  \hat{\beta}_d = \frac{n_d}{n_d + \kappa} \tilde{\beta}_d + \frac{\kappa}{n_d + \kappa} \bar{\beta}_{\text{national}},
  \label{eq:eb_module_b}
\end{equation}
where $n_d = \sum_{i \in \mathcal{D}_{\text{val}}, d(i)=d} y_i$ is the observed mortality event count in division $d$, and $\kappa > 0$ is the empirical Bayes shrinkage constant, fixed at $\kappa = 40.0$ based on held-out validation likelihood. For high-burden divisions (e.g., Sylhet, $n_d = 62$ in validation), $\frac{n_d}{n_d + \kappa} \approx 0.608$, preserving regional signal; for low-burden divisions (e.g., Khulna, $n_d = 32$), $\frac{n_d}{n_d + \kappa} \approx 0.444$, shrinking estimates toward the national baseline.

The division-shrunk logit score $z_i^{(B)}$ is then computed as:
\begin{equation}
  z_i^{(B)} = \hat{\alpha}_{d(i)} + \hat{\beta}_{d(i)} \cdot z_i.
  \label{eq:shrunk_logit}
\end{equation}

\subsubsection{Module C: Drift-Aware Platt Recalibration}
\label{sec:form_module_c}
The division-adjusted logit $z_i^{(B)}$ remains unanchored to the absolute prevalence scale of prospective cohorts. Module~C applies a parametric Platt scaling transformation fitted on out-of-fold predictions from the independent BDHS 2017--18 validation split ($N_{\text{val}} = 8{,}684$, $356$ deaths).

The calibrated posterior mortality probability $\hat{p}_i^*$ is formulated as:
\begin{equation}
  \hat{p}_i^* = \sigma\big( a^* \cdot z_i^{(B)} + b^* \big) = \frac{1}{1 + \exp\Big( -(a^* \cdot z_i^{(B)} + b^*) \Big)},
  \label{eq:module_c_platt}
\end{equation}
where scalar calibration parameters $(a^*, b^*)$ are determined by minimizing survey-weighted negative log-likelihood over the validation set:
\begin{equation}
  (a^*, b^*) = \arg\min_{a, b} - \sum_{j \in \mathcal{D}_{\text{val}}} w_j^{\text{val}} \Big[ y_j \log \sigma(a \cdot z_j^{(B)} + b) + (1 - y_j) \log(1 - \sigma(a \cdot z_j^{(B)} + b)) \Big].
  \label{eq:module_c_opt}
\end{equation}
Optimization is performed using L-BFGS-B with bound constraints ($a \in [0.1, 5.0]$, $b \in [-10.0, 5.0]$). Because the 2017--18 validation split serves as an independent temporal anchor immediately preceding the 2022 test round, $(a^*, b^*)$ explicitly absorbs secular drift in baseline logits without modifying the underlying ranking order.

\subsubsection{Operational Threshold and Prioritization Layer}
\label{sec:form_decision_layer}
To translate continuous calibrated probabilities $\hat{p}_i^*$ into public health decision rules, SWIFT-U5 establishes two operational regimes:
\begin{enumerate}
    \item \textbf{Validation-Locked Youden Operating Point ($\tau_w$):} The decision threshold is selected by maximizing Youden's $J$ statistic on out-of-fold validation predictions ($J(\tau) = \text{Sensitivity}_w(\tau) + \text{Specificity}_w(\tau) - 1$), yielding a locked threshold of $\tau_w = 0.04546$. A child is classified as high-risk if $\hat{p}_i^* \ge \tau_w$.
    \item \textbf{Capacity-Constrained Outreach Prioritization (Top $K\%$):} In community outreach triage, public health resources restrict visitation to a fixed population proportion (e.g., top $K = 10\%$ highest-risk records, corresponding to $N_{\text{prioritized}} = \lfloor 0.10 \times N \rfloor = 1{,}312$ records out of $13{,}119$). Children are ranked in descending order of $\hat{p}_i^*$, selecting the top $N_{\text{prioritized}}$ instances for simulated intervention.
\end{enumerate}

\subsection{Algorithm and Computational Procedure}
\label{sec:swift_algorithm}

Algorithm~\ref{alg:swift} formalizes the complete end-to-end computational procedure for training, calibrating, and prospectively evaluating SWIFT-U5.

\begin{algorithm}[H]
\caption{Complete SWIFT-U5 Training, Calibration, and Prospective Evaluation Workflow}
\label{alg:swift}
\begin{algorithmic}[1]
\Require Historical development cohort $\mathcal{D}_{\text{dev}} = \{(\mathbf{x}_i, y_i, w_i, t_i, d_i)\}_{i=1}^{N_{\text{dev}}}$ (BDHS 2004--2014)
\Require Independent validation cohort $\mathcal{D}_{\text{val}} = \{(\mathbf{x}_j, y_j, w_j, t_j, d_j)\}_{j=1}^{N_{\text{val}}}$ (BDHS 2017--18)
\Require Unlabeled prospective target cohort $\mathcal{D}_{\text{test}} = \{(\mathbf{x}_k, w_k, t_{\text{target}}, d_k)\}_{k=1}^{N_{\text{test}}}$ (BDHS 2022)
\Require Hyperparameters: recency decay $\lambda = 0.05$, shrinkage constant $\kappa = 40.0$, clipping $[0.1, 10.0]$
\Ensure Calibrated prospective child risk probabilities $\hat{\mathbf{p}}_{\text{test}}^* \in [0, 1]^{N_{\text{test}}}$, binary triage decisions $\hat{\mathbf{y}}_{\text{test}}$, and prioritized cohort $\mathcal{S}_{\text{top10\%}}$

\Statex \textbf{// Phase 1: Preprocessing and Stable Feature Consensus}
\State Fit imputation scalers $T_{\text{prep}}$ on $\mathcal{D}_{\text{dev}}$ only; transform $\mathcal{D}_{\text{dev}}$, $\mathcal{D}_{\text{val}}$, and $\mathcal{D}_{\text{test}}$
\State Run dual-method PSU stability consensus selection on $\mathcal{D}_{\text{dev}}$ across 40 bootstrap iterations to identify stable feature set $\mathcal{F}^*$

\Statex \textbf{// Phase 2: Module A Transductive Covariate Shift Weighting}
\State Construct domain dataset $\mathcal{D}_{\text{domain}} = \{(\mathbf{x}_i[\mathcal{F}^*], 0)\}_{i \in \mathcal{D}_{\text{dev}}} \cup \{(\mathbf{x}_k[\mathcal{F}^*], 1)\}_{k \in \mathcal{D}_{\text{test}}}$
\State Train logistic domain classifier $g(\mathbf{x}) \gets \hat{P}(S=1 \mid \mathbf{x})$ on $\mathcal{D}_{\text{domain}}$ via Eq.~\eqref{eq:domain_mle}
\For{each training instance $i \in \mathcal{D}_{\text{dev}}$}
    \State Compute density ratio $r(\mathbf{x}_i) \gets \frac{g(\mathbf{x}_i)}{1 - g(\mathbf{x}_i)} \cdot \frac{N_{\text{dev}}}{N_{\text{test}}}$
    \State Clip density ratio: $r_i \gets \min(\max(r(\mathbf{x}_i), 0.1), 10.0)$
    \State Compute composite weight: $w_i^* \gets w_i \cdot \exp(-\lambda(t_{\text{target}} - t_i)) \cdot r_i$
\EndFor
\State Normalize composite weights: $\tilde{w}_i^* \gets w_i^* \times \frac{N_{\text{dev}}}{\sum_{j} w_j^*}$

\Statex \textbf{// Phase 3: Base Model Training and Validation Extraction}
\State Train base gradient booster $f_{\text{base}}(\mathbf{x}; \boldsymbol{\theta})$ on $\mathcal{D}_{\text{dev}}[\mathcal{F}^*]$ with sample weights $\tilde{w}_i^*$ via Eq.~\eqref{eq:swift_objective}
\State Generate validation logits: $z_j \gets \text{logit}(f_{\text{base}}(\mathbf{x}_j[\mathcal{F}^*]))$ for all $j \in \mathcal{D}_{\text{val}}$

\Statex \textbf{// Phase 4: Module B Empirical Bayes Division Shrinkage}
\State Compute national validation logit calibration parameters $(\bar{\alpha}, \bar{\beta})$ on $\mathcal{D}_{\text{val}}$
\For{each administrative division $d \in \{1, \dots, D\}$}
    \State Fit unconstrained division calibration parameters $(\tilde{\alpha}_d, \tilde{\beta}_d)$ on $\mathcal{D}_{\text{val}}$
    \State Shrink parameters: $\hat{\alpha}_d \gets \frac{n_d}{n_d + \kappa} \tilde{\alpha}_d + \frac{\kappa}{n_d + \kappa} \bar{\alpha}$, \quad $\hat{\beta}_d \gets \frac{n_d}{n_d + \kappa} \tilde{\beta}_d + \frac{\kappa}{n_d + \kappa} \bar{\beta}$
\EndFor
\State Compute division-shrunk validation logits: $z_j^{(B)} \gets \hat{\alpha}_{d(j)} + \hat{\beta}_{d(j)} \cdot z_j$ for all $j \in \mathcal{D}_{\text{val}}$

\Statex \textbf{// Phase 5: Module C Drift-Aware Platt Recalibration and Threshold Locking}
\State Fit calibration parameters $(a^*, b^*)$ on $(z_j^{(B)}, y_j, w_j^{\text{val}})$ via Eq.~\eqref{eq:module_c_opt}
\State Compute calibrated validation probabilities: $\hat{p}_j^* \gets \sigma(a^* \cdot z_j^{(B)} + b^*)$
\State Lock Youden decision threshold: $\tau_w \gets \arg\max_{\tau} \big( \text{Sensitivity}_w(\tau) + \text{Specificity}_w(\tau) - 1 \big)$ on $\mathcal{D}_{\text{val}}$

\Statex \textbf{// Phase 6: Prospective Inference on Locked 2022 Test Cohort}
\State Compute base test logits: $z_k \gets \text{logit}(f_{\text{base}}(\mathbf{x}_k[\mathcal{F}^*]))$ for all $k \in \mathcal{D}_{\text{test}}$
\State Apply Module B shrinkage: $z_k^{(B)} \gets \hat{\alpha}_{d(k)} + \hat{\beta}_{d(k)} \cdot z_k$
\State Apply Module C recalibration: $\hat{p}_k^* \gets \sigma(a^* \cdot z_k^{(B)} + b^*)$
\State Assign binary threshold decisions: $\hat{y}_k \gets \mathbb{I}(\hat{p}_k^* \ge \tau_w)$
\State Identify top 10\% prioritization cohort: $\mathcal{S}_{\text{top10\%}} \gets \text{argsort}_{\text{desc}}(\hat{\mathbf{p}}_{\text{test}}^*)[1 : \lfloor 0.10 \times N_{\text{test}} \rfloor]$
\State \Return Calibrated probabilities $\hat{\mathbf{p}}_{\text{test}}^*$, binary decisions $\hat{\mathbf{y}}_{\text{test}}$, and prioritized cohort $\mathcal{S}_{\text{top10\%}}$
\end{algorithmic}
\end{algorithm}

\subsection{Main Contributions and Characteristics of the Proposed Method}
\label{sec:swift_contributions}

The SWIFT-U5 framework establishes several distinct technical characteristics that distinguish it from existing machine learning approaches in public health:
\begin{enumerate}
    \item \textbf{Leakage-Free Feature Architecture:} By establishing that retrospective illness recall and breastfeeding duration act as survival-contingent proxies, SWIFT-U5 grounds its feature space exclusively on durable maternal reproductive indicators and household physical assets known prior to birth, ensuring true prospective validity.
    \item \textbf{Transductive Covariate Adaptation without Label Peeking:} Module~A leverages unlabeled target covariates ($X_{2022}$) to correct for population distribution drift across survey waves, strictly isolating target mortality outcome labels ($Y_{2022}$) from the development lifecycle.
    \item \textbf{Synergistic Calibration and Regularization:} Rather than viewing calibration as an isolated post-processing step, SWIFT-U5 couples empirical Bayes regional shrinkage (Module~B) with drift-aware Platt scaling (Module~C), directly resolving the logit distortion introduced by class-imbalance boosting.
    \item \textbf{Translational Decision-Analytic Utility:} By coupling probability calibration with decision-curve analysis and capacity-constrained outreach metrics (NNS and yield per 1,000 prioritized), SWIFT-U5 bridges the gap between abstract statistical ranking and concrete community healthcare worker triage operations.
\end{enumerate}

\section{Experiments}
\label{sec:experiments}

This section details the experimental environment, dataset characteristics, cohort partitions, experimental procedures, hyperparameter configurations, and evaluation metrics used to evaluate the SWIFT-U5 framework.

\subsection{Experimental Setup and Conditions}
\label{sec:exp_setup}

All computational experiments were developed and executed in a containerized Unix environment (macOS ARM64 / Darwin 24.1.0) using Python 3.9.6. Core computational libraries included NumPy 1.24.3, Pandas 2.0.3, SciPy 1.11.1, Scikit-learn 1.3.0, XGBoost 1.7.6, LightGBM 4.1.0, and SHAP 0.42.1. 

To guarantee complete computational reproducibility, all pseudorandom number generators (including NumPy, Python random, and scikit-learn splitters) were initialized with a global fixed random seed of $42$. Gradient tree boosting algorithms were configured with deterministic single-threaded histogram building options where applicable. All pre-processing transformers, imputation statistics, and feature scalers were fitted exclusively on development folds and applied unchanged to subsequent validation and test sets.

\subsection{Dataset Introduction}
\label{sec:exp_dataset}

The experimental corpus was constructed from individual Birth Recode (BR) microdata files of the Bangladesh Demographic and Health Survey across all six sequential survey waves conducted between 2004 and 2022: BDHS 2004, BDHS 2007, BDHS 2011, BDHS 2014, BDHS 2017--18, and BDHS 2022 \cite{niport2024}. 

\paragraph{Inclusion and Exclusion Criteria}
The analytical cohort includes all live births born to interviewed women aged 15--49 years within the five-year retrospective recall window preceding the survey interview date ($0 \le \texttt{b3} - \texttt{v008} \le 59$ months). Stillbirths and miscarriages were excluded in accordance with standard DHS birth recode protocols. Records with missing survival status or incomplete sampling weights were audited; across all six survey waves, exposure completeness exceeded 99.8\%, yielding an aggregate master dataset of $N = \mathbf{51{,}489}$ live births and $\mathbf{2{,}338}$ documented under-five child deaths (Table~\ref{tab:cohort_summary}).

\begin{table}[pos=htbp]
  \centering
  \caption{Longitudinal characteristics of the harmonized BDHS under-five mortality cohort across six survey rounds (2004--2022) and administrative divisions.}
  \label{tab:cohort_summary}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\centering\arraybackslash}p{0.14\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}X@{}}
    \toprule
    Survey Wave & Administrative Division & Sample ($N$) & Deaths ($n$) & Weighted U5MR (per 1,000) \\
    \midrule
    \textbf{BDHS 2004} & National (Total) & 6{,}908 & 484 & 71.9 \\
    & Barishal & 752 & 43 & 57.9 \\
    & Chattogram & 1{,}503 & 103 & 67.7 \\
    & Dhaka & 1{,}531 & 122 & 78.2 \\
    & Khulna & 866 & 41 & 50.2 \\
    & Rajshahi & 1{,}312 & 89 & 70.8 \\
    & Sylhet & 944 & 86 & 99.2 \\
    \midrule
    \textbf{BDHS 2007} & National (Total) & 6{,}150 & 361 & 55.9 \\
    & Barishal & 791 & 42 & 54.7 \\
    & Chattogram & 1{,}275 & 74 & 62.2 \\
    & Dhaka & 1{,}285 & 60 & 47.7 \\
    & Khulna & 714 & 33 & 45.2 \\
    & Rajshahi & 967 & 50 & 52.9 \\
    & Sylhet & 1{,}118 & 102 & 88.6 \\
    \midrule
    \textbf{BDHS 2011} & National (Total) & 8{,}742 & 410 & 44.8 \\
    & Barishal & 976 & 52 & 55.1 \\
    & Chattogram & 1{,}749 & 66 & 35.5 \\
    & Dhaka & 1{,}442 & 66 & 46.1 \\
    & Khulna & 981 & 35 & 36.8 \\
    & Rajshahi & 995 & 46 & 45.4 \\
    & Rangpur & 1{,}255 & 61 & 48.0 \\
    & Sylhet & 1{,}344 & 84 & 60.8 \\
    \midrule
    \textbf{BDHS 2014} & National (Total) & 7{,}886 & 319 & 40.4 \\
    & Barishal & 900 & 46 & 49.3 \\
    & Chattogram & 1{,}506 & 70 & 46.5 \\
    & Dhaka & 1{,}498 & 53 & 36.6 \\
    & Khulna & 992 & 35 & 35.8 \\
    & Rajshahi & 981 & 27 & 28.5 \\
    & Rangpur & 896 & 28 & 31.7 \\
    & Sylhet & 1{,}113 & 60 & 52.9 \\
    \midrule
    \textbf{BDHS 2017--18} & National (Total) & 8{,}684 & 356 & 39.9 \\
    & Barishal & 907 & 45 & 50.8 \\
    & Chattogram & 1{,}445 & 60 & 41.5 \\
    & Dhaka (incl.\ Mymensingh) & 2{,}335 & 99 & 38.6 \\
    & Khulna & 857 & 32 & 38.2 \\
    & Rajshahi & 964 & 27 & 29.5 \\
    & Rangpur & 869 & 31 & 35.3 \\
    & Sylhet & 1{,}307 & 62 & 46.7 \\
    \midrule
    \textbf{BDHS 2022} & National (Total) & 13{,}119 & 408 & 29.9 \\
    & Barishal & 1{,}464 & 50 & 33.0 \\
    & Chattogram & 2{,}219 & 69 & 30.9 \\
    & Dhaka (incl.\ Mymensingh) & 3{,}558 & 98 & 27.7 \\
    & Khulna & 1{,}448 & 34 & 21.7 \\
    & Rajshahi & 1{,}309 & 32 & 27.1 \\
    & Rangpur & 1{,}505 & 54 & 38.1 \\
    & Sylhet & 1{,}616 & 71 & 44.3 \\
    \bottomrule
  \end{tabularx}
\end{table}

\paragraph{Outcome Definition and Right-Censoring Qualification}
The analytical estimand is defined as: \textit{Retrospective risk modeling of under-five mortality among survey-reported live births born within the five-year recall window, where survival status is ascertained at maternal interview}. Formally:
\begin{equation}
  \text{target} = (\texttt{b5} = 0) \land (\texttt{b7} < 60),
  \label{eq:target_def}
\end{equation}
where \texttt{b5} indicates survival status ($0 = \text{deceased}$, $1 = \text{alive}$) and \texttt{b7} is recorded age at death in completed months. 

We explicitly articulate the epidemiologic character of this outcome: in retrospective demographic surveys, live births occurring within five years prior to the interview date have variable lengths of follow-up. Children alive at interview who are younger than 60 months have not completed the full five-year exposure window. Under the binary classification convention standard across the DHS literature \cite{oduse2024,naznin2025,bitew2020}, living children are categorized as non-events at interview time ($Y = 0$). In Bangladesh, under-five child mortality is heavily front-loaded in early life: according to national vital registration statistics \cite{niport2024}, over 71\% of all under-five deaths occur during the neonatal period ($t < 1$ month), and over 85\% occur within the first year of life ($t < 12$ months). For a child who has survived past their first birthday to the survey interview, the residual conditional probability of dying before age five is under 0.4\%. Consequently, the degree of label misclassification among living children is modest. Nonetheless, living children under 60 months are right-censored in the strict epidemiologic sense; they must not be characterized as confirmed five-year survivors.

\paragraph{Audited Predictor Feature Space}
Table~\ref{tab:predictor_rationale} documents the measurement timing, DHS source variable, reference period, and temporal stability rationale for all candidate predictors. To eliminate survival leakage, all retrospective child illness recall variables (\texttt{h11} diarrhea, \texttt{h22} fever, \texttt{h31} cough) and postnatal feeding duration (\texttt{m4}) were strictly excluded. Household assets (wealth quintile, toilet facility, water source, flooring material, electricity) were ascertained at survey interview; for children who died prior to the interview, these indicators serve as durable structural background proxies for household socioeconomic vulnerability.

\begin{table}[pos=htbp]
  \centering
  \caption{Predictor measurement timing, DHS source, reference period, and availability rationale.}
  \label{tab:predictor_rationale}
  \small
  \setlength{\tabcolsep}{3pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}X@{}}
    \toprule
    Predictor Domain & Variables & DHS Source & At Birth? & At Survey? & Temporal Stability / Rationale \\
    \midrule
    Maternal Core & \texttt{maternal\_age\_at\_birth} & \texttt{v011}, \texttt{b3} & Yes & Yes & Derived exactly from maternal birth date (\texttt{v011}) and child birth date (\texttt{b3}). \\
    Reproductive History & \texttt{birth\_order}, \texttt{parity\_before}, \texttt{is\_first\_birth} & \texttt{bord}, \texttt{v201} & Yes & Yes & Parity and birth sequence established prior to index birth. \\
    Birth Spacing & \texttt{preceding\_birth\_interval} & \texttt{b11} & Yes & Yes & Interval in completed months preceding index birth; first births encoded via explicit indicator. \\
    Multiplicity & \texttt{is\_twin} & \texttt{b0} & Yes & Yes & Plurality indicator available at birth. \\
    Antenatal Care & \texttt{anc\_visits}, \texttt{tetanus\_toxoid}, \texttt{iron\_supp} & \texttt{m14}, \texttt{m1}, \texttt{m45} & Yes & Yes & Clinical care during index pregnancy, recalled at interview. Missingness encoded explicitly. \\
    Birth Characteristics & \texttt{birth\_size}, \texttt{facility\_delivery}, \texttt{caesarean} & \texttt{m18}, \texttt{m15}, \texttt{m17} & Yes & Yes & Mother-recalled subjective birth size proxy for birth weight; delivery location and mode. \\
    Maternal Education & \texttt{maternal\_education} & \texttt{v106} & Invariant & Yes & Completed educational attainment; adult schooling is highly stable post-childbirth in rural Bangladesh. \\
    Household Wealth & \texttt{wealth\_quintile} & \texttt{v190} & Proxy & Yes & Asset index measured at interview; durable structural proxy for socioeconomic status. \\
    WASH Infrastructure & \texttt{water\_source}, \texttt{toilet\_type}, \texttt{electricity}, \texttt{floor} & \texttt{v113}, \texttt{v116}, \texttt{v119}, \texttt{v127} & Proxy & Yes & Household physical infrastructure; durable assets with zero skip-pattern missingness. \\
    Geographic & \texttt{division}, \texttt{residence} & \texttt{v024}, \texttt{v025} & Yes & Yes & Administrative division and urban/rural classification established at survey sampling. \\
    \bottomrule
  \end{tabularx}
\end{table}

\paragraph{Prospective Cohort Partitions}
The dataset was partitioned into three mutually exclusive, chronologically ordered partitions:
\begin{enumerate}
    \item \textbf{Historical Development Partition ($df\_train$):} BDHS 2004, 2007, 2011, and 2014 ($N = 29{,}686$, deaths = $1{,}574$). Used for feature selection, baseline model training, and density-ratio estimation.
    \item \textbf{Independent Validation Anchor ($df\_val$):} BDHS 2017--18 ($N = 8{,}684$, deaths = $356$). Strictly held out from development training; used exclusively for Module~B division shrinkage estimation, Module~C drift recalibration, and Youden threshold optimization.
    \item \textbf{Locked Prospective Out-of-Time Test Cohort ($df\_test$):} BDHS 2022 ($N = 13{,}119$, deaths = $408$). Evaluated exactly once at study conclusion.
\end{enumerate}
The partitions are completely exhaustive: $29{,}686 + 8{,}684 + 13{,}119 = \mathbf{51{,}489}$ children; deaths: $1{,}574 + 356 + 408 = \mathbf{2{,}338}$ deaths. We declare that BDHS 2022 test mortality labels were quarantined throughout all model training, hyperparameter tuning, shrinkage estimation, calibration fitting, and threshold selection. Module~A incorporated unlabeled 2022 covariates within a transductive domain adaptation framework, accessing strictly feature vectors without target mortality labels.

\subsection{Experimental Procedure}
\label{sec:exp_procedure}

The experimental evaluation was structured around two distinct operational tracks and a controlled factorial benchmark:
\begin{enumerate}
    \item \textbf{Track A (Prospective Forward Temporal Benchmark):} Models are trained on historical development data (2004--2014), calibrated on the 2017--18 validation anchor, and prospectively evaluated on the locked 2022 test cohort. This represents the primary prospective deployment benchmark of SWIFT-U5.
    \item \textbf{Track B (Retrospective In-Distribution Random Diagnostic):} For each forward cut, matched-N random splits are drawn by pooling contemporary survey records, randomly allocating live births into training (70\%) and testing (30\%) splits. Track B serves strictly as a methodological diagnostic to measure design inflation.
    \item \textbf{Controlled $2 \times 2$ Factorial Benchmark:} Evaluated across five sequential forward temporal cuts (Table~\ref{tab:temporal_cuts}), crossing Factor 1 (Split Strategy: Random vs. Temporal) with Factor 2 (Training Weights: Unweighted vs. Survey/Module A Weighted).
    \item \textbf{Stepwise Component Ablation:} The full SWIFT-U5 pipeline is systematically ablated by removing Module~A (no shift/recency weights), Module~B (no division shrinkage), Module~C (no drift recalibration), and all modules simultaneously (base model).
    \item \textbf{Design-Based Cluster Bootstrapping:} All 95\% confidence intervals were generated via 1,000 cluster bootstrap iterations, resampling primary sampling units (PSUs, \texttt{v021}) with replacement within survey strata (\texttt{v022}).
\end{enumerate}

\begin{table}[pos=htbp]
  \centering
  \caption{Definitive Forward-Temporal Validation Partitions: Training cohorts, evaluation test waves, sample sizes, and death counts across all five sequential temporal cuts.}
  \label{tab:temporal_cuts}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabularx}{\linewidth}{@{}>{\centering\arraybackslash}p{0.08\linewidth}>{\raggedright\arraybackslash}p{0.34\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\centering\arraybackslash}p{0.14\linewidth}>{\centering\arraybackslash}X@{}}
    \toprule
    Cut & Training Waves & Evaluation Test Wave & Test Sample ($N$) & Test Deaths ($n$) \\
    \midrule
    Cut 1 & BDHS 2004 ($N = 6{,}908$) & BDHS 2007 & 6{,}150 & 361 \\
    Cut 2 & BDHS 2004--2007 ($N = 13{,}058$) & BDHS 2011 & 8{,}742 & 410 \\
    Cut 3 & BDHS 2004--2011 ($N = 21{,}800$) & BDHS 2014 & 7{,}886 & 319 \\
    Cut 4 & BDHS 2004--2014 ($N = 29{,}686$) & BDHS 2017--18 & 8{,}684 & 356 \\
    Cut 5 & BDHS 2004--2017--18 ($N = 38{,}370$) & \textbf{BDHS 2022 (Locked)} & \textbf{13{,}119} & \textbf{408} \\
    \bottomrule
  \end{tabularx}
\end{table}

\subsection{Parameter Configuration}
\label{sec:exp_parameters}

Model architectures and pipeline hyperparameters were configured as follows:
\begin{itemize}
    \item \textbf{XGBoost (Base Learner):} \texttt{n\_estimators} = 250, \texttt{max\_depth} = 4, \texttt{learning\_rate} = 0.05, \texttt{subsample} = 0.8, \texttt{colsample\_bytree} = 0.8, \texttt{min\_child\_weight} = 3, \texttt{gamma} = 0.1, \texttt{scale\_pos\_weight} = $32.4$, objective = \texttt{binary:logistic}, evaluation metric = \texttt{logloss}.
    \item \textbf{LightGBM:} \texttt{n\_estimators} = 250, \texttt{max\_depth} = 4, \texttt{learning\_rate} = 0.05, \texttt{num\_leaves} = 15, \texttt{subsample} = 0.8, \texttt{colsample\_bytree} = 0.8, \texttt{scale\_pos\_weight} = $32.4$.
    \item \textbf{Logistic Regression:} $L_2$ regularization with inverse penalty $C = 1.0$, solver = \texttt{lbfgs}, \texttt{class\_weight} = \texttt{balanced}.
    \item \textbf{Random Forest:} \texttt{n\_estimators} = 100, \texttt{max\_depth} = 8, \texttt{min\_samples\_split} = 10, \texttt{class\_weight} = \texttt{balanced\_subsample}.
    \item \textbf{Module A Configuration:} Temporal decay parameter $\lambda = 0.05$, domain classifier $C_g = 1.0$, density-ratio clipping bounds $[r_{\min}, r_{\max}] = [0.1, 10.0]$, composite weight clipping bounds $[0.01, 10.0]$.
    \item \textbf{Module B Configuration:} Shrinkage intensity prior constant $\kappa = 40.0$.
    \item \textbf{Module C Configuration:} Platt logistic parameters optimized via L-BFGS-B with bounds $a \in [0.1, 5.0]$ and $b \in [-10.0, 5.0]$.
    \item \textbf{Operational Decision Thresholds:} Validation-locked Youden cutoff $\tau_w = 0.04546$; community outreach prioritization capacity fixed at top $10\%$ ($N_{\text{prioritized}} = 1{,}312$).
\end{itemize}

\subsection{Evaluation Metrics}
\label{sec:exp_metrics}

All evaluation metrics are formulated incorporating normalized DHS sampling weights $w_i$ ($W = \sum_{i=1}^N w_i$):
\begin{enumerate}
    \item \textbf{Weighted Sensitivity (Recall):}
    \begin{equation}
      \text{Sensitivity}_w = \frac{\sum_{i=1}^N w_i \cdot y_i \cdot \hat{y}_i}{\sum_{i=1}^N w_i \cdot y_i} = \frac{\text{TP}_w}{\text{TP}_w + \text{FN}_w}.
    \end{equation}
    \item \textbf{Weighted Specificity:}
    \begin{equation}
      \text{Specificity}_w = \frac{\sum_{i=1}^N w_i \cdot (1 - y_i) \cdot (1 - \hat{y}_i)}{\sum_{i=1}^N w_i \cdot (1 - y_i)} = \frac{\text{TN}_w}{\text{TN}_w + \text{FP}_w}.
    \end{equation}
    \item \textbf{Weighted Precision (Positive Predictive Value, PPV):}
    \begin{equation}
      \text{PPV}_w = \frac{\sum_{i=1}^N w_i \cdot y_i \cdot \hat{y}_i}{\sum_{i=1}^N w_i \cdot \hat{y}_i} = \frac{\text{TP}_w}{\text{TP}_w + \text{FP}_w}.
    \end{equation}
    \item \textbf{Weighted Accuracy and Balanced Accuracy:}
    \begin{equation}
      \text{Accuracy}_w = \frac{\text{TP}_w + \text{TN}_w}{W}, \quad \text{Balanced Accuracy}_w = \frac{\text{Sensitivity}_w + \text{Specificity}_w}{2}.
    \end{equation}
    \item \textbf{Weighted F1-Score and Matthews Correlation Coefficient (MCC):}
    \begin{equation}
      \text{F1}_w = \frac{2 \cdot \text{PPV}_w \cdot \text{Sensitivity}_w}{\text{PPV}_w + \text{Sensitivity}_w}, \quad \text{MCC}_w = \frac{\text{TP}_w \text{TN}_w - \text{FP}_w \text{FN}_w}{\sqrt{(\text{TP}_w + \text{FP}_w)(\text{TP}_w + \text{FN}_w)(\text{TN}_w + \text{FP}_w)(\text{TN}_w + \text{FN}_w)}}.
    \end{equation}
    \item \textbf{Survey-Weighted ROC-AUC and PR-AUC:} Computed by integrating the weighted receiver operating characteristic and precision-recall curves across all possible decision thresholds $\tau \in [0, 1]$ \cite{saito2015}.
    \item \textbf{Survey-Weighted Brier Score and Brier Skill Score (BSS):}
    \begin{equation}
      \text{Brier}_w = \frac{1}{W} \sum_{i=1}^N w_i (\hat{p}_i^* - y_i)^2, \quad \text{BSS}_w = 1 - \frac{\text{Brier}_w}{\text{Brier}_{\text{null}}},
    \end{equation}
    where $\text{Brier}_{\text{null}} = \bar{\pi}_w (1 - \bar{\pi}_w) \approx 0.0290$ for survey prevalence $\bar{\pi}_w = 0.0299$.
    \item \textbf{Calibration Intercept and Slope:} Estimated via weighted logistic regression: $\text{logit}(P(Y=1)) = \alpha + \beta \cdot \text{logit}(\hat{p}^*)$. Perfect calibration corresponds to $\alpha = 0$ and $\beta = 1$.
    \item \textbf{Decision-Curve Analysis Net Benefit (NB):}
    \begin{equation}
      \text{NB}(p_t) = \frac{\text{TP}_w}{W} - \frac{\text{FP}_w}{W} \left( \frac{p_t}{1 - p_t} \right),
    \end{equation}
    evaluated across clinical threshold probabilities $p_t \in [0.01, 0.10]$ \cite{vickers2006}.
    \item \textbf{Number Needed to Screen (NNS) and Prioritization Yield:}
    \begin{equation}
      \text{NNS} = \frac{1}{\text{PPV}_w}, \quad \text{Yield}_{\text{prioritized}} = \text{PPV}_w \times 1{,}000.
    \end{equation}
\end{enumerate}

\section{Results}
\label{sec:results}

This section reports the empirical findings of SWIFT-U5 on the locked BDHS 2022 test cohort ($N = 13{,}119$, $408$ deaths, survey-weighted prevalence $2.99\%$). In accordance with faculty writing guidelines, results are organized metric-by-metric, followed by comparative benchmarks, ablation analyses, statistical inference, and in-depth discussion.

\subsection{Results Based on Evaluation Metrics}
\label{sec:res_metrics}

Table~\ref{tab:binary_metrics} presents the complete binary classification performance profile of SWIFT-U5 across two operational regimes: the validation-locked Youden operating point ($\tau_w = 0.04546$) and the simulated 10\% community outreach prioritization capacity.

\begin{table}[pos=htbp]
  \centering
  \caption{Standard binary classification performance of SWIFT-U5 on the locked BDHS 2022 test cohort ($N = 13{,}119$, $408$ deaths, survey-weighted prevalence $2.99\%$) under two distinct operational threshold regimes.}
  \label{tab:binary_metrics}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.28\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}X@{}}
    \toprule
    Operating Regime & Threshold ($\tau$) & Sens (Recall) & Spec & Prec (PPV) & Accuracy & Bal Acc & F1-Score \\
    \midrule
    Validation Youden ($J$) & 0.04546 & 47.18\% & 70.61\% & 4.72\% & 69.91\% & 58.90\% & 0.0858 \\
    10\% National Prioritization Capacity & Rank Top 10\% & 24.60\% & 90.45\% & 7.70\% & 88.43\% & 57.53\% & 0.1173 \\
    Naive All-Negative Baseline & N/A & 0.00\% & 100.00\% & 0.00\% & 97.01\% & 50.00\% & 0.0000 \\
    \bottomrule
  \end{tabularx}
  \vspace{4pt}
  \raggedright\footnotesize
  \textbf{Survey-Weighted Contingency Mass (Total Population Weight = $13{,}169.9$, Survey-Weighted Estimated Event Total = $394.2$):}\\
  Validation Youden Regime ($\tau = 0.04546$): $\text{TP}_w = 185.3$ weighted true-positive event equivalents ($207$ unw deaths), $\text{FP}_w = 3{,}740.6$ ($3{,}947$ unw), $\text{TN}_w = 8{,}985.7$ ($8{,}764$ unw), $\text{FN}_w = 207.4$ ($201$ unw).\\
  10\% Prioritization Capacity: $\text{TP}_w = 97.0$ weighted true-positive event equivalents ($107$ unw deaths; $\text{Sens}_w = 97.0 / 394.2 = 24.60\%$), $\text{FP}_w = 1{,}163.0$ ($1{,}205$ unw), $\text{TN}_w = 11{,}612.6$ ($11{,}506$ unw), $\text{FN}_w = 297.3$ ($301$ unw) ($\text{PPV}_w = 97.0 / 1{,}260.0 = 7.70\%$; $\text{NNS} = 13.0$; weighted estimated prioritization yield $\approx 77.0$/1k prioritized records).
\end{table}

\subsubsection{Accuracy}
Under the validation-locked Youden regime ($\tau_w = 0.04546$), SWIFT-U5 achieved a survey-weighted accuracy of \textbf{69.91\%}. In simulated 10\% community prioritization capacity, accuracy rose to \textbf{88.43\%}. Under extreme class imbalance ($33:1$), an uninformative all-negative classifier trivially attains $97.01\%$ accuracy by predicting zero deaths. SWIFT-U5's $69.91\%$ accuracy reflects an intentional, clinically calibrated operational tradeoff to elevate mortality sensitivity.

\subsubsection{Precision (Positive Predictive Value)}
At the Youden threshold, precision was \textbf{4.72\%} ($185.3$ weighted true-positive event mass over $3{,}925.9$ weighted positive predictions; unweighted PPV: $207 / 4{,}154 = 4.98\%$). Under 10\% prioritization capacity, precision increased to \textbf{7.70\%} ($96.97$ weighted true-positive event equivalents over $1{,}260.02$ prioritized population weight; unweighted PPV: $107 / 1{,}312 = 8.16\%$), representing a $2.57\times$ enrichment over the baseline national survey prevalence ($2.99\%$).

\subsubsection{Recall / Sensitivity}
SWIFT-U5 achieved a survey-weighted Sensitivity of \textbf{47.18\%} at the Youden operating point ($185.3$ weighted true-positive event equivalents out of $394.2$ survey-weighted estimated event total; unweighted sensitivity: $207 / 408 = 50.74\%$). Under 10\% prioritization capacity, the model captured \textbf{24.60\%} of survey-weighted mortality ($96.97 / 394.24$; unweighted sensitivity: $107 / 408 = 26.23\%$).

\subsubsection{Specificity}
Specificity was \textbf{70.61\%} under the Youden regime ($\text{TN}_w = 8{,}985.7$ out of $12{,}726.3$ non-event weight). Under 10\% prioritization capacity, specificity reached \textbf{90.45\%} ($\text{TN}_w = 11{,}612.6$), confirming that $90.5\%$ of non-event households are correctly spared from outreach prioritization.

\subsubsection{F1-Score}
The harmonic mean of precision and recall yielded an F1-score of \textbf{0.0858} under the Youden operating regime. Under 10\% prioritization capacity, the F1-score reached \textbf{0.1173}. In rare-event demographic settings where baseline prevalence is under 3\%, F1-scores are bounded by low precision; nonetheless, SWIFT-U5 achieves a substantial lift over the trivial baseline ($\text{F1} = 0.0000$).

\subsubsection{Balanced Accuracy and Matthews Correlation Coefficient}
Accounting for class imbalance, SWIFT-U5 achieved a Balanced Accuracy of \textbf{58.90\%} at the Youden threshold and \textbf{57.53\%} under 10\% prioritization capacity (both outperforming the trivial baseline of $50.00\%$). The Matthews Correlation Coefficient was \textbf{+0.0662} at the Youden threshold, confirming positive chance-corrected association between predicted risk categories and recorded deaths.

\subsubsection{ROC-AUC}
On the locked BDHS 2022 test cohort, primary SWIFT-U5 achieved a survey-weighted ROC-AUC of \textbf{0.625} ($0.6253$, 95\% PSU cluster-bootstrap CI: [0.587, 0.663]). Uncalibrated baseline XGBoost achieved an AUROC of $0.638$ [95\% CI: 0.603, 0.673]. As detailed in Section~\ref{sec:res_comparative}, SWIFT-U5 trades marginal linear ranking concordance to restore probability calibration and population representativeness.

\subsubsection{PR-AUC}
SWIFT-U5 achieved a survey-weighted Precision-Recall Area Under the Curve (PR-AUC) of \textbf{0.089} (Track A baseline: $0.079$). This represents a $3.0\times$ precision enrichment over the survey-weighted baseline mortality prevalence ($\bar{\pi}_w = 0.0299$), establishing strong utility for high-risk triage.

\subsubsection{Brier Score and Brier Skill Score}
SWIFT-U5 achieved a survey-weighted Brier score of \textbf{0.0287}, outperforming the uninformative constant-prevalence null benchmark of $0.0290$ ($\pi = 2.99\%$). This yields a positive Brier Skill Score of:
\begin{equation}
  \text{BSS} = 1 - \frac{0.0287}{0.0290} = \mathbf{+0.0103} \quad (+1.03\% \text{ skill over null}).
\end{equation}
In contrast, uncalibrated class-rebalanced baseline XGBoost produced a Brier score of $0.1649$ ($\text{BSS} = -4.69$), reflecting severe probability distortion.

\subsubsection{Calibration Analysis}
Assessed via logistic regression of observed mortality on logit predicted probabilities, SWIFT-U5 achieved a Calibration Slope of $\beta = \mathbf{0.806}$ and a Calibration Intercept of $\alpha = \mathbf{-0.905}$ (Figure~\ref{fig:cal_curves}). In contrast, uncalibrated baseline XGBoost exhibited severe logit distortion ($\alpha = -3.135$, $\beta = 1.012$), over-predicting absolute child mortality risk by more than $300\%$.

\subsubsection{Decision-Curve Analysis}
Decision Curve Analysis (Figure~\ref{fig:dca}) on the pre-specified 0.5-percentage-point threshold grid ($p_t \in [0.010, 0.100]$) confirmed that SWIFT-U5 achieves positive standardized net benefit over Treat-All and Treat-None across risk thresholds from $2.5\%$ to $4.5\%$ ($NB = +0.00062$ to $+0.00628$). Between $5.0\%$ and $6.0\%$, the net benefit is negligible ($+0.00005$ to $+0.00020$), representing clinical equivalence with Treat-None. We explicitly disclose that empirical decision curves do not guarantee an unbroken continuous utility interval.

\subsubsection{Prioritization Capacity and Yield}
Under simulated 10\% outreach capacity ($N = 1{,}312$ records prioritized out of $13{,}119$), SWIFT-U5 achieved a Number Needed to Screen of $\text{NNS} = \mathbf{13.0}$ (95\% CI: $10.4$--$16.9$) and a survey-weighted estimated prioritization yield of \textbf{approximately 77 per 1,000 prioritized records} ($76.96 \approx 77.0$ per $1{,}000$; 95\% CI: $59.2$--$95.7$), representing a $\mathbf{2.57\times}$ efficiency multiplier over random household visits. In high-burden northern divisions, estimated yield reached $\mathbf{123.3}$ per $1{,}000$ in Rangpur ($\text{NNS} = 8.1$, $3.24\times$ efficiency) and $\mathbf{101.2}$ per $1{,}000$ in Mymensingh ($\text{NNS} = 9.9$, $3.23\times$ efficiency).

\subsection{Comparative Results}
\label{sec:res_comparative}

\paragraph{Prospective Baseline Comparison}
Table~\ref{tab:table4_baselines} compares SWIFT-U5 against alternative prospective machine learning architectures on the locked BDHS 2022 test set. All baseline architectures (Weighted XGBoost, Weighted Logistic Regression, and Weighted Random Forest) output uncalibrated scores distorted by class imbalance reweighting, producing severe probability overconfidence (Brier scores $0.1346$--$0.2091$ and calibration intercepts $-2.92$ to $-3.38$). SWIFT-U5 is the only model that restores population calibration ($\text{Brier} = 0.0287$) while preserving prioritization triage efficiency ($\text{NNS} = 13.0$).

\begin{table}[pos=htbp]
  \centering
  \caption{Prospective benchmark on locked BDHS 2022 test cohort ($N = 13{,}119$, $408$ deaths): SWIFT-U5 framework vs.\ baseline machine learning architectures and the uninformative null predictor.}
  \label{tab:table4_baselines}
  \small
  \setlength{\tabcolsep}{3pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}>{\centering\arraybackslash}p{0.14\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}X@{}}
    \toprule
    Model Pipeline & AUROC (W) [95\% CI] & Brier Score & Cal Slope & Cal Intercept & NNS @ 10\% \\
    \midrule
    \textbf{SWIFT-U5 (Full Framework)} & \textbf{0.625} [0.587, 0.663] & \textbf{0.0287} & \textbf{0.806} & \textbf{-0.905} & \textbf{13.0} \\
    Weighted XGBoost (Baseline)        & 0.638 [0.603, 0.673]          & 0.1649          & 1.012          & -3.135          & 12.2 \\
    Weighted Logistic (Baseline)       & 0.609 [0.571, 0.643]          & 0.2091          & 0.830          & -3.385          & 11.8 \\
    Weighted Random Forest (Baseline)  & 0.634 [0.599, 0.669]          & 0.1346          & 0.994          & -2.919          & 13.4 \\
    \midrule
    Null Predictor ($\hat{p} = \bar{\pi}_w = 0.030$) & 0.500 [Reference] & 0.0290 & N/A & 0.000 & N/A \\
    \bottomrule
  \end{tabularx}
\end{table}

\paragraph{Temporal vs. Random Split Benchmark ($2 \times 2$ Factorial Results)}
Table~\ref{tab:table2_results} and Figure~\ref{fig:inflation} report the controlled $2 \times 2$ factorial validation benchmark across four experimental arms and five forward temporal cuts. In random split validation (Arms~A and~B), live births from the same survey round are randomly allocated across training and testing, producing an apparent $\text{ROC-AUC}$ of $0.706$--$0.718$ on the 2022 cohort. When evaluated across a prospective temporal gap (Arms~C and~D), discrimination declines to $0.648$--$0.657$, isolating an apparent statistical inflation gap of $+0.062$ to $+0.064$ AUROC points. Crucially, random validation severely underestimates outreach workload: on the 2022 cohort, random validation suggested an $\text{NNS}_{5\%}$ of $4.6$, whereas prospective temporal validation revealed a workload of $\text{NNS}_{5\%} = 8.1$--$8.9$ (nearly double the required prioritization effort).

\begin{table}[pos=htbp]
  \centering
  \caption{Main $2 \times 2$ Factorial Validation Benchmark: Discrimination, calibration, and screening workload (NNS) across four experimental arms and five forward temporal cuts. For each cut, Arms A and B evaluate matched-N stratified random splits ($N_{\text{test}} \in \{6{,}150; 8{,}742; 7{,}886; 8{,}684; 13{,}119\}$) drawn from the contemporary cohort, directly isolating the statistical inflation induced by random splitting over prospective temporal validation.}
  \label{tab:table2_results}
  \small
  \setlength{\tabcolsep}{3pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}X@{}}
    \toprule
    Experimental Arm & Test Round & AUROC (W) & Brier (W) & Cal Slope & NNS @ 5\% (W) \\
    \midrule
    Arm A: Random / Unweighted & 2007 & 0.722 [0.698, 0.743] & 0.1659 & 0.745 & 4.3 \\
    Arm B: Random / Weighted   & 2007 & 0.725 [0.702, 0.747] & 0.1612 & 0.791 & 3.8 \\
    Arm C: Temporal / Unweighted & 2007 & 0.747 [0.732, 0.773] & 0.1568 & 0.876 & 3.9 \\
    Arm D: Temporal / Weighted   & 2007 & 0.750 [0.728, 0.771] & 0.1511 & 0.914 & 3.5 \\
    \midrule
    Arm A: Random / Unweighted & 2011 & 0.764 [0.739, 0.780] & 0.1699 & 1.029 & 3.7 \\
    Arm B: Random / Weighted   & 2011 & 0.752 [0.731, 0.774] & 0.1664 & 0.996 & 3.8 \\
    Arm C: Temporal / Unweighted & 2011 & 0.716 [0.670, 0.742] & 0.2682 & 1.178 & 4.8 \\
    Arm D: Temporal / Weighted   & 2011 & 0.699 [0.663, 0.735] & 0.2494 & 1.010 & 5.1 \\
    \midrule
    Arm A: Random / Unweighted & 2014 & 0.760 [0.732, 0.788] & 0.1802 & 1.082 & 3.5 \\
    Arm B: Random / Weighted   & 2014 & 0.759 [0.728, 0.787] & 0.1784 & 1.103 & 3.4 \\
    Arm C: Temporal / Unweighted & 2014 & 0.655 [0.611, 0.678] & 0.2668 & 0.501 & 8.8 \\
    Arm D: Temporal / Weighted   & 2014 & 0.648 [0.615, 0.682] & 0.2586 & 0.494 & 9.7 \\
    \midrule
    Arm A: Random / Unweighted & 2017--18 & 0.734 [0.704, 0.759] & 0.1812 & 0.984 & 4.6 \\
    Arm B: Random / Weighted   & 2017--18 & 0.728 [0.699, 0.758] & 0.1772 & 0.968 & 4.3 \\
    Arm C: Temporal / Unweighted & 2017--18 & 0.694 [0.654, 0.717] & 0.1584 & 1.052 & 4.7 \\
    Arm D: Temporal / Weighted   & 2017--18 & 0.681 [0.651, 0.713] & 0.1551 & 0.974 & 5.0 \\
    \midrule
    Arm A: Random / Unweighted & 2022 & 0.718 [0.689, 0.734] & 0.1852 & 0.961 & 4.7 \\
    Arm B: Random / Weighted   & 2022 & 0.706 [0.684, 0.730] & 0.1831 & 0.922 & 4.6 \\
    Arm C: Temporal / Unweighted & 2022 & 0.657 [0.615, 0.682] & 0.1763 & 0.971 & 8.1 \\
    Arm D: Temporal / Weighted   & 2022 & 0.648 [0.617, 0.679] & 0.1727 & 0.929 & 8.9 \\
    \bottomrule
  \end{tabularx}
\end{table}

\casinlinefigtall[\linewidth]{fig1_inflation_plot.png}{Validation design inflation plot comparing random split validation (Arms A and B) versus forward temporal validation (Arms C and D) across all five sequential test rounds.}{fig:inflation}

\paragraph{Compound Contrast: Factorial Arm C vs. Prospective Track A SWIFT-U5}
We explicitly distinguish the retrospective factorial diagnostic benchmark (Arm~C, Table~\ref{tab:table2_results}) from the prospective deployment framework (Track~A SWIFT-U5, Table~\ref{tab:table4_baselines}). Factorial Arm~C evaluates raw unweighted XGBoost trained across all five historical waves ($N_{\text{train}} = 38{,}370$, 2004--2018) without held-out validation or recalibration modules, achieving an uncalibrated AUROC of $0.657$ ($0.6568$) with severe probability distortion (Brier score $0.1763$). Conversely, prospective Track~A SWIFT-U5 trains the base learner on historical development waves ($N_{\text{dev}} = 29{,}686$, 2004--2014), incorporates transductive covariate-shift instance weighting (Module~A), and reserves the 2017--18 wave as an independent validation anchor for empirical Bayes shrinkage (Module~B) and drift recalibration (Module~C), achieving a calibrated AUROC of $0.625$ ($0.6253$) and Brier score of $0.0287$. The observed AUROC difference reflects a compound contrast between experimental configurations, including differences in training-data scope, validation design, instance weighting, and post-processing. Therefore, it cannot be attributed to any single component independently.

\subsection{Ablation Study}
\label{sec:res_ablation}

Table~\ref{tab:table5_ablation} isolates the incremental contributions of individual pipeline components from the ablation experiments. Relative to the uncalibrated base model lacking all modules ($\text{Brier} = 0.1627$, calibration intercept $-3.137$), the introduction of drift-aware Platt recalibration (Module~C) is the primary driver of logit scale restoration, improving the calibration slope from $0.493$ to $0.806$ and adjusting the intercept from $-1.640$ to $-0.905$ (reducing Brier error from $0.0293$ to $0.0287$). Removing Module~A (recency and density-shift weighting) preserved Brier error ($0.0286$) with an intercept of $-0.641$, while omitting Module~B (empirical Bayes shrinkage) resulted in an intercept of $-0.902$. These ablation findings confirm that the overall Brier-score reduction from the $0.16$ baseline regime reflects the complete pipeline architecture, with Module~C specifically providing the final parametric recalibration on held-out validation logits.

\begin{table}[pos=htbp]
  \centering
  \caption{Ablation analysis on BDHS 2022 test cohort: Stepwise removal of SWIFT-U5 modules.}
  \label{tab:table5_ablation}
  \small
  \begin{tabular}{lcccc}
    \toprule
    Ablation Configuration & AUROC (W) & Brier Score (W) & Cal Intercept & NNS @ 5\% \\
    \midrule
    \textbf{Full SWIFT-U5 (Modules A+B+C)} & \textbf{0.625} & \textbf{0.0287} & \textbf{-0.905} & \textbf{8.8} \\
    SWIFT-U5 w/o Module A (No Shift/Recency Weighting) & 0.638 & 0.0286 & -0.641 & 8.7 \\
    SWIFT-U5 w/o Module B (No Division Shrinkage Head)  & 0.626 & 0.0287 & -0.902 & 8.7 \\
    SWIFT-U5 w/o Module C (No Drift Recalibration)     & 0.625 & 0.0293 & -1.640 & 8.8 \\
    Base Model (No Modules A, B, C)                    & 0.638 & 0.1627 & -3.137 & 8.8 \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Statistical Analysis}
\label{sec:res_statistical}

In accordance with complex survey methodology, statistical uncertainty was quantified using 1,000 design-based PSU cluster bootstrap resamples. The 95\% confidence interval for primary SWIFT-U5 ROC-AUC was [0.587, 0.663], overlapping with the baseline XGBoost interval [0.603, 0.673]. For prioritization metrics at 10\% capacity, the 95\% CI for sensitivity was [19.8\%, 30.4\%], for NNS was [10.4, 16.9], and for estimated prioritization yield was [59.2, 95.7] per 1,000 prioritized. No claim of statistically significant superiority over baseline models is made; comparisons are presented as descriptive empirical contrasts between distinct methodological configurations.

\subsection{Discussion of Key Findings}
\label{sec:res_discussion}

\paragraph{Overall Performance and Calibration Restoration}
The prospective empirical results demonstrate that when child mortality models are stripped of retrospective survival leakage and evaluated across an out-of-time temporal gap, discrimination is moderate ($\text{AUROC} = 0.625$). In rare-event demographic settings ($33:1$ imbalance), SWIFT-U5's primary clinical contribution is restoring probability calibration (Brier score $0.0287$, slope $0.806$, intercept $-0.905$) over raw class-weighted boosting ($\text{Brier} = 0.1649$, intercept $-3.135$), preventing systematic overconfidence in triage resource planning (Figure~\ref{fig:cal_curves} and Figure~\ref{fig:dca}).

\casinlinefig[\linewidth]{fig2_calibration_curves.png}{Reliability calibration curves across Arms A, B, C, and D, demonstrating severe probability distortion under unweighted learning and restoration under survey-aware calibration.}{fig:cal_curves}

\casinlinefig[\linewidth]{decision_curve_analysis.png}{Decision Curve Analysis (DCA): Standardized net benefit of SWIFT-U5 versus Treat-All and Treat-None across risk thresholds from 1\% to 10\% on the locked BDHS 2022 test cohort.}{fig:dca}

\paragraph{Longitudinal Predictor Stability and Explainable AI}
Figure~\ref{fig:stability} displays feature selection frequencies across 40 PSU-bootstrapped iterations for weighted LASSO and XGBoost permutation importance. Four features emerged as an immutable \textbf{Stable Core} across nearly two decades of survey microdata: antenatal care visits (\texttt{anc\_visits}, selection frequency = $1.00$), preceding birth interval (\texttt{preceding\_birth\_interval}, frequency = $0.955$), multiple birth indicator (\texttt{is\_twin}, frequency = $0.800$), and maternal education (\texttt{maternal\_education}, frequency = $0.790$). TreeSHAP attributions on BDHS 2022 confirmed that these features dominate model decisions (Figure~\ref{fig:shap_drift}A), while the SHAP Drift Index ($\text{SDI}$) remained bounded below $0.25$ across all historical cuts (declining to $0.007$ in 2022, Figure~\ref{fig:shap_drift}C).

\casinlinefig[\linewidth]{fig3_feature_stability.png}{Dual-method PSU stability selection matrix: Joint selection frequency across weighted LASSO and XGBoost permutation importance, distinguishing stable core predictors from temporally drifting covariates.}{fig:stability}

\casinlinefigtall[\linewidth]{fig5_shap_drift_division.png}{Post-hoc TreeSHAP interpretation and temporal drift: (A) Global top 12 risk drivers on BDHS 2022; (B) Regional driver attribution heatmap across eight divisions; (C) Temporal SHAP Drift Index (SDI) across five sequential forward temporal cuts (2007--2022).}{fig:shap_drift}

\paragraph{Division Policy Layer and Geographic Heterogeneity}
Table~\ref{tab:table6_capacity} and Figure~\ref{fig:atlas} summarize division-level operational capacity on the BDHS 2022 test cohort. Substantial geographic variation was observed: in high-burden northern divisions, estimated prioritization yield reached \textbf{123.3 per 1,000 prioritized} in Rangpur ($\text{NNS} = 8.1$, $3.24\times$ efficiency) and \textbf{101.2 per 1,000 prioritized} in Mymensingh ($\text{NNS} = 9.9$, $3.23\times$ efficiency), compared to $38.1$ per $1{,}000$ in Dhaka ($\text{NNS} = 26.2$). These findings demonstrate that population risk stratification can effectively identify geographic pockets of vulnerability.

\begin{table}[pos=htbp]
  \centering
  \caption{Division-level prioritization capacity, sensitivity, NNS, and survey-weighted estimated prioritization yield on BDHS 2022 test cohort ($N = 13{,}119$, $408$ deaths) with 95\% PSU-stratified cluster-bootstrap confidence intervals. Note: Metrics evaluate retrospective prioritization on survey-recorded deaths rather than prospective clinical screening.}
  \label{tab:table6_capacity}
  \small
  \setlength{\tabcolsep}{2pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\centering\arraybackslash}p{0.08\linewidth}>{\centering\arraybackslash}p{0.08\linewidth}>{\centering\arraybackslash}p{0.16\linewidth}>{\centering\arraybackslash}p{0.16\linewidth}>{\centering\arraybackslash}p{0.18\linewidth}>{\centering\arraybackslash}X@{}}
    \toprule
    Division & Sample ($N$) & Deaths & Obs Rate (‰) & Sens @ 10\% [95\% CI] & NNS @ 10\% [95\% CI] & Yield/1k Prioritized @ 10\% & Efficiency \\
    \midrule
    Barishal   & 1{,}464 & 50 & 33.0 & 21.6\% [10.8, 37.2] & 15.6 [8.7, 32.0] & 64.2 [31.3, 115.0] & 1.94$\times$ \\
    Chattogram & 2{,}219 & 69 & 30.9 & 30.2\% [17.2, 42.4] & 11.1 [7.5, 20.7] & 90.0 [48.4, 132.5] & 2.91$\times$ \\
    Dhaka      & 1{,}995 & 48 & 24.8 & 15.4\% [6.6, 27.2]  & 26.2 [14.6, 69.3] & 38.1 [14.8, 68.4]  & 1.54$\times$ \\
    Khulna     & 1{,}448 & 34 & 21.7 & 21.5\% [7.2, 39.5]  & 21.4 [10.4, 77.6] & 46.8 [12.9, 96.3]  & 2.15$\times$ \\
    Mymensingh & 1{,}563 & 50 & 31.4 & 32.0\% [16.9, 44.4] & 9.9 [6.9, 20.1]   & 101.2 [50.0, 144.3] & 3.23$\times$ \\
    Rajshahi   & 1{,}309 & 32 & 27.1 & 29.6\% [12.4, 44.2] & 13.0 [7.6, 42.1]  & 77.2 [23.7, 131.0] & 2.85$\times$ \\
    Rangpur    & 1{,}505 & 54 & 38.1 & 32.1\% [21.1, 46.4] & \textbf{8.1} [5.0, 14.2]  & \textbf{123.3} [70.6, 198.7] & 3.24$\times$ \\
    Sylhet     & 1{,}616 & 71 & 44.3 & 20.7\% [11.9, 29.1] & \textbf{11.3} [7.0, 21.5] & \textbf{88.2} [46.6, 143.7] & 1.99$\times$ \\
    \midrule
    \textbf{National} & \textbf{13{,}119} & \textbf{408} & \textbf{29.9} & \textbf{24.6\%} [19.8, 30.4] & \textbf{13.0} [10.4, 16.9] & \textbf{77.0} [59.2, 95.7] & \textbf{2.57$\times$} \\
    \bottomrule
  \end{tabularx}
\end{table}

\casinlinefigtall[\linewidth]{fig4_division_risk_atlas.png}{Division Policy Layer: (A) 2022 Observed mortality vs.\ SWIFT-U5 calibrated predicted risk across eight administrative divisions; (B) Retrospective prioritization yield (survey-weighted estimated yield per 1,000 prioritized records at 10\% capacity) vs.\ random baseline; (C) Longitudinal U5MR trajectories across Bangladesh (2004--2022).}{fig:atlas}

\paragraph{Algorithmic Equity Audit}
Table~\ref{tab:table7_equity} reports the demographic equity audit across residence, wealth quintiles, and maternal education. Sensitivity was comparable across rural ($23.3\%$, $\text{NNS} = 13.8$) and urban ($28.5\%$, $\text{NNS} = 10.7$) strata. Across wealth quintiles, prioritization exhibited a pro-poor distribution: Quintile~1 (poorest) achieved an NNS of \textbf{9.2} (capturing $27.1\%$ of deaths), whereas Quintile~5 (wealthiest) rose to an NNS of $20.7$. Mothers with no formal education exhibited high sensitivity ($35.1\%$, $\text{NNS} = 7.7$); however, because this stratum contains only $28$ death events ($n = 28 < 50$), it is formally designated as \textit{Insufficient Evidence} rather than certified fairness.

\begin{table}[pos=htbp]
  \centering
  \caption{Demographic Equity Audit of SWIFT-U5 risk predictions across residence, wealth quintiles, and maternal education on the BDHS 2022 test cohort ($N = 13{,}119$). Note: Event counts within certain subgroups are modest (e.g., 28 deaths in the uneducated cohort); results reflect descriptive subgroup ranking performance rather than definitive fairness certification.}
  \label{tab:table7_equity}
  \small
  \setlength{\tabcolsep}{3pt}
  \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\centering\arraybackslash}p{0.08\linewidth}>{\centering\arraybackslash}p{0.08\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}X@{}}
    \toprule
    Demographic Domain & Subgroup & Sample ($N$) & Deaths & AUROC (W) & Sens @ 10\% & NNS @ 10\% & Status \\
    \midrule
    \textbf{Overall}   & National & 13{,}119 & 408 & 0.625 & 24.6\% & 13.0 & Flagged (Cal Drift) \\
    \midrule
    \textbf{Residence} & Rural & 8{,}761 & 275 & 0.633 & 23.3\% & 13.8 & Flagged (Cal Drift) \\
                       & Urban & 4{,}358 & 133 & 0.601 & 28.5\% & 10.7 & Flagged (Cal Drift) \\
    \midrule
    \textbf{Wealth Quintile} & Q1 (Poorest) & 2{,}910 & 117 & 0.628 & 27.1\% & \textbf{9.2} & Satisfied \\
                             & Q2 (Poorer)  & 2{,}604 & 101 & 0.629 & 26.4\% & 9.8 & Flagged (Cal Drift) \\
                             & Q3 (Middle)  & 2{,}551 & 78  & 0.597 & 18.7\% & 16.6 & Flagged (Cal Drift) \\
                             & Q4 (Richer)  & 2{,}518 & 56  & 0.570 & 21.4\% & 18.8 & Flagged (Cal Drift) \\
                             & Q5 (Richest) & 2{,}536 & 56  & 0.611 & 20.2\% & 20.7 & Flagged (Cal Drift) \\
    \midrule
    \textbf{Maternal Education} & No Education & 824 & 28 & \textbf{0.709} & \textbf{35.1\%} & \textbf{7.7} & Insufficient Evidence ($n < 50$) \\
                                & Primary      & 3{,}092 & 110 & 0.594 & 20.5\% & 13.6 & Flagged (Cal Drift) \\
                                & Secondary    & 6{,}788 & 226 & 0.613 & 23.7\% & 12.6 & Flagged (Cal Drift) \\
                                & Higher       & 2{,}415 & 44  & 0.536 & 19.4\% & 31.5 & Flagged (Cal Drift; $n < 50$) \\
    \bottomrule
  \end{tabularx}
\end{table}

\section{Conclusion}
\label{sec:conclusion}

This research developed, formalized, and prospectively evaluated SWIFT-U5---a survey-weighted and drift-calibrated machine learning framework for under-five child mortality risk stratification and community outreach prioritization simulation using Bangladesh Demographic and Health Survey microdata. By auditing feature timing to eliminate retrospective survival leakage and grounding predictions on durable pre-birth maternal reproductive indicators and household infrastructure, we established an honest, reproducible benchmark of prospective predictive performance. The proposed architecture integrated transductive covariate-shift density weighting with recency decay (Module~A), empirical Bayes division logit shrinkage (Module~B), and drift-aware Platt recalibration (Module~C) across six sequential BDHS survey rounds (2004 to 2022). On the locked prospective BDHS 2022 test cohort ($N = 13{,}119$, $408$ deaths, survey-weighted prevalence $2.99\%$), primary SWIFT-U5 achieved a survey-weighted ROC-AUC of $0.625$ [95\% CI: 0.587--0.663] compared to $0.638$ [95\% CI: 0.603--0.673] for uncalibrated baseline XGBoost. In descriptive comparison between configurations, SWIFT-U5 substantially restored probability calibration, achieving a survey-weighted Brier score of $0.0287$ compared to $0.1649$ for uncalibrated class-rebalanced baseline XGBoost (outperforming the constant-prevalence null benchmark of $0.0290$), with a calibration slope of $0.806$ and intercept of $-0.905$. In simulated community outreach prioritization at 10\% capacity ($1{,}312$ records prioritized containing $107$ unweighted recorded deaths, corresponding to approximately $97$ weighted true-positive event equivalents), the model achieved a survey-weighted sensitivity of $24.60\%$ ($\text{NNS} = 13.0$) with a weighted estimated prioritization yield of approximately 77 per 1,000 prioritized records ($77.0$ per 1,000; $2.57\times$ efficiency over random selection), rising to an estimated yield of $123.3$ per 1,000 prioritized in high-burden Rangpur division ($\text{NNS} = 8.1$). Antenatal care compliance, birth spacing, multiple gestation, and maternal education emerged as the dominant stable predictors.

The findings demonstrate both the practical utility and inherent boundaries of population-level demographic risk stratification. Several methodological limitations define directions for future inquiry: (i) living children under five years of age at interview have incomplete follow-up, introducing potential right-censoring that warrants extension to survival hazard models with censored loss formulations; (ii) household physical infrastructure and assets are ascertained at interview rather than birth, serving as durable background proxies rather than time-varying measurements; (iii) regional sub-strata with low event counts limit statistical power for subgroup fairness certification; and (iv) because these results evaluate retrospective prioritization on cross-sectional survey-recorded deaths, prospective clinical evaluation among eligible living children enrolled at primary Community Clinics is required before operational deployment claims can be made. Future research should evaluate geographic external validity across South Asian and Sub-Saharan African DHS settings and extend spatial modeling to upazila-level micro-planning.

\section*{CRediT Authorship Contribution Statement}
\noindent\textbf{Md. Rafiul Azam Chowdhury:} Conceptualization, Methodology, Software, Formal Analysis, Writing -- Original Draft.\\
\textbf{Rezaul Mursalin Khan:} Conceptualization, Investigation, Data Curation, Writing -- Review \& Editing.\\
\textbf{Fatema Tuz Zohura Rifa:} Validation, Software, Formal Analysis, Visualization.\\
\textbf{Tanjim Ahmed Irfad:} Software, Data Curation, Validation, Investigation.\\
\textbf{Arafat Hussain Nahid:} Formal Analysis, Visualization, Writing -- Review \& Editing.\\
\textbf{Ohidujjaman:} Supervision, Project Administration, Methodology, Writing -- Review \& Editing.

\section*{Declaration of Competing Interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Funding Statement}
This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

\section*{Data Availability Statement}
The Demographic and Health Survey microdata analyzed in this study are publicly available from the DHS Program (\url{https://dhsprogram.com}) upon registration and approved project application. The complete Python implementation, feature processing crosswalks, model weights, and replication scripts are archived in the study's public repository.

\section*{Code Availability Statement}
All analytical code, data processing pipelines, and experimental evaluation scripts are implemented in Python and made available under an open-source research license at the project repository.

\section*{Ethical Approval Statement}
This study constitutes secondary analysis of de-identified, publicly available Demographic and Health Survey microdata. The original survey protocols were approved by the ICF Institutional Review Board and the Bangladesh Medical Research Council (BMRC) National Ethics Review Committee. Informed consent was obtained from all survey respondents prior to interview.

\section*{Acknowledgements}
The authors acknowledge the DHS Program, ICF International, and the National Institute of Population Research and Training (NIPORT), Bangladesh, for providing access to the BDHS survey microdata. Computational infrastructure was provided by the Department of Computer Science and Engineering, United International University.

\begin{thebibliography}{46}
\expandafter\ifx\csname natexlab\endcsname\relax\def\natexlab#1{#1}\fi
\providecommand{\url}[1]{\texttt{#1}}
\providecommand{\href}[2]{#2}
\providecommand{\path}[1]{#1}
\providecommand{\DOIprefix}{doi:}
\providecommand{\ArXivprefix}{arXiv:}
\providecommand{\URLprefix}{URL: }
\providecommand{\Pubmedprefix}{pmid:}
\providecommand{\doi}[1]{\href{http://dx.doi.org/#1}{\path{#1}}}
\providecommand{\Pubmed}[1]{\href{pmid:#1}{\path{#1}}}
\providecommand{\biboptions}[1]{}
\biboptions{numbers,sort&compress}

\bibitem{who2023}
World Health Organization, Levels and trends in child mortality: Report 2023, Technical Report, Geneva: World Health Organization (2023). URL: \url{https://www.who.int/publications/i/item/9789240081587}.

\bibitem{unicef2023}
United Nations Inter-agency Group for Child Mortality Estimation, Levels \& trends in child mortality: Report 2023, UNICEF, New York (2023). URL: \url{https://data.unicef.org/resources/levels-and-trends-in-child-mortality/}.

\bibitem{black2010}
R.~E. Black, S.~Cousens, H.~L. Johnson, J.~E. Lawn, I.~Rudan, D.~G. Bassani, P.~Jha, H.~Campbell, C.~F. Walker, R.~Cibulskis, T.~Eisele, L.~Liu, C.~Mathers, Global, regional, and national causes of child mortality in 2008: a systematic analysis, \textit{The Lancet} 375~(9730) (2010) 1969--1987. \doi{10.1016/S0140-6736(10)60549-1}.

\bibitem{you2015}
D.~You, L.~Hug, S.~Ejdemyr, P.~Idele, D.~Hogan, C.~Mathers, P.~Gerland, J.~R. New, L.~Alkema, Global, regional, and national levels and trends in under-5 mortality between 1990 and 2015, with scenario-based projections to 2030: a systematic analysis by the UN Inter-agency Group for Child Mortality Estimation, \textit{The Lancet} 386~(10010) (2015) 2275--2286. \doi{10.1016/S0140-6736(15)00120-8}.

\bibitem{dhsrecode2018}
The DHS Program, DHS-VII Recode Manual, Technical Report, ICF, Rockville, MD (2018). URL: \url{https://dhsprogram.com/pubs/pdf/DHSG4/Recode7_DHS_10Sep2018_DHSG4.pdf}.

\bibitem{croft2023}
T.~N. Croft, C.~K. Allen, B.~W. Zachary, Guide to DHS Statistics, Technical Report, ICF, Rockville, MD (2023). URL: \url{https://dhsprogram.com/data/guide-to-dhs-statistics/}.

\bibitem{niport2024}
National Institute of Population Research and Training (NIPORT) and ICF, Bangladesh Demographic and Health Survey 2022: Final Report, Dhaka, Bangladesh and Rockville, MD (2024).

\bibitem{omran2005}
A.~R. Omran, The Epidemiologic Transition: A Theory of the Epidemiology of Population Change, \textit{The Milbank Quarterly} 83~(4) (2005) 731--757. \doi{10.1111/j.1468-0009.2005.00398.x}.

\bibitem{khan2017}
J.~R. Khan, N.~Awan, A comprehensive analysis on child mortality and its determinants in Bangladesh using frailty models, \textit{Archives of Public Health} 75~(1) (2017) 58. \doi{10.1186/s13690-017-0224-6}.

\bibitem{khan2021}
M.~A. Khan, N.~Khan, O.~Rahman, G.~Mustagir, K.~Hossain, R.~Islam, Trends and projections of under-5 mortality in Bangladesh including the effects of maternal high-risk fertility behaviours and use of healthcare services, \textit{PLOS ONE} 16~(2) (2021) e0246210. \doi{10.1371/journal.pone.0246210}.

\bibitem{mazumder2024causes}
T.~Mazumder, I.~Mohanty, D.~Ahmad, T.~Niyonsenga, An analysis of cause-specific under-5 mortality in Bangladesh using the demographic and health survey 2011 and 2017--2018, \textit{BMC Pediatrics} 24~(1) (2024) 572. \doi{10.1186/s12887-024-04979-6}.

\bibitem{mazumder2024stagnant}
T.~Mazumder, I.~Mohanty, D.~Ahmad, T.~Niyonsenga, An explanation of the stagnant under-5 mortality rate in Bangladesh using multilevel, multivariable analysis of three Demographic and Health Surveys, \textit{Scientific Reports} 14~(1) (2024) 19823. \doi{10.1038/s41598-024-69924-0}.

\bibitem{chowdhury2013}
A.~M.~R. Chowdhury, A.~Bhuiya, M.~E. Chowdhury, S.~Rasheed, Z.~Hussain, L.~C. Chen, The Bangladesh paradox: exceptional health achievement despite economic poverty, \textit{The Lancet} 382~(9906) (2013) 1734--1745. \doi{10.1016/S0140-6736(13)62148-0}.

\bibitem{arifeen2013}
S.~E. Arifeen, A.~Christou, L.~Reichenbach, F.~A. Osman, K.~Azad, K.~S. Islam, F.~Ahmed, H.~Perry, D.~H. Peters, Community-based approaches and partnerships: innovations in health-service delivery in Bangladesh, \textit{The Lancet} 382~(9909) (2013) 2012--2026. \doi{10.1016/S0140-6736(13)62149-2}.

\bibitem{naznin2025}
S.~Naznin, M.~J. Uddin, A.~Kabir, Identifying determinants of under-5 mortality in Bangladesh: A machine learning approach with BDHS 2022 data, \textit{PLOS ONE} 20~(6) (2025) e0324825. \doi{10.1371/journal.pone.0324825}.

\bibitem{fahim2026}
M.~M.~M. Fahim, M.~M. Huq, M.~Sabiruzzaman, M.~R. Karim, Temporal Validation Changes the Apparent Public-Health Utility of Under-Five Mortality Prediction in Bangladesh: A Four-Round DHS Machine-Learning Study, \textit{arXiv preprint arXiv:2602.03957} (2026).

\bibitem{haque2026}
M.~A. Haque, M.~A. Kabir, Machine Learning-Based Prediction of Childhood Stunting in Bangladesh: Fairness and Temporal Robustness Assessment, \textit{arXiv preprint arXiv:2609.24386} (2026).

\bibitem{oduse2024}
S.~Oduse, T.~Zewotir, D.~North, Application of machine learning methods for predicting under-five mortality: analysis of Nigerian demographic health survey 2018 dataset, \textit{BMC Medical Informatics and Decision Making} 24~(1) (2024) 86. \doi{10.1186/s12911-024-02476-5}.

\bibitem{mabula2025}
S.~Mabula, R.~Too, G.~Kerich, Comparative analysis of machine learning algorithms for predicting under-five mortality: evidence from Tanzania Demographic and Health Survey, \textit{Machine Learning Research} 10~(2) (2025) 110--123. \doi{10.11648/j.mlr.20251002.12}.

\bibitem{bitew2020}
F.~H. Bitew, S.~H. Nyarko, L.~Potter, C.~S. Sparks, Machine learning approach for predicting under-five mortality determinants in Ethiopia: evidence from the 2016 Ethiopian Demographic and Health Survey, \textit{Genus} 76~(1) (2020) 37. \doi{10.1186/s41118-020-00106-2}.

\bibitem{saroj2022}
R.~K. Saroj, P.~K. Yadav, R.~Singh, O.~N. Chilyabanyama, Machine learning algorithms for understanding the determinants of under-five mortality, \textit{BioData Mining} 15~(1) (2022) 20. \doi{10.1186/s13040-022-00308-8}.

\bibitem{asmare2026}
A.~A. Asnake, et~al., Prediction of under-five mortality using supervised machine learning algorithms in the 23 sub-Saharan African countries, \textit{Scientific Reports} 16 (2026) 9131. \doi{10.1038/s41598-026-40401-0}.

\bibitem{chen2016}
T.~Chen, C.~Guestrin, XGBoost: A Scalable Tree Boosting System, in: \textit{Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, pp. 785--794 (2016). \doi{10.1145/2939672.2939785}.

\bibitem{ke2017}
G.~Ke, Q.~Meng, T.~Finley, T.~Wang, W.~Chen, W.~Ma, Q.~Ye, T.-Y. Liu, LightGBM: A Highly Efficient Gradient Boosting Decision Tree, in: \textit{Advances in Neural Information Processing Systems} 30 (2017) 3146--3154.

\bibitem{breiman2001}
L.~Breiman, Random Forests, \textit{Machine Learning} 45~(1) (2001) 5--32. \doi{10.1023/A:1010933404324}.

\bibitem{tibshirani1996}
R.~Tibshirani, Regression Shrinkage and Selection via the Lasso, \textit{Journal of the Royal Statistical Society: Series B (Methodological)} 58~(1) (1996) 267--288. \doi{10.1111/j.2517-6161.1996.tb02080.x}.

\bibitem{meinshausen2010}
N.~Meinshausen, P.~B{\"u}hlmann, Stability selection, \textit{Journal of the Royal Statistical Society: Series B (Statistical Methodology)} 72~(4) (2010) 417--473. \doi{10.1111/j.1467-9868.2010.00740.x}.

\bibitem{kursa2010}
M.~B. Kursa, W.~R. Rudnicki, Feature Selection with the Boruta Package, \textit{Journal of Statistical Software} 36~(11) (2010) 1--13. \doi{10.18637/jss.v036.i11}.

\bibitem{platt1999}
J.~C. Platt, Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods, in: \textit{Advances in Large Margin Classifiers}, MIT Press, pp. 61--74 (1999).

\bibitem{vickers2006}
A.~J. Vickers, E.~B. Elkin, Decision Curve Analysis: A Novel Method for Evaluating Prediction Models, \textit{Medical Decision Making} 26~(6) (2006) 565--574. \doi{10.1177/0272989X06295361}.

\bibitem{vanCalster2019}
B.~Van~Calster, D.~J. McLernon, M.~van Smeden, L.~Wynants, E.~W. Steyerberg, Calibration: the Achilles heel of predictive analytics, \textit{BMC Medicine} 17~(1) (2019) 230. \doi{10.1186/s12916-019-1466-7}.

\bibitem{collins2024}
G.~S. Collins, K.~G.~M. Moons, P.~Dhiman, R.~D. Riley, A.~L. Beam, B.~Van~Calster, M.~Ghassemi, X.~Liu, J.~B. Reitsma, M.~van Smeden, et~al., TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods, \textit{BMJ} 385 (2024) e078378. \doi{10.1136/bmj-2023-078378}.

\bibitem{wolff2019}
R.~F. Wolff, K.~G.~M. Moons, R.~D. Riley, P.~F. Whiting, M.~Westwood, G.~S. Collins, J.~B. Reitsma, J.~Kleijnen, S.~Mallett, PROBAST: A Tool to Assess the Risk of Bias and Applicability of Prediction Model Studies, \textit{Annals of Internal Medicine} 170~(1) (2019) 51--58. \doi{10.7326/M18-1376}.

\bibitem{lumley2004}
T.~Lumley, Analysis of Complex Survey Samples, \textit{Journal of Statistical Software} 9~(1) (2004) 1--19. \doi{10.18637/jss.v009.i08}.

\bibitem{wieczorek2022}
J.~Wieczorek, C.~Guerin, T.~McMahon, K-fold cross-validation for complex sample surveys, \textit{Stat} 11~(1) (2022) e454. \doi{10.1002/sta4.454}.

\bibitem{oh2026}
Y.~Oh, H.~W. Zheng, J.~Feng, A.~A.~T. Bui, Survey-aware machine learning: a guideline for valid population health inference based on scoping review, in: \textit{Proceedings of the 7th Conference on Health, Inference, and Learning}, volume 333 of \textit{Proceedings of Machine Learning Research}, PMLR, 2026, pp. 198--220.

\bibitem{shimodaira2000}
H.~Shimodaira, Improving predictive inference under covariate shift by weighting the log-likelihood function, \textit{Journal of Statistical Planning and Inference} 90~(2) (2000) 227--244. \doi{10.1016/S0378-3758(00)00115-4}.

\bibitem{sugiyama2012}
M.~Sugiyama, M.~Kawanabe, Machine Learning in Non-Stationary Environments: Introduction to Covariate Shift Adaptation, MIT Press (2012).

\bibitem{casella1985}
G.~Casella, An introduction to empirical Bayes data analysis, \textit{The American Statistician} 39~(2) (1985) 83--87. \doi{10.1080/00031305.1985.10479401}.

\bibitem{efron2012}
B.~Efron, Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction, Cambridge University Press (2012).

\bibitem{lundberg2017}
S.~M. Lundberg, S.-I. Lee, A Unified Approach to Interpreting Model Predictions, in: \textit{Advances in Neural Information Processing Systems} 30 (2017) 4765--4774.

\bibitem{vyas2020}
D.~A. Vyas, L.~G. Eisenstein, D.~S. Jones, Hidden in Plain Sight --- Reconsidering the Use of Race Correction in Clinical Algorithms, \textit{New England Journal of Medicine} 383~(9) (2020) 874--882. \doi{10.1056/NEJMp2008447}.

\bibitem{obermeyer2019}
Z.~Obermeyer, B.~Powers, C.~Vogeli, S.~Mullainathan, Dissecting racial bias in an algorithm used to manage the health of populations, \textit{Science} 366~(6464) (2019) 447--453. \doi{10.1126/science.aax2342}.

\bibitem{chawla2002}
N.~V. Chawla, K.~W. Bowyer, L.~O. Hall, W.~P. Kegelmeyer, SMOTE: Synthetic Minority Over-sampling Technique, \textit{Journal of Artificial Intelligence Research} 16 (2002) 321--357. \doi{10.1613/jair.953}.

\bibitem{macnell2023}
N.~MacNell, L.~Feinstein, J.~Wilkerson, et~al., Implementing machine learning methods with complex survey data: lessons learned on the impacts of accounting sampling weights in gradient boosting, \textit{PLOS ONE} 18~(3) (2023) e0280387. \doi{10.1371/journal.pone.0280387}.

\bibitem{saito2015}
T.~Saito, M.~Rehmsmeier, The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets, \textit{PLOS ONE} 10~(3) (2015) e0118432. \doi{10.1371/journal.pone.0118432}.

\end{thebibliography}

\end{document}
"""
    with open('docs/manuscript.tex', 'w', encoding='utf-8') as f:
        f.write(tex.strip() + '\n')
    print("Successfully generated docs/manuscript.tex")

if __name__ == '__main__':
    generate_manuscript()
