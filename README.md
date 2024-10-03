# SAP HANA Machine Learning with Predictive Analysis Library (PAL) and Automated Predictive Library (APL)

## Overview

This course is designed to help participants understand how machine learning is implemented in SAP HANA through the use of regression, classification, and time series analysis. By leveraging the capabilities of the **SAP HANA Predictive Analysis Library (PAL)** and the **SAP HANA Automated Predictive Library (APL)**, developers and data scientists will be able to efficiently build, train, and optimize machine learning models.

Machine Learning Workflow:
![workflow](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/MLworkflow.JPG)

At the end of this course, participants will be able to:
- Implement machine learning techniques in SAP HANA using Python clients.
- Apply **regression**, **classification**, and **time series analysis** models.
- Effectively utilize SAP HANA's machine learning libraries (PAL and APL) for predictive analytics.

## Learning Objectives

Participants will gain the following skills and knowledge:
- Understand how the **Python machine learning client for SAP HANA** integrates with machine learning models.
- Build and train machine learning models using various algorithms for **Regression**, **Classification**, and **Time Series** forecasting.
- Learn to use the advanced features of **SAP HANA PAL** and **SAP HANA APL** for robust predictive modeling.

## Prerequisites

Participants are expected to have:
- A **basic understanding of machine learning** principles and algorithms.
- Familiarity with machine learning terminology and concepts.
- **Programming knowledge**, with a working proficiency in **Python**.
- A foundational understanding of data science practices, especially in **predictive modeling** and **data analysis**.

---

## Understanding the Foundations of Data Analysis and Predictive Modeling

### Objective
After completing this section, you will be able to define and explain the concepts of **data analysis** and **predictive modeling** and understand their role in the machine learning workflow within **SAP HANA**.

### Overview of Data Analysis and Predictive Modeling
Predictive analytics plays a crucial role in machine learning, especially in business applications. It is a branch of advanced analytics focused on forecasting **future events, behaviors,** and **outcomes** by leveraging historical data. 

SAP HANA utilizes predictive analytics through machine learning algorithms to help businesses make informed decisions based on insights from past data trends. In this course, we will explore the key elements of data analysis and predictive modeling and understand how these concepts drive **data science** initiatives in SAP HANA.

### Predictive Analytics in the Business Landscape
In today's fast-paced business environment, organizations face significant challenges, including **trade disruptions**, **supply chain fluctuations**, and evolving **market demands**. Predictive analytics helps businesses anticipate future trends with a reasonable degree of accuracy, providing a significant competitive edge.

Key benefits of predictive analytics include:
- **Anticipating future trends**: It helps businesses look into the future, allowing them to react quickly to changing demands or risks.
- **Proactive decision-making**: By predicting potential disruptions or opportunities, businesses can optimize their strategies and mitigate negative impacts.
- **Strategic foresight**: Organizations can forecast supply chain issues, market shifts, or customer behavior, helping them navigate uncertain market conditions.

**Fact**: Businesses that base their decisions on data are **19 times more likely** to be profitable than those that rely solely on intuition (Source: McKinsey).

### Importance of Data in Decision-Making

Data plays a **central role** in decision-making within organizations, especially in the domain of **predictive analytics** and **machine learning**. These technologies rely on historical data to predict future outcomes and retrain models for continual improvement.

Here are some ways businesses leverage data to enhance their decision-making processes:
- **Strategic and operational decisions**: Data analysis helps guide decision-making processes at all organizational levels, from day-to-day operations to long-term strategy formulation.
- **Optimizing operations**: Businesses can identify inefficiencies, optimize resource allocation, and streamline processes using data insights.
- **Mitigating risks**: Advanced analytics help detect potential risks early, allowing companies to take proactive steps to prevent them.
- **Customer insights**: Data-driven personalization helps improve customer experiences, build loyalty, and anticipate customer needs.
- **Competitive advantage**: Data-driven organizations are better equipped to adapt to market changes, driving innovation and long-term growth.

**Fact**: 
- **94%** of companies believe analytics is crucial to their **growth** and **digital transformation** (Source: McKinsey).
- **59%** of organizations are already utilizing **advanced** and **predictive analytics** in their operations.

---

## Predictive Analytics Process

The Predictive Analytics process generally consists of six steps:
![processes](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/PredictiveAnalyticProcess.JPG)


# Applying Classification, Regression, and Time-series Analysis

## Objective

After completing this lesson, you will be able to **identify use cases where classification, regression, and time-series analysis techniques can be applied**.

## Classic Machine Learning Scenarios

This lesson covers three key Machine Learning methods for SAP HANA: **Regression**, **Classification**, and **Time Series Analysis**. These techniques can be applied in various real-world scenarios:

- **Regression**: 
  - Predicting car prices based on model characteristics and market trends.
  
- **Classification**:
  - Predicting customer behaviors, including churn, fraud detection, and purchasing patterns.
  
- **Time Series Analysis**: 
  - Forecasting future sales, demand, costs, and other metrics based on historical data.

---

## Linear Regression: Predicting House Price Based on Living Area

### What is Linear Regression?

**Linear Regression** is one of the most widely known statistical techniques. It helps uncover **linear relationships** between input and output numerical variables. This technique is popular for predictive and statistical modeling, and understanding it is essential for more advanced models like **Generalized Linear Models**.

In the simplest form of linear regression, we map a **predictor variable (x)**, such as the living area of houses, to an **output variable (y)**, like house prices. This is represented as a linear function:

$$
Y = mx + c
$$

Where:
- `Y` is the predicted house price,
- `m` is the slope of the line (the rate at which the price increases with area),
- `c` is the y-intercept.



### Training a Regression Model

The goal of the regression model is to take the living area of a house (input variable) and **predict the house price** (output variable). The model fits a line to the dataset, where the slope (`m`) and y-intercept (`c`) determine the prediction.

The data points in the plot represent **observations** from the dataset, and by minimizing the differences between the predicted and actual values, the model can make accurate predictions.

### Evaluating the Model: Mean Squared Error (MSE)
![SLR](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/SimpleLinearRegression.JPG)

Once the model is built, it’s important to measure how well it fits the data. This is done using the **Mean Squared Error (MSE)**, which calculates the average squared differences between predicted values (Ŷ) and actual values (Yi).

The MSE formula is:

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (Y_i - \hat{Y}_i)^2
$$

Where:
- `N` is the number of data points,
- `Y_i` are the observed values,
- `$\hat{Y}_i$` are the predicted values.


---

## What is Classification?

**Classification** is a machine learning technique used to organize input data into distinct classes. For example, a **spam classifier** determines whether an email is spam or not.

- The model is trained on a **Training Dataset** and evaluated on a **Testing Dataset**.
- After evaluation, the model can classify **new, unseen data** into the appropriate category.

A key distinction between **Classification** and **Regression**:
- **Classification** deals with **discrete target variables** (e.g., spam or not-spam).
- **Regression** deals with **continuous output variables** (e.g., predicting house prices).

![class](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/Classification.JPG)

---

## Time-series Analysis

**Time-series data** refers to data collected about a subject at different points in time. For example:
- A country’s exports over years,
- A company’s sales over months,
- A person’s blood pressure measured every minute.

### Example: Temperature Trends in the UK

The figure below illustrates the **United Kingdom's annual mean temperatures**, measured from 1800 to recent years. This is a **time-series dataset** showing how temperatures have changed over time. The **Met Office** provides historical data revealing a clear warming trend over the years.

According to the **Met Office blog** ([source](https://blog.metoffice.gov.uk/2023/07/14/how-have-daily-temperatures-shifted-in-the-uks-changing-climate/)), the latest 30-year meteorological averages (1991-2020) show almost a **1°C increase** in the UK's annual mean temperature compared to the previous period (1961-1990). 

This data exemplifies how **time-series analysis** can uncover long-term trends, such as climate change.

![time](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/TimeSeries.JPG)

---

## Overview of Algorithms in SAP HANA

**SAP HANA** offers a wide variety of algorithms for **Classification**, **Regression**, and **Time-series analysis**, among others.

### Common Classification Algorithms:
- **Decision Tree Analysis** (CART, C4.5, CHAID),
- **Logistic Regression**,
- **Support Vector Machine**,
- **K-Nearest Neighbor**,
- **Naïve Bayes**,
- **Confusion Matrix** and **AUC**,
- **Online Multi-class Logistic Regression**.

### Common Regression Algorithms:
- **Multiple Linear Regression**,
- **Online Linear Regression**.

### Common Time-series Analysis Algorithms:
- **Auto Exponential Smoothing**,
- **Unified Exponential Smoothing**,
- **Linear Regression** (with damped trend and seasonal adjustments),
- **Hierarchical Forecasting**.

For more details on available algorithms, refer to the **SAP HANA Predictive Analysis Library documentation**.

![al](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/AlgorithmsInSAP.JPG)

---

# Distinguishing Between Supervised and Unsupervised Learning

## Objective

After completing this lesson, you will be able to **distinguish between supervised and unsupervised learning approaches**.

## Key Differences Between Supervised and Unsupervised Learning

### Supervised Learning

Supervised learning involves using **labeled datasets**, meaning each data point includes an input and a known output (target variable). The algorithm learns to predict the output based on the input data. This method is commonly used for:

- **Classification** (e.g., predicting customer churn, fraud detection).
- **Regression** (e.g., forecasting weather, stock prices).

### Unsupervised Learning

![UL](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/UnsupervisedLearning.JPG)

Unsupervised learning works with **unlabeled data**. It aims to discover hidden patterns or groupings in the data without any guidance on the outcome. This method is used for:
- **Clustering** (e.g., anomaly detection, big data visualization).
![ca](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/ClusterOutput.png)

In supervised learning, the goal is to **map input data to known outcomes**, while in unsupervised learning, the algorithm **explores the intrinsic structure** of the data to find patterns.

## Supervised Learning: Training on Labeled Data

Supervised learning models are trained on **labeled datasets**, where each instance has a target variable (label). The algorithm learns to map input features to the correct output. This process is applied in tasks such as:
- **Classification**: Predicting categorical values (e.g., spam vs. non-spam emails).
- **Regression**: Predicting continuous values (e.g., house prices).

### Example: California Housing Dataset
![TD](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/SLTD.png)

A classic example is the **California Housing Dataset**, where the target variable is the **Median House Value**. Features like **median house age**, **average number of rooms per household**, and **average number of bedrooms per household** are used to predict the value of a house.

- **Input Data**: Features like house age, number of rooms, etc.
- **Target**: Median house value.

![TV](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/ExampleTD.png)

After training, the model can predict the **median house value** of a block group based on these input features.

### Model Prediction

Once trained, a supervised learning model can accept new input data (e.g., a block group from the U.S. Census) and predict an outcome, such as the **median house value** for a particular district.

## Unsupervised Learning: Training on Unlabeled Data

In unsupervised learning, the dataset lacks labels. The algorithm looks for patterns within the data. **Clustering** is a common application of unsupervised learning, where similar data points are grouped together based on shared features.

### Example: Clustering in Housing Markets

Using the same **California Housing Dataset**, unsupervised learning can identify **clusters** of similar housing markets. For example, one cluster may represent regions with **high median income** and **an average of 5 or more rooms per household**.

These clusters are not predefined but emerge from the data, providing insights that could be valuable for **home buyers** and **property investors**.

### Model Prediction

After training, an unsupervised learning model can generate predictions or cluster new input data into similar groups. In this case, it may categorize a block group based on its similarity to other regions in the housing market.
![pre](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/ULprediction.png)

## Summary of Differences

- **Supervised Learning**: Uses labeled data, typically applied in **classification** and **regression** tasks. It learns to map inputs to known outputs.
- **Unsupervised Learning**: Works with unlabeled data, often used in **clustering** tasks. It uncovers hidden patterns or groupings in the data.

### Advantages of Unsupervised Learning

Unsupervised learning is valuable when **labeling data is labor-intensive** or **unavailable**. It can reveal patterns or groupings that are otherwise difficult to discover, such as insights into the **Californian housing market** for real estate investments.

By identifying these patterns, unsupervised learning provides **valuable insights** into data that would be hard to obtain manually.


# Implementing a Machine Learning Workflow

![workflow](https://github.com/MohidulHaqueTushar/SAP-Skills-Portfolio/blob/main/AI%20Models%20Deploying%20for%20SAP%20HANA/Images/MLworkflow.JPG)

## Objective

After completing this lesson, you will be able to understand the **basic steps of the machine learning workflow**.

## Overview of the Machine Learning Workflow

In this lesson, we will provide an overview of a typical **Machine Learning workflow**, which involves an **end-to-end process**. This starts with **data sourcing** and **data preparation**, and culminates in the delivery of a **Machine Learning model**. However, there are several steps in between, and we will cover these in detail.

The main goal of a Machine Learning project is to **build a mathematical model** that can solve a given problem by using input data and applying **Machine Learning algorithms**.

### Dataset Analysis using the California Housing Dataset

For illustration, let's consider the **California Housing Dataset**, which includes features such as:

- Median house age
- Average number of rooms per household
- Average number of bedrooms per household
- Median house value

In this case, the **target feature** is the "Median house value for California districts," which is the outcome we aim to predict or understand. This scenario is an example of **Supervised Learning**, where the model learns to predict house values based on historical data.

## The Four Steps of the Machine Learning Workflow

As Data Scientists and developers, you may already be familiar with the **four key steps** in the Machine Learning workflow. Let's revisit these generic steps before delving into the details with the **California housing dataset**.

### Step 1: Data Extraction
This stage involves **carefully selecting relevant features** and structuring the dataset for analysis. Proper data selection ensures that the model has the right inputs for training.

### Step 2: Dataset Partitioning
The dataset is divided into distinct subsets for **training** and **testing**. This partitioning is crucial for effective model training and evaluation, ensuring the model is not overfitting to the training data.

### Step 3: Training an ML Model
In this step, the prepared **training dataset** is fed into a machine learning algorithm to train the model. The goal is to enable the model to **learn patterns** from the data and generate predictions.

### Step 4: Model Evaluation
Once the model is trained, its performance is assessed. If the model does not meet the desired outcomes, it can be **iteratively fine-tuned**, and the training process is repeated until satisfactory results are achieved.


## Conclusion

We have gained practical experience in using SAP HANA’s machine learning libraries to build predictive models that can solve real-world business problems. The skills acquired through this course will enable developers and data scientists to better understand the intricacies of **predictive analytics**, and apply them to enhance business decision-making through data-driven insights.

