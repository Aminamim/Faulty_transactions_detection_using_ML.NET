This project implements a fault detection system using ML.NET. It analyzes business transactions (eg: sales, purchases) to identify potentially fraudulent activities or faulty transactions. The model is trained on structured transaction data and leverages machine learning algorithms for anomaly detection.

🛠 Tech Stack
  - .NET 8
  - ML.NET for machine learning
  - C# for implementation
  - Visual Studio 2022 for development

📊 Dataset
The dataset consists of structured business transactions, including:
  - Date, Description, Item Name, Quantity, Unit Cost, Stock on Hand, Running Unit Cost
  - Label (IsFaulty) – indicating whether a transaction is faulty or fraudulent

🔍 Key Features
- Data Preprocessing: Cleans and transforms raw transaction data
- Feature Engineering: Extracts relevant features
- Model Training: Uses ML.NET for binary classification
- Evaluation & Metrics: Assesses model performance with accuracy, AUC, and F1 score
