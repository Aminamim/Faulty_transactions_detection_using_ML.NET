using System;
using Microsoft.ML;
using Transaction_Anomaly_Detection.Models;
using Transaction_Anomaly_Detection.PreProcessing;
using static Transaction_Anomaly_Detection.PreProcessing.DataLoader;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        string dataPath = @"..\..\..\Data\raw\Faulty_data_raw.csv";  // Update with your data path
        var dataView = DataLoader.LoadData(mlContext, dataPath);

        var processedData = DataPreprocessor.PreprocessData(mlContext, dataView);

        var preview = processedData.Preview();
        foreach (var row in preview.RowView)
        {
            Console.WriteLine(string.Join(", ", row.Values));
        }

        Console.WriteLine("Data preprocessing complete.");

        // Split the data into training and testing sets
        var trainTestData = mlContext.Data.TrainTestSplit(processedData, testFraction: 0.2);

        // Explicitly access trainData and testData
        IDataView trainData = trainTestData.TrainSet;
        IDataView testData = trainTestData.TestSet;

        // Train the model
        var model = ModelBuilder.BuildModel(mlContext, trainData);

        // Evaluate the model
        var predictions = model.Transform(testData);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions);

        // Output evaluation metrics
        Console.WriteLine($"Accuracy: {metrics.Accuracy}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");
        Console.WriteLine($"F1 Score: {metrics.F1Score}");

        // Use the model for predictions (example with a single sample)
        var sample = new TransactionData
        {
            DESCRIPTION = "Sample transaction",
            QUANTITY = 100,
            UNITCOST = 25.5f,
            STOCKONHAND = 500,
            RUNNINGUNITCOST = 20.0f
        };

        var predictionFunction = mlContext.Model.CreatePredictionEngine<TransactionData, TransactionPrediction>(model);
        var prediction = predictionFunction.Predict(sample);

        Console.WriteLine($"Predicted Fraud: {prediction.IsFaulty}");
    }
}