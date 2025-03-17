using FraudDetectionAPI.Models;
using Microsoft.ML;

namespace FraudDetectionAPI.Services
{
    public class ModelTrainingService
    {
        private IWebHostEnvironment _environment;
        private readonly ILogger<ModelTrainingService> _logger;
        public ModelTrainingService(IWebHostEnvironment environment, ILogger<ModelTrainingService> logger)
        {
            _environment = environment;
            _logger = logger;
        }
        public void Train()
        {
            string FilePath = Path.Combine(_environment.WebRootPath, "Data\\");

            string dataPath = Directory.GetFiles(FilePath)?.FirstOrDefault(f => f == FilePath + "Faulty_data_raw.csv");

            if (dataPath != null)
            {
                var mlContext = new MLContext();

                var dataView = LoadData(mlContext, dataPath);

                var processedData = PreprocessData(mlContext, dataView);

                var preview = processedData.Preview();
                foreach (var row in preview.RowView)
                {
                    _logger.LogInformation(string.Join(", ", row.Values));
                }
                _logger.LogInformation("Data preprocessing complete.");

                // Split the data into training and testing sets
                var trainTestData = mlContext.Data.TrainTestSplit(processedData, testFraction: 0.2);

                // Explicitly access trainData and testData
                IDataView trainData = trainTestData.TrainSet;
                IDataView testData = trainTestData.TestSet;

                foreach (var column in trainData.Schema)
                {
                    _logger.LogInformation($"{column.Name} - {column.Type}");
                }

                // Train the model
                var model = BuildModel(mlContext, trainData);

                // Evaluate the model
                var predictions = model.Transform(testData);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions);

                // Output evaluation metrics
                _logger.LogInformation($"Accuracy: {metrics.Accuracy}");
                _logger.LogInformation($"AUC: {metrics.AreaUnderRocCurve}");
                _logger.LogInformation($"F1 Score: {metrics.F1Score}");

                string TrainedModelPath = Path.Combine(_environment.WebRootPath, "TrainedModel\\") + "trained_model.zip";
                mlContext.Model.Save(model, null, TrainedModelPath);
            }
        }
        public static IDataView LoadData(MLContext mlContext, string dataPath)
        {
            var dataView = mlContext.Data.LoadFromTextFile<TransactionData>(
                path: dataPath,
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true);

            return dataView;
        }
        public static IDataView PreprocessData(MLContext mlContext, IDataView dataView)
        {
            // Preprocess pipeline:
            // 1. One-hot encode the categorical variable (TransactionDescription).
            // 2. Normalize numerical values (Cost, Quantity, StockOnHand).
            // 3. Concatenate all features into a single "Features" column.

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("DATE")
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding("DESCRIPTION"))
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding("ITEMNAME"))
                           .Append(mlContext.Transforms.NormalizeMinMax("UNITCOST"))
                           .Append(mlContext.Transforms.NormalizeMinMax("QUANTITY"))
                           .Append(mlContext.Transforms.NormalizeMinMax("STOCKONHAND"))
                           .Append(mlContext.Transforms.NormalizeMinMax("RUNNINGUNITCOST"))
                           .Append(mlContext.Transforms.Concatenate("Features", "DATE", "DESCRIPTION", "ITEMNAME",
                               "UNITCOST", "QUANTITY", "STOCKONHAND", "RUNNINGUNITCOST"))
                           .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));


            // Apply the transformations to the data
            var processedData = pipeline.Fit(dataView).Transform(dataView);

            return processedData;
        }
        public static ITransformer BuildModel(MLContext mlContext, IDataView trainingData)
        {
            // Define the pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("DATE")
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding("DESCRIPTION"))
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding("ITEMNAME"))
                           .Append(mlContext.Transforms.NormalizeMinMax("UNITCOST"))
                           .Append(mlContext.Transforms.NormalizeMinMax("QUANTITY"))
                           .Append(mlContext.Transforms.NormalizeMinMax("STOCKONHAND"))
                           .Append(mlContext.Transforms.NormalizeMinMax("RUNNINGUNITCOST"))
                           .Append(mlContext.Transforms.Concatenate("Features", "DATE", "DESCRIPTION", "ITEMNAME",
                               "UNITCOST", "QUANTITY", "STOCKONHAND", "RUNNINGUNITCOST"))
                           .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainingData);

            return model;
        }
    }
}
