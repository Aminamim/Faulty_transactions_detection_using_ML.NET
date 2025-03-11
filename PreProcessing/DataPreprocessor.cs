using Microsoft.ML;

namespace Transaction_Anomaly_Detection.PreProcessing
{
    public class DataPreprocessor
    {
        public static IDataView PreprocessData(MLContext mlContext, IDataView dataView)
        {
            // Preprocess pipeline:
            // 1. One-hot encode the categorical variable (TransactionDescription).
            // 2. Normalize numerical values (Cost, Quantity, StockOnHand).
            // 3. Concatenate all features into a single "Features" column.


            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("DESCRIPTION")
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding("ITEMNAME"))
                           .Append(mlContext.Transforms.NormalizeMinMax("UNITCOST"))
                           .Append(mlContext.Transforms.NormalizeMinMax("QUANTITY"))
                           .Append(mlContext.Transforms.NormalizeMinMax("STOCKONHAND"))
                           .Append(mlContext.Transforms.NormalizeMinMax("RUNNINGUNITCOST"))
                           .Append(mlContext.Transforms.Concatenate("Features", "DESCRIPTION", "ITEMNAME",
                               "UNITCOST", "QUANTITY", "STOCKONHAND", "RUNNINGUNITCOST"))
                           .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));


            // Apply the transformations to the data
            var processedData = pipeline.Fit(dataView).Transform(dataView);

            return processedData;
        }
    }
}
