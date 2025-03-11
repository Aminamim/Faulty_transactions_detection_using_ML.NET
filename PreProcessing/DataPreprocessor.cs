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

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("DescriptionEncoded", "DESCRIPTION")
                     .Append(mlContext.Transforms.NormalizeMinMax("NormalizedUNITCOST", "UNITCOST"))
                     .Append(mlContext.Transforms.NormalizeMinMax("NormalizedQUANTITY", "QUANTITY"))
                     .Append(mlContext.Transforms.NormalizeMinMax("NormalizedSTOCKONHAND", "STOCKONHAND"))
                     .Append(mlContext.Transforms.NormalizeMinMax("NormalizedRUNNINGUNITCOST", "RUNNINGUNITCOST"))
                     .Append(mlContext.Transforms.Concatenate("Features", "DescriptionEncoded", "NormalizedUNITCOST", "NormalizedQUANTITY", "NormalizedSTOCKONHAND", "NormalizedRUNNINGUNITCOST"))
                     .Append(mlContext.Transforms.DropColumns("DESCRIPTION", "UNITCOST", "QUANTITY", "STOCKONHAND", "RUNNINGUNITCOST"));

            // Apply the transformations to the data
            var processedData = pipeline.Fit(dataView).Transform(dataView);

            return processedData;
        }
    }
}
